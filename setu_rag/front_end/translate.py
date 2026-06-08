"""IndicTrans2 translation for query expansion + KB pivoting (real-with-fallback).

Powers two of the multi-view query expansions (novel contribution #2):
  * ``to_english``  -> the English-pivot view and the CRAG corrective re-query
  * ``to_lang``     -> the matrix-canonical view (round-trip into the matrix
                       language's native script via an English pivot)
and the KB-side ``en_pivot`` form in ``kb_ingest``.

Live path: AI4Bharat IndicTrans2 distilled checkpoints (200M/320M, T4-friendly)
driven through ``IndicTransToolkit``'s ``IndicProcessor``. Three direction models
are loaded on demand and cached:
  en->indic, indic->en, indic->indic.
If the toolkit/weights are unavailable (offline / not installed / ``force_offline``),
every method returns "" (no translation) — callers already treat an empty pivot as
"skip this view", so the pipeline keeps running on the surface + native views.

Pass one into ``build_pipeline(enable_translation=True)`` (or ``translator=...``)
to bring the English-pivot / matrix-canonical views and the corrective pass alive.
"""
from __future__ import annotations
from typing import Optional
from ..config import SETTINGS, SCHEDULED_LANGS

# 3-letter family -> default native FLORES code IndicTrans2 expects (e.g. hin -> hin_Deva).
_FAM2FLORES = {code.split("_")[0]: code for code in SCHEDULED_LANGS}
_FAM2FLORES["eng"] = "eng_Latn"


def _flores(lang: str) -> str:
    """Accept 'hin', 'hin_Deva', or 'eng' and return a full FLORES code."""
    if "_" in lang:
        return lang
    return _FAM2FLORES.get(lang, lang)


def _is_eng(code: str) -> bool:
    return code.split("_")[0] == "eng"


class IndicTrans2Translator:
    def __init__(self, settings=SETTINGS, device: str = "cuda", force_offline: bool = False):
        self.s = settings
        self.device = device
        self.force_offline = force_offline or settings.force_offline
        self._ip = None                  # IndicProcessor
        self._models = {}                # direction -> (tokenizer, model)
        self._loaded = False
        self.live = False

    # ---- loading -----------------------------------------------------------
    def _ensure_processor(self):
        if self._loaded:
            return
        self._loaded = True
        if self.force_offline:
            self.live = False
            return
        try:
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except Exception:
                from IndicTransToolkit import IndicProcessor  # older layout
            self._ip = IndicProcessor(inference=True)
            self.live = True
            print("[translate] IndicTrans2 toolkit ready")
        except Exception as e:
            print(f"[translate] IndicTrans2 unavailable ({type(e).__name__}); pivots disabled.")
            self.live = False

    def _direction_model(self, src_code: str, tgt_code: str):
        """Return (tokenizer, model) for the right distilled checkpoint, loaded on demand."""
        if _is_eng(src_code):
            key = "mt_en_indic"
        elif _is_eng(tgt_code):
            key = "mt_indic_en"
        else:
            key = "mt_indic_indic"
        if key in self._models:
            return self._models[key]
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_id = self.s.models[key]
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype)
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        print(f"[translate] loaded {model_id}")
        self._models[key] = (tok, model)
        return tok, model

    def free(self):
        """Drop the cached direction models to reclaim VRAM between turns."""
        self._models.clear()
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- core --------------------------------------------------------------
    def _translate(self, text: str, src: str, tgt: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self._ensure_processor()
        if not self.live:
            return ""
        src_code, tgt_code = _flores(src), _flores(tgt)
        if src_code == tgt_code:
            return text
        try:
            import torch
            tok, model = self._direction_model(src_code, tgt_code)
            batch = self._ip.preprocess_batch([text], src_lang=src_code, tgt_lang=tgt_code)
            inputs = tok(batch, truncation=True, padding="longest",
                         return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_length=256, num_beams=5,
                                     num_return_sequences=1)
            decoded = tok.batch_decode(gen.detach().cpu().tolist(), skip_special_tokens=True)
            out = self._ip.postprocess_batch(decoded, lang=tgt_code)
            return out[0].strip() if out else ""
        except Exception as e:
            print(f"[translate] {src_code}->{tgt_code} failed ({type(e).__name__}); returning ''.")
            return ""

    # ---- public API expected by the pipeline -------------------------------
    def to_english(self, text: str, src_lang: str) -> str:
        if _is_eng(_flores(src_lang)):
            return (text or "").strip()
        return self._translate(text, src_lang, "eng_Latn")

    def to_lang(self, text: str, src: str, tgt: str) -> str:
        """Render ``text`` into ``tgt``'s native script.

        Used for the matrix-canonical view where src==tgt: we pivot through English
        (romanized/code-mixed surface -> clean native-script paraphrase), which is
        more robust than feeding romanized text straight to an indic->indic model.
        """
        src_code, tgt_code = _flores(src), _flores(tgt)
        if _is_eng(tgt_code):
            return self.to_english(text, src)
        pivot = self.to_english(text, src) if not _is_eng(src_code) else text
        if not (pivot or "").strip():
            return ""
        return self._translate(pivot, "eng_Latn", tgt_code)
