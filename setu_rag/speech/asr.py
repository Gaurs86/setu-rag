"""Code-switching-aware ASR (speech -> text), runnable with a 3-tier backend.

Primary: AI4Bharat IndicConformer (``ai4bharat/indic-conformer-600m-multilingual``,
loaded via ``AutoModel(trust_remote_code=True)`` and called as
``model(wav, lang, "ctc")``) — 22 languages, trained on IndicVoices incl.
code-switched speech, so it is the best transcriber for this project.
Fallback: Whisper (``openai/whisper-large-v3-turbo``) via the transformers ASR
pipeline — pip-installable, un-gated, gives word timestamps.
Final fallback: if no backend is available, ``transcribe()`` returns an empty
transcript with a warning (so the text pipeline still runs).

Returns transcript + word timestamps (timestamps only on the Whisper path).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .audio_io import Audio

_FAMILY2WHISPER = {"hin": "hindi", "ben": "bengali", "tam": "tamil", "tel": "telugu",
                   "mar": "marathi", "guj": "gujarati", "kan": "kannada", "mal": "malayalam",
                   "pan": "punjabi", "ory": "oriya", "urd": "urdu", "eng": "english"}

# IndicConformer expects ISO 2-letter language codes.
_FAMILY2IC = {"hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu",
              "kan": "kn", "mal": "ml", "pan": "pa", "ory": "or", "urd": "ur", "asm": "as",
              "npi": "ne", "san": "sa", "snd": "sd", "kas": "ks", "mai": "mai", "kok": "kok",
              "doi": "doi", "brx": "brx", "mni": "mni", "sat": "sat", "eng": "en"}


@dataclass
class ASRWord:
    text: str
    start_s: float = 0.0
    end_s: float = 0.0
    conf: float = 1.0


@dataclass
class ASRResult:
    text: str
    words: List[ASRWord] = field(default_factory=list)
    acoustic_lang: Optional[str] = None
    lang_posteriors: dict = field(default_factory=dict)


class ASR:
    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual",
                 fallback_id: str = "openai/whisper-large-v3-turbo",
                 device: str = "cuda", force_offline: bool = False,
                 default_lang: str = "hi"):
        self.model_id, self.fallback_id = model_id, fallback_id
        self.device, self.force_offline = device, force_offline
        self.default_lang = default_lang
        self._ic = None          # IndicConformer model
        self._pipe = None        # Whisper pipeline
        self.backend = None      # "indicconformer" | "whisper" | None
        self.live = False

    def load(self):
        if self.backend is not None or self.force_offline:
            return self
        # 1) primary: IndicConformer
        try:
            import torch
            from transformers import AutoModel
            self._ic = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
            if self.device == "cuda" and torch.cuda.is_available():
                self._ic = self._ic.to("cuda")
            self._ic.eval()
            self.backend = "indicconformer"; self.live = True
            print(f"[asr] loaded {self.model_id} (IndicConformer)")
            return self
        except Exception as e:
            print(f"[asr] IndicConformer unavailable ({type(e).__name__}: {e}); trying Whisper fallback.")
        # 2) fallback: Whisper
        try:
            import torch
            from transformers import pipeline
            self._pipe = pipeline(
                "automatic-speech-recognition", model=self.fallback_id,
                device=0 if (self.device == "cuda" and torch.cuda.is_available()) else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
            self.backend = "whisper"; self.live = True
            print(f"[asr] loaded {self.fallback_id} (Whisper fallback)")
        except Exception as e:
            print(f"[asr] no ASR backend ({type(e).__name__}); transcribe() will return ''.")
            self.backend = None; self.live = False
        return self

    def transcribe(self, audio: Audio, lang_hint: Optional[str] = None) -> ASRResult:
        self.load()
        if not self.live:
            return ASRResult(text="")
        if self.backend == "indicconformer":
            return self._transcribe_ic(audio, lang_hint)
        return self._transcribe_whisper(audio, lang_hint)

    # ---- IndicConformer (primary) -----------------------------------------
    def _transcribe_ic(self, audio: Audio, lang_hint: Optional[str]) -> ASRResult:
        try:
            import torch
            fam = (lang_hint or "").split("_")[0]
            lang = _FAMILY2IC.get(fam, self.default_lang)
            wav = torch.from_numpy(audio.samples).float().unsqueeze(0)  # (1, n)
            if next(self._ic.parameters()).is_cuda:
                wav = wav.cuda()
            with torch.no_grad():
                out = self._ic(wav, lang, "ctc")
            text = self._coerce_text(out).strip()
            return ASRResult(text=text, acoustic_lang=fam or None)
        except Exception as e:
            print(f"[asr] IndicConformer transcribe failed ({type(e).__name__}); empty transcript.")
            return ASRResult(text="")

    @staticmethod
    def _coerce_text(out) -> str:
        """IndicConformer may return a str, a 1-element list/tuple, or a dict —
        extract the transcript without leaking a Python repr into the text."""
        if isinstance(out, str):
            return out
        if isinstance(out, (list, tuple)):
            return " ".join(ASR._coerce_text(o) for o in out)
        if isinstance(out, dict):
            for k in ("text", "transcription", "transcript"):
                if k in out:
                    return ASR._coerce_text(out[k])
        return str(out)

    # ---- Whisper (fallback) -----------------------------------------------
    def _transcribe_whisper(self, audio: Audio, lang_hint: Optional[str]) -> ASRResult:
        gen_kw = {}
        if lang_hint:
            wl = _FAMILY2WHISPER.get(lang_hint.split("_")[0])
            if wl:
                gen_kw["language"] = wl
        try:
            res = self._pipe({"raw": audio.samples, "sampling_rate": audio.sr},
                             return_timestamps="word", generate_kwargs=gen_kw)
        except Exception:
            res = self._pipe({"raw": audio.samples, "sampling_rate": audio.sr})
        text = res.get("text", "").strip() if isinstance(res, dict) else str(res)
        words = []
        for ch in (res.get("chunks", []) if isinstance(res, dict) else []):
            ts = ch.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
            words.append(ASRWord(text=ch.get("text", "").strip(),
                                 start_s=ts[0] or 0.0, end_s=ts[1] or 0.0))
        return ASRResult(text=text, words=words)
