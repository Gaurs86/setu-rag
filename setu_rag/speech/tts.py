"""CMI-Conditioned Text-to-Speech  ---  SPEECH NOVELTY #2 (runnable).

Live path: AI4Bharat Indic Parler-TTS, whose natural-language style description is
synthesised from the user's matrix language and Code-Mixing Index so the spoken
reply mirrors their register. If Parler-TTS is unavailable, synthesize() returns
None (the pipeline then returns the answer text without audio, no crash).
"""
from __future__ import annotations
from typing import Optional
from .audio_io import Audio
from ..config import SETTINGS

_VOICE_BY_LANG = {
    "hin": "a clear female Hindi voice", "tam": "a clear female Tamil voice",
    "ben": "a clear Bengali voice", "tel": "a clear Telugu voice",
    "mar": "a clear Marathi voice", "guj": "a clear Gujarati voice",
    "eng": "a clear neutral Indian-English voice",
}

def style_description(matrix_lang: str, cmi: float) -> str:
    fam = matrix_lang.split("_")[0]
    voice = _VOICE_BY_LANG.get(fam, f"a clear {fam} voice")
    if cmi < 0.05:
        mix = "speaking purely in the target language"
    elif cmi < 0.20:
        mix = "with a few naturally pronounced English words"
    else:
        mix = "in a natural code-mixed style, English words pronounced naturally"
    return (f"{voice}, {mix}. The recording is clean and close, "
            f"with neutral pace and slight expressiveness.")

class TTS:
    def __init__(self, model_id: str = "ai4bharat/indic-parler-tts",
                 device: str = "cuda", force_offline: bool = False):
        self.model_id, self.device, self.force_offline = model_id, device, force_offline
        self._model = None
        self._desc_tok = None
        self._prompt_tok = None
        self.live = False

    def load(self):
        if self._model is not None or self.force_offline:
            return self
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            self._model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_id).to(
                self.device if torch.cuda.is_available() else "cpu")
            self._desc_tok = AutoTokenizer.from_pretrained(self.model_id)
            # Indic Parler-TTS uses a separate prompt tokenizer; fall back to the same one.
            try:
                self._prompt_tok = AutoTokenizer.from_pretrained(
                    self._model.config.text_encoder._name_or_path)
            except Exception:
                self._prompt_tok = self._desc_tok
            self.live = True
            print(f"[tts] loaded {self.model_id}")
        except Exception as e:
            print(f"[tts] backend unavailable ({type(e).__name__}); synthesize() returns None.")
            self._model = None; self.live = False
        return self

    def synthesize(self, text: str, matrix_lang: str, cmi: float) -> Optional[Audio]:
        self.load()
        if not self.live or not text.strip():
            return None
        import torch
        desc = style_description(matrix_lang, cmi)
        dev = self._model.device
        d = self._desc_tok(desc, return_tensors="pt").to(dev)
        p = self._prompt_tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = self._model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                       prompt_input_ids=p.input_ids,
                                       prompt_attention_mask=p.attention_mask)
        wav = gen.cpu().numpy().squeeze()
        sr = int(self._model.config.sampling_rate)
        return Audio(samples=wav.astype("float32"), sr=sr)
