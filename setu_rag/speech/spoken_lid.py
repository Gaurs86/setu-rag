"""Spoken (acoustic) language identification (real-with-fallback).

Live path: IndicConformer's LID head, called as ``model(wav, None, "lid")``.
Rather than loading a second copy of the ~1.2 GB model, we share the instance
already loaded by the ASR stage: pass ``asr=self.asr`` from SpeechSetuRAG and
this class borrows the pre-loaded ``_ic`` model.  A separate load is tried only
if no shared ASR model is available yet. Fallback: returns ``{"hin": 1.0}`` so
the pipeline proceeds with a fixed (Hindi-biased) acoustic prior rather than
crashing — CMI/routing still work from the lexical LID signal alone.
"""
from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
from .audio_io import Audio

if TYPE_CHECKING:
    from .asr import ASR


class SpokenLID:
    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual",
                 device: str = "cuda", asr: Optional["ASR"] = None):
        self.model_id = model_id
        self.device = device
        self._asr = asr          # optional shared ASR instance → avoids double-loading
        self._m = None
        self.live = False

    def load(self):
        if self._m is not None:
            return self
        # Prefer borrowing ASR's already-loaded IndicConformer model.
        if (self._asr is not None and
                getattr(self._asr, "backend", None) == "indicconformer" and
                self._asr._ic is not None):
            self._m = self._asr._ic
            self.live = True
            print("[spoken_lid] sharing IndicConformer model from ASR")
            return self
        # Otherwise load our own copy.
        try:
            import torch
            from transformers import AutoModel
            self._m = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
            if self.device == "cuda" and torch.cuda.is_available():
                self._m = self._m.to("cuda")
            self._m.eval()
            self.live = True
            print(f"[spoken_lid] loaded {self.model_id}")
        except Exception as e:
            print(f"[spoken_lid] IndicConformer LID unavailable ({type(e).__name__}); "
                  "returning fixed Hindi prior.")
            self._m = "fallback"; self.live = False
        return self

    def identify(self, audio: Audio) -> Dict[str, float]:
        """Return {{lang_family: posterior}}, e.g. {{'hin': 0.7, 'eng': 0.3}}.

        Calls IndicConformer's LID head (``model(wav, None, 'lid')``). The model
        returns a dict keyed by 2-letter ISO codes; we normalise to 3-letter FLORES
        families to match the text-core's convention. Fallback: ``{{'hin': 1.0}}``.
        """
        self.load()
        if not self.live:
            return {"hin": 1.0}
        try:
            import torch
            wav = torch.from_numpy(audio.samples).float().unsqueeze(0)
            if next(self._m.parameters()).is_cuda:
                wav = wav.cuda()
            with torch.no_grad():
                result = self._m(wav, None, "lid")
            if isinstance(result, dict) and result:
                # Normalise 2-letter -> 3-letter family codes.
                return _normalise_posteriors(result)
            # Some versions return a tensor of log-probs — treat as uniform fallback.
            return {"hin": 1.0}
        except Exception as e:
            print(f"[spoken_lid] LID predict failed ({type(e).__name__}); fixed prior.")
            return {"hin": 1.0}


# 2-letter ISO -> 3-letter FLORES family (inverse of _FAMILY2IC in asr.py)
_ISO2FAM = {"hi": "hin", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar",
            "gu": "guj", "kn": "kan", "ml": "mal", "pa": "pan", "or": "ory",
            "ur": "urd", "as": "asm", "ne": "npi", "sa": "san", "sd": "snd",
            "ks": "kas", "mai": "mai", "kok": "kok", "doi": "doi",
            "brx": "brx", "mni": "mni", "sat": "sat", "en": "eng"}


def _normalise_posteriors(raw: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in raw.items():
        fam = _ISO2FAM.get(k, k[:3])
        out[fam] = out.get(fam, 0.0) + float(v)
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}
