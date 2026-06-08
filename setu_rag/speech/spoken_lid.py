"""Spoken (acoustic) language identification.

Gives a language posterior directly from the audio, independent of the ASR
transcript. Useful because in code-switched speech the dominant *acoustic*
language (accent, phonotactics) is a strong, noise-robust prior for the matrix
language even when the transcript is short or noisy. Backed by IndicConformer's
language-ID head (or a dedicated wav2vec2 spoken-LID).
"""
from __future__ import annotations
from typing import Dict
from .audio_io import Audio

class SpokenLID:
    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual",
                 device: str = "cuda"):
        self.model_id, self.device, self._m = model_id, device, None

    def load(self):
        if self._m is None:
            ...  # TODO load LID head / wav2vec2 spoken-LID
        return self

    def identify(self, audio: Audio) -> Dict[str, float]:
        """Return {lang_family: posterior}, e.g. {'hin': 0.7, 'eng': 0.3}."""
        self.load()
        # return self._m.predict_language(audio.samples)
        return {"hin": 1.0}  # TODO
