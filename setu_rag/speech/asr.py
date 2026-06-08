"""Code-switching-aware ASR (speech -> text), runnable.

Default runnable backend: Whisper (openai/whisper-large-v3-turbo) via the
transformers ASR pipeline — pip-installable, un-gated, handles 22-language audio
and reasonable code-switching. For the dissertation's best code-switched accuracy,
swap `model_id` to AI4Bharat IndicConformer (NeMo) — same interface.

Returns transcript + word timestamps. If no ASR backend is available, transcribe()
returns an empty transcript with a warning (so the text pipeline still runs).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .audio_io import Audio

_FAMILY2WHISPER = {"hin": "hindi", "ben": "bengali", "tam": "tamil", "tel": "telugu",
                   "mar": "marathi", "guj": "gujarati", "kan": "kannada", "mal": "malayalam",
                   "pan": "punjabi", "ory": "oriya", "urd": "urdu", "eng": "english"}

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
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo",
                 device: str = "cuda", force_offline: bool = False):
        self.model_id, self.device, self.force_offline = model_id, device, force_offline
        self._pipe = None
        self.live = False

    def load(self):
        if self._pipe is not None or self.force_offline:
            return self
        try:
            import torch
            from transformers import pipeline
            self._pipe = pipeline(
                "automatic-speech-recognition", model=self.model_id,
                device=0 if (self.device == "cuda" and torch.cuda.is_available()) else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
            self.live = True
            print(f"[asr] loaded {self.model_id}")
        except Exception as e:
            print(f"[asr] backend unavailable ({type(e).__name__}); transcribe() will return ''.")
            self._pipe = None; self.live = False
        return self

    def transcribe(self, audio: Audio, lang_hint: Optional[str] = None) -> ASRResult:
        self.load()
        if not self.live:
            return ASRResult(text="")
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
