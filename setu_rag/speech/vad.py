"""Voice-activity detection and utterance segmentation.

Turn-based now, streaming-ready: `segment()` works on a whole clip; `stream()`
exposes the same logic over a chunk iterator so a live-mic front-end can reuse it
later without touching ASR. Backed by silero-vad (tiny, CPU-friendly).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List
import numpy as np
from .audio_io import Audio, TARGET_SR

@dataclass
class Segment:
    start_s: float
    end_s: float
    audio: Audio

class VAD:
    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 300):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self._model = None

    def load(self):
        if self._model is None:
            # import torch
            # self._model, self._utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
            ...
        return self

    def segment(self, audio: Audio) -> List[Segment]:
        """Whole-clip segmentation. Replace stub with silero get_speech_timestamps."""
        self.load()
        # ts = get_speech_timestamps(audio.samples, self._model, sampling_rate=TARGET_SR)
        # return [Segment(t['start']/TARGET_SR, t['end']/TARGET_SR, slice_audio(audio,t)) ...]
        return [Segment(0.0, len(audio.samples) / audio.sr, audio)]  # TODO

    def stream(self, chunks: Iterator[Audio]) -> Iterator[Segment]:
        """Streaming hook: yield a Segment whenever an utterance boundary is hit."""
        buf: List[np.ndarray] = []
        for ch in chunks:                    # TODO: real online VAD state machine
            buf.append(ch.samples)
        merged = Audio(np.concatenate(buf) if buf else np.zeros(0, "float32"), TARGET_SR)
        yield from self.segment(merged)
