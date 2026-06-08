"""Voice-activity detection and utterance segmentation (real-with-fallback).

Live path: silero-vad (``snakers4/silero-vad`` via torch.hub) — tiny (~1 MB),
CPU-only, returns per-sample speech timestamps so the pipeline can split a long
recording into clean utterance segments before sending them to ASR.
Fallback: treat the entire clip as a single segment (keeps the pipeline running
when silero is unavailable, e.g. offline or no internet on first run).
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
        self._utils = None
        self.live = False

    def load(self):
        if self._model is not None:
            return self
        try:
            import torch
            self._model, self._utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True)
            self.live = True
            print("[vad] loaded silero-vad")
        except Exception as e:
            print(f"[vad] silero-vad unavailable ({type(e).__name__}); "
                  "treating each clip as one segment.")
            self._model = "passthrough"; self.live = False
        return self

    def _get_timestamps(self, audio: Audio):
        """Return silero speech timestamps (list of {start, end} sample indices)."""
        import torch
        get_speech_timestamps = self._utils[0]
        wav = torch.from_numpy(audio.samples)
        return get_speech_timestamps(
            wav, self._model, sampling_rate=TARGET_SR,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_ms)

    def segment(self, audio: Audio) -> List[Segment]:
        self.load()
        if not self.live:
            return [Segment(0.0, len(audio.samples) / audio.sr, audio)]
        try:
            ts = self._get_timestamps(audio)
            if not ts:
                return [Segment(0.0, len(audio.samples) / audio.sr, audio)]
            segs = []
            for t in ts:
                s, e = int(t["start"]), int(t["end"])
                clip = Audio(audio.samples[s:e].copy(), audio.sr)
                segs.append(Segment(s / audio.sr, e / audio.sr, clip))
            return segs
        except Exception as e:
            print(f"[vad] segmentation failed ({type(e).__name__}); one segment fallback.")
            return [Segment(0.0, len(audio.samples) / audio.sr, audio)]

    def stream(self, chunks: Iterator[Audio]) -> Iterator[Segment]:
        """Streaming hook: buffer chunks then segment the accumulated clip."""
        buf: List[np.ndarray] = []
        for ch in chunks:
            buf.append(ch.samples)
        merged = Audio(np.concatenate(buf) if buf else np.zeros(0, "float32"), TARGET_SR)
        yield from self.segment(merged)
