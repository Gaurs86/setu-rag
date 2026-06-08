"""Audio loading, resampling and saving for the speech pipeline.

All speech models in this project expect 16 kHz mono float32 PCM. This module
centralises that contract so every stage (VAD, ASR, TTS) speaks the same format.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

TARGET_SR = 16000

@dataclass
class Audio:
    samples: np.ndarray   # float32, mono, shape (n,)
    sr: int = TARGET_SR

def load(path: str, target_sr: int = TARGET_SR) -> "Audio":
    """Load any audio file to 16 kHz mono float32. Uses soundfile + librosa."""
    import soundfile as sf, librosa
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:                       # stereo -> mono
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return Audio(samples=wav.astype("float32"), sr=target_sr)

def save(path: str, audio: "Audio") -> str:
    import soundfile as sf
    sf.write(path, audio.samples, audio.sr)
    return path

def from_array(samples: np.ndarray, sr: int) -> "Audio":
    return Audio(samples=np.asarray(samples, dtype="float32"), sr=sr)

def duration_s(audio: "Audio") -> float:
    return len(audio.samples) / audio.sr
