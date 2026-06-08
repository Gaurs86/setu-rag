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

def resample(audio: "Audio", target_sr: int = TARGET_SR) -> "Audio":
    """Resample to target_sr (16 kHz by default). No-op if already at rate.

    The whole speech stack assumes 16 kHz mono; mic/upload audio from Gradio
    usually arrives at 44.1/48 kHz, so this must run before VAD/ASR or the
    timeline and acoustic features are all wrong.
    """
    if audio.sr == target_sr or audio.samples.size == 0:
        return Audio(samples=audio.samples.astype("float32"), sr=target_sr)
    src = audio.samples.astype("float32")
    try:
        import librosa
        wav = librosa.resample(src, orig_sr=audio.sr, target_sr=target_sr)
    except Exception:
        # Fallback: numpy linear interpolation (no SciPy/librosa needed).
        n_out = int(round(src.shape[0] * target_sr / audio.sr))
        x_old = np.linspace(0.0, 1.0, num=src.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        wav = np.interp(x_new, x_old, src)
    return Audio(samples=wav.astype("float32"), sr=target_sr)

def duration_s(audio: "Audio") -> float:
    return len(audio.samples) / audio.sr
