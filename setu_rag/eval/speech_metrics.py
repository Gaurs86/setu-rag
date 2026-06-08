"""Speech-side evaluation metrics (WER / CER implemented).

Extends the text CS-RAGAS harness with the acoustic axes shown in the speech
architecture figure: ASR accuracy on code-switched audio, spoken-LID accuracy,
and TTS intelligibility (by re-transcribing the synthesised audio and comparing).
"""
from __future__ import annotations
from typing import List

def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i-1] != b[j-1]))
            prev = cur
    return dp[m]

def wer(reference: str, hypothesis: str) -> float:
    r, h = reference.split(), hypothesis.split()
    return _edit_distance(r, h) / max(len(r), 1)

def cer(reference: str, hypothesis: str) -> float:
    r, h = list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
    return _edit_distance(r, h) / max(len(r), 1)

def spoken_lid_accuracy(pred_langs: List[str], gold_langs: List[str]) -> float:
    if not gold_langs:
        return 0.0
    return sum(p == g for p, g in zip(pred_langs, gold_langs)) / len(gold_langs)

def tts_intelligibility(original_text: str, reasr_text: str) -> float:
    """1 - CER between the TTS input text and a re-ASR of the synthesised audio.

    A cheap, reference-free proxy for intelligibility (higher is better). Pair with
    a small human MOS panel for naturalness, which cannot be automated reliably.
    """
    return round(1.0 - cer(original_text, reasr_text), 4)

def lid_fusion_gain(matrix_text_only: List[str], matrix_fused: List[str],
                    gold_matrix: List[str]) -> float:
    """Accuracy(fused) - Accuracy(text-only) for matrix-language decisions."""
    def acc(preds):
        return sum(p == g for p, g in zip(preds, gold_matrix)) / max(len(gold_matrix), 1)
    return round(acc(matrix_fused) - acc(matrix_text_only), 4)
