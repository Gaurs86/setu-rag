"""Code-Mixing Index and matrix-language detection (fully implemented).

CMI (Das & Gamback, 2014) quantifies how code-mixed an utterance is.
For an utterance with N language-bearing tokens, where max(w_i) is the count of
tokens in the most frequent language and P is the number of language switch
points:

    CMI = 100 * (1 - max(w_i)/N) ... combined with a switch-point term.

We return a normalised CMI in [0,1], the matrix (dominant) language, the set of
embedded languages, and the number of switch points. These features drive the
CMI-Adaptive Retrieval Router (novel contribution #1).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import List
from .language_id import Token

_NON_LANG = {"und_Latn"}  # punctuation/numbers are skipped upstream

@dataclass
class CMIResult:
    cmi: float                 # 0 (monolingual) .. ~1 (maximally mixed)
    matrix_lang: str
    embedded_langs: List[str]
    switch_points: int
    n_tokens: int

def _lang_family(lang: str) -> str:
    # collapse romanized/native of same language: 'hin_Latn' & 'hin_Deva' -> 'hin'
    return lang.split("_")[0]

def compute_cmi(tokens: List[Token]) -> CMIResult:
    langs = [_lang_family(t.lang) for t in tokens
             if t.lang not in _NON_LANG and t.text.isalnum()]
    n = len(langs)
    if n == 0:
        return CMIResult(0.0, "und", [], 0, 0)
    counts = Counter(langs)
    matrix, max_w = counts.most_common(1)[0]
    # switch points = transitions between adjacent differing languages
    sp = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
    # Das-Gamback combined CMI, scaled to [0,1]
    frac_embedded = 1.0 - (max_w / n)
    sp_term = (sp / (n - 1)) if n > 1 else 0.0
    cmi = 0.5 * frac_embedded + 0.5 * sp_term * (1 if len(counts) > 1 else 0)
    embedded = [l for l in counts if l != matrix]
    return CMIResult(round(cmi, 4), matrix, embedded, sp, n)
