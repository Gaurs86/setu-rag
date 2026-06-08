"""Code-switching-native evaluation metrics (fully implemented).

Beyond RAG quality, we measure whether the system *behaves* correctly under
code-switching -- the novelty needs CS-aware metrics:

  cmi_alignment        : 1 - |CMI(query) - CMI(answer)|  (style mirroring)
  language_consistency : answer uses the user's matrix language as its frame
  translit_robustness  : retrieval-quality delta between a native-script query
                         and its romanized version (lower |delta| = more robust)
"""
from __future__ import annotations
from typing import List
from ..front_end.language_id import LanguageIdentifier
from ..front_end.cmi import compute_cmi

def cmi_alignment(query: str, answer: str, lid: LanguageIdentifier) -> float:
    cq = compute_cmi(lid.tag(query)).cmi
    ca = compute_cmi(lid.tag(answer)).cmi
    return round(1.0 - abs(cq - ca), 4)

def language_consistency(query: str, answer: str, lid: LanguageIdentifier) -> float:
    mq = compute_cmi(lid.tag(query)).matrix_lang
    ma = compute_cmi(lid.tag(answer)).matrix_lang
    return 1.0 if mq == ma else 0.0

def translit_robustness(native_scores: List[float], roman_scores: List[float]) -> float:
    """Mean absolute nDCG delta between native and romanized variants of queries."""
    if not native_scores:
        return 0.0
    deltas = [abs(a - b) for a, b in zip(native_scores, roman_scores)]
    return round(1.0 - (sum(deltas) / len(deltas)), 4)
