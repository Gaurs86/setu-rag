"""Corrective RAG grader (CRAG-style).

After reranking, a lightweight grader labels the evidence
{CORRECT, AMBIGUOUS, INCORRECT} from the top rerank score and its margin:

  CORRECT   -> proceed to generation with the retrieved context
  AMBIGUOUS -> proceed but also trigger ONE corrective re-query (English pivot)
  INCORRECT -> drop context; re-query via translation, else fall back to a
               generic-help template or human handoff (customer-support safe).

This guards against the documented failure where retrievers silently return
off-topic chunks for code-switched inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

class Grade(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"

@dataclass
class GradeResult:
    grade: Grade
    top_score: float
    needs_requery: bool

def grade(reranked: List[Tuple[dict, float]],
          hi: float = 0.55, lo: float = 0.30) -> GradeResult:
    if not reranked:
        return GradeResult(Grade.INCORRECT, 0.0, True)
    top = reranked[0][1]
    if top >= hi:
        return GradeResult(Grade.CORRECT, top, False)
    if top >= lo:
        return GradeResult(Grade.AMBIGUOUS, top, True)
    return GradeResult(Grade.INCORRECT, top, True)
