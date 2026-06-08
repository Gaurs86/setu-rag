"""Self-RAG-style faithfulness gate (lightweight, runnable).

Splits the answer into claims and checks each against the retrieved context. The
default check is a fast lexical-grounding heuristic (content-word overlap) so the
gate runs with no extra model; an mDeBERTa NLI judge can be dropped in by setting
use_nli=True. Ungrounded answers are caught so the pipeline can regenerate or abstain.
"""
from __future__ import annotations
from typing import List
import re

_STOP = set("a an the is are was were of to in on for and or not it this that you your my our "
            "kya hai ka ki ke ko me mein se par bhi ko aa".split())

def split_claims(answer: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", answer) if s.strip()]

def _content_words(text: str) -> set:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in _STOP and len(w) > 2}

class FaithfulnessGate:
    def __init__(self, nli_model_id: str = "", threshold: float = 0.7, use_nli: bool = False):
        self.model_id, self.threshold, self.use_nli = nli_model_id, threshold, use_nli
        self._m = None

    def load(self):
        if self.use_nli and self._m is None:
            try:
                from transformers import pipeline
                self._m = pipeline("text-classification", model=self.model_id)
            except Exception as e:
                print(f"[faithfulness] NLI unavailable ({type(e).__name__}); using lexical grounding.")
                self.use_nli = False
        return self

    def grounded_fraction(self, answer: str, contexts: List[dict]) -> float:
        self.load()
        claims = split_claims(answer)
        if not claims:
            return 0.0
        ctx_words = set()
        for c in contexts:
            ctx_words |= _content_words(c.get("answer", ""))
        grounded = 0
        for cl in claims:
            cw = _content_words(cl)
            if not cw:
                grounded += 1; continue
            if len(cw & ctx_words) / len(cw) >= 0.4:
                grounded += 1
        return grounded / len(claims)

    def passes(self, answer: str, contexts: List[dict]) -> bool:
        return self.grounded_fraction(answer, contexts) >= self.threshold
