"""Cross-encoder reranking (BGE-reranker-v2-m3) with a lexical fallback.

Live path: sentence-transformers CrossEncoder scores (query, doc) pairs; raw logits
are squashed with a sigmoid to [0,1] so the CRAG thresholds are meaningful. Offline
fallback: Jaccard-style lexical overlap, which keeps ordering sensible without the
model.
"""
from __future__ import annotations
from typing import List, Tuple
import math

class CrossReranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda",
                 force_offline: bool = False):
        self.model_id, self.device, self.force_offline = model_id, device, force_offline
        self._m = None
        self.live = False

    def load(self):
        if self._m is not None:
            return self
        if self.force_offline:
            self._m = "lex"; self.live = False; return self
        try:
            from sentence_transformers import CrossEncoder
            self._m = CrossEncoder(self.model_id, device=self.device, max_length=512)
            self.live = True
            print(f"[reranker] loaded {self.model_id}")
        except Exception as e:
            print(f"[reranker] real model unavailable ({type(e).__name__}); using lexical fallback.")
            self._m = "lex"; self.live = False
        return self

    def rerank(self, query: str, candidates: List[dict], top_k: int) -> List[Tuple[dict, float]]:
        self.load()
        if not candidates:
            return []
        if self.live:
            pairs = [[query, c.get("answer", "")] for c in candidates]
            raw = self._m.predict(pairs)
            scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw]   # sigmoid -> [0,1]
        else:
            qset = set(query.lower().split())
            scores = []
            for c in candidates:
                dset = set((c.get("answer", "") + " " + (c.get("question") or "")).lower().split())
                scores.append(len(qset & dset) / (len(qset) or 1))
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
