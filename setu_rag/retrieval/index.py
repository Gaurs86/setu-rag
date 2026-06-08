"""Hybrid index (dense FAISS + lexical BM25) with Reciprocal Rank Fusion.

Fully runnable: builds from a list of KB records, searches every query view with
both retrievers, and fuses the ranked lists with RRF. FAISS and BM25 are optional
— if absent, a NumPy brute-force cosine search and an empty lexical list keep the
pipeline working.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60,
                           weights: List[float] | None = None) -> List[Tuple[str, float]]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: Dict[str, float] = defaultdict(float)
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(lst):
            scores[doc_id] += w * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class HybridIndex:
    def __init__(self, embedder, settings):
        self.embedder = embedder
        self.s = settings
        self._faiss = None
        self._mat = None
        self._bm25 = None
        self._corpus_tokens: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.docs: Dict[str, dict] = {}

    def build(self, records: List[dict]):
        texts, self.doc_ids, self.docs, self._corpus_tokens = [], [], {}, []
        for r in records:
            self.docs[r["id"]] = r
            forms = r.get("forms", {})
            blob = " ".join([r.get("question") or "", r.get("answer") or "",
                             forms.get("roman", ""), forms.get("en_pivot", "")]).strip()
            texts.append(blob); self.doc_ids.append(r["id"])
            self._corpus_tokens.append(blob.lower().split())
        self._mat = self.embedder.encode(texts)["dense"]
        try:
            import faiss
            idx = faiss.IndexFlatIP(self._mat.shape[1]); idx.add(self._mat); self._faiss = idx
        except Exception:
            self._faiss = None
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus_tokens)
        except Exception:
            self._bm25 = None
        print(f"[index] built over {len(self.doc_ids)} docs "
              f"(faiss={'yes' if self._faiss else 'numpy'}, bm25={'yes' if self._bm25 else 'no'})")
        return self

    def dense_search(self, qvec, k: int) -> List[str]:
        if self._mat is None or len(self.doc_ids) == 0:
            return []
        q = np.asarray(qvec, dtype="float32").reshape(1, -1)
        if self._faiss is not None:
            _, I = self._faiss.search(q, min(k, len(self.doc_ids)))
            return [self.doc_ids[i] for i in I[0] if i >= 0]
        sims = self._mat @ q[0]
        return [self.doc_ids[i] for i in np.argsort(-sims)[:k]]

    def bm25_search(self, tokens, k: int) -> List[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores([t.lower() for t in tokens])
        return [self.doc_ids[i] for i in np.argsort(-scores)[:k]]

    def search_views(self, view_encodings: List[dict]) -> List[Tuple[str, float]]:
        ranked: List[List[str]] = []
        for enc in view_encodings:
            ranked.append(self.dense_search(enc["dense"], self.s.retrieve_k))
            ranked.append(self.bm25_search(enc.get("tokens", []), self.s.retrieve_k))
        return reciprocal_rank_fusion(ranked, k=self.s.rrf_k)[: self.s.retrieve_k]
