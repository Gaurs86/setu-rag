"""Cross-encoder reranking (BGE-reranker-v2-m3) with a lexical fallback.

Live path — tries two strategies in order:
  1. sentence-transformers CrossEncoder (clean API, works on ST < 3.x)
  2. transformers AutoModelForSequenceClassification directly (ST 3.x broke
     CrossEncoder for some architectures with an AssertionError — loading the
     model directly via transformers is equally fast and always works)
Offline fallback: Jaccard-style lexical overlap so ordering stays sensible
without the model.
"""
from __future__ import annotations
from typing import List, Tuple
import math


class CrossReranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda",
                 force_offline: bool = False):
        self.model_id, self.device, self.force_offline = model_id, device, force_offline
        self._m = None          # CrossEncoder | "direct" | "lex"
        self._tok = None        # tokenizer when _m == "direct"
        self._mdl = None        # raw model when _m == "direct"
        self.live = False

    def load(self):
        if self._m is not None:
            return self
        if self.force_offline:
            self._m = "lex"; self.live = False; return self

        import torch
        dev = self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"

        # Strategy 1 — sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            self._m = CrossEncoder(self.model_id, device=dev, max_length=512)
            self.live = True
            print(f"[reranker] loaded {self.model_id} (CrossEncoder, device={dev})")
            return self
        except Exception as e1:
            print(f"[reranker] CrossEncoder failed ({type(e1).__name__}: {e1}); "
                  "trying transformers direct load.")

        # Strategy 2 — transformers AutoModelForSequenceClassification (ST 3.x compat)
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            dtype = torch.float16 if dev == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._mdl = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, torch_dtype=dtype)
            if dev == "cuda":
                self._mdl = self._mdl.to("cuda")
            self._mdl.eval()
            self._m = "direct"
            self.live = True
            print(f"[reranker] loaded {self.model_id} (transformers direct, device={dev})")
        except Exception as e2:
            print(f"[reranker] transformers load also failed ({type(e2).__name__}: {e2}); "
                  "using lexical fallback.")
            self._m = "lex"; self.live = False
        return self

    def rerank(self, query: str, candidates: List[dict], top_k: int) -> List[Tuple[dict, float]]:
        self.load()
        if not candidates:
            return []
        if self._m == "direct":
            scores = self._score_direct(query, candidates)
        elif self.live:
            pairs = [[query, c.get("answer", "")] for c in candidates]
            raw = self._m.predict(pairs)
            scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw]
        else:
            qset = set(query.lower().split())
            scores = [
                len(qset & set((c.get("answer", "") + " " + (c.get("question") or ""))
                               .lower().split())) / (len(qset) or 1)
                for c in candidates
            ]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _score_direct(self, query: str, candidates: List[dict]) -> List[float]:
        import torch
        pairs = [[query, c.get("answer", "")] for c in candidates]
        enc = self._tok(pairs, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        enc = {k: v.to(self._mdl.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._mdl(**enc).logits
        # BGE-reranker outputs a single logit; sigmoid → [0, 1]
        scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores
