"""BGE-M3 dense embedder with a deterministic offline fallback.

Live path: sentence-transformers loads BAAI/bge-m3 and returns L2-normalised dense
vectors (cosine == dot product). If the library or the weights are unavailable
(no GPU / no internet / force_offline), a hashing embedding is used instead so the
pipeline still runs end-to-end — useful for unit tests and for trying the system
on CPU before downloading models.
"""
from __future__ import annotations
from typing import List, Dict
import hashlib
import numpy as np

class M3Embedder:
    def __init__(self, model_id: str = "BAAI/bge-m3", device: str = "cuda",
                 fp16: bool = True, dim: int = 1024, force_offline: bool = False):
        self.model_id, self.device, self.fp16 = model_id, device, fp16
        self.dim = dim
        self.force_offline = force_offline
        self._m = None
        self.live = False

    def load(self):
        if self._m is not None:
            return self
        if self.force_offline:
            self._m = "hash"; self.live = False; return self
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            try:
                self._m = SentenceTransformer(self.model_id, device=dev)
            except RuntimeError as cuda_err:
                if dev == "cuda":
                    print(f"[embedder] CUDA load failed ({cuda_err}); retrying on CPU.")
                    self._m = SentenceTransformer(self.model_id, device="cpu")
                else:
                    raise
            self.dim = self._m.get_sentence_embedding_dimension()
            self.live = True
            print(f"[embedder] loaded {self.model_id} (dim={self.dim}, device={self._m.device})")
        except Exception as e:
            print(f"[embedder] real model unavailable ({type(e).__name__}: {e}); using hashing fallback.")
            self._m = "hash"; self.live = False
        return self

    def encode(self, texts: List[str]) -> Dict[str, np.ndarray]:
        self.load()
        if self.live:
            v = self._m.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
            return {"dense": v.astype("float32")}
        return {"dense": np.stack([self._hash(t) for t in texts]).astype("float32")}

    def _hash(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in text.lower().split():
            for gram in (tok, tok[:3], tok[-3:]):
                if not gram:
                    continue
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
