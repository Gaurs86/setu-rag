"""End-to-end SETU-RAG orchestration (runnable).

    query -> front-end (LID, CMI, matrix lang)
          -> CMI-adaptive router            [novel #1]
          -> transliteration-robust views   [novel #2]
          -> hybrid retrieval (dense + BM25) + RRF
          -> cross-encoder rerank
          -> CRAG grade (one corrective pass if a translator is available)
          -> CMI-conditioned generation     [novel #3]
          -> faithfulness gate
          -> grounded, style-matched answer

Construct via setu_rag.app.build_pipeline(), or directly with a built HybridIndex.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .config import SETTINGS
from .front_end.language_id import LanguageIdentifier
from .front_end.transliterate import Transliterator
from .front_end.cmi import compute_cmi
from .router.adaptive_router import decide_route, Route
from .query.multi_view import MultiViewBuilder
from .rerank.reranker import CrossReranker
from .correction.crag import grade, Grade
from .generation.generator import Generator
from .generation.faithfulness import FaithfulnessGate

@dataclass
class Trace:
    query: str = ""
    cmi: float = 0.0
    matrix_lang: str = ""
    route: str = ""
    n_views: int = 0
    grade: str = ""
    faithful: bool = False
    answer: str = ""
    retrieved: list = field(default_factory=list)   # (id, score) after rerank
    contexts: list = field(default_factory=list)

class SetuRAG:
    def __init__(self, index, translator=None, settings=SETTINGS, transliterator=None):
        self.s = settings
        self.index = index
        self.translator = translator
        self.lid = LanguageIdentifier(device=settings.device, force_offline=settings.force_offline)
        self.translit = transliterator if transliterator is not None else \
            Transliterator(device=settings.device, force_offline=settings.force_offline)
        self.mvq = MultiViewBuilder(self.translit, translator)
        self.reranker = CrossReranker(device=settings.device, force_offline=settings.force_offline)
        self.generator = Generator(settings)
        self.gate = FaithfulnessGate(settings.models.get("nli", ""), settings.faithfulness_threshold)

    def answer(self, query: str) -> Trace:
        tokens = self.lid.tag(query)
        cmi = compute_cmi(tokens)
        frac_roman = sum(t.script == "Latn" for t in tokens) / max(len(tokens), 1)
        decision = decide_route(cmi, frac_roman)
        tr = Trace(query=query, cmi=cmi.cmi, matrix_lang=cmi.matrix_lang,
                   route=decision.route, n_views=decision.n_views)

        if decision.route == Route.NO_RETRIEVAL:
            tr.answer = self.generator.generate(query, [], cmi.matrix_lang, cmi.cmi)
            return tr

        views = self.mvq.build(query, tokens, cmi, decision)
        tr.n_views = len(views)
        encs = [self._encode_view(v.text) for v in views]
        fused = self.index.search_views(encs)
        candidates = [self.index.docs[d] for d, _ in fused if d in self.index.docs]
        reranked = self.reranker.rerank(query, candidates, self.s.rerank_k)

        g = grade(reranked)
        tr.grade = g.grade
        if g.needs_requery and self.translator is not None:
            reranked = self._corrective_pass(query, cmi, reranked)
        if not reranked:
            tr.answer = "Sorry, I couldn't find this in our help centre — connecting you to an agent."
            return tr

        tr.retrieved = [(c["id"], round(s, 3)) for c, s in reranked]
        contexts = [c for c, _ in reranked[: self.s.max_ctx_docs]]
        tr.contexts = contexts
        ans = self.generator.generate(query, contexts, cmi.matrix_lang, cmi.cmi)
        if not self.gate.passes(ans, contexts):
            ans = self.generator.generate(query, contexts, cmi.matrix_lang, cmi.cmi)
        tr.faithful = self.gate.passes(ans, contexts)
        tr.answer = ans
        return tr

    def _encode_view(self, text: str) -> dict:
        enc = self.index.embedder.encode([text])
        return {"dense": enc["dense"][0], "tokens": text.split()}

    def _corrective_pass(self, query, cmi, reranked):
        try:
            pivot = self.translator.to_english(query, cmi.matrix_lang)
        except Exception:
            return reranked
        enc = self.index.embedder.encode([pivot])
        fused = self.index.search_views([{"dense": enc["dense"][0], "tokens": pivot.split()}])
        cands = [self.index.docs[d] for d, _ in fused if d in self.index.docs]
        return self.reranker.rerank(pivot, cands, self.s.rerank_k) or reranked
