"""Transliteration-Robust Multi-View Query  ---  NOVEL CONTRIBUTION #2 (runnable).

Expands a code-switched query into up to four views, each embedded separately so at
least one lands in the KB's representation space. The transliterator and translator
are OPTIONAL: when a translator is not provided (quick Colab demo), the English-pivot
and matrix-canonical views are skipped and the system still benefits from the
surface + native-script views.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..router.adaptive_router import Route, RouteDecision

@dataclass
class QueryView:
    name: str
    text: str
    lang: str

class MultiViewBuilder:
    def __init__(self, translit=None, translator=None):
        self.translit = translit
        self.translator = translator

    def build(self, raw: str, tokens, cmi, decision: RouteDecision) -> List[QueryView]:
        views: List[QueryView] = [QueryView("surface", raw, cmi.matrix_lang)]
        wants_native = decision.route in (Route.MONO_ROMAN, Route.AMBIGUOUS, Route.CROSS_LINGUAL)
        if wants_native and self.translit is not None:
            try:
                native = self.translit.normalize_query(tokens)
                if native.strip() and native != raw:
                    views.append(QueryView("native", native, cmi.matrix_lang))
            except Exception:
                pass
        if self.translator is not None and decision.route in (Route.AMBIGUOUS, Route.CROSS_LINGUAL):
            try:
                views.append(QueryView("english_pivot",
                                       self.translator.to_english(raw, cmi.matrix_lang), "eng_Latn"))
            except Exception:
                pass
        if self.translator is not None and decision.route == Route.CROSS_LINGUAL:
            try:
                canon = self.translator.to_lang(raw, src=cmi.matrix_lang, tgt=cmi.matrix_lang)
                views.append(QueryView("matrix_canon", canon, cmi.matrix_lang))
            except Exception:
                pass
        seen, uniq = set(), []
        for v in views:
            key = v.text.strip().lower()
            if key and key not in seen:
                seen.add(key); uniq.append(v)
        return uniq
