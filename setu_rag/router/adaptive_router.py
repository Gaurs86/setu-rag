"""CMI-Adaptive Retrieval Router  ---  NOVEL CONTRIBUTION #1.

Standard Adaptive-RAG routes by *reasoning complexity*. We instead route by the
*linguistic* profile of the query (Code-Mixing Index + matrix language), because
the failure mode we target is retrieval breakage under code-switching, not
multi-hop reasoning. The router chooses the cheapest path that will still match
the relevant KB chunk:

    MONO_NATIVE   : CMI < cmi_low and script native  -> single dense query
    MONO_ROMAN    : CMI < cmi_low and script roman   -> transliterate then dense
    CROSS_LINGUAL : CMI >= cmi_high                  -> full multi-view fan-out
    AMBIGUOUS     : cmi_low <= CMI < cmi_high         -> light fan-out (2 views)

A trainable variant replaces the thresholds with a small logistic head over
features [cmi, switch_points/N, frac_roman, n_embedded, matrix_is_english].
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from ..config import SETTINGS
from ..front_end.cmi import CMIResult

class Route(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    MONO_NATIVE = "mono_native"
    MONO_ROMAN = "mono_roman"
    AMBIGUOUS = "ambiguous"
    CROSS_LINGUAL = "cross_lingual"

@dataclass
class RouteDecision:
    route: Route
    n_views: int
    target_langs: list

def decide_route(cmi: CMIResult, frac_roman: float, chit_chat: bool = False) -> RouteDecision:
    s = SETTINGS
    if chit_chat:
        return RouteDecision(Route.NO_RETRIEVAL, 0, [])
    if cmi.cmi >= s.cmi_high:
        langs = [cmi.matrix_lang, *cmi.embedded_langs] or [cmi.matrix_lang]
        return RouteDecision(Route.CROSS_LINGUAL, 4, langs)
    if cmi.cmi < s.cmi_low:
        if frac_roman > 0.5:
            return RouteDecision(Route.MONO_ROMAN, 2, [cmi.matrix_lang])
        return RouteDecision(Route.MONO_NATIVE, 1, [cmi.matrix_lang])
    return RouteDecision(Route.AMBIGUOUS, 2, [cmi.matrix_lang, *cmi.embedded_langs][:2])
