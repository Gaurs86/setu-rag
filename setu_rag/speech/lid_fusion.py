"""Acoustic + Lexical LID Fusion  ---  SPEECH NOVELTY #1.

Code-switched speech gives two complementary language signals:
  * acoustic  : spoken-LID posterior over the whole utterance (robust, coarse)
  * lexical   : token-level IndicLID over the ASR transcript (fine, but inherits
                ASR errors and is unreliable on short romanised spans)

We fuse them into a single matrix-language decision and a refined per-token tag
sequence, then hand that to the text core's CMI module. The fusion both stabilises
the matrix language (acoustic prior) and keeps token-level resolution (lexical),
which neither signal achieves alone. This makes the downstream CMI / routing
decisions far more reliable than running the text pipeline on a raw transcript.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from ..front_end.language_id import Token

@dataclass
class FusedLID:
    matrix_lang: str
    token_langs: List[str]          # one family code per transcript token
    acoustic_weight: float

def fuse(acoustic: Dict[str, float], tokens: List[Token],
         alpha: float = 0.4) -> FusedLID:
    """alpha = weight on the acoustic prior when (re)deciding each token's language.

    For each token we combine its lexical posterior (here, a hard tag from IndicLID
    with its confidence) with the acoustic posterior, and we pick the global matrix
    language from the alpha-blended distribution over the whole utterance.
    """
    from collections import defaultdict
    fam = lambda l: l.split("_")[0]
    # lexical vote weighted by token confidence
    lex_dist: Dict[str, float] = defaultdict(float)
    token_langs: List[str] = []
    for t in tokens:
        f = fam(t.lang)
        lex_dist[f] += t.conf
        token_langs.append(f)
    # normalise lexical distribution
    tot = sum(lex_dist.values()) or 1.0
    lex_dist = {k: v / tot for k, v in lex_dist.items()}
    # blended global distribution
    keys = set(lex_dist) | set(acoustic)
    blended = {k: (1 - alpha) * lex_dist.get(k, 0.0) + alpha * acoustic.get(k, 0.0)
               for k in keys}
    matrix = max(blended, key=blended.get) if blended else "und"
    # re-tag tokens that IndicLID marked low-confidence/undetermined toward the
    # acoustically-dominant language (helps short romanised spans)
    ac_top = max(acoustic, key=acoustic.get) if acoustic else matrix
    refined = [tl if tl not in ("und",) else ac_top for tl in token_langs]
    return FusedLID(matrix_lang=matrix, token_langs=refined, acoustic_weight=alpha)
