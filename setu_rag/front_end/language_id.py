"""Token-level language identification for code-switched Indian text.

Wraps AI4Bharat IndicLID, the first LID covering all 22 scheduled languages in
BOTH native and romanized script (47 classes incl. English/Other). We run it at
the *token* level so a single utterance can carry per-token language tags, which
the CMI module then turns into a code-mixing measure.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)

@dataclass
class Token:
    text: str
    lang: str       # e.g. "hin_Deva", "eng_Latn", "hin_Latn" (romanized Hindi)
    script: str     # "Latn" | "Deva" | "Beng" | ...
    conf: float

class LanguageIdentifier:
    """Real-with-fallback token LID.

    Live path: AI4Bharat IndicLID (``pip install``-from-GitHub; see
    github.com/AI4Bharat/IndicLID). It is the only LID covering all 22 scheduled
    languages in BOTH native and romanized script, so it discriminates English vs
    romanized Indic far better than the heuristic. We query it per word to obtain
    token-level tags. If the package/weights are unavailable (offline / CPU /
    ``force_offline``), we fall back to the deterministic script+cue heuristic so
    the pipeline keeps running.
    """

    def __init__(self, model_id: str = "ai4bharat/IndicLID", device: str = "cuda",
                 force_offline: bool = False):
        self.model_id = model_id
        self.device = device
        self.force_offline = force_offline
        self._model = None  # lazy
        self.live = False

    def load(self):
        if self._model is not None:
            return self
        if self.force_offline:
            self._model = "heuristic"; self.live = False
            return self
        try:
            from IndicLID import IndicLID
            # input_threshold gates native-vs-roman routing; roman_lid_threshold gates
            # the romanized classifier's confidence (defaults from the IndicLID README).
            self._model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
            self.live = True
            print(f"[lid] loaded IndicLID ({self.model_id})")
        except Exception as e:
            print(f"[lid] IndicLID unavailable ({type(e).__name__}); using script+cue heuristic.")
            self._model = "heuristic"; self.live = False
        return self

    @staticmethod
    def _script_of(tok: str) -> str:
        for ch in tok:
            o = ord(ch)
            if 0x0900 <= o <= 0x097F: return "Deva"
            if 0x0980 <= o <= 0x09FF: return "Beng"
            if 0x0B80 <= o <= 0x0BFF: return "Taml"
            if 0x0C00 <= o <= 0x0C7F: return "Telu"
            if 0x0A00 <= o <= 0x0A7F: return "Guru"
            if 0x0A80 <= o <= 0x0AFF: return "Gujr"
            if 0x0D00 <= o <= 0x0D7F: return "Mlym"
            if 0x0C80 <= o <= 0x0CFF: return "Knda"
            if 0x0B00 <= o <= 0x0B7F: return "Orya"
        return "Latn"

    # Heuristic fallback ONLY for the stub so demos produce realistic CMI without
    # the real model. Replace `tag()` with an IndicLID call in production.
    _SCRIPT2LANG = {"Deva": "hin_Deva", "Beng": "ben_Beng", "Taml": "tam_Taml",
                    "Telu": "tel_Telu", "Guru": "pan_Guru", "Gujr": "guj_Gujr",
                    "Mlym": "mal_Mlym", "Knda": "kan_Knda", "Orya": "ory_Orya"}

    def tag(self, text: str) -> List[Token]:
        """Return per-token language tags.

        Live path: IndicLID per word (FLORES-style ``xxx_Yyyy`` labels map straight
        onto Token.lang). Fallback: deterministic script+cue heuristic.
        """
        self.load()
        if self.live:
            try:
                return self._tag_indiclid(text)
            except Exception as e:
                print(f"[lid] IndicLID predict failed ({type(e).__name__}); heuristic for this call.")
        return self._tag_heuristic(text)

    def _tag_indiclid(self, text: str) -> List[Token]:
        words = _WORD.findall(text)
        # Only language-bearing words go to the model; punctuation/numbers are tagged
        # locally so we don't waste model calls (and so CMI skips them as before).
        alpha = [w for w in words if w.isalpha()]
        preds = {}
        if alpha:
            # batch_predict returns rows of [input, predicted_label, score, model_type].
            for row in self._model.batch_predict(alpha, len(alpha)):
                w, label, score = row[0], row[1], float(row[2]) if len(row) > 2 else 0.7
                preds[w] = (label, score)
        toks: List[Token] = []
        for w in words:
            script = self._script_of(w)
            if not w.isalpha():
                toks.append(Token(text=w, lang="und_Latn" if script == "Latn" else f"und_{script}",
                                  script=script, conf=1.0))
                continue
            label, score = preds.get(w, (None, 0.7))
            if not label or label == "other":
                # fall back to script for native, romanized-Hindi for Latin
                lang = self._SCRIPT2LANG.get(script, "hin_Latn" if script == "Latn" else f"und_{script}")
            else:
                lang = label
                script = label.split("_")[1] if "_" in label else script
            toks.append(Token(text=w, lang=lang, script=script, conf=score))
        return toks

    def _tag_heuristic(self, text: str) -> List[Token]:
        """Deterministic fallback: native script -> that language; Latin alphabetic ->
        English if in the cue list, else assumed romanized Indic (default Hindi)."""
        toks: List[Token] = []
        for w in _WORD.findall(text):
            script = self._script_of(w)
            if script != "Latn":
                lang = self._SCRIPT2LANG.get(script, f"und_{script}")
            elif not w.isalpha():
                lang = "und_Latn"                       # numbers / punctuation
            elif w.lower() in _EN_CUES:
                lang = "eng_Latn"
            else:
                lang = "hin_Latn"                       # romanized Indic (assume Hindi)
            toks.append(Token(text=w, lang=lang, script=script, conf=0.7))
        return toks

# Small English cue list used ONLY by the stub; the real system uses IndicLID,
# which discriminates English vs romanized Indic far more reliably.
_EN_CUES = {"the","is","am","are","my","your","why","how","when","what","will",
            "i","a","an","to","of","and","or","not","on","in","for","please",
            "order","refund","cancel","coupon","cart","track","return","payment",
            "page","status","help","agent","delivery","business","days","method"}
