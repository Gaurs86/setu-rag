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
    def __init__(self, model_id: str = "ai4bharat/IndicLID", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None  # lazy

    def load(self):
        if self._model is None:
            # from IndicLID import IndicLID
            # self._model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
            ...  # TODO: load IndicLID (see github.com/AI4Bharat/IndicLID)
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
        """Return per-token language tags. Replace the stub with real IndicLID.

        Stub policy: native script -> that language; Latin alphabetic -> English
        if in the English cue list, else assumed romanized Indic (default Hindi).
        """
        self.load()
        toks: List[Token] = []
        for w in _WORD.findall(text):
            script = self._script_of(w)
            # TODO: real call -> lang, conf = self._model.predict(w)
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
