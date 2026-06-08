"""Transliteration normalisation (romanized <-> native script).

Indian-language code-switching online is overwhelmingly written in Roman script
("mujhe refund chahiye"). KB documents and embedders are strongest in native
script, so we normalise romanized tokens to native script with AI4Bharat
IndicXlit (trained on the 26M-pair Aksharantar corpus). This single step is what
makes downstream crosslingual retrieval robust to script — a core contribution.

Real-with-fallback: live path uses ``ai4bharat.transliteration.XlitEngine``
(``pip install ai4bharat-transliteration``). If unavailable (offline / not
installed / ``force_offline``), every method is an identity pass-through so the
pipeline still builds — the native-script query view simply collapses to surface.
"""
from __future__ import annotations
from typing import List

# 3-letter FLORES family -> 2-letter IndicXlit language code.
_FAM2XLIT = {
    "hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu",
    "kan": "kn", "mal": "ml", "ory": "or", "pan": "pa", "asm": "as", "urd": "ur",
    "npi": "ne", "san": "sa", "snd": "sd", "kas": "ks", "mai": "mai", "kok": "gom",
    "doi": "doi", "brx": "brx", "mni": "mni", "sat": "sat",
}


def _xlit_code(lang: str) -> str:
    """Accept 'hin', 'hin_Deva', or 'hi' and return the IndicXlit 2-letter code."""
    fam = lang.split("_")[0]
    return _FAM2XLIT.get(fam, fam)


class Transliterator:
    def __init__(self, device: str = "cuda", force_offline: bool = False):
        self.device = device
        self.force_offline = force_offline
        self._r2n = None   # roman -> native engine
        self._n2r = None   # native -> roman engine
        self.live = False

    def load(self):
        if self._r2n is not None or self.force_offline:
            if self.force_offline:
                self._r2n = self._n2r = "identity"; self.live = False
            return self
        try:
            from ai4bharat.transliteration import XlitEngine
            # Two engines: one per direction (source script is fixed at construction).
            self._r2n = XlitEngine(beam_width=4, src_script_type="roman")
            self._n2r = XlitEngine(beam_width=4, src_script_type="indic")
            self.live = True
            print("[translit] loaded IndicXlit (ai4bharat-transliteration)")
        except Exception as e:
            print(f"[translit] IndicXlit unavailable ({type(e).__name__}); identity pass-through.")
            self._r2n = self._n2r = "identity"; self.live = False
        return self

    def roman_to_native(self, word: str, lang: str) -> str:
        self.load()
        if not self.live:
            return word
        try:
            code = _xlit_code(lang)
            out = self._r2n.translit_word(word, code, topk=1)
            cand = out.get(code) if isinstance(out, dict) else None
            return cand[0] if cand else word
        except Exception:
            return word

    def native_to_roman(self, word: str, lang: str) -> str:
        self.load()
        if not self.live:
            return word
        try:
            code = _xlit_code(lang)
            out = self._n2r.translit_word(word, code, topk=1)
            cand = out.get(code) if isinstance(out, dict) else None
            return cand[0] if cand else word
        except Exception:
            return word

    def normalize_query(self, tokens, default_lang: str = "hi") -> str:
        """Rebuild the query with romanized Indic tokens converted to native script.

        Each romanized Indic token is transliterated using its own detected language
        (from the LID tag) when available, else ``default_lang``.
        """
        out: List[str] = []
        for t in tokens:
            if t.script == "Latn" and t.lang.startswith(("hin", "ben", "tam", "tel",
                    "mar", "guj", "kan", "mal", "ory", "pan", "und")) and t.lang != "eng_Latn":
                lang = default_lang if t.lang.startswith("und") else t.lang
                out.append(self.roman_to_native(t.text, lang))
            else:
                out.append(t.text)
        return " ".join(out)
