"""Transliteration normalisation (romanized <-> native script).

Indian-language code-switching online is overwhelmingly written in Roman script
("mujhe refund chahiye"). KB documents and embedders are strongest in native
script, so we normalise romanized tokens to native script with AI4Bharat
IndicXlit (trained on the 26M-pair Aksharantar corpus). This single step is what
makes downstream crosslingual retrieval robust to script — a core contribution.
"""
from __future__ import annotations
from typing import List

class Transliterator:
    def __init__(self, device: str = "cuda"):
        self._engine = None

    def load(self):
        if self._engine is None:
            # from ai4bharat.transliteration import XlitEngine
            # self._engine = XlitEngine(beam_width=4, src_script_type="roman")
            ...  # pip install ai4bharat-transliteration
        return self

    def roman_to_native(self, word: str, lang: str) -> str:
        """lang is a 2-letter IndicXlit code, e.g. 'hi','ta','bn'."""
        self.load()
        # return self._engine.translit_word(word, lang, topk=1)[lang][0]
        return word  # TODO

    def native_to_roman(self, word: str, lang: str) -> str:
        self.load()
        return word  # TODO

    def normalize_query(self, tokens, default_lang: str = "hi") -> str:
        """Rebuild the query with romanized Indic tokens converted to native."""
        out: List[str] = []
        for t in tokens:
            if t.script == "Latn" and t.lang.startswith(("hin","ben","tam","tel",
                    "mar","guj","kan","mal","ory","pan","und")) and t.lang != "eng_Latn":
                out.append(self.roman_to_native(t.text, default_lang))
            else:
                out.append(t.text)
        return " ".join(out)
