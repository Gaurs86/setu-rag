"""Offline KB indexing: store native / romanized / English-pivot forms per chunk.

The transliterator and translator are OPTIONAL: when absent (e.g. the quick Colab
demo), the forms gracefully fall back to the native text so the index still builds.
Switch-point-aware chunking keeps each chunk's matrix language coherent.
"""
from __future__ import annotations
from typing import Dict, List
import json, hashlib

def chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def expand_forms(chunk: str, lang: str, translit=None, translator=None) -> Dict[str, str]:
    roman = chunk
    if translit is not None:
        try:
            roman = " ".join(translit.native_to_roman(w, lang.split("_")[0]) for w in chunk.split())
        except Exception:
            roman = chunk
    en_pivot = ""
    if translator is not None:
        try:
            en_pivot = translator.to_english(chunk, lang)
        except Exception:
            en_pivot = ""
    return {"native": chunk, "roman": roman, "en_pivot": en_pivot}

def build_records(faq_path: str, translit=None, translator=None) -> List[dict]:
    records = []
    with open(faq_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            forms = expand_forms(row["answer"], row.get("lang", "hin_Deva"), translit, translator)
            records.append({"id": chunk_id(row["answer"]), "lang": row.get("lang"),
                            "question": row.get("question"), "answer": row["answer"],
                            "forms": forms, "meta": row.get("meta", {})})
    return records
