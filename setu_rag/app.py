"""One-call builder for a runnable SETU-RAG text pipeline.

    from setu_rag.app import build_pipeline
    rag = build_pipeline()                 # loads sample FAQs, builds the index
    print(rag.answer("mera refund kab aayega").answer)

On Colab/Kaggle with the deps installed and a GPU, this uses the real models
(BGE-M3, BGE-reranker-v2-m3, a 4-bit instruct LLM). Anywhere else it falls back to
lightweight stand-ins so the structure still runs. Pass force_offline=True to skip
all downloads.
"""
from __future__ import annotations
import os
from .config import SETTINGS
from .retrieval.embedder import M3Embedder
from .retrieval.index import HybridIndex
from .retrieval.kb_ingest import build_records
from .pipeline import SetuRAG

_HERE = os.path.dirname(__file__)
_DEFAULT_FAQ = os.path.normpath(os.path.join(_HERE, "..", "data", "faqs.sample.jsonl"))

def build_pipeline(faq_path: str | None = None, device: str | None = None,
                   force_offline: bool = False, translit=None, translator=None) -> SetuRAG:
    if device:
        SETTINGS.device = device
    SETTINGS.force_offline = force_offline
    faq_path = faq_path or _DEFAULT_FAQ
    records = build_records(faq_path, translit=translit, translator=translator)
    embedder = M3Embedder(SETTINGS.models["embedder"], device=SETTINGS.device,
                          force_offline=force_offline)
    index = HybridIndex(embedder, SETTINGS).build(records)
    return SetuRAG(index=index, translator=translator, settings=SETTINGS, transliterator=translit)

def demo(queries=None, **kw):
    rag = build_pipeline(**kw)
    queries = queries or [
        "mera refund kab tak aayega, maine order cancel kiya tha",
        "how do I track my order",
        "coupon apply nahi ho raha hai",
    ]
    for q in queries:
        tr = rag.answer(q)
        print(f"\nQ: {q}")
        print(f"   route={tr.route} cmi={tr.cmi:.2f} matrix={tr.matrix_lang} "
              f"grade={tr.grade} faithful={tr.faithful}")
        print(f"   retrieved={tr.retrieved}")
        print(f"   A: {tr.answer}")
    return rag

if __name__ == "__main__":
    demo(force_offline=True)
