"""Build the index and run a couple of queries end-to-end.

    python scripts/build_index.py                       # uses sample FAQs, real models if available
    python scripts/build_index.py --offline             # deterministic fallbacks, no downloads
    python scripts/build_index.py --faqs my_faqs.jsonl
"""
import argparse
import os, sys
# Make `setu_rag` importable when run as `python scripts/build_index.py` (Python puts
# the script's own dir on sys.path, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setu_rag.app import build_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faqs", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    rag = build_pipeline(faq_path=args.faqs, device=args.device, force_offline=args.offline)
    for q in ["mera refund kab tak aayega", "how do I cancel my order"]:
        tr = rag.answer(q)
        print(f"\nQ: {q}\n   route={tr.route} cmi={tr.cmi:.2f} -> {tr.answer}")

if __name__ == "__main__":
    main()
