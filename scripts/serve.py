"""Minimal Gradio chat over SETU-RAG (T4-friendly demo).

    python scripts/serve.py                 # real models if available, fallbacks otherwise
    python scripts/serve.py --offline       # deterministic fallbacks, no downloads
"""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setu_rag.app import build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faqs", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    import gradio as gr
    rag = build_pipeline(faq_path=args.faqs, force_offline=args.offline)

    def respond(msg, history):
        tr = rag.answer(msg)
        return (f"{tr.answer}\n\n"
                f"_route={tr.route} · cmi={tr.cmi:.2f} · matrix={tr.matrix_lang} · "
                f"grade={tr.grade} · faithful={tr.faithful}_")

    gr.ChatInterface(respond, title="SETU-RAG — Code-Switch-Aware Support").launch(
        share=args.share)


if __name__ == "__main__":
    main()
