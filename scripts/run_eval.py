"""Run the CS-RAGAS evaluation harness over the sample code-switched QA pairs.

    python scripts/run_eval.py --offline     # deterministic fallbacks, no downloads
    python scripts/run_eval.py               # real models if available
    python scripts/run_eval.py --translate   # also bring IndicTrans2 pivot views alive
"""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setu_rag.eval.run_eval import main


def _run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval.sample.jsonl"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--translate", action="store_true")
    args = ap.parse_args()
    main(eval_path=args.eval, offline=args.offline, enable_translation=args.translate)


if __name__ == "__main__":
    _run()
