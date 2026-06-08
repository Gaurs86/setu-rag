"""Driver: run SETU-RAG over an eval set and print CS-RAGAS."""
from __future__ import annotations
import json, argparse
from ..front_end.language_id import LanguageIdentifier
from . import cs_metrics

def main(eval_path: str, pipeline):
    lid = LanguageIdentifier().load()
    rows, cmi_al, lang_con = [], [], []
    for line in open(eval_path, encoding="utf-8"):
        ex = json.loads(line)
        tr = pipeline.answer(ex["question"])
        cmi_al.append(cs_metrics.cmi_alignment(ex["question"], tr.answer, lid))
        lang_con.append(cs_metrics.language_consistency(ex["question"], tr.answer, lid))
        rows.append({"question": ex["question"], "answer": tr.answer,
                     "contexts": [c["answer"] for c in tr.contexts],
                     "ground_truth": ex.get("ground_truth", "")})
    print(f"CMI-alignment        : {sum(cmi_al)/len(cmi_al):.3f}")
    print(f"Language-consistency : {sum(lang_con)/len(lang_con):.3f}")
    # from .ragas_eval import evaluate_ragas; print(evaluate_ragas(rows))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eval", default="data/eval.sample.jsonl")
    print("Wire a built pipeline into main(); see scripts/serve.py")
