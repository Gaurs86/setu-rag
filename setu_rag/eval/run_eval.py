"""Real CS-RAGAS evaluation driver — runs a handful of code-switched QA pairs
through a built SETU-RAG pipeline and prints a results table.

Per pair it reports:
  * route / measured CMI            (the router's linguistic decision)
  * retrieval hit@k                 (did a retrieved doc match the gold topic)
  * CMI-alignment                   (style mirroring: 1 - |CMI(q) - CMI(a)|)
  * language-consistency            (answer uses the user's matrix language)
  * answer WER / CER vs reference   (lexical closeness to the gold answer;
                                     reuses the speech-side WER/CER implementation)
  * faithful                        (faithfulness gate verdict)

Runs anywhere: with real models on a GPU, or with deterministic fallbacks
(``force_offline=True``) so the harness itself is testable without downloads.

    python scripts/run_eval.py --offline
    python -m setu_rag.eval.run_eval --offline
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Optional

from ..front_end.language_id import LanguageIdentifier
from . import cs_metrics
from .speech_metrics import wer, cer


@dataclass
class RowResult:
    question: str
    route: str
    cmi_q: float
    retrieval_hit: bool
    cmi_alignment: float
    language_consistency: float
    answer_wer: float
    answer_cer: float
    faithful: bool
    answer: str = ""


@dataclass
class EvalSummary:
    rows: List[RowResult] = field(default_factory=list)

    def _mean(self, attr):
        vals = [getattr(r, attr) for r in self.rows]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def means(self) -> dict:
        return {
            "retrieval_hit@k": self._mean("retrieval_hit"),
            "cmi_alignment": self._mean("cmi_alignment"),
            "language_consistency": self._mean("language_consistency"),
            "answer_wer": self._mean("answer_wer"),
            "answer_cer": self._mean("answer_cer"),
            "faithful": self._mean("faithful"),
        }


def _retrieval_hit(trace, gold_topic: Optional[str]) -> bool:
    """True if any retrieved context carries the gold topic (top-k after rerank)."""
    if not gold_topic:
        return False
    for ctx in getattr(trace, "contexts", []) or []:
        if (ctx.get("meta", {}) or {}).get("topic") == gold_topic:
            return True
    return False


def evaluate(pipeline, eval_path: str, lid: Optional[LanguageIdentifier] = None) -> EvalSummary:
    lid = lid or LanguageIdentifier(
        force_offline=getattr(pipeline.s, "force_offline", False)).load()
    summary = EvalSummary()
    with open(eval_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            q, gt = ex["question"], ex.get("ground_truth", "")
            tr = pipeline.answer(q)
            summary.rows.append(RowResult(
                question=q,
                route=str(tr.route),
                cmi_q=round(tr.cmi, 2),
                retrieval_hit=_retrieval_hit(tr, ex.get("topic")),
                cmi_alignment=cs_metrics.cmi_alignment(q, tr.answer, lid),
                language_consistency=cs_metrics.language_consistency(q, tr.answer, lid),
                answer_wer=round(wer(gt, tr.answer), 3),
                answer_cer=round(cer(gt, tr.answer), 3),
                faithful=bool(tr.faithful),
                answer=tr.answer,
            ))
    return summary


def print_table(summary: EvalSummary) -> None:
    cols = [("question", 34), ("route", 20), ("cmi", 5), ("hit", 4),
            ("cmi_al", 7), ("lang", 5), ("wer", 6), ("cer", 6), ("faith", 6)]
    header = " | ".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in summary.rows:
        route = r.route.replace("Route.", "")
        cells = [
            r.question[:34].ljust(34),
            route[:20].ljust(20),
            f"{r.cmi_q:.2f}".ljust(5),
            ("Y" if r.retrieval_hit else "n").ljust(4),
            f"{r.cmi_alignment:.2f}".ljust(7),
            f"{r.language_consistency:.0f}".ljust(5),
            f"{r.answer_wer:.2f}".ljust(6),
            f"{r.answer_cer:.2f}".ljust(6),
            ("Y" if r.faithful else "n").ljust(6),
        ]
        print(" | ".join(cells))
    print("-" * len(header))
    m = summary.means
    print(f"\nAggregate over {len(summary.rows)} code-switched pairs:")
    print(f"  retrieval hit@k      : {m['retrieval_hit@k']:.3f}")
    print(f"  CMI-alignment        : {m['cmi_alignment']:.3f}")
    print(f"  language-consistency : {m['language_consistency']:.3f}")
    print(f"  answer WER (vs gold) : {m['answer_wer']:.3f}")
    print(f"  answer CER (vs gold) : {m['answer_cer']:.3f}")
    print(f"  faithful             : {m['faithful']:.3f}")
    # RAGAS quality axes (faithfulness / answer-relevancy / context precision+recall)
    # need a multilingual judge LLM; see eval/ragas_eval.py to enable on a GPU run.


def main(eval_path: str = "data/eval.sample.jsonl", offline: bool = True,
         enable_translation: bool = False, pipeline=None) -> EvalSummary:
    if pipeline is None:
        from ..app import build_pipeline
        pipeline = build_pipeline(force_offline=offline, enable_translation=enable_translation)
    summary = evaluate(pipeline, eval_path)
    print_table(summary)
    return summary


if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval.sample.jsonl")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--translate", action="store_true", help="enable IndicTrans2 pivot views")
    args = ap.parse_args()
    main(eval_path=args.eval, offline=args.offline, enable_translation=args.translate)
