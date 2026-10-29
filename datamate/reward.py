"""Verifiable reward for RL fine-tuning (RLVR), following ReViSQL [R40] §5.1.

ReViSQL rewards a rollout with +1 when its final SQL is equivalent to the gold query under the
question's grading method, 0 when it fails to execute or gives a different answer, and -1 when
the model produces no final query. Here the gold is the organisers' answer and "equivalent"
is the official per-question scoring rule (`scorer.is_correct`, aliases included). An
`unanswerable` declaration is an answer like any other: +1 if the gold is unanswerable, 0
otherwise.

This module is only used by `scripts/export_rlvr_data.py` and by external trainers; the
inference pipeline never calls it. Whether a fine-tuned checkpoint is admissible under rule 1
("models with weights publicly released before 1 Oct 2026") must be confirmed with the
organisers before such a checkpoint is submitted.
"""
from __future__ import annotations

from .formatter import ANSWER_TYPES
from .scorer import is_correct


def score_completion(agent, item: dict, text: str, gold: dict) -> tuple[float, dict]:
    """-> (reward, info). `agent` provides the shop context and the executor."""
    typ, lang, code = agent._parse(text)
    if typ == "unanswerable":
        ok = gold.get("type") == "unanswerable"
        return (1.0 if ok else 0.0), {"pred": {"type": "unanswerable"}}
    if not code:
        return -1.0, {"error": "no final program"}
    ctx = agent.context(item["business_id"])
    if ctx is None:
        return 0.0, {"error": "business not found"}
    from .question import analyze
    spec = analyze(item["question"])
    tctx = agent.resolver.resolve(item["question"], item["as_of_date"])
    atype = typ if typ in ANSWER_TYPES else spec.answer_type
    res = agent._execute(ctx, lang, code, item["as_of_date"])
    fa = agent._fmt(res, atype, tctx.month_answer or spec.entity == "month")
    if fa is None:
        return 0.0, {"error": res.get("err") or "empty / unreadable result"}
    pred = {"type": fa[0], "answer": fa[1]}
    return (1.0 if is_correct(pred, gold) else 0.0), {"pred": pred}
