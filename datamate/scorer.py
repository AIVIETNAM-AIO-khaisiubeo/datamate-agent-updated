"""Local re-implementation of the official DataMate scoring.

Per question: correct (+1) / abstain (0) / wrong (-1); PA = mean.
* abstain or missing answer               -> 0
* type must equal the reference type       (otherwise wrong)
* number   : exact match (refs are ints or 2-dp)       -> compared after round(.,2)
* percent  : exact after rounding to 2 dp, sign must match
* string   : metric_norm(pred) in {metric_norm(form) for form in accepted forms}
* date     : exact string (yyyy-mm-dd, or yyyy-mm for month questions)
* boolean  : exact
* list     : same length, position-wise match; elements by string/number rule
* table    : columns by normalised name, else by position; rows as a multiset;
             name cells string rule, numeric cells number rule
* unanswerable: correct iff reference is unanswerable

Reference format accepted by this scorer (dev/synthetic gold):
  {"id":..., "type":"string", "answer": "Sting dâu 330ml", "accepted": ["sting dâu 330ml", "sp0042"]}
  - "answer" may itself be a list of accepted forms for strings;
  - list elements / table cells may be lists of accepted forms.
"""
from __future__ import annotations

import math

from .textnorm import metric_norm

ABSTAIN = "abstain"
UNANSWERABLE = "unanswerable"


def _to_float(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def num_eq(p, g) -> bool:
    a, b = _to_float(p), _to_float(g)
    if a is None or b is None:
        return False
    return round(a, 2) == round(b, 2)


def pct_eq(p, g) -> bool:
    a, b = _to_float(p), _to_float(g)
    if a is None or b is None:
        return False
    ra, rb = round(a, 2), round(b, 2)
    return ra == rb and (math.copysign(1, ra) == math.copysign(1, rb) or ra == 0)


def _forms(g, extra=None) -> set[str]:
    forms = g if isinstance(g, (list, tuple, set)) else [g]
    forms = list(forms) + list(extra or [])
    return {metric_norm(f) for f in forms if f is not None}


def str_eq(p, g, extra=None) -> bool:
    if p is None or isinstance(p, (list, dict)):
        return False
    return metric_norm(p) in _forms(g, extra)


def cell_eq(p, g) -> bool:
    if isinstance(g, (list, tuple)):          # accepted forms of a name cell
        return str_eq(p, g)
    if _to_float(g) is not None and not isinstance(g, str):
        return num_eq(p, g)
    if isinstance(g, str) and _to_float(g) is not None and _to_float(p) is not None:
        return num_eq(p, g)
    return str_eq(p, g)


def _table_eq(pa, ga) -> bool:
    if not isinstance(pa, dict) or "rows" not in pa:
        return False
    gcols = [metric_norm(c) for c in ga.get("columns", [])]
    pcols = [metric_norm(c) for c in pa.get("columns", [])]
    if len(pcols) != len(gcols):
        return False
    if sorted(pcols) == sorted(gcols) and len(set(gcols)) == len(gcols):
        perm = [pcols.index(c) for c in gcols]
    else:
        perm = list(range(len(gcols)))
    prows = [[r[i] if i < len(r) else None for i in perm] for r in pa.get("rows", [])]
    grows = ga.get("rows", [])
    if len(prows) != len(grows):
        return False
    used = [False] * len(prows)
    for g in grows:
        j = next((j for j, p in enumerate(prows)
                  if not used[j] and len(p) == len(g) and all(cell_eq(x, y) for x, y in zip(p, g))), None)
        if j is None:
            return False
        used[j] = True
    return True


def _alias_at(gold, i):
    al = gold.get("alias") or []
    if i < len(al):
        a = al[i]
        return a if isinstance(a, list) else [a]
    return []


def is_correct(pred: dict, gold: dict) -> bool:
    pt, gt = pred.get("type"), gold.get("type")
    if pt != gt or pt is None:
        return False
    pa, ga = pred.get("answer"), gold.get("answer")
    if gt == UNANSWERABLE:
        return True
    if gt == "number":
        return num_eq(pa, ga)
    if gt == "percent":
        return pct_eq(pa, ga)
    if gt == "string":
        al = gold.get("alias") or []
        al = [x for a in al for x in (a if isinstance(a, list) else [a])]
        return str_eq(pa, ga, list(gold.get("accepted") or []) + al)
    if gt == "date":
        return isinstance(pa, str) and pa.strip() == str(ga).strip()
    if gt == "boolean":
        return isinstance(pa, bool) and pa == (ga if isinstance(ga, bool) else str(ga).lower() == "true")
    if gt == "list":
        if not isinstance(pa, list) or len(pa) != len(ga):
            return False
        for i, (p, g) in enumerate(zip(pa, ga)):
            al = _alias_at(gold, i)
            if isinstance(g, (int, float)) and not isinstance(g, bool):
                if not num_eq(p, g):
                    return False
            elif not str_eq(p, g if isinstance(g, list) else [g], al):
                return False
        return True
    if gt == "table":
        ga2 = ga
        al = gold.get("alias") or []
        if al and isinstance(ga, dict):  # alias per row applies to the first (name) column
            rows = []
            for i, r in enumerate(ga.get("rows", [])):
                a = al[i] if i < len(al) else []
                a = a if isinstance(a, list) else [a]
                rows.append([[r[0]] + a] + list(r[1:]) if a and r else r)
            ga2 = {"columns": ga.get("columns", []), "rows": rows}
        return _table_eq(pa, ga2)
    return False


def score_one(pred: dict | None, gold: dict) -> int:
    if pred is None or pred.get("type") in (None, ABSTAIN):
        return 0
    return 1 if is_correct(pred, gold) else -1


def penalized_accuracy(preds: dict, golds: dict) -> float:
    if not golds:
        return 0.0
    return sum(score_one(preds.get(i), g) for i, g in golds.items()) / len(golds)


def report(preds: dict, golds: dict) -> dict:
    """PA plus breakdown by reference type: answered / correct / wrong / abstain."""
    from collections import defaultdict
    by = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0, "abstain": 0})
    for i, g in golds.items():
        s = score_one(preds.get(i), g)
        b = by[g.get("type")]
        b["n"] += 1
        b["correct" if s == 1 else "wrong" if s == -1 else "abstain"] += 1
    tot = {k: sum(b[k] for b in by.values()) for k in ("n", "correct", "wrong", "abstain")}
    answered = tot["correct"] + tot["wrong"]
    return {"PA": penalized_accuracy(preds, golds), **tot,
            "coverage": answered / max(1, tot["n"]),
            "selective_precision": tot["correct"] / max(1, answered),
            "by_type": dict(by)}
