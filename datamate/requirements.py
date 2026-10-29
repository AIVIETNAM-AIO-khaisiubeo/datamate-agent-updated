"""Explicit requirements of a question, and a deterministic check of a program against them.

Adapted from ReViSQL (Zhu, Jin, Choi, Kang, 2026 [R40]):

* *Evidence supervision.* ReViSQL's system prompt makes the model translate every
  piece of external knowledge into a requirement ("Requirement of external
  knowledge i (...): ...") before writing SQL, and verify each one before the
  final answer ("Verification of external knowledge i (...): ..."). In DataMate
  the "external knowledge" is what our deterministic layer already resolved:
  the time ranges, the linked entities and filters, and the rulebook rules that
  apply. We list them as R1..Rn in the prompt and ask for one CHECK line each.

* *Reconciliation.* ReViSQL groups candidate SQL queries by execution result and
  keeps the groups that a model judges to cover all explicit constraints of the
  question and external knowledge (Algorithm 1). Most of our constraints are
  literal (date literals, stored values, numbers), so a large part of that
  judgement can be done deterministically on the program text: `check()` below.
  It is deliberately permissive (a requirement fails only when no acceptable
  form of it appears in the program) because ReViSQL's reconciliation is
  valuable only with a low false-rejection rate (1.3-3.9 % in their Table 7).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta

from .textnorm import fold

_CANCEL_WORDS = (r"\bhuy\b|\btra hang\b|\btra lai\b|\bhoan (tra|hang|tien)\b|\bno\b|\bcong no\b|\bton\b|\bkho\b|"
                 r"\bnhap\b|nha cung cap|het han|\blo\b")
_SALES_INTENTS = {"revenue", "n_orders", "aov", "pct_change", "share", "best_period", "top_items", "top_customers",
                  "top_staff", "n_customers", "n_customers_min_orders", "pct_customers_min_orders",
                  "n_days_threshold", "n_orders_threshold", "n_items_min_orders", "qty", "qty_size", "monotonic",
                  "revenue_series", "multi_filter", "shift_diff_pct"}


@dataclass
class Requirement:
    rid: str
    kind: str                      # time | snapshot | valid | filter | threshold | rank | shift | weekday | type
    text: str                      # shown to the LLM
    tokens: list = field(default_factory=list)   # acceptable folded forms; any one present = satisfied
    checkable: bool = True
    fn: object = None                              # custom check on the raw program text

    def check(self, code: str) -> bool | None:
        """True / False on the program text, None when the requirement cannot be read off the text."""
        if not self.checkable or not code:
            return None
        if self.fn is not None:
            return bool(self.fn(code))
        c = _norm_code(code)
        return any(t and t in c for t in self.tokens)


def _time_check(r):
    """The program bounds the period: its first day and (exclusive end or last day) appear as date literals,
    or the period is selected as a calendar month / year by name."""
    start, end, last = r.start.isoformat(), r.end.isoformat(), (r.end - timedelta(days=1)).isoformat()
    month_like = r.start.day == 1 and (r.end - r.start).days in (28, 29, 30, 31)

    def fn(code):
        lits = set(re.findall(r"\d{4}-\d{2}-\d{2}", code))
        if start in lits and (end in lits or last in lits):
            return True
        low = code.lower()
        if month_like:
            ym = f"{r.start:%Y-%m}"
            if re.search(rf"['\"]{ym}['\"]", code):
                return True
            if start in lits and re.search(r"date_trunc|to_period|\.month|month\s*\(", low):
                return True
            if re.search(rf"month\s*\([^)]*\)\s*=+\s*{r.start.month}\b", low) and str(r.start.year) in low:
                return True
        if r.start.month == 1 and r.start.day == 1 and r.end == r.start.replace(year=r.start.year + 1):
            if re.search(rf"year\s*\([^)]*\)\s*=+\s*{r.start.year}\b|\.year\s*==\s*{r.start.year}\b", low):
                return True
        return False
    return fn


def _norm_code(code: str) -> str:
    """Folded program text with SQL/Python comments removed; digits kept, separators in numbers dropped."""
    code = re.sub(r"--[^\n]*|#[^\n]*", " ", code)
    c = fold(code.replace("_", " "))
    c = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", c)
    return " " + re.sub(r"\s+", " ", c) + " "


def _fold_iso(d) -> str:
    return fold(d.isoformat())          # '2026 05 01'


def _value_tokens(value: str) -> list:
    """A stored value is 'present' when its folded form, or its longest distinctive part, appears."""
    f = fold(str(value))
    toks = [f]
    words = [w for w in f.split() if len(w) >= 3]
    if len(words) >= 2:
        toks.append(" ".join(words[:2]))
    return toks


def build(spec, tctx, links=(), attr_filters=(), rb=None, entity_link=None) -> list[Requirement]:
    """Requirements the answer program must satisfy, in the order they are shown to the LLM."""
    from .templates import CHAN_LEX, PAY_LEX
    R: list[Requirement] = []

    def add(kind, text, tokens=(), checkable=True, fn=None):
        R.append(Requirement(f"R{len(R) + 1}", kind, text, [t for t in tokens if t], checkable, fn))

    for r in tctx.ranges:
        add("time", f"{'compare with' if r.role == 'comparison' else 'restrict to'} {r.label}: "
                    f"order_date >= DATE '{r.start}' AND order_date < DATE '{r.end}' "
                    f"(use the date column of the table you query if it is not v_orders / v_sales)", fn=_time_check(r))
    if tctx.snapshot is not None:
        add("snapshot", f"state at the END of {tctx.snapshot} (records up to and including that day)",
            [_fold_iso(tctx.snapshot), _fold_iso(tctx.snapshot + timedelta(days=1))])
    q = spec.q or ""
    if (spec.intent in _SALES_INTENTS or (spec.intent is None and re.search(r"doanh thu|\bdon\b|ban duoc|khach", q))) \
            and not re.search(_CANCEL_WORDS, q) and (rb is None or rb.has("valid_orders")):
        add("valid", "count only valid orders (rulebook): v_orders.is_valid / v_sales.is_valid", [" is valid "])
    if entity_link is not None:
        add("filter", f"only {entity_link.role or 'value'} = '{entity_link.value}' "
                      f"(stored in {entity_link.table}.{entity_link.column})", _value_tokens(entity_link.value))
    if spec.group_name:
        add("filter", f"only the product group named like '{spec.group_name}' (exact group name in the data)",
            _value_tokens(spec.group_name))
    if spec.cust_group:
        add("filter", f"only customers of group '{spec.cust_group}'", _value_tokens(spec.cust_group))
    if spec.payment:
        add("filter", f"only orders paid by {spec.payment}", [f" {w} " for w in PAY_LEX.get(spec.payment, [spec.payment])]
            + [fold(spec.payment)])
    if spec.channel:
        add("filter", f"only orders of sales channel {spec.channel}",
            [f" {w} " for w in CHAN_LEX.get(spec.channel, [spec.channel])] + [fold(spec.channel)])
    for f in attr_filters:
        if f.attr.is_flag:
            add("filter", f"only items with {f.attr.table}.{f.attr.column} = {'TRUE' if f.value else 'FALSE/blank'}",
                [fold(f.attr.column.replace("_", " "))])
        else:
            add("filter", f"only items with {f.attr.table}.{f.attr.column} = '{f.value}' exactly",
                _value_tokens(f.value) + [fold(f.attr.column.replace("_", " "))])
    if spec.threshold is not None:
        t = spec.threshold
        num = str(int(t)) if float(t).is_integer() else f"{t:g}"
        add("threshold", f"threshold {num} ({'>=' if spec.threshold_inclusive else '>'})",
            [num, f"{t:g}", f"{t / 1e6:g}e6" if t >= 1e6 else ""])
    if spec.k:
        add("rank", f"top {spec.k} by the criterion of the question / rulebook default, descending, ties by code ascending",
            [f"limit {spec.k}", f"head {spec.k}", f"nlargest {spec.k}", f"[ {spec.k}]", f": {spec.k}]"])
    if tctx.shift:
        add("shift", f"only orders created in ca {tctx.shift} (rulebook hours)",
            [" hour ", " hours ", "strftime", "time", "%h", " minute"])
    if tctx.weekday is not None:
        add("weekday", f"only {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][tctx.weekday]}s",
            ["dayofweek", "isodow", "weekday", " dow ", "strftime", "dayname", " %w", " %a", " %u"])
    if spec.intent is not None:
        add("type", f"answer TYPE {spec.answer_type}", checkable=False)
    return R


def coverage(code: str, reqs: list[Requirement]) -> dict:
    res = {r.rid: r.check(code) for r in reqs}
    checked = [v for v in res.values() if v is not None]
    return {"n_checked": len(checked), "n_pass": sum(checked), "failed": [k for k, v in res.items() if v is False],
            "frac": (sum(checked) / len(checked)) if checked else 1.0}


def prompt_block(reqs: list[Requirement]) -> str:
    if not reqs:
        return ""
    lines = ["## REQUIREMENTS (derived from the time ranges, linked values and rulebook above)"]
    lines += [f"- {r.rid}: {r.text}" for r in reqs]
    lines.append("Before the program write one line per requirement: `CHECK R<i>: <how the program satisfies it>` "
                 "(or why it does not apply). Fix the program if a requirement is not satisfied.")
    return "\n".join(lines)
