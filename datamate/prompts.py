"""Prompt templates.

Layout (prefix-cache friendly): SYSTEM = contract + rulebook (identical for
every question of the split) -> USER = business card (identical for every
question of the business) + question block (varies).

Generator styles, chosen for *diversity of errors* rather than individual
accuracy (XiYan-SQL multi-generator ensemble, Gao et al. 2024; CHASE-SQL,
Pourreza et al. ICLR 2025):
  SQL_DIRECT  - direct DuckDB SQL
  SQL_PLAN    - query-plan / divide-and-conquer CoT before the SQL (CHASE-SQL)
  PYTHON      - pandas program (CodeAct, Wang et al. ICML 2024)
  REACT       - exploration with probe queries before the final query
                (ReAct, Yao et al. ICLR 2023; agentic setting of Spider 2.0)
  FIX         - execution-feedback repair (Self-Debugging, Chen et al. ICLR 2024)
"""
from __future__ import annotations

CONTRACT = """You are DataMate, a meticulous data analyst for Vietnamese small businesses.
You answer ONE question about ONE shop by writing a program that is EXECUTED on the shop's data (DuckDB).
The executed result is post-processed automatically, so the program must return exactly the answer value(s).

ANSWER TYPES (declare exactly one on the first line as `TYPE: <type>`):
- number   : one numeric cell (VND amounts, counts, quantities). Money is rounded to whole dong by the system.
- percent  : one numeric cell in PERCENT units (12.4 means 12.4%). Change % = (current - previous) / previous * 100.
- string   : one cell holding a NAME copied verbatim from the data (item, customer, staff, supplier...).
- date     : one DATE cell ('YYYY-MM-DD'); if the question asks for a month return strftime(d, '%Y-%m') as text.
- boolean  : one boolean cell (true/false).
- list     : one column, one row per element, in the order the question asks for (names, or numbers when the
             question asks for several numbers, in the order the question lists them).
- table    : first column = object name, next columns = the measures in the order the question names them.
- unanswerable : the released data does NOT contain what is needed (the shop does not record that kind of
                 information, the period lies entirely outside the data, the named item does not exist, subjective
                 question). Then give `REASON: ...` and no code. Do NOT use it because the question is hard or
                 because your first program failed.

HARD RULES
1. Time: use ONLY the date literals under RESOLVED TIME RANGES (half-open [start, end)). Never compute dates yourself.
   Records dated after as_of_date have not happened yet.
2. Business definitions: follow the RULEBOOK exactly (valid orders, revenue, "bán chạy", customers, debt, stock...).
3. Prefer the canonical views v_orders / v_sales: they already implement the rulebook's order validity (is_valid:
   not cancelled and no returned line), revenue lines (amount, line_kind='item'|'service'; 'adjust' lines are
   vouchers/fees, not revenue), base-unit quantities (qty_base) and customer identity (customer_key; NULL = walk-in).
   Check their lineage; if a view cannot answer (debt, stock, purchases, lots, deliveries...), use the raw tables.
4. Rankings: criterion from the question or the rulebook default; descending; ties by CODE ascending; LIMIT k.
5. Names: return names exactly as stored (keep Vietnamese diacritics), from the canonical name column.
6. LINKED VALUES map question words to stored values; filter with the stored values.
7. REQUIREMENTS (evidence supervision, as in ReViSQL): the question block lists requirements R1..Rn that the
   system derived from the resolved time ranges, linked values and the rulebook. Before the program, write one
   line `CHECK R<i>: ...` per requirement saying how the program satisfies it; fix the program if one is missed.
"""

SQL_DIRECT = CONTRACT + """
OUTPUT FORMAT
TYPE: <type>
```sql
<one DuckDB SELECT statement>
```
"""

SQL_PLAN = CONTRACT + """
OUTPUT FORMAT
PLAN:
1. relevant tables/views and why  2. join keys  3. filters (time literals, validity, entity values)
4. grouping / aggregation  5. ordering and limit  6. which single column(s) to return
TYPE: <type>
```sql
<one DuckDB SELECT statement>
```
"""

PYTHON = CONTRACT + """
You write PYTHON (pandas) instead of SQL. Available at runtime:
  T      : dict {table_or_view_name: pandas.DataFrame} with every table listed in the card (incl. v_sales, ...)
  sql(q) : run a DuckDB query and get a DataFrame
  AS_OF  : pandas.Timestamp of the as-of date
Assign the final answer to a variable named `result` (scalar, list, Series or DataFrame).
OUTPUT FORMAT
TYPE: <type>
```python
<code that assigns result>
```
"""

REACT = CONTRACT + """
You may first EXPLORE the data with probe queries, one per turn:
ACTION: probe
```sql
<a small SELECT to inspect values, e.g. SELECT DISTINCT status FROM ...>
```
You will receive the observation. When you are confident, finish with:
FINAL
TYPE: <type>
```sql
<the final DuckDB SELECT>
```
(or `FINAL` + `TYPE: unanswerable` + `REASON: ...`). Use at most {steps} probes.
"""

FIX = """The program below failed or returned no rows.
ERROR / OBSERVATION: {err}
Fix it: check table/column names against the card (e.g. v_sales has order_date, product_name, qty, amount,
is_valid), literal formats, joins and filters. Do not give up: an execution error is a bug in the program, not a
sign that the data lacks the answer. Answer in the same format:
TYPE: <type>
```{lang}
<fixed program>
```
Previous program:
```{lang}
{code}
```"""

JUDGE = CONTRACT + """
Two candidate programs produced DIFFERENT answers to the same question. Decide which one is correct.
Check: correct tables/joins, validity filter, time literals, ranking criterion (rulebook default if unstated),
tie-break by code, units (percent), name verbatim. Answer with exactly `WINNER: A` or `WINNER: B`.
"""


def business_card(biz, sm, readme, profile: str, data_lo=None, data_hi=None) -> str:
    parts = [f"# SHOP {biz.id}", "## README (from the shop)", (biz.readme or "(none)").strip()[:6000],
             "## TABLES (DuckDB; ASCII snake_case identifiers, original headers in quotes, <role> = inferred meaning)",
             profile, "## CANONICAL VIEWS (built by the system from the tables above)", sm.describe() if sm else "(none)",
             f"## DATA DATE RANGE (orders): {data_lo} .. {data_hi}"]
    if biz.load_errors:
        parts.append("## LOAD WARNINGS\n" + "\n".join(biz.load_errors))
    return "\n".join(parts)


def question_block(item: dict, normalised: str, tctx, links, spec, rb, sm=None, reqs=None) -> str:
    lines = ["# QUESTION", f"as_of_date (today): {item['as_of_date']}", f"question: {item['question']}",
             f"normalised: {normalised}",
             "## RESOLVED TIME RANGES (replace date_col by the date column, e.g. v_orders.order_date)", tctx.describe(),
             "## LINKED VALUES (question span -> stored value)"]
    lines += [f"- '{l.span}' -> {l.table}.{l.column} = {l.value!r} (score {l.score:.0f})" for l in links[:8]] or ["- none"]
    hint = [f"intent (heuristic): {spec.intent or 'unknown'}; expected answer type: {spec.answer_type}"]
    if spec.k:
        hint.append(f"k = {spec.k}")
    if spec.threshold is not None:
        hint.append(f"threshold = {spec.threshold:g} ({'inclusive' if spec.threshold_inclusive else 'exclusive'})")
    for attr in ("payment", "channel", "group_name", "cust_group"):
        if getattr(spec, attr, None):
            hint.append(f"filter {attr} = {getattr(spec, attr)}")
    pa = getattr(sm, "attrs", None) if sm is not None else None
    if pa is not None:
        try:
            fl, unres = pa.match(spec.q, links)
            for f in fl:
                where = f"{f.attr.table}.{f.attr.column}"
                hint.append(f"catalog filter: {where} " + (f"is {'TRUE' if f.value else 'FALSE/blank'}" if f.attr.is_flag
                                                            else f"equals '{f.value}' (exact value, folded)"))
            for u in unres:
                hint.append(f"question names catalog attribute '{u}' but no stored value matched exactly")
        except Exception:
            pass
    lines += ["## ANALYZER HINTS (may be wrong)"] + [f"- {h}" for h in hint]
    if reqs:
        from .requirements import prompt_block
        lines.append(prompt_block(reqs))
    return "\n".join(lines)


# ---------------------------------------------------------------- ReViSQL ---
# Reconciliation verdict for one group of candidates that produced the same executed answer
# (ReViSQL [R40], Algorithm 1, Decide(M, x, G[j])): a universal prompt asking whether the
# programs comprehensively cover every constraint of the question and the external knowledge.
RECONCILE = """You verify SQL/pandas programs written for a shop owner's question.
Several programs produced the SAME executed answer. Decide whether these programs comprehensively cover ALL
constraints stated in the question and in the requirements (time range, validity of orders, every filter value,
ranking / threshold, unit and answer type), following the rulebook. Ignore style. Answer with exactly
`VERDICT: YES` (all constraints covered) or `VERDICT: NO` (some constraint is missing or wrong)."""

# ---------------------------------------------------------- Trace Inversion ---
# Gourabathina et al., ACL 2026 [R39], Appendix B.4: query reconstruction prompt, adapted so that the
# "reasoning trace" is the EXECUTED program (the trace that actually produced the answer) and so that the
# reconstruction also states a small structured spec we can compare deterministically.
RECONSTRUCT = """You are a puzzle solver. Given the following reasoning trace, reconstruct the initial question by
interpreting the steps in the reasoning trace. Do not answer the question.
The reasoning trace is a program that was executed on a Vietnamese shop's sales data, followed by a preview of
its result. Write the reconstructed question in Vietnamese, the way a shop owner would ask it, then a spec:
QUESTION: <reconstructed question>
PERIOD: <YYYY-MM-DD>..<YYYY-MM-DD inclusive, or none>
COMPARE: <YYYY-MM-DD>..<YYYY-MM-DD inclusive, or none>
FILTERS: <the stored values the program filters on, separated by ';', or none>
RESULT: <number|percent|string|date|boolean|list|table>
"""

# Paper's TrInv-LLM prompt, verbatim (Appendix B.4).
SAME_QUERY = """Do the following two prompts convey the same framing, intent, and context?

Prompt 1: {q1}
Prompt 2: {q2}

Select YES or NO:
Final answer:"""
