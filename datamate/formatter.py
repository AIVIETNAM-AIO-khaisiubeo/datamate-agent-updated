"""Convert an execution result (DataFrame / scalar) into an answer of the
required type, in the exact shape the scorer expects, plus a canonical key
used to cluster equivalent candidates (execution-result voting)."""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

from .textnorm import fold, metric_norm, nfc

ANSWER_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table")


def round2(v):
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return None
    r = round(f + 0.0, 2)
    if r == 0:
        r = 0.0
    return int(r) if r == int(r) else r


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, str):
        from .textnorm import parse_number_vn
        v = parse_number_vn(v)
        if v is None:
            return None
    try:
        if pd.isna(v):
            return None
        return round2(v)
    except (TypeError, ValueError):
        return None


def _is_num_col(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _name_col(df: pd.DataFrame):
    """Pick the column holding the entity NAME (answers must be names/codes copied from data)."""
    text = [c for c in df.columns if not _is_num_col(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
    if not text:
        return df.columns[0]
    for pref in ("name", "ten", "product_name", "customer_name"):
        for c in text:
            if pref in fold(str(c)).replace("_", " ").split() or str(c).endswith("_name"):
                return c
    # avoid pure code columns if a longer text column exists
    def avglen(c):
        return df[c].dropna().astype(str).str.len().mean() or 0
    return max(text, key=avglen)


def _cell(v):
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, bool):
        return v
    n = _num(v) if not isinstance(v, str) else None
    if n is not None:
        return n
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return nfc(str(v)).strip()


def _date_str(v, month: bool):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if len(s) == 7 and s[4] == "-":
            return s
        try:
            v = pd.Timestamp(s)
        except Exception:
            return None
    try:
        ts = pd.Timestamp(v)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m") if month else ts.strftime("%Y-%m-%d")


def to_frame(result) -> pd.DataFrame | None:
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.reset_index()
    if isinstance(result, (list, tuple)):
        return pd.DataFrame({"value": list(result)})
    if isinstance(result, dict) and "rows" in result:
        return pd.DataFrame(result["rows"], columns=result.get("columns"))
    return pd.DataFrame({"value": [result]})


def format_answer(result, atype: str, month_answer: bool = False):
    """-> (type, answer) or None when the result cannot be expressed as atype."""
    df = to_frame(result)
    if df is None or atype not in ANSWER_TYPES:
        return None
    if atype == "list":
        if df.empty:
            return ("list", [])
        col = _name_col(df) if not all(_is_num_col(df[c]) for c in df.columns) else df.columns[0]
        vals = [_cell(v) for v in df[col].tolist()]
        return None if any(v is None for v in vals) else ("list", vals)
    if atype == "table":
        if df.empty:
            return ("table", {"columns": [str(c) for c in df.columns], "rows": []})
        return ("table", {"columns": [str(c) for c in df.columns],
                          "rows": [[_cell(v) for v in r] for r in df.itertuples(index=False)]})
    if df.empty:
        return None
    row = df.iloc[0]
    if atype in ("number", "percent"):
        if len(df) > 1 and len(df.columns) == 1:
            return None                       # ambiguous: many rows for a scalar question
        nums = [c for c in df.columns if _is_num_col(df[c])] or list(df.columns)
        v = _num(row[nums[-1]] if len(nums) > 1 and len(df.columns) > 1 else row[nums[0]])
        return None if v is None else (atype, v)
    if atype == "boolean":
        v = row.iloc[0]
        if isinstance(v, str):
            f = fold(v)
            if f in ("true", "co", "dung", "yes", "1"):
                return ("boolean", True)
            if f in ("false", "khong", "sai", "no", "0"):
                return ("boolean", False)
            return None
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return ("boolean", bool(v))
    if atype == "date":
        dcols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        v = row[dcols[0]] if dcols else row.iloc[0]
        s = _date_str(v, month_answer)
        return None if s is None else ("date", s)
    if atype == "string":
        col = _name_col(df)
        v = row[col]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return ("string", _cell(v) if isinstance(v, str) else str(_cell(v)))
    return None


def answer_key(typ, ans) -> tuple:
    """Canonical key: two candidates with the same key would be scored identically."""
    if typ in ("number", "percent"):
        return (typ, round(float(ans), 2))
    if typ == "string":
        return (typ, metric_norm(ans))
    if typ == "list":
        return (typ, tuple(metric_norm(x) if isinstance(x, str) else round(float(x), 2) for x in ans))
    if typ == "table":
        rows = tuple(sorted(tuple(metric_norm(x) if isinstance(x, str) else repr(x) for x in r) for r in ans["rows"]))
        return (typ, len(ans["columns"]), rows)
    return (typ, str(ans).lower())


def rank_with_tiebreak(df: pd.DataFrame, metric_col: str, code_col: str, k: int | None = None,
                       ascending: bool = False):
    """Order by metric (desc by default), ties by code ascending.
    Returns (top-k frame, tie_at_boundary)."""
    d = df.sort_values([metric_col, code_col], ascending=[ascending, True], kind="mergesort").reset_index(drop=True)
    tie = False
    if k is not None and len(d) > k:
        tie = bool(round(float(d.loc[k - 1, metric_col]), 6) == round(float(d.loc[k, metric_col]), 6))
    return (d.head(k) if k else d), tie
