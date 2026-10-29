"""Sandboxed execution of generated programs.

* SQL  -> DuckDB cursor on the business connection, SELECT/WITH only, hard
          timeout via connection.interrupt() (execution-guided decoding,
          Wang et al. 2018; execution feedback for self-debugging, Chen et al.
          ICLR 2024 "Teaching LLMs to Self-Debug").
* Python (CodeAct, Wang et al. ICML 2024) -> separate process, read-only
          DuckDB snapshot, wall-clock timeout, result must be assigned to
          `result`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading

import pandas as pd

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|"
                       r"pragma|set|call|checkpoint|vacuum|write_csv|read_csv|read_parquet|read_json|glob)\b", re.I)


def extract_blocks(text: str, lang: str) -> list[str]:
    blocks = re.findall(rf"```{lang}\s*\n?(.*?)```", text or "", re.S | re.I)
    return [b.strip() for b in blocks if b.strip()]


def clean_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    # keep only the last statement if several were emitted
    parts = [p.strip() for p in re.split(r";\s*\n", sql) if p.strip()]
    return parts[-1] if parts else sql


def run_sql(con, sql: str, timeout_s: float = 8.0, max_rows: int = 500) -> dict:
    sql = clean_sql(sql)
    head = sql.lstrip("( \n").lower()
    if not (head.startswith("select") or head.startswith("with") or head.startswith("from")):
        return {"ok": False, "df": None, "empty": True, "err": "only SELECT/WITH queries are allowed"}
    body = re.sub(r"'(?:[^']|'')*'", "''", sql)   # ignore string literals
    if FORBIDDEN.search(body):
        return {"ok": False, "df": None, "empty": True, "err": "forbidden statement/function"}
    cur = con.cursor()
    timer = threading.Timer(timeout_s, lambda: _interrupt(cur))
    timer.start()
    try:
        df = cur.execute(sql).fetchdf()
        return {"ok": True, "df": df.head(max_rows), "empty": len(df) == 0 or bool(df.isna().all().all()),
                "err": None, "nrows": len(df)}
    except Exception as e:
        msg = str(e).split("\n")[0][:400]
        return {"ok": False, "df": None, "empty": True, "err": f"{type(e).__name__}: {msg}"}
    finally:
        timer.cancel()
        try:
            cur.close()
        except Exception:
            pass


def _interrupt(cur):
    try:
        cur.interrupt()
    except Exception:
        pass


PY_PRELUDE = r'''
import sys, json, math, re, datetime as _dt
import duckdb, pandas as pd, numpy as np
_con = duckdb.connect(sys.argv[1], read_only=True)
T = {r[0]: _con.execute(f'SELECT * FROM "{r[0]}"').fetchdf() for r in _con.execute("SHOW TABLES").fetchall()}
def sql(q):
    return _con.execute(q).fetchdf()
AS_OF = pd.Timestamp(sys.argv[2])
result = None
'''

PY_EPILOGUE = r'''
_r = result
def _default(o):
    if isinstance(o, (pd.Timestamp, _dt.date, _dt.datetime)):
        return o.isoformat()[:10]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
if isinstance(_r, pd.Series):
    _r = _r.reset_index()
if isinstance(_r, pd.DataFrame):
    _r = _r.head(500)
    for _c in _r.columns:
        if pd.api.types.is_datetime64_any_dtype(_r[_c]):
            _r[_c] = _r[_c].dt.strftime("%Y-%m-%d")
    print("__RESULT__" + json.dumps({"df": {"columns": [str(c) for c in _r.columns],
                                             "data": _r.astype(object).where(_r.notna(), None).values.tolist()}},
                                     default=_default, ensure_ascii=False))
else:
    print("__RESULT__" + json.dumps({"scalar": _r}, default=_default, ensure_ascii=False))
'''

PY_FORBIDDEN = re.compile(r"\b(os\.system|subprocess|shutil|socket|requests|urllib|open\(|__import__|eval\(|exec\(|"
                          r"to_csv|to_excel|to_parquet|remove|unlink|rmdir)\b")


def run_python(db_path: str, code: str, as_of: str, timeout_s: float = 15.0) -> dict:
    if PY_FORBIDDEN.search(code):
        return {"ok": False, "df": None, "empty": True, "err": "forbidden call in python code"}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(PY_PRELUDE + "\n" + code + "\n" + PY_EPILOGUE)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path, db_path, as_of], capture_output=True, text=True,
                           timeout=timeout_s, env={**os.environ, "OMP_NUM_THREADS": "1"})
        out = [l for l in p.stdout.splitlines() if l.startswith("__RESULT__")]
        if p.returncode != 0 or not out:
            err = (p.stderr.strip().splitlines() or ["no result printed"])[-1][:400]
            return {"ok": False, "df": None, "empty": True, "err": err}
        res = json.loads(out[-1][len("__RESULT__"):])
        if "df" in res:
            df = pd.DataFrame(res["df"]["data"], columns=res["df"]["columns"])
            for c in df.columns:  # recover numeric columns
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() == df[c].notna().sum() and df[c].notna().any():
                    df[c] = conv
        else:
            df = pd.DataFrame({"value": [res["scalar"]]})
        empty = len(df) == 0 or bool(df.isna().all().all())
        return {"ok": True, "df": df, "empty": empty, "err": None, "nrows": len(df)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "df": None, "empty": True, "err": f"timeout after {timeout_s}s"}
    except Exception as e:
        return {"ok": False, "df": None, "empty": True, "err": f"{type(e).__name__}: {e}"[:400]}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def preview(df: pd.DataFrame | None, n: int = 8) -> str:
    if df is None:
        return "(no result)"
    with pd.option_context("display.max_columns", 12, "display.width", 160):
        return f"{len(df)} row(s)\n" + df.head(n).to_string(index=False)[:1500]
