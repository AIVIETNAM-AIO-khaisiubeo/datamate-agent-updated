"""Product attributes recorded in a shop's catalog tables, usable as filters.

Shops record per-item attributes beside the name: an active ingredient, a brand, a
category, a "prescription" flag, a size... Questions filter on them ("thuốc có hoạt
chất Cetirizin", "hàng không kê đơn", "các món nhóm lẩu"). This module turns every
catalog column into either

  * a VALUE attribute: the question names one of its values exactly
    (rulebook: "hoạt chất ghi đúng như vậy" -> exact match, a combination is its
    own value), linked through the value index; or
  * a FLAG attribute: a boolean-like column whose meaning ("kê đơn") comes from its
    header and from README sentences like "`thong_tin_thuoc.BanTheoDon` đúng
    (TRUE/1) là thuốc kê đơn"; negations ("không kê đơn", "trừ thuốc kê đơn ra")
    select the complement.

Templates call `match()`; a question that names an attribute column ("hoạt chất")
without a value we can resolve is reported as unresolved so templates do not answer
it unfiltered (schema-linking guard in the spirit of CHESS [R7] / PRACTIQ [R25]).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loader import name_key
from .question import VOCAB
from .textnorm import fold

_TRUE = {"true", "1", "1.0", "co", "x", "yes", "y", "dung", "v"}
_FALSE = {"false", "0", "0.0", "khong", "no", "n", "sai", ""}
_FILLER = {"thuoc", "la", "co", "can", "phai", "hang", "san", "pham", "mat", "duoc", "so", "ma", "ten", "loai", "id"}
_NEG = r"(?:khong|ko|chua|ngoai|tru|tru cac|tru nhung)"
_SKIP_ROLES = {"product_code", "product_name", "unit", "base_unit", "unit_price", "price", "cost_unit", "qty",
               "qty_base", "factor", "line_amount", "order_id", "order_date", "order_datetime", "customer_code",
               "customer_name", "staff", "stock_qty"}


@dataclass
class Attr:
    table: str
    column: str
    label: str                      # folded header (+ README phrase for flags)
    phrases: list                   # folded phrases that name the attribute in a question
    is_flag: bool
    role: str | None = None
    by_key: dict = field(default_factory=dict)   # folded product code / name -> folded value (or bool)


@dataclass
class AttrFilter:
    attr: Attr
    value: object                   # folded value, or bool for flags
    desc: str


def _norm(x) -> str | None:
    """Folded alphanumeric tokens: 'Paracetamol + Cafein' -> 'paracetamol cafein' (matches qnorm output)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    t = " ".join(re.findall(r"[0-9a-z]+", fold(str(x))))
    return t or None


def _flagify(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return {0: False, 1: True}.get(float(v))
    f = fold(str(v))
    if f in _TRUE:
        return True
    if f in _FALSE:
        return False
    return None


def _readme_flag_phrases(readme_text: str) -> dict:
    """'`thong_tin_thuoc.BanTheoDon` đúng (TRUE/1) là thuốc kê đơn' -> {'ban theo don': 'thuoc ke don'}"""
    out = {}
    rx = re.compile(r"(?:`([^`]+)`|\*\*([^*]+)\*\*)\s*(?:đúng|=\s*true|true|có|bằng 1)\s*(?:\([^)]*\))?\s*(?:là|nghĩa là)\s+([^.;\n]+)",
                    re.I)
    for m in rx.finditer(readme_text or ""):
        col = (m.group(1) or m.group(2)).split(".")[-1]
        out[name_key(col)] = fold(m.group(3))
    return out


# everyday paraphrases of common catalog flags (kept tiny and generic)
_FLAG_SYNONYMS = {"ke don": ["can don", "don bac si", "toa bac si", "theo don", "can toa", "theo toa"]}


def _phrases(label: str) -> list:
    toks = [t for t in label.split() if t not in _FILLER]
    out = {label}
    if toks:
        out.add(" ".join(toks))
    for k, syn in _FLAG_SYNONYMS.items():
        if k in label:
            out.update(syn)
    return sorted((p for p in out if len(p) >= 4), key=len, reverse=True)


class ProductAttributes:
    def __init__(self, biz, sm, readme_text: str = ""):
        self.attrs: list[Attr] = []
        flag_meaning = _readme_flag_phrases(readme_text)
        kinds = getattr(sm, "kinds", {}) or {}
        roles = getattr(sm, "roles", {}) or {}
        for tn, t in biz.tables.items():
            if tn.startswith("v_") or kinds.get(tn) not in ("product",):
                continue
            df = t.df
            keycols = [c for c in df.columns if roles.get((tn, c)) in ("product_code", "product_name")]
            if not keycols or len(df) == 0:
                continue
            keys = [df[c].map(_norm) for c in keycols]
            for c in df.columns:
                if c in keycols or c == "_sheet" or roles.get((tn, c)) in _SKIP_ROLES:
                    continue
                s = df[c]
                nn = s.dropna()
                if nn.empty or nn.nunique() < 2 or pd.api.types.is_datetime64_any_dtype(s):
                    continue
                label = name_key(t.orig.get(c, c))
                flags = nn.map(_flagify)
                is_flag = bool(flags.notna().all() and nn.nunique() <= 3)
                if not is_flag and (pd.api.types.is_numeric_dtype(s) or nn.nunique() > max(200, 0.9 * len(nn))):
                    continue
                if is_flag:
                    phrases = _phrases(label)
                    if label in flag_meaning:
                        phrases = sorted(set(phrases) | set(_phrases(flag_meaning[label])), key=len, reverse=True)
                    vals = s.map(lambda v: bool(_flagify(v)))
                else:
                    phrases = _phrases(label)
                    vals = s.map(_norm)
                a = Attr(tn, c, label, phrases, is_flag, role=roles.get((tn, c)))
                for kser in keys:
                    for k, v in zip(kser, vals):
                        if k:
                            a.by_key.setdefault(k, v)
                self.attrs.append(a)

    # ------------------------------------------------------------------
    def match(self, qf: str, links) -> tuple[list[AttrFilter], list[str]]:
        """-> (filters, unresolved attribute labels named in the question)."""
        filters, unresolved = [], []
        qf = _norm(qf) or ""
        qs = f" {qf} "
        # value attributes: a catalog value written in the question exactly (longest wins)
        exact = []
        for a in self.attrs:
            if a.is_flag:
                continue
            for v in {x for x in a.by_key.values() if isinstance(x, str) and x}:
                if len(v) >= 3 and v not in _FILLER and v not in VOCAB and f" {v} " in qs:
                    exact.append((len(v), v, a))
        exact.sort(key=lambda x: (-x[0], x[2].label))
        # a value inside a named item ("Sơn" in "Bồn nước inox Sơn Hà") is part of the item, not a filter
        used_spans = [_norm(l.span) or "" for l in links
                      if (l.role or "").startswith("product_") and l.score >= 95 and len(l.span.split()) >= 2]
        for _, v, a in exact:
            if any(f" {v} " in f" {u} " for u in used_spans):
                continue
            used_spans.append(v)
            if a.column in {f.attr.column for f in filters if f.attr.table == a.table}:
                continue        # one value per attribute
            filters.append(AttrFilter(a, v, f"{a.label}={v}"))
        named = {f.attr.column for f in filters}
        for a in self.attrs:
            if a.is_flag:
                for p in a.phrases:
                    m = re.search(rf"\b{re.escape(p)}\b", qf)
                    if not m:
                        continue
                    before = qf[:m.start()].split()[-4:]
                    neg = bool(re.search(rf"\b{_NEG}\b", " ".join(before))) or \
                        bool(re.search(rf"\btru\b.{{0,40}}{re.escape(p)}.{{0,30}}\bra\b", qf))
                    filters.append(AttrFilter(a, not neg, f"{a.label}={'không ' if neg else ''}{p}"))
                    break
            elif a.column not in named and a.role != "product_group":   # "nhóm X" is handled by the group filter
                if any(len(p.split()) >= 2 and re.search(rf"\b{re.escape(p)}\b", qf) for p in a.phrases):
                    unresolved.append(a.label)
        return filters, unresolved

    @staticmethod
    def mask(lines: pd.DataFrame, filters: list[AttrFilter]) -> pd.Series:
        m = pd.Series(True, index=lines.index)
        codes = lines["product_code"].map(_norm)
        names = lines["product_name"].map(_norm)
        for f in filters:
            bk = f.attr.by_key
            val = [bk.get(c, bk.get(n)) if (c in bk or n in bk) else None for c, n in zip(codes, names)]
            if f.attr.is_flag:
                # rulebook: everything not marked is the complement ("mọi sản phẩm còn lại")
                hit = [bool(v) is f.value if v is not None else (f.value is False) for v in val]
            else:
                hit = [v == f.value for v in val]
            m &= pd.Series(hit, index=lines.index)
        return m
