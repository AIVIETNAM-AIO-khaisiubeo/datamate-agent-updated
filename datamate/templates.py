"""Deterministic template programs over the canonical sales views.

They cover the head of the question distribution (revenue, orders, cancellations,
average order value, % change, best day/month, top items/customers, shares,
quantities, price effects...). Each template
  * fires only when every slot (period, filters, linked item) is resolved,
  * declares the rulebook clauses it implements (`needs`); if the split's
    rulebook does not contain them, the template is disabled (rulebook drift),
  * returns a typed answer plus a `strength` in [0, 1] used as a calibration
    feature, and an `evidence` string written to the per-question log.
Role in the system: one independent "family" in the candidate pool. Agreement
with LLM programs is a strong confidence signal (cross-family self-consistency);
in LLM-free mode the calibrated template strength alone decides answer/abstain.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .attributes import ProductAttributes, _norm as _anorm
from .linker import resolve_entity
from .question import VOCAB, QSpec, unexplained_tokens
from .textnorm import fold

DAY = timedelta(days=1)


@dataclass
class TemplateResult:
    name: str
    answer: tuple            # (type, value) or ("unanswerable", reason)
    strength: float
    detail: str = ""
    plan: dict | None = None  # intent, period labels and filters actually applied (for template inversion)


def half_up(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def r2(x: float) -> float:
    v = round(float(x) + 0.0, 2)
    return 0.0 if v == 0 else v


def money(x: float):
    return int(half_up(float(x)))


class Ctx:
    """Everything a template needs, prepared once per question."""

    def __init__(self, spec: QSpec, tctx, sm, links, rb, data_lo, data_hi, weekday_names):
        self.spec, self.t, self.sm, self.links, self.rb = spec, tctx, sm, links, rb
        self.lo, self.hi = data_lo, data_hi
        self.weekday_names = weekday_names
        self.as_of = tctx.as_of
        self.L = sm.lines
        self.O = sm.orders
        self.applied: list = []          # (kind, value) filters really applied -> TemplateResult.plan
        pa = getattr(sm, "attrs", None)
        try:
            self.attr_filters, self.attr_unresolved = pa.match(spec.q, links) if pa is not None else ([], [])
        except Exception:
            self.attr_filters, self.attr_unresolved = [], []

    # ---- periods ---------------------------------------------------------
    def clip(self, r):
        end = min(r.end, self.as_of + DAY)
        return r.start, end

    def outside(self, r) -> bool:
        if self.lo is None or self.hi is None:
            return False
        s, e = self.clip(r)
        return e <= self.lo or s > self.hi

    def orders(self, r, valid=True, shift=None):
        O = self.O
        s, e = self.clip(r)
        m = (O["order_date"] >= pd.Timestamp(s)) & (O["order_date"] < pd.Timestamp(e))
        if valid:
            m &= O["is_valid"]
        o = O[m]
        return self.shift_filter(o, shift)

    def lines(self, r, valid=True, items_only=False, shift=None):
        L = self.L
        if L is None:
            return None
        s, e = self.clip(r)
        m = (L["order_date"] >= pd.Timestamp(s)) & (L["order_date"] < pd.Timestamp(e))
        if valid:
            m &= L["is_valid"]
        if items_only:
            m &= L["line_kind"].eq("item")
        l = L[m & L["line_kind"].ne("adjust")]
        l = l[~((l["qty"] < 0) | (l["amount"] < 0))]
        return self.shift_filter(l, shift)

    def shift_filter(self, df, shift):
        if not shift or df is None:
            return df
        win = self.rb.get("time.shift_morning" if shift == "sang" else "time.shift_afternoon")
        mins = df["order_ts"].dt.hour * 60 + df["order_ts"].dt.minute
        return df[(mins >= win[0]) & (mins <= win[1])]


def _match_value(series: pd.Series, phrase: str) -> pd.Series:
    """Rows whose value equals the phrase (folded); falls back to containment."""
    f = series.map(lambda v: fold(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "")
    p = fold(phrase)
    exact = f.eq(p)
    if exact.any():
        return exact
    return f.map(lambda v: bool(v) and (f" {p} " in f" {v} " or f" {v} " in f" {p} "))


PAY_LEX = {"cash": ["tien mat", "cash", "tm"], "transfer": ["chuyen khoan", "transfer", "ck", "bank"],
           "ewallet": ["vi dien tu", "ewallet", "e wallet", "momo", "zalopay", "vi", "shopeepay", "vnpay"],
           "cod": ["cod", "giao hang thu tien"], "card": ["the", "card", "quet the", "pos"],
           "debt": ["ghi no", "debt", "no", "cong no", "mua chiu"]}


CHAN_LEX = {"store": ["store", "tai quay", "quay", "cua hang", "offline", "pos", "truc tiep", "instore", "in store"],
            "zalo": ["zalo", "zalo oa", "zalooa"], "phone": ["phone", "dien thoai", "hotline", "goi dien", "call"],
            "shopee": ["shopee"], "lazada": ["lazada"], "tiktok": ["tiktok", "tiktok shop"], "grab": ["grab", "grabfood"],
            "facebook": ["facebook", "fb", "fanpage", "messenger"], "online": ["online", "website", "web", "truc tuyen"]}


def _chan_mask(series: pd.Series, ch: str) -> pd.Series:
    words = CHAN_LEX.get(ch, [ch])

    def hit(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        f = " " + fold(str(v).replace("_", " ")) + " "
        return any(f" {w} " in f for w in words)
    return series.map(hit)


def _pay_mask(series: pd.Series, pay: str) -> pd.Series:
    words = PAY_LEX.get(pay, [pay])
    return series.map(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))
                      and fold(v) in words or any(f" {w} " in f" {fold(v)} " for w in words if len(w) > 2)
                      if v is not None and not (isinstance(v, float) and np.isnan(v)) else False)


# ============================================================== dispatcher ===
def run_templates(spec: QSpec, tctx, sm, links, rb, data_lo, data_hi, weekday_names) -> list[TemplateResult]:
    if sm is None or sm.orders is None or spec.intent is None:
        return []
    c = Ctx(spec, tctx, sm, links, rb, data_lo, data_hi, weekday_names)
    fn = INTENTS.get(spec.intent)
    if fn is None:
        return []
    needs = NEEDS.get(spec.intent, ())
    if not rb.has(*needs):
        return []
    try:
        res = fn(c)
    except Exception as e:  # a template bug must never crash the run
        return [TemplateResult(f"{spec.intent}:error", None, 0.0, repr(e)[:200])]
    if res is None:
        return []
    res = [r for r in (res if isinstance(res, list) else [res]) if r is not None and r.answer is not None]
    plan = {"intent": spec.intent, "period": c.t.primary.label if c.t.primary else None,
            "compare": c.t.comparison.label if c.t.comparison else None,
            "filters": list(dict.fromkeys(c.applied))}
    for r in res:
        r.plan = plan
    # weaken when the question has words the template does not understand
    covered = set()
    for f in c.attr_filters:
        covered.update(str(f.value).split() if not f.attr.is_flag else [])
        covered.update(" ".join(f.attr.phrases).split())
    unk = [t for t in unexplained_tokens(spec.q, links) if t not in covered]
    for r in res:
        if unk and r.answer[0] != "unanswerable":
            r.strength *= 0.6 ** min(len(unk), 4)
            r.detail += f" | unexplained={unk[:6]}"
        r.strength *= _view_quality(sm)
    return res


def _view_quality(sm) -> float:
    q = 1.0
    q *= max(0.3, 1 - 5 * sm.flags.get("unknown_status_share", 0.0))
    if any("no status column" in w for w in sm.warnings):
        q *= 0.85
    if any("failed" in w for w in sm.warnings):
        q *= 0.3
    return q


# ---------------------------------------------------------------- helpers ---
def _period(c: Ctx):
    p = c.t.primary
    if p is None and not c.t.explicit:
        return None
    return p


def _ua_period(c: Ctx, p, name):
    if p is not None and c.outside(p):
        return TemplateResult(f"{name}:outside_data", ("unanswerable", f"kỳ {p.label} nằm ngoài khoảng dữ liệu {c.lo}..{c.hi}"), 0.9)
    return None


def _filtered_lines_and_orders(c: Ctx, p):
    """Apply product / group / customer-group / payment / channel filters. Returns
    (lines, orders, description, ok, filter_level) where filter_level='line'|'order'|None."""
    spec, sm = c.spec, c.sm
    O = c.orders(p, shift=c.t.shift)
    L = c.lines(p, shift=c.t.shift)
    desc, level = [], None
    desc_spans = [f.value for f in c.attr_filters if not f.attr.is_flag]
    # catalog attributes ("hoạt chất X", "thuốc kê đơn", "danh mục bia")
    if c.attr_unresolved:
        return None, None, f"attribute {c.attr_unresolved} named without a known value", False, None
    attr_vals = [f.value for f in c.attr_filters if not f.attr.is_flag]
    if c.attr_filters:
        if L is None:
            return None, None, "no lines", False, None
        L = L[ProductAttributes.mask(L, c.attr_filters)]
        desc += [f.desc for f in c.attr_filters]
        c.applied += [("flag", f.attr.label) if f.attr.is_flag else ("attribute", f.value) for f in c.attr_filters]
        level = "line"
    # product (linked item) — only when the question names one
    link, amb = resolve_entity(c.links, ("product_name", "product_code"), min_score=88)
    grp = [_anorm(spec.group_name) or ""] if spec.group_name else []
    if link is not None and any(f" {_anorm(link.span)} " in f" {v} " for v in attr_vals + grp):
        link = None                       # the "item" is really the attribute / group value
    if link is not None and spec.intent not in ("top_items", "n_items_min_orders", "n_unsold_items", "qty_size"):
        if amb:
            return None, None, "ambiguous item", False, None
        if L is None:
            return None, None, "no lines", False, None
        v = fold(link.value)
        m = L["product_name"].map(lambda x: fold(x) == v if x is not None else False) | \
            L["product_code"].map(lambda x: fold(x) == v if x is not None else False)
        if "product_name_raw" in L:
            m |= L["product_name_raw"].map(lambda x: fold(x) == v if x is not None else False)
        if not m.any():
            # value linked from a catalog but no sale in the period -> real zero
            pass
        L = L[m]
        desc.append(f"item={link.value}")
        c.applied.append(("item", link.value))
        desc_spans += [link.span, link.value]
        level = "line"
    if spec.group_name:
        if L is None or L["product_group"].isna().all():
            return None, None, "no product group column", False, None
        gm = _match_value(L["product_group"], spec.group_name)
        if not gm.any():
            allg = c.L["product_group"].dropna().unique()
            if not any(fold(spec.group_name) == fold(g) or fold(g) in fold(spec.group_name) for g in allg):
                return None, None, f"group '{spec.group_name}' not found", False, None
        L = L[gm]
        desc.append(f"group={spec.group_name}")
        c.applied.append(("group", spec.group_name))
        level = "line"
    if spec.cust_group:
        if O["customer_group"].isna().all():
            return None, None, "no customer group", False, None
        om = _match_value(O["customer_group"], spec.cust_group)
        O = O[om]
        if L is not None:
            L = L[L["order_id"].isin(set(O["order_id"]))]
        desc.append(f"customer_group={spec.cust_group}")
        c.applied.append(("cust_group", spec.cust_group))
        level = level or "order"
    if spec.payment:
        if O["payment"].isna().all():
            return None, None, "no payment column", False, None
        om = _pay_mask(O["payment"], spec.payment)
        O = O[om]
        if L is not None:
            L = L[L["order_id"].isin(set(O["order_id"]))]
        desc.append(f"payment={spec.payment}")
        c.applied.append(("payment", spec.payment))
        level = level or "order"
    # a named staff member / customer (codes or full names only)
    for attr, cols, skip in (("staff", ("staff",), ("top_staff",)),
                             ("customer", ("customer_code", "customer_name"),
                              ("top_customers", "n_customers", "n_customers_min_orders", "pct_customers_min_orders"))):
        if spec.intent in skip:
            continue
        link, amb = resolve_entity(c.links, ("staff", "staff_name") if attr == "staff" else ("customer_code", "customer_name"),
                                   min_score=95)
        if link is None or not (len(link.span.split()) >= 2 or re.search(r"\d", link.span)):
            continue
        if amb:
            return None, None, f"ambiguous {attr}", False, None
        v = fold(link.value)
        keys = {v} | set((sm.aliases.get(attr) or {}).get(v, ()))
        cols = [k for k in cols if k in O.columns]
        if not cols:
            return None, None, f"no {attr} column", False, None
        hit = lambda df: np.logical_or.reduce([df[k].map(lambda x: isinstance(x, str) and fold(x) in keys).to_numpy()
                                               for k in cols])
        if not hit(sm.orders).any():
            return None, None, f"{attr} {link.value} not in orders", False, None
        O = O[hit(O)]
        if L is not None:
            L = L[L["order_id"].isin(set(O["order_id"]))]
        desc.append(f"{attr}={link.value}")
        c.applied.append((attr, link.value))
        desc_spans += [link.span, link.value]
        level = level or "order"
    if spec.channel:
        if O["channel"].isna().all():
            return None, None, "no channel column", False, None
        om = _chan_mask(O["channel"], spec.channel)
        if not _chan_mask(sm.orders["channel"], spec.channel).any():
            return None, None, f"channel {spec.channel} not found", False, None
        O = O[om]
        if L is not None:
            L = L[L["order_id"].isin(set(O["order_id"]))]
        desc.append(f"channel={spec.channel}")
        c.applied.append(("channel", spec.channel))
        level = level or "order"
    if c.t.shift and not sm.flags.get("has_time", True):
        return None, None, "no time of day", False, None
    if c.t.weekday is not None:
        O = O[O["order_date"].dt.weekday == c.t.weekday]
        if L is not None:
            L = L[L["order_date"].dt.weekday == c.t.weekday]
        desc.append(f"weekday={c.t.weekday}")
    # a named entity we did not turn into a filter means the template would answer a different question
    consumed = [_anorm(x) or "" for x in desc_spans] + ([fold(spec.group_name)] if spec.group_name else [])
    for l in c.links:
        if l.score < 97 or (l.role or "") not in ("product_name", "product_code", "customer_name", "customer_code",
                                                  "staff", "staff_name", "supplier", "branch"):
            continue
        sp = _anorm(l.span) or ""
        if (len(sp.split()) < 2 and not re.search(r"\d", sp)) or all(t in VOCAB for t in sp.split()):
            continue
        if any(sp in cs or cs in sp for cs in consumed if cs):
            continue
        return None, None, f"unhandled entity '{l.value}' ({l.role})", False, None
    return L, O, ", ".join(desc), True, level


def _revenue(c: Ctx, p):
    L, O, desc, ok, level = _filtered_lines_and_orders(c, p)
    if not ok:
        return None, desc
    if level == "line":
        return float(L["amount"].sum()), desc
    return float(O["revenue"].sum()), desc


# ================================================================ intents ===
def t_multi_filter(c: Ctx):
    """"doanh thu đơn Zalo và doanh thu đơn tại quầy (lần lượt)" -> list in the order asked."""
    import copy
    p = _period(c)
    if p is None or len(c.spec.multi) < 2:
        return None
    ua = _ua_period(c, p, "revenue")
    if ua:
        return ua
    vals, descs = [], []
    for attr, val in c.spec.multi:
        sp = copy.copy(c.spec)
        sp.channel = sp.payment = None
        setattr(sp, attr, val)
        c2 = Ctx(sp, c.t, c.sm, c.links, c.rb, c.lo, c.hi, c.weekday_names)
        L, O, desc, ok, level = _filtered_lines_and_orders(c2, p)
        if not ok:
            return None
        if c.spec.base_intent == "n_orders":
            vals.append(int(L["order_id"].nunique() if level == "line" else len(O)))
        else:
            vals.append(money(float(L["amount"].sum()) if level == "line" else float(O["revenue"].sum())))
        descs.append(desc)
    return TemplateResult("multi_filter", ("list", vals), 0.8, f"{p.label} {descs}")


def t_revenue(c: Ctx):
    p = _period(c)
    if p is None:
        return None
    ua = _ua_period(c, p, "revenue")
    if ua:
        return ua
    v, desc = _revenue(c, p)
    if v is None:
        return None
    return TemplateResult("revenue", ("number", money(v)), 0.9, f"{p.label} {desc}")


def t_n_orders(c: Ctx):
    p = _period(c)
    if p is None:
        return None
    ua = _ua_period(c, p, "n_orders")
    if ua:
        return ua
    L, O, desc, ok, level = _filtered_lines_and_orders(c, p)
    if not ok:
        return None
    n = L["order_id"].nunique() if level == "line" else len(O)
    return TemplateResult("n_orders", ("number", int(n)), 0.9, f"{p.label} {desc}")


def t_n_cancelled(c: Ctx):
    p = _period(c)
    if p is None:
        return None
    ua = _ua_period(c, p, "n_cancelled")
    if ua:
        return ua
    O = c.orders(p, valid=False, shift=c.t.shift)
    return TemplateResult("n_cancelled", ("number", int(O["is_cancelled"].sum())), 0.85, p.label)


def t_cancel_rate(c: Ctx):
    p = _period(c)
    if p is None:
        return None
    ua = _ua_period(c, p, "cancel_rate")
    if ua:
        return ua
    O = c.orders(p, valid=False, shift=c.t.shift)
    if len(O) == 0:
        return None
    return TemplateResult("cancel_rate", ("percent", r2(100 * O["is_cancelled"].sum() / len(O))), 0.85, p.label)


def t_aov(c: Ctx):
    p = _period(c)
    if p is None:
        return None
    ua = _ua_period(c, p, "aov")
    if ua:
        return ua
    L, O, desc, ok, level = _filtered_lines_and_orders(c, p)
    if not ok or level == "line" or len(O) == 0:
        return None
    return TemplateResult("aov", ("number", money(O["revenue"].sum() / len(O))), 0.9, f"{p.label} {desc}")


def _metric_value(c: Ctx, p, metric):
    if metric == "orders":
        L, O, desc, ok, level = _filtered_lines_and_orders(c, p)
        if not ok:
            return None
        return float(L["order_id"].nunique() if level == "line" else len(O))
    if metric == "quantity":
        L, O, desc, ok, level = _filtered_lines_and_orders(c, p)
        if not ok or L is None:
            return None
        return float(L["qty_base"].sum())
    v, _ = _revenue(c, p)
    return v


def _previous_period(c: Ctx, p):
    from dateutil.relativedelta import relativedelta
    from .timeresolver import TimeRange
    if p.grain == "month":
        ps = p.start - relativedelta(months=1)
        if p.to_date and c.rb.get("time.compare_same_days", True):
            last = p.start - DAY
            end_day = min(c.as_of.day, last.day)
            return TimeRange("tháng trước (cùng số ngày)", ps, date(ps.year, ps.month, end_day) + DAY, "month", "comparison")
        return TimeRange("tháng liền trước", ps, p.start, "month", "comparison")
    if p.grain == "week":
        return TimeRange("tuần liền trước", p.start - 7 * DAY, p.start, "week", "comparison")
    if p.grain == "quarter":
        return TimeRange("quý liền trước", p.start - relativedelta(months=3), p.start, "quarter", "comparison")
    if p.grain == "year":
        return TimeRange("năm liền trước", date(p.start.year - 1, 1, 1), p.start, "year", "comparison")
    if p.grain == "day":
        return TimeRange("ngày liền trước", p.start - DAY, p.start, "day", "comparison")
    return None


def t_pct_change(c: Ctx):
    p, q = c.t.primary, c.t.comparison
    if p is None:
        return None
    if q is None:
        q = _previous_period(c, p)
        if q is None:
            return None
    for r in (p, q):
        ua = _ua_period(c, r, "pct_change")
        if ua:
            return ua
    metric = c.spec.metric or "revenue"
    a, b = _metric_value(c, p, metric), _metric_value(c, q, metric)
    if a is None or b is None or b == 0:
        return None
    return TemplateResult("pct_change", ("percent", r2((a - b) / b * 100)), 0.85, f"{p.label} vs {q.label} ({metric})")


def t_share(c: Ctx):
    p = _period(c)
    if p is None or not (c.spec.group_name or c.spec.cust_group or c.spec.payment or c.spec.channel or c.attr_filters or
                         resolve_entity(c.links, ("product_name", "product_code"), 88)[0] is not None):
        return None
    ua = _ua_period(c, p, "share")
    if ua:
        return ua
    part, desc = _revenue(c, p)
    if part is None:
        return None
    total = float(c.orders(p, shift=c.t.shift)["revenue"].sum())
    if total == 0:
        return None
    return TemplateResult("share", ("percent", r2(100 * part / total)), 0.8, f"{p.label} {desc}")


def t_shift_diff(c: Ctx):
    p = _period(c)
    if p is None or not c.sm.flags.get("has_time"):
        return None
    ua = _ua_period(c, p, "shift_diff")
    if ua:
        return ua
    om, oa = c.shift_filter(c.orders(p), "sang"), c.shift_filter(c.orders(p), "chieu")
    if c.spec.metric == "orders":
        m, a = float(len(om)), float(len(oa))
    else:
        m, a = float(om["revenue"].sum()), float(oa["revenue"].sum())
    if m == 0:
        return None
    # rulebook: comparing ca chiều with ca sáng, the base is ca sáng
    return TemplateResult("shift_diff_pct", ("percent", r2((a - m) / m * 100)), 0.75, f"{p.label} {c.spec.metric}")


def t_best_period(c: Ctx):
    from .timeresolver import TimeRange
    p = next((r for r in c.t.ranges if r.grain == "month_span"), None) or _period(c)
    if p is None:
        return None
    listed = [r for r in c.t.ranges if r.grain == "month" and r.role == "primary"]
    if c.spec.entity == "month" and len(listed) > 1:   # "xét 3 tháng 1/2026, 2/2026 và 3/2026"
        listed.sort(key=lambda r: r.start)
        p = TimeRange("các tháng được liệt kê", listed[0].start, listed[-1].end, "month_span")
    ua = _ua_period(c, p, "best_period")
    if ua:
        return ua
    O = c.orders(p, shift=c.t.shift)
    if O.empty:
        return None
    key = O["order_date"].dt.strftime("%Y-%m-%d" if c.spec.entity == "day" else "%Y-%m")
    g = (O.groupby(key)["revenue"].sum() if c.spec.metric != "orders" else O.groupby(key).size()).rename("m").reset_index()
    g.columns = ["k", "m"]
    g = g.sort_values(["m", "k"], ascending=[c.spec.ascending, True])
    tie = len(g) > 1 and g.iloc[0]["m"] == g.iloc[1]["m"]
    return TemplateResult("best_period", ("date", str(g.iloc[0]["k"])), 0.85 * (0.4 if tie else 1.0), p.label)


def _rank(df, key, metric, k, ascending, code_col):
    g = df.groupby(key, dropna=True)[metric].sum().rename("m").reset_index()
    g = g.sort_values(["m", code_col], ascending=[ascending, True], kind="mergesort").reset_index(drop=True)
    tie = k is not None and len(g) > k and g.loc[k - 1, "m"] == g.loc[k, "m"]
    return (g.head(k) if k else g), tie


def t_top_items(c: Ctx):
    p = _period(c)
    if p is None or c.L is None:
        return None
    ua = _ua_period(c, p, "top_items")
    if ua:
        return ua
    L, O, desc, ok, level = _filtered_lines_and_orders(c, p)
    if not ok:
        return None
    L = L[L["line_kind"].eq("item")]
    if L.empty:
        return None
    metric = "amount" if c.spec.metric == "revenue" else "qty_base"
    k = c.spec.k or 1
    names = L.groupby("product_code")["product_name"].first()
    g, tie = _rank(L, "product_code", metric, k, c.spec.ascending, "product_code")
    out = [str(names.get(cd, cd)) for cd in g["product_code"]]
    s = 0.85 * (0.35 if tie else 1.0)
    if c.spec.answer_type == "string":
        return TemplateResult("top_items", ("string", out[0]), s, f"{p.label} by {metric}")
    return TemplateResult("top_items", ("list", out), s, f"{p.label} by {metric} k={k}")


def t_top_customers(c: Ctx):
    p = _period(c)
    if p is None:
        return None
    ua = _ua_period(c, p, "top_customers")
    if ua:
        return ua
    O = c.orders(p, shift=c.t.shift)
    O = O[O["customer_key"].notna()]
    if O.empty:
        if c.O["customer_key"].isna().all():
            return TemplateResult("top_customers:no_customers", ("unanswerable", "shop không ghi khách"), 0.6)
        return None
    O = O.assign(_one=1)
    metric = "_one" if c.spec.metric == "orders" else "revenue"
    k = c.spec.k or 1
    name = O.groupby("customer_key")["customer_name"].first()
    code = O.groupby("customer_key")["customer_code"].first()
    g = O.groupby("customer_key")[metric].sum().rename("m").reset_index()
    g["code"] = g["customer_key"].map(code).fillna(g["customer_key"])
    g = g.sort_values(["m", "code"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    tie = len(g) > k and g.loc[k - 1, "m"] == g.loc[k, "m"]
    out = [str(name.get(x) or x) for x in g.head(k)["customer_key"]]
    s = 0.8 * (0.35 if tie else 1.0)
    if c.spec.answer_type == "string" or k == 1:
        return TemplateResult("top_customers", ("string", out[0]), s, f"{p.label} by {c.spec.metric}")
    return TemplateResult("top_customers", ("list", out), s, f"{p.label} by {c.spec.metric} k={k}")


def t_top_staff(c: Ctx):
    p = _period(c)
    if p is None or c.O["staff"].isna().all():
        return None
    O = c.orders(p)
    O = O[O["staff"].notna()]
    if O.empty:
        return None
    g = O.groupby("staff")["revenue"].sum().rename("m").reset_index().sort_values(["m", "staff"], ascending=[False, True])
    k = c.spec.k or 1
    out = [str(x) for x in g.head(k)["staff"]]
    if c.spec.answer_type == "string" or k == 1:
        return TemplateResult("top_staff", ("string", out[0]), 0.5, p.label)
    return TemplateResult("top_staff", ("list", out), 0.5, p.label)


def t_n_customers(c: Ctx):
    p = _period(c)
    if p is None:
        return None
    ua = _ua_period(c, p, "n_customers")
    if ua:
        return ua
    O = c.orders(p, shift=c.t.shift)
    if c.O["customer_key"].isna().all():
        return None
    return TemplateResult("n_customers", ("number", int(O["customer_key"].dropna().nunique())), 0.75, p.label)


def _cust_orders(c: Ctx, p):
    O = c.orders(p, shift=c.t.shift)
    O = O[O["customer_key"].notna()]
    return O.groupby("customer_key").size()


def _thr_ok(v, c: Ctx):
    th = c.spec.threshold
    return v >= th if c.spec.threshold_inclusive else v > th


def t_n_customers_min_orders(c: Ctx):
    p = _period(c)
    if p is None or c.spec.threshold is None or c.O["customer_key"].isna().all():
        return None
    ua = _ua_period(c, p, "n_customers_min_orders")
    if ua:
        return ua
    n = _cust_orders(c, p)
    return TemplateResult("n_customers_min_orders", ("number", int(sum(_thr_ok(v, c) for v in n))), 0.8, p.label)


def t_pct_customers_min_orders(c: Ctx):
    p = _period(c)
    if p is None or c.spec.threshold is None or c.O["customer_key"].isna().all():
        return None
    ua = _ua_period(c, p, "pct_customers_min_orders")
    if ua:
        return ua
    n = _cust_orders(c, p)
    if len(n) == 0:
        return None
    return TemplateResult("pct_customers_min_orders", ("percent", r2(100 * sum(_thr_ok(v, c) for v in n) / len(n))), 0.8, p.label)


def t_n_days_threshold(c: Ctx):
    p = _period(c)
    if p is None or c.spec.threshold is None:
        return None
    ua = _ua_period(c, p, "n_days_threshold")
    if ua:
        return ua
    O = c.orders(p, shift=c.t.shift)
    d = O.groupby(O["order_date"].dt.date)["revenue"].sum()
    return TemplateResult("n_days_threshold", ("number", int(sum(_thr_ok(v, c) for v in d))), 0.8, p.label)


def t_n_orders_threshold(c: Ctx):
    p = _period(c)
    if p is None or c.spec.threshold is None:
        return None
    ua = _ua_period(c, p, "n_orders_threshold")
    if ua:
        return ua
    O = c.orders(p, shift=c.t.shift)
    return TemplateResult("n_orders_threshold", ("number", int(sum(_thr_ok(v, c) for v in O["revenue"]))), 0.8, p.label)


def t_n_items_min_orders(c: Ctx):
    p = _period(c)
    if p is None or c.spec.threshold is None or c.L is None:
        return None
    L = c.lines(p, items_only=True, shift=c.t.shift)
    n = L.groupby("product_code")["order_id"].nunique()
    return TemplateResult("n_items_min_orders", ("number", int(sum(_thr_ok(v, c) for v in n))), 0.75, p.label)


def _base_unit_map(c: Ctx) -> dict:
    L = c.L
    bu = {}
    if "base_unit" in L and L["base_unit"].notna().any():
        for cd, u in zip(L["product_code"], L["base_unit"]):
            if u is not None and isinstance(u, str):
                bu.setdefault(cd, fold(u))
    one = L[(L["factor"] == 1) & L["unit"].notna()]
    for cd, u in one.groupby("product_code")["unit"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).items():
        if u is not None:
            bu.setdefault(cd, fold(u))
    return bu


def t_qty(c: Ctx):
    p = _period(c)
    if p is None or c.L is None:
        return None
    ua = _ua_period(c, p, "qty")
    if ua:
        return ua
    L, O, desc, ok, level = _filtered_lines_and_orders(c, p)
    if not ok:
        return None
    L = L[L["line_kind"].eq("item")]
    uw = c.spec.unit_word
    bu = _base_unit_map(c)
    if level == "line" and "item=" in desc:
        units = {bu.get(cd) for cd in L["product_code"].unique()} - {None}
        if uw and units and uw not in units:
            # asked in a bigger unit than the base unit -> not expressible without more rules
            return None
        return TemplateResult("qty_item", ("number", r2(L["qty_base"].sum())), 0.85, f"{p.label} {desc}")
    if not uw:
        return None
    sel = L[L["product_code"].map(lambda cd: bu.get(cd) == uw)]
    if sel.empty and not any(v == uw for v in bu.values()):
        return None
    return TemplateResult("qty_unit", ("number", r2(sel["qty_base"].sum())), 0.7, f"{p.label} unit={uw} {desc}")


def t_qty_size(c: Ctx):
    p = _period(c)
    if p is None or c.L is None or not c.spec.size:
        return None
    L = c.lines(p, items_only=True, shift=c.t.shift)
    sz = c.spec.size.lower()
    m = L["product_name"].map(lambda n: isinstance(n, str) and n.strip().lower().endswith(f"({sz})"))
    if not m.any():
        return None
    return TemplateResult("qty_size", ("number", r2(L[m]["qty_base"].sum())), 0.8, p.label)


def _months_of(c: Ctx):
    """Consecutive calendar months asked about: an explicit span, or the listed months."""
    from dateutil.relativedelta import relativedelta
    from .timeresolver import TimeRange
    span = next((r for r in c.t.ranges if r.grain == "month_span"), None)
    if span is not None:
        out, s = [], span.start
        while s < span.end:
            e = s + relativedelta(months=1)
            out.append(TimeRange(f"{s:%Y-%m}", s, min(e, span.end), "month"))
            s = e
        return out
    return sorted([r for r in c.t.ranges if r.grain == "month"], key=lambda r: r.start)


def t_monotonic(c: Ctx):
    months = _months_of(c)
    if len(months) < 2:
        return None
    vals = []
    for r in months:
        if c.outside(r):
            return None
        v, _ = _revenue(c, r)
        if v is None:
            return None
        vals.append(v)
    inc = "decreasing" not in c.spec.flags
    ok = all((b > a) if inc else (b < a) for a, b in zip(vals, vals[1:]))
    return TemplateResult("monotonic", ("boolean", bool(ok)), 0.75, f"{[m.label for m in months]} {vals}")


def t_revenue_series(c: Ctx):
    from dateutil.relativedelta import relativedelta
    from .timeresolver import TimeRange
    months = [r for r in c.t.ranges if r.grain == "month_span"]
    listed = [r for r in c.t.ranges if r.grain == "month" and r.role == "primary"]
    if months:
        rs = _months_of(c)
    elif len(listed) >= 2:
        rs = listed                      # "tháng 8, tháng 9 và tháng 10 ... lần lượt": the order asked
    else:
        p = c.t.primary
        if p is None or p.grain != "month":
            return None
        k = c.spec.k or 3
        rs = []
        for i in range(k - 1, -1, -1):
            s = p.start - relativedelta(months=i)
            rs.append(TimeRange(f"{s:%Y-%m}", s, s + relativedelta(months=1), "month"))
    out = []
    for r in rs:
        if c.outside(r):
            return None
        v, _ = _revenue(c, r)
        out.append(money(v))
    return TemplateResult("revenue_series", ("list", out), 0.75, f"{[r.label for r in rs]}")


def t_price_effect(c: Ctx):
    p, q = c.t.primary, c.t.comparison
    if p is None or q is None:
        return None
    link, amb = resolve_entity(c.links, ("product_name", "product_code"), min_score=88)
    if link is None or amb:
        return None
    a = _metric_value(c, p, "quantity")
    b = _metric_value(c, q, "quantity")
    if a is None or b is None or b == 0:
        return None
    return TemplateResult("price_effect", ("percent", r2((a - b) / b * 100)), 0.75, f"{link.value} {p.label} vs {q.label}")


def t_unit_price_on_day(c: Ctx):
    p = _period(c)
    if p is None or p.grain != "day" or c.L is None:
        return None
    link, amb = resolve_entity(c.links, ("product_name", "product_code"), min_score=88)
    if link is None or amb:
        return None
    L = c.L
    s, e = p.start, p.end
    m = (L["order_date"] >= pd.Timestamp(s)) & (L["order_date"] < pd.Timestamp(e)) & L["line_kind"].eq("item")
    v = fold(link.value)
    m &= L["product_name"].map(lambda x: fold(x) == v if x is not None else False) | \
        L["product_code"].map(lambda x: fold(x) == v if x is not None else False)
    d = L[m & L["unit_price"].notna() & (L["qty"] > 0)]
    if d.empty:
        return None
    per_base = (d["unit_price"] / d["factor"].replace(0, np.nan)).dropna().round(2)
    if per_base.empty:
        return None
    mode = per_base.mode()
    s_ = 0.8 if per_base.nunique() == 1 else 0.4
    val = float(mode.iloc[0])
    return TemplateResult("unit_price_on_day", ("number", int(val) if val.is_integer() else r2(val)), s_, link.value)


def t_n_unsold(c: Ctx):
    p = _period(c)
    if p is None or c.L is None:
        return None
    L = c.L[c.L["is_valid"] & c.L["line_kind"].eq("item") & (c.L["qty"] > 0)]
    s, e = c.clip(p)
    before = set(L[L["order_date"] < pd.Timestamp(s)]["product_code"])
    during = set(L[(L["order_date"] >= pd.Timestamp(s)) & (L["order_date"] < pd.Timestamp(e))]["product_code"])
    unsold = before - during
    active = c.sm.products
    if active is not None and "active" in active:
        act = set(active.loc[active["active"].astype(bool), "product_code"])
        unsold &= act
    return TemplateResult("n_unsold_items", ("number", len(unsold)), 0.6, p.label)


def _purchases(c: Ctx, p):
    P = c.sm.purchases
    if P is None:
        return None
    s, e = c.clip(p)
    m = (P["import_date"] >= pd.Timestamp(s)) & (P["import_date"] < pd.Timestamp(e)) & ~P["is_cancelled"].fillna(False)
    return P[m]


def t_import_total(c: Ctx):
    p = _period(c)
    if p is None or c.sm.purchases is None:
        return None
    P = _purchases(c, p)
    if P["receipt_total"].notna().any():
        v = P.groupby("receipt_id")["receipt_total"].first().sum()
    else:
        v = P["amount"].sum()
    return TemplateResult("import_total", ("number", money(v)), 0.8, p.label)


def t_import_avg_price(c: Ctx):
    p = _period(c)
    if p is None or c.sm.purchases is None:
        return None
    link, amb = resolve_entity(c.links, ("product_name", "product_code"), min_score=88)
    if link is None or amb:
        return None
    P = _purchases(c, p)
    v = fold(link.value)
    m = P["product_name"].map(lambda x: fold(x) == v if isinstance(x, str) else False) | \
        P["product_code"].map(lambda x: fold(x) == v if isinstance(x, str) else False)
    P = P[m & P["qty"].notna() & (P["qty"] > 0)]
    if P.empty or (P["unit"].notna().any() and P["unit"].map(fold).nunique() > 1):
        return None
    tot = P["amount"] if P["amount"].notna().all() else P["qty"] * P["price"]
    return TemplateResult("import_avg_price", ("number", money(tot.sum() / P["qty"].sum())), 0.75, f"{link.value} {p.label}")


def t_top_supplier(c: Ctx):
    if c.spec.metric == "receipts" and not c.rb.has("main_supplier_by_receipts"):
        return None
    p = _period(c)
    if p is None or c.sm.purchases is None or c.sm.purchases["supplier"].isna().all():
        return None
    P = _purchases(c, p)
    P = P[P["supplier"].notna()]
    if P.empty:
        return None
    if c.spec.metric == "receipts":
        g = P.groupby("supplier")["receipt_id"].nunique()
    else:
        g = (P.groupby(["supplier", "receipt_id"])["receipt_total"].first().groupby("supplier").sum()
             if P["receipt_total"].notna().any() else P.groupby("supplier")["amount"].sum())
    g = g.rename("m").reset_index().sort_values(["m", "supplier"], ascending=[False, True])
    tie = len(g) > 1 and g.iloc[0]["m"] == g.iloc[1]["m"]
    return TemplateResult("top_supplier", ("string", str(g.iloc[0]["supplier"])), 0.75 * (0.4 if tie else 1), p.label)


INTENTS = {
    "import_total": t_import_total, "import_avg_price": t_import_avg_price, "top_supplier": t_top_supplier,
    "multi_filter": t_multi_filter, "revenue": t_revenue, "n_orders": t_n_orders, "n_cancelled": t_n_cancelled, "cancel_rate": t_cancel_rate,
    "aov": t_aov, "pct_change": t_pct_change, "share": t_share, "shift_diff_pct": t_shift_diff,
    "best_period": t_best_period, "top_items": t_top_items, "top_customers": t_top_customers,
    "top_staff": t_top_staff, "n_customers": t_n_customers, "n_customers_min_orders": t_n_customers_min_orders,
    "pct_customers_min_orders": t_pct_customers_min_orders, "n_days_threshold": t_n_days_threshold,
    "n_orders_threshold": t_n_orders_threshold, "n_items_min_orders": t_n_items_min_orders, "qty": t_qty,
    "qty_size": t_qty_size, "monotonic": t_monotonic, "revenue_series": t_revenue_series,
    "price_effect": t_price_effect, "unit_price_on_day": t_unit_price_on_day, "n_unsold_items": t_n_unsold,
}
_SALES = ("valid_orders", "partial_return_whole_order", "revenue_lines", "period_by_order_date")
NEEDS = {
    "import_total": ("import_total_by_receipt_date",), "import_avg_price": ("import_avg_price_qty_weighted",),
    "top_supplier": ("top_supplier_by_amount",),
    "multi_filter": _SALES + ("orders_valid_only",), "revenue": _SALES, "n_orders": _SALES + ("orders_valid_only",), "n_cancelled": ("period_by_order_date",),
    "cancel_rate": ("cancel_rate_all_orders", "period_by_order_date"), "aov": _SALES + ("aov_round_half_up",),
    "pct_change": _SALES + ("pct_change_vs_previous",), "share": _SALES, "shift_diff_pct": _SALES + ("shift_hours",),
    "best_period": _SALES + ("tie_by_code",), "top_items": _SALES + ("bestseller_by_base_qty", "tie_by_code"),
    "top_customers": _SALES + ("customer_by_code", "tie_by_code"), "top_staff": _SALES + ("staff_by_revenue",),
    "n_customers": _SALES + ("customer_by_code",), "n_customers_min_orders": _SALES + ("customer_by_code",),
    "pct_customers_min_orders": _SALES + ("customer_by_code",), "n_days_threshold": _SALES,
    "n_orders_threshold": _SALES, "n_items_min_orders": _SALES, "qty": _SALES + ("bestseller_by_base_qty",),
    "qty_size": _SALES, "monotonic": _SALES, "revenue_series": _SALES, "price_effect": _SALES,
    "unit_price_on_day": (), "n_unsold_items": _SALES + ("unsold_definition",),
}
