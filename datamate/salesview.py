"""Canonical, rulebook-aware sales views built from heterogeneous shop files.

  v_sales   one row per sold line
            order_id, order_date, order_ts, status, is_cancelled, is_returned, is_valid,
            product_code, product_name, product_group, line_kind (item|service|adjust),
            unit, qty, factor, qty_base, base_unit, unit_price, amount, cost_unit,
            customer_key, customer_code, customer_name, customer_group, payment, channel, staff
  v_orders  one row per order (same order-level columns + revenue, n_lines)

Rulebook semantics implemented once here (train/public rulebook §1, §3, §8):
  * order status in {hoàn thành, huỷ, trả hàng, đang giao}; an order with ANY
    returned line (status, negative quantity/amount line, returned qty > 0)
    is a returned order and is excluded as a whole;
  * revenue = sum of line amounts after line discounts; order-level vouchers,
    points, shipping fees are not revenue; "Phí phục vụ" is revenue but not an item;
  * quantities converted to the item's base unit;
  * blank cells in follow-up rows of an order inherit the first row; identical
    rows of the same order are export duplicates (counted once);
  * walk-in customers ("khách lẻ") are not customers.
Every inference is recorded in `lineage`/`warnings` and shown to the LLM, which
may still query raw tables when the view is wrong for a question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loader import name_key
from .readme import parse_conversion_text
from .roles import tag_business
from .textnorm import fold, metric_norm

STATUS_LEX = [  # (status, folded phrases) — checked in this order
    ("tra_hang", ["tra hang", "doi tra", "hang tra ve", "tra lai", "khach tra", "khach mang tra", "return", "returned",
                  "refund", "refunded", "hoan tra", "hoan hang", "da hoan hang", "da tra hang", "tra ve", "doi hang"]),
    ("huy", ["huy", "cancel", "cancelled", "canceled", "void", "bo don", "khach bo", "khong giao duoc", "that bai",
             "failed"]),
    ("dang_giao", ["dang giao", "chua giao", "shipping", "delivering", "dang van chuyen", "cho giao", "in transit",
                   "pending", "dang ship", "dang gui", "cho lay hang", "dang giao dich", "dang xu ly", "processing"]),
    ("hoan_thanh", ["hoan thanh", "xong", "da lay", "completed", "complete", "done", "ok", "thanh cong", "da giao",
                    "success", "da thanh toan", "paid", "finished", "giao thanh cong", "hoan tat", "da ban",
                    "delivered", "closed", "received"]),
]
WALKIN = {"khach le", "kl", "le", "khach vang lai", "vang lai", "khach", "walk in", "walkin", "guest", "khach ghe",
          "khong ten", "khach qua duong", "na", "none", "-"}
ADJUST_RX = re.compile(r"^(voucher|dung \d* ?diem|diem|phi ship|phi giao( hang)?|giam gia don|khuyen mai don|tru diem)\b")
SERVICE_RX = re.compile(r"^phi phuc vu")


def norm_status(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    f = f" {fold(v)} "
    for st, phrases in STATUS_LEX:
        for p in phrases:
            if f" {p} " in f:
                return st
    return "unknown"


def customer_key(name):
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    f = fold(name)
    if not f or f in WALKIN or f.startswith("khach le"):
        return None
    # rulebook: names that differ only in how they are written are one customer
    # ("An - 0932 225 622" == "An (0932225622)"); different letters or digits are different customers
    return re.sub(r"(?<=\d) (?=\d)", "", metric_norm(name))


@dataclass
class SalesModel:
    lines: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    products: pd.DataFrame | None = None
    purchases: pd.DataFrame | None = None
    roles: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    kinds: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    attrs: object = None                 # attributes.ProductAttributes (catalog filters)
    aliases: dict = field(default_factory=dict)   # {"staff"|"customer": {folded name/code: folded code}}

    def describe(self) -> str:
        out = []
        for v, df in (("v_sales", self.lines), ("v_orders", self.orders), ("v_products", self.products),
                      ("v_purchases", self.purchases)):
            if df is not None:
                out.append(f'VIEW "{v}" ({len(df)} rows) built from: {self.lineage.get(v, "?")}')
                out.append("  columns: " + ", ".join(df.columns))
        if self.warnings:
            out.append("VIEW WARNINGS: " + "; ".join(self.warnings[:8]))
        return "\n".join(out) if out else "(no canonical sales view could be built; use raw tables)"


def _cols(t, roles, role):
    return [c for c in t.df.columns if roles.get((t.name, c)) == role]


def _first(t, roles, role):
    c = _cols(t, roles, role)
    return c[0] if c else None


ORDER_LEVEL = ["order_date", "order_datetime", "order_time", "status", "note", "customer_name", "customer_code",
               "payment", "channel", "staff", "customer_group", "order_total", "order_discount", "paid", "debt",
               "shipping_fee", "order_goods_total"]


def build(biz, readme, rulebook=None) -> SalesModel:
    sm = SalesModel()
    roles, ev, kinds = tag_business(biz, readme)
    # context fixes: in customer / staff tables a generic "name" column is the person's name
    for (tn, c), r in list(roles.items()):
        if kinds.get(tn) == "customer" and r == "product_name":
            roles[(tn, c)] = "customer_name"
        if kinds.get(tn) == "customer" and r == "product_code":
            roles[(tn, c)] = "customer_code"
    sm.roles, sm.evidence, sm.kinds = roles, ev, kinds
    tabs = {n: t for n, t in biz.tables.items() if not n.startswith("v_")}
    try:
        _build_products(sm, tabs, readme)
        _build_sales(sm, biz, tabs, readme)
    except Exception as e:  # never break the run; the LLM can still use raw tables
        sm.warnings.append(f"sales view failed: {type(e).__name__}: {e}"[:200])
        sm.lines = sm.orders = None
    try:
        _build_purchases(sm, tabs)
    except Exception as e:
        sm.warnings.append(f"purchase view failed: {type(e).__name__}: {e}"[:200])
        sm.purchases = None
    try:
        _build_aliases(sm, tabs)
    except Exception as e:
        sm.warnings.append(f"aliases failed: {type(e).__name__}: {e}"[:200])
    for name, df in (("v_sales", sm.lines), ("v_orders", sm.orders), ("v_products", sm.products),
                     ("v_purchases", sm.purchases)):
        if df is not None and len(df):
            biz.add_view_table(name, df, sm.lineage.get(name, ""))
    return sm


def _build_aliases(sm, tabs):
    """Staff / customer lists: name -> code, so "nhân viên Bùi Dũng Lan" filters orders that store NV003."""
    R = sm.roles
    for kind, code_role, name_role in (("staff", "staff", "staff_name"), ("customer", "customer_code", "customer_name")):
        amap = {}
        for t in tabs.values():
            if sm.kinds.get(t.name) != kind:
                continue
            cc, nc = _first(t, R, code_role), _first(t, R, name_role)
            if not cc or not nc:
                continue
            for code, name in zip(t.df[cc], t.df[nc]):
                if isinstance(code, str) and isinstance(name, str) and code.strip() and name.strip():
                    amap.setdefault(fold(name), set()).add(fold(code))
                    amap.setdefault(fold(code), set()).add(fold(code))
        sm.aliases[kind] = amap


# ------------------------------------------------------------- products ---
def _build_products(sm, tabs, readme):
    R = sm.roles
    best = None
    for t in tabs.values():
        if sm.kinds.get(t.name) not in ("product", "other", "stock", "conversion"):
            continue
        code, name = _first(t, R, "product_code"), _first(t, R, "product_name")
        if not name:
            continue
        if sm.kinds.get(t.name) == "conversion" and not code:
            continue
        score = len(t.df[name].dropna().unique()) + (1000 if code else 0) + (500 if sm.kinds.get(t.name) == "product" else 0)
        if best is None or score > best[0]:
            best = (score, t, code, name)
    if best is None:
        return
    _, t, code, name = best
    df = pd.DataFrame({"product_code": t.df[code].astype(str).str.strip() if code else None,
                       "product_name": t.df[name]})
    for role, out in (("product_group", "product_group"), ("base_unit", "base_unit"), ("unit", "unit"),
                      ("cost", "cost"), ("list_price", "list_price"), ("unit_price", "list_price")):
        c = _first(t, R, role)
        if c and out not in df:
            df[out] = t.df[c].values
    if "base_unit" not in df and "unit" in df:
        df["base_unit"] = df.pop("unit")
    elif "unit" in df:
        df.pop("unit")
    df = df.dropna(subset=["product_name"]).drop_duplicates()
    if code:
        df = df.drop_duplicates("product_code")
    sm.products = df.reset_index(drop=True)
    sm.lineage["v_products"] = f'"{t.name}"' + (f" code={code}" if code else "") + f" name={name}"


def _factor_cell(f):
    """12 -> (12.0, None); '15 vỉ' -> (15.0, 'vi'); junk -> None."""
    if f is None or (isinstance(f, float) and np.isnan(f)):
        return None
    if isinstance(f, (int, float, np.integer, np.floating)):
        return (float(f), None) if f > 0 else None
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(\D.{0,15})?$", str(f))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return (v, fold(m.group(2)).strip() or None if m.group(2) else None) if v > 0 else None


def _conversion_map(sm, tabs, readme):
    """-> {(key, unit_folded): factor to the base unit}; key is product code or folded name.
    Factors written against an intermediate unit ("1 hộp = 15 vỉ", "1 vỉ = 12 viên") are chained."""
    R, raw = sm.roles, {}
    for t in tabs.values():
        kind = sm.kinds.get(t.name)
        fac = _first(t, R, "factor")
        unit = _first(t, R, "big_unit") or _first(t, R, "unit") or _first(t, R, "import_unit")
        prod = _first(t, R, "product_code") or _first(t, R, "product_name")
        if fac and unit and prod and kind in ("conversion", "product", "other"):
            for p, u, f in zip(t.df[prod], t.df[unit], t.df[fac]):
                fc = _factor_cell(f)
                if p is not None and u is not None and fc is not None:
                    raw[(fold(str(p)), fold(str(u)))] = fc
        # free-text conversions in a catalog column ("1 thùng = 4 bịch; 1 lốc = ...")
        for c in t.df.columns:
            if "quy doi" in name_key(t.orig.get(c, c)) and prod and t.df[c].dtype == object:
                for p, txt in zip(t.df[prod], t.df[c]):
                    for big, f, _base in parse_conversion_text(txt):
                        raw[(fold(str(p)), fold(big))] = (f, None)
    for item, big, f, _base in readme.conversions if readme else []:
        raw.setdefault((fold(item), fold(big)), (f, None))
    conv = {}
    for (p, u), (f, via) in raw.items():
        seen, total = {u}, f
        while via is not None and via not in seen and (p, via) in raw:
            seen.add(via)
            f2, via = raw[(p, via)]
            total *= f2
        conv[(p, u)] = total
    return conv


def _glossary(sm, tabs):
    """Glossary sheets ("Ghi tắt"): shorthand -> full name, full name -> group."""
    R, names, groups = sm.roles, {}, {}
    for t in tabs.values():
        if sm.kinds.get(t.name) != "glossary":
            continue
        full = _first(t, R, "product_name")
        grp = _first(t, R, "product_group")
        if not full:
            continue
        others = [c for c in t.df.columns if c not in (full, grp) and t.df[c].map(lambda v: isinstance(v, str)).mean() > 0.8]
        for c in others[:1]:
            for a, f in zip(t.df[c], t.df[full]):
                if isinstance(a, str) and isinstance(f, str):
                    names[fold(a)] = f
        if grp:
            for f, g in zip(t.df[full], t.df[grp]):
                if isinstance(f, str) and isinstance(g, str):
                    groups[fold(f)] = g
    return names, groups


# ---------------------------------------------------------------- sales ---
def _pick_line_table(sm, tabs):
    R, cands = sm.roles, []
    for t in tabs.values():
        if sm.kinds.get(t.name) != "sales":
            continue
        has_prod = _first(t, R, "product_name") or _first(t, R, "product_code")
        has_q = _first(t, R, "qty") or _first(t, R, "qty_base") or _first(t, R, "line_amount")
        if has_prod and has_q:
            cands.append((len(t.df), t))
    return max(cands, key=lambda x: x[0])[1] if cands else None


def _pick_header(sm, tabs, line_t, oid_vals):
    R, best = sm.roles, None
    for t in tabs.values():
        if t is line_t or sm.kinds.get(t.name) not in ("sales", "other", "debt", "payment"):
            continue
        oid = _first(t, R, "order_id")
        if not oid:
            continue
        useful = [r for r in ORDER_LEVEL if _first(t, R, r)]
        if not useful:
            continue
        right = t.df[oid].astype(str).str.strip()
        if not right.is_unique and sm.kinds.get(t.name) != "sales":
            continue
        cov = pd.Series(oid_vals).isin(set(right)).mean() if len(oid_vals) else 0
        score = cov * 10 + len(useful) + (2 if sm.kinds.get(t.name) == "sales" else 0)
        if cov > 0.5 and (best is None or score > best[0]):
            best = (score, t, oid)
    return best


def _group_fill(df, key, cols):
    if not cols:
        return df
    g = df.groupby(key, sort=False)
    for c in cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(g[c].transform("first"))
    return df


def _build_sales(sm, biz, tabs, readme):
    R = sm.roles
    lt = _pick_line_table(sm, tabs)
    if lt is None:
        _build_orders_only(sm, tabs)
        return
    d = lt.df.copy()
    lin = [f'lines="{lt.name}"']
    oid = _first(lt, R, "order_id")
    if not oid:
        # a running number shared by several consecutive lines ("các dòng cùng STT là một đơn")
        for c in lt.df.columns:
            if R.get((lt.name, c)) == "line_no" and pd.api.types.is_numeric_dtype(d[c]):
                v = d[c].dropna()
                if len(v) and v.duplicated().mean() > 0.1 and (v.diff().dropna() >= 0).mean() > 0.9:
                    oid = c
                    lin.append(f"order id = repeated running number {c}")
                    break
    if oid:
        if d[oid].isna().any():
            prod = _first(lt, R, "product_name") or _first(lt, R, "product_code")
            qv = _first(lt, R, "qty") or _first(lt, R, "qty_base") or _first(lt, R, "unit_price")
            blank = d[oid].isna()
            looks_line = d.loc[blank, prod].notna().mean() > 0.8 if prod else False
            if qv:
                looks_line = looks_line and d.loc[blank, qv].notna().mean() > 0.8
            if looks_line:
                d[oid] = d[oid].ffill()
                lin.append(f"{oid} forward-filled")
            else:   # subtotal / note rows without an order id
                d = d[~blank]
                sm.warnings.append(f"{int(blank.sum())} row(s) without order id dropped")
        d["__oid"] = d[oid].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else
                                (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()))
        # export duplicates: identical rows of the same order count once
        data_cols = [c for c in lt.df.columns if c != "_sheet"]
        dup = d.duplicated(subset=data_cols, keep="first") & d["__oid"].notna()
        if dup.any():
            d = d[~dup]
            sm.warnings.append(f"{int(dup.sum())} duplicate line(s) of the same order removed")
        fill_cols = [c for c in d.columns if R.get((lt.name, c)) in ORDER_LEVEL]
        d = _group_fill(d, "__oid", fill_cols)
    else:
        d["__oid"] = [f"row{i}" for i in range(len(d))]
        sm.warnings.append("no order id column: each line treated as one order")

    # header table join
    hdr = _pick_header(sm, tabs, lt, d["__oid"].dropna().unique())
    hcols = {}
    if hdr:
        _, ht, hoid = hdr
        h = ht.df.copy()
        h["__oid"] = h[hoid].map(lambda v: None if v is None else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()))
        h = h.drop_duplicates("__oid")
        for c in h.columns:
            r = R.get((ht.name, c))
            if r in ORDER_LEVEL and not _first(lt, R, r):
                hcols[c] = r
        if hcols:
            h = h[["__oid"] + list(hcols)].rename(columns={c: f"__h_{c}" for c in hcols})
            d = d.merge(h, on="__oid", how="left")
            lin.append(f'header="{ht.name}" ({", ".join(sorted(set(hcols.values())))})')

    def col(role):
        c = _first(lt, R, role)
        if c is not None and c in d:
            return d[c]
        for hc, r in hcols.items():
            if r == role:
                return d[f"__h_{hc}"]
        return None

    out = pd.DataFrame(index=d.index)
    out["order_id"] = d["__oid"]
    # dates / timestamps
    dtc, dc, tc = col("order_datetime"), col("order_date"), col("order_time")
    ts = None
    if dtc is not None:
        ts = pd.to_datetime(dtc, errors="coerce")
    if dc is not None:
        dd = pd.to_datetime(dc, errors="coerce")
        if ts is None:
            ts = dd
            if tc is not None:
                if pd.api.types.is_datetime64_any_dtype(tc):
                    tt = pd.to_datetime(tc, errors="coerce")
                    ts = dd.dt.normalize() + (tt - tt.dt.normalize())
                else:
                    secs = pd.to_timedelta(tc.astype(str).where(tc.notna(), None), errors="coerce")
                    ts = dd.dt.normalize() + secs.fillna(pd.Timedelta(0))
        else:
            ts = ts.fillna(dd)
    if ts is None:
        raise ValueError("no order date column")
    out["order_ts"] = ts
    has_time = bool((ts.dropna().dt.hour != 0).mean() > 0.3)
    sm.flags["has_time"] = has_time
    # products
    pc, pn = col("product_code"), col("product_name")
    out["product_code"] = pc.astype(str).str.strip().where(pc.notna(), None) if pc is not None else None
    out["product_name"] = pn if pn is not None else None
    gl_name, gl_group = _glossary(sm, tabs)
    if pn is not None and gl_name:
        # shorthand written in a hand-kept ledger ("CL 9D") -> full item name from the shop's glossary sheet
        out["product_name_raw"] = pn
        full = pn.map(lambda n: gl_name.get(fold(n)) if isinstance(n, str) else None)
        out["product_name"] = full.fillna(pn)
        lin.append(f"{int(full.notna().sum())} shorthand item names expanded from the glossary")
    P = sm.products
    if P is not None:
        by_code = dict(zip(P["product_code"].astype(str), P["product_name"])) if P["product_code"].notna().any() else {}
        by_name = {fold(n): c for c, n in zip(P["product_code"], P["product_name"]) if c is not None}
        if pc is None and by_name:
            out["product_code"] = out["product_name"].map(lambda n: by_name.get(fold(n)) if n is not None else None)
        if pn is None and by_code:
            out["product_name"] = out["product_code"].map(by_code)
        elif by_code:
            # canonical catalog name; keep line spelling when unmapped
            canon = out["product_code"].map(by_code)
            out["product_name_raw"] = out["product_name"]
            out["product_name"] = canon.fillna(out["product_name"])
        key = out["product_code"].where(out["product_code"].notna(), out["product_name"].map(lambda n: None if n is None else fold(n)))
        for extra in ("product_group", "base_unit", "cost", "list_price"):
            if extra in P:
                m1 = dict(zip(P["product_code"].astype(str), P[extra])) if P["product_code"].notna().any() else {}
                m2 = {fold(n): v for n, v in zip(P["product_name"], P[extra])}
                out[extra] = [m1.get(str(c)) if c is not None and str(c) in m1 else m2.get(fold(n) if n is not None else None)
                              for c, n in zip(out["product_code"], out["product_name"])]
    if "product_code" not in out or out["product_code"] is None or out["product_code"].isna().all():
        out["product_code"] = out["product_name"]
    lg = col("product_group")
    if lg is not None:
        out["product_group"] = lg
    elif gl_group and ("product_group" not in out or pd.Series(out.get("product_group")).isna().all()):
        out["product_group"] = out["product_name"].map(lambda n: gl_group.get(fold(n)) if isinstance(n, str) else None)
    # kinds of lines
    pn_f = out["product_name"].map(lambda n: fold(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else "")
    out["line_kind"] = np.where(pn_f.str.match(ADJUST_RX.pattern), "adjust",
                                np.where(pn_f.str.match(SERVICE_RX.pattern), "service", "item"))
    # quantities
    q, qb, unit = col("qty"), col("qty_base"), col("unit")
    out["unit"] = unit if unit is not None else out.get("base_unit")
    out["qty"] = pd.to_numeric(q, errors="coerce") if q is not None else (pd.to_numeric(qb, errors="coerce") if qb is not None else 1.0)
    if qb is not None:
        out["qty_base"] = pd.to_numeric(qb, errors="coerce")
        out["factor"] = (out["qty_base"] / out["qty"]).where(out["qty"] != 0)
        lin.append("qty_base from file")
    else:
        conv = _conversion_map(sm, tabs, readme)
        fac = []
        nconv = 0
        for c, n, u, bu in zip(out["product_code"], out["product_name"], out["unit"],
                               out["base_unit"] if "base_unit" in out else [None] * len(out)):
            uf = fold(u) if u is not None and not (isinstance(u, float) and np.isnan(u)) else None
            if uf is None or (bu is not None and isinstance(bu, str) and uf == fold(bu)):
                fac.append(1.0)
                continue
            f = conv.get((fold(c), uf)) if c is not None else None
            if f is None and n is not None:
                f = conv.get((fold(n), uf))
            if f is not None:
                nconv += 1
            fac.append(f if f is not None else 1.0)
        out["factor"] = fac
        out["qty_base"] = out["qty"] * out["factor"]
        lin.append(f"qty_base = qty x factor ({nconv} converted lines, {len(conv)} conversion rules)")
    # money
    amt, price, disc = col("line_amount"), col("unit_price"), col("line_discount")
    if disc is None:
        dc_ = _first(lt, R, "discount")
        disc = d[dc_] if dc_ else None
    out["unit_price"] = pd.to_numeric(price, errors="coerce") if price is not None else None
    if amt is not None:
        out["amount"] = pd.to_numeric(amt, errors="coerce")
        lin.append(f"amount={_first(lt, R, 'line_amount')}")
    elif price is not None:
        out["amount"] = out["qty"] * out["unit_price"] - (pd.to_numeric(disc, errors="coerce").fillna(0) if disc is not None else 0)
        lin.append("amount = qty x unit_price" + (" - discount" if disc is not None else ""))
    else:
        out["amount"] = np.nan
        sm.warnings.append("no line amount / price: revenue unavailable from lines")
    cst = col("cost")
    out["cost_unit"] = pd.to_numeric(cst, errors="coerce") if cst is not None and _first(lt, R, "cost") else None
    # customers
    cc, cn = col("customer_code"), col("customer_name")
    out["customer_code"] = cc.astype(str).where(cc.notna(), None) if cc is not None else None
    out["customer_name"] = cn if cn is not None else None
    if cc is not None and cn is None:
        ct = next((t for t in tabs.values() if sm.kinds.get(t.name) == "customer" and _first(t, R, "customer_code")
                   and _first(t, R, "customer_name")), None)
        if ct is not None:
            m = dict(zip(ct.df[_first(ct, R, "customer_code")].astype(str), ct.df[_first(ct, R, "customer_name")]))
            out["customer_name"] = out["customer_code"].map(m)
    if cc is not None:
        out["customer_key"] = out["customer_code"].where(out["customer_code"].map(lambda v: v is not None and fold(v) not in WALKIN and fold(v) != ""), None)
    else:
        out["customer_key"] = out["customer_name"].map(customer_key) if cn is not None else None
    for r in ("customer_group", "payment", "channel", "staff"):
        s = col(r)
        out[r] = s if s is not None else None
    if out["customer_group"] is None or (isinstance(out["customer_group"], pd.Series) and out["customer_group"].isna().all()):
        ct = next((t for t in tabs.values() if sm.kinds.get(t.name) == "customer" and _first(t, R, "customer_code")
                   and _first(t, R, "customer_group")), None)
        if ct is not None and cc is not None:
            m = dict(zip(ct.df[_first(ct, R, "customer_code")].astype(str), ct.df[_first(ct, R, "customer_group")]))
            out["customer_group"] = out["customer_code"].map(m)
    # status
    stc, note = col("status"), col("note")
    st = stc.map(norm_status) if stc is not None else pd.Series(None, index=out.index)
    if note is not None:
        nst = note.map(lambda v: norm_status(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None)
        nst = nst.where(nst != "unknown", None)
        st = st.where(st.notna() & (st != "unknown"), nst) if stc is not None else nst
    out["status_raw"] = stc if stc is not None else (note if note is not None else None)
    out["line_status"] = st
    neg = (out["qty"] < 0) | (out["amount"] < 0)
    neg &= out["line_kind"].eq("item")
    rq = col("returned_qty")
    ret_line = st.eq("tra_hang") | neg | (pd.to_numeric(rq, errors="coerce").fillna(0) > 0 if rq is not None else False)
    # returned qty recorded in a lot table (e.g. "SL trả (ĐVT cơ bản)")
    for t in tabs.values():
        if t is lt:
            continue
        toid, trq = _first(t, R, "order_id"), _first(t, R, "returned_qty")
        if toid and trq:
            rets = set(t.df.loc[pd.to_numeric(t.df[trq], errors="coerce").fillna(0) > 0, toid].astype(str).str.strip())
            ret_line = ret_line | out["order_id"].isin(rets)
            lin.append(f'returns also from "{t.name}".{trq}')
    g = out.groupby("order_id", sort=False)
    o_ret = ret_line.groupby(out["order_id"]).transform("any")
    o_can = st.eq("huy").groupby(out["order_id"]).transform("any")
    o_ship = st.eq("dang_giao").groupby(out["order_id"]).transform("any")
    known = st.notna() & st.ne("unknown")
    o_known = known.groupby(out["order_id"]).transform("any")
    status = np.where(o_can, "huy", np.where(o_ret, "tra_hang", np.where(o_ship, "dang_giao",
                      np.where(o_known | (stc is None), "hoan_thanh", "unknown"))))
    out["status"] = status
    out["is_cancelled"] = out["status"].eq("huy")
    out["is_returned"] = out["status"].eq("tra_hang")
    out["is_valid"] = ~(out["is_cancelled"] | out["is_returned"])
    unk = st.eq("unknown").mean() if stc is not None else 0.0
    sm.flags["unknown_status_share"] = float(unk)
    if unk > 0.001:
        vals = sorted(set(map(str, stc[st.eq("unknown")].dropna().unique())))[:6]
        sm.warnings.append(f"{unk:.1%} lines with unrecognised status {vals} treated as completed")
    if stc is None and note is None:
        sm.warnings.append("no status column: every order treated as completed")
    # order date = first non-return line of the order (return lines may carry the return time)
    base_ts = out["order_ts"].where(~(neg | st.eq("tra_hang")))
    first_ts = base_ts.groupby(out["order_id"]).transform("min").fillna(g["order_ts"].transform("min"))
    out["order_ts"] = first_ts
    out["order_date"] = first_ts.dt.normalize()
    keep = ["order_id", "order_date", "order_ts", "status", "is_cancelled", "is_returned", "is_valid",
            "product_code", "product_name", "product_group", "line_kind", "unit", "qty", "factor", "qty_base",
            "base_unit", "unit_price", "amount", "cost_unit", "customer_key", "customer_code", "customer_name",
            "customer_group", "payment", "channel", "staff", "status_raw", "line_status"]
    for c in keep:
        if c not in out:
            out[c] = None
    lines = out[keep].reset_index(drop=True)
    sm.lines = lines
    lin.append(f"status={_first(lt, R, 'status') or (list(hcols)[list(hcols.values()).index('status')] if 'status' in hcols.values() else None) or ('note' if note is not None else 'none')}")
    sm.lineage["v_sales"] = "; ".join(lin)
    # orders
    rev_line = lines["amount"].where(lines["line_kind"].ne("adjust") & lines["is_valid"] | lines["line_kind"].ne("adjust"))
    lines_rev = lines.assign(_rev=lines["amount"].where(lines["line_kind"].ne("adjust"), 0.0))
    lines_rev = lines_rev.assign(_rev=lines_rev["_rev"].where(~((lines_rev["qty"] < 0) | (lines_rev["amount"] < 0)), 0.0))
    agg = lines_rev.groupby("order_id", sort=False).agg(
        order_date=("order_date", "first"), order_ts=("order_ts", "first"), status=("status", "first"),
        is_cancelled=("is_cancelled", "first"), is_returned=("is_returned", "first"), is_valid=("is_valid", "first"),
        customer_key=("customer_key", "first"), customer_code=("customer_code", "first"),
        customer_name=("customer_name", "first"), customer_group=("customer_group", "first"),
        payment=("payment", "first"), channel=("channel", "first"), staff=("staff", "first"),
        revenue=("_rev", "sum"), n_lines=("product_name", "size"))
    sm.orders = agg.reset_index()
    sm.lineage["v_orders"] = "v_sales grouped by order_id; revenue = sum(amount) of item+service lines (sale lines only)"
    del rev_line
    _self_checks(sm, out, col, hcols)


def _self_checks(sm, out, col, hcols):
    """Internal consistency of the inferred roles (used as confidence features):
    qty x unit_price - discount == amount on lines; sum of line amounts == order goods total."""
    try:
        q, pr, am = out["qty"], out["unit_price"], out["amount"]
        if pr is not None and isinstance(pr, pd.Series) and pr.notna().any() and am.notna().any():
            ok = ((q * pr - am).abs() <= 1) | ((q * pr - am).abs() <= 0.002 * am.abs())
            disc_ok = (am <= q * pr + 1)  # line discounts only lower the amount
            sm.flags["line_math_agree"] = float((ok | disc_ok)[q.notna() & pr.notna() & am.notna()].mean())
        gt = col("order_goods_total")
        if gt is not None:
            per = out.assign(_gt=pd.to_numeric(gt, errors="coerce")).groupby("order_id").agg(
                s=("amount", lambda x: x[x > 0].sum()), g=("_gt", "first"))
            per = per[per["g"].notna()]
            if len(per):
                sm.flags["order_total_agree"] = float(((per["s"] - per["g"]).abs() <= 1).mean())
    except Exception:
        pass


def _build_orders_only(sm, tabs):
    """Shops that only keep one row per order (no item lines)."""
    R = sm.roles
    best = None
    for t in tabs.values():
        if sm.kinds.get(t.name) != "sales":
            continue
        oid = _first(t, R, "order_id")
        money = _first(t, R, "order_goods_total") or _first(t, R, "line_amount") or _first(t, R, "order_total")
        dcol = _first(t, R, "order_date") or _first(t, R, "order_datetime")
        if money and dcol and (best is None or (bool(oid), len(t.df)) > (bool(best[1]), len(best[0].df))):
            best = (t, oid, money, dcol)
    if best is None:
        sm.warnings.append("no sales table recognised")
        return
    t, oid, money, dcol = best
    d = t.df
    if oid:
        o = pd.DataFrame({"order_id": d[oid].astype(str).str.strip()})
    else:   # ledgers with one row per ticket and no ticket number
        o = pd.DataFrame({"order_id": [f"row{i}" for i in range(len(d))]}, index=d.index)
        sm.warnings.append(f'"{t.name}": no order id, each row treated as one order')
    ts = pd.to_datetime(d[dcol], errors="coerce")
    tc = _first(t, R, "order_datetime")
    if tc and tc != dcol:
        ts = pd.to_datetime(d[tc], errors="coerce").fillna(ts)
    o["order_ts"] = ts
    o["order_date"] = ts.dt.normalize()
    stc, note = _first(t, R, "status"), _first(t, R, "note")
    st = d[stc].map(norm_status) if stc else (d[note].map(norm_status) if note else pd.Series("hoan_thanh", index=d.index))
    st = st.fillna("hoan_thanh").replace({"unknown": "hoan_thanh"} if not stc else {})
    o["status"] = st
    o["is_cancelled"] = st.eq("huy")
    o["is_returned"] = st.eq("tra_hang")
    o["is_valid"] = ~(o["is_cancelled"] | o["is_returned"])
    cc, cn = _first(t, R, "customer_code"), _first(t, R, "customer_name")
    o["customer_code"] = d[cc].astype(str) if cc else None
    o["customer_name"] = d[cn] if cn else None
    o["customer_key"] = o["customer_code"] if cc else (o["customer_name"].map(customer_key) if cn else None)
    for r in ("customer_group", "payment", "channel", "staff"):
        c = _first(t, R, r)
        o[r] = d[c] if c else None
    o["revenue"] = pd.to_numeric(d[money], errors="coerce")
    o["n_lines"] = None
    sm.orders = o.drop_duplicates("order_id").reset_index(drop=True)
    sm.lineage["v_orders"] = f'"{t.name}" one row per order; revenue={money}'
    sm.flags["orders_only"] = True
    if R.get((t.name, money)) == "order_total":
        sm.warnings.append("revenue taken from the order total column (may include order discounts/fees)")


# ------------------------------------------------------------ purchases ---
def _build_purchases(sm, tabs):
    """v_purchases: one row per purchase line (receipt_id, import_date, supplier, product_code/name,
    unit, qty, price, amount, receipt_total, is_cancelled). Header/detail files are joined on the
    receipt id; product names come from the catalog when lines only carry codes."""
    R = sm.roles
    pts = [t for t in tabs.values() if sm.kinds.get(t.name) == "purchase"]
    if not pts:
        return
    lines = [t for t in pts if (_first(t, R, "product_code") or _first(t, R, "product_name"))
             and (_first(t, R, "import_qty") or _first(t, R, "qty"))]
    if not lines:
        return
    lt = max(lines, key=lambda t: len(t.df))
    d = lt.df.copy()

    def c(t, *roles):
        for r in roles:
            x = _first(t, R, r)
            if x:
                return x
        return None
    rid = c(lt, "receipt_id", "order_id")
    out = pd.DataFrame(index=d.index)
    out["receipt_id"] = d[rid].astype(str).str.strip() if rid else [f"r{i}" for i in range(len(d))]
    hdr = None
    if rid:
        for h in pts:
            hr = c(h, "receipt_id", "order_id")
            if h is lt or not hr or not c(h, "import_date", "order_date"):
                continue
            if out["receipt_id"].isin(set(h.df[hr].astype(str).str.strip())).mean() > 0.8:
                hdr = (h, hr)
                break
    def pick(role_list):
        col_ = c(lt, *role_list)
        if col_:
            return d[col_]
        if hdr:
            h, hr = hdr
            hc = c(h, *role_list)
            if hc:
                m = dict(zip(h.df[hr].astype(str).str.strip(), h.df[hc]))
                return out["receipt_id"].map(m)
        return None
    dt_ = pick(["import_date", "order_date", "order_datetime"])
    if dt_ is None:
        return
    out["import_date"] = pd.to_datetime(dt_, errors="coerce").dt.normalize()
    sup = pick(["supplier"])
    out["supplier"] = sup if sup is not None else None
    stc = pick(["import_status", "status"])
    note = pick(["note"])
    st = stc.map(norm_status) if stc is not None else pd.Series(None, index=out.index)
    out["is_cancelled"] = st.eq("huy")
    if note is not None:
        out["is_cancelled"] |= note.map(lambda v: norm_status(v) == "huy" if isinstance(v, str) else False)
    pc, pn = c(lt, "product_code"), c(lt, "product_name")
    out["product_code"] = d[pc].astype(str).str.strip() if pc else None
    out["product_name"] = d[pn] if pn else None
    P = sm.products
    if P is not None and pc and not pn:
        out["product_name"] = out["product_code"].map(dict(zip(P["product_code"].astype(str), P["product_name"])))
    if pn and not pc and P is not None:
        m = {fold(n): cd for cd, n in zip(P["product_code"], P["product_name"])}
        out["product_code"] = out["product_name"].map(lambda n: m.get(fold(n)) if isinstance(n, str) else None)
    u = c(lt, "import_unit", "unit")
    out["unit"] = d[u] if u else None
    q = c(lt, "import_qty", "qty")
    out["qty"] = pd.to_numeric(d[q], errors="coerce")
    pr = c(lt, "import_price", "unit_price", "cost")
    out["price"] = pd.to_numeric(d[pr], errors="coerce") if pr else np.nan
    am = c(lt, "import_amount", "line_amount")
    out["amount"] = pd.to_numeric(d[am], errors="coerce") if am else out["qty"] * out["price"]
    tot = pick(["import_total", "order_total"])
    out["receipt_total"] = pd.to_numeric(tot, errors="coerce") if tot is not None else None
    sm.purchases = out.reset_index(drop=True)
    sm.lineage["v_purchases"] = f'lines="{lt.name}"' + (f'; header="{hdr[0].name}"' if hdr else "") + \
        f"; qty={q}; price={pr}; amount={am or 'qty*price'}"
