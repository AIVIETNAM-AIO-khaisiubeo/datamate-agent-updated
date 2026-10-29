"""Synthetic DataMate-style benchmark generator (dev set + calibration data).

Following the data-synthesis recipe of OmniSQL (Li et al., VLDB 2025) and the
unanswerable taxonomy of PRACTIQ (Dong et al., NAACL 2025), we generate:
  * businesses with deliberately MESSY files (title rows, merged 2-row headers
    avoided, 'Tổng cộng' rows, VND strings, day-first dates, name variants
    between files, cancelled/returned orders, multi-sheet Excel, SQLite);
  * questions in everyday Vietnamese (with/without diacritics, teencode);
  * gold answers computed by an INDEPENDENT oracle on the clean data (before
    the messy files are written), so loader bugs show up as errors;
  * unanswerable questions: nonexistent metric (no cost column -> profit),
    nonexistent filter value (product not sold by this business), period
    outside the data.
Ranking questions with a tie at the k boundary are dropped, exactly like the
official test sets.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import date, timedelta

import pandas as pd

from .lunar import LunarCalendar, solar_to_lunar_algo
from .textnorm import fold

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

CATALOG = {
    "vlxd": {
        "desc": "Cửa hàng vật liệu xây dựng {name} tại {city}. Bán xi măng, sắt thép, gạch, cát đá, sơn cho "
                "thợ và nhà thầu; nhiều khách mua chịu và trả dần (công nợ).",
        "unit": "bao",
        "items": [("Xi măng Hà Tiên PCB40 bao 50kg", 92000, "bao"), ("Xi măng Insee Đa Dụng bao 50kg", 95000, "bao"),
                  ("Thép Hòa Phát D10 cây 11.7m", 118000, "cây"), ("Thép Pomina D12 cây 11.7m", 172000, "cây"),
                  ("Gạch ống Tuynel 8x8x18", 1450, "viên"), ("Gạch thẻ Đồng Nai 4x8x18", 1250, "viên"),
                  ("Cát xây dựng hạt to", 380000, "m3"), ("Đá 1x2 xanh", 420000, "m3"),
                  ("Sơn Dulux nội thất 5L", 765000, "thùng"), ("Sơn chống thấm Kova CT11A 4kg", 520000, "thùng"),
                  ("Ống nhựa Bình Minh phi 21", 21000, "cây"), ("Tôn lạnh Hoa Sen 0.4mm", 98000, "m")],
        "customers": ["Anh Tư thợ hồ", "Công ty XD Phúc Thịnh", "Chị Lan Bình Chánh", "Anh Hùng nhà thầu",
                      "Cô Ba Tân Phú", "Anh Dũng Củ Chi", "Công ty TNHH Minh Long", "Chú Sáu Hóc Môn"],
    },
    "trasua": {
        "desc": "Chuỗi quán trà sữa {name} có 3 chi nhánh ở {city}. Dữ liệu bán hàng xuất từ phần mềm POS (SQLite).",
        "unit": "ly",
        "items": [("Trà sữa trân châu đường đen", 35000, "ly"), ("Trà đào cam sả", 39000, "ly"),
                  ("Cà phê sữa đá", 29000, "ly"), ("Bạc xỉu", 32000, "ly"), ("Trà sữa matcha", 42000, "ly"),
                  ("Sữa tươi trân châu", 38000, "ly"), ("Trà vải", 36000, "ly"), ("Hồng trà kem cheese", 45000, "ly"),
                  ("Nước ép cam", 40000, "ly"), ("Sinh tố bơ", 45000, "ly")],
        "branches": ["CN Quận 1", "CN Thủ Đức", "CN Gò Vấp"],
    },
    "nhathuoc": {
        "desc": "Nhà thuốc {name} ở {city}. Quản lý thuốc theo lô có hạn sử dụng; hóa đơn bán lẻ ghi trong file CSV.",
        "unit": "hộp",
        "items": [("Paracetamol 500mg hộp 10 vỉ", 25000, "hộp"), ("Panadol Extra hộp 12 vỉ", 185000, "hộp"),
                  ("Berberin 100mg lọ 100 viên", 18000, "lọ"), ("Vitamin C 500mg lọ 100 viên", 45000, "lọ"),
                  ("Amoxicillin 500mg hộp 10 vỉ", 68000, "hộp"), ("Oresol gói 27.9g", 3500, "gói"),
                  ("Smecta gói 3g", 4500, "gói"), ("Efferalgan 500mg hộp 16 viên", 52000, "hộp"),
                  ("Siro ho Prospan 100ml", 89000, "chai"), ("Salonpas hộp 20 miếng", 32000, "hộp"),
                  ("Nước muối sinh lý 0.9% 500ml", 6000, "chai")],
    },
}
NAMES = ["Phát Đạt", "Hưng Thịnh", "Minh Anh", "Thành Công", "An Khang", "Kim Ngân", "Bảo Châu", "Tâm An"]
CITIES = ["TP.HCM", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Biên Hòa", "Nha Trang"]


def money_str(v, rng):
    v = int(round(v))
    style = rng.random()
    if style < 0.5:
        return f"{v:,}".replace(",", ".") + "đ"
    if style < 0.8:
        return f"{v:,}".replace(",", ".")
    return str(v)


def strip_accents_lower(s):
    return fold(s)


def variant(name, rng):
    r = rng.random()
    if r < 0.4:
        return name.lower()
    if r < 0.6:
        return name.upper()
    return name + " "  # trailing space (classic Excel mess)


# ----------------------------------------------------------------- business --
def gen_lines(kind, items, as_of, rng, customers=None, branches=None, start=date(2024, 1, 1)):
    rows, oid = [], 1000
    d = start
    while d <= as_of:
        base = rng.randint(4, 12) if kind != "trasua" else rng.randint(15, 35)
        if d.weekday() >= 5:
            base = int(base * 1.4)
        for _ in range(base):
            oid += 1
            status = rng.choices(["Hoàn thành", "Đã hủy", "Trả hàng"], [0.9, 0.07, 0.03])[0]
            cust = rng.choice(customers) if customers else None
            br = rng.choice(branches) if branches else None
            for i in rng.sample(range(len(items)), min(len(items), rng.randint(1, 3))):  # distinct items per order
                qty = rng.randint(1, 4) if kind != "vlxd" else rng.choice([1, 2, 5, 10, 20, 50, 100])
                price = items[i][1]
                rows.append(dict(order_id=f"HD{oid}", date=d, status=status, item=i, qty=qty, price=price,
                                 amount=qty * price, cust=cust, branch=br))
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def write_business(root, bid, kind, rng, as_of):
    cat = CATALOG[kind]
    name, city = rng.choice(NAMES), rng.choice(CITIES)
    items = cat["items"][:]
    rng.shuffle(items)
    items = items[: rng.randint(7, len(items))]
    codes = [f"{'SP' if kind != 'nhathuoc' else 'TH'}{i + 1:04d}" for i in range(len(items))]
    customers = None
    if kind == "vlxd":
        customers = [(f"KH{i + 1:03d}", n) for i, n in enumerate(cat["customers"])]
    lines = gen_lines(kind, items, as_of, rng, customers=[c[0] for c in customers] if customers else None,
                      branches=cat.get("branches"))
    lines["code"] = [codes[i] for i in lines["item"]]
    lines["name"] = [items[i][0] for i in lines["item"]]
    d = os.path.join(root, bid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {name}\n\n" + cat["desc"].format(name=name, city=city) +
                "\n\nCác file trong gói dữ liệu được chủ cửa hàng tự nhập/xuất, có thể có dòng tiêu đề, dòng tổng.\n")
    clean = {"lines": lines, "items": list(zip(codes, [i[0] for i in items], [i[2] for i in items])),
             "variants": {c: {n} for c, n in zip(codes, [i[0] for i in items])}}

    if kind == "vlxd":
        pd.DataFrame({"Mã SP": codes, "Tên sản phẩm": [i[0] for i in items], "ĐVT": [i[2] for i in items],
                      "Đơn giá": [money_str(i[1], rng) for i in items]}).to_csv(
            os.path.join(d, "san_pham.csv"), index=False, encoding="utf-8-sig")
        orders = lines.drop_duplicates("order_id")[["order_id", "date", "cust", "status"]]
        with pd.ExcelWriter(os.path.join(d, "ban_hang.xlsx")) as xw:
            o = pd.DataFrame({"Số HĐ": orders["order_id"], "Ngày bán": orders["date"].map(lambda x: x.strftime("%d/%m/%Y")),
                              "Mã KH": orders["cust"], "Trạng thái": orders["status"]})
            title = pd.DataFrame([[f"SỔ BÁN HÀNG - CỬA HÀNG {name.upper()}", None, None, None], [None] * 4,
                                  list(o.columns)])
            pd.concat([title, pd.DataFrame(o.values)], ignore_index=True).to_excel(xw, sheet_name="Đơn hàng",
                                                                                   header=False, index=False)
            det = pd.DataFrame({"Số HĐ": lines["order_id"], "Mã SP": lines["code"], "Số lượng": lines["qty"],
                                "Đơn giá": lines["price"], "Thành tiền": [money_str(v, rng) for v in lines["amount"]]})
            total = pd.DataFrame([{"Số HĐ": "Tổng cộng", "Mã SP": None, "Số lượng": int(lines["qty"].sum()),
                                   "Đơn giá": None, "Thành tiền": money_str(lines["amount"].sum(), rng)}])
            pd.concat([det, total]).to_excel(xw, sheet_name="Chi tiết", index=False)
        pd.DataFrame({"Mã KH": [c[0] for c in customers], "Tên khách hàng": [c[1] for c in customers],
                      "SĐT": [f"09{rng.randint(10000000, 99999999)}" for _ in customers]}).to_excel(
            os.path.join(d, "khach_hang.xlsx"), index=False)
        # receivables ledger: invoices on credit and payments
        led = []
        for c, _ in customers:
            owe = rng.random() < 0.6
            for _ in range(rng.randint(1, 4)):
                amt = rng.randint(5, 80) * 100000
                paid = amt if not owe else rng.randint(0, amt // 100000) * 100000
                led.append({"Mã KH": c, "Ngày": (as_of - timedelta(days=rng.randint(1, 200))).strftime("%d/%m/%Y"),
                            "Phát sinh": amt, "Đã trả": paid})
        led = pd.DataFrame(led)
        led.to_csv(os.path.join(d, "cong_no.csv"), index=False, encoding="utf-8-sig")
        clean["ledger"] = led
        clean["customers"] = customers
    elif kind == "trasua":
        con = sqlite3.connect(os.path.join(d, "pos.sqlite"))
        orders = lines.drop_duplicates("order_id")
        pd.DataFrame({"order_id": orders["order_id"], "created_at": orders["date"].map(lambda x: f"{x} 0{rng.randint(7, 9)}:15:00"),
                      "branch": orders["branch"], "status": orders["status"].map(
                {"Hoàn thành": "completed", "Đã hủy": "cancelled", "Trả hàng": "refunded"})}).to_sql(
            "orders", con, index=False)
        pd.DataFrame({"order_id": lines["order_id"], "item_code": lines["code"], "item_name": lines["name"],
                      "quantity": lines["qty"], "unit_price": lines["price"]}).to_sql("order_items", con, index=False)
        con.close()
    else:  # nhathuoc
        pd.DataFrame({"Mã thuốc": codes, "Tên thuốc": [i[0] for i in items], "ĐVT": [i[2] for i in items]}).to_csv(
            os.path.join(d, "danh_muc_thuoc.csv"), index=False, encoding="utf-8-sig")
        var = {c: variant(n, rng) for c, n in zip(codes, [i[0] for i in items])}
        for c, v in var.items():
            clean["variants"][c].add(v.strip())
        out = pd.DataFrame({"Ngày": lines["date"].map(lambda x: x.strftime("%d/%m/%Y")), "Số HĐ": lines["order_id"],
                            "Mã thuốc": lines["code"], "Tên thuốc": [var[c] for c in lines["code"]],
                            "SL": lines["qty"], "Thành tiền": [money_str(v, rng) for v in lines["amount"]],
                            "Trạng thái": lines["status"]})
        out.to_csv(os.path.join(d, "hoa_don_ban_le.csv"), index=False, sep=";", encoding="utf-8-sig")
        batches = []
        for c in codes:
            for b in range(rng.randint(1, 3)):
                exp = as_of + timedelta(days=rng.randint(-20, 400))
                batches.append({"Mã thuốc": c, "Số lô": f"L{rng.randint(1000, 9999)}",
                                "Hạn sử dụng": exp.strftime("%d/%m/%Y"), "Tồn kho": rng.choice([0, 3, 5, 12, 30, 48])})
        bt = pd.DataFrame(batches)
        with pd.ExcelWriter(os.path.join(d, "kho_thuoc.xlsx")) as xw:
            title = pd.DataFrame([["BÁO CÁO TỒN KHO THEO LÔ", None, None, None], [f"Ngày lập: {as_of:%d/%m/%Y}", None, None, None],
                                  list(bt.columns)])
            pd.concat([title, pd.DataFrame(bt.values)], ignore_index=True).to_excel(xw, sheet_name="Tồn kho",
                                                                                    header=False, index=False)
        clean["batches"] = bt.assign(exp=pd.to_datetime(bt["Hạn sử dụng"], dayfirst=True).dt.date)
    clean["kind"] = kind
    return clean


# ----------------------------------------------------------------- questions --
class Oracle:
    def __init__(self, clean, as_of, rb_default_metric="quantity", week_start=0, tet_last=3, horizon=30):
        self.c, self.t = clean, as_of
        L = clean["lines"].copy()
        L["valid"] = L["status"] == "Hoàn thành"
        self.L = L
        self.codes = {c: n for c, n, _ in clean["items"]}
        self.metric = rb_default_metric
        self.week_start, self.tet_last, self.horizon = week_start, tet_last, horizon

    def accepted(self, code):
        return sorted(self.c["variants"][code]) + [code]

    def periods(self):
        t = self.t
        m0 = t.replace(day=1)
        wk = t - timedelta(days=(t.weekday() - self.week_start) % 7)
        pm = (m0 - timedelta(days=1)).replace(day=1)
        return {
            "tháng này": (m0, t + timedelta(days=1)),
            "tháng trước": (pm, m0),
            "tuần rồi": (wk - timedelta(days=7), wk),
            "năm nay": (date(t.year, 1, 1), t + timedelta(days=1)),
            "7 ngày qua": (t - timedelta(days=6), t + timedelta(days=1)),
        }

    def sel(self, a, b, valid=True):
        L = self.L
        m = (L["date"] >= a) & (L["date"] < b)
        return L[m & L["valid"]] if valid else L[m]

    def rank(self, a, b, metric, k, by="code"):
        s = self.sel(a, b)
        col = "amount" if metric == "revenue" else "qty"
        g = s.groupby(by)[col].sum().reset_index().sort_values([col, by], ascending=[False, True])
        vals = g[col].tolist()
        if len(vals) > k and vals[k - 1] == vals[k]:
            return None  # tie at boundary -> excluded
        return g.head(k)


def _noaccent(q):
    from .textnorm import nfc
    import unicodedata
    s = unicodedata.normalize("NFD", nfc(q).replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").lower()


def style(q, rng):
    r = rng.random()
    if r < 0.25:
        return _noaccent(q)
    if r < 0.45:
        for a, b in [("sản phẩm", "sp"), ("được", "dc"), ("không", "ko"), ("bao nhiêu", "bn"), ("khách hàng", "kh")]:
            q = q.replace(a, b)
        return q.lower().rstrip("?")
    return q


def make_questions(bid, clean, as_of, rng, weekday_names=WEEKDAYS):
    O = Oracle(clean, as_of)
    P = O.periods()
    Q, G = [], []
    kind = clean["kind"]
    unit = CATALOG[kind]["unit"]

    def add(text, typ, ans, accepted=None, cat=""):
        qid = f"q_{bid}_{len(Q):03d}"
        Q.append({"id": qid, "business_id": bid, "question": style(text, rng), "as_of_date": as_of.isoformat()})
        if typ == "string" and accepted:
            g = {"id": qid, "type": typ, "answer": ans, "alias": [a for a in accepted if a != ans]}
        elif typ == "list" and ans and isinstance(ans[0], list):
            g = {"id": qid, "type": typ, "answer": [a[0] for a in ans], "alias": [a[1:] for a in ans]}
        else:
            g = {"id": qid, "type": typ, "answer": ans, "alias": []}
        g["category"] = cat
        G.append(g)

    # top-1 by default metric (rulebook) in a few periods
    for per, phr in [("tháng này", "Tháng này bán chạy nhất mặt hàng gì?"),
                     ("tuần rồi", "sản phẩm bán được nhiều nhất tuần rồi là gì"),
                     ("tháng trước", "Mặt hàng nào bán chạy nhất tháng trước?")]:
        r = O.rank(*P[per], O.metric, 1)
        if r is not None and len(r):
            c = r.iloc[0]["code"]
            add(phr, "string", O.codes[c], O.accepted(c), "top1_default")
    # top-3 by quantity
    r = O.rank(*P["tháng trước"], "quantity", 3)
    if r is not None and len(r) == 3:
        add("Top 3 sản phẩm bán nhiều nhất theo số lượng tháng trước?", "list",
            [O.accepted(c) for c in r["code"]], None, "topk_qty")
    # totals
    s = O.sel(*P["tháng trước"])
    add("Doanh thu tháng trước là bao nhiêu?", "number", int(s["amount"].sum()), None, "total_revenue")
    s = O.sel(*P["tháng này"])
    add("Tháng này có bao nhiêu đơn hàng thành công?", "number", int(s["order_id"].nunique()), None, "count_orders")
    # quantity of a specific product last week
    codes = sorted(O.codes)
    c = rng.choice(codes)
    s = O.sel(*P["tuần rồi"])
    pname = O.codes[c]
    short = pname if rng.random() < 0.5 else _noaccent(pname)
    add(f"Tuần rồi bán được bao nhiêu {clean['items'][codes.index(c)][2]} {short}?", "number",
        int(s[s["code"] == c]["qty"].sum()), None, "qty_product")
    # percent change vs same period last year
    a, b = P["tháng này"]
    from dateutil.relativedelta import relativedelta
    cur = O.sel(a, b)["amount"].sum()
    prev = O.sel(a - relativedelta(years=1), b - relativedelta(years=1))["amount"].sum()
    if prev > 0:
        add("Doanh thu tháng này so với cùng kỳ năm ngoái tăng hay giảm bao nhiêu %?", "percent",
            round((cur - prev) / prev * 100, 2), None, "pct_change")
    # best weekday last month (by revenue)
    s = O.sel(*P["tháng trước"]).copy()
    s["wd"] = s["date"].map(lambda d: d.weekday())
    g = s.groupby("wd")["amount"].sum().sort_values(ascending=False)
    if len(g) > 1 and g.iloc[0] != g.iloc[1]:
        add("Tháng trước thứ mấy trong tuần bán được nhiều tiền nhất?", "string", weekday_names[g.index[0]],
            None, "best_weekday")
    # best day (date)
    g = O.sel(*P["tháng trước"]).groupby("date")["amount"].sum().sort_values(ascending=False)
    if len(g) > 1 and g.iloc[0] != g.iloc[1]:
        add("Ngày nào trong tháng trước có doanh thu cao nhất?", "date", g.index[0].isoformat(), None, "best_day")
    # best month last year (yyyy-mm)
    s = O.sel(date(as_of.year - 1, 1, 1), date(as_of.year, 1, 1)).copy()
    if len(s):
        s["m"] = s["date"].map(lambda d: f"{d.year}-{d.month:02d}")
        g = s.groupby("m")["amount"].sum().sort_values(ascending=False)
        if len(g) > 1 and g.iloc[0] != g.iloc[1]:
            add("Năm ngoái tháng nào doanh thu cao nhất?", "date", g.index[0], None, "best_month")
    # boolean threshold
    rev = O.sel(*P["tuần rồi"])["amount"].sum()
    thr = int(round(rev * rng.choice([0.8, 1.25]) / 1e6)) * 1_000_000 or 1_000_000
    if rev != thr:
        add(f"Doanh thu tuần rồi có vượt {thr // 1_000_000} triệu không?", "boolean", bool(rev > thr), None, "bool_threshold")
    # ---- harder questions that the template library does NOT cover (LLM families needed)
    s = O.sel(*P["tháng trước"])
    if s["order_id"].nunique():
        aov = s.groupby("order_id")["amount"].sum().mean()
        add("Tháng trước trung bình mỗi đơn hàng bao nhiêu tiền?", "number", int(float(aov) + 0.5), None, "hard_avg_order")
    sold = set(O.sel(*P["tháng trước"])["code"])
    unsold = sorted(c for c in codes if c not in sold)
    add("Tháng trước có mặt hàng nào không bán được không?", "boolean", bool(unsold), None, "hard_unsold_bool")
    s = O.sel(*P["năm nay"])
    g = s.groupby("code")["order_id"].nunique().reset_index().sort_values(["order_id", "code"], ascending=[False, True])
    if len(g) > 2 and g.iloc[1]["order_id"] != g.iloc[2]["order_id"]:
        add("Năm nay 2 sản phẩm nào xuất hiện trong nhiều đơn hàng nhất?", "list",
            [O.accepted(c) for c in g.head(2)["code"]], None, "hard_top_by_orders")
    # unanswerable
    add("Lợi nhuận tháng trước là bao nhiêu?", "unanswerable", None, None, "ua_metric")
    add("Doanh thu năm 2019 là bao nhiêu?", "unanswerable", None, None, "ua_period")
    foreign = {"vlxd": "Trà sữa trân châu", "trasua": "Xi măng Hà Tiên", "nhathuoc": "Thép Hòa Phát D10"}[kind]
    add(f"Tuần rồi bán được bao nhiêu {foreign}?", "unanswerable", None, None, "ua_value")

    if kind == "vlxd":
        led = clean["ledger"]
        debt = (led["Phát sinh"] - led["Đã trả"]).groupby(led["Mã KH"]).sum().rename("debt").reset_index()
        debt = debt[debt["debt"] > 0].sort_values(["debt", "Mã KH"], ascending=[False, True])
        names = dict(clean["customers"])
        if len(debt):
            add("Còn ai nợ tiền chưa trả?", "list", [[names[c], c] for c in debt["Mã KH"]], None, "debtors")
            add("Tổng số tiền khách còn nợ là bao nhiêu?", "number", int(debt["debt"].sum()), None, "total_debt")
        s = O.sel(*P["năm nay"])
        g = s.groupby("cust")["amount"].sum().sort_values(ascending=False)
        if len(g) > 1 and g.iloc[0] != g.iloc[1]:
            c = g.index[0]
            add("Khách hàng nào mua nhiều tiền nhất năm nay?", "string", names[c], [names[c], c], "top_customer")
    if kind == "trasua":
        s = O.sel(*P["7 ngày qua"])
        c = rng.choice(codes)
        add(f"7 ngày qua bán được bao nhiêu ly {O.codes[c].lower()}?", "number", int(s[s["code"] == c]["qty"].sum()),
            None, "qty_product_rolling")
        s = O.sel(*P["tháng trước"])
        g = s.groupby("branch")["amount"].sum().reset_index().sort_values("branch")
        add("Doanh thu tháng trước của từng chi nhánh?", "table",
            {"columns": ["chi nhánh", "doanh thu"], "rows": [[b, int(v)] for b, v in g.values]}, None, "table_branch")
    if kind == "nhathuoc":
        bt = clean["batches"]
        h = as_of + timedelta(days=30)
        e = bt[(bt["exp"] >= as_of) & (bt["exp"] <= h) & (bt["Tồn kho"] > 0)]
        e = e.sort_values(["exp", "Mã thuốc"]).drop_duplicates("Mã thuốc")
        if len(e):
            add("Thuốc nào sắp hết hạn trong kho?", "list", [O.accepted(c) for c in e["Mã thuốc"]], None, "expiring")
        else:
            add("Thuốc nào sắp hết hạn trong kho?", "list", [], None, "expiring")
    return Q, G


RULEBOOK_MD = """# DataMate — Rulebook (bộ synthetic)

**Hôm nay** của mỗi câu là trường `as_of_date`.

| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn **không huỷ** và **không trả hàng**. Đơn có trả hàng, kể cả chỉ trả một phần, là đơn trả hàng: bỏ cả đơn |
| Đơn thuộc kỳ nào | theo ngày đặt |
| Số đơn | chỉ tính đơn không huỷ, không trả |
| Doanh thu một đơn | tổng thành tiền các dòng hàng, không trừ giảm giá trên cả đơn |
| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ trong mọi đơn đặt trong kỳ |
| Phần trăm thay đổi, tăng trưởng | so với kỳ trước; giảm thì ra số âm |
| Tuần | bắt đầu thứ Hai. Tuần trước: cả tuần liền trước |
| Tháng này | từ ngày 1 đến hết hôm nay. So với tháng trước thì lấy cùng số ngày của tháng trước |
| N ngày gần nhất | N ngày liền nhau kể cả hôm nay |
| Bán chạy | theo số lượng quy về đơn vị cơ bản |
| Bằng nhau | hai đối tượng bằng nhau ở tiêu chí câu hỏi thì xếp theo mã tăng dần |
| Giá trị trung bình một đơn | làm tròn về đồng, nửa đồng làm tròn lên |
| Xác định khách | mỗi mã khách là một khách; khách lẻ không phải khách |
| Không trả lời được | kỳ nằm hoàn toàn ngoài khoảng ngày có dữ liệu thì không trả lời được; kỳ chỉ có một phần nằm trong khoảng ngày có dữ liệu thì tính trên phần có dữ liệu |
"""


def write_lunar_csv(path, y0=2023, y1=2027):
    rows, d = [], date(y0, 1, 1)
    while d <= date(y1, 12, 31):
        ld, lm, ly, lp = solar_to_lunar_algo(d)
        note = "Tết Nguyên Đán" if (ld, lm) == (1, 1) and not lp else ""
        rows.append({"date": d.isoformat(), "weekday": d.isoweekday(), "lunar_day": ld, "lunar_month": lm,
                     "lunar_year": ly, "lunar_leap": str(bool(lp)).lower(), "holiday": note})
        d += timedelta(days=1)
    pd.DataFrame(rows).to_csv(path, index=False)


def generate(out_dir: str, n_business: int = 6, seed: int = 0):
    """Writes the official split layout: data/<biz>/, rulebook.md, calendar.csv, questions.jsonl, answers.jsonl."""
    rng = random.Random(seed)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    kinds = ["vlxd", "trasua", "nhathuoc"]
    allq, allg = [], []
    for i in range(n_business):
        bid = f"syn_{seed:02d}{i:03d}"
        as_of = date(2026, 3, 4) + timedelta(days=rng.randint(-40, 40))
        clean = write_business(os.path.join(out_dir, "data"), bid, kinds[i % 3], rng, as_of)
        q, g = make_questions(bid, clean, as_of, rng)
        allq += q
        allg += g
    with open(os.path.join(out_dir, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in allq:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "answers.jsonl"), "w", encoding="utf-8") as f:
        for g in allg:
            f.write(json.dumps(g, ensure_ascii=False, default=str) + "\n")
    open(os.path.join(out_dir, "rulebook.md"), "w", encoding="utf-8").write(RULEBOOK_MD)
    write_lunar_csv(os.path.join(out_dir, "calendar.csv"))
    return len(allq)
