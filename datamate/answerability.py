"""Positive evidence that a question CANNOT be answered from the released data.

Emitting `unanswerable` is itself an answer that scores -1 when wrong, so we only
use it on positive evidence (never because generation failed). Categories follow
PRACTIQ (Dong et al., NAACL 2025: nonexistent SELECT column / WHERE column /
filter value, unsupported join) plus rules of the DataMate rulebook §7:
  * period entirely outside the shop's data range        (rulebook: not 0)
  * the shop does not record the needed kind of information (payment method,
    channel, customer group, address, gender, stock, purchases, cost, debt,
    expiry, return/cancel reasons, time of day ...)
  * subjective / out-of-data questions (staff attitude, medical dosage ...)
  * a named item / customer that does not exist in this shop's data
"""
from __future__ import annotations

import re

from .loader import name_key

SUBJECTIVE = r"lanh nghe|tan tam|le phep|chu dao|quy men|hay khen|tay nghe|nhiet tinh|than thien|vui ve|de thuong|" \
             r"uy tin|tot bung|chuyen nghiep|lich su|dep|xinh|nen uong|lieu dung|cach dung|uong moi lan|tac dung phu|" \
             r"chua (benh|duoc)|co tot khong|ngon nhat|hai long"

NEEDS = [  # (question regex, roles or column-name regex that must exist with data, label)
    (r"gioi tinh|\bkhach (hang )?(la )?(nu|nam|phu nu|dan ong|dan ba)\b|\b(nu|nam) gioi\b|(bao nhieu|may) (khach )?(phu nu|dan ong)",
     None, r"gioi tinh|nam nu|gender|sex", "giới tính khách"),
    (r"dia chi|nha o|\b(o|tai|khach|pho) duong\b|khach (o )?quan \d", "address", r"dia chi|addr", "địa chỉ khách"),
    (r"\bvi (loi|mua nham|giao tre|hong|het hang|doi y)|ly do", None, r"ly do|nguyen nhan|reason", "lý do huỷ/trả"),
    (r"tien mat|chuyen khoan|vi dien tu|\bcod\b|quet the|hinh thuc thanh toan|tra bang", "payment", None, "hình thức thanh toán"),
    (r"zalo|tai quay|tai tiem|qua dien thoai|san tmdt|shopee|lazada|tiktok|online|kenh ban|giao di|grab", "channel", None, "kênh bán"),
    (r"nhom khach|khach (than thiet|si|vip|thanh vien)", "customer_group", None, "nhóm khách"),
    (r"\bton\b|ton kho|con lai .{0,20}trong kho|trong kho|het hang", "stock", r"ton|the kho", "tồn kho"),
    (r"\bnhap\b|nha cung cap|\bncc\b", "import_qty|import_amount|import_price|import_total|receipt_id", None, "nhập hàng"),
    (r"lai gop|loi gop|lai rong|loi nhuan|gia von|hang loi nhat|lai nhat", "cost", r"gia von|cost", "giá vốn"),
    (r"\bno\b|cong no|mua chiu|tra no|thu no|ghi no|thieu tien|con thieu", "debt|paid", r"no|thieu|tra|thu", "công nợ"),
    (r"het han|han dung|han su dung|\blo\b", "expiry", r"han|hsd|exp", "hạn dùng"),
    (r"ca sang|ca chieu|gio cao diem|khung gio|\bgio\b", "__time__", None, "giờ bán"),
    (r"nhan vien|thu ngan|\btho\b", "staff", r"nhan vien|tho|nv", "nhân viên"),
]


def evidence(biz, sm, spec, tctx, item_linked, data_lo, data_hi, links=()) -> list[tuple[str, str]]:
    """-> [(strength 'strong'|'weak', reason)]"""
    q = spec.q
    # words inside a named item are not requests ("Giò heo" is not "giờ", "Bay thợ hồ" is not a staff member)
    for l in links:
        if (l.role or "").startswith("product_") and l.score >= 95 and len(l.span.split()) >= 2:
            q = q.replace(l.span, " ")
    ev = []
    if re.search(SUBJECTIVE, q):
        ev.append(("strong", "câu hỏi chủ quan / ngoài dữ liệu bán hàng"))
    p = tctx.primary
    if p is not None and data_lo is not None and data_hi is not None:
        end = min(p.end, tctx.as_of + __import__("datetime").timedelta(days=1))
        if end <= data_lo or p.start > data_hi:
            ev.append(("strong", f"kỳ {p.label} nằm hoàn toàn ngoài khoảng dữ liệu {data_lo}..{data_hi}"))
    q2 = tctx.comparison
    if q2 is not None and data_lo is not None and (q2.end <= data_lo or q2.start > data_hi):
        ev.append(("strong", f"kỳ so sánh {q2.label} nằm ngoài khoảng dữ liệu"))
    roles = {}
    for (tn, c), r in sm.roles.items():
        s = biz.tables[tn].df[c] if tn in biz.tables else None
        if s is not None and s.notna().any():
            roles.setdefault(r, []).append((tn, c))
    allcols = [(t.name, c, name_key(t.orig.get(c, c))) for t in biz.tables.values() if not t.name.startswith("v_")
               for c in t.df.columns if t.df[c].notna().any()]
    kinds = set(sm.kinds.values())
    for qrx, need_roles, colrx, label in NEEDS:
        if not re.search(qrx, q):
            continue
        ok = False
        if need_roles == "__time__":
            ok = bool(sm.flags.get("has_time"))
        elif need_roles:
            ok = any(r in roles for r in need_roles.split("|"))
            if label == "nhập hàng":
                ok = ok or "purchase" in kinds
            if label == "tồn kho":
                ok = ok or "stock" in kinds or "purchase" in kinds
            if label == "công nợ":
                ok = ok or "debt" in kinds or "payment" in kinds
            if label == "nhóm khách" and sm.orders is not None and sm.orders["customer_group"].notna().any():
                ok = True
            if label == "hình thức thanh toán" and sm.orders is not None and sm.orders["payment"].notna().any():
                ok = True
            if label == "kênh bán" and sm.orders is not None and sm.orders["channel"].notna().any():
                ok = True
        if not ok and colrx:
            ok = any(re.search(colrx, nk) for _, _, nk in allcols)
        if not ok:
            ev.append(("strong", f"shop không ghi {label}"))
    if item_linked is False:
        ev.append(("strong", "mặt hàng được hỏi không có trong dữ liệu của shop"))
    cust_intents = ("top_customers", "n_customers", "n_customers_min_orders", "pct_customers_min_orders")
    if spec.intent in cust_intents and sm.orders is not None and sm.orders["customer_key"].isna().all():
        ev.append(("strong", "shop không ghi khách (chỉ có khách lẻ)"))
    return ev
