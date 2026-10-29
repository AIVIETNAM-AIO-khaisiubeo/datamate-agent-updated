"""Column-role tagging (schema linking) and table-kind classification.

Evidence, strongest first:
  1. README data dictionary description of the column ("thành tiền sau giảm",
     "trạng thái đơn", "số lượng đã đổi ra đơn vị cơ bản"...);
  2. the column header (Vietnamese / English / camelCase / abbreviations);
  3. the column dtype (a role is dropped when the values cannot carry it).
Schema linking is a core stage of RESDSQL (Li et al., AAAI 2023), CHESS
(Talaei et al., 2024) and CHASE-SQL (Pourreza et al., ICLR 2025); here the
schemas are small, so we tag every column instead of pruning.
"""
from __future__ import annotations

import re

import pandas as pd

from .loader import name_key
from .textnorm import fold

# (phrase in folded README description, role) — most specific first
DESC_RULES = [
    ("cot trong", "empty"),
    ("so luong da doi ra don vi co ban", "qty_base"), ("so luong quy ve don vi co ban", "qty_base"),
    ("so luong ban lay tu lo", "lot_qty"), ("so luong khach tra lai", "returned_qty"),
    ("so luong nhap", "import_qty"), ("so luong giao", "delivery_qty"), ("so luong ton", "stock"),
    ("so thu tu dong", "line_no"), ("so thu tu don", "order_id"), ("so hoa don", "order_id"),
    ("ma don", "order_id"), ("ma hoa don", "order_id"), ("ma chung tu", "order_id"),
    ("ma phieu nhap", "receipt_id"), ("ma phieu thu", "payment_id"),
    ("ma nha cung cap", "supplier"), ("nha cung cap", "supplier"),
    ("ma khach", "customer_code"), ("ten khach", "customer_name"), ("nhom khach", "customer_group"),
    ("ma nhan vien", "staff"), ("nhan vien", "staff"), ("tho lam", "staff"),
    ("ma hang", "product_code"), ("ma san pham", "product_code"), ("ma thuoc", "product_code"),
    ("ten hang", "product_name"), ("hang ban", "product_name"), ("ten thuoc", "product_name"),
    ("ten san pham", "product_name"), ("ten mon", "product_name"),
    ("nhom hang", "product_group"), ("loai hang", "product_group"),
    ("don vi co ban", "base_unit"), ("don vi lon", "big_unit"), ("don vi nhap", "import_unit"),
    ("don vi tinh", "unit"), ("he so", "factor"),
    ("thoi diem dat don", "order_datetime"), ("gio dat don", "order_time"),
    ("ngay ban", "order_date"), ("ngay dat", "order_date"), ("ngay mua", "order_date"),
    ("ngay nhap", "import_date"), ("ngay giao", "delivery_date"), ("ngay sua", "order_date"),
    ("han su dung", "expiry"), ("so lo", "lot"),
    ("trang thai phieu nhap", "import_status"), ("trang thai don", "status"), ("trang thai", "status"),
    ("thanh tien sau giam", "line_amount"), ("thanh tien cua dong", "line_amount"),
    ("tien nhap cua ca dong", "import_amount"), ("tong tien phieu nhap", "import_total"),
    ("tien giam cua ca dong", "line_discount"),
    ("giam gia cua ca don", "order_discount"), ("tien bot cho ca don", "order_discount"),
    ("tong tien hang cua ca don", "order_goods_total"), ("tong tien hang cua don", "order_goods_total"),
    ("tien hang cua don", "order_goods_total"),
    ("so khach phai tra", "order_total"), ("khach can tra", "order_total"), ("tong tien don", "order_total"),
    ("tong tien cua ca phieu", "order_total"), ("tong tien ca phieu", "order_total"),
    ("so tien khach tra cho ca don", "order_total"),
    ("phi giao", "shipping_fee"),
    ("khach da tra", "paid"), ("da thanh toan", "paid"),
    ("khach con no", "debt"), ("con no", "debt"),
    ("hinh thuc thanh toan", "payment"), ("kenh ban", "channel"),
    ("gia nhap", "import_price"), ("gia von", "cost"), ("don gia", "unit_price"),
    ("gia ban hien tai", "list_price"), ("ton kho", "stock"),
    ("so dien thoai", "phone"), ("dia chi", "address"), ("ghi chu", "note"),
    ("giam gia", "discount"), ("thanh tien", "line_amount"), ("so luong", "qty"),
]

# (regex on folded/camel-split header, role) — evaluated in order
NAME_RULES = [
    (r"^(stt|so tt|tt dong|so thu tu|thu tu|line|no|dong|dong so)$", "line_no"),
    (r"(sl|so luong|slg)\s*((dv|don vi)\s*)?(quy doi|quy ra|co ban|goc|le|quy ra dv le|quy ra le)|quy ra (dv )?le|sl co ban|slquy doi|so luong le", "qty_base"),
    (r"sl tra|so luong tra|tra lai", "returned_qty"),
    (r"(sl|so luong) nhap", "import_qty"),
    (r"(sl|so luong) giao|thuc giao", "delivery_qty"),
    (r"^(ma (hoa don|hd|don|don hang|chung tu|giao dich|bill)|so (hd|hoa don|bill|don|chung tu)|(don|hd|hoa don|bill) so|order id|order no|ma bill|hd|bill|invoice|ma hoadon|ma don ban)$", "order_id"),
    (r"^(ma (phieu nhap|nhap hang|pn)|so phieu|ma phieu)$", "receipt_id"),
    (r"^(ma (kh|khach|khach hang)|so kh|customer id|ma kh mua|id khach)$", "customer_code"),
    (r"^(ten (kh|khach|khach hang)|(khach hang|khach|kh) ten|khach hang|khach|ben mua|doi tac|customer|nguoi mua|ten khach mua|customer name)$", "customer_name"),
    (r"(nhom|loai) khach|hang khach|khach le si|loai khach hang", "customer_group"),
    (r"^(ma (nv|nhan vien)|nhan vien|nv ban|nguoi ban|thu ngan|ma nv ban|tho)$", "staff"),
    (r"^(ma (hang|sp|san pham|hh|hang hoa|mon|thuoc|vach)|id sp|sku|ma hanghoa|ma|(item|product|sku) (code|id)|product id|item no)$", "product_code"),
    (r"^(ten (hang|sp|san pham|hang hoa|mon|thuoc|mat hang)|hang|hang hoa|mat hang|san pham|mon|noi dung|chi tiet hang|item|thuoc|ten|ten goi|ten day du|(item|product) name|product|description)$", "product_name"),
    (r"nhom (hang|hh|hang hoa|sp)|danh muc|phan loai|nhom|loai$|category", "product_group"),
    (r"(don vi|dv|dvt)\s*(goc|co ban|co so)|don vi co so", "base_unit"),
    (r"^(dv|don vi) (nhap|lon|quy doi|doi|quy cach)$|^dvqd$|^don vi quy doi( lon)?$|^quy cach$", "big_unit"),
    (r"^(dvt|don vi( tinh)?|dv|d vi|unit|dv tinh)$", "unit"),
    (r"^(?!gia\b)(?!.*\bgia (ban|nhap)\b).*(he so|quy doi|\bbang\b)", "factor"),
    (r"^(gio|gio tao don|gio dat|time)$", "order_time"),
    (r"thoi (gian|diem) (lap|tao|dat)|gio tao don|created", "order_datetime"),
    (r"^(ngay( ban| dat| giao dich| lap| mua| thang| hd| tao)?|ngay ban hang|ngaylap|ngay ban le|ngay chung tu|thoi gian)$", "order_date"),
    (r"ngay nhap", "import_date"), (r"ngay giao", "delivery_date"),
    (r"han su dung|hsd|han dung|expiry", "expiry"), (r"^(lo|so lo|ma lo)$", "lot"),
    (r"trang thai|status|ket qua|xu ly|tinh trang", "status"),
    (r"^(thanh tien|t tien|tri gia|tt|tt dong|thanh tien dong|tien|thanh tien sau giam|thanhtien|amount|line (total|amount)|subtotal)$", "line_amount"),
    (r"^(giam|giam dong|ck dong|chiet khau dong)$", "line_discount"),
    (r"tien hang|tong tien hang", "order_goods_total"),
    (r"^(tong tien|tong bill|tong thanh toan|tong don|phai thu|tong sau giam gia|khach phai tra|can tra)$", "order_total"),
    (r"chiet khau|giam gia|khuyen mai|^ck$|^bot$", "discount"),
    (r"phi (giao|ship|van chuyen)", "shipping_fee"),
    (r"^(da tra|khach da tra|da thu|khach tra)$", "paid"),
    (r"con no|chua tra|cong no|con lai|con phai thu", "debt"),
    (r"phuong thuc|hinh thuc (tt|thanh toan)|thanh toan|payment", "payment"),
    (r"^(chi nhanh|ma chi nhanh|cua hang|co so|branch|store|diem ban)$", "branch"),
    (r"kenh|ban qua|nguon don|^nguon$|channel", "channel"),
    (r"gia von", "cost"), (r"gia nhap", "import_price"),
    (r"^(don gia|gia|gia ban|gia le|gia niem yet|gia don vi|don gia ban|dg|unit price|price|gia san pham|gia (1|mot) (dv|don vi))$", "unit_price"),
    (r"^(sl|s l|so luong|slg|qty|so luong ban|sl ban|quantity|qty sold|so luong san pham)$", "qty"),
    (r"^(barcode|ma barcode|ma vach)$", "barcode"),
    (r"ton (kho|cuoi|thuc)|^ton$", "stock"),
    (r"nha cung cap|ncc", "supplier"),
    (r"dien thoai|sdt|phone", "phone"), (r"dia chi|addr", "address"),
    (r"ghi chu|note|dien giai", "note"),
]
NUMERIC_ROLES = {"qty", "qty_base", "returned_qty", "import_qty", "delivery_qty", "lot_qty", "factor", "unit_price",
                 "line_amount", "line_discount", "discount", "order_discount", "order_goods_total", "order_total",
                 "shipping_fee", "paid", "debt", "cost", "import_price", "import_amount", "import_total", "stock",
                 "list_price"}
DATE_ROLES = {"order_date", "order_datetime", "import_date", "delivery_date", "expiry"}
TEXT_ROLES = {"order_id", "receipt_id", "payment_id", "customer_code", "customer_name", "product_code",
              "product_name", "product_group", "unit", "base_unit", "big_unit", "import_unit", "status", "note",
              "payment", "channel", "staff", "supplier", "customer_group", "lot", "phone", "address", "branch"}


def _dtype_ok(role: str, s: pd.Series) -> bool:
    nn = s.dropna()
    if nn.empty:
        return role == "empty"
    if role == "factor" and not pd.api.types.is_numeric_dtype(s):
        # "12" / "15 vỉ" (factor relative to the next smaller unit)
        return nn.astype(str).str.match(r"^\s*\d+([.,]\d+)?(\s+\D.{0,15})?\s*$").mean() > 0.9
    if role in NUMERIC_ROLES:
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if role in DATE_ROLES:
        return pd.api.types.is_datetime64_any_dtype(s)
    if role == "order_time":
        return pd.api.types.is_datetime64_any_dtype(s) or nn.astype(str).str.match(r"^\d{1,2}:\d{2}").mean() > 0.9
    if role in ("order_id", "line_no", "product_code", "customer_code"):
        return True  # codes may be numeric (STT)
    if role in TEXT_ROLES:
        return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)
    return True


def role_of(table, col: str, readme) -> tuple[str | None, str]:
    """-> (role, evidence)"""
    orig = table.orig.get(col, col)
    s = table.df[col]
    if col == "_sheet":
        return None, ""
    desc = readme.desc_for(table.files, orig) if readme is not None else None
    if desc:
        d = f" {fold(desc)}"
        best = None   # the phrase the description STARTS with wins ("hệ số: 1 đơn vị lớn = ..." is a factor)
        for phrase, role in DESC_RULES:
            i = d.find(f" {phrase}")
            if 0 <= i <= 12 and (best is None or (i, -len(phrase)) < (best[0], -best[2])):
                best = (i, role, len(phrase))
        if best is not None and _dtype_ok(best[1], s):
            return best[1], "readme"
    key = name_key(orig)
    # "Thành tiền (đ)", "Giảm (VNĐ)" -> drop the currency suffix before matching
    bare = re.sub(r"\s+(d|vnd|dong|vnđ|k)$", "", key).strip()
    for k in dict.fromkeys((key, bare)):
        for rx, role in NAME_RULES:
            if re.search(rx, k) and _dtype_ok(role, s):
                return role, "name"
    return None, ""


# ------------------------------------------------------------- table kinds ---
KIND_RULES = [
    ("purchase", r"nhap hang|phieu nhap|chi tiet nhap|nhap kho|so nhap|don nhap|^nhap|\bnhap\b"),
    ("stock", r"ton kho|bao cao ton|^ton\b|\bton\b|the kho"),
    ("price_history", r"bang gia|lich su gia"),
    ("lot", r"lo hang|ban theo lo|chi tiet lo|don hang lo|\blo\b"),
    ("payment", r"thanh toan|phieu thu|so quy|thu tien"),
    ("delivery", r"giao hang|phieu giao|so giao"),
    ("debt", r"cong no|han no|so no\b|tra gop|so gop"),
    ("expense", r"chi phi"),
    ("glossary", r"ghi tat|viet tat|tu dien|chu thich|giai thich"),
    ("warranty", r"bao hanh"),
    ("return", r"tra hang|hoan tra|doi tra|hang tra"),
    ("customer", r"khach hang|^khach|thanh vien"),
    ("supplier", r"nha cung cap|\bncc\b"),
    ("branch", r"chi nhanh|\bco so\b"),
    ("conversion", r"quy doi|don vi|\bdvt\b"),
    ("sales", r"hoa don|don hang|chi tiet don|chi tiet ban|bill"),
    ("product", r"san pham|hang hoa|danh muc|\bthuoc\b|\bmon\b|mat hang"),
    ("staff", r"nhan vien|\btho\b"),
    ("sales", r"ban hang|ban le|so ban|pos|du lieu ban|sua xe|giao dich|chi tiet"),
]


def table_kind(table, roles: dict) -> str:
    """Kind from the table's own title first (sheet / SQL table), then from its file names."""
    title = name_key(getattr(table, "title", "") or table.name)
    fnames = "" if getattr(table, "is_sql", False) else \
        " ".join(name_key(re.sub(r"\.(xlsx|xls|csv|tsv|txt)$", "", f)) for f in table.files)
    r = {roles.get((table.name, c)) for c in table.df.columns}
    has_prod = bool(r & {"product_name", "product_code"})
    has_qty_amt = bool(r & {"qty", "qty_base", "line_amount"})
    for tname in (title, fnames):
        for kind, rx in KIND_RULES:
            if tname and re.search(rx, tname):
                if kind == "product" and (r & {"order_id"}) and has_qty_amt:
                    return "sales"
                if kind == "customer" and has_prod and has_qty_amt:
                    return "sales"
                return kind
    if has_prod and has_qty_amt and (r & {"order_id", "order_date", "order_datetime"}):
        return "sales"
    if (r & {"order_id"}) and (r & {"order_date", "order_datetime", "status"}):
        return "sales"
    if has_prod and not (r & {"order_date", "order_datetime"}):
        return "product"
    return "other"


def tag_business(biz, readme) -> tuple[dict, dict, dict]:
    """-> (roles {(table,col): role}, evidence {(table,col): 'readme'|'name'}, kinds {table: kind})"""
    roles, ev = {}, {}
    for t in biz.tables.values():
        if t.name.startswith("v_"):
            continue
        for c in t.df.columns:
            r, e = role_of(t, c, readme)
            if r and r != "empty":
                roles[(t.name, c)] = r
                ev[(t.name, c)] = e
        # a barcode is the item key only when the table has no other item code
        bc = [c for c in t.df.columns if roles.get((t.name, c)) == "barcode"]
        if bc and not any(roles.get((t.name, c)) == "product_code" for c in t.df.columns):
            roles[(t.name, bc[0])] = "product_code"
    kinds = {t.name: table_kind(t, roles) for t in biz.tables.values() if not t.name.startswith("v_")}
    # a "name" column means different things in different lists
    remap = {"staff": {"product_name": "staff_name", "customer_name": "staff_name", "product_code": None},
             "customer": {"product_name": "customer_name"},
             "supplier": {"product_name": "supplier", "customer_name": "supplier", "product_code": None},
             "branch": {"product_name": "branch", "customer_name": "branch", "product_code": None}}
    for t in biz.tables.values():           # "Hạng" (tier) of a customer folds to "hang" like "Hàng" (goods)
        if kinds.get(t.name) == "customer":
            for c in t.df.columns:
                if re.fullmatch(r"(hang|hang khach|hang thanh vien|phan hang|cap do|hang kh)", name_key(t.orig.get(c, c))):
                    roles[(t.name, c)] = "customer_group"
    for (tn, c), r in list(roles.items()):
        m = remap.get(kinds.get(tn), {})
        if r in m:
            if m[r] is None:
                roles.pop((tn, c))
                ev.pop((tn, c), None)
            else:
                roles[(tn, c)] = m[r]
    return roles, ev, kinds
