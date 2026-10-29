"""Rule-based question analysis: intent family, answer type and slots.

A cheap deterministic "router" (no LLM call). Its outputs
  (a) select a deterministic template program (templates.py),
  (b) are shown to the LLM as hints,
  (c) are calibration features (agreement between LLM-declared type and this one).
Classifying a question before generating SQL follows DIN-SQL (Pourreza & Rafiei,
NeurIPS 2023). All matching happens on qnorm(question): folded, Telex typos
fixed, chat abbreviations expanded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .textnorm import parse_number_vn, qnorm

_NUMW = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}


@dataclass
class QSpec:
    raw: str
    q: str                          # canonical folded question
    intent: str | None = None       # see INTENTS below
    answer_type: str = "number"
    k: int | None = None
    ascending: bool = False
    metric: str | None = None       # revenue | quantity | orders | customers | days
    entity: str | None = None       # item | customer | staff | day | month | weekday | hour | group | payment | channel
    threshold: float | None = None
    threshold_inclusive: bool = True
    group_name: str | None = None   # "nhóm X" phrase (folded)
    cust_group: str | None = None
    payment: str | None = None
    channel: str | None = None
    multi: list = field(default_factory=list)   # ordered [(attr, value)] for "A và B lần lượt" list questions
    base_intent: str | None = None
    unit_word: str | None = None    # "bao nhiêu <ly|lon|chai...>"
    size: str | None = None         # coffee size M/L
    flags: list = field(default_factory=list)


PAY_WORDS = {"tien mat": "cash", "tien tuoi": "cash", "chuyen khoan": "transfer", "vi dien tu": "ewallet", "vi": "ewallet", "cod": "cod",
             "the": "card", "quet the": "card", "ghi no": "debt", "no": "debt", "momo": "ewallet", "zalopay": "ewallet"}


CHANNEL_WORDS = [  # (question phrase, canonical channel)
    (r"tai quay|tai cua hang|tai tiem|tai shop|truc tiep tai|offline", "store"),
    (r"zalo", "zalo"), (r"dien thoai|goi dien|hotline", "phone"), (r"shopee", "shopee"), (r"lazada", "lazada"),
    (r"tiktok", "tiktok"), (r"grab", "grab"), (r"facebook|\bfb\b|fanpage|messenger", "facebook"),
    (r"website|\bweb\b|online|truc tuyen", "online")]


def _filter_mentions(q: str) -> list:
    """Ordered channel / payment mentions: [("channel", "zalo"), ("payment", "cod"), ...]."""
    hits = []
    for rx, ch in CHANNEL_WORDS:
        for m in re.finditer(rf"\b(ban|don|mua|dat|kenh)\b.{{0,15}}?\b({rx})", q):
            hits.append((m.start(2), "channel", ch))
    for w, p in PAY_WORDS.items():
        for m in re.finditer(rf"\b(tra|thanh toan|bang|qua|hinh thuc) ?(bang |qua )?({w})\b|\bdon ({w})\b", q):
            hits.append((m.start(), "payment", p))
    hits.sort()
    out = []
    for _, k, v in hits:
        if (k, v) not in out:
            out.append((k, v))
    return out


def _int(tok):
    return int(tok) if tok.isdigit() else _NUMW.get(tok)


def _money(s: str) -> float | None:
    s = s.strip()
    m = re.match(r"^([\d.,]+)\s*(trieu|tr|nghin|ngan|k|ty|dong|d)?$", s)
    if not m:
        return None
    v = parse_number_vn(m.group(1).replace(" ", ""))
    if v is None:
        return None
    return v * {"trieu": 1e6, "tr": 1e6, "nghin": 1e3, "ngan": 1e3, "k": 1e3, "ty": 1e9}.get(m.group(2) or "", 1)


def analyze(question: str) -> QSpec:
    q = qnorm(question)
    s = QSpec(raw=question, q=q)

    # ---------- threshold "từ X trở lên", "≥ X", "ít nhất X", "trên X"
    raw_l = question.lower()
    unit = r"(?:\s*(?:trieu|tr|nghin|ngan|k|ty|dong|d)\b)?"
    m = re.search(r"(?:≥|>=)\s*([\d.,]+)" + unit, raw_l) or re.search(r"(?:≥|>=)\s*([\d.,]+)" + unit, q)
    if m:
        s.threshold, s.threshold_inclusive = _money(m.group(0).lstrip("≥>= ")), True
    else:
        m = re.search(r"\b(?:tu|it nhat|toi thieu|dat|bang)\s+([\d.,]+" + unit + r")\s*(?:don|don hang|lan|ly|ngay)?\s*(?:tro len|tro di)?", q)
        if m and (re.search(r"tro (len|di)", q[m.start():m.end() + 12]) or re.match(r"(it nhat|toi thieu)", m.group(0))):
            s.threshold, s.threshold_inclusive = _money(m.group(1)), True
        else:
            m = re.search(r"\b(?:tren|hon|qua|vuot|lon hon)\s+([\d.,]+" + unit + r")", q)
            if m:
                s.threshold, s.threshold_inclusive = _money(m.group(1)), False

    # ---------- k
    m = re.search(r"\btop\s*(\d+)\b", q) or re.search(
        r"(?<![/\d])\b(\d{1,2}|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\s+(san pham|mat hang|mon|hang|khach hang|khach|thuoc|nhan vien|nhom|loai)\b", q)
    if m and not re.search(r"(thang|nam|quy|ngay)\s*$", q[:m.start()]):
        k = _int(m.group(1))
        if k and 1 <= k <= 50:
            s.k = k

    # ---------- filters
    m = re.search(r"(?<!khach )(?<!khach hang )(?<!khach thuoc )\bnhom (?!khach)(?!hang nao)(?:hang )?(.+?)"
                  r"(?= (ban|chiem|trong|thang|tuan|nam|quy|hom|tu|duoc|la|co|doanh|dong gop|thu|chi|ti|ty|cua|"
                  r"tong|so)\b|$)", q)
    if m and not re.search(r"nhom hang nao|nhom nao", q):
        s.group_name = m.group(1).strip()
    m = re.search(r"\b(?:nhom )?khach (?:hang )?(?:thuoc )?(?:nhom )?(than thiet|si|le|vip|thanh vien|moi|quen)\b", q)
    if m and re.search(r"nhom khach|khach (hang )?(thuoc )?(nhom )?(than thiet|si|vip)", q):
        s.cust_group = m.group(1)
    for w, p in PAY_WORDS.items():
        if re.search(rf"\b(tra|thanh toan|bang|qua|hinh thuc) ?(bang |qua )?{w}\b|\bdon {w}\b", q):
            s.payment = p
            break
    for rx, ch in CHANNEL_WORDS:
        if re.search(rf"\b(ban|don|mua|dat|kenh)\b.{{0,15}}\b({rx})", q):
            s.channel = ch
            break
    m = re.search(r"\bsize ([ml])\b", q)
    if m:
        s.size = m.group(1).upper()

    # ---------- intent ----------------------------------------------------------
    q_nv = re.sub(r"\d+([.,]\d+)?\s*%", " ", q)            # "kem 5%" is part of an item name, not a % question
    pct = bool(re.search(r"phan tram|%|\bti le\b|\bty le\b|ti trong|ty trong", q_nv))
    has = lambda rx: re.search(rx, q) is not None
    q_ng = q.replace(s.group_name, " ") if s.group_name else q   # group names like "giảm đau" are not "giảm"
    has_ng = lambda rx: re.search(rx, q_ng) is not None
    purchase = has(r"\bnhap\b|nha cung cap|\bncc\b")
    if purchase and has(r"gia nhap binh quan|binh quan.{0,40}(mot|1) \w+|nhap.{0,60}binh quan"):
        s.intent, s.answer_type = "import_avg_price", "number"
    elif purchase and has(r"nha cung cap (nao|chinh)|ncc (nao|chinh)") and not has(r"bao nhieu (phieu|lan)"):
        s.intent, s.answer_type = "top_supplier", "string"
        s.metric = "receipts" if has(r"chinh|nhieu phieu|nhieu lan") else "amount"
    elif purchase and has(r"tong (tien|gia tri)|tien nhap|gia tri.{0,20}phieu nhap|nhap (hang )?(het )?bao nhieu tien|chi.{0,20}nhap") \
            and not has(r"tru|chenh|so ban|ban ra"):
        s.intent, s.answer_type = "import_total", "number"
    elif has(r"tang gia|len gia|doi gia|gia moi|tang len") and has(r"ke tu|ngay ke"):
        s.intent, s.answer_type = "price_effect", "percent"
    elif has(r"(ban|gia ban|don gia|gia).{0,40}(bao nhieu|may).{0,20}mot (ly|lon|chai|hop|goi|vien|dia|binh|cai|phan|cay|bao|kg|vi|tuyp|bich|thung)") \
            or has(r"voi don gia bao nhieu") or has(r"gia ban .{0,60}ngay \d"):
        s.intent, s.answer_type = "unit_price_on_day", "number"
    elif has(r"tang lien tuc|giam lien tuc|lien tuc tang|lien tuc giam|thang sau (co )?(luon )?(cao|thap) hon thang truoc"):
        s.intent, s.answer_type = "monotonic", "boolean"
        s.flags.append("decreasing" if has(r"giam lien tuc|lien tuc giam|thap hon thang truoc") else "increasing")
    elif has(r"doanh thu") and ((has(r"moi thang|tung thang|theo thu tu thoi gian") and
                                 (has(r"(\d+|hai|ba|bon|nam|sau) thang lien (nhau|tiep)") or has(r"tu thang \d")))
                                or (len(re.findall(r"\b\d{1,2}/\d{4}\b", q)) >= 2
                                    and has(r"lan luot|theo thu tu|tung thang|moi thang"))):
        s.intent, s.answer_type = "revenue_series", "list"
        m = re.search(r"(\d+|hai|ba|bon|nam|sau) thang lien", q)
        s.k = _int(m.group(1)) if m else 3
    elif has(r"\bhuy\b") and pct and not has(r"khach.{0,20}huy nhieu"):
        s.intent, s.answer_type = "cancel_rate", "percent"
    elif has(r"\bhuy\b") and has(r"(may|bao nhieu|so) (don|hoa don)|don (da |bi )?huy|bi huy may|huy may") \
            and not has(r"khach"):
        s.intent, s.answer_type = "n_cancelled", "number"
    elif (has(r"ca chieu.{0,30}(chenh|hon|kem|so).{0,20}ca sang|ca sang.{0,30}(chenh|hon|kem|so).{0,20}ca chieu")
          or (has(r"ca sang") and has(r"ca chieu") and has(r"chenh|hon|kem|so voi|so vs"))) and pct:
        s.intent, s.answer_type = "shift_diff_pct", "percent"
        s.metric = "orders" if has(r"so don|bao nhieu don|may don") else "revenue"
    elif (has(r"ti trong|chiem bao nhieu|chiem may|phan tram doanh thu|% doanh thu|% tong doanh thu|phan cua")
          or (pct and has(r"tong doanh thu"))) and not has_ng(r"tang|giam|thay doi|so voi") \
            and not (s.threshold is not None and has(r"\bkhach\b") and has(r"\bdon\b")):
        s.intent, s.answer_type = "share", "percent"
    elif pct and has_ng(r"tang|giam|thay doi|hon|kem|chenh|tang truong|so voi|so vs"):
        s.intent, s.answer_type = "pct_change", "percent"
        s.metric = "orders" if has(r"so don|bao nhieu don") else "revenue"
    elif has(r"(trung binh|binh quan).{0,30}(moi|tren|mot|1) (don|hoa don|bill)|gia tri trung binh (mot|moi) don"):
        s.intent, s.answer_type = "aov", "number"
    elif has(r"\bngay nao\b|\bhom nao\b") and has(r"nhat|cao|dat|nhieu"):
        s.intent, s.answer_type, s.entity = "best_period", "date", "day"
        s.metric = "orders" if has(r"nhieu don|dong khach|dat khach") else "revenue"
    elif has(r"\bthang nao\b") and has(r"nhat|cao|nhieu"):
        s.intent, s.answer_type, s.entity = "best_period", "date", "month"
        s.metric = "orders" if has(r"nhieu don") else "revenue"
    elif has(r"(dem|bao nhieu|may|so|co nhieu) ngay") and s.threshold is not None and has(r"doanh thu|ban duoc|tien"):
        s.intent, s.answer_type = "n_days_threshold", "number"
    elif has(r"(bao nhieu|may|so|dem) (don|hoa don)") and s.threshold is not None and has(r"doanh thu|tien hang|gia tri|tri gia"):
        s.intent, s.answer_type = "n_orders_threshold", "number"
    elif has(r"\bkhach\b") and s.threshold is not None and has(r"\bdon\b") and pct:
        s.intent, s.answer_type = "pct_customers_min_orders", "percent"
    elif has(r"\bkhach\b") and s.threshold is not None and has(r"\bdon\b"):
        s.intent, s.answer_type = "n_customers_min_orders", "number"
    elif has(r"mat hang|san pham|mon") and s.threshold is not None and has(r"\bdon\b") and has(r"co mat|xuat hien|nam trong"):
        s.intent, s.answer_type = "n_items_min_orders", "number"
    elif (has(r"\bkhach\b") or has(r"\bai\b.{0,20}\bmua\b")) and not purchase and \
            has(r"nhieu tien nhat|chi tieu nhieu nhat|mua nhieu nhat|nhieu don nhat|mua nhieu don|nhieu lan nhat|mua hang nhieu|"
                r"(tong tien|tien mua|doanh thu|so tien).{0,12}(cao nhat|nhieu nhat|lon nhat)"):
        s.intent, s.entity = "top_customers", "customer"
        s.metric = "orders" if has(r"nhieu don|nhieu lan|theo so don|so don.{0,15}nhieu nhat") else "revenue"
        s.answer_type = "list" if (s.k or 1) > 1 else "string"
    elif has(r"nhan vien|thu ngan|nguoi ban\b") and has(r"nhat|gioi"):
        s.intent, s.entity, s.metric = "top_staff", "staff", "revenue"
        s.answer_type = "list" if (s.k or 1) > 1 else "string"
    elif has(r"ban chay|ban nhieu nhat|ban duoc nhieu nhat|hot nhat|ban it nhat|ban cham|ban e\b|it duoc mua|nhieu tien nhat|doanh thu cao nhat|ban nhieu nhat") \
            and has(r"mat hang|san pham|mon|hang|thuoc|gi|nao|top|loai|do uong|nuoc") and not purchase \
            and not has(r"ngay nao|thang nao|khach|nhan vien|nhom nao|khong ban duoc|lan nao|bao nhieu$|la bao nhieu"):
        s.intent, s.entity = "top_items", "item"
        s.ascending = has(r"ban it nhat|ban cham|it duoc mua|ban e")
        s.metric = "revenue" if has(r"nhieu tien|doanh thu cao|doanh thu lon") else "quantity"
        s.answer_type = "list" if (s.k or 1) > 1 or has(r"nhung (mat hang|san pham|mon)|liet ke") else "string"
    elif has(r"(khong ban|chua ban) (duoc|ra)|bi e\b|\bhang e\b|\bmat hang e\b|khong ban duoc lan nao") and has(r"bao nhieu|may|so|dem"):
        s.intent, s.answer_type = "n_unsold_items", "number"
    elif has(r"so ly size|ly size|size [ml]") and has(r"bao nhieu|may|so"):
        s.intent, s.answer_type = "qty_size", "number"
    elif has(r"(bao nhieu|may|so|dem) (khach|nguoi mua)") and not has(r"\bno\b"):
        s.intent, s.answer_type = "n_customers", "number"
    elif has(r"(bao nhieu|may|so|tong|dem) (don|hoa don|bill|don hang)") or has(r"\bso don\b"):
        s.intent, s.answer_type = "n_orders", "number"
    elif has(r"ban (duoc|ra)? ?(bao nhieu|may|tong cong bao nhieu|tong cong may) (ly|lon|chai|hop|goi|vien|dia|binh|cai|phan|cay|bao|kg|vi|tuyp|bich|thung|cuon|m3|bo|chiec|lo|suat|to|tan|tam|vien nang|goi|o|ky|lit)\b") \
            or has(r"\bso (ly|lon|chai|hop|goi|vien|dia|binh|cai|phan|cay|bao|kg|vi|tuyp|bich|thung|cuon) .{0,60}ban (duoc|ra)") \
            or has(r"tong so (ly|lon|chai|hop|goi|vien|dia|binh|cai|phan|cay|bao|kg|vi|tuyp|bich|thung|cuon)"):
        s.intent, s.answer_type = "qty", "number"
        units = r"(ly|lon|chai|hop|goi|vien|dia|binh|cai|phan|cay|bao|kg|vi|tuyp|bich|thung|cuon|m3|bo|chiec|lo|suat|to|tan|tam|lit|ong|noi|o)"
        m = re.search(rf"\b(?:bao nhieu|may|tong so|so|tong cong|nhieu)\s+{units}\b(?! nhieu)", q)
        s.unit_word = m.group(1) if m else None
    elif not purchase and has(r"doanh thu|ban (duoc|ra)? ?(bao nhieu|bn|nhieu|may) tien|thu (duoc|ve)( bao nhieu)? tien|tien (hang|ban)|bao nhieu tien|ban duoc bao nhieu$"):
        s.intent, s.answer_type, s.metric = "revenue", "number", "revenue"
        if pct:
            s.intent, s.answer_type = "share", "percent"
    # questions the template library does not cover -> leave to the LLM with the right type hint
    if has(r"\bthu (may|nao)\b|ngay nao trong tuan"):
        s.intent, s.answer_type = None, "string"
    if s.intent not in ("monotonic",) and re.search(r"\b(co|da)\b.*\b(khong|chua)\s*(a|ha|nhi|vay|z)?\s*$", q) \
            and not has(r"\b(bao nhieu|may|nao|gi|ai)\b"):
        s.intent, s.answer_type = None, "boolean"
    if has(r"\b(tung|moi|theo) (chi nhanh|nhan vien|kenh|nhom|thang|ngay|san pham|mat hang|khach|hinh thuc|ca)\b") \
            and s.intent in ("revenue", "n_orders", "qty", "aov"):
        s.intent, s.answer_type = None, "table"
    if has(r"\bkho\b|ton kho|\bcon ton\b|het han|han dung|han su dung|\blo hang\b|\blo\b.{0,30}\bhan\b") \
            and s.intent not in ("import_total", "import_avg_price", "top_supplier"):
        s.intent = None
    # several values requested ("mỗi loại", "lần lượt") -> list answer; leave to the LLM
    if s.intent in ("revenue", "n_orders", "qty", "aov") and has(r"moi ben|tung loai|moi nhom|tung nhom|moi kenh") \
            or s.intent in ("qty", "aov") and has(r"moi loai|lan luot"):
        s.intent, s.answer_type = None, "list"
    if s.intent in ("revenue", "n_orders", "aov") and (len(re.findall(r"doanh thu|\bso don\b", q)) >= 2
                                                       or has(r"\bva (cac )?(don|doanh thu)\b|lan luot|moi loai")):
        fm = _filter_mentions(q)
        if len(fm) >= 2 and len({k for k, _ in fm}) == 1 and s.intent in ("revenue", "n_orders"):
            s.base_intent, s.intent, s.answer_type, s.multi = s.intent, "multi_filter", "list", fm
            s.channel = s.payment = None
        else:
            s.intent, s.answer_type = None, "list"
    if s.intent and not purchase and has(r"cong no|tra no|thu no|mua chiu|thu hoi|\bno\b|thieu tien|con thieu|tra gop"):
        s.intent = None
    # profit / cost questions are not revenue
    if s.intent and has(r"\blai\b|loi nhuan|gia von|\blo\b (bao nhieu|may)|\bloi\b (bao nhieu|duoc)"):
        s.intent = None
    # a payment / channel phrase we could not map to a filter: do not answer the unfiltered question
    if s.intent and not s.payment and not s.multi and has(r"\b(tra|thanh toan) (bang|qua)\b"):
        s.intent = None
    if s.intent == "revenue" and has(r"cao hon|thap hon") and has(r"co phai|khong\b|\ba\b$"):
        s.intent, s.answer_type = None, "boolean"
    if s.intent is None:
        s.answer_type = _fallback_type(q)
    return s


def _fallback_type(q: str) -> str:
    if re.search(r"\b(co|da)\b.*\b(khong|chua)\s*$", q) and not re.search(r"\b(bao nhieu|nao|gi|ai)\b", q):
        return "boolean"
    if re.search(r"phan tram|%|ti le|ti trong", q):
        return "percent"
    if re.search(r"\b(ngay nao|hom nao)\b", q):
        return "date"
    if re.search(r"\bthang nao\b", q):
        return "date"
    if re.search(r"\b(liet ke|nhung|cac|danh sach|top)\b", q):
        return "list"
    if re.search(r"\b(la gi|cai gi|nao|ai)\b", q):
        return "string"
    return "number"


# Vocabulary the deterministic layer understands; unexplained tokens weaken a template.
VOCAB = set("""
la gi nao bao nhieu nhat cua cho trong tuan thang nam quy nay roi truoc qua ngay hom kia co khong va voi tai o cac
nhung mot ai con da chua duoc ban mua chay the nhi vay a ha ah oi nhe giup minh toi em anh chi xem hay hoac so cung ky
ngoai tang giam top theo tung moi hien gan day vua nua tu den dau cuoi tet ram am lich duong thu hai ba bon sau bay
tam chin muoi chu may bang tong tat ca het han sap kho hang san pham mat mon thuoc ly cai hop chai cay vien goi lo thung
m m3 kg khach doanh luong tien don hoa nhanh nhan vien phan tram trieu nghin ngan ty vuot hon cao thap it lon nho dung
phai tra no thanh gia tri dat duoi tren nguoi lieu ke danh sach dong nhieu ra loai nhom thu ve
lai gop rong loi nhuan thanh toan cong ty le ti so voi nhu thoi diem luc dip le giong gio ve tinh lien tiep do dua
ton su dung lai check gium coi cho biet hoi nhe nha ta z e ee voi a oi dum dem binh quan trung bill tro len it
tung ca sang chieu chenh kem lech tuan le dau rieng xet theo dat bao doanh thu chiem trong ti trong cua hang quan
nha thuoc tiem shop dem rieng vay do day khach le nhung mang vs total revenue sp tong cong tinh den het lien truoc
huy bi don hang hoa don vua dang khong nhieu tien chi tieu mua dia lon chai tuyp bich vi binh ve cap tat
thay doi ghi thu tu lan luot hot ban ra so sanh cac ngay ngay tieu ngay ke giai doan khoang hay
""".split())


def unexplained_tokens(q_folded: str, links=()) -> list[str]:
    covered = set()
    for l in links:
        if getattr(l, "score", 0) >= 85:
            covered.update(l.span.split())
    return [t for t in q_folded.split()
            if t not in VOCAB and t not in covered and not re.match(r"^[\d.,/%≥]+([.,]\d+)?(k|tr|d|m|kg|ml|l|g|mg)?$", t)]
