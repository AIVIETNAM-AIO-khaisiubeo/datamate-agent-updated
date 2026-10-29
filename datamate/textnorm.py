"""Vietnamese text normalisation.

Two different normal forms are used on purpose:

* ``metric_norm`` reproduces the competition's string-matching rule exactly
  (NFC -> lowercase -> non letter/digit/space -> space -> collapse). It is used
  by the local scorer and to compare candidate answers when voting.
* ``fold`` is an accent-insensitive key used ONLY for matching user text
  against data (questions are often typed without diacritics or with teencode).
  It must never be used to produce an answer: the metric does NOT strip
  diacritics, so answers are always copied verbatim from the data.
"""
from __future__ import annotations

import re
import unicodedata

__all__ = ["nfc", "metric_norm", "fold", "fold_soft", "parse_qty_vn", "expand_abbr", "parse_number_vn",
           "tokens", "contains_phrase"]


def nfc(s) -> str:
    return unicodedata.normalize("NFC", "" if s is None else str(s))


def metric_norm(s) -> str:
    """Exact re-implementation of the official `string` normalisation."""
    s = nfc(s).lower()
    s = "".join(ch if (ch.isalpha() or ch.isdigit() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def fold(s) -> str:
    """Accent/case/punctuation-insensitive key (matching only)."""
    s = metric_norm(s).replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.split())


def fold_soft(s) -> str:
    """Like fold() but keeps punctuation such as '/', '-', '%' (dates, numbers)."""
    s = nfc(s).lower().replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[?!,;:\"'()\[\]]", " ", s)
    return " ".join(s.split())


def tokens(s) -> list[str]:
    return fold(s).split()


def contains_phrase(text_folded: str, phrase_folded: str) -> bool:
    """Whole-token phrase containment on folded strings."""
    if not phrase_folded:
        return False
    return f" {phrase_folded} " in f" {text_folded} "


# Common Vietnamese chat abbreviations / teencode seen in shop-owner questions.
# Keys are FOLDED tokens. Expansion is only used to help intent detection and
# is shown to the LLM next to the raw question (never replaces it).
ABBR = {
    "sp": "sản phẩm", "spham": "sản phẩm", "mh": "mặt hàng", "hh": "hàng hóa",
    "dc": "được", "đc": "được", "dk": "được", "duoc": "được",
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "hong": "không", "khg": "không",
    "kh.hang": "khách hàng", "khach": "khách", "kq": "kết quả",
    "sl": "số lượng", "sluong": "số lượng", "dt": "doanh thu", "ds": "doanh số",
    "bn": "bao nhiêu", "bnh": "bao nhiêu", "bnhieu": "bao nhiêu", "baonhieu": "bao nhiêu",
    "j": "gì", "gi": "gì", "ji": "gì",
    "nv": "nhân viên", "cn": "chi nhánh", "ch": "cửa hàng",
    "dh": "đơn hàng", "đh": "đơn hàng", "hd": "hóa đơn", "hđ": "hóa đơn",
    "ng": "người", "ngta": "người ta", "tr": "triệu", "trieu": "triệu",
    "thg": "tháng", "th": "tháng", "t": "tháng",
    "r": "rồi", "rui": "rồi", "roi": "rồi", "ms": "mới", "vs": "với",
    "z": "vậy", "v": "vậy", "nhiu": "nhiêu", "nhìu": "nhiều", "nhiu.": "nhiều",
    "hsd": "hạn sử dụng", "tk": "tồn kho", "lnhuan": "lợi nhuận", "ln": "lợi nhuận",
    "ntn": "như thế nào", "hnay": "hôm nay", "hqua": "hôm qua", "hum": "hôm",
    "wa": "qua", "qa": "qua", "tuan": "tuần", "thang": "tháng",
    "bnhiu": "bao nhiêu", "bnhieu.": "bao nhiêu", "bao nhiu": "bao nhiêu", "nhiu": "nhiêu", "trc": "trước",
    "dthu": "doanh thu", "dt": "doanh thu", "rev": "doanh thu", "revenue": "doanh thu", "bill": "đơn",
    "total": "tổng", "sp": "sản phẩm", "kl": "khách lẻ", "hdon": "hóa đơn", "đơn": "đơn",
    "bữa": "hôm", "bua": "hôm", "order": "đơn", "orders": "đơn", "don hang": "đơn hàng", "nay.": "nay", "thangs": "tháng", "ngayf": "ngày", "khachs": "khách",
}


def expand_abbr(q: str) -> str:
    out = []
    for tok in nfc(q).split():
        key = fold(tok)
        out.append(ABBR.get(key, tok) if key in ABBR and len(key) <= 8 else tok)
    return " ".join(out)


_NUM_RE = re.compile(r"^([-+]?)(\d[\d.,\s]*)(\s*)([a-zđ%]*)$")


def parse_number_vn(x):
    """Parse Vietnamese-formatted numbers/money.

    '1.250.000đ' -> 1250000 ; '1,25tr' -> 1250000 ; '500k' -> 500000 ;
    '12,5' -> 12.5 ; '(1.000)' -> -1000 ; '15%' -> 15.0 . Returns None if not a number.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = nfc(str(x)).strip().lower()
    if not s or s in {"-", "nan", "none", "null"}:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    s = s.replace("vnđ", "").replace("vnd", "").replace("đồng", "").replace("dong", "").strip()
    s = s.rstrip("đ").strip()
    m = re.match(r"^([-+]?)(\d+)\s*(tr|triệu|trieu|k|nghìn|ngàn|tỷ|ty)\s*(\d{1,3})$", s)   # 1tr440, 1tr2, 2k5
    if m:
        sign, whole, unit, frac = m.groups()
        mult = {"tr": 1e6, "triệu": 1e6, "trieu": 1e6, "k": 1e3, "nghìn": 1e3, "ngàn": 1e3, "tỷ": 1e9, "ty": 1e9}[unit]
        v = (int(whole) + float("0." + frac)) * mult
        v = -v if sign == "-" else v
        return -v if neg else v
    m = _NUM_RE.match(s)
    if not m:
        return None
    sign, num, _, unit = m.groups()
    num = num.replace(" ", "")
    mult = {"": 1, "k": 1e3, "nghìn": 1e3, "ngàn": 1e3, "tr": 1e6, "triệu": 1e6,
            "tỷ": 1e9, "ty": 1e9, "%": 1}.get(unit)
    if mult is None:
        return None
    # decide decimal separator
    if "," in num and "." in num:
        if num.rfind(",") > num.rfind("."):          # 1.234,5
            num = num.replace(".", "").replace(",", ".")
        else:                                         # 1,234.5
            num = num.replace(",", "")
    elif "." in num:
        parts = num.split(".")
        if len(parts) > 2 or (len(parts[-1]) == 3 and mult == 1 and len(parts[0]) <= 3):
            num = num.replace(".", "")                # 1.250.000 thousand separators
    elif "," in num:
        parts = num.split(",")
        if len(parts) > 2 or (len(parts[-1]) == 3 and mult == 1 and len(parts[0]) <= 3):
            num = num.replace(",", "")
        else:
            num = num.replace(",", ".")               # 12,5 decimal comma
    try:
        v = float(num) * mult
    except ValueError:
        return None
    if sign == "-":
        v = -v
    return -v if neg else v


_QTY_RE = re.compile(r"^\s*([-+]?\d[\d.,]*)\s+([^\d\s].{0,20})$")


def parse_qty_vn(x):
    """Quantity cells may carry their unit: '1 ly', '5 viên', '2 bình' -> 1, 5, 2.
    Falls back to parse_number_vn. A trailing word is accepted only if it is short
    and contains no digits (so '7 Up 330ml' stays text)."""
    v = parse_number_vn(x)
    if v is not None or x is None or isinstance(x, (int, float)):
        return v
    m = _QTY_RE.match(nfc(str(x)).lower())
    if not m:
        return None
    return parse_number_vn(m.group(1))


_TELEX_PAIRS = (("dd", "d"), ("aa", "a"), ("aw", "a"), ("ee", "e"), ("oo", "o"), ("ow", "o"), ("uw", "u"))
_VALID_END = re.compile(r"(?:[aeiouy]|c|ch|m|n|ng|nh|p|t)$")
_SYLLABLE = re.compile(r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?"
                       r"(?:[aeiouy]{1,3})(?:ch|ng|nh|[cmnpt])?$")


def telex_fix(tok: str) -> str:
    """Folded token typed with leftover Telex marks -> folded syllable.
    'toongr'->'tong', 'dduwowcj'->'duoc', 'thangs'->'thang', 'tieenf'->'tien', 'nawm'->'nam'.
    Vietnamese syllables never end in s/f/r/x/j and never contain the Telex digraphs,
    so the rewrite only fires on tokens that cannot be Vietnamese as typed."""
    if not tok.isalpha() or len(tok) < 3:
        return tok
    t = tok
    changed = False
    for a, b in _TELEX_PAIRS:
        if a in t:
            t, changed = t.replace(a, b), True
    if (len(t) >= 3 or (changed and len(t) >= 2)) and t[-1] in "sfrxj" and _VALID_END.search(t[:-1]):
        t, changed = t[:-1], True
    if "w" in t:
        t, changed = t.replace("w", "u"), True
    # only accept the repair if it yields ONE Vietnamese syllable ("acyclovir" stays a drug name)
    return t if changed and _SYLLABLE.match(t) else tok


# folded abbreviation -> accepted surface forms (anything else with diacritics is a real word)
ABBR_PLAIN_ONLY = {"hong": {"hong", "hông"}, "ch": {"ch"}, "th": {"th"}, "ng": {"ng"}, "t": {"t"},
                   "v": {"v"}, "r": {"r"}, "gi": {"gi", "gì"}, "duoc": {"duoc", "được", "đuoc", "dươc", "đươc"},
                   "thang": {"thang", "tháng", "thangs"}, "tuan": {"tuan", "tuần"}, "roi": {"roi", "rồi"},
                   "khach": {"khach", "khách"}, "trieu": {"trieu", "triệu"}}


# loanwords that look like Telex leftovers but must be kept as typed
ENGLISH_KEEP = set("""order orders bill bills revenue total top ship shipper shop store sales sale voucher combo size
online offline phone zalo cod ok best seller items item customer customers new news app web website service
staff cashier report status cancel cancelled return returned refund""".split())


def qnorm(q: str) -> str:
    """Canonical folded question: accents/case/punct removed, Telex leftovers fixed,
    chat abbreviations expanded. Used by every rule-based analyser."""
    toks = []
    orig_by_fold: dict = {}
    for w in re.findall(r"\w+", nfc(q).lower()):
        orig_by_fold.setdefault(fold(w), set()).add(w)
    for tok in fold_soft(q).replace("/", " / ").split():
        if tok == "/":
            toks.append("/")
            continue
        t = tok.strip(".,;:!?")
        if t not in ABBR and t not in ENGLISH_KEEP and not any(ch.isdigit() for ch in t):
            t = telex_fix(t)
        exp = ABBR.get(t)
        if exp and t in ABBR_PLAIN_ONLY and orig_by_fold.get(t) and not orig_by_fold[t] & ABBR_PLAIN_ONLY[t]:
            exp = None   # "họng" (throat) is not the chat form "hông/hong" (= không)
        toks.append(fold(exp) if exp else t)
    return re.sub(r"\s*/\s*", "/", " ".join(toks))
