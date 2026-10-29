"""Value linking: find which data values the question refers to.

Questions are typed without diacritics, abbreviated or partially ("trà sữa
trân châu" for "Trà sữa trân châu đường đen"). CHESS (Talaei et al., 2024)
retrieves values with LSH + edit-distance re-ranking; CHASE-SQL (Pourreza et
al., ICLR 2025) uses keyword extraction + LSH for the same purpose. Business
databases here are small, so we can afford an exact variant: every contiguous
token span of the folded question is compared with every distinct folded
value (rapidfuzz C++ cdist), plus a containment score for partial mentions.

The linker also produces an "unlinked item mention" signal: if the question
names an item ("bán được bao nhiêu X") and X matches nothing in the data,
that is positive evidence for the PRACTIQ category "nonexistent filter value"
-> `unanswerable`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .loader import Business, is_text
from .textnorm import fold, metric_norm

STOP = set("""ban duoc bao nhieu la gi nao nhat cua cho trong tuan thang nam nay roi truoc qua ngay hom
co khong va voi tai o cac nhung mot cai ly hop chai bao cay vien goi m kg thung lo ai con da chua the
san pham mat hang hang mon thuoc khach doanh thu so luong top theo tung moi gan day vua toi nua tu den
tien don hoa nhieu it cao thap loai mua chay sap het han kho tong vay a ha nhi nhe ta do day the z oi ah u
dum giup gium cho voi minh toi em anh chi la duoc roi ban ra tong cong luot lan""".split())

UNIT_W = r"(?:ly|cai|hop|chai|bao|cay|vien|goi|lo|thung|m3|m|kg|tan|phan|suat|to|dia|chiec|bo|cuon|tam|vi|tuyp|lon|lit|l)"
TIME_W = r"(?:tuan|thang|nam|hom|ngay|quy|tu|den|trong|vao|dip|tet|\d+ ngay|o|tai|cua|cho)"


@dataclass
class Link:
    value: str
    table: str
    column: str
    role: str | None
    score: float
    span: str


class ValueIndex:
    def __init__(self, biz: Business, roles: dict | None = None, max_distinct: int = 5000):
        self.entries = []          # (fold, orig, table, col, role)
        roles = roles or {}
        for t in biz.tables.values():
            for c in t.df.columns:
                s = t.df[c]
                if c == "_sheet" or not is_text(s):
                    continue
                u = s.dropna().astype(str).unique()
                if len(u) > max_distinct or len(u) == 0:
                    continue
                role = roles.get((t.name, c))
                avg = np.mean([len(x) for x in u[:200]])
                if role is None and (avg > 60 or len(u) > 0.9 * len(s) > 200):
                    continue    # free text / ids with no role: not useful for linking
                for v in u:
                    f = fold(v)
                    if len(f) >= 2:
                        self.entries.append((f, v, t.name, c, role))
        self.keys = [e[0] for e in self.entries]

    def _spans(self, qf: str, max_len: int = 8):
        toks = qf.split()
        out = set()
        for i in range(len(toks)):
            for j in range(i + 1, min(len(toks), i + max_len) + 1):
                span = toks[i:j]
                if all(t in STOP for t in span):
                    continue
                out.add(" ".join(span))
        return sorted(out, key=len, reverse=True)

    def link(self, question: str, top: int = 12, cutoff: float = 80) -> list[Link]:
        if not self.entries:
            return []
        qf = fold(question)
        spans = self._spans(qf)
        if not spans:
            return []
        best: dict[int, tuple[float, str]] = {}
        # 1) full-string similarity span <-> value (handles typos / missing accents)
        from .question import VOCAB as _V0
        M = process.cdist(spans, self.keys, scorer=fuzz.ratio, workers=1, score_cutoff=cutoff)
        si, vi = np.nonzero(M)
        vocab_only = [all(t in _V0 for t in sp.split()) for sp in spans]
        for a, b in zip(si, vi):
            sc = float(M[a, b])
            if vocab_only[a] and sc < 100:
                continue          # "nha thuoc ban" is question wording, not a near-miss of "Nhà thuốc Lan"
            if sc > best.get(b, (0, ""))[0]:
                best[b] = (sc, spans[a])
        # 2) partial mention: multi-token span contained in the value (e.g. a product prefix)
        from .question import VOCAB as _V
        # partial containment needs at least one content word ("nha thuoc" alone is question vocabulary)
        multi = [s for s in spans if (len(s.split()) >= 2 or re.match(r"^[a-z]+\d+$", s))
                 and not all(t in _V for t in s.split())]
        for b, key in enumerate(self.keys):
            pk = f" {key} "
            for s in multi:
                if f" {s} " in pk:
                    cov = len(s) / max(1, len(key))
                    sc = 82 + 16 * cov
                    if sc > best.get(b, (0, ""))[0]:
                        best[b] = (sc, s)
                    break
            # 3) value fully contained in question (short codes like "sp0042")
            if f" {key} " in f" {qf} " and len(key) >= 3 and key not in STOP and self.entries[b][4] != "unit":
                sc = 100.0
                if sc > best.get(b, (0, ""))[0]:
                    best[b] = (sc, key)
        # accents disambiguate short values: in an accented question "trả" is not the drink "Trà"
        accented = fold(question) != metric_norm(question).replace("đ", "d")
        qa = f" {metric_norm(question)} "
        from .question import VOCAB
        for b, (sc, sp) in list(best.items()):
            if len(sp.split()) > 1 or len(self.entries[b][0]) > 12:
                continue
            va = metric_norm(self.entries[b][1])
            if accented and f" {va} " not in qa:
                best[b] = (sc - 30, sp)
            elif not accented and sp in VOCAB and self.entries[b][0] == sp:
                best[b] = (sc - 15, sp)       # "tra" in "tra bang chuyen khoan" is a question word first
        links = [Link(self.entries[b][1], self.entries[b][2], self.entries[b][3], self.entries[b][4], sc, sp)
                 for b, (sc, sp) in best.items() if sc >= cutoff]
        links.sort(key=lambda l: (-l.score, -len(l.span), l.value))
        # de-duplicate same value in the same column
        seen, out = set(), []
        for l in links:
            k = (l.table, l.column, l.value)
            if k not in seen:
                seen.add(k)
                out.append(l)
        return out[:top]


def unit_value(mention: str | None, vindex: "ValueIndex") -> str | None:
    """The mention is exactly a unit of measure recorded by the shop ("nồi", "ly", "bao") -> folded unit."""
    if not mention:
        return None
    m = fold(mention)
    for e in vindex.entries:
        if e[4] in ("unit", "base_unit") and e[0] == m:
            return m
    return None


def mention_linked(mention: str | None, vindex: "ValueIndex", cutoff: float = 80) -> bool | None:
    """None if no mention; True if the mention matches some value; False if it matches nothing."""
    if not mention or unit_value(mention, vindex):
        return None
    links = vindex.link(mention, top=3, cutoff=cutoff)
    return any(l.score >= cutoff and (l.role or "").split("_")[0] in ("product", "category", None) for l in links)


def item_mention(question: str) -> str | None:
    """The item phrase in "bán được bao nhiêu (ly) X ..." questions, folded."""
    from .textnorm import qnorm
    q = qnorm(question)
    m = re.search(rf"\bban (?:duoc )?(?:bao nhieu|may|bn) (?:{UNIT_W} )?(.+?)(?:\s+{TIME_W}\b|$)", q)
    if not m:
        return None
    toks = m.group(1).strip().split()
    while toks and toks[0] in STOP:
        toks.pop(0)
    if toks and toks[0] in ("size", "loai", "nhom", "tat", "moi", "cac", "nhung", "hang", "do", "mon", "khach", "tien",
                            "dong", "doanh", "don", "hoa", "bill", "lan", "luot", "phan", "%"):
        return None
    while toks and toks[-1] in STOP:
        toks.pop()
    return " ".join(toks) if toks else None


def resolve_entity(links: list[Link], roles_prefix=("product",), min_score=85.0):
    """Best linked entity of a kind; returns (Link|None, ambiguous: bool)."""
    ok = lambda r: bool(r) and (r in roles_prefix or r.split("_")[0] in roles_prefix)
    cands = [l for l in links if ok(l.role) and l.score >= min_score]
    if not cands:
        return None, False
    top = cands[0]
    rivals = [l for l in cands[1:] if fold(l.value) != fold(top.value) and l.role.split("_")[0] == top.role.split("_")[0]
              and (l.span == top.span or l.score >= top.score - 1e-9)]
    exact = top.score >= 99 and len(top.span) >= 0.9 * len(fold(top.value))
    return top, bool(rivals) and not exact
