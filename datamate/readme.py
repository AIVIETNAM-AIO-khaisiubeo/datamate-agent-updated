"""Parse a business README.md into machine-usable hints.

READMEs vary from a full data dictionary to almost nothing (dataset spec), so
every field here is optional and only *adds* evidence to the column-role
tagger; the README text itself is always shown verbatim to the LLM.

Extracted:
  * columns[file][column] = description  (sections "## Cột trong `file`")
  * conventions: bullet list of the "## Quy ước" section
  * data_end: "Dữ liệu có đến hết dd/mm/yyyy"
  * conversions: "Item: 1 thùng = 24 lon" lines -> (item, big_unit, factor, base_unit)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .textnorm import fold, nfc


@dataclass
class ReadmeInfo:
    text: str = ""
    columns: dict = field(default_factory=dict)       # file basename -> {column header -> description}
    conventions: list = field(default_factory=list)
    data_end: date | None = None
    data_start: date | None = None
    conversions: list = field(default_factory=list)   # (item, big_unit, factor, base_unit)

    def desc_for(self, files: list[str], column: str) -> str | None:
        """Description of `column` in any of `files` (falls back to any file)."""
        col_f = fold(column)
        for f in files:
            for fk, cols in self.columns.items():
                if fold(fk) == fold(f) or fold(fk.split(":")[-1]) == fold(f):
                    for c, d in cols.items():
                        if fold(c) == col_f:
                            return d
        hits = {d for cols in self.columns.values() for c, d in cols.items() if fold(c) == col_f}
        return hits.pop() if len(hits) == 1 else None


_SEC = re.compile(r"^##\s+(.*)$", re.M)
_COL = re.compile(r"^\s*[-*]\s+\*\*(.+?)\*\*\s*[:：]\s*(.+?)\s*$", re.M)
_FILE_IN_HDR = re.compile(r"`([^`]+)`")
_END = re.compile(r"(?:dữ liệu|du lieu)[^.\n]{0,40}?(?:đến hết|den het|tới hết|toi het|đến|den)\s*(?:ngày\s*)?(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", re.I)
_START = re.compile(r"(?:dữ liệu|du lieu)[^.\n]{0,30}?(?:từ|tu|bắt đầu|bat dau)\s*(?:ngày\s*)?(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", re.I)
_CONV = re.compile(r"^\s*[-*]\s+(.+?)\s*:\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$", re.M)
_CONV_INLINE = re.compile(r"1\s+([^\d=;,]+?)\s*=\s*([\d.,]+)\s+([^\d;,.]+)")


def parse_readme(text: str) -> ReadmeInfo:
    info = ReadmeInfo(text=nfc(text or ""))
    t = info.text
    if not t:
        return info
    secs = [(m.start(), m.end(), m.group(1)) for m in _SEC.finditer(t)] + [(len(t), len(t), "")]
    for (s0, e0, title), (s1, _, _) in zip(secs, secs[1:]):
        body = t[e0:s1]
        tf = fold(title)
        if tf.startswith("cot trong") or tf.startswith("cot cua") or "columns" in tf:
            fm = _FILE_IN_HDR.search(title)
            fname = fm.group(1) if fm else title.split()[-1]
            info.columns[fname] = {m.group(1).strip(): m.group(2).strip() for m in _COL.finditer(body)}
        elif tf.startswith("quy uoc") or tf.startswith("ghi chu") or tf.startswith("luu y"):
            info.conventions += [ln.strip()[1:].strip() for ln in body.splitlines() if ln.strip().startswith(("-", "*"))]
    m = _END.search(t)
    if m:
        try:
            info.data_end = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    m = _START.search(t)
    if m:
        try:
            info.data_start = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    for m in _CONV.finditer(t):
        item, big, fac, base = m.groups()
        f = _num(fac)
        if f and len(item) < 80:
            info.conversions.append((item.strip(" *`"), big.strip(), f, base.strip(" .")))
    return info


def parse_conversion_text(s: str):
    """'1 lốc = 4 gói; 1 thùng = 24 gói' -> [('lốc', 4, 'gói'), ('thùng', 24, 'gói')]"""
    out = []
    for m in _CONV_INLINE.finditer(nfc(str(s or ""))):
        f = _num(m.group(2))
        if f:
            out.append((m.group(1).strip(), f, m.group(3).strip()))
    return out


def _num(s):
    try:
        return float(str(s).replace(",", "."))
    except ValueError:
        return None
