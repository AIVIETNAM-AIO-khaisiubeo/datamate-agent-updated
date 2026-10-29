"""Deterministic resolution of Vietnamese time expressions against as_of_date.

The LLM never does date arithmetic: every recognised expression becomes a
half-open interval [start, end) passed to generators as SQL literals ("evidence"
in the sense of BIRD, Li et al. NeurIPS 2023, generated automatically). The
conventions come from the split's rulebook (rulebook.settings), e.g. in the
official rulebook: weeks start Monday; "tuần trước" = the full previous week;
"tháng này" = day 1 .. today, compared with the same number of days of the
previous month; "N ngày gần nhất" ends today; "tháng M" without a year = this
year unless after today; Tết Y = 23 tháng Chạp (lunar Y-1) .. mùng 10 (lunar Y);
"so với trước Tết" = the same number of days right before 23 tháng Chạp.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

from .lunar import LunarCalendar
from .textnorm import qnorm

DAY = timedelta(days=1)


@dataclass
class TimeRange:
    label: str
    start: date
    end: date               # exclusive
    role: str = "primary"   # primary | comparison
    grain: str = "day"      # day | week | month | quarter | year | range | tet
    to_date: bool = False   # period containing as_of, cut at as_of

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def sql(self, col: str = "<date_col>") -> str:
        return f"{col} >= DATE '{self.start.isoformat()}' AND {col} < DATE '{self.end.isoformat()}'"

    def as_dict(self):
        return {"label": self.label, "start": self.start.isoformat(), "end_exclusive": self.end.isoformat(),
                "role": self.role, "grain": self.grain}


@dataclass
class TimeContext:
    as_of: date
    ranges: list[TimeRange] = field(default_factory=list)
    horizon_days: int | None = None      # "hết hạn trong N ngày tới"
    month_answer: bool = False           # answer is a month -> "YYYY-MM"
    shift: str | None = None             # "sang" | "chieu"
    snapshot: date | None = None         # "đến hết ngày X" / "cuối ngày hôm nay"
    weekday: int | None = None           # "các ngày thứ X"
    explicit: bool = False               # any time expression found

    @property
    def primary(self) -> TimeRange | None:
        return next((r for r in self.ranges if r.role == "primary"), None)

    @property
    def comparison(self) -> TimeRange | None:
        return next((r for r in self.ranges if r.role == "comparison"), None)

    def describe(self) -> str:
        lines = [f"- {r.role.upper()} '{r.label}': {r.sql('date_col')}   ({r.start} .. {r.end - DAY} inclusive, {r.days} days)"
                 for r in self.ranges]
        if self.horizon_days is not None:
            lines.append(f"- HORIZON: from DATE '{self.as_of}' to DATE '{self.as_of + timedelta(days=self.horizon_days)}' inclusive")
        if self.snapshot is not None:
            lines.append(f"- SNAPSHOT: state at the end of {self.snapshot} (all records up to and including that day)")
        if self.shift:
            lines.append(f"- SHIFT: ca {self.shift} (filter on order time-of-day, see rulebook)")
        if not lines:
            return "- (no time expression detected: use all data up to and including as_of_date)"
        return "\n".join(lines)


_NUMW = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9,
         "muoi": 10, "muoi lam": 15, "hai muoi": 20, "ba muoi": 30}
WEEKDAY_W = {"thu hai": 0, "thu 2": 0, "thu ba": 1, "thu 3": 1, "thu tu": 2, "thu 4": 2, "thu nam": 3, "thu 5": 3,
             "thu sau": 4, "thu 6": 4, "thu bay": 5, "thu 7": 5, "chu nhat": 6, "cn": 6}


def _n(tok: str) -> int | None:
    return int(tok) if tok.isdigit() else _NUMW.get(tok)


class TimeResolver:
    def __init__(self, rulebook, lunar: LunarCalendar | None = None):
        self.rb = rulebook
        self.lunar = lunar or LunarCalendar(None)

    def _week_start(self, d: date) -> date:
        ws = 6 if self.rb.get("time.week_start", "monday") == "sunday" else 0
        return d - timedelta(days=(d.weekday() - ws) % 7)

    def _cut(self, start: date, natural_end: date, t: date) -> tuple[date, bool]:
        if self.rb.get("time.current_period_to_date", True) and start <= t < natural_end:
            return t + DAY, True
        return natural_end, False

    def tet_window(self, lunar_year: int) -> tuple[date, date]:
        a = int(self.rb.get("time.tet_start_chap_day", 23))
        b = int(self.rb.get("time.tet_last_day", 10))
        s = self.lunar.to_solar(lunar_year - 1, 12, a)
        e = self.lunar.to_solar(lunar_year, 1, b) + DAY
        return s, e

    # ---------------------------------------------------------------- main --
    def resolve(self, question: str, as_of) -> TimeContext:
        t = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of)[:10])
        q = qnorm(question)
        ctx = TimeContext(as_of=t)
        R = ctx.ranges

        def add(label, s, e, grain="day", role="primary", to_date=False):
            if s is None or e is None or e <= s:
                return
            if not any(r.start == s and r.end == e and r.role == role for r in R):
                R.append(TimeRange(label, s, e, role, grain, to_date))

        m0 = t.replace(day=1)
        q0 = date(t.year, 3 * ((t.month - 1) // 3) + 1, 1)
        y0 = date(t.year, 1, 1)
        wk = self._week_start(t)

        # ---- shift / weekday / snapshot -------------------------------------
        if re.search(r"\bca sang\b", q):
            ctx.shift = "sang"
        elif re.search(r"\bca chieu\b", q):
            ctx.shift = "chieu"
        if not re.search(r"\b(thu may|thu nao)\b", q):
            for w, i in WEEKDAY_W.items():
                if w == "cn":
                    continue
                if re.search(rf"\b(cac|moi|nhung|vao|ngay|toi|sang|chieu) (ngay )?{w}\b", q) or \
                        (w == "chu nhat" and re.search(r"\bchu nhat\b", q)):
                    ctx.weekday = i
                    break

        # ---- explicit dates ---------------------------------------------------
        full = []
        for m in re.finditer(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{2,4})(?![\d/])", q):
            y_ = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
            full.append((m.start(), m.end(), y_, int(m.group(2)), int(m.group(1))))
        for m in re.finditer(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])", q):
            if re.search(r"(thang|quy)\s*$", q[:m.start()]):
                continue
            mo, d_ = int(m.group(2)), int(m.group(1))
            if 1 <= mo <= 12 and 1 <= d_ <= 31:
                yy = t.year if date(t.year, mo, min(d_, 28)) <= t else t.year - 1
                full.append((m.start(), m.end(), yy, mo, d_))
        dates = []
        for st_, en_, y_, mo, d_ in sorted(full):
            try:
                dates.append((st_, en_, date(y_, mo, d_)))
            except ValueError:
                pass
        consumed = []
        kt = re.search(r"(\d+) ngay ke tu", q)
        if kt and dates:
            consumed = [(st_, en_) for st_, en_, _ in dates]       # "N ngày kể từ D" handled below
        elif len(dates) >= 2 and dates[0][2] != dates[1][2] and re.search(r"\b(tu|den|toi)\b|-|giai doan|trong khoang", q):
            a, b = dates[0][2], dates[1][2]
            add(f"{min(a, b)}..{max(a, b)}", min(a, b), max(a, b) + DAY, "range")
            consumed = [(dates[0][0], dates[1][1])]
        elif dates and re.search(r"\btu (ngay )?\d{1,2}/\d{1,2}(/\d{2,4})? (nam nay )?(den|toi) (het )?(hom nay|nay|bay gio)", q):
            d0 = dates[0][2]
            add(f"{d0}..hôm nay", d0, t + DAY, "range")
            consumed = [(dates[0][0], dates[0][1])]
            q = re.sub(r"(den|toi) (het )?(hom nay|nay|bay gio)", " ", q)
        elif dates:
            st_, en_, d0 = dates[0]
            pre, post = q[max(0, st_ - 25):st_], q[en_:en_ + 30]
            if re.search(r"(den het|tinh den|tinh toi|cuoi) (ngay )?$", pre) or re.search(r"^\s*(tro ve truoc)", post):
                ctx.snapshot = d0
            elif re.search(r"\bke tu\b", pre) or re.search(r"\d+ ngay ke tu", q):
                pass  # "N ngày kể từ D" handled below
            else:
                add(f"ngày {d0}", d0, d0 + DAY, "day")
            consumed = [(st_, en_)]
        for st_, en_ in consumed:
            q = q[:st_] + " " * (en_ - st_) + q[en_:]

        # ---- N days from D / before D (price-change questions) -----------------
        m = kt
        if m and dates:
            n, d0 = int(m.group(1)), dates[0][2]
            add(f"{n} ngày kể từ {d0}", d0, d0 + timedelta(days=n), "range")
            add(f"{n} ngày liền trước {d0}", d0 - timedelta(days=n), d0, "range", role="comparison")

        # ---- snapshot words ------------------------------------------------------
        if re.search(r"(cuoi ngay hom nay|\bhet hom nay|tinh den hom nay|hien tai|bay gio|hien nay|con (ton|no|lai)|"
                     r"con bao nhieu|trong kho con|ton kho)", q) \
                and ctx.snapshot is None:
            ctx.snapshot = t
        if re.search(r"cuoi ngay hom qua|den het hom qua", q):
            ctx.snapshot = t - DAY

        # ---- relative days -----------------------------------------------------
        if re.search(r"\bhom nay\b", q) and not re.search(r"(den het|tinh den|cuoi ngay|toi het|den|toi) hom nay", q):
            add("hôm nay", t, t + DAY)
        if re.search(r"\bhom qua\b", q) and not re.search(r"(den het|cuoi ngay) hom qua", q):
            add("hôm qua", t - DAY, t)
        if re.search(r"\bhom kia\b", q):
            add("hôm kia", t - 2 * DAY, t - DAY)

        # ---- rolling windows ---------------------------------------------------
        for m in re.finditer(r"\b(\d+|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\s+(ngay|tuan|thang)\s+"
                             r"(qua|vua qua|gan day|gan nhat|vua roi|cuoi|tro lai day|lien tiep)\b", q):
            n, unit = _n(m.group(1)), m.group(2)
            if not n:
                continue
            end = t + DAY if self.rb.get("time.rolling_includes_today", True) else t
            start = end - (timedelta(days=n) if unit == "ngay" else timedelta(days=7 * n) if unit == "tuan"
                           else relativedelta(months=n))
            add(f"{n} {unit} gần nhất", start, end, "range")

        # ---- forward horizon ---------------------------------------------------
        m = re.search(r"(\d+)\s*(ngay|tuan|thang)\s*(toi|nua|sap toi|ke tiep|tiep theo)", q)
        if m:
            ctx.horizon_days = int(m.group(1)) * {"ngay": 1, "tuan": 7, "thang": 30}[m.group(2)]
        elif re.search(r"sap het han|gan het han", q):
            ctx.horizon_days = int(self.rb.get("time.expiring_horizon_days", 30))

        # ---- "this X last year" -------------------------------------------------
        m = re.search(r"\b(thang|tuan|quy) nay (cua )?(nam ngoai|nam truoc|nam roi)\b", q)
        if m:
            unit = m.group(1)
            s0 = {"thang": m0, "tuan": wk, "quy": q0}[unit]
            e0 = {"thang": m0 + relativedelta(months=1), "tuan": wk + 7 * DAY, "quy": q0 + relativedelta(months=3)}[unit]
            add(f"{unit} này năm ngoái", s0 - relativedelta(years=1), e0 - relativedelta(years=1),
                {"thang": "month", "tuan": "week", "quy": "quarter"}[unit])
            q = q.replace(m.group(0), " ")

        # ---- weeks ----------------------------------------------------------------
        if re.search(r"\btuan (nay|hien tai)\b", q):
            e, td = self._cut(wk, wk + 7 * DAY, t)
            add("tuần này", wk, e, "week", to_date=td)
        if re.search(r"\btuan (roi|truoc|qua|vua roi|vua qua|truoc do)\b", q) and not re.search(r"\d+\s+tuan (qua|truoc)", q):
            role = "comparison" if (re.search(r"so (voi|vs) tuan", q) and any(r.grain == "week" for r in R)) else "primary"
            add("tuần trước", wk - 7 * DAY, wk, "week", role=role)

        # ---- months -----------------------------------------------------------------
        if re.search(r"\bthang (nay|hien tai)\b", q) or re.search(r"\btu dau thang\b", q):
            e, td = self._cut(m0, m0 + relativedelta(months=1), t)
            add("tháng này", m0, e, "month", to_date=td)
        for m in re.finditer(r"\bthang\s*(\d{1,2})\s*(?:/|nam|-)\s*(\d{4})\b|\bthang\s*(\d{1,2})\b(?!\s*/)|(?<![\d/])(\d{1,2})/(\d{4})\b", q):
            mo = int(m.group(1) or m.group(3) or m.group(4))
            yr = m.group(2) or m.group(5)
            if not 1 <= mo <= 12 or re.search(r"quy\s*$", q[:m.start()]):
                continue
            tail = q[m.end():m.end() + 14]
            if re.match(r"\s*(am lich|al\b)", tail) or re.match(r"\s*(ngay|tuan|thang) (qua|gan)", tail):
                continue
            if yr is None:
                if re.match(r"\s*(cua )?(nam ngoai|nam truoc|nam roi)", tail):
                    yr = t.year - 1
                elif re.match(r"\s*(cua )?nam nay", tail):
                    yr = t.year
                else:
                    yr = t.year if date(t.year, mo, 1) <= t else t.year - 1
            s = date(int(yr), mo, 1)
            e, td = self._cut(s, s + relativedelta(months=1), t)
            add(f"tháng {mo}/{yr}", s, e, "month", to_date=td)
        if re.search(r"\bthang (roi|truoc|qua|vua roi|vua qua|truoc do|lien truoc)\b", q) and \
                not re.search(r"\d+\s+thang (qua|truoc)", q):
            q = re.sub(r"cung ky (cua )?(thang (roi|truoc|qua))", r"so voi \2", q)
            explicit_month = [r for r in R if r.grain == "month" and r.role == "primary"]
            comparing = bool(re.search(r"so (voi|vs)|thay doi|tang|giam|chenh|\bhon\b|\bkem\b", q))
            if explicit_month and comparing:
                p = explicit_month[0]
                ps = p.start - relativedelta(months=1)
                if p.to_date and self.rb.get("time.compare_same_days", True):
                    last = (p.start - DAY)
                    end_day = min(t.day, last.day)
                    add(f"tháng trước (cùng {end_day} ngày đầu)", ps, date(ps.year, ps.month, end_day) + DAY, "month",
                        role="comparison")
                else:
                    add("tháng liền trước", ps, p.start, "month", role="comparison")
            else:
                pm = m0 - relativedelta(months=1)
                add("tháng trước", pm, m0, "month")

        m = re.search(r"\btu thang (\d{1,2})(?:/(\d{4}))? (?:den|toi) (?:het )?thang (\d{1,2})(?:/(\d{4}))?", q)
        if m:
            a_m, a_y, b_m, b_y = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
            b_y = int(b_y) if b_y else (t.year if date(t.year, b_m, 1) <= t else t.year - 1)
            a_y = int(a_y) if a_y else (b_y if a_m <= b_m else b_y - 1)
            s_, e_ = date(a_y, a_m, 1), date(b_y, b_m, 1) + relativedelta(months=1)
            R[:] = [r for r in R if not (r.grain == "month" and (s_ <= r.start < e_ or re.search(r"thang (sau|truoc)", q)))]
            ee, td = self._cut(s_, e_, t)
            add(f"tháng {a_m}/{a_y}..{b_m}/{b_y}", s_, ee, "month_span", to_date=td)
            q = q[:m.start()] + " " * (m.end() - m.start()) + q[m.end():]

        # ---- quarters ---------------------------------------------------------------
        if re.search(r"\bquy (nay|hien tai)\b", q):
            e, td = self._cut(q0, q0 + relativedelta(months=3), t)
            add("quý này", q0, e, "quarter", to_date=td)
        for m in re.finditer(r"\bquy (\d|i{1,3}|iv)\b(?:\s*(?:/|nam|-)\s*(\d{4}))?", q):
            g = m.group(1)
            n = {"i": 1, "ii": 2, "iii": 3, "iv": 4}.get(g, int(g) if g.isdigit() else 0)
            if not 1 <= n <= 4:
                continue
            yr = int(m.group(2)) if m.group(2) else (t.year if date(t.year, 3 * (n - 1) + 1, 1) <= t else t.year - 1)
            s = date(yr, 3 * (n - 1) + 1, 1)
            e, td = self._cut(s, s + relativedelta(months=3), t)
            add(f"quý {n}/{yr}", s, e, "quarter", to_date=td)
        if re.search(r"\bquy (truoc|roi|qua|lien truoc|truoc do)\b", q):
            pq = [r for r in R if r.grain == "quarter" and r.role == "primary"]
            if pq and re.search(r"so (voi|vs)|thay doi|tang|giam|\bhon\b|\bkem\b|chenh", q):
                add("quý liền trước", pq[0].start - relativedelta(months=3), pq[0].start, "quarter", role="comparison")
            else:
                add("quý trước", q0 - relativedelta(months=3), q0, "quarter")

        # ---- years ------------------------------------------------------------------
        if (re.search(r"\bnam (nay|hien tai)\b", q) and not any(r.grain in ("month", "quarter") for r in R)
                and not re.search(r"\btet (nam )?nay\b", q)) or re.search(r"\btu dau nam\b", q):
            e, td = self._cut(y0, date(t.year + 1, 1, 1), t)
            add("năm nay", y0, e, "year", to_date=td)
        for m in re.finditer(r"\bnam (\d{4})\b", q):
            yr = int(m.group(1))
            if any(r.start.year == yr and r.grain in ("month", "quarter", "tet") for r in R) or re.search(rf"tet (nam )?{yr}", q):
                continue
            s = date(yr, 1, 1)
            e, td = self._cut(s, date(yr + 1, 1, 1), t)
            add(f"năm {yr}", s, e, "year", to_date=td)
        if re.search(r"\bnam (ngoai|truoc|roi|vua roi|qua)\b", q) and not re.search(r"cung ky", q) \
                and not any(r.grain in ("month", "quarter", "week") for r in R) and not re.search(r"\btet\b", q):
            add("năm ngoái", date(t.year - 1, 1, 1), y0, "year")

        # ---- Tết ----------------------------------------------------------------------
        if re.search(r"\btet\b", q) and not re.search(r"tet duong", q):
            m = re.search(r"\btet (nam |nguyen dan )?(\d{4})\b", q)
            if m:
                ly = int(m.group(2))
            else:
                ly = t.year
                s, _ = self.tet_window(ly)
                if s > t:
                    ly -= 1
                # the FIRST Tết mention decides the primary year ("Tết năm nay so với Tết năm ngoái")
                first = re.search(r"\btet\b(?: nam| nguyen dan)?(?: (nay|ngoai|truoc|roi))?", q)
                if first and first.group(1) == "nay":
                    ly = t.year
                elif first and first.group(1) in ("ngoai", "truoc", "roi"):
                    ly = t.year - 1
            s, e = self.tet_window(ly)
            if s and e:
                if e > t + DAY and s <= t:
                    e = t + DAY
                add(f"Tết {ly} (23 Chạp – mùng 10)", s, e, "tet")
                if re.search(r"truoc tet", q):
                    n = (e - s).days
                    add(f"{n} ngày ngay trước 23 Chạp", s - timedelta(days=n), s, "range", role="comparison")
                if (len(re.findall(r"\btet\b", q)) >= 2 or re.search(r"so (voi|vs) (dip )?tet", q)) \
                        and not re.search(r"truoc tet", q):
                    s2, e2 = self.tet_window(ly - 1)
                    add(f"Tết {ly - 1}", s2, e2, "tet", role="comparison")

        # ---- comparisons --------------------------------------------------------------
        prim = [r for r in R if r.role == "primary"]
        if re.search(r"cung ky|cung (thang|tuan|quy|thoi diem|thoi gian) (cua )?nam (truoc|ngoai|roi)", q) and prim \
                and not any(r.role == "comparison" for r in R):
            p = prim[0]
            add(f"cùng kỳ năm trước của {p.label}", p.start - relativedelta(years=1), p.end - relativedelta(years=1),
                p.grain, role="comparison")
        elif len(prim) >= 2 and re.search(r"\bso (voi|vs)\b|chenh|hon|kem", q) and not any(r.role == "comparison" for r in R):
            # "A so với B": B is the comparison period
            pos = {id(r): q.find(r.label.split()[0]) for r in prim}
            a, b = prim[0], prim[1]
            if re.search(r"so (voi|vs)", q):
                idx = q.find("so voi") if "so voi" in q else q.find("so vs")
                before = [r for r in prim[:2] if q.find(_kw(r)) < idx]
                after = [r for r in prim[:2] if r not in before]
                if before and after:
                    a, b = before[0], after[0]
            b.role = "comparison"
            del pos
        ctx.month_answer = bool(re.search(r"\bthang (nao|may)\b", q))
        ctx.explicit = bool(R) or ctx.snapshot is not None or ctx.horizon_days is not None
        R.sort(key=lambda r: r.role != "primary")
        return ctx


def _kw(r: TimeRange) -> str:
    l = r.label.lower()
    for k in ("tháng này", "tháng trước", "tuần này", "tuần trước", "hôm nay", "hôm qua", "năm nay", "quý"):
        if l.startswith(k):
            return {"tháng này": "thang nay", "tháng trước": "thang truoc", "tuần này": "tuan nay",
                    "tuần trước": "tuan truoc", "hôm nay": "hom nay", "hôm qua": "hom qua", "năm nay": "nam nay",
                    "quý": "quy"}[k]
    m = re.match(r"tháng (\d+)/(\d+)", l)
    if m:
        return f"thang {m.group(1)}"
    return l.split()[0]
