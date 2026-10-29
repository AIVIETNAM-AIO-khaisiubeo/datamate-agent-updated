"""Vietnamese solar <-> lunar calendar.

Primary source: the organisers' shared solar-lunar calendar file (loaded by
``LunarCalendar``). Fallback: Ho Ngoc Duc's astronomical algorithm (UTC+7),
the de-facto reference implementation used by Vietnamese calendar software
("Am lich Viet Nam", Ho Ngoc Duc, 2004; based on Jean Meeus, *Astronomical
Algorithms*, 1998). The file always wins when it covers the date.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import date

PI = math.pi
TZ = 7.0


def _int(x):
    return math.floor(x)


def jd_from_date(dd, mm, yy):
    a = (14 - mm) // 12
    y = yy + 4800 - a
    m = mm + 12 * a - 3
    jd = dd + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    if jd < 2299161:
        jd = dd + (153 * m + 2) // 5 + 365 * y + y // 4 - 32083
    return jd


def jd_to_date(jd):
    if jd > 2299160:
        a = jd + 32044
        b = (4 * a + 3) // 146097
        c = a - (b * 146097) // 4
    else:
        b, c = 0, jd + 32082
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = b * 100 + d - 4800 + m // 10
    return day, month, year


def _new_moon(k):
    T = k / 1236.85
    T2, T3 = T * T, T * T * T
    dr = PI / 180
    jd1 = 2415020.75933 + 29.53058868 * k + 0.0001178 * T2 - 0.000000155 * T3
    jd1 += 0.00033 * math.sin((166.56 + 132.87 * T - 0.009173 * T2) * dr)
    M = 359.2242 + 29.10535608 * k - 0.0000333 * T2 - 0.00000347 * T3
    Mpr = 306.0253 + 385.81691806 * k + 0.0107306 * T2 + 0.00001236 * T3
    F = 21.2964 + 390.67050646 * k - 0.0016528 * T2 - 0.00000239 * T3
    C1 = (0.1734 - 0.000393 * T) * math.sin(M * dr) + 0.0021 * math.sin(2 * dr * M)
    C1 = C1 - 0.4068 * math.sin(Mpr * dr) + 0.0161 * math.sin(dr * 2 * Mpr)
    C1 = C1 - 0.0004 * math.sin(dr * 3 * Mpr)
    C1 = C1 + 0.0104 * math.sin(dr * 2 * F) - 0.0051 * math.sin(dr * (M + Mpr))
    C1 = C1 - 0.0074 * math.sin(dr * (M - Mpr)) + 0.0004 * math.sin(dr * (2 * F + M))
    C1 = C1 - 0.0004 * math.sin(dr * (2 * F - M)) - 0.0006 * math.sin(dr * (2 * F + Mpr))
    C1 = C1 + 0.0010 * math.sin(dr * (2 * F - Mpr)) + 0.0005 * math.sin(dr * (2 * Mpr + M))
    if T < -11:
        deltat = 0.001 + 0.000839 * T + 0.0002261 * T2 - 0.00000845 * T3 - 0.000000081 * T * T3
    else:
        deltat = -0.000278 + 0.000265 * T + 0.000262 * T2
    return jd1 + C1 - deltat


def _sun_longitude(jdn):
    T = (jdn - 2451545.0) / 36525
    T2 = T * T
    dr = PI / 180
    M = 357.52910 + 35999.05030 * T - 0.0001559 * T2 - 0.00000048 * T * T2
    L0 = 280.46645 + 36000.76983 * T + 0.0003032 * T2
    DL = (1.914600 - 0.004817 * T - 0.000014 * T2) * math.sin(dr * M)
    DL += (0.019993 - 0.000101 * T) * math.sin(dr * 2 * M) + 0.000290 * math.sin(dr * 3 * M)
    L = (L0 + DL) * dr
    return L - PI * 2 * _int(L / (PI * 2))


def _new_moon_day(k, tz=TZ):
    return _int(_new_moon(k) + 0.5 + tz / 24)


def _sun_long_sector(day_number, tz=TZ):
    return _int(_sun_longitude(day_number - 0.5 - tz / 24) / PI * 6)


def _lunar_month11(yy, tz=TZ):
    off = jd_from_date(31, 12, yy) - 2415021
    k = _int(off / 29.530588853)
    nm = _new_moon_day(k, tz)
    if _sun_long_sector(nm, tz) >= 9:
        nm = _new_moon_day(k - 1, tz)
    return nm


def _leap_month_offset(a11, tz=TZ):
    k = _int((a11 - 2415021.076998695) / 29.530588853 + 0.5)
    i = 1
    arc = _sun_long_sector(_new_moon_day(k + i, tz), tz)
    while True:
        last = arc
        i += 1
        arc = _sun_long_sector(_new_moon_day(k + i, tz), tz)
        if not (arc != last and i < 14):
            break
    return i - 1


def solar_to_lunar_algo(d: date, tz=TZ):
    """-> (lunar_day, lunar_month, lunar_year, is_leap)"""
    day_number = jd_from_date(d.day, d.month, d.year)
    k = _int((day_number - 2415021.076998695) / 29.530588853)
    month_start = _new_moon_day(k + 1, tz)
    if month_start > day_number:
        month_start = _new_moon_day(k, tz)
    a11 = _lunar_month11(d.year, tz)
    b11 = a11
    if a11 >= month_start:
        lunar_year = d.year
        a11 = _lunar_month11(d.year - 1, tz)
    else:
        lunar_year = d.year + 1
        b11 = _lunar_month11(d.year + 1, tz)
    lunar_day = day_number - month_start + 1
    diff = _int((month_start - a11) / 29)
    leap = 0
    lunar_month = diff + 11
    if b11 - a11 > 365:
        leap_diff = _leap_month_offset(a11, tz)
        if diff >= leap_diff:
            lunar_month = diff + 10
            if diff == leap_diff:
                leap = 1
    if lunar_month > 12:
        lunar_month -= 12
    if lunar_month >= 11 and diff < 4:
        lunar_year -= 1
    return lunar_day, lunar_month, lunar_year, bool(leap)


def lunar_to_solar_algo(ld, lm, ly, leap=False, tz=TZ):
    if lm < 11:
        a11, b11 = _lunar_month11(ly - 1, tz), _lunar_month11(ly, tz)
    else:
        a11, b11 = _lunar_month11(ly, tz), _lunar_month11(ly + 1, tz)
    k = _int(0.5 + (a11 - 2415021.076998695) / 29.530588853)
    off = lm - 11
    if off < 0:
        off += 12
    if b11 - a11 > 365:
        leap_off = _leap_month_offset(a11, tz)
        leap_month = leap_off - 2
        if leap_month < 0:
            leap_month += 12
        if leap and lm != leap_month:
            return None
        if leap or off >= leap_off:
            off += 1
    month_start = _new_moon_day(k + off, tz)
    d, m, y = jd_to_date(month_start + ld - 1)
    return date(y, m, d)


class LunarCalendar:
    """Lookup table from the organisers' file with algorithmic fallback.

    The file format is not yet known, so the loader is permissive: CSV/TSV/JSON
    rows; a solar date column (name contains 'solar'/'duong'/'date'/'ngay') and
    either lunar_day/lunar_month/lunar_year(/leap) columns or a single lunar
    date string column ('dd/mm/yyyy' or 'yyyy-mm-dd'). Holiday/note columns are
    kept in ``self.notes`` for the time resolver.
    """

    def __init__(self, path: str | None = None):
        self.s2l: dict[date, tuple[int, int, int, bool]] = {}
        self.l2s: dict[tuple[int, int, int, bool], date] = {}
        self.notes: dict[date, str] = {}
        self.source = "algorithm"
        if path and os.path.exists(path):
            try:
                self._load(path)
                if self.s2l:
                    self.source = path
            except Exception:  # never crash the run because of the calendar file
                self.s2l, self.l2s = {}, {}

    # ---- loading -------------------------------------------------------
    def _rows(self, path):
        if path.endswith(".json"):
            data = json.load(open(path, encoding="utf-8"))
            return data if isinstance(data, list) else data.get("rows", [])
        with open(path, encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return list(csv.DictReader(f, dialect=dialect))

    def _load(self, path):
        from .textnorm import fold
        rows = self._rows(path)
        if not rows:
            return
        keys = {fold(k): k for k in rows[0].keys()}

        def find(*cands, exclude=()):
            for fk, k in keys.items():
                if any(c in fk for c in cands) and not any(e in fk for e in exclude):
                    return k
            return None

        solar = find("solar", "duong", "gregorian") or find("date", "ngay", exclude=("lunar", "am"))
        lday, lmon = find("lunar day", "lunar_day", "ngay am"), find("lunar month", "lunar_month", "thang am")
        lyear, leap = find("lunar year", "lunar_year", "nam am"), find("leap", "nhuan")
        lstr = find("lunar", "am lich", exclude=("day", "month", "year", "leap"))
        note = find("holiday", "le", "note", "ghi chu", "event", "su kien")
        for r in rows:
            sd = _parse_date(r.get(solar)) if solar else None
            if sd is None:
                continue
            if lday and lmon and lyear:
                ld, lm, ly = int(float(r[lday])), int(float(r[lmon])), int(float(r[lyear]))
            elif lstr and r.get(lstr):
                p = _parse_date(r[lstr])
                if p is None:
                    continue
                ld, lm, ly = p.day, p.month, p.year
            else:
                continue
            lp = str(r.get(leap, "0")).strip().lower() in {"1", "true", "yes", "co", "có", "x"} if leap else False
            self.s2l[sd] = (ld, lm, ly, lp)
            self.l2s.setdefault((ly, lm, ld, lp), sd)
            if note and r.get(note):
                self.notes[sd] = str(r[note])

    # ---- api -----------------------------------------------------------
    def to_lunar(self, d: date):
        return self.s2l.get(d) or solar_to_lunar_algo(d)

    def to_solar(self, ly: int, lm: int, ld: int, leap: bool = False) -> date | None:
        return self.l2s.get((ly, lm, ld, leap)) or lunar_to_solar_algo(ld, lm, ly, leap)

    def tet(self, lunar_year: int) -> date:
        return self.to_solar(lunar_year, 1, 1)

    def find_note(self, keyword_folded: str, year: int | None = None):
        from .textnorm import fold
        hits = [d for d, n in self.notes.items() if keyword_folded in fold(n) and (year is None or d.year == year)]
        return sorted(hits)


def _parse_date(s):
    if s is None:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None
