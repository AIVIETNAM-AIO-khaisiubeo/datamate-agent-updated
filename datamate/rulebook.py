"""Rulebook of the split being answered.

The challenge ships one rulebook per split and warns that it may differ between
splits, so nothing here hard-codes the training rulebook:

1. `text`      -> injected verbatim into every LLM prompt (system prefix, shared
                  by all questions of the split -> served from the prefix cache).
2. `settings`  -> structured conventions used by deterministic code, parsed from
                  the rulebook TEXT (regexes over the folded text); values not
                  found fall back to configs/rulebook_overrides.yaml.
3. `clauses`   -> "rule fingerprints". Each deterministic template declares the
                  rulebook clauses it implements; if a clause is not found in
                  the split's rulebook (rule changed / removed), the template
                  is disabled and the question goes to the LLM, which reads the
                  actual text. This is the safety valve against rulebook drift.
"""
from __future__ import annotations

import json
import os
import re

import yaml

from .config import ROOT, deep_merge
from .textnorm import fold

# clause name -> regex over the folded rulebook text
CLAUSES = {
    "valid_orders": r"khong huy.{0,15}khong tra hang",
    "partial_return_whole_order": r"(ke ca|ca) chi tra mot phan.{0,60}bo ca don|tra mot phan.{0,40}bo ca don",
    "revenue_lines": r"doanh thu mot don.{0,40}(tong )?thanh tien cac dong",
    "revenue_no_order_discount": r"khong.{0,5}tru giam gia tren ca don",
    "period_by_order_date": r"don thuoc ky nao.{0,20}theo ngay dat",
    "orders_valid_only": r"so don.{0,30}chi tinh don khong huy.{0,10}khong tra",
    "cancel_rate_all_orders": r"ti le don huy.{0,80}moi don dat trong ky",
    "pct_change_vs_previous": r"phan tram thay doi.{0,60}so voi.{0,10}ky truoc",
    "diff_current_minus_previous": r"ky hoi.{0,6}ky so",
    "week_monday": r"tuan.{0,30}bat dau.{0,10}thu hai",
    "last_week_full": r"tuan truoc.{0,40}ca.{0,10}tuan lien truoc",
    "this_month_to_date": r"thang nay.{0,30}tu ngay 1 den.{0,10}het hom nay",
    "same_days_previous_month": r"cung so ngay.{0,20}thang truoc",
    "month_without_year": r"thang m.{0,20}khong ghi nam.{0,60}nam nay.{0,60}nam truoc",
    "rolling_includes_today": r"n ngay gan nhat.{0,80}ke ca hom nay",
    "from_to_inclusive": r"tu x den y.{0,40}gom ca x va y",
    "tet_window": r"23 thang chap.{0,160}mung 10",
    "bestseller_by_base_qty": r"ban chay.{0,40}so luong quy ve don vi co ban",
    "service_fee_not_item": r"phi phuc vu.{0,40}khong phai mat hang",
    "tie_by_code": r"bang nhau.{0,120}ma tang dan",
    "aov_round_half_up": r"gia tri trung binh mot don.{0,40}lam tron ve dong",
    "customer_by_code": r"moi ma khach.{0,10}la mot khach",
    "walkin_not_customer": r"khach le.{0,40}khong phai khach",
    "outside_range_unanswerable": r"hoan toan.{0,10}ngoai khoang ngay co du lieu",
    "partial_period_ok": r"chi co mot phan.{0,60}tinh tren phan co du lieu",
    "money_round_dong": r"tien ra so le lam tron ve dong",
    "percent_2dp": r"phan tram.{0,80}(2|hai) chu so thap phan",
    "shift_hours": r"ca sang.{0,40}06 00.{0,10}13 59",
    "staff_by_revenue": r"nhan vien ban gioi nhat.{0,30}doanh thu",
    "customer_most_money_by_revenue": r"khach mua nhieu tien nhat.{0,30}doanh thu",
    "payment_by_order_field": r"theo hinh thuc ghi tren don",
    "group_exact_name": r"nhom hang.{0,40}dung ten nhom",
    "unsold_definition": r"(ban duoc trong ky|khong ban duoc trong ky|\be\b).{0,80}da tung ban truoc ky",
    "list_order_by_measure_then_code": r"liet ke.{0,60}so do giam dan.{0,60}ma tang dan",
    "import_total_by_receipt_date": r"tien nhap hang trong ky.{0,80}ngay phieu.{0,40}bo phieu huy",
    "import_avg_price_qty_weighted": r"gia nhap binh quan.{0,80}theo so luong nhap",
    "top_supplier_by_amount": r"nha cung cap nhap nhieu nhat.{0,40}tien nhap cao nhat",
    "main_supplier_by_receipts": r"nha cung cap chinh.{0,60}nhieu phieu nhap nhat",
}


class RuleBook:
    def __init__(self, text: str = "", settings: dict | None = None):
        self.text = text or ""
        self.s = settings or {}
        f = fold(self.text)
        self.clauses = {k: bool(re.search(rx, f)) for k, rx in CLAUSES.items()} if self.text else {}

    def has(self, *clauses) -> bool:
        """True when every clause is present (or when no rulebook text exists at all,
        in which case the overrides file defines the behaviour)."""
        if not self.text:
            return True
        return all(self.clauses.get(c, False) for c in clauses)

    def get(self, dotted: str, default=None):
        d = self.s
        for k in dotted.split("."):
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    @classmethod
    def load(cls, path: str | None, overrides_path: str | None = None) -> "RuleBook":
        overrides_path = overrides_path or os.path.join(ROOT, "configs", "rulebook_overrides.yaml")
        settings = yaml.safe_load(open(overrides_path, encoding="utf-8")) if os.path.exists(overrides_path) else {}
        text = ""
        if path and os.path.exists(path):
            raw = open(path, encoding="utf-8").read()
            if path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(raw) or {}
                text = raw
                if isinstance(data, dict):
                    settings = deep_merge(settings, data.get("datamate_settings", {}))
            elif path.endswith(".json"):
                text = json.dumps(json.loads(raw), ensure_ascii=False, indent=1)
            else:
                text = raw
        settings = deep_merge(settings, infer_from_text(text))
        return cls(text, settings)

    # ---- valid orders (used only by the legacy synthetic generator) ---------
    def is_valid_status(self, status) -> bool:
        if status is None or (isinstance(status, float) and status != status):
            return True
        f = f" {fold(status)} "
        bad = [fold(k) for k in self.get("valid_orders.invalid_status_keywords", []) or []]
        return not any(f" {k} " in f or f.strip().startswith(k) for k in bad)

    def summary(self) -> str:
        on = [k for k, v in self.clauses.items() if v]
        off = [k for k, v in self.clauses.items() if not v]
        return f"{len(on)}/{len(self.clauses)} known clauses found" + (f"; missing: {', '.join(off)}" if off else "")


def infer_from_text(text: str) -> dict:
    """Extract conventions that deterministic code needs. Only explicit phrasings are
    recognised; anything else keeps the overrides-file value (the LLM reads the text)."""
    out: dict = {}
    if not text:
        return out
    t = fold(text)

    def put(path, v):
        d = out
        ks = path.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = v

    m = re.search(r"ban chay[^|\n]{0,40}?theo\s*(so luong|so tien|doanh thu)", t)
    if m:
        put("ranking.bestseller_metric", "quantity" if "so luong" in m.group(1) else "revenue")
    m = re.search(r"tuan[^|\n]{0,30}bat dau[^|\n]{0,10}(thu hai|chu nhat)", t)
    if m:
        put("time.week_start", "monday" if "hai" in m.group(1) else "sunday")
    if re.search(r"thang nay[^|\n]{0,40}het hom nay", t):
        put("time.current_period_to_date", True)
    if re.search(r"cung so ngay[^|\n]{0,20}thang truoc", t):
        put("time.compare_same_days", True)
    if re.search(r"n ngay gan nhat[^|\n]{0,80}ke ca hom nay", t):
        put("time.rolling_includes_today", True)
    m = re.search(r"(\d{1,2}) thang chap[^|\n]{0,160}?mung (\d{1,2})", t)
    if m:
        put("time.tet_start_chap_day", int(m.group(1)))
        put("time.tet_last_day", int(m.group(2)))
    m = re.search(r"ca sang\D{0,30}?(\d{1,2}) (\d{2}) (\d{1,2}) (\d{2})\D{0,30}?ca chieu\D{0,30}?(\d{1,2}) (\d{2}) (\d{1,2}) (\d{2})", t)
    if m:
        a = [int(x) for x in m.groups()]
        put("time.shift_morning", [a[0] * 60 + a[1], a[2] * 60 + a[3]])
        put("time.shift_afternoon", [a[4] * 60 + a[5], a[6] * 60 + a[7]])
    m = re.search(r"het han trong n ngay toi[^|\n]{0,80}tu d den het ngay d \+ n", t)
    if m:
        put("time.expiry_window_inclusive", True)
    return out
