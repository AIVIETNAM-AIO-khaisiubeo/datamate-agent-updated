"""Trace Inversion for abstention (Gourabathina et al., ACL 2026 [R39]), adapted to data questions.

Query-misalignment view of the paper: a model that should abstain is usually answering a
*different* question q* than the user's q. Trace Inversion (i) generates a reasoning trace,
(ii) reconstructs, from the trace ONLY, the question it answers, (iii) compares q* with q and
abstains when they diverge. Distance = majority vote of three modules: sentence-embedding
similarity (SE), an LLM asked whether both prompts convey the same framing, intent and context
(TrInv-LLM), and a groundedness detector (GROUND, Granite-Guardian-3.3-8B).

How it maps onto DataMate
* The trace is the EXECUTED program of the winning answer cluster (plus a preview of its
  result): it is exactly the interpretation that produced the answer, so a mis-read period,
  a missing filter or the wrong measure shows up in q*. Comments are stripped so the
  question text cannot leak into the trace.
* SE: all-MiniLM-L6-v2 in the paper is English-only. We use a character 3-gram cosine on the
  normalised Vietnamese text by default (no extra parameters), or any local multilingual
  sentence-embedding model if `inversion.se_model` is set (counted in the 15B budget).
* TrInv-LLM: the paper's prompt verbatim, answered by the same Qwen3 model (thinking off).
* GROUND needs an 8B guardian model, which does not fit next to Qwen3-14B under the 15B cap
  (competition rule 1). It is replaced by STRUCT, a deterministic comparison of the
  reconstructed spec (period, comparison period, filters, answer type) with what our analyzer
  resolved from q. STRUCT is the module that can be checked exactly in this domain.
* Votes are combined as in the authors' ensemble runner (abstain if at least half of the
  modules flag misalignment), but the result is passed to the calibrator as features rather
  than used as a hard abstention: under PA the decision stays Chow's rule on P(correct).

Templates have no free-text trace, but they do have an explicit plan (intent, period, filters
actually applied). `template_inversion` renders that plan as a question and runs the same SE
comparison plus a grounding check of every applied filter value against the question text.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from datetime import date, timedelta

from .textnorm import fold, qnorm

_DATE = r"(\d{4}-\d{2}-\d{2})"


# --------------------------------------------------------------- similarity --
def _grams(s: str, n: int = 3) -> Counter:
    s = f"  {s}  "
    return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


_TIME_W = set("""ngay hom thang nam tuan quy tu den het nay qua truoc sau roi kia trong khoang thoi gian giai doan
ky cung lien gan nhat dau cuoi tet le sang chieu ca vao luc""".split())
_FILLER_W = set("""cho hoi xem giup gium check coi minh toi em anh chi oi a a nhe nhi vay z ha la bao nhieu may nhieu
duoc cua cac nhung the nao tinh tong cong bang khong co""".split())


def content_text(s: str) -> str:
    """Question text without time expressions and chat filler: time is compared exactly by STRUCT, so SE
    looks at the measure and the filters ("doanh thu tháng trước" vs "tổng doanh thu 1/2 - 28/2")."""
    toks = [t for t in qnorm(s or "").split() if not re.search(r"\d", t) and t not in _TIME_W and t not in _FILLER_W]
    return " ".join(toks)


def lexical_similarity(a: str, b: str) -> float:
    """Character 3-gram cosine of the normalised (accent-folded, Telex-repaired) texts."""
    ga, gb = _grams(qnorm(a or "")), _grams(qnorm(b or ""))
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb.get(k, 0) for k, v in ga.items())
    return dot / (math.sqrt(sum(v * v for v in ga.values())) * math.sqrt(sum(v * v for v in gb.values())))


class SentenceEncoder:
    """Optional local sentence-embedding model (sentence-transformers), loaded lazily, offline."""

    def __init__(self, path: str | None):
        self.path, self._m, self._lock = path, None, threading.Lock()

    def available(self) -> bool:
        return bool(self.path)

    def similarity(self, a: str, b: str) -> float:
        if not self.path:
            ca, cb = content_text(a), content_text(b)
            return lexical_similarity(ca, cb) if ca and cb else lexical_similarity(a, b)
        with self._lock:
            if self._m is None:
                from sentence_transformers import SentenceTransformer
                self._m = SentenceTransformer(self.path, device="cpu")
            e = self._m.encode([a, b], normalize_embeddings=True)
        return float((e[0] * e[1]).sum())


# --------------------------------------------------------- reconstruction --
def strip_comments(code: str) -> str:
    code = re.sub(r"--[^\n]*", "", code or "")
    code = re.sub(r"(?m)^\s*#.*$", "", code)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    return code.strip()


def parse_reconstruction(text: str) -> dict:
    t = text or ""

    def rng(label):
        m = re.search(rf"{label}\s*:\s*{_DATE}\s*\.\.\s*{_DATE}", t, re.I)
        if not m:
            return None
        try:
            return date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        except ValueError:
            return None

    def any_dates():
        """Fallback when PERIOD is not filled in: ISO or d/m/yyyy dates anywhere in the reconstruction."""
        out = []
        for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})|\b(\d{1,2})/(\d{1,2})/(\d{4})\b", t):
            try:
                out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m.group(1)
                           else date(int(m.group(6)), int(m.group(5)), int(m.group(4))))
            except ValueError:
                continue
        return (out[0], out[1]) if len(out) >= 2 else None

    q = re.search(r"QUESTION\s*:\s*(.+)", t, re.I)
    fl = re.search(r"FILTERS\s*:\s*(.+)", t, re.I)
    rs = re.search(r"RESULT\s*:\s*([a-z]+)", t, re.I)
    filters = []
    if fl and fold(fl.group(1)) not in ("none", "khong", ""):
        filters = [x.strip() for x in fl.group(1).split(";") if x.strip()]
    question = (q.group(1).strip() if q else t.strip().splitlines()[0] if t.strip() else "")
    period = rng("PERIOD")
    if period is None:            # small models often copy the date bounds into FILTERS instead
        period = any_dates()
    return {"question": question.strip().strip('"'), "period": period, "compare": rng("COMPARE"), "filters": filters,
            "result": rs.group(1).lower() if rs else None}


def struct_compare(rec: dict, spec, tctx, reqs) -> dict:
    """Deterministic STRUCT module: does q* resolve to the same period(s), filters and answer type as q?"""
    comps = {}
    p, c = tctx.primary, tctx.comparison
    same = lambda got, r: got is not None and got[0] == r.start and got[1] in (r.end - timedelta(days=1), r.end)
    if p is not None:
        comps["period"] = same(rec["period"], p)
    if c is not None:
        comps["compare"] = same(rec["compare"], c)
    hay = " " + fold(" ; ".join(rec["filters"]) + " " + rec["question"]) + " "
    for r in reqs:
        if r.kind == "filter" and r.tokens:
            comps[f"filter:{r.rid}"] = any(t in hay or t.strip() in hay for t in r.tokens if t.strip())
    if spec.intent is not None and rec["result"]:
        comps["type"] = rec["result"] == spec.answer_type
    score = (sum(comps.values()) / len(comps)) if comps else 1.0
    return {"score": score, "aligned": all(comps.values()), "components": comps}


class TraceInverter:
    def __init__(self, llm, cfg: dict | None = None, encoder: SentenceEncoder | None = None):
        cfg = cfg or {}
        self.llm = llm
        self.se_threshold = float(cfg.get("se_threshold", 0.35))
        self.max_tokens = int(cfg.get("max_tokens", 320))
        self.encoder = encoder or SentenceEncoder(cfg.get("se_model"))

    def invert(self, question: str, program: str, result_preview: str, spec, tctx, reqs, seed) -> dict:
        from .prompts import RECONSTRUCT, SAME_QUERY
        trace = f"```\n{strip_comments(program)}\n```\nResult preview:\n{(result_preview or '')[:400]}\n\nReconstructed query:"
        rec_txt = self.llm.chat([{"role": "user", "content": RECONSTRUCT + "\nReasoning trace:\n" + trace}],
                                n=1, temperature=0.0, thinking=False, max_tokens=self.max_tokens, seed=seed)[0]
        rec = parse_reconstruction(rec_txt)
        se = self.encoder.similarity(question, rec["question"])
        verdict = self.llm.chat([{"role": "user", "content": SAME_QUERY.format(q1=question, q2=rec["question"])}],
                                n=1, temperature=0.0, thinking=False, max_tokens=8, seed=seed + 1)[0]
        same = 1.0 if re.search(r"\bYES\b", verdict.upper()) else 0.0 if re.search(r"\bNO\b", verdict.upper()) else 0.5
        st = struct_compare(rec, spec, tctx, reqs)
        votes = [se < self.se_threshold, same < 0.5, not st["aligned"]]
        return {"ti_used": 1.0, "ti_se": float(se), "ti_llm_same": same, "ti_struct": float(st["score"]),
                "ti_misaligned": float(sum(votes) >= len(votes) / 2),
                "_trace": {"reconstructed": rec, "verdict": verdict[:20], "struct": st["components"],
                           "votes": {"se": votes[0], "llm": votes[1], "struct": votes[2]}}}


# ------------------------------------------------------ template inversion --
INTENT_VI = {
    "revenue": "doanh thu", "n_orders": "số đơn", "n_cancelled": "số đơn huỷ", "cancel_rate": "tỉ lệ đơn huỷ",
    "aov": "trung bình mỗi đơn", "pct_change": "doanh thu thay đổi bao nhiêu phần trăm so với",
    "share": "chiếm bao nhiêu phần trăm doanh thu", "shift_diff_pct": "ca chiều so với ca sáng chênh bao nhiêu phần trăm",
    "best_period": "ngày tháng nào doanh thu cao nhất", "top_items": "mặt hàng bán chạy nhất",
    "top_customers": "khách mua nhiều nhất", "top_staff": "nhân viên bán giỏi nhất", "n_customers": "bao nhiêu khách",
    "n_customers_min_orders": "bao nhiêu khách mua từ ít nhất đơn", "pct_customers_min_orders": "phần trăm khách mua từ đơn",
    "n_days_threshold": "mấy ngày doanh thu đạt", "n_orders_threshold": "bao nhiêu đơn có tiền hàng đạt",
    "n_items_min_orders": "bao nhiêu mặt hàng có mặt trong ít nhất đơn", "qty": "bán được bao nhiêu",
    "qty_size": "bán được bao nhiêu ly size", "monotonic": "doanh thu tăng liên tục", "revenue_series": "doanh thu từng tháng",
    "price_effect": "sau khi tăng giá bán được nhiều hay ít hơn bao nhiêu phần trăm",
    "unit_price_on_day": "giá bán một đơn vị ngày", "n_unsold_items": "mấy mặt hàng không bán được",
    "import_total": "tiền nhập hàng", "import_avg_price": "giá nhập bình quân", "top_supplier": "nhà cung cấp nào",
    "multi_filter": "doanh thu lần lượt",
}


def render_plan(plan: dict) -> str:
    parts = [INTENT_VI.get(plan.get("intent"), plan.get("intent") or "")]
    for k in ("period", "compare"):
        if plan.get(k):
            parts.append(str(plan[k]))
    parts += [str(v) for _, v in plan.get("filters", [])]
    return " ".join(p for p in parts if p)


def _grounded(value: str, qf: str) -> float:
    toks = [t for t in fold(value).split() if len(t) >= 2]
    if not toks:
        return 1.0
    qs = set(qf.split())
    return sum(t in qs for t in toks) / len(toks)


def template_inversion(plan: dict | None, question: str) -> dict:
    """Deterministic inversion of a template answer: rendered plan vs question, and filter grounding."""
    if not plan:
        return {"tpl_ti_se": 0.0, "tpl_grounded": 0.0}
    qf = qnorm(question)
    vals = [v for k, v in plan.get("filters", []) if k in ("item", "staff", "customer", "group", "attribute")]
    g = min([_grounded(v, qf) for v in vals], default=1.0)
    return {"tpl_ti_se": lexical_similarity(question, render_plan(plan)), "tpl_grounded": float(g)}
