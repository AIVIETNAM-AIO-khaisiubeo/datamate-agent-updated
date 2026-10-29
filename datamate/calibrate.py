"""Confidence calibration + the abstention decision.

Decision theory. With s = +1 / 0 / -1 and p = P(answer correct | x):
    E[s | answer] = p - (1 - p) = 2p - 1,   E[s | abstain] = 0
so the PA-optimal rule is Chow's reject rule (Chow, IEEE Trans. IT 1970):
answer iff p > 1/2. The same holds for emitting `unanswerable` (it is just
another answer that is right or wrong). Everything therefore hinges on a
well-calibrated p (Guo et al., ICML 2017; Platt 1999; Zadrozny & Elkan 2002).

Confidence signals (features):
  * execution-result agreement among sampled programs = semantic-entropy style
    uncertainty where "same meaning" := same executed answer (Farquhar et al.,
    Nature 2024; self-consistency, Wang et al. ICLR 2023). TrustSQL (Lee et al.
    2024) found plain voting a strong abstention baseline.
  * cross-family agreement (SQL vs pandas vs template vs second model)
  * execution errors / empty results (strongest single filter in EHRSQL-2024:
    Jo et al. 2024; Kim et al. 2024)
  * rule-based answerability evidence (PRACTIQ categories)
A logistic model (optionally isotonic-corrected) maps features -> p. Before
any labelled dev data exists, a hand-set prior is used.
"""
from __future__ import annotations

import math
import pickle

import numpy as np

FEATS = ["agree", "margin", "cross_family", "n_families", "tpl_fired", "tpl_agree", "tpl_strength",
         "err_frac", "empty_frac", "type_agree", "winner_is_ua", "ua_strong", "ua_any",
         "link_score", "link_ambiguous", "time_in_range", "log_n_cand", "llm_used", "escalated",
         "has_intent", "time_explicit", "view_line_math", "view_order_total", "view_unknown_status",
         # ReViSQL reconciliation [R40]: execution groups, share passing Decide, winner changed, requirement coverage
         "rc_groups", "rc_pass_frac", "rc_filtered", "req_cov",
         # Trace Inversion [R39]: modules SE / TrInv-LLM / STRUCT and their majority vote; template inversion
         "ti_used", "ti_se", "ti_llm_same", "ti_struct", "ti_misaligned", "tpl_ti_se", "tpl_grounded"]
INTENTS = ["revenue", "n_orders", "n_cancelled", "cancel_rate", "aov", "pct_change", "share", "shift_diff_pct",
           "best_period", "top_items", "top_customers", "top_staff", "n_customers", "n_customers_min_orders",
           "pct_customers_min_orders", "n_days_threshold", "n_orders_threshold", "n_items_min_orders", "qty",
           "qty_size", "monotonic", "revenue_series", "price_effect", "unit_price_on_day", "n_unsold_items",
           "import_total", "import_avg_price", "top_supplier", "multi_filter"]


def vec(f: dict) -> np.ndarray:
    base = [float(f.get(k, 0.0) or 0.0) for k in FEATS]
    it = f.get("intent")
    return np.array(base + [1.0 if it == i else 0.0 for i in INTENTS] + [1.0 if it is None else 0.0])


def _sig(z):
    return 1 / (1 + math.exp(-z))


class Calibrator:
    def __init__(self, offline_template_prior: float = 0.75):
        self.lr = None
        self.iso = None
        self.offline_prior = offline_template_prior
        self.extra: dict = {}            # e.g. tuned trace-inversion SE threshold

    # ---- prior used before fitting -------------------------------------
    def prior(self, f: dict) -> float:
        if not f:
            return 0.0
        if f.get("winner_is_ua"):
            base = 0.82 if f.get("ua_strong") else 0.35
            if f.get("llm_used"):
                # 'the data cannot answer this' needs positive evidence; a fired template that computed
                # a value is evidence AGAINST it
                base = _sig(-2.0 + 2.5 * f.get("agree", 0) + 1.5 * f.get("ua_strong", 0) + 0.8 * f.get("ua_any", 0)
                            + 0.8 * f.get("cross_family", 0) - 1.5 * f.get("tpl_fired", 0) * f.get("tpl_strength", 0))
            return base
        if not f.get("llm_used"):
            g = f.get("tpl_grounded", 1.0) if f.get("tpl_ti_se") else 1.0
            return self.offline_prior * f.get("tpl_strength", 0.0) * (0.4 + 0.6 * g) if f.get("tpl_agree") else 0.0
        z = (-2.2 + 3.2 * f.get("agree", 0) + 1.2 * f.get("cross_family", 0) + 1.2 * f.get("tpl_agree", 0)
             - 1.2 * f.get("ti_misaligned", 0) + 0.5 * f.get("ti_used", 0) * (f.get("ti_struct", 0) - 0.5)
             - 1.0 * (1 - f.get("req_cov", 1.0)) * float(bool(f.get("req_cov")))
             + 0.6 * f.get("type_agree", 0) - 1.5 * f.get("err_frac", 0) - 2.0 * (1 - f.get("time_in_range", 1))
             - 1.5 * f.get("ua_strong", 0) - 1.0 * f.get("link_ambiguous", 0)
             - 1.0 * f.get("tpl_fired", 0) * (1 - f.get("tpl_agree", 0)) * f.get("tpl_strength", 0))
        return _sig(z)

    # ---- learned ---------------------------------------------------------
    def fit(self, feats: list[dict], labels: list[int], C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        X, y = np.stack([vec(f) for f in feats]), np.array(labels)
        if len(set(y)) < 2:
            return self
        self.lr = LogisticRegression(C=C, max_iter=2000).fit(X, y)
        if len(y) >= 300:
            from sklearn.isotonic import IsotonicRegression
            self.iso = IsotonicRegression(out_of_bounds="clip").fit(self.lr.predict_proba(X)[:, 1], y)
        return self

    def p(self, f: dict) -> float:
        if not f:
            return 0.0
        if self.lr is None:
            return self.prior(f)
        raw = float(self.lr.predict_proba(vec(f)[None])[0, 1])
        return float(self.iso.predict([raw])[0]) if self.iso is not None else raw

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"lr": self.lr, "iso": self.iso, "offline_prior": self.offline_prior,
                         "feats": FEATS, "intents": INTENTS, "extra": self.extra}, fh)

    @classmethod
    def load(cls, path, offline_template_prior=0.75):
        c = cls(offline_template_prior)
        try:
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            if d.get("feats") == FEATS and d.get("intents") == INTENTS:
                c.lr, c.iso = d.get("lr"), d.get("iso")
                c.extra = d.get("extra") or {}
            else:   # fitted with another feature set: fall back to the prior rather than mis-read features
                import sys
                print(f"[datamate] calibrator {path} was fitted with a different feature set; using the prior",
                      file=sys.stderr)
        except Exception:
            pass
        return c


def expected_pa(ps, correct, tau):
    ps, c = np.asarray(ps), np.asarray(correct)
    return float(np.where(ps > tau, np.where(c == 1, 1, -1), 0).mean()) if len(ps) else 0.0


def best_threshold(ps, correct, grid=None, n_boot: int = 200, seed: int = 0):
    """Threshold maximising PA, averaged over bootstrap resamples (robust on small dev sets).
    Returns (tau, PA at tau on full data)."""
    grid = np.linspace(0.3, 0.9, 61) if grid is None else grid
    ps, c = np.asarray(ps, float), np.asarray(correct, int)
    if len(ps) == 0:
        return 0.5, 0.0
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(grid))
    for _ in range(n_boot):
        idx = rng.integers(0, len(ps), len(ps))
        acc += np.array([expected_pa(ps[idx], c[idx], t) for t in grid])
    tau = float(grid[int(np.argmax(acc))])
    return tau, expected_pa(ps, c, tau)
