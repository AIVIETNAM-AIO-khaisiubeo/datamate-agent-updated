"""Fit the confidence calibrator on a labelled run and report honest PA estimates.

    # 1) run the agent on a labelled split (train) and dump features
    python scripts/run.py --data path/to/train --out runs/train_sub.jsonl --dump runs/train_dump.jsonl
    # 2) fit (grouped cross-validation by BUSINESS, so the estimate reflects unseen shops)
    python scripts/fit_calibrator.py runs/train_dump.jsonl path/to/train/answers.jsonl --out artifacts/calibrator.pkl

Label of a question = 1 if the agent's candidate answer (value or unanswerable) is
correct under the official rule. Questions without a candidate are abstentions in
every setting and only enter the PA denominator.
The report compares: prior (hand-set), fitted (out-of-fold), and the PA-optimal
threshold from bootstrap; keep tau = 0.5 (Chow) unless the bootstrap optimum is
stable across folds.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.calibrate import Calibrator, best_threshold, expected_pa  # noqa: E402
from datamate.scorer import is_correct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("gold")
ap.add_argument("--out", default="artifacts/calibrator.pkl")
ap.add_argument("--C", type=float, default=0.5)
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()

golds = {d["id"]: d for d in map(json.loads, open(a.gold, encoding="utf-8"))}
rows = [json.loads(l) for l in open(a.dump, encoding="utf-8")]
X, y, grp, ids = [], [], [], []
for r in rows:
    pred = r.get("pred") or {}
    if r["id"] in golds and pred.get("type") not in (None, "abstain") and r.get("feats"):
        X.append(r["feats"])
        y.append(int(is_correct(pred, golds[r["id"]])))
        grp.append(r.get("business_id", "?"))
        ids.append(r["id"])
N = len([r for r in rows if r["id"] in golds])
y = np.array(y)
print(f"{len(y)} questions with a candidate out of {N}; candidate accuracy {y.mean() if len(y) else 0:.3f}")
if len(set(y.tolist())) < 2:
    sys.exit("need both correct and wrong candidates to fit")

# grouped K-fold by business: estimate how the calibrator transfers to unseen shops
biz = sorted(set(grp))
rng = np.random.default_rng(0)
rng.shuffle(biz)
fold_of = {b: i % a.folds for i, b in enumerate(biz)}
oof = np.zeros(len(y))
for f in range(a.folds):
    te = np.array([fold_of[g] == f for g in grp])
    if te.sum() == 0 or (~te).sum() == 0:
        continue
    c = Calibrator().fit([X[i] for i in np.where(~te)[0]], y[~te].tolist(), C=a.C)
    oof[te] = [c.p(X[i]) for i in np.where(te)[0]]
prior = np.array([Calibrator().prior(x) for x in X])
scale = len(y) / max(1, N)        # PA is averaged over all questions; no-candidate questions score 0
tau, pa = best_threshold(oof, y)
print(f"answer-everything PA            = {(2 * y.mean() - 1) * scale:.4f}")
print(f"prior     @0.5                  = {expected_pa(prior, y, 0.5) * scale:.4f}")
print(f"fitted    @0.5 (grouped OOF)    = {expected_pa(oof, y, 0.5) * scale:.4f}")
print(f"fitted    @best tau={tau:.2f}         = {pa * scale:.4f}   (optimistic: tau chosen on the same OOF scores)")
per = defaultdict(lambda: [0, 0, 0])
for x, yy, p in zip(X, y, oof):
    k = x.get("intent") or "-"
    per[k][0] += 1
    per[k][1] += yy
    per[k][2] += (1 if yy else -1) if p > 0.5 else 0
print(f"\n{'intent':26s}{'n':>5s}{'acc':>7s}{'PA@0.5':>8s}")
for k, (n, ok, s) in sorted(per.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:26s}{n:5d}{ok / n:7.2f}{s / n:8.2f}")
# Trace Inversion [R39]: tune the SE vote threshold as the authors' runner does (grid search that maximises the
# accuracy of the questions that would still be answered), only when the dump contains inversion features.
ti = [(x.get("ti_se", 0.0), yy) for x, yy in zip(X, y) if x.get("ti_used")]
extra = {}
if len(ti) >= 30:
    best = None
    for t in np.linspace(0.05, 0.95, 91):
        kept = [yy for s_, yy in ti if s_ >= t]
        if len(kept) >= max(10, 0.3 * len(ti)):
            acc = sum(kept) / len(kept)
            if best is None or acc > best[1]:
                best = (float(t), acc)
    if best:
        extra["ti_se_threshold"] = round(best[0], 3)
        print(f"trace-inversion SE threshold = {best[0]:.3f} (answered accuracy {best[1]:.3f} on {len(ti)} LLM answers)")
cal = Calibrator().fit(X, y.tolist(), C=a.C)
cal.extra = extra
cal.save(a.out)
print(f"\nsaved {a.out}")
