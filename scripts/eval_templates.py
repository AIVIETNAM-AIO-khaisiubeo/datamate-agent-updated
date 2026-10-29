"""Evaluate the deterministic layer (templates + rules, no LLM) on a labelled split.

    python scripts/eval_templates.py --split /path/to/train [--biz biz_101,biz_105] [--show 20]

Prints per-intent and per-business correct / wrong / abstain (at the template's own
firing, before calibration) — the main diagnostic for the sales views.
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.agent import Agent  # noqa: E402
from datamate.calibrate import Calibrator  # noqa: E402
from datamate.config import load_config  # noqa: E402
from datamate.lunar import LunarCalendar  # noqa: E402
from datamate.rulebook import RuleBook  # noqa: E402
from datamate.scorer import is_correct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--biz", default=None)
ap.add_argument("--intent", default=None)
ap.add_argument("--show", type=int, default=15)
ap.add_argument("--out", default=None, help="write per-question jsonl (features, template answers, gold)")
a = ap.parse_args()

cfg = load_config(overrides=["llm.backend=none"])
cfg["data"]["root"] = a.split
rb = RuleBook.load(os.path.join(a.split, "rulebook.md"))
agent = Agent(cfg, rb, LunarCalendar(os.path.join(a.split, "calendar.csv")), None, None, Calibrator())
Q = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
G = {g["id"]: g for g in map(json.loads, open(os.path.join(a.split, "answers.jsonl"), encoding="utf-8"))}
if a.biz:
    keep = set(a.biz.split(","))
    Q = [q for q in Q if q["business_id"] in keep]
Q.sort(key=lambda q: (q["business_id"], q["id"]))
by_int = collections.defaultdict(lambda: [0, 0, 0, 0])
by_biz = collections.defaultdict(lambda: [0, 0, 0, 0])
wrong = []
fo = open(a.out, "w", encoding="utf-8") if a.out else None
t0 = time.time()
for q in Q:
    pred, feats, trace = agent.solve(q)
    g = G[q["id"]]
    intent = trace.get("spec", {}).get("intent")
    if a.intent and intent != a.intent:
        continue
    tpl = [c for c in trace.get("candidates", []) if c["family"].startswith(("tpl:", "rule"))]
    fired = pred.get("type") not in (None, "abstain")
    ok = fired and is_correct(pred, g)
    for d in (by_int[intent], by_biz[q["business_id"]]):
        d[0] += 1
        d[1] += fired and ok
        d[2] += fired and not ok
        d[3] += not fired
    if fired and not ok:
        wrong.append((q, g, pred, trace))
    if fo:
        fo.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "intent": intent, "pred": pred,
                             "gold": g, "correct": bool(ok), "feats": feats,
                             "tpl": tpl}, ensure_ascii=False, default=str) + "\n")
print(f"{len(Q)} questions in {time.time() - t0:.0f}s")
print(f"{'intent':28s}{'n':>5s}{'ok':>5s}{'bad':>5s}{'none':>6s}")
for k, v in sorted(by_int.items(), key=lambda kv: -kv[1][0]):
    print(f"{str(k):28s}{v[0]:5d}{v[1]:5d}{v[2]:5d}{v[3]:6d}")
print(f"\n{'business':12s}{'n':>5s}{'ok':>5s}{'bad':>5s}{'none':>6s}")
for k, v in sorted(by_biz.items()):
    print(f"{k:12s}{v[0]:5d}{v[1]:5d}{v[2]:5d}{v[3]:6d}")
tot = [sum(v[i] for v in by_int.values()) for i in range(4)]
print(f"TOTAL n={tot[0]} ok={tot[1]} bad={tot[2]} none={tot[3]}  PA(fire-all)={(tot[1]-tot[2])/max(1,tot[0]):.4f}")
for q, g, pred, tr in wrong[: a.show]:
    print(f"\n✗ {q['business_id']} {q['as_of_date']} [{tr.get('spec', {}).get('intent')}] {q['question'][:110]}")
    print(f"   gold={g['type']}:{json.dumps(g.get('answer'), ensure_ascii=False)[:100]}  pred={pred.get('type')}:{json.dumps(pred.get('answer'), ensure_ascii=False)[:100]}")
    print(f"   time={[(r['start'], r['end_exclusive']) for r in tr.get('time', [])]} cands={[(c['family'], c['key'][:50], c.get('detail', '')[:80]) for c in tr.get('candidates', [])][:3]}")
