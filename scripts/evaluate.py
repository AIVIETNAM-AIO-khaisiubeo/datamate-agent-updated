"""Score a submission against gold with the official PA rule (local re-implementation).

    python scripts/evaluate.py submission.jsonl data/synthetic/gold.jsonl [--dump runs/dev.jsonl]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.scorer import report, score_one  # noqa: E402


def load(p):
    return {d["id"]: d for d in map(json.loads, (l for l in open(p, encoding="utf-8") if l.strip()))}


ap = argparse.ArgumentParser()
ap.add_argument("submission")
ap.add_argument("gold")
ap.add_argument("--dump", default=None, help="run dump to print per-category errors")
ap.add_argument("--show_wrong", type=int, default=15)
a = ap.parse_args()
preds, golds = load(a.submission), load(a.gold)
r = report(preds, golds)
print(f"PA = {r['PA']:.4f}   (n={r['n']}, correct={r['correct']}, wrong={r['wrong']}, abstain={r['abstain']}, "
      f"coverage={r['coverage']:.2%}, selective precision={r['selective_precision']:.2%})")
print(f"{'ref type':14s} {'n':>4s} {'correct':>8s} {'wrong':>6s} {'abstain':>8s}")
for t, b in sorted(r["by_type"].items()):
    print(f"{t:14s} {b['n']:4d} {b['correct']:8d} {b['wrong']:6d} {b['abstain']:8d}")
cats = defaultdict(lambda: [0, 0, 0, 0])
for i, g in golds.items():
    s = score_one(preds.get(i), g)
    c = cats[g.get("category", "-")]
    c[0] += 1
    c[{1: 1, -1: 2, 0: 3}[s]] += 1
if any(k != "-" for k in cats):
    print("\nby category: n / correct / wrong / abstain")
    for k, v in sorted(cats.items()):
        print(f"  {k:22s} {v[0]:4d} {v[1]:4d} {v[2]:4d} {v[3]:4d}")
dump = load(a.dump) if a.dump else {}
shown = 0
for i, g in golds.items():
    if score_one(preds.get(i), g) == -1 and shown < a.show_wrong:
        shown += 1
        print(f"\n✗ {i} [{g.get('category', '')}] gold={g['type']}:{json.dumps(g.get('answer'), ensure_ascii=False)[:120]}")
        print(f"  pred={json.dumps(preds.get(i), ensure_ascii=False)[:200]}")
        if i in dump:
            d = dump[i]
            print(f"  p={d.get('p'):.3f} cands={[(c['family'], c['key'][:60]) for c in d['trace'].get('candidates', [])][:6]}")
