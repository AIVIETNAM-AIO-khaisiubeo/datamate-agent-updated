"""Export a labelled split as RLVR training data (ReViSQL [R40] recipe, data side only).

    python scripts/export_rlvr_data.py --split path/to/train --out runs/rlvr_train.jsonl [--val_frac 0.15]

Each line holds the exact chat messages the agent sends at inference (system = contract +
rulebook, user = shop card + question block with requirements R1..Rn), the gold answer, and
curation flags. Following ReViSQL's lesson that noisy labels produce spurious rewards, every
question is also answered by the deterministic templates: when a confident template disagrees
with the gold, the item is flagged `suspect` for a human check before training (ReViSQL used an
LLM reviewer to flag items, then SQL experts to correct them). The split is made BY SHOP, like
the calibrator, so validation measures transfer to unseen shops.

Training itself (GRPO/CISPO with the reward in `datamate/reward.py`) needs GPUs and is not part
of this repository; see ReViSQL's `scripts/train.py`. Check with the organisers that a
fine-tuned checkpoint is admissible under rule 1 before using one.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.agent import Agent  # noqa: E402
from datamate.calibrate import Calibrator  # noqa: E402
from datamate.config import load_config  # noqa: E402
from datamate.lunar import LunarCalendar  # noqa: E402
from datamate.prompts import SQL_DIRECT, question_block  # noqa: E402
from datamate.question import analyze  # noqa: E402
from datamate.rulebook import RuleBook  # noqa: E402
from datamate.scorer import is_correct  # noqa: E402
from datamate.templates import run_templates  # noqa: E402
from datamate.textnorm import qnorm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val_frac", type=float, default=0.15)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

cfg = load_config(overrides=["llm.backend=none"])
cfg["data"]["root"] = a.split
rb = RuleBook.load(os.path.join(a.split, "rulebook.md"))
agent = Agent(cfg, rb, LunarCalendar(os.path.join(a.split, "calendar.csv")), None, None, Calibrator())
Q = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
G = {g["id"]: g for g in map(json.loads, open(os.path.join(a.split, "answers.jsonl"), encoding="utf-8"))}
Q.sort(key=lambda q: (q["business_id"], q["id"]))
if a.limit:
    Q = Q[: a.limit]
shops = sorted({q["business_id"] for q in Q})
random.Random(0).shuffle(shops)
val = set(shops[: max(1, round(a.val_frac * len(shops)))]) if len(shops) > 1 else set()

n = sus = 0
with open(a.out, "w", encoding="utf-8") as fo:
    for i, it in enumerate(Q):
        ctx = agent.context(it["business_id"])
        if ctx is None:
            continue
        q = it["question"]
        spec = analyze(q)
        tctx = agent.resolver.resolve(q, it["as_of_date"])
        links = ctx.vindex.link(q)
        reqs = agent._requirements(ctx, spec, tctx, links)
        user = ctx.card + "\n\n" + question_block(it, qnorm(q), tctx, links, spec, rb, ctx.sm, reqs)
        tpl = run_templates(spec, tctx, ctx.sm, links, rb, ctx.data_lo, ctx.data_hi, agent.weekdays)
        conf = [t for t in tpl if t.strength >= 0.6 and t.answer[0] != "unanswerable"]
        gold = G.get(it["id"])
        suspect = bool(conf) and gold is not None and not is_correct({"type": conf[0].answer[0],
                                                                       "answer": conf[0].answer[1]}, gold)
        sus += suspect
        fo.write(json.dumps({
            "id": it["id"], "business_id": it["business_id"], "split": "val" if it["business_id"] in val else "train",
            "as_of_date": it["as_of_date"], "question": q,
            "messages": [{"role": "system", "content": agent._system(SQL_DIRECT)}, {"role": "user", "content": user}],
            "gold": gold, "requirements": [r.text for r in reqs], "suspect": suspect,
            "template_answer": list(conf[0].answer) if conf else None,
        }, ensure_ascii=False, default=str) + "\n")
        n += 1
        if i + 1 == len(Q) or Q[i + 1]["business_id"] != it["business_id"]:
            agent.release(it["business_id"])
print(f"wrote {n} items to {a.out} ({len(val)} validation shops); {sus} flagged suspect for review")
