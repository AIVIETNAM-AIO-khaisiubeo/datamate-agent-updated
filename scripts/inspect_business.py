"""Print what the agent "sees" for one business: tables, roles, canonical views, card.

    python scripts/inspect_business.py data/synthetic/businesses/biz_00000 [--rulebook data/synthetic/rulebook.md]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate import semantic  # noqa: E402
from datamate.loader import Business  # noqa: E402
from datamate.prompts import business_card  # noqa: E402
from datamate.rulebook import RuleBook  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("biz_dir")
ap.add_argument("--rulebook", default=None)
a = ap.parse_args()
biz = Business(a.biz_dir)
sem = semantic.build(biz, RuleBook.load(a.rulebook))
print(business_card(biz, sem, biz.profile(roles=sem.roles)))
for name in ("v_sales", "v_debts", "v_batches"):
    df = getattr(sem, {"v_sales": "sales", "v_debts": "debts", "v_batches": "batches"}[name])
    if df is not None:
        print(f"\n--- {name} (head) ---\n{df.head(5).to_string(index=False)}")
if biz.load_errors:
    print("\nLOAD ERRORS:", *biz.load_errors, sep="\n  ")
