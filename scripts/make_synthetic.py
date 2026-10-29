"""Generate a synthetic DataMate-style dev set (messy files + questions + gold).

    python scripts/make_synthetic.py --out data/synthetic --n 9 --seed 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.synthetic import generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--n", type=int, default=9, help="number of businesses")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
n = generate(a.out, a.n, a.seed)
print(f"wrote {n} questions for {a.n} businesses to {a.out}")
