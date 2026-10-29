"""Answer a split: <split>/questions.jsonl -> submission.jsonl (+ per-question logs).

    python scripts/run.py --data /data --out /result/submission.jsonl --logs /result/logs
    python scripts/run.py --data path/to/public_test --out runs/public.jsonl --set llm.backend=none   # no LLM

Competition requirements handled here:
  * one line per input id, written as soon as it is decided (append + flush), so an
    interrupted run still scores everything written so far; the file is rewritten in
    input order at the end;
  * a per-question log (id, final answer, elapsed time, intent, time ranges, every
    candidate program with its execution output) in --logs;
  * wall-clock budget (run.time_budget_s): after run.fast_after_s questions are
    answered without the LLM (templates + rules), after the hard limit the rest abstain;
  * offline only; total parameters of declared models checked against the 15B cap;
  * deterministic: per-question seeds derived from the question text.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.agent import Agent  # noqa: E402
from datamate.calibrate import Calibrator  # noqa: E402
from datamate.config import load_config, resolve_shared_paths  # noqa: E402
from datamate.llm import build_llms  # noqa: E402
from datamate.lunar import LunarCalendar  # noqa: E402
from datamate.rulebook import RuleBook  # noqa: E402


def check_param_budget(cfg):
    L = cfg["llm"]
    if L.get("backend", "openai") == "none":
        return 0.0
    used = {L["main_model"]}
    if L.get("sql_model"):
        used.add(L["sql_model"])
    se = (cfg.get("inversion") or {}).get("se_model")
    if se and (cfg.get("inversion") or {}).get("enabled", True):
        # a local path is accepted; its declared name is the last two path parts (org/model)
        used.add(se if se in (L.get("params_b") or {}) else "/".join(se.rstrip("/").split("/")[-2:]))
    table = L.get("params_b", {}) or {}
    total = sum(float(table.get(m, 0.0)) for m in used)
    missing = [m for m in used if m not in table]
    if missing:
        print(f"[datamate] WARNING: parameter count not declared for {missing} (llm.params_b)", file=sys.stderr)
    cap = float(cfg.get("run", {}).get("max_total_params_b", 15.0))
    if total > cap:
        raise SystemExit(f"declared models total {total}B parameters > {cap}B cap: {sorted(used)}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="split directory (questions.jsonl, rulebook.md, calendar.csv, data/)")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--logs", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="override config, e.g. --set agent.n_sql=5")
    ap.add_argument("--dump", default=None, help="features + traces jsonl (for calibration / error analysis)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    if a.data:
        cfg["data"]["root"] = a.data
    qpath = a.questions or os.path.join(cfg["data"]["root"], "questions.jsonl")
    rb_path, cal_path = resolve_shared_paths(cfg)
    rb = RuleBook.load(rb_path)
    lunar = LunarCalendar(cal_path)
    total_b = check_param_budget(cfg)
    llm, sql_llm = build_llms(cfg)
    D = cfg["decision"]
    cal = Calibrator.load(D["calibrator_path"], D.get("offline_template_prior", 0.75))
    agent = Agent(cfg, rb, lunar, llm, sql_llm, cal)
    print(f"[datamate] rulebook={rb_path} ({rb.summary()}) calendar={lunar.source} "
          f"llm={'off' if llm is None else llm.model} params={total_b}B "
          f"calibrator={'fitted' if cal.lr is not None else 'prior'}", file=sys.stderr)

    items = [json.loads(l) for l in open(qpath, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[: a.limit]
    order = [it["id"] for it in items]
    by_biz = OrderedDict()
    for it in items:
        by_biz.setdefault(it["business_id"], []).append(it)

    R = cfg.get("run", {})
    budget = float(R.get("time_budget_s", 6 * 3600 - 900))
    fast_after = float(R.get("fast_after_s", budget * 0.85))
    t_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if a.logs:
        os.makedirs(a.logs, exist_ok=True)
    lock = threading.Lock()
    results = {}
    partial = open(a.out, "w", encoding="utf-8")
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None

    def one(it):
        t0 = time.time()
        elapsed = t0 - t_start
        if elapsed > budget:
            out, p, pred, feats, trace = {"type": "abstain"}, 0.0, {"type": None}, {}, {"note": "time budget exhausted"}
        else:
            try:
                pred, feats, trace = agent.solve(it, fast=elapsed > fast_after)
                out, p = agent.decide(pred, feats)
            except Exception as e:  # never lose a line: abstain scores 0
                pred, feats, trace = {"type": None}, {}, {"error": repr(e)[:500]}
                out, p = {"type": "abstain"}, 0.0
        sub = {"id": it["id"], **out}
        dt = time.time() - t0
        with lock:
            results[it["id"]] = sub
            partial.write(json.dumps(sub, ensure_ascii=False) + "\n")
            partial.flush()
            if dump:
                dump.write(json.dumps({"id": it["id"], "business_id": it["business_id"], "p": p, "pred": pred,
                                       "feats": feats, "decision": out["type"]}, ensure_ascii=False, default=str) + "\n")
        if a.logs:
            log = {"id": it["id"], "final": sub, "elapsed_s": round(dt, 3), "p_correct": round(p, 4),
                   "candidate_answer": pred, "features": feats, **trace}
            with open(os.path.join(a.logs, f"{it['id']}.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=1, default=str)
        return it["id"]

    workers = max(1, int(cfg["agent"].get("workers", 8)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for bid, its in by_biz.items():           # one shop at a time: shared prompt prefix, bounded memory
            list(ex.map(one, its))
            agent.release(bid)
    partial.close()
    with open(a.out, "w", encoding="utf-8") as fo:     # final file in input order
        for qid in order:
            fo.write(json.dumps(results.get(qid, {"id": qid, "type": "abstain"}), ensure_ascii=False) + "\n")
    if dump:
        dump.close()
    dt = time.time() - t_start
    n_ans = sum(1 for s in results.values() if s["type"] != "abstain")
    n_ua = sum(1 for s in results.values() if s["type"] == "unanswerable")
    print(f"[datamate] {len(items)} questions in {dt:.1f}s ({dt / max(1, len(items)):.2f}s/q); "
          f"answered {n_ans} (unanswerable {n_ua}); llm={llm.stats if llm else {}}", file=sys.stderr)


if __name__ == "__main__":
    main()
