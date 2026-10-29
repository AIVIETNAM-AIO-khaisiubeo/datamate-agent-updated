"""End-to-end tests on a freshly generated synthetic split (official layout, messy files)."""
import json
import os
import re
import subprocess
import sys

from datamate import salesview
from datamate.agent import Agent
from datamate.calibrate import Calibrator, best_threshold
from datamate.loader import Business
from datamate.lunar import LunarCalendar
from datamate.readme import parse_readme
from datamate.rulebook import RuleBook
from datamate.scorer import report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def _agent(cfg, synth_dir, llm=None):
    rb = RuleBook.load(os.path.join(synth_dir, "rulebook.md"))
    return Agent(cfg, rb, LunarCalendar(os.path.join(synth_dir, "calendar.csv")), llm, None, Calibrator())


def test_loader_and_sales_view_on_messy_files(synth_dir, cfg):
    rb = RuleBook.load(os.path.join(synth_dir, "rulebook.md"))
    for bid in sorted(os.listdir(os.path.join(synth_dir, "data"))):
        biz = Business(os.path.join(synth_dir, "data", bid), cache_dir=cfg["data"]["cache_dir"])
        assert not biz.load_errors
        sm = salesview.build(biz, parse_readme(biz.readme), rb)
        assert sm.lines is not None and len(sm.lines) > 100, sm.warnings
        assert sm.lines["order_ts"].notna().all()
        assert 0.8 < sm.orders["is_valid"].mean() < 0.97          # cancelled / returned orders excluded
        assert sm.lines["amount"].notna().mean() > 0.99          # VND strings parsed
        assert sm.flags.get("unknown_status_share", 0) == 0


def run_all(agent, qs):
    preds = {}
    for it in qs:
        pred, feats, _ = agent.solve(it)
        out, _ = agent.decide(pred, feats)
        preds[it["id"]] = out
    return preds


def test_offline_agent_is_precise(cfg, synth_dir):
    qs = _load(os.path.join(synth_dir, "questions.jsonl"))
    golds = {g["id"]: g for g in _load(os.path.join(synth_dir, "answers.jsonl"))}
    r = report(run_all(_agent(cfg, synth_dir), qs), golds)
    assert r["PA"] > 0.4, r
    assert r["selective_precision"] > 0.9, r                   # abstention must protect precision


class FakeLLM:
    """Scripted OpenAI-compatible stand-in: the first answer is buggy SQL (exercises the
    execution-feedback repair); the repair returns a correct query for 'average order value'
    questions; pandas returns a constant; ReAct probes once then gives up; judge picks A."""
    model = "fake"

    def __init__(self):
        self.calls = 0
        self.stats = {}

    def chat(self, messages, n=1, temperature=None, thinking=None, max_tokens=None, seed=None):
        self.calls += 1
        sys_p, last = messages[0]["content"], messages[-1]["content"]
        if "WINNER" in sys_p:
            return ["WINNER: A"] * n
        if "VERDICT" in sys_p:                       # ReViSQL reconciliation verdict
            return ["VERDICT: YES"] * n
        if sys_p.startswith("You are a puzzle solver"):   # trace inversion: reconstruction
            return ["QUESTION: doanh thu trung bình mỗi đơn là bao nhiêu\nPERIOD: none\nFILTERS: none\nRESULT: number"]
        if "same framing, intent, and context" in sys_p:  # trace inversion: TrInv-LLM
            return ["YES"]
        if "failed or returned no rows" in last:
            rng = re.search(r"PRIMARY .*?: date_col >= DATE '([\d-]+)' AND date_col < DATE '([\d-]+)'", messages[1]["content"])
            a, b = rng.group(1), rng.group(2)
            return ["TYPE: number\n```sql\nSELECT AVG(revenue) FROM v_orders "
                    f"WHERE is_valid AND order_date >= DATE '{a}' AND order_date < DATE '{b}'\n```"] * n
        if "pandas" in sys_p:
            return ["TYPE: number\n```python\nresult = len(T['v_orders'])\n```"] * n
        if "EXPLORE" in sys_p:
            if "OBSERVATION" in last:
                return ["FINAL\nTYPE: unanswerable\nREASON: probe done"]
            return ["ACTION: probe\n```sql\nSELECT DISTINCT status FROM v_orders\n```"]
        return ["TYPE: number\n```sql\nSELECT SUM(revenu) FROM v_orders\n```"] * n   # typo -> binder error


def test_llm_path_with_fake_llm(cfg, synth_dir):
    qs = [q for q in _load(os.path.join(synth_dir, "questions.jsonl")) if "trung b" in q["question"].lower()
          or "trung binh" in q["question"].lower()][:2]
    golds = {g["id"]: g for g in _load(os.path.join(synth_dir, "answers.jsonl"))}
    llm = FakeLLM()
    agent = _agent(cfg, synth_dir, llm)
    assert qs
    for it in qs:
        pred, feats, trace = agent.solve(it)
        fams = [c["family"] for c in trace["candidates"]]
        assert any(f.endswith("+fix") for f in fams)              # repair path executed
        assert "react" in " ".join(fams)                          # escalation path executed
        assert pred["type"] == "number"
        assert abs(pred["answer"] - golds[it["id"]]["answer"]) <= 1   # money rounded to dong
        assert 0 <= agent.cal.p(feats) <= 1
        assert feats["ti_used"] == 1.0 and 0 <= feats["ti_se"] <= 1        # trace inversion ran on the winner
        assert "rc_pass_frac" in feats and trace["requirements"]           # reconciliation + requirement list
    assert llm.calls > 0


def test_reward_function(cfg, synth_dir):
    from datamate.reward import score_completion
    qs = _load(os.path.join(synth_dir, "questions.jsonl"))
    golds = {g["id"]: g for g in _load(os.path.join(synth_dir, "answers.jsonl"))}
    agent = _agent(cfg, synth_dir)
    it = next(q for q in qs if golds[q["id"]]["type"] == "number")
    r, info = score_completion(agent, it, "no program here", golds[it["id"]])
    assert r == -1.0
    r, info = score_completion(agent, it, "TYPE: number\n```sql\nSELECT -12345.678\n```", golds[it["id"]])
    assert r == 0.0 and info["pred"]["type"] == "number"
    g = golds[it["id"]]
    r, _ = score_completion(agent, it, f"TYPE: number\n```sql\nSELECT {g['answer']}\n```", g)
    assert r == 1.0


def test_run_script_writes_submission_and_logs(synth_dir, tmp_path):
    out, logs = tmp_path / "sub.jsonl", tmp_path / "logs"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run.py"), "--data", synth_dir, "--out", str(out),
                        "--logs", str(logs), "--set", "llm.backend=none", "--set", f"data.cache_dir={tmp_path / 'c'}",
                        "--set", "decision.calibrator_path=/nonexistent.pkl"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr
    qs = _load(os.path.join(synth_dir, "questions.jsonl"))
    sub = _load(str(out))
    assert [s["id"] for s in sub] == [q["id"] for q in qs]
    for s in sub:
        assert s["type"] in ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain")
        if s["type"] == "abstain":
            assert set(s) == {"id", "type"}
        if s["type"] == "unanswerable":
            assert isinstance(s.get("reason", ""), str)
    log = json.load(open(logs / f"{qs[0]['id']}.json", encoding="utf-8"))
    assert {"id", "final", "elapsed_s"} <= set(log)


def test_threshold_search():
    tau, pa = best_threshold([0.9, 0.8, 0.7, 0.4, 0.3, 0.2], [1, 1, 0, 0, 0, 1], n_boot=20)
    assert 0.3 <= tau <= 0.9 and pa >= 0
