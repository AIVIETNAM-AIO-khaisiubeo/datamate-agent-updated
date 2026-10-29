import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    """A synthetic split in the official layout (data/<biz>, rulebook.md, calendar.csv, answers.jsonl)."""
    from datamate.synthetic import generate
    d = tmp_path_factory.mktemp("synth")
    generate(str(d), n_business=3, seed=7)
    return str(d)


@pytest.fixture(scope="session")
def cfg(synth_dir, tmp_path_factory):
    from datamate.config import load_config
    c = load_config(overrides=["llm.backend=none", "agent.workers=2"])
    c["data"]["root"] = synth_dir
    c["data"]["cache_dir"] = str(tmp_path_factory.mktemp("cache"))
    c["decision"]["calibrator_path"] = "/nonexistent.pkl"
    return c
