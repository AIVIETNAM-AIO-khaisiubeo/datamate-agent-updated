from __future__ import annotations

import copy
import glob
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _set(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    path = path or os.path.join(ROOT, "configs", "default.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        _set(cfg, k.strip(), v.strip())
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def discover(root: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


def resolve_shared_paths(cfg: dict) -> tuple[str | None, str | None]:
    root = cfg["data"]["root"]
    rb = cfg["data"].get("rulebook") or discover(root, ["rulebook*.md", "rulebook*.txt", "rulebook*.yaml",
                                                          "rulebook*.json", "*rule*book*", "*quy*tac*"])
    cal = cfg["data"].get("lunar_calendar") or discover(root, ["*lunar*.csv", "*lunar*.json", "*am*lich*",
                                                               "*calendar*.csv", "*calendar*.json", "*lich*.csv"])
    return rb, cal
