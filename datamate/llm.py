"""OpenAI-compatible chat client (vLLM / SGLang / any API) with a disk cache.

Serving recommendation: vLLM (PagedAttention, Kwon et al., SOSP 2023) with
automatic prefix caching, or SGLang (RadixAttention, Zheng et al., NeurIPS
2024). Prompts are laid out as [rulebook+contract | business card | question]
so the first two segments are shared across questions and hit the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time


class DiskCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    def get(self, k):
        with self.lock:
            r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
            self.db.commit()


_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 1200,
                 timeout_s: float = 120, enable_thinking: bool = False, cache_path: str | None = None,
                 thinking_max_tokens: int = 8192):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s, max_retries=2)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_max_tokens = thinking_max_tokens
        self.cache = DiskCache(cache_path) if cache_path else None
        self.n_supported = True
        self.stats = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def chat(self, messages: list[dict], n: int = 1, temperature: float | None = None, thinking: bool | None = None,
             max_tokens: int | None = None, seed: int | None = None) -> list[str]:
        temp = self.temperature if temperature is None else temperature
        think = self.enable_thinking if thinking is None else thinking
        mt = max_tokens or (self.thinking_max_tokens if think else self.max_tokens)
        key = hashlib.sha256(json.dumps([self.model, messages, n, temp, think, mt, seed], ensure_ascii=False).encode()).hexdigest()
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cached"] += 1
                return hit
        t0 = time.time()
        outs = self._call(messages, n, temp, think, mt, seed)
        self.stats["seconds"] += time.time() - t0
        outs = [_THINK.sub("", o or "").strip() for o in outs]
        if self.cache:
            self.cache.put(key, outs)
        return outs

    def _call(self, messages, n, temp, think, mt, seed=None):
        extra = {"chat_template_kwargs": {"enable_thinking": bool(think)}}
        kw = {"seed": seed} if seed is not None else {}
        if n > 1 and self.n_supported:
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, n=n, temperature=temp,
                                                        max_tokens=mt, extra_body=extra, **kw)
                self._account(r)
                outs = [c.message.content or "" for c in r.choices]
                if len(outs) == n:
                    return outs
            except Exception as e:
                if "extra_body" in str(e) or "chat_template_kwargs" in str(e):
                    extra = {}
                self.n_supported = False
        outs = []
        for i in range(n):
            kwi = {"seed": seed + i} if seed is not None else {}
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temp,
                                                        max_tokens=mt, extra_body=extra or None, **kwi)
            except Exception as e:
                if extra and ("chat_template_kwargs" in str(e) or "extra" in str(e).lower()):
                    extra = {}
                    r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temp,
                                                            max_tokens=mt, **kwi)
                else:
                    raise
            self._account(r)
            outs.append(r.choices[0].message.content or "")
        return outs

    def _account(self, r):
        self.stats["calls"] += 1
        u = getattr(r, "usage", None)
        if u:
            self.stats["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.stats["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0


def build_llms(cfg: dict):
    L = cfg["llm"]
    if L.get("backend", "openai") == "none":
        return None, None
    cache_dir = cfg["data"].get("cache_dir", ".cache/")
    mk = lambda url, model: LLM(url, L.get("api_key", "EMPTY"), model, L.get("temperature", 0.7),
                                L.get("max_tokens", 1500), L.get("timeout_s", 300), L.get("enable_thinking", False),
                                os.path.join(cache_dir, "llm_cache.sqlite") if L.get("cache", True) else None,
                                L.get("thinking_max_tokens", 8192))
    main = mk(L["base_url"], L["main_model"])
    sql = mk(L.get("sql_base_url") or L["base_url"], L["sql_model"]) if L.get("sql_model") else None
    return main, sql
