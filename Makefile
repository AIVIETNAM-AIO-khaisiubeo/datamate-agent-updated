PY ?= python3
DATA ?= data/synthetic
OUT ?= runs

.PHONY: install test synth offline eval serve run fit clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q tests

synth:            ## synthetic dev + held-out splits
	$(PY) scripts/make_synthetic.py --out data/synthetic --n 6 --seed 0
	$(PY) scripts/make_synthetic.py --out data/synthetic_test --n 6 --seed 1

offline:          ## LLM-free run (templates + rules) — works on any laptop
	$(PY) scripts/run.py --questions $(DATA)/questions.jsonl --data_root $(DATA) --out $(OUT)/sub_offline.jsonl \
	    --dump $(OUT)/dump_offline.jsonl --set llm.backend=none
	$(PY) scripts/evaluate.py $(OUT)/sub_offline.jsonl $(DATA)/gold.jsonl --dump $(OUT)/dump_offline.jsonl

serve:            ## vLLM OpenAI-compatible server (GPU)
	bash scripts/serve_vllm.sh

run:              ## full agent against the server started by `make serve`
	$(PY) scripts/run.py --questions $(DATA)/questions.jsonl --data_root $(DATA) --out $(OUT)/sub.jsonl --dump $(OUT)/dump.jsonl
	$(PY) scripts/evaluate.py $(OUT)/sub.jsonl $(DATA)/gold.jsonl --dump $(OUT)/dump.jsonl

fit:              ## fit calibrator on a labelled run, then re-run `make run`
	$(PY) scripts/fit_calibrator.py $(OUT)/dump.jsonl $(DATA)/gold.jsonl --out artifacts/calibrator.pkl

clean:
	rm -rf .cache runs artifacts/*.pkl
