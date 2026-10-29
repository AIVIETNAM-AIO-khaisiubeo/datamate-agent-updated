# DataMate Agent

**Zalo AI Challenge 2026 · DataMate **

DataMate is a Vietnamese natural-language data agent for small-business spreadsheets and databases. It answers questions over CSV, Excel, and SQLite data by combining deterministic data preparation, auditable query templates, LLM-generated programs, execution-based verification, candidate reconciliation, and calibrated abstention.

The central principle is **answer only when the evidence supports the answer**.

> **Scoring objective:** Penalized Accuracy (PA) gives +1 for a correct answer, −1 for an incorrect answer, and 0 for abstention. If the estimated probability that an answer is correct is p, answering has expected score 2p − 1. The basic decision rule is therefore to answer only when p > 0.5.

## Contents

- [What the agent does](#what-the-agent-does)
- [Input and output contract](#input-and-output-contract)
- [Architecture](#architecture)
- [Recorded results](#recorded-results)
- [First-round workflow](#first-round-workflow)
- [Repository map](#repository-map)
- [Limitations and validation status](#limitations-and-validation-status)
- [Documentation and research](#documentation-and-research)

## What the agent does

- **Normalizes messy business data:** repairs common spreadsheet headers, removes repeated headers and summary rows, parses mixed dates and numbers, and loads tables into DuckDB.
- **Builds a semantic layer:** uses each shop's column descriptions and inferred column roles to construct canonical business views and consistency checks.
- **Understands question constraints:** extracts intent, answer type, periods, filters, thresholds, ranking constraints, and evidence relevant to answerability.
- **Generates diverse candidates:** runs deterministic templates and, in LLM mode, multiple SQL samples with optional plan-to-SQL, Python/pandas, ReAct exploration, and execution-feedback repair.
- **Checks before voting:** executes candidates, normalizes their outputs, groups equivalent answers, and checks explicit requirements against candidate programs.
- **Calibrates the decision:** combines candidate agreement, execution signals, constraint coverage, and question–program alignment into a correctness estimate before answering or abstaining.

## Input and output contract

A split contains question metadata, a rulebook, an optional calendar, and one folder per business. The loader searches recursively, so business data may be organized into nested folders.

~~~text
<split>/
├── questions.jsonl
├── rulebook.md
├── calendar.csv                 # optional solar–lunar calendar
└── <business_id>/
    ├── sales.xlsx
    ├── customers.csv
    ├── inventory.sqlite
    └── README.md                 # optional column descriptions
~~~

Each line in questions.jsonl is a JSON object with fields such as id, business_id, question, and as_of_date. Challenge questions are Vietnamese; the English example below illustrates the record schema only.

~~~json
{
  "id": "q_example",
  "business_id": "biz_example",
  "question": "Which product sold the most last week?",
  "as_of_date": "2026-03-04"
}
~~~

The runner writes one output record per input question to a JSONL submission file. Optional per-question logs record the decision, timing, parsed context, candidate programs, execution results, and confidence features.


## Recorded results

The figures below are from **template-and-rules mode without an LLM**. They are not measurements of the full Qwen3-14B pipeline.

| Split / configuration | Correct | Wrong | Abstain | Penalized Accuracy |
|---|---:|---:|---:|---:|
| Train: 3,000 questions, 30 businesses; in-sample | 2,667 | 3 | 330 | **0.888** |
| Public: 1,000 questions, 14 businesses; no gold labels | — | — | 365 | Not available |

The public run returned 611 ordinary answers and 24 evidence-based unanswerable outputs, for 635 non-abstaining outputs. Because public gold labels are unavailable in this repository, **public-test PA cannot be calculated**. The train score is in-sample: errors were inspected during development, so it is not an unbiased estimate of generalization.

The tracked round-one output is [submissions/public_test_round1.jsonl](submissions/public_test_round1.jsonl). Detailed counts and error analysis are documented in [docs/RESULTS.md](docs/RESULTS.md).

## First-round workflow

### 1. Install and test

Use Python 3.10 or newer in a clean virtual environment.

~~~bash
python -m pip install -r requirements.txt
make test
~~~

### 2. Choose the inference path

**Template-only fallback (CPU-friendly; no LLM call):**

~~~bash
python scripts/run.py --data path/to/public_test --out submissions/public_test_round1.jsonl --logs runs/public_logs --set llm.backend=none --set decision.calibrator_path=artifacts/calibrator_offline.pkl
~~~

**Full LLM path using Qwen3-14B:**

Start an OpenAI-compatible vLLM endpoint, for example:

~~~bash
vllm serve Qwen/Qwen3-14B --tensor-parallel-size 2 --enable-prefix-caching --max-model-len 32768 --seed 0
~~~

Fit the calibrator using labeled candidate traces generated with the same LLM configuration, then run the public split:

~~~bash
python scripts/run.py --data path/to/train --out runs/train_submission.jsonl --dump runs/train_dump.jsonl
python scripts/fit_calibrator.py runs/train_dump.jsonl path/to/train/answers.jsonl --out artifacts/calibrator.pkl
python scripts/run.py --data path/to/public_test --out submissions/public_test_round1.jsonl --logs runs/public_logs
~~~

Do not reuse the template-only calibrator as though it were calibrated for LLM-generated candidates. Confirm paths, the model endpoint, model-budget settings, and the current competition instructions before running the final command.

### 3. Evaluate a labeled split

~~~bash
python scripts/eval_templates.py --split path/to/train --out runs/train_offline_dump.jsonl
python scripts/evaluate.py submission_train.jsonl path/to/train/answers.jsonl
~~~

Use configs/default.yaml and the command-line help from scripts/run.py for supported options.

## Repository map

~~~text
datamate-agent/
├── README.md
├── data/
│   └── README.md                  # Expected split and business-data layout
├── datamate/
│   ├── agent.py                   # Candidate orchestration and selection
│   ├── answerability.py           # Evidence for unanswerable questions
│   ├── attributes.py              # Catalog-attribute extraction
│   ├── calibrate.py               # Correctness probability calibration
│   ├── executor.py                # Bounded SQL and Python execution
│   ├── formatter.py               # Typed answer normalization
│   ├── inversion.py               # Question–program alignment features
│   ├── linker.py                  # Entity and value linking
│   ├── loader.py                  # CSV / Excel / SQLite ingestion
│   ├── lunar.py                   # Lunar-calendar fallback
│   ├── prompts.py                 # LLM task prompts
│   ├── question.py                # Intent and constraint analysis
│   ├── readme.py                  # Shop README parsing
│   ├── requirements.py            # Explicit constraints and program checks
│   ├── roles.py                   # Table and column role inference
│   ├── rulebook.py                # Split-specific rule handling
│   ├── salesview.py               # Canonical business views
│   ├── scorer.py                  # Official scoring helpers
│   ├── synthetic.py               # Synthetic development fixtures
│   ├── templates.py               # Deterministic answer templates
│   ├── textnorm.py                # Vietnamese text normalization
│   └── timeresolver.py             # Date and time resolution
├── configs/
│   ├── default.yaml
│   └── rulebook_overrides.yaml
├── scripts/
│   ├── eval_templates.py
│   ├── evaluate.py
│   ├── export_rlvr_data.py        # Optional research utility; not needed for inference
│   ├── fit_calibrator.py
│   ├── inspect_business.py
│   ├── make_synthetic.py
│   └── run.py
├── tests/
├── submissions/
│   └── public_test_round1.jsonl
└── docs/
    ├── DESIGN.md
    ├── REFERENCES.md
    ├── RESULTS.md
    └── figures/
        └── architecture_round1.mmd
~~~

## Limitations and validation status

- The recorded PA of 0.888 is an in-sample training result from template-and-rules mode.
- The public split has no gold labels in this repository, so its accuracy and PA cannot be reported here.
- The full Qwen3-14B GPU run and its model-specific calibration have not been validated in the documented development environment. A mock backend and a small CPU model were used only for smoke tests.
- Template coverage is incomplete by design; unsupported or ambiguous questions may be abstained on.
- ReViSQL-inspired requirement checks/reconciliation and Trace Inversion are inference-time adaptations. This repository does not reproduce ReViSQL's expert-labeling or RLVR fine-tuning pipeline.

## Documentation and research

- [Detailed design](docs/DESIGN.md) — architecture, implementation rationale, paper adaptations, and limitations.
- [Results and reproduction notes](docs/RESULTS.md) — measured outcomes, error categories, and commands.
- [Curated references](docs/REFERENCES.md) — the smaller set of papers and technical references that directly support the methods used here.
- [Default configuration](configs/default.yaml) — inference model, candidate generation, reconciliation, inversion, calibration, and time budget.

The reference list is intentionally curated. A method is cited where it informs a real design choice; the repository does not claim to reproduce any cited system end to end.
