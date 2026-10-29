# DataMate Agent

**Zalo AI Challenge 2026 — DataMate**  
*A calibrated Vietnamese natural-language data agent for small-business spreadsheets and databases.*

DataMate answers everyday Vietnamese questions over business data stored in CSV, Excel, or SQLite. It combines deterministic data preparation and query templates with LLM-generated programs, execution-based verification, answer reconciliation, and a calibrated abstention decision. Its central design principle is **answer only when the available evidence supports the answer**.

> **Objective:** maximise Penalized Accuracy (PA), where a correct answer scores +1, abstention scores 0, and a wrong answer scores −1. A confident-looking but unsupported answer is therefore worse than abstaining.

Example input:

```json
{
  "id": "q_06de33fa21cc",
  "business_id": "biz_118",
  "question": "sp bán dc nhiều nhất tuần rồi là gì",
  "as_of_date": "2026-03-04"
}
```

Example answer:

```json
{
  "id": "q_06de33fa21cc",
  "type": "string",
  "answer": "Xi măng Hà Tiên PCB40 bao 50kg"
}
```

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How a question is answered](#how-a-question-is-answered)
- [Research foundations](#research-foundations)
- [Results and evaluation caveats](#results-and-evaluation-caveats)
- [Getting started](#getting-started)
- [Run inference](#run-inference)
- [Docker submission](#docker-submission)
- [Repository layout](#repository-layout)
- [Model and runtime constraints](#model-and-runtime-constraints)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [References](#references)

## Overview

### Design goals

- **Evidence-first answering:** answers are based on programs that are executed against the shop's data, not on free-form text generation alone.
- **Correct business semantics:** a rulebook-aware semantic layer normalises tables and applies business rules consistently before analysis.
- **Multiple solution paths:** deterministic templates, SQL, plan-then-SQL, pandas, and ReAct-style exploration can independently produce candidate answers.
- **Execution-based verification:** candidate programs are executed and grouped by their resulting answers; requirements are checked before voting.
- **Calibrated abstention:** a probability calibrator estimates answer correctness, and the decision rule reflects the competition's +1 / 0 / −1 scoring.
- **Reproducible and resource-aware inference:** fixed seeds, response caching, structured JSONL output, per-question logs, an offline fallback, and model-budget checks.

### Supported inputs and output

The runner consumes a split containing question metadata, a rulebook, an optional lunar calendar, and one or more data files per business. Data files can be CSV, Excel workbooks, or SQLite databases. The expected competition input structure is:

```text
<split>/
├── questions.jsonl
├── rulebook.md
├── calendar.csv
└── data/
    └── <business_id>/
        ├── ... CSV / XLSX / SQLite files ...
        └── README.md                 # optional column descriptions
```

The runner emits one JSON object per input question in `submission.jsonl`. Detailed execution traces can be written to a separate log directory.

## Architecture


At a high level, a question travels through six stages:

| Stage | Main components | Responsibility |
|---|---|---|
| 1. Ingest | `loader.py`, `readme.py`, `roles.py` | Load heterogeneous files into DuckDB, interpret column documentation, infer column roles, and normalise values. |
| 2. Semantic layer | `salesview.py`, `rulebook.py`, `attributes.py` | Build standard views such as `v_sales`, `v_orders`, `v_products`, and `v_purchases`; apply rulebook semantics and run consistency checks. |
| 3. Understand the question | `textnorm.py`, `question.py`, `timeresolver.py`, `linker.py`, `answerability.py`, `requirements.py` | Normalise Vietnamese text, extract intent and constraints, resolve time ranges, link mentioned entities to data, identify unanswerable questions, and build explicit requirements. |
| 4. Generate candidates | `templates.py`, `prompts.py`, `agent.py` | Produce deterministic candidates first, then generate SQL and escalate to alternative reasoning/programming paths when needed. |
| 5. Execute and reconcile | `executor.py`, `formatter.py`, `requirements.py` | Execute candidates, standardise outputs, group equivalent answers, check requirement coverage, and vote among supported candidates. |
| 6. Estimate reliability | `inversion.py`, `calibrate.py`, `scorer.py` | Inspect whether the winning program appears to answer the original question, calibrate correctness probability, and answer or abstain. |


## Research foundations

DataMate adapts ideas from prior work rather than reproducing any paper's complete system. The most relevant connections are:

| Method in DataMate | Research connection | How it is used |
|---|---|---|
| Schema and entity linking | RESDSQL [R6], CHESS [R7], CHASE-SQL [R8], AmbiQT [R25] | Interpret column roles and map Vietnamese mentions to values in the shop's actual data. |
| Decomposition and candidate diversity | DIN-SQL [R3], CHASE-SQL [R8], XiYan-SQL [R5], Self-Consistency [R11] | Use structured query understanding and multiple independently generated candidates. |
| Tool use and executable programs | ReAct [R12], CodeAct [R13], execution-guided decoding [R16], Self-Debugging [R17] | Use execution outcomes and errors to validate or repair candidate programs. |
| Candidate reconciliation | ReViSQL [R40], Self-Consistency [R11] | Check explicit requirements on candidate programs, then vote on answers derived from execution. This is an inference-time adaptation, not the ReViSQL training method. |
| Abstention and reliability | Chow [R27], selective classification [R28], TrustSQL [R19], EHRSQL [R18], conformal abstention [R26] | Treat abstention as a first-class action and account for unanswerable or ambiguous questions. |
| Probability calibration | Guo et al. [R29], Platt [R30], Zadrozny and Elkan [R31] | Calibrate correctness features with logistic/Platt-style and isotonic methods. |
| Trace Inversion | Gourabathina et al. [R39] | Adapt query reconstruction to the winning executable program and its result, with a structural consistency signal suited to this task. |
| Fast model serving and analytics | vLLM/PagedAttention [R32], DuckDB [R34], Qwen3 [R35] | Serve the chosen open-weight model and execute analytical queries locally. |

For full design decisions, what was adapted, and why alternatives were not used, see [`docs/DESIGN.md`](docs/DESIGN.md). The full paper bibliography is included below and maintained separately in [`docs/REFERENCES.md`](docs/REFERENCES.md).

## Results and evaluation caveats

The figures below describe the **template + rules configuration without an LLM**, which is the CPU-compatible fallback. They must not be interpreted as the measured performance of the full Qwen3-14B pipeline.

### Training split

| Split / configuration | Correct | Wrong | Abstain | Penalized Accuracy |
|---|---:|---:|---:|---:|
| Train: 3,000 questions, 30 shops; official scorer, in-sample | 2,667 | 3 | 330 | **0.888** |

The training result is **in-sample**: failure cases were inspected during development and used to improve general-purpose data handling, linking, and query parsing. It is not an unbiased estimate of held-out performance. The calibrator is designed to use cross-validation grouped by shop, but the overall implementation has still been iteratively developed on the training set.

### Public test split

| Outcome | Questions |
|---|---:|
| Ordinary answers | 611 |
| Returned as `unanswerable` | 24 |
| Abstained | 365 |
| **Total** | **1,000** |

The public test questions have no available gold labels in this repository, so a public-test PA score **cannot be reported**. This run produced 635 answers, counting `unanswerable` as an answer, and abstained on 365 questions. It ran in approximately 6 minutes 40 seconds on the development CPU environment for the documented run.

The tracked round-one output is [`submissions/public_test_round1.jsonl`](submissions/public_test_round1.jsonl). Full tables, known error categories, and reproduction commands are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Getting started

### Requirements

- Python **3.10 or newer**.
- A local environment capable of installing the packages in `requirements.txt`.
- GPU(s) and a compatible CUDA/vLLM environment only for the full LLM configuration. Unit tests and the template-only path do not require a GPU.

### Install and test

```bash
python -m pip install -r requirements.txt
make test
```

The test command runs the unit and pipeline tests under `tests/`. The repository also includes synthetic-data utilities for local development.

## Run inference

The following commands assume the competition split has the structure shown above. Replace placeholder paths with the actual local split paths.

### Option A — CPU-friendly template-only mode

This mode does not call an LLM. It is useful for local smoke tests and as a fast fallback.

```bash
python scripts/run.py --data path/to/public_test --out submission.jsonl --logs runs/public_logs --set llm.backend=none --set decision.calibrator_path=artifacts/calibrator_offline.pkl
```

### Option B — full LLM mode with Qwen3-14B

Serve Qwen3-14B through an OpenAI-compatible vLLM endpoint using two GPUs:

```bash
vllm serve Qwen/Qwen3-14B --tensor-parallel-size 2 --enable-prefix-caching --max-model-len 32768 --seed 0
```

In a second terminal, first run the agent on the labelled training split and dump candidate features, then fit the calibrator on that run:

```bash
python scripts/run.py --data path/to/train --out runs/train_sub.jsonl --dump runs/train_dump.jsonl
python scripts/fit_calibrator.py runs/train_dump.jsonl path/to/train/answers.jsonl --out artifacts/calibrator.pkl
```

Then run the public split:

```bash
python scripts/run.py --data path/to/public_test --out submission.jsonl --logs runs/public_logs
```

**Important:** when enabling LLM inference, fit the calibrator using a labelled run generated by the same LLM configuration. Do not assume that a calibrator fitted on template-only candidates remains calibrated for the LLM candidate distribution.

### Evaluate a labelled split

```bash
python scripts/eval_templates.py --split path/to/train --out runs/train_offline_dump.jsonl
python scripts/evaluate.py submission_train.jsonl path/to/train/answers.jsonl
```

See `python scripts/run.py --help` and the configuration in [`configs/default.yaml`](configs/default.yaml) for additional options.

## Docker submission

The repository includes a competition entry point and a Dockerfile template. The entry point reads `/data`, writes `/result/submission.jsonl`, and saves per-question logs under `/result/logs/`.

Before building the image, prepare the local model directory expected by the Dockerfile (by default, `models/Qwen3-14B`). The weights need to be included in the image because inference is configured to run offline. Use the base image/CUDA version required by the competition organisers; the checked-in Dockerfile uses a vLLM OpenAI image as a starting point, and its compatibility with the final organiser environment must be verified.

For a template-only run without starting vLLM, set `NO_LLM=1`. See [`Dockerfile`](Dockerfile) and [`predict.sh`](predict.sh) before building the final submission image.

## Repository layout

```text
.
├── datamate/              # Agent implementation
│   ├── agent.py           # End-to-end orchestration
│   ├── loader.py          # CSV / Excel / SQLite ingestion and DuckDB loading
│   ├── readme.py          # Column dictionary and business-data conventions
│   ├── roles.py           # Column roles and table classification
│   ├── salesview.py       # Canonical sales/order/product/purchase views
│   ├── rulebook.py        # Split-specific business rules and rule fingerprints
│   ├── textnorm.py        # Vietnamese text normalisation
│   ├── question.py        # Intent, answer type, filters, thresholds and top-k
│   ├── timeresolver.py    # Relative expressions to explicit date intervals
│   ├── linker.py          # Question-to-data entity linking
│   ├── answerability.py   # Evidence for unanswerable questions
│   ├── requirements.py    # Requirements and candidate-program checks
│   ├── templates.py       # Deterministic query templates
│   ├── prompts.py         # SQL / plan / pandas / ReAct / repair prompts
│   ├── executor.py        # Read-only SQL and isolated Python execution
│   ├── formatter.py       # Output normalisation and candidate grouping
│   ├── inversion.py       # Trace and template inversion features
│   ├── calibrate.py       # Correctness calibration and decision threshold
│   └── scorer.py          # Official scoring logic
├── configs/
│   ├── default.yaml       # Main runtime configuration
│   └── rulebook_overrides.yaml
├── scripts/               # Runner, evaluation, calibration, data export and utilities
├── tests/                 # Unit and pipeline tests
├── docs/
│   ├── DESIGN.md          # Detailed architecture and method-by-method rationale
│   ├── REFERENCES.md      # Standalone reference list
│   ├── RESULTS.md         # Evaluation details and known limitations
│   └── figures/           # Pipeline source and rendered diagram
├── submissions/           # Tracked round-one submission
├── predict.sh             # Docker/competition entry point
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## Model and runtime constraints

| Component | Role | Notes |
|---|---|---|
| `Qwen/Qwen3-14B` | Main LLM for SQL/program generation, ReAct, candidate judgement, verdict reconciliation, and query reconstruction | Declared as 14.8B parameters; Apache-2.0 model licence per project configuration. |
| `intfloat/multilingual-e5-small` | Optional embedding-based signal for Trace Inversion | Disabled by default; the default character 3-gram signal does not require another model. If enabled, its parameters must be included in the total model budget. |
| vLLM | Local OpenAI-compatible inference server | Two-GPU tensor parallelism is used in the documented full configuration. |
| DuckDB | Analytical execution engine | Queries use the local semantic layer; SQL execution is read-only and has a timeout. |

The competition configuration declares a total inference-time model budget of at most 15B parameters. The optional encoder is therefore disabled by default when Qwen3-14B uses the main budget. The runtime configuration also includes a time budget and a template-only fast fallback. Check the current competition rules and hardware environment before making a submission.

## Limitations

- **Full-model performance is not yet measured in the documented development environment.** The reported scores are from template + rules on CPU. The LLM execution path has been smoke-tested with a mock LLM and a small local model, but the full Qwen3-14B GPU run and its final calibration must be evaluated separately.
- **Training performance is in-sample.** It should not be presented as an unbiased generalisation score.
- **Public-test ground truth is unavailable here.** Public answer coverage is reported, but public PA and accuracy cannot be calculated.
- **Template coverage is intentionally incomplete.** In template-only mode, questions outside the implemented sales and purchasing intents may be abstained on. The LLM path is intended to handle additional business-specific questions.
- **Deployment depends on the organiser's runtime.** The Dockerfile is a starting point, not a guarantee of compatibility with a particular GPU/CUDA base image.

## Documentation

- [**Design and method rationale**](docs/DESIGN.md) — architecture, design decisions, the connection to each cited method, adaptations, ablations, and limitations.
- [**Detailed evaluation**](docs/RESULTS.md) — split counts, measured results, known errors, and reproduction commands.
- [**Reference list**](docs/REFERENCES.md) — citation keys used in code comments and design documentation.
- [**Pipeline source**](docs/figures/pipeline_v2.mmd) — Mermaid source for the architecture figure.
- [**Configuration**](configs/default.yaml) — model serving, candidate generation, reconciliation, inversion, calibration, and time budget.

---

## References

The citation keys below are used in the design documentation and are included here so readers can trace the technical foundations directly from the README. Titles link to arXiv or publisher pages when a stable public link is available. These are bibliographic references, not copied passages from the papers.

### Text-to-SQL and schema linking

- **[R1] BIRD.** J. Li et al. “Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database-Grounded Text-to-SQLs.” NeurIPS 2023 (Datasets and Benchmarks). [arXiv:2305.03111](https://arxiv.org/abs/2305.03111).
- **[R2] Spider 2.0.** F. Lei et al. “Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows.” ICLR 2025. [arXiv:2411.07763](https://arxiv.org/abs/2411.07763).
- **[R3] DIN-SQL.** M. Pourreza and D. Rafiei. “DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction.” NeurIPS 2023. [arXiv:2304.11015](https://arxiv.org/abs/2304.11015).
- **[R4] Alpha-SQL.** B. Li, J. Zhang, J. Fan, Y. Xu, C. Chen, N. Tang, and Y. Luo. “Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search.” ICML 2025. [arXiv:2502.17248](https://arxiv.org/abs/2502.17248).
- **[R5] XiYan-SQL.** Y. Gao et al. “XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL.” [arXiv:2411.08599](https://arxiv.org/abs/2411.08599).
- **[R6] RESDSQL.** H. Li, J. Zhang, C. Li, and H. Chen. “RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL.” AAAI 2023. [arXiv:2302.05965](https://arxiv.org/abs/2302.05965).
- **[R7] CHESS.** S. Talaei, M. Pourreza, Y.-C. Chang, A. Mirhoseini, and A. Saberi. “CHESS: Contextual Harnessing for Efficient SQL Synthesis.” [arXiv:2405.16755](https://arxiv.org/abs/2405.16755).
- **[R8] CHASE-SQL.** M. Pourreza et al. “CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL.” ICLR 2025. [arXiv:2410.01943](https://arxiv.org/abs/2410.01943).
- **[R9] OmniSQL.** H. Li et al. “OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale.” PVLDB 18(11), 2025. [arXiv:2503.02240](https://arxiv.org/abs/2503.02240).
- **[R10] Arctic-Text2SQL-R1.** Z. Yao, G. Sun, Ł. Borchmann, Z. Shen, M. Deng, B. Zhai, H. Zhang, A. Li, and Y. He. “Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL.” Findings of ACL 2026. [arXiv:2505.20315](https://arxiv.org/abs/2505.20315).
- DAIL-SQL. D. Gao et al. “Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation.” PVLDB 17(5), 2024. [arXiv:2308.15363](https://arxiv.org/abs/2308.15363).
- MAC-SQL. B. Wang et al. “MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL.” COLING 2025. [arXiv:2312.11242](https://arxiv.org/abs/2312.11242).

### Reasoning, agents, and table question answering

- **[R11] Self-Consistency.** X. Wang et al. “Self-Consistency Improves Chain of Thought Reasoning in Language Models.” ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).
- **[R12] ReAct.** S. Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models.” ICLR 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
- **[R13] CodeAct.** X. Wang et al. “Executable Code Actions Elicit Better LLM Agents.” ICML 2024. [arXiv:2402.01030](https://arxiv.org/abs/2402.01030).
- **[R14] Chain-of-Table.** Z. Wang et al. “Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding.” ICLR 2024. [arXiv:2401.04398](https://arxiv.org/abs/2401.04398).
- **[R15] InfiAgent-DABench.** X. Hu et al. “InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks.” ICML 2024. [arXiv:2401.05507](https://arxiv.org/abs/2401.05507).
- **[R16] Execution-guided decoding.** C. Wang et al. “Robust Text-to-SQL Generation with Execution-Guided Decoding.” [arXiv:1807.03100](https://arxiv.org/abs/1807.03100).
- **[R17] Self-Debugging.** X. Chen, M. Lin, N. Schärli, and D. Zhou. “Teaching Large Language Models to Self-Debug.” ICLR 2024. [arXiv:2304.05128](https://arxiv.org/abs/2304.05128).

### Reliability, ambiguity, and abstention

- **[R18] EHRSQL 2024 shared task.** G. Lee, S. Kweon, S. Bae, and E. Choi. “Overview of the EHRSQL 2024 Shared Task on Reliable Text-to-SQL Modeling on Electronic Health Records.” ClinicalNLP 2024. [arXiv:2405.06673](https://arxiv.org/abs/2405.06673).
- **[R19] TrustSQL.** G. Lee, W. Chay, S. Cho, and E. Choi. “TrustSQL: A Reliability Benchmark for Text-to-SQL Models with Diverse Unanswerable Questions.” [arXiv:2403.15879](https://arxiv.org/abs/2403.15879).
- **[R20] PLUQ (1st place EHRSQL 2024).** Y. Jo, S. Lee, M. Seo, S. J. Hwang, and M. Lee. “LG AI Research & KAIST at EHRSQL 2024: Self-Training Large Language Models with Pseudo-Labeled Unanswerable Questions for a Reliable Text-to-SQL System on EHRs.” ClinicalNLP 2024. [arXiv:2405.11162](https://arxiv.org/abs/2405.11162).
- **[R21] ProbGate (3rd place EHRSQL 2024).** S. Kim, D. Han, and S. Kim. “ProbGate at EHRSQL 2024: Enhancing SQL Query Generation Accuracy through Probabilistic Threshold Filtering and Error Handling.” [arXiv:2404.16659](https://arxiv.org/abs/2404.16659).
- **[R22] PRACTIQ.** M. Dong, N. Ashok Kumar et al. “PRACTIQ: A Practical Conversational Text-to-SQL Dataset with Ambiguous and Unanswerable Queries.” NAACL 2025. [arXiv:2410.11076](https://arxiv.org/abs/2410.11076).
- **[R23] Semantic entropy.** S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal. “Detecting hallucinations in large language models using semantic entropy.” *Nature* 630, 625–630 (2024).
- **[R24] Semantic uncertainty.** L. Kuhn, Y. Gal, and S. Farquhar. “Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.” ICLR 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664).
- **[R25] AmbiQT.** A. Bhaskar et al. “Benchmarking and Improving Text-to-SQL Generation under Ambiguity.” EMNLP 2023. [arXiv:2310.13659](https://arxiv.org/abs/2310.13659).
- **[R26] Conformal abstention.** Y. Abbasi-Yadkori, I. Kuzborskij et al. “Mitigating LLM Hallucinations via Conformal Abstention.” [arXiv:2405.01563](https://arxiv.org/abs/2405.01563).
- Abstention survey. B. Wen, J. Yao, S. Feng, C. Xu, Y. Tsvetkov, B. Howe, and L. L. Wang. “Know Your Limits: A Survey of Abstention in Large Language Models.” TACL 2025. [arXiv:2407.18418](https://arxiv.org/abs/2407.18418).
- P(True). S. Kadavath et al. “Language Models (Mostly) Know What They Know.” [arXiv:2207.05221](https://arxiv.org/abs/2207.05221).

### Decision theory and calibration

- **[R27] Chow's reject rule.** C. K. Chow. “On optimum recognition error and reject tradeoff.” *IEEE Transactions on Information Theory*, 16(1):41–46, 1970.
- **[R28] Selective classification.** Y. Geifman and R. El-Yaniv. “Selective Classification for Deep Neural Networks.” NeurIPS 2017. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500).
- **[R29] Neural-network calibration.** C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. “On Calibration of Modern Neural Networks.” ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- **[R30] Platt scaling.** J. Platt. “Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.” 1999.
- **[R31] Multiclass probability calibration.** B. Zadrozny and C. Elkan. “Transforming Classifier Scores into Accurate Multiclass Probability Estimates.” KDD 2002.

### Systems and foundation models

- **[R32] vLLM / PagedAttention.** W. Kwon et al. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- **[R33] SGLang.** L. Zheng et al. “SGLang: Efficient Execution of Structured Language Model Programs.” NeurIPS 2024. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104).
- **[R34] DuckDB.** M. Raasveldt and H. Mühleisen. “DuckDB: an Embeddable Analytical Database.” SIGMOD 2019 (demo).
- **[R35] Qwen3.** Qwen Team. “Qwen3 Technical Report.” [arXiv:2505.09388](https://arxiv.org/abs/2505.09388). Related coding model reference: B. Hui et al., “Qwen2.5-Coder Technical Report,” [arXiv:2409.12186](https://arxiv.org/abs/2409.12186).

### Vietnamese language and calendar handling

- **[R36] ViText2SQL.** A. T. Nguyen, M. H. Dao, and D. Q. Nguyen. “A Pilot Study of Text-to-SQL Semantic Parsing for Vietnamese.” Findings of EMNLP 2020.
- **[R37] Vietnamese lunar calendar.** Hồ Ngọc Đức. “Thuật toán tính âm lịch” (*Âm lịch Việt Nam*), 2004. Used as a reference for the fallback lunar-calendar implementation.
- **[R38] Astronomical algorithms.** J. Meeus. *Astronomical Algorithms*, 2nd edition. Willmann-Bell, 1998.

### Additional methods in the round-one revision

- **[R39] Trace Inversion.** A. Gourabathina, I. Padhi, M. Nagireddy, S. Chaudhury, and P. Sattigeri. “Answering the Wrong Question: Reasoning Trace Inversion for Abstention in LLMs.” *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026), Long Papers*, pp. 13307–13324. [ACL Anthology](https://aclanthology.org/2026.acl-long.608/) · [arXiv:2604.02230](https://arxiv.org/abs/2604.02230).
- **[R40] ReViSQL.** Y. Zhu, T. Jin, Y. Choi, and D. Kang. “ReViSQL: Achieving Human-Level Text-to-SQL.” [arXiv:2603.20004](https://arxiv.org/abs/2603.20004). [Project code and BIRD-Verified data](https://github.com/uiuc-kang-lab/ReViSQL).
- **[R41] CISPO / MiniMax-M1.** MiniMax. “MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention.” [arXiv:2506.13585](https://arxiv.org/abs/2506.13585).

---

**Citation note:** bracketed identifiers such as `[R40]` connect implementation choices to the bibliography; see [`docs/DESIGN.md`](docs/DESIGN.md) for the detailed method-by-method discussion. Paper titles and bibliographic metadata are provided for attribution; this README does not reproduce the full text of copyrighted papers.
