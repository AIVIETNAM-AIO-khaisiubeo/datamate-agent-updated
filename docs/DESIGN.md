# DataMate Agent — System Design and Method Rationale

This document describes the implementation in this repository, the research ideas that influenced it, and the limitations of those adaptations. It is scoped to the **round-one DataMate inference pipeline**; it does not describe a deployment-image submission workflow.

Related documents: [README](../README.md), [Results](RESULTS.md), [Curated references](REFERENCES.md), [Architecture source](figures/architecture_round1.mmd).

## 1. Design scope and evidence

DataMate answers Vietnamese business questions over small-shop data stored in CSV, Excel, and SQLite. Each question is associated with a business, an as-of date, and the rulebook for its split. Business folders may include a README with column definitions and operational conventions.

The design addresses five practical properties of the task:

| Observed property | Design consequence |
|---|---|
| Tables and workbooks vary in headers, sheet organization, date formats, numeric formats, and summary rows. | Normalize data before query generation; keep raw tables available for fallback reasoning. |
| Column names and shop documentation are inconsistent. | Infer table and column roles using README descriptions, header names, and observed value types. |
| The rulebook can differ across splits. | Load the active rulebook for each split and disable templates when a required rule is absent. |
| Questions contain colloquial Vietnamese, abbreviations, typing errors, entity names, and relative date expressions. | Normalize text conservatively, resolve time ranges explicitly, and link mentioned values to the business data. |
| Incorrect answers are penalized while abstention is free. | Treat answerability, candidate agreement, execution quality, requirement coverage, and calibration as first-class features. |

In the recorded training split, common answer types include numbers, percentages, lists, dates, strings, booleans, tables, and unanswerable questions. Many questions target sales metrics such as revenue, order counts, average order value, period changes, top-k entities, shares, and quantities. These observations guide template coverage; they do not guarantee generalization to unseen businesses.

## 2. Decision rule under Penalized Accuracy

The challenge score for an answer is +1 when correct and −1 when incorrect; abstention scores 0. Let p be the estimated probability that a candidate is correct.

- Expected score if the system answers: p(+1) + (1 − p)(−1) = 2p − 1.
- Expected score if it abstains: 0.

The simple reject-option rule is therefore:

**Answer when p > 0.5; otherwise abstain.**

This is the reject rule associated with Chow [R27] and relates to selective classification [R28]. An unanswerable response is still an answer, not an abstention, so it is subject to the same score rule. The pipeline estimates p from candidate consistency, cross-family agreement, execution and formatting signals, constraint coverage, answerability evidence, and question–program alignment.

## 3. Architecture

The complete round-one flow is rendered in the [README architecture section](../README.md#architecture). The editable Mermaid source is [architecture_round1.mmd](figures/architecture_round1.mmd).

The implementation has six logical layers:

1. **Inputs:** question record, business data, shop README, split rulebook, and optional calendar.
2. **Data preparation:** parse source files, infer schema roles, build canonical business views in DuckDB, and record consistency warnings.
3. **Question understanding:** normalize Vietnamese text, infer intent and constraints, resolve periods, link entities and catalog attributes, collect answerability evidence, and create requirements R1…Rn.
4. **Candidate generation:** run deterministic templates and, when enabled, LLM SQL samples followed by plan-to-SQL, Python/pandas, ReAct, or repair escalation.
5. **Execution and reconciliation:** run programs with limits, normalize typed results, group equivalent outputs, check constraints, and select a candidate.
6. **Reliability and output:** build alignment and agreement features, estimate correctness probability, apply the PA decision rule, and write submission/log records.

Business-data context can be reused across questions for the same business. Question understanding is per question. Template-only mode skips LLM generation and Trace Inversion but retains template evidence and the decision rule.

## 4. Data preparation and semantic layer

### 4.1 Loading and normalization

Implementation: **datamate/loader.py**. The loader discovers CSV, Excel, and SQLite inputs and prepares them as DuckDB tables. It addresses messy exports, including:

- title rows above real headers, multi-row headers, repeated header rows, and footnotes;
- day or monthly totals mixed with transaction rows;
- multiple monthly sheets or files that share a schema;
- numeric values stored as currency strings or quantities with units;
- mixed date formats, Excel serial dates, and years inferred from the source sheet or filename;
- table names where the sheet name is more informative than the workbook filename.

Normalization should be conservative. A summary-row detector must examine the row label rather than discard a row merely because a customer name resembles a total label. The loader caches parsed tables and keeps source context available for diagnostics.

### 4.2 Shop documentation and schema roles

Implementation: **datamate/readme.py** and **datamate/roles.py**.

The system uses column descriptions from the shop README first, normalized column names second, and observed value patterns third. A table is classified using sheet/table names before generic filenames. This context helps distinguish fields with generic names such as “name” by the type of table in which they occur.

The linking approach is intentionally practical for small per-shop schemas: compare question spans with distinct observed values, handle aliases, and preserve ambiguity signals. It is not a reproduction of CHESS's retrieval architecture; CHESS [R7] and AmbiQT [R25] are references for schema/value selection and ambiguity handling.

### 4.3 Canonical business views

Implementation: **datamate/salesview.py**.

The semantic layer retains raw tables and builds canonical views where the source data supports them:

- **v_sales:** line-level sales with normalized dates, status, quantities, amounts, and linked entities.
- **v_orders:** order-level metrics and validity fields.
- **v_products:** catalog information.
- **v_purchases:** purchase/receipt information when supported by the source schema.

Business conventions are applied centrally, rather than reimplemented in each query candidate. Examples include excluding invalid or returned orders where the active rulebook requires it, computing revenue from the correct line fields, converting quantities to base units, inheriting repeated order-level fields where the export format omits them on continuation rows, and deduplicating exact duplicate export rows.

Consistency checks compare quantities and unit prices against line amounts and compare line totals with recorded totals when both are present. Warnings and lineage become confidence features and can be shown to the LLM. The check is diagnostic evidence; it is not a guarantee that the source data is correct.

### 4.4 Split-specific rulebook and calendar

Implementation: **datamate/rulebook.py**, **datamate/timeresolver.py**, and **datamate/lunar.py**.

The active rulebook is supplied to the LLM and interpreted for deterministic settings such as week boundaries, time-of-day categories, and required business metrics. Configuration overrides are fallback settings, not a replacement for the split's rulebook. Each template declares the rules it implements; if a required clause is missing, the template can be disabled.

Relative time phrases are resolved into explicit half-open intervals of the form [start, end). Supported patterns include yesterday, recent-day windows, this or last week/month, quarters, explicit date intervals, comparisons with prior periods, and holiday-related periods. An official calendar file takes precedence for solar–lunar mappings. The built-in fallback follows Hồ Ngọc Đức's Vietnamese lunar-calendar algorithm [R37] and the astronomical calculations described by Meeus [R38].

## 5. Question understanding and explicit requirements

### 5.1 Normalization and analysis

Implementation: **datamate/textnorm.py** and **datamate/question.py**.

The parser extracts the likely intent, answer type, top-k, numeric thresholds, period phrases, comparison periods, and filters such as product/customer groups, payment methods, channels, staff, shifts, and catalog attributes. Text normalization handles Vietnamese diacritics, Telex typing errors, and selected abbreviations conservatively. It avoids rewriting a name or technical term unless the correction has sufficient linguistic evidence.

### 5.2 Entity and attribute linking

Implementation: **datamate/linker.py** and **datamate/attributes.py**.

The linker compares candidate spans with values present in the business data and can use fuzzy similarity and known aliases. It records unresolved or ambiguous matches. Catalog attributes, where available, can become explicit filters rather than being left for the LLM to infer from a generic name.

A deterministic template must not silently ignore a meaningful constraint. If a question names a specific entity or attribute and the template cannot turn it into a reliable filter, the template should withdraw instead of answering a broader question.

### 5.3 Answerability evidence and requirements

Implementation: **datamate/answerability.py** and **datamate/requirements.py**.

The system collects positive evidence that a question may be unanswerable: for example, the requested period is entirely outside the available data, a required field is not recorded, a named entity is absent, or the question requests a subjective judgment that the tables cannot support. A partially overlapping time period is not automatically treated as unanswerable.

The requirement builder turns already-resolved question information into explicit R1…Rn constraints. Requirements can cover time bounds, valid-order rules, entity and attribute filters, thresholds, ranking, shifts, weekdays, and the requested answer type. These requirements support both candidate prompting and post-execution reconciliation.

## 6. Candidate generation, execution, and reconciliation

### 6.1 Deterministic templates

Implementation: **datamate/templates.py**.

Templates are small, auditable programs for supported patterns such as revenue, order count, average order value, change rates, best periods, top-k products or customers, group shares, and selected purchase metrics. Each template has a strength signal. Unrecognized question tokens and unresolved constraints reduce confidence or cause the template to withdraw.

Templates are an independent candidate family. Agreement between a template and an independently generated LLM candidate is useful evidence, but agreement alone does not establish correctness.

### 6.2 LLM candidate families

Implementation: **datamate/agent.py**, **datamate/prompts.py**, and **datamate/executor.py**.

When an LLM endpoint is enabled, the first round samples direct SQL candidates with configured temperatures. If early agreement is insufficient, the agent can escalate to:

- plan-to-SQL candidates, inspired by decomposition and execution-plan reasoning such as CHASE-SQL [R8];
- Python/pandas programs, an independent data-analysis path related to CodeAct [R13];
- ReAct-style data exploration [R12];
- execution-feedback repair, related to execution-guided decoding [R16] and Self-Debugging [R17];
- a short judge call when the leading answer groups are close.

Multiple candidates and answer voting are inspired by self-consistency [R11], but the implementation votes on **executed, normalized answers** rather than raw natural-language responses. Candidate families have different failure modes, so cross-family agreement is informative, not conclusive.

### 6.3 Bounded execution and reconciliation

SQL is executed read-only against DuckDB with time limits. Python candidates run in a separate process with a timeout and result-size limits. The formatter normalizes result types and captures failures before answer groups are compared.

Reconciliation is adapted from ReViSQL [R40]'s inference-time approach. DataMate checks many explicit requirements directly against candidate-program text and may use a short LLM verdict for remaining candidate groups. It then selects among the surviving groups, with a weighted vote and an optional tie-break judge.

This is an **inference-time adaptation only**. The repository does not reproduce ReViSQL's expert-curated training data or RLVR fine-tuning, and it does not claim ReViSQL's benchmark results.

A generated unanswerable declaration during an execution-repair attempt is treated as a failed repair rather than an independent vote. This avoids allowing a candidate to bypass debugging simply by declaring the task impossible.

## 7. Question–program alignment and calibration

### 7.1 Trace Inversion adaptation

Implementation: **datamate/inversion.py**.

Trace Inversion [R39] frames some hallucinations as the model answering a different question from the one asked: reconstruct the question implied by the model's trace and compare it with the original. DataMate adapts this idea to executable data programs:

- For an LLM candidate, it uses the winning program with comments removed and a preview of the executed result as a compact proxy for the model's reasoning trace.
- It asks the model to reconstruct a concise question/specification from that evidence.
- It compares the original question and reconstructed question using normalized character n-gram similarity by default, with an optional local multilingual sentence encoder.
- A deterministic STRUCT comparison checks periods, filters, and answer type when those fields can be recovered.
- For a template candidate, the system uses the template's explicit plan and checks whether applied filters are grounded in the question.

These are **features for the calibrator**, not an unconditional hard-abstention rule. The data-agent adaptation differs from the paper's original setup, which operates on model reasoning traces; the repository's behavior should not be described as an exact reproduction. The default similarity score is lexical similarity, not a semantic-entropy estimate.

### 7.2 Correctness calibration

Implementation: **datamate/calibrate.py** and **scripts/fit_calibrator.py**.

The calibrator estimates the probability that a proposed answer is correct. Features include candidate agreement and margin, cross-family agreement, template strength, execution failures and empty results, requirement coverage, value-linking ambiguity, semantic-view consistency checks, answerability evidence, and question–program alignment.

The calibration script supports shop-grouped cross-validation so questions from one business are not split between fitting and validation folds. Isotonic correction is enabled only when the labeled sample count is large enough under the configured minimum. When LLM generation is enabled, fit the calibrator on traces produced with the same model and configuration; a template-only calibrator should not be assumed calibrated for the LLM candidate distribution.

## 8. Research choices: what is used and what is not

The project borrows ideas at the component level. It does not implement the cited papers end to end.

| DataMate subsystem | Key inspiration | Actual adaptation and boundary |
|---|---|---|
| Schema/value linking and ambiguity | CHESS [R7], AmbiQT [R25] | Per-shop role inference and value matching; no reproduction of CHESS's complete retrieval pipeline. |
| Diverse SQL generation and selection | CHASE-SQL [R8], self-consistency [R11] | Multiple candidate families, execution-result grouping, weighted voting, and a lightweight judge; not CHASE-SQL's full pairwise selection architecture. |
| Tool use and repair | ReAct [R12], CodeAct [R13], execution-guided decoding [R16], Self-Debugging [R17] | Bounded DuckDB/Python execution and limited repair attempts using execution results. |
| Answerability and abstention | TrustSQL [R19], PRACTIQ [R22] | Positive evidence for unsupported or unanswerable questions, combined with candidate-quality signals. |
| Reject option and calibration | Chow [R27], selective classification [R28], Guo et al. [R29] | The PA-optimal p > 0.5 baseline, then a learned probability estimate from validation traces. |
| Requirements and reconciliation | ReViSQL [R40] | Explicit requirement construction and execution-based candidate-group filtering at inference time; no RLVR fine-tuning. |
| Query misalignment | Trace Inversion [R39] | Program/result-based reconstruction plus deterministic structural checks, with results passed to calibration rather than directly forcing abstention. |
| Runtime and data execution | Qwen3 [R35], vLLM [R32], DuckDB [R34] | Model, serving, and analytical execution dependencies—not evidence that the full configured model has been benchmarked here. |

## 9. Results, reproducibility, and limitations

See [RESULTS.md](RESULTS.md) for the exact recorded counts and commands. The key limitations are:

- The reported PA of 0.888 is in-sample and was measured with template-and-rules mode.
- The public split has no ground-truth labels in the repository; its answer coverage is not public accuracy or PA.
- A complete GPU run with Qwen3-14B and a calibrator fitted on matching LLM traces has not been validated in the documented development environment.
- The template-only path has incomplete coverage for questions outside the implemented patterns. Unsupported or ambiguous cases may be abstained on.
- Trace Inversion and ReViSQL-inspired components have been smoke-tested, but their effect on PA requires an evaluation with the intended model, split, and calibration procedure.

For the first-round workflow, use the commands in the [README](../README.md#first-round-workflow). The curated bibliography in [REFERENCES.md](REFERENCES.md) keeps the works directly relevant to implemented methods and the few runtime components that need technical attribution.
