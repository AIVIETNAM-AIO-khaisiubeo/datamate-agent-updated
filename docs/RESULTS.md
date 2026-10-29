# Round-One Results — Template-and-Rules Mode

The documented development environment had a two-core CPU. The figures below therefore describe **template-and-rules mode without an LLM** (llm.backend = none). They are not results from the full Qwen3-14B configuration.

## 1. Training split

Training contains 3,000 questions from 30 businesses. The following versions were evaluated using the challenge scoring helper in scripts/evaluate.py.

| Version / change set | Correct | Wrong | Abstain | PA |
|---|---:|---:|---:|---:|
| v2 — before the general data-cleaning revision | 1,988 | 315 | 697 | 0.558 |
| v3 — sheet-name classification, total-row cleanup, channel/payment filters, catalog attributes | 2,473 | 73 | 454 | 0.800 |
| v5 — chained unit conversion, safer value linking, unresolved-constraint checks, holiday-year handling | 2,526 | 13 | 461 | 0.838 |
| v6 — product/customer groups, name-to-code aliases, customer tiers | 2,666 | 5 | 329 | 0.887 |
| v6 through run.py with the calibrator | 2,667 | 3 | 330 | **0.888** |

**Interpretation:** this is an in-sample score, not an unbiased generalization estimate. Development errors were found by inspecting incorrect training examples and then fixing general behavior in the loader, role inference, linker, or question parser. The code does not depend on question IDs or business IDs, but the training set was still used iteratively during development.

The calibrator uses cross-validation grouped by business. That reduces question-level leakage between fitting and validation folds, but it does not undo the fact that overall implementation choices were refined after reviewing the training set. When LLM candidates are enabled, calibration must be repeated using traces from that same LLM configuration.

## 2. Public split

The tracked round-one run contains 1,000 questions from 14 businesses. The repository does not include public gold labels, so public PA and accuracy cannot be calculated.

| Outcome | Questions |
|---|---:|
| Ordinary typed answers | 611 |
| Evidence-based unanswerable responses | 24 |
| Abstentions | 365 |
| **Total** | **1,000** |

The run completed in roughly 6 minutes 40 seconds on the CPU development environment (a cached rerun took about 4 minutes 10 seconds). A non-abstaining output count of 635 is **coverage only**. It must not be reported as public-test accuracy or PA.

The tracked output is [submissions/public_test_round1.jsonl](../submissions/public_test_round1.jsonl). The output was checked for 1,000 lines in input order, the expected output fields, and the abstention/unanswerable formats. Per-question logs were written under the run's logs directory.

## 3. General data errors found and fixed

The issues below were identified from training failures and addressed with general rules rather than question-specific exceptions.

| Failure pattern | Why it mattered | General correction |
|---|---|---|
| A legitimate customer name was mistaken for a total-row marker. | Real transactions were removed and revenue was understated. | Determine a total row from its label field, rather than matching a substring anywhere in the row. |
| A total row escaped filtering because its label was in a later column or another cell was missing. | Summary amounts were counted as sales. | Inspect row cells consistently and treat missing values as empty. |
| A workbook contained multiple unrelated tables but shared a generic filename. | Every sheet could be classified as the same table type. | Prefer sheet/table names to the generic filename for role classification. |
| A unit-conversion table used an unexpected column label. | Quantities were not converted to the base unit, affecting rankings. | Improve the role rules and multiply chained conversion factors. |
| A question constrained a catalog attribute but the template returned a whole-shop metric. | The numeric result could look plausible while answering a broader question. | Parse the attribute and force the template to withdraw if the constraint cannot become a filter. |
| A payment phrase was linked to a similar product string. | The wrong filter could produce an empty or unrelated result. | Use accent-aware evidence, fuzzy matching, and explicit ambiguity signals. |
| Text normalization changed a product name or valid word. | Intent and entity resolution became unreliable. | Apply Telex repair only when the result is a plausible Vietnamese syllable, and expand ambiguous abbreviations conservatively. |
| A comparison across holiday periods used the wrong year. | A valid question could be incorrectly declared unanswerable. | Resolve the year from the first explicit holiday mention and use the split calendar when available. |
| A customer tier was confused with a similarly spelled generic word. | The filter was omitted or misinterpreted. | Use table context when assigning column roles and interpreting group labels. |
| A question mentioned a staff member but the sales table stored only a staff code. | A whole-shop result could be returned. | Build name-to-code aliases from the relevant staff/customer table. |

## 4. ReViSQL-inspired reconciliation and Trace Inversion smoke tests

The repository adds two inference-time adaptations: explicit requirements/reconciliation motivated by ReViSQL [R40], and question–program alignment features inspired by Trace Inversion [R39]. These additions are not evidence that either method improves the final score in the full LLM configuration.

| Check | Recorded outcome | Interpretation |
|---|---|---|
| Template-only train run after the new features | 2,667 correct, 4 wrong, 329 abstentions in the template evaluator; grouped out-of-fold PA at threshold 0.5 was about 0.887. | The new LLM-specific features do not materially change the template-only public output. |
| Public template-only output comparison | All 1,000 output lines matched the earlier round-one file. | This was a compatibility check, not an accuracy evaluation. |
| Automated tests | 25/25 tests passed in the recorded run. Tests covered requirement creation/checking, candidate-group reconciliation, Trace Inversion modules, reward helpers, and a pipeline run with a mock LLM. | Passing tests confirms code-path behavior, not benchmark performance. |
| CPU smoke test with a quantized 1.5B model | The run completed and produced a correctly formatted submission; its SQL candidates failed on the tiny test case, so the template path supplied the answers. | This validates only the execution path, not LLM answer quality. |
| Trace Inversion comparison on one correct and one incorrect program | Lexical similarity and the structural comparison separated some signals, but the small model's LLM comparison flagged both candidates. | These features should inform calibration instead of acting as an unconditional abstention rule. |
| Requirement verdict smoke test | A row-count query for a revenue question was marked as failing the requirements. | The requirement path can identify at least this type of intent/metric mismatch. |

The effect of these additions on PA must be measured with the intended Qwen3-14B setup and a calibrator fitted on matching LLM traces. The CPU smoke tests do not establish that result.

## 5. Reproduction commands

Evaluate the template path on a labeled split:

~~~bash
python scripts/eval_templates.py --split path/to/train --out runs/train_offline_dump.jsonl
python scripts/fit_calibrator.py runs/train_offline_dump.jsonl path/to/train/answers.jsonl --out artifacts/calibrator_offline.pkl
python scripts/run.py --data path/to/public_test --out runs/public_submission.jsonl --logs runs/public_logs --set llm.backend=none --set decision.calibrator_path=artifacts/calibrator_offline.pkl
~~~

The run is configured to be deterministic with candidate seeds derived from question content rather than question ID, wall-clock time, or processing order. Before reporting results, record the configuration, inference backend, model identifier, split, calibration source, run duration, submission file, and per-question logs.

## 6. Limitations

- The train score is in-sample because errors were inspected and corrected during development.
- Public labels are not available in this repository, so no public-test PA should be claimed.
- The full Qwen3-14B GPU path and its model-specific calibration remain unvalidated here.
- The template path covers only a subset of business-question intents; unsupported questions can be abstained on.
- Research-inspired components have been tested for behavior, but their score contribution must be measured by controlled ablations on the intended model and evaluation setup.
