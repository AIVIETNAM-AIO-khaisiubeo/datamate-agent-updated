# References

Mã `[R#]` được dùng trong `docs/DESIGN.md` và docstring của code.

## Text-to-SQL
- **[R1] BIRD** — J. Li et al. *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs.* NeurIPS 2023 (D&B). arXiv:2305.03111
- **[R2] Spider 2.0** — F. Lei et al. *Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows.* ICLR 2025. arXiv:2411.07763
- **[R3] DIN-SQL** — M. Pourreza, D. Rafiei. *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction.* NeurIPS 2023. arXiv:2304.11015
- **[R4] Alpha-SQL** — B. Li, J. Zhang, J. Fan, Y. Xu, C. Chen, N. Tang, Y. Luo. *Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search.* ICML 2025. arXiv:2502.17248
- **[R5] XiYan-SQL** — Y. Gao et al. *XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL.* arXiv:2411.08599
- **[R6] RESDSQL** — H. Li, J. Zhang, C. Li, H. Chen. *RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL.* AAAI 2023. arXiv:2302.05965
- **[R7] CHESS** — S. Talaei, M. Pourreza, Y.-C. Chang, A. Mirhoseini, A. Saberi. *CHESS: Contextual Harnessing for Efficient SQL Synthesis.* arXiv:2405.16755
- **[R8] CHASE-SQL** — M. Pourreza et al. *CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL.* ICLR 2025. arXiv:2410.01943
- **[R9] OmniSQL** — H. Li et al. *OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale.* PVLDB 18(11), 2025. arXiv:2503.02240
- **[R10] Arctic-Text2SQL-R1** — Z. Yao, G. Sun, Ł. Borchmann, Z. Shen, M. Deng, B. Zhai, H. Zhang, A. Li, Y. He. *Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL.* Findings of ACL 2026. arXiv:2505.20315
- DAIL-SQL — D. Gao et al. *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation.* PVLDB 17(5), 2024. arXiv:2308.15363
- MAC-SQL — B. Wang et al. *MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL.* COLING 2025. arXiv:2312.11242

## Reasoning, agents, table QA
- **[R11] Self-Consistency** — X. Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. arXiv:2203.11171
- **[R12] ReAct** — S. Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023. arXiv:2210.03629
- **[R13] CodeAct** — X. Wang et al. *Executable Code Actions Elicit Better LLM Agents.* ICML 2024. arXiv:2402.01030
- **[R14] Chain-of-Table** — Z. Wang et al. *Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding.* ICLR 2024. arXiv:2401.04398
- **[R15] InfiAgent-DABench** — X. Hu et al. *InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks.* ICML 2024. arXiv:2401.05507
- **[R16] Execution-guided decoding** — C. Wang et al. *Robust Text-to-SQL Generation with Execution-Guided Decoding.* arXiv:1807.03100
- **[R17] Self-Debugging** — X. Chen, M. Lin, N. Schärli, D. Zhou. *Teaching Large Language Models to Self-Debug.* ICLR 2024. arXiv:2304.05128

## Reliability, unanswerable questions, abstention
- **[R18] EHRSQL 2024 shared task** — G. Lee, S. Kweon, S. Bae, E. Choi. *Overview of the EHRSQL 2024 Shared Task on Reliable Text-to-SQL Modeling on Electronic Health Records.* ClinicalNLP 2024. arXiv:2405.06673
- **[R19] TrustSQL** — G. Lee, W. Chay, S. Cho, E. Choi. *TrustSQL: A Reliability Benchmark for Text-to-SQL Models with Diverse Unanswerable Questions.* arXiv:2403.15879
- **[R20] PLUQ (1st place EHRSQL 2024)** — Y. Jo, S. Lee, M. Seo, S. J. Hwang, M. Lee. *LG AI Research & KAIST at EHRSQL 2024: Self-Training Large Language Models with Pseudo-Labeled Unanswerable Questions for a Reliable Text-to-SQL System on EHRs.* ClinicalNLP 2024. arXiv:2405.11162
- **[R21] ProbGate (3rd place EHRSQL 2024)** — S. Kim, D. Han, S. Kim. *ProbGate at EHRSQL 2024: Enhancing SQL Query Generation Accuracy through Probabilistic Threshold Filtering and Error Handling.* ClinicalNLP 2024. arXiv:2404.16659
- **[R22] PRACTIQ** — M. Dong, N. Ashok Kumar et al. *PRACTIQ: A Practical Conversational Text-to-SQL dataset with Ambiguous and Unanswerable Queries.* NAACL 2025. arXiv:2410.11076
- **[R23] Semantic entropy** — S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 625–630 (2024)
- **[R24] Semantic uncertainty** — L. Kuhn, Y. Gal, S. Farquhar. *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023. arXiv:2302.09664
- **[R25] AmbiQT** — A. Bhaskar et al. *Benchmarking and Improving Text-to-SQL Generation under Ambiguity.* EMNLP 2023. arXiv:2310.13659
- **[R26] Conformal abstention** — Y. Abbasi-Yadkori, I. Kuzborskij et al. *Mitigating LLM Hallucinations via Conformal Abstention.* arXiv:2405.01563
- Abstention survey — B. Wen, J. Yao, S. Feng, C. Xu, Y. Tsvetkov, B. Howe, L. L. Wang. *Know Your Limits: A Survey of Abstention in Large Language Models.* TACL 2025. arXiv:2407.18418
- P(True) — S. Kadavath et al. *Language Models (Mostly) Know What They Know.* arXiv:2207.05221

## Decision theory & calibration
- **[R27]** C. K. Chow. *On optimum recognition error and reject tradeoff.* IEEE Trans. Information Theory 16(1):41–46, 1970
- **[R28]** Y. Geifman, R. El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017. arXiv:1705.08500
- **[R29]** C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. arXiv:1706.04599
- **[R30]** J. Platt. *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.* 1999
- **[R31]** B. Zadrozny, C. Elkan. *Transforming Classifier Scores into Accurate Multiclass Probability Estimates.* KDD 2002

## Systems & models
- **[R32] vLLM** — W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180
- **[R33] SGLang** — L. Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024. arXiv:2312.07104
- **[R34] DuckDB** — M. Raasveldt, H. Mühleisen. *DuckDB: an Embeddable Analytical Database.* SIGMOD 2019 (demo)
- **[R35] Qwen3** — Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388. Qwen2.5-Coder: B. Hui et al., arXiv:2409.12186

## Vietnamese language & calendar
- **[R36] ViText2SQL** — A. T. Nguyen, M. H. Dao, D. Q. Nguyen. *A Pilot Study of Text-to-SQL Semantic Parsing for Vietnamese.* Findings of EMNLP 2020
- **[R37]** Hồ Ngọc Đức. *Thuật toán tính âm lịch* (Âm lịch Việt Nam), 2004 — reference implementation used by Vietnamese calendar software
- **[R38]** J. Meeus. *Astronomical Algorithms*, 2nd ed., Willmann-Bell, 1998

## Added in the round-1 revision (methods suggested by the team)
- **[R39] Trace Inversion** — A. Gourabathina, I. Padhi, M. Nagireddy, S. Chaudhury, P. Sattigeri. *Answering the Wrong Question: Reasoning Trace Inversion for Abstention in LLMs.* Proceedings of ACL 2026 (Long Papers), pp. 13307–13324. Code: https://github.com/abinithago/trace-inversion
- **[R40] ReViSQL** — Y. Zhu, T. Jin, Y. Choi, D. Kang. *ReViSQL: Achieving Human-Level Text-to-SQL.* arXiv:2603.20004 (2026), PVLDB. Code and BIRD-Verified data: https://github.com/uiuc-kang-lab/ReViSQL
- **[R41] CISPO** (RL objective used by ReViSQL) — MiniMax. *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention.* arXiv:2506.13585

