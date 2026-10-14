# Reproduction notes

*Using Single- and Multi-Task Transformers for Classifying Controlled Experiment
Protocols in Software Engineering: Initial Investigation* (Grande, Cordeiro,
OliveiraJr; SBQS 2026).

## Status

This package is an **independent re-implementation** of the study, built
from the paper, the undergraduate thesis it is based on and the source
spreadsheet. The tables of the paper are the results of the original runs;
this package does not replace them. This document records how the
re-implementation was built and how far it reproduces them.

## Corpus: 200 studies

The mapping study lists 211 studies. The paper excludes 11 ("could not be
downloaded or successfully processed"), but only 8 are identifiable. The other
3 were reconstructed from the "Before" column of Table 1, which describes the
input corpus, never from any metric:

1. Populated cells are counted with the empty-cell rule that reproduces
   Table 1 (only `""`, `"---"` and `"nan"` are empty; `"----"` counts), which
   gives Impact = 8 and Parameters = 45 of Table 1.
2. Excluding studies alone cannot reach Table 1 exactly: with all 211
   studies, Methods exceeds the published count by 12 cells, and 11 studies
   remove at most 11.
3. Keeping the 8 identifiable exclusions, 293 trios of the remaining studies
   match every class except +1 in Background, Methods and Problem Statement.
4. Rule, fixed before any training: keep the trio with the most studies whose
   PDF text could not be processed (no grounded span anchors); ties go to the
   smallest trio. Result: studies **151, 174, 205**.

The 11 exclusions are in `corpus/excluded_studies.csv`; all 293 trios and the
summary in `reports/exclusion_search.{csv,txt}`.

Resulting corpus (`reports/table1_comparison.csv`):

- Before augmentation: 3,221 cells by the original rule versus 3,218
  published (the +1 in three classes above); 3,218 original excerpts enter the
  dataset, because cells such as `"----"` hold no text.
- After augmentation and balancing: **4,298 instances, identical to Table 1 in
  every class** (Limitations 77 = 11 x 7; Impact 42 = 6 x 7).

## Pipeline

| Step | Code | Notes |
|---|---|---|
| Grounding | `run_extract.py`, `extract/` | gemini-2.5-flash locates the literal excerpt for each spreadsheet value in the PDF text; a span is used only if it is found verbatim in the PDF text, otherwise the spreadsheet text is kept (`reports/span_verification.csv`). All answers are in `cache/gemini_grounding.jsonl` |
| Labels | `pipeline/stage01` to `stage03`, `pipeline/schema.py` | 22 spreadsheet columns to 23 classes in 4 tasks (Variables split into Independent and Dependent) |
| Augmentation | `pipeline/stage04` to `stage06` | AugGPT prompt; up to 6 paraphrases per excerpt, only as many as the balancing needs; label and study of the original kept. All answers are in `cache/gemini_paraphrases.jsonl` |
| Folds | `pipeline/stage07`, `data/07_training/folds.csv` | 5 folds, grouped by study and stratified by class; a study, its excerpts and their paraphrases are in the same test fold; the same folds for STL and MTL |
| Training | `run_train.py`, `train/` | BERT, SciBERT, DistilBERT, RoBERTa, DeBERTa; STL (one model per task) and MTL (shared encoder, four heads, uncertainty weighting); 10 epochs, batch 16, AdamW 2e-5, weight decay 0.01, warmup 10%, dropout 0.1, gradient clipping 1.0 |
| Evaluation | `run_eval.py`, `analysis/` | Every test item of a fold (originals and paraphrases), as in the paper; macro F1 per task is the primary metric |

Not specified in the paper and fixed here: class weights only in STL (as in
the thesis), maximum length 256 tokens, the same learning rate for the
uncertainty-weighting parameters, "eager" attention, gradient checkpointing
for DeBERTa only (a fold does not fit an 8 GB GPU without it).

## Comparison with the paper

Criterion fixed before training (`analysis/reproduction_check.py`, output
`reports/reproduction_verdict.csv`), tolerance T = max(2 pp, published SD):
(a) the best encoder of each task and mode is within T of the published best;
(b) the difference best MTL - best STL keeps its direction, or stays a tie
(all published differences are ties, below 2 pp); (c) Welch in the published
form (best STL vs best MTL, two-sided) stays non-significant.

Best Macro-F1 (%):

| Task | Mode | Paper | Re-implementation |
|---|---|---|---|
| CT-Abstract | STL | 81.48 | 78.46 |
| CT-Abstract | MTL | 81.79 | 78.66 |
| CT-Introduction | STL | 89.84 | 88.81 |
| CT-Introduction | MTL | 88.80 | 88.79 |
| CT-Planning | STL | 65.95 | 79.08 |
| CT-Planning | MTL | 64.91 | 79.41 |
| CT-Conclusion | STL | 98.31 | 74.57 |
| CT-Conclusion | MTL | 97.89 | 76.61 |

- Result: **partial reproduction**. (b) and (c) pass in every task except (b)
  in CT-Conclusion; (a) passes only in CT-Introduction.
- **Main conclusion reproduced:** MTL does not significantly outperform STL in
  any task, under Welch and under every test below.
- **CT-Conclusion** is below the paper because of Impact (6 original
  excerpts, one per study; F1 about 32 here, 88.8 to 94.1 published) and, in
  CT-Abstract, Limitations (about 68 here, 91.0 to 95.0 published). These
  classes have very few original excerpts, each expanded with paraphrases, so
  their scores depend strongly on how the paraphrases are distributed across
  folds. With folds grouped by study, the model never sees, in training, a
  paraphrase of a test excerpt; the published scores of these classes should
  be read as optimistic (Section 6 of the paper).
- **CT-Planning** is above the paper, mostly in the Independent and Dependent
  Variables classes; this difference is reported as it is.

## Statistics for dependent folds

Fold scores are not independent: training sets overlap, and STL and MTL are
evaluated on the same folds and items. `analysis/stats_dependent.py` (called
by `run_eval.py`; self-test `python analysis/stats_dependent.py --selftest`)
adds:

- the corrected resampled paired t-test (Nadeau and Bengio, 2003) on the
  per-fold differences MTL - STL, variance (1/k + n_test/n_train) s^2,
  df = k - 1, with Holm's adjustment over the 20 encoder x task pairs;
- a paired bootstrap of the pooled out-of-fold predictions that resamples
  whole studies (2,000 resamples, seed 42), 95% percentile interval of the
  difference in macro F1;
- both tests also in the published form (best STL vs best MTL);
- the published Welch test recomputed with the Nadeau-Bengio variance
  correction (`reports/stats_dependent_published.csv`): 0.76 to 0.84,
  0.22 to 0.40, 0.36 to 0.54, 0.59 to 0.72 (Section 3.5 of the paper). The paper
  reports means and standard deviations, not per-fold scores, so this is the
  only correction applicable to the published numbers.

Results (`reports/stats_dependent.csv`):

- corrected t per encoder, with Holm: 0/20 significant (lowest adjusted p 0.19);
- bootstrap intervals per encoder excluding 0: 0/20;
- best vs best: no task significant; every interval contains 0.

No evidence of positive (or negative) transfer. With five folds the power is
low, so this does not show that STL and MTL are equivalent.

## Reporting completeness of the corpus

`analysis/reporting_completeness.py` counts, for each element of the
guideline, the studies with at least one original excerpt
(`reports/reporting_completeness.csv`). Lowest: Impact 6/200 (3.0%),
Limitations in the abstract 11/200 (5.5%), Parameters 44/200 (22%),
Conclusions in the abstract 70/200 (35%), Analysis Procedure 82/200 (41%),
Hypotheses 84/200 (42%), Tasks 96/200 (48%) (Section 5 of the paper).

## Ablation without augmentation

Not performed; future work (Section 7 of the paper).

## Determinism and environment

- Every fold starts from seed 42 + fold (Python, numpy, torch, CUDA and the
  DataLoader shuffling); cuDNN deterministic; `torch.use_deterministic_algorithms(True)`
  enforced; "eager" attention for every encoder.
- Hugging Face checkpoints pinned to a revision (`pipeline/config.py`).
- Every `run_info.json` records the signature of the run (model, revision,
  hyperparameters, seed, hashes of `instances.csv` and `folds.csv`), the
  environment (Python, torch, transformers, numpy, pandas, CUDA, cuDNN, GPU),
  the per-epoch losses and the peak GPU memory of each fold.
- Every `run_info.json` also records `code_sha256`, a fingerprint of the
  code of this package (every `.py` file and the `requirements-*.txt` files;
  `pipeline/io_utils.code_sha256`). `verify.py` recomputes it and checks that
  all runs in `metrics/` were trained with this code and on these tables.
- Environment of the runs: Python 3.12.3, Linux, torch 2.13.0+cu132,
  transformers 5.17.0, numpy 2.5.2, pandas 3.0.6, CUDA runtime 13.2, NVIDIA
  GeForce RTX 4060 8 GB. Pinned in `requirements-train.txt`; full
  `pip freeze` in `reports/train_environment.txt`.
- The same GPU, driver and library versions give byte-identical predictions;
  other hardware or versions may give slightly different values.

## Dataset hashes (content with LF line endings)

| File | SHA-256 |
|---|---|
| `data/07_training/instances.csv` | `664e1fed540bc24f62795c8e7f3ca56e446eceff80660932c28575bf41cf2655` |
| `data/07_training/folds.csv` | `19a29d882ec900c2cb5df6888df9491b5ff55a05e14a642c266499e1baa485f5` |

`verify.py` checks them; every `run_info.json` stores the hashes of the
tables it was trained on.
