# Reproduction notes

*Using Single- and Multi-Task Transformers for Classifying Controlled Experiment
Protocols in Software Engineering: Initial Investigation* (Grande, Cordeiro,
OliveiraJr; SBQS 2026).

## Status

This package is an **independent re-implementation** of the study. It was
built from:
- the paper;
- the undergraduate thesis the paper is based on;
- the source spreadsheet;
- the data-preparation rules of the original study.

The tables of the paper are the results of the original runs; this package
does not replace them. This document records how the re-implementation was
built and how far it reproduces them.

## Provenance: original versus reconstructed

### Availability of the original materials

| Material | Available | Where |
|---|---|---|
| Source spreadsheet of the mapping study (labels, 211 studies) | Yes, unchanged | `corpus/source.xlsx`, a copy of https://doi.org/10.5281/zenodo.3893302 |
| Published results: Table 1 counts; means and SDs per encoder, task and mode; Welch p-values | Yes | The paper (`paper/author_version.pdf`) |
| Data-preparation scripts of the original study: spreadsheet extraction with its empty-cell rule; augmentation prompt and Gemini model | Not distributed. Their rules are re-implemented and cited below | Authors' private repository |
| List of the PDFs held by the original study | Used only to identify 8 of the 11 exclusions | `corpus/excluded_studies.csv` (`evidence` column) |
| Original dataset files (instances, folds), Gemini outputs of the original runs | **No** | — |
| Original training and evaluation implementation | **No** | — |
| Original per-fold scores, predictions, checkpoints, logs | **No**: the paper reports means and standard deviations only | — |

As a consequence, no original execution record (log, per-fold score,
prediction) is available to calibrate or check the re-implementation. The
only records of the original execution are the published numbers.

### Files of this package

| File | Status |
|---|---|
| `corpus/source.xlsx` | **Original** (third-party, unchanged) |
| `corpus/catalog.csv` | Derived from the spreadsheet (title, year, venue, inclusion) |
| `corpus/excluded_studies.csv` | 8 exclusions **from the original study's files** (`confirmed = yes`); 3 **reconstructed** from the Table 1 counts (`confirmed = no`; see "Corpus" below) |
| `cache/*.jsonl` | **Produced by the re-implementation** (new Gemini requests, gemini-2.5-flash), not the answers of the original runs |
| `data/extract/grounded_instances.csv` | **Reconstructed** (grounding of the re-implementation) |
| `data/07_training/instances.csv` | **Reconstructed**. The class counts after augmentation equal Table 1; the texts are not the original ones (not available) |
| `data/07_training/folds.csv` | **Reconstructed** (grouped by study, stratified by class, seed 42). The original fold assignment is not available |
| `metrics/`, `reports/` | **Produced by the re-implementation**, except the published values that `analysis/` compares against, which are copied from the paper |

### Source of each decision

| Decision | Value used here | Source |
|---|---|---|
| Corpus of 200 studies; 8 exclusions | Studies whose full text is absent from the original study's files | Original study's files |
| 3 further exclusions | Studies 151, 174, 205 | **Inferred** from the "Before" counts of Table 1 (rule below) |
| Empty-cell rule | Only `""`, `"---"`, `"nan"` are empty | Original extraction script; reproduces Impact = 8 and Parameters = 45 of Table 1 |
| 22 columns to 23 classes; Variables split into Independent and Dependent | As in the paper | Paper (Section 3.2.2) |
| Grounding model | gemini-2.5-flash | Paper (Gemini API) and original scripts (model name) |
| Span verification against the PDF text | Kept only if found verbatim; otherwise the spreadsheet text | **Added here.** The paper states that the original pipeline did not verify spans (Section 6) |
| Augmentation prompt | AugGPT prompt, verbatim | Original augmentation script |
| Number of paraphrases | At most 6 per excerpt, only as many as the balancing needs | **Inferred.** The paper generates 6 per excerpt and then caps each class; both give the counts of Table 1 |
| Balancing target | Each class capped at the largest original class of its task | Paper (Section 3.2.3) |
| Folds | 5, stratified by class, grouped by study, the same for STL and MTL | Paper (Sections 3.2.3 and 3.4). Grouping in the original runs is uncertain (Section 6 of the paper) |
| Hyperparameters | 10 epochs, batch 16, AdamW 2e-5, weight decay 0.01, warmup 10%, dropout 0.1, clipping 1.0 | Paper (Section 3.4) |
| MTL loss | Uncertainty weighting | Paper (Section 3.3) |
| Class weights | In STL only | Thesis |
| Learning rate of the uncertainty parameters | Same as the encoder | **Not specified; fixed here** |
| Maximum length | 256 tokens | **Not specified; fixed here** |
| Attention implementation | Eager | **Not specified; fixed here** (determinism) |
| Gradient checkpointing | DeBERTa only | **Not specified; fixed here** (8 GB GPU) |
| Checkpoint revisions, seeds | Pinned (`pipeline/config.py`); seed 42 + fold | **Not specified; fixed here** |
| Evaluation | Every test item of a fold, originals and paraphrases; macro F1 per task | Paper (Section 3.5) |

### Differences that affect comparability

| Aspect | Original study (paper) | This package | Effect on comparability |
|---|---|---|---|
| **Grounding** | Gemini spans used without substring verification (Section 6) | A span is kept only if found verbatim in the PDF text; otherwise the spreadsheet text is kept. 2,906 of the 3,390 original excerpts come from the PDF and 484 from the spreadsheet (`reports/span_verification.csv`) | The instance texts differ from the original ones, which are not available, so the size of the effect cannot be measured. It is concentrated in CT-Planning. Experiment Design keeps the spreadsheet text in all 141 excerpts, a closed vocabulary that is easy to classify (SciBERT STL F1 94.0 here, 67.6 published). Independent and Dependent Variables reach 84.7 and 83.3 here, against 36.0 and 52.6 published |
| **Grouping by study** | Paraphrases bound to their study's group (Section 3.2.3). Section 6 acknowledges that test folds may contain paraphrases of training excerpts | Strict: a study, its excerpts and their paraphrases are always in the same fold | Strict grouping lowers the classes with very few originals, each expanded by paraphrases: Impact (6 originals; STL F1 31.6 here, mean of 5 encoders, against 88.8 to 94.1 published) and Limitations (11 originals; 67.8 against 91.0 to 95.0). This explains most of the CT-Conclusion and part of the CT-Abstract gap. The published scores of these classes should be read as optimistic |
| **Loss weighting** | Uncertainty weighting in MTL; class weights not stated in the paper | Uncertainty weighting in MTL; class weights in STL (thesis) | Same choice for every encoder, so the STL–MTL comparison inside this package is consistent. The effect relative to the paper was not tested |
| **Preprocessing** | Empty-cell rule and marker normalisation of the original scripts; maximum length not stated | Same empty-cell rule; 256 tokens | Excerpts longer than 256 tokens are truncated, mostly in Procedure and Summary. The effect relative to the paper was not tested |
| **Augmentation** | 6 paraphrases per excerpt, then capping (Section 3.2.3) | Only the paraphrases the balancing needs (at most 6 per excerpt) | The class totals are identical, but which excerpts have paraphrases, and how many, may differ. This changes the composition of the test folds |
| **Corpus** | 200 studies, 11 exclusions (3 not identifiable) | 3 exclusions inferred from Table 1 | "Before" differs by +1 in three classes (3 cells of 3,218); after augmentation the counts are identical. Small expected effect |

Comparability is therefore highest for the conclusion of the paper (STL
versus MTL inside the same pipeline), and lower for the absolute scores of
individual classes and tasks.

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
  published (the +1 in three classes above). Like the "Total" of Table 1,
  these counts include the Variables cell once.
- Of the 3,221 cells, 7 hold no text (for example `"----"`), so 3,214 cells
  enter the dataset. The Variables cell of 185 studies is split into one
  Independent and one Dependent Variables excerpt (181 and 180, because some
  cells hold only one of them). This gives the **3,390 original excerpts** of
  `instances.csv`, plus 908 paraphrases.
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
