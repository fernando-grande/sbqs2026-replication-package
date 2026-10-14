# Replication package: Single- and Multi-Task Transformers for Classifying Controlled Experiment Protocols in SE

Replication package of the paper

> Fernando S. Grande, André F. R. Cordeiro, Edson OliveiraJr. 2026.
> *Using Single- and Multi-Task Transformers for Classifying Controlled
> Experiment Protocols in Software Engineering: Initial Investigation.*
> SBQS 2026, CTICQS track.

- Package DOI: https://doi.org/10.5281/zenodo.22760606
- Source repository: https://github.com/fernando-grande/sbqs2026-replication-package
- Paper (author version): `paper/author_version.pdf`; the version of record
  will be available in the SBQS 2026 proceedings

## What this package is

The dataset, the full pipeline (corpus selection, extractive grounding with
Gemini, LLM paraphrase augmentation, balancing, cross-validation folds), the
training code for the five encoders in the STL and MTL settings, every
prediction of every trained model, and the evaluation and statistics code.

It is an **independent re-implementation** of the study, built from the
paper, the undergraduate thesis it is based on and the source spreadsheet.
The tables of the paper are the results of the original runs.
The re-implementation reproduces the CT-Introduction results and the main
conclusion (MTL does not significantly outperform STL in any task); it gives
lower CT-Conclusion and CT-Abstract scores and higher CT-Planning scores
(Reproducibility paragraph, Section 6 of the paper). Details in
`REPRODUCTION.md`.

## Contents

| Path | Content |
|---|---|
| `README.md` | This file |
| `REPRODUCTION.md` | How the re-implementation was built, comparison with the paper, statistics, determinism, dataset hashes |
| `paper/author_version.pdf` | The paper (author version) |
| `requirements-eval.txt` | Packages for verification and evaluation (CPU only) |
| `requirements-train.txt`, `reports/train_environment.txt` | Exact environment of the training runs (GPU) |
| `requirements-dataset.txt` | Packages for rebuilding the dataset |
| `corpus/` | Source spreadsheet of the mapping study (`source.xlsx`), study catalog, the 11 excluded studies. PDFs are not distributed |
| `cache/` | Every Gemini answer (grounding and paraphrases) |
| `data/07_training/` | **The dataset**: `instances.csv` (4,298 instances: 3,218 original excerpts and their paraphrases) and `folds.csv` (5 folds grouped by study) |
| `data/extract/grounded_instances.csv` | Excerpts grounded in the PDFs, before augmentation |
| `metrics/` | Predictions of the 25 training runs (5 encoders; STL, one run per task, and MTL; 5 folds each) and a `run_info.json` per run |
| `reports/` | All metrics and statistics (below) |
| `pipeline/`, `extract/` | Dataset construction (stages 01 to 07) and grounding |
| `train/` | Training (STL, MTL with uncertainty weighting) |
| `analysis/` | Metrics, statistics, comparison with the paper |
| `run_extract.py`, `run_dataset.py`, `run_train.py`, `run_eval.py`, `verify.py` | Entry points |

## Requirements

- **Verification and evaluation (recommended):** any computer with Python
  3.10 or newer (tested with 3.12 on Linux and 3.14 on Windows), about 400 MB
  of disk including the virtual environment, no GPU.
- **Training (optional):** NVIDIA GPU with at least 8 GB (runs were made on
  an RTX 4060 8 GB), Linux, Python 3.12, CUDA 13.2 runtime.
- **Rebuilding the dataset (optional):** the full-text PDFs of the 200
  studies, which are copyrighted and not distributed (`corpus/pdfs/HOW_TO_ADD.txt`).

## Quick start: verify and recompute every result (CPU)

From the folder of this README:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements-eval.txt

python verify.py
python analysis/stats_dependent.py --selftest
python run_eval.py
```

Expected results:

1. `verify.py` prints `[PASS]` lines: every study is in exactly one test fold,
   every class has original excerpts in every test fold, the SHA-256 of
   `instances.csv` and `folds.csv` match `REPRODUCTION.md`, and all 25 runs in
   `metrics/` were trained with the code of this package and on these tables.
   One line is `[SKIP]`: the check that each paraphrase keeps the study of its
   original needs the intermediate tables of `run_dataset.py`, which are not
   versioned (it passes when the dataset is rebuilt).
2. `stats_dependent.py --selftest` prints only `PASS` lines, including the
   two sets of p-values of the paper, recomputed from the published means and
   standard deviations: Welch (Section 4.3) `0.76, 0.22, 0.36, 0.59` and with
   the Nadeau-Bengio correction (Section 3.5) `0.84, 0.40, 0.54, 0.72`.
3. `run_eval.py` recomputes every report in `reports/` from the 125
   prediction files in `metrics/`, prints the tables and the comparison with
   the paper ("partial reproduction"). The regenerated files are identical to
   the versioned ones: with git, `git diff --stat` prints nothing (`git status`
   may list the CSV files, which Python writes with CRLF line endings; their
   content is unchanged).

Main reports in `reports/`:

| File | Content |
|---|---|
| `metrics_overall.csv`, `metrics_by_task.csv`, `metrics_by_class.csv` | Accuracy and macro precision, recall and F1, mean and SD over the 5 folds, per run |
| `stl_vs_mtl.csv` | STL vs MTL per encoder (Welch) |
| `welch_published.csv` | Best STL vs best MTL per task, as in the paper |
| `stats_dependent.csv` | Corrected resampled paired t-test (Nadeau and Bengio) with Holm's adjustment and study-level paired bootstrap |
| `stats_dependent_published.csv` | The published Welch p-values with the Nadeau-Bengio correction (Section 3.5) |
| `reproduction_verdict.csv` | Comparison with the published values (criterion in `REPRODUCTION.md`) |
| `table1_comparison.csv` | Class distribution before and after augmentation vs Table 1 |
| `reporting_completeness.csv` | Share of the 200 studies that report each guideline element (Section 5) |

Every test item of a fold is evaluated, original excerpts and paraphrases,
as in the paper.

## Retraining (GPU, optional)

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements-train.txt
python verify.py
python run_train.py --mode both --encoder all --force   # the 25 runs
python run_eval.py
```

`--force` retrains runs that are already in `metrics/` (otherwise a run with
the same signature is skipped). To retrain a single run and compare it with
the published one, byte by byte, retrain in a copy of the package
(`../retrain`) and compare it with the package, from the folder of this
README:

```bash
cp -r . ../retrain
(cd ../retrain && python run_train.py --mode stl --encoder scibert --task CT-Planning --force)
python analysis/compare_runs.py metrics/stl/scibert/CT-Planning ../retrain/metrics/stl/scibert/CT-Planning
```

Training is deterministic (fixed seeds, deterministic algorithms, eager
attention): the same GPU, driver and library versions give byte-identical
predictions; other hardware or versions may give slightly different values.
`verify.py` also checks that every run in `metrics/` was trained with the
code of this package (`code_sha256` in each `run_info.json`).

## Rebuilding the dataset (optional)

```bash
pip install -r requirements-dataset.txt
python run_extract.py --llm --cache-only   # needs the PDFs in corpus/pdfs/
python run_dataset.py
python verify.py
```

The Gemini answers are in `cache/`, so no API key is needed. Without the PDFs
the extraction falls back to the spreadsheet text and the result differs from
the published dataset, which is why the dataset itself is in `data/`.

## Data, ethics and licenses

- Code: MIT (`LICENSE`). Data: CC BY 4.0 (`DATA_LICENSE.md`).
- Labels and source spreadsheet: data extraction of the systematic mapping
  study by Furtado et al., CC BY 4.0, https://doi.org/10.5281/zenodo.3893302.
- The excerpts are short verbatim quotations of 200 published papers, kept
  for research and linked to their source studies; the PDFs are not
  distributed. Paraphrases were generated with Google Gemini
  (gemini-2.5-flash). The data contains no personal or human-subject data.

## How to cite

See `CITATION.cff`.
