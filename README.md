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

The package contains:
- the dataset;
- the full pipeline: corpus selection, extractive grounding with Gemini, LLM paraphrase augmentation, balancing and cross-validation folds;
- the training code for the five encoders in the STL and MTL settings;
- every prediction of every trained model;
- the evaluation and statistics code.

It is an **independent re-implementation** of the study, built from the
paper, the undergraduate thesis it is based on, the source spreadsheet and
the data-preparation rules of the original study. The original training
implementation, dataset files, per-fold scores and predictions are not
available: the paper reports means and standard deviations only.

**The tables of the paper are the results of the original runs. This package
does not reproduce them; it re-runs the study and reports how far its results
agree.** `REPRODUCTION.md` lists which files are original and which are
reconstructed, where each decision came from, and how each difference affects
comparability.

## What each workflow verifies

Every workflow below checks the **re-implementation**. None of them
recomputes the original runs of the paper.

| # | Workflow | Hardware and inputs | What a successful run shows | What it does **not** show |
|---|---|---|---|---|
| 1 | `verify.py` | CPU | The dataset files are the published ones (SHA-256). Every study is in exactly one test fold. All 25 runs in `metrics/` were trained with the code of this package and on these tables | That the original study used these files |
| 2 | `analysis/stats_dependent.py --selftest` | CPU | The arithmetic of the published p-values. Welch (Section 4.3) and the Nadeau-Bengio correction (Section 3.5) are recomputed **from the published means and standard deviations** | Anything about the original training runs or their predictions, which are not available. The self-test checks a calculation, not a result |
| 3 | `run_eval.py` | CPU | Every metric and statistic in `reports/` is recomputed from the 125 prediction files of the re-implementation, with the same values | That the re-implementation matches the paper. The comparison with the paper is a separate output, the verdict below |
| 4 | `run_dataset.py --offline` | CPU; no PDFs, no API key | Stages 01 to 07 rebuild `instances.csv` and `folds.csv` byte-identical to the published ones. The stages are labels, cleaning, augmentation from the cached Gemini answers, balancing and folds. Input: the versioned grounded excerpts | The grounding step, which needs the PDFs (workflow 5) |
| 5 | `run_extract.py --llm --cache-only`, then workflow 4 | CPU; the 200 PDFs (not distributed) | The grounding of every excerpt in the PDF text, from the cached Gemini answers | — |
| 6 | `run_train.py` | NVIDIA GPU | The training code produces the published predictions of the re-implementation (byte-identical with the same GPU, driver and libraries) | The original runs |

**Agreement with the paper** (`reports/reproduction_verdict.csv`, criterion
fixed before training, details in `REPRODUCTION.md`): **partial reproduction**.

| Level | Result |
|---|---|
| Dataset | The class counts after augmentation are identical to Table 1 (4,298). The counts before augmentation differ by +1 in three classes. The dataset files themselves are reconstructed, not the original ones |
| Performance | CT-Introduction is within tolerance. CT-Abstract and CT-Conclusion are lower; CT-Planning is higher |
| Main conclusion | MTL does not significantly outperform STL in any task, in the paper and here |

Agreement on a **non-significant** STL–MTL comparison is narrower evidence
than reproducing the performance results. With five folds the tests have low
power, so this agreement does not show that STL and MTL are equivalent, and it
does not confirm the published scores.

## Execution scenarios

```
                        Zenodo archive or git clone
                                    |
        +---------------------------+----------------------------+
        |                           |                            |
  A. VERIFY AND EVALUATE     B. REBUILD THE DATASET       C. RETRAIN
     CPU, minutes               CPU, minutes                 NVIDIA GPU, hours
     requirements-eval.txt      requirements-dataset.txt     torch + requirements-train.txt
        |                           |                            |
     verify.py               [run_extract.py --llm           verify.py
     stats_dependent.py        --cache-only]  <- only        run_train.py
       --selftest              with the 200 PDFs             run_eval.py
     run_eval.py             run_dataset.py --offline        compare_runs.py
        |                    verify.py                           |
        v                           v                            v
  reports/ identical to      instances.csv and folds.csv   predictions byte-identical
  the versioned reports      match the published hashes    to metrics/ (same GPU,
  (workflows 1-3)            (workflows 4-5)               driver, libraries) (wf. 6)
```

Scenario A is the recommended one. Scenarios B and C are optional and
independent of each other; each starts from the archive as published.

## Contents

| Path | Content |
|---|---|
| `README.md` | This file |
| `REPRODUCTION.md` | Provenance (original versus reconstructed), decisions, comparability, comparison with the paper, statistics, determinism, dataset hashes |
| `SHA256SUMS` | SHA-256 of every file of the archive |
| `paper/author_version.pdf` | The paper (author version) |
| `requirements-eval.txt` | Packages for scenario A (CPU only) |
| `requirements-dataset.txt` | Packages for scenario B |
| `requirements-train.txt`, `reports/train_environment.txt` | Exact environment of the training runs (GPU) |
| `corpus/` | Source spreadsheet of the mapping study (`source.xlsx`, original), study catalog, the 11 excluded studies. PDFs are not distributed |
| `cache/` | Every Gemini answer of the re-implementation (grounding and paraphrases) |
| `data/07_training/` | **The dataset** (reconstructed): `instances.csv` with 4,298 instances (3,390 original excerpts and 908 paraphrases), and `folds.csv` with 5 folds grouped by study |
| `data/extract/grounded_instances.csv` | Excerpts grounded in the PDFs, before augmentation (input of scenario B) |
| `metrics/` | Predictions of the 25 training runs and a `run_info.json` per run. The runs are 5 encoders; STL is one run per task, MTL one run per encoder; each run has 5 folds |
| `reports/` | All metrics and statistics (below) |
| `pipeline/`, `extract/` | Dataset construction (stages 01 to 07) and grounding |
| `train/` | Training (STL, and MTL with uncertainty weighting) |
| `analysis/` | Metrics, statistics, comparison with the paper |
| `run_extract.py`, `run_dataset.py`, `run_train.py`, `run_eval.py`, `verify.py` | Entry points |

**Archive completeness.** The CPU scenarios A and B were run on 2026-11-04
in a fresh extraction of the Zenodo archive, in a new virtual environment with
only the packages listed in the requirements files. Every check passed and
every regenerated file was identical. This was on Windows (Python 3.14);
scenario A was also run on Linux (Python 3.12). To check that an archive is
complete and unmodified:

```bash
sha256sum -c SHA256SUMS --quiet       # Linux; macOS: shasum -a 256 -c SHA256SUMS --quiet
```

The command prints nothing when every file is present and unchanged.

## Requirements

- **Scenario A (recommended):** any computer with **Python 3.12 or newer**
  and about 400 MB of disk, including the virtual environment. No GPU. The
  pinned versions (pandas 3.0.6, numpy 2.5.2) are not available for Python
  3.11 or older.
- **Scenario B:** the same, plus `requirements-dataset.txt`. Grounding (workflow 5) also needs the full-text PDFs of the 200 studies, which are copyrighted and not distributed (`corpus/pdfs/HOW_TO_ADD.txt`).
- **Scenario C:** NVIDIA GPU with at least 8 GB (runs were made on an RTX
  4060 8 GB), Linux, Python 3.12, CUDA 13.2 runtime.

On Linux, use `python3` where the commands below say `python` if `python` is
not available.

## Scenario A: verify and recompute every result (CPU)

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

1. **`verify.py` prints `[PASS]` lines:**
   - every study is in exactly one test fold;
   - every class has original excerpts in every test fold;
   - the SHA-256 of `instances.csv` and `folds.csv` match `REPRODUCTION.md`;
   - all 25 runs in `metrics/` were trained with the code of this package and on these tables.

   One line is `[SKIP]`: the check that each paraphrase keeps the study and the fold of its original. It needs the intermediate tables of scenario B, and passes after `run_dataset.py`.
2. **`stats_dependent.py --selftest` prints only `PASS` lines.** Among them are the two sets of p-values of the paper, recomputed from the published means and standard deviations:
   - Welch (Section 4.3): `0.76, 0.22, 0.36, 0.59`;
   - with the Nadeau-Bengio correction (Section 3.5): `0.84, 0.40, 0.54, 0.72`.

   This checks the calculation of the published p-values, not the original runs.
3. **`run_eval.py` recomputes every report in `reports/`** from the 125 prediction files in `metrics/`. It prints the tables and the comparison with the paper ("partial reproduction"). The regenerated files are identical to the versioned ones. To check:
   - with git: `git diff --stat` prints nothing. `git status` may list CSV files that Python writes with CRLF line endings on Windows; their content is unchanged;
   - without git (Linux/macOS), copy the reports first and compare them after:
     ```bash
     cp -r reports /tmp/reports_published
     python run_eval.py
     diff -r --strip-trailing-cr /tmp/reports_published reports    # prints nothing
     ```

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

## Scenario B: rebuild the dataset (CPU)

**From the grounded excerpts (no PDFs, no API key):**

```bash
pip install -r requirements-dataset.txt
python run_dataset.py --offline
python verify.py
```

`run_dataset.py` runs stages 01 to 07. The inputs are:
- the spreadsheet;
- the versioned grounded excerpts (`data/extract/grounded_instances.csv`);
- the cached paraphrases (`cache/gemini_paraphrases.jsonl`).

`--offline` means that no API request is made. A paraphrase missing from the cache would become a marked placeholder; every paraphrase is cached, so none appears.

`verify.py` then prints `[PASS]` for the SHA-256 of `instances.csv` and `folds.csv`, which are rebuilt byte-identical. The paraphrase checks that were `[SKIP]` in scenario A now pass.

**With grounding (needs the 200 PDFs in `corpus/pdfs/`):**

```bash
python run_extract.py --llm --cache-only
python run_dataset.py --offline
python verify.py
```

The Gemini answers are in `cache/`, so no API key is needed. Without the PDFs,
`run_extract.py` falls back to the spreadsheet text and the result differs
from the published dataset. That is why the grounded excerpts are versioned.

## Scenario C: retrain (GPU, optional)

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements-train.txt
python verify.py
python run_train.py --mode both --encoder all --force   # the 25 runs
python run_eval.py
```

`--force` retrains runs that are already in `metrics/`; without it, a run
with the same signature is skipped.

To retrain a single run and compare it, byte by byte, with the published
one:
1. Retrain in a copy of the package (`../retrain`).
2. Compare the copy with the package.

From the folder of this README:

```bash
cp -r . ../retrain
(cd ../retrain && python run_train.py --mode stl --encoder scibert --task CT-Planning --force)
python analysis/compare_runs.py metrics/stl/scibert/CT-Planning ../retrain/metrics/stl/scibert/CT-Planning
```

Training is deterministic: fixed seeds, deterministic algorithms and eager
attention. The same GPU, driver and library versions give byte-identical
predictions; other hardware or versions may give slightly different values.
`verify.py` also checks that every run in `metrics/` was trained with the
code of this package (`code_sha256` in each `run_info.json`).

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
