"""Paths and settings of the replication package.

The corpus is the 200 studies of the paper (Table 1): the 211 studies of the
mapping minus the 11 listed in corpus/excluded_studies.csv.
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKBOOK = PACKAGE_ROOT / "corpus" / "source.xlsx"
SOURCE_SHEET = "Extracted Data"
EXCLUDED_STUDIES = PACKAGE_ROOT / "corpus" / "excluded_studies.csv"
PDF_DIR = PACKAGE_ROOT / "corpus" / "pdfs"

DATA = PACKAGE_ROOT / "data"
CORPUS_DIR = DATA / "01_corpus"
EXTRACTED_DIR = DATA / "02_extracted"
CLEAN_DIR = DATA / "03_clean"
AUGMENTED_DIR = DATA / "04_augmented"
CLEAN_SYNTHETIC_DIR = DATA / "05_clean_synthetic"
FINAL_DIR = DATA / "06_final"
TRAINING_DIR = DATA / "07_training"
PDF_TEXT_DIR = DATA / "extract" / "texts"
PDF_FIELDS_DIR = DATA / "extract" / "fields"
# Stage 02 instances with the text replaced by the span grounded in the PDF.
GROUNDED_INSTANCES = DATA / "extract" / "grounded_instances.csv"

CACHE_DIR = PACKAGE_ROOT / "cache"
AUGMENTATION_CACHE = CACHE_DIR / "gemini_paraphrases.jsonl"
GROUNDING_CACHE = CACHE_DIR / "gemini_grounding.jsonl"

REPORTS = PACKAGE_ROOT / "reports"
METRICS_DIR = PACKAGE_ROOT / "metrics"
METRICS_SMOKE_DIR = PACKAGE_ROOT / "metrics_smoke"

def metrics_dir(smoke: bool = False):
    return METRICS_SMOKE_DIR if smoke else METRICS_DIR

SEED = 42
MIN_TEXT_LENGTH = 15
PARAPHRASES_PER_EXCERPT = 6
AUGMENTATION_BATCH_SIZE = 10
GEMINI_MODEL = "gemini-2.5-flash"
# The same model served through OpenRouter. Cached answers are keyed by
# GEMINI_MODEL, so both routes share one cache.
OPENROUTER_MODEL = "google/gemini-2.5-flash"
# "auto": OpenRouter when OPENROUTER_API_KEY is set, the Google API otherwise.
GROUNDING_PROVIDER = "auto"
# Bump when the grounding prompt changes, so cached answers are not reused.
GROUNDING_PROMPT_VERSION = "v1"
# Seconds between grounding calls (the free tier allows ~10 requests/minute).
GROUNDING_SLEEP = 7
GROUNDING_MAX_ATTEMPTS = 5
# Twice the longest valid answer seen (~10k tokens); a looping answer stops here.
GROUNDING_MAX_OUTPUT_TOKENS = 20000
N_FOLDS = 5
# "Before" column of the per-class table in the SBQS 2026 paper (200 studies).
# The paper reports Variables as one class (187) before splitting it.
PAPER_BEFORE = {
    "Background": 196,
    "Objective": 198,
    "Methods": 192,
    "Results": 154,
    "Limitations": 11,
    "Conclusions": 71,
    "ProblemStatement": 197,
    "ResearchObjectives": 197,
    "Context": 200,
    "Goals": 192,
    "ExperimentalUnits": 196,
    "ExperimentalMaterial": 199,
    "Tasks": 96,
    "Hypotheses": 84,
    "Parameters": 45,
    "Variables": 187,
    "ExperimentDesign": 141,
    "Procedure": 197,
    "AnalysisProcedure": 83,
    "Summary": 200,
    "Impact": 8,
    "FutureWork": 174,
}
PAPER_BEFORE_TOTAL = 3218

# "After" column of the same table (after augmentation and balancing).
PAPER_AFTER = {
    "Background": 198, "Objective": 198, "Methods": 198, "Results": 198, "Limitations": 77, "Conclusions": 198,
    "ProblemStatement": 200, "ResearchObjectives": 200, "Context": 200,
    "Goals": 199, "ExperimentalUnits": 199, "ExperimentalMaterial": 199, "Tasks": 199, "Hypotheses": 199,
    "Parameters": 199, "IndependentVariables": 199, "DependentVariables": 199, "ExperimentDesign": 199,
    "Procedure": 199, "AnalysisProcedure": 199,
    "Summary": 200, "Impact": 42, "FutureWork": 200,
}
PAPER_AFTER_TOTAL = 4298

# Hugging Face checkpoints pinned to a commit, so a later update of a model
# repository cannot change what is downloaded. Names are the canonical ones
# (the short aliases such as "bert-base-uncased" redirect to them).
# gradient_checkpointing: off everywhere by default. It is switched on for a
# model only after a memory diagnosis shows that one fold
# does not fit the GPU, and then stays fixed for every run of that model.
ENCODERS = {
    "bert": {"name": "google-bert/bert-base-uncased", "revision": "86b5e0934494bd15c9632b12f734a8a67f723594", "gradient_checkpointing": False},
    "scibert": {"name": "allenai/scibert_scivocab_uncased", "revision": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1", "gradient_checkpointing": False},
    "distilbert": {"name": "distilbert/distilbert-base-uncased", "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be", "gradient_checkpointing": False},
    "roberta": {"name": "FacebookAI/roberta-base", "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b", "gradient_checkpointing": False},
    # Diagnosis on an RTX 4060 8 GB, eager attention, batch 16 x 256 tokens,
    # fp32: out of memory already in fold 1 of STL CT-Abstract (7.19 GiB
    # allocated by PyTorch), so checkpointing is on for DeBERTa only, for
    # every run (REPRODUCTION.md).
    "deberta": {"name": "microsoft/deberta-base", "revision": "0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195", "gradient_checkpointing": True},
}

# Same attention implementation for every encoder: "eager" is deterministic on
# GPU (SDPA's memory-efficient backward is not) and DeBERTa v1 has no SDPA.
ATTN_IMPLEMENTATION = "eager"

TRAINING = {
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "dropout": 0.1,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "max_grad_norm": 1.0,
    "max_length": 256,
}


def ensure_dirs() -> None:
    for directory in (
        CORPUS_DIR,
        EXTRACTED_DIR,
        CLEAN_DIR,
        AUGMENTED_DIR,
        CLEAN_SYNTHETIC_DIR,
        FINAL_DIR,
        TRAINING_DIR,
        PDF_TEXT_DIR,
        PDF_FIELDS_DIR,
        CACHE_DIR,
        REPORTS,
        PDF_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
