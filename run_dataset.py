"""Build the dataset (stages 01 to 07) from the spreadsheet, the grounded
excerpts (run_extract.py) and the cached paraphrases.

    python run_extract.py --llm --cache-only
    python run_dataset.py
"""
from pipeline import (
    stage01_select_corpus,
    stage02_extract_labels,
    stage03_clean,
    stage04_augment,
    stage05_clean_synthetic,
    stage06_balance,
    stage07_build_training_sets,
)
import argparse

from analysis import table1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run augmentation without the API, using marked placeholder text",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="enable deduplication and the minimum-length filter in stage 03",
    )
    parser.add_argument(
        "--text-source",
        choices=("auto", "pdf", "sheet"),
        default="auto",
        help="instance text: PDF spans from run_extract.py --llm, or the spreadsheet "
        "(default auto: PDF spans when data/extract/grounded_instances.csv exists)",
    )
    args = parser.parse_args()

    stage01_select_corpus.run()
    print()
    stage02_extract_labels.run()
    print()
    stage03_clean.run(
        dedupe=args.strict,
        min_length=15 if args.strict else 0,
        text_source=args.text_source,
    )
    print()
    stage04_augment.run(offline=args.offline)
    print()
    stage05_clean_synthetic.run()
    print()
    stage06_balance.run()
    print()
    stage07_build_training_sets.run()
    print()
    table1.run()


if __name__ == "__main__":
    main()
