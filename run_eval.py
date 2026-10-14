"""Aggregate the saved predictions and compare STL with MTL.

    python run_eval.py

Every test item of a fold is evaluated (original excerpts and paraphrases),
as in the paper. Writes, in reports/:

    metrics_overall.csv            precision, recall, F1, accuracy (all tasks together)
    metrics_by_task.csv            the same per task (macro and weighted)
    metrics_by_class.csv           the same per class, with support
    stl_vs_mtl.csv                 Welch per encoder (one- and two-sided)
    welch_published.csv            best STL vs best MTL, as in the paper
    stats_dependent.csv            tests for dependent folds (corrected paired t
                                   with Holm, study-level paired bootstrap)
    stats_dependent_published.csv  published Welch with the Nadeau-Bengio correction
    reproduction_verdict.csv       comparison with the published values (REPRODUCTION.md)

Means and SDs are over the five folds; "sd" is the population SD (ddof=0)
used in the published tables, "sd_sample" the ddof=1 one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import reproduction_check, stats_dependent, welch  # noqa: E402
from analysis.metrics import aggregate  # noqa: E402
from analysis.results import load_predictions, report_tables, run_metrics  # noqa: E402
from pipeline import config, io_utils, run_signature, schema  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="all", help="encoder key or 'all'")
    parser.add_argument("--metrics-root", default=None, help="default: metrics/")
    args = parser.parse_args()
    standard_root = config.metrics_dir()
    metrics_root = Path(args.metrics_root) if args.metrics_root else standard_root
    default_root = metrics_root.resolve() == standard_root.resolve()
    # Reports of another metrics root (e.g. a retraining) never overwrite the
    # published ones.
    reports_dir = config.REPORTS if default_root else metrics_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"metrics: {metrics_root}  reports: {reports_dir}")

    encoders = list(config.ENCODERS) if args.encoder == "all" else [args.encoder]
    predictions, excluded = {}, []
    for mode in ("stl", "mtl"):
        for encoder in encoders:
            expected = run_signature.current(encoder, hyperparameters=dict(config.TRAINING))
            folds, problems = load_predictions(metrics_root, mode, encoder, expected)
            excluded += problems
            if folds:
                predictions[(mode, encoder)] = folds
    for (mode, encoder), folds in sorted(predictions.items()):
        print(f"counted {mode} {encoder}: folds {sorted(folds)}")
    for problem in excluded:
        print(f"NOT COUNTED {problem}")
    if not predictions:
        raise SystemExit(f"No complete run with the current signature under {metrics_root}.")

    if default_root:
        pd.DataFrame(stats_dependent.published_corrected()).to_csv(reports_dir / "stats_dependent_published.csv", index=False)
    any_folds = next(iter(predictions.values()))
    ratios = {task: stats_dependent.test_train_ratio(any_folds, task) for task in schema.TASKS}

    fold_results = {key: run_metrics(folds) for key, folds in predictions.items()}
    summaries = {key: aggregate(results) for key, results in fold_results.items()}
    overall, by_task, by_class = report_tables(summaries)
    overall.to_csv(reports_dir / "metrics_overall.csv", index=False)
    by_task.to_csv(reports_dir / "metrics_by_task.csv", index=False)
    by_class.to_csv(reports_dir / "metrics_by_class.csv", index=False)

    f1 = {
        (mode, encoder, task): [r["tasks"][task]["f1"] for r in results if task in r["tasks"]]
        for (mode, encoder), results in fold_results.items()
        for task in schema.TASKS
    }
    per_encoder = pd.DataFrame(welch.per_encoder(f1, encoders, list(schema.TASKS)))
    published = pd.DataFrame(welch.published_form(f1, encoders, list(schema.TASKS)))
    per_encoder.to_csv(reports_dir / "stl_vs_mtl.csv", index=False)
    published.to_csv(reports_dir / "welch_published.csv", index=False)
    dependent = _dependent_tests(predictions, f1, encoders, ratios)
    dependent.to_csv(reports_dir / "stats_dependent.csv", index=False)

    reimpl = {
        key: (float(np.mean(values)) * 100, float(np.std(values)) * 100)
        for key, values in f1.items()
        if values
    }
    welch_p = {row["task"]: row["p_two_sided_published_form"] for _, row in published.iterrows()} if not published.empty else {}
    level, verdict_rows = reproduction_check.verdict(reimpl, welch_p, excluded)
    verdict_frame = pd.DataFrame(verdict_rows)
    verdict_frame["level"] = level
    verdict_frame.to_csv(reports_dir / "reproduction_verdict.csv", index=False)

    table = by_task[by_task["metric"].isin(["f1", "precision", "recall", "accuracy"])]
    if not table.empty:
        wide = table.pivot_table(index=["mode", "encoder", "task"], columns="metric", values="mean")
        print(wide.to_string(float_format=lambda v: f"{v:.4f}"))
    if not published.empty:
        print("\nWelch, published form (best STL vs best MTL, two-sided):")
        print(published.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not dependent.empty:
        print("\nDependent folds (corrected paired t, Holm; study-level paired bootstrap):")
        columns = ["comparison", "task", "stl_encoder", "mtl_encoder", "delta_fold_mean", "p_two_sided",
                   "p_two_sided_holm", "ci95_low", "ci95_high", "ci_excludes_zero"]
        print(dependent[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nComparison with the published values (REPRODUCTION.md): {level}")
    columns = ["task", "missing"] if level == "incomplete" else ["task", "a", "b", "b_rule", "c", "cells_within_tolerance"]
    print(verdict_frame[columns].to_string(index=False))
    io_utils.banner("eval", f"wrote {reports_dir}")


def _dependent_tests(predictions, f1, encoders, ratios) -> pd.DataFrame:
    """stats_dependent tests per encoder and in the published best-vs-best form."""
    tasks = list(schema.TASKS)
    pooled = {
        (mode, encoder, task): [row for fold in sorted(folds) for row in folds[fold] if row["task"] == task]
        for (mode, encoder), folds in predictions.items()
        for task in tasks
    }
    pairs = [
        ("per_encoder", task, encoder, encoder)
        for task in tasks
        for encoder in encoders
        if f1.get(("stl", encoder, task)) and f1.get(("mtl", encoder, task))
    ]
    for task in tasks:
        best = {
            mode: max((e for e in encoders if f1.get((mode, e, task))), key=lambda e: np.mean(f1[(mode, e, task)]), default=None)
            for mode in ("stl", "mtl")
        }
        if best["stl"] and best["mtl"]:
            pairs.append(("published_form", task, best["stl"], best["mtl"]))
    if not pairs:
        return pd.DataFrame()
    labels = {task: list(labels) for task, labels in schema.LABELS_BY_TASK.items()}
    return pd.DataFrame(stats_dependent.compare(pairs, f1, pooled, ratios, labels, seed=config.SEED))


if __name__ == "__main__":
    main()
