"""Collect saved predictions and turn them into metrics and report tables.

Everything is rebuilt from ``metrics/<mode>/<encoder>[/<task>]/fold{k}_predictions.csv``
through analysis.metrics, so the reports never depend on numbers computed
during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.metrics import fold_metrics
from pipeline import config, run_signature, schema


def section_classes() -> dict[str, list[str]]:
    return {task: list(labels) for task, labels in schema.LABELS_BY_TASK.items()}


def _fold_files(folder: Path) -> dict[int, Path]:
    files = {}
    for path in folder.glob("fold*_predictions.csv"):
        files[int(path.stem.removeprefix("fold").removesuffix("_predictions"))] = path
    return files


def _read(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype={"instance_id": str, "study_id": str, "origin": str})
    return frame.to_dict("records")


def check_run(folder: Path, expected_signature: dict | None) -> str:
    """Empty string when the run in ``folder`` can be counted, else the reason.

    Counted only if run_info.json says "completed", its signature equals the
    expected one (same model, revision, attention, determinism, checkpointing,
    hyperparameters and dataset hashes) and exactly config.N_FOLDS prediction
    files exist.
    """
    info_path = folder / "run_info.json"
    if not info_path.exists():
        return "no run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    folds = len(_fold_files(folder))
    if info.get("status") != "completed":
        return f"not completed ({folds}/{config.N_FOLDS} folds)"
    if folds != config.N_FOLDS:
        return f"{folds}/{config.N_FOLDS} folds"
    if expected_signature is not None:
        diff = run_signature.differences(info.get("signature"), expected_signature)
        if diff:
            return "different signature: " + ", ".join(diff)
    return ""


def load_predictions(
    metrics_root: Path,
    mode: str,
    encoder: str,
    expected_signature: dict | None = None,
) -> tuple[dict[int, list[dict]], list[str]]:
    """Predictions per fold, and the reasons for any run that is not counted.

    STL folds join the four task runs of that fold. An STL encoder is counted
    only when all four task runs pass ``check_run``, so the overall level is
    computed over the same items as in MTL. ``expected_signature=None`` skips
    the signature check (ad-hoc metric roots only).
    """
    base = metrics_root / mode / encoder
    if mode == "mtl":
        if not base.exists():
            return {}, [f"mtl/{encoder}: not trained"]
        reason = check_run(base, expected_signature)
        if reason:
            return {}, [f"mtl/{encoder}: {reason}"]
        return {fold: _read(path) for fold, path in _fold_files(base).items()}, []

    problems = []
    for task in schema.TASKS:
        folder = base / task
        reason = check_run(folder, expected_signature) if folder.exists() else "not trained"
        if reason:
            problems.append(f"stl/{encoder}/{task}: {reason}")
    if problems:
        return {}, problems
    per_task = {task: _fold_files(base / task) for task in schema.TASKS}
    folds = sorted(set.intersection(*(set(files) for files in per_task.values())))
    return {fold: [row for task in schema.TASKS for row in _read(per_task[task][fold])] for fold in folds}, []


def run_metrics(records_by_fold: dict[int, list[dict]]) -> list[dict]:
    results = []
    for fold, records in sorted(records_by_fold.items()):
        result = fold_metrics(records, section_classes())
        result["fold"] = fold
        results.append(result)
    return results


def _stat_rows(prefix: dict, block: dict) -> list[dict]:
    rows = []
    for metric, stats in block.items():
        if isinstance(stats, dict) and "mean" in stats:
            rows.append({**prefix, "metric": metric, **stats})
    return rows


def report_tables(summaries: dict[tuple[str, str], dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Long tables (mode, encoder, [task, label], metric, mean, sd, sd_sample, folds)."""
    overall, by_task, by_class = [], [], []
    for (mode, encoder), summary in summaries.items():
        base = {"mode": mode, "encoder": encoder}
        overall += _stat_rows(base, summary["overall"])
        for task, block in summary["tasks"].items():
            by_task += _stat_rows({**base, "task": task}, {k: v for k, v in block.items() if k != "classes"})
            for label, class_block in block["classes"].items():
                support = class_block.get("support")
                by_class += [
                    {**row, "support_all_folds": support}
                    for row in _stat_rows({**base, "task": task, "label": label}, class_block)
                ]
    return pd.DataFrame(overall), pd.DataFrame(by_task), pd.DataFrame(by_class)
