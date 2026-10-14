"""Evaluation metrics, computed from saved predictions.

One implementation is shared by training (run_train.py) and evaluation
(run_eval.py), so every number in the reports can be
regenerated from ``metrics/.../fold{k}_predictions.csv`` without retraining.
Only numpy is used, so results do not depend on the scikit-learn version.

Levels (precision, recall, F1 and accuracy at each):

* class   - one-vs-rest inside its task: precision, recall, F1, support and
            accuracy (TP + TN) / N;
* task    - macro precision/recall/F1 (the primary metric of the paper is the
            task macro-F1), weighted precision/recall/F1, and accuracy
            (micro: correct / all, as in the paper), plus the confusion matrix;
* overall - all four tasks together, classes identified by (task, label) so
            that ids of different tasks never mix: accuracy over every test
            item, macro precision/recall/F1 over the classes, and weighted.

Macro averages cover the classes that have at least one test item in the fold
(support > 0); a class absent from a fold is listed in ``classes_absent`` and
does not enter the average as a zero. Undefined precision (no prediction of a
class) counts as 0.

Every test item of a fold is evaluated, original excerpts and paraphrases
alike, as in the paper.
"""
from __future__ import annotations

import math

import numpy as np

METRIC_KEYS = ("precision", "recall", "f1", "accuracy")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _class_table(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Confusion matrix and one-vs-rest metrics for one label space."""
    position = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    confusion = np.zeros((n, n), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        confusion[position[true], position[pred]] += 1
    total = int(confusion.sum())
    classes = {}
    for label, i in position.items():
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum()) - tp
        fn = int(confusion[i, :].sum()) - tp
        tn = total - tp - fp - fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        classes[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": _safe_div(tp + tn, total),
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return {"classes": classes, "confusion": confusion.tolist(), "labels": labels, "total": total}


def _averages(classes: dict, correct: int, total: int) -> dict:
    present = [name for name, c in classes.items() if c["support"] > 0]
    support = sum(classes[name]["support"] for name in present)
    out = {"accuracy": _safe_div(correct, total), "n": total}
    for key in ("precision", "recall", "f1"):
        out[key] = float(np.mean([classes[name][key] for name in present])) if present else 0.0
        out[f"weighted_{key}"] = (
            _safe_div(sum(classes[name][key] * classes[name]["support"] for name in present), support)
        )
    out["classes_present"] = present
    out["classes_absent"] = [name for name in classes if name not in present]
    return out


def fold_metrics(records: list[dict], section_classes: dict[str, list[str]]) -> dict:
    """All levels for one fold. ``records`` need task, true_name, pred_name."""
    tasks = {}
    overall_true, overall_pred = [], []
    for task, labels in section_classes.items():
        subset = [row for row in records if row["task"] == task]
        if not subset:
            continue
        y_true = [row["true_name"] for row in subset]
        y_pred = [row["pred_name"] for row in subset]
        table = _class_table(y_true, y_pred, list(labels))
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        tasks[task] = {**_averages(table["classes"], correct, len(subset)), **table}
        overall_true += [f"{task}|{label}" for label in y_true]
        overall_pred += [f"{task}|{label}" for label in y_pred]

    overall = {}
    if overall_true:
        space = [f"{task}|{label}" for task in tasks for label in section_classes[task]]
        table = _class_table(overall_true, overall_pred, space)
        correct = sum(t == p for t, p in zip(overall_true, overall_pred))
        overall = _averages(table["classes"], correct, len(overall_true))
    return {"overall": overall, "tasks": tasks}


def _stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {"mean": None, "sd": None, "sd_sample": None, "folds": 0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        # Population SD (ddof=0) is what the published tables report.
        "sd": float(array.std(ddof=0)),
        "sd_sample": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "folds": len(values),
    }


def aggregate(folds: list[dict]) -> dict:
    """Mean and SD over folds at every level."""
    out = {"overall": {}, "tasks": {}}
    keys = ("precision", "recall", "f1", "accuracy", "weighted_precision", "weighted_recall", "weighted_f1")
    for key in keys:
        out["overall"][key] = _stats([f["overall"].get(key) for f in folds if f.get("overall")])
    task_names = sorted({task for f in folds for task in f["tasks"]})
    for task in task_names:
        entries = [f["tasks"][task] for f in folds if task in f["tasks"]]
        block = {key: _stats([e[key] for e in entries]) for key in keys}
        classes = {}
        for label in entries[0]["classes"]:
            present = [e["classes"][label] for e in entries if e["classes"][label]["support"] > 0]
            classes[label] = {key: _stats([c[key] for c in present]) for key in METRIC_KEYS}
            classes[label]["support"] = sum(e["classes"][label]["support"] for e in entries)
        block["classes"] = classes
        out["tasks"][task] = block
    return out
