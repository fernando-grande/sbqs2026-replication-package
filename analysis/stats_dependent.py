"""STL vs MTL tests that account for dependent cross-validation folds.

The published Welch test treats the five fold scores of STL and of MTL as two
independent samples. They are not: the training sets of any two folds share
three quarters of the data, and STL and MTL were evaluated on the same folds
and the same test items. This module adds tests that respect both facts
(reviewer comment of SBQS 2026; see REPRODUCTION.md):

* ``corrected_paired_t`` - paired t-test on the per-fold differences with the
  variance correction of Nadeau & Bengio (2003, "Inference for the
  Generalization Error"; also Bouckaert & Frank 2004): the variance of the
  mean difference is (1/k + n_test/n_train) * s^2 instead of s^2 / k. Holm's
  adjustment is applied over the encoder x task comparisons.
* ``cluster_bootstrap`` - paired bootstrap of the pooled out-of-fold
  predictions that resamples whole studies, so the excerpts of one paper and
  their paraphrases move together. Reports the 95% percentile interval of
  macro-F1(MTL) - macro-F1(STL).
* ``published_corrected`` - the published best-vs-best Welch test recomputed
  with the Nadeau-Bengio variance inflation. The paper reports only means and
  standard deviations, so this is the only correction that can be applied to the published
  numbers; it inflates the variance but cannot pair the folds.

    python analysis/stats_dependent.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ALPHA = 0.05
N_BOOTSTRAP = 2000


def corrected_paired_t(stl: list[float], mtl: list[float], test_train_ratio: float) -> dict:
    """Corrected resampled paired t-test, df = k - 1, on d = MTL - STL."""
    from scipy import stats

    d = np.asarray(mtl, dtype=float) - np.asarray(stl, dtype=float)
    k = len(d)
    if k < 2:
        return {"t": None, "p_two_sided": None, "p_mtl_greater": None}
    variance = (1.0 / k + test_train_ratio) * d.var(ddof=1)
    if variance == 0:
        same = float(d.mean()) == 0.0
        return {"t": 0.0 if same else float("inf") * np.sign(d.mean()), "p_two_sided": 1.0 if same else 0.0,
                "p_mtl_greater": 0.5 if same else float(d.mean() < 0)}
    t = float(d.mean() / np.sqrt(variance))
    return {
        "t": t,
        "p_two_sided": float(2 * stats.t.sf(abs(t), k - 1)),
        "p_mtl_greater": float(stats.t.sf(t, k - 1)),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _macro_f1(confusion: np.ndarray) -> float:
    """Macro-F1 over the classes with support, as analysis.metrics does."""
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0


def _study_confusions(records: list[dict], labels: list[str], studies: list[str]) -> np.ndarray:
    position = {label: i for i, label in enumerate(labels)}
    index = {study: i for i, study in enumerate(studies)}
    out = np.zeros((len(studies), len(labels), len(labels)), dtype=np.int64)
    for row in records:
        out[index[str(row["study_id"])], position[row["true_name"]], position[row["pred_name"]]] += 1
    return out


def cluster_bootstrap(
    stl_records: list[dict],
    mtl_records: list[dict],
    labels: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over studies of pooled macro-F1(MTL) - macro-F1(STL).

    Both record lists hold the out-of-fold predictions of one task for the
    same test items (checked by instance id).
    """
    if sorted(str(r["instance_id"]) for r in stl_records) != sorted(str(r["instance_id"]) for r in mtl_records):
        raise ValueError("STL and MTL predictions do not cover the same test items")
    studies = sorted({str(r["study_id"]) for r in stl_records})
    conf_stl = _study_confusions(stl_records, labels, studies)
    conf_mtl = _study_confusions(mtl_records, labels, studies)
    observed = _macro_f1(conf_mtl.sum(axis=0)) - _macro_f1(conf_stl.sum(axis=0))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        weights = np.bincount(rng.integers(0, len(studies), len(studies)), minlength=len(studies))
        deltas[b] = _macro_f1(np.tensordot(weights, conf_mtl, axes=1)) - _macro_f1(np.tensordot(weights, conf_stl, axes=1))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "pooled_delta": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "n_studies": len(studies),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def test_train_ratio(records_by_fold: dict[int, list[dict]], task: str) -> float:
    """Mean n_test / n_train of the folds for one task (every item, paraphrases included)."""
    sizes = [sum(1 for row in records if row["task"] == task) for records in records_by_fold.values()]
    total = sum(sizes)
    return float(np.mean([size / (total - size) for size in sizes]))


def compare(
    pairs: list[tuple[str, str, str, str]],
    f1: dict[tuple[str, str, str], list[float]],
    pooled: dict[tuple[str, str, str], list[dict]],
    ratios: dict[str, float],
    labels_by_task: dict[str, list[str]],
    seed: int,
) -> list[dict]:
    """``pairs`` = (comparison, task, stl_encoder, mtl_encoder)."""
    rows = []
    for comparison, task, stl_encoder, mtl_encoder in pairs:
        stl, mtl = f1[("stl", stl_encoder, task)], f1[("mtl", mtl_encoder, task)]
        t_test = corrected_paired_t(stl, mtl, ratios[task])
        boot = cluster_bootstrap(
            pooled[("stl", stl_encoder, task)], pooled[("mtl", mtl_encoder, task)], labels_by_task[task], seed=seed
        )
        rows.append(
            {
                "comparison": comparison,
                "task": task,
                "stl_encoder": stl_encoder,
                "mtl_encoder": mtl_encoder,
                "stl_macro_f1": float(np.mean(stl)),
                "mtl_macro_f1": float(np.mean(mtl)),
                "delta_fold_mean": float(np.mean(mtl) - np.mean(stl)),
                "test_train_ratio": ratios[task],
                "t_corrected": t_test["t"],
                "p_two_sided": t_test["p_two_sided"],
                "p_mtl_greater": t_test["p_mtl_greater"],
                **boot,
            }
        )
    for comparison in sorted({row["comparison"] for row in rows}):
        block = [row for row in rows if row["comparison"] == comparison]
        for row, adjusted in zip(block, holm([row["p_two_sided"] for row in block])):
            row["p_two_sided_holm"] = adjusted
            row["significant_holm"] = adjusted < ALPHA
    return rows


def published_corrected(k: int = 5, test_train_ratio: float = 0.25) -> list[dict]:
    """Published best-vs-best Welch test with the Nadeau-Bengio variance inflation."""
    from scipy import stats

    from analysis.welch import PUBLISHED_BEST

    inflation = np.sqrt((1.0 / k + test_train_ratio) / (1.0 / k))
    rows = []
    for task, ((m_stl, s_stl), (m_mtl, s_mtl), reported) in PUBLISHED_BEST.items():
        result = stats.ttest_ind_from_stats(m_mtl, s_mtl * inflation, k, m_stl, s_stl * inflation, k, equal_var=False)
        rows.append(
            {
                "task": task,
                "p_published": reported,
                "sd_inflation": float(inflation),
                "p_corrected": float(result.pvalue),
            }
        )
    return rows


def self_test() -> None:
    from analysis.welch import published_check

    for row in published_check():
        assert round(row["p_recomputed"], 2) == row["p_reported"], row
    print("PASS  published Welch p-values (paper, Section 4.3):", {r["task"]: round(r["p_recomputed"], 2) for r in published_check()})

    expected = {"CT-Abstract": 0.84, "CT-Introduction": 0.40, "CT-Planning": 0.54, "CT-Conclusion": 0.72}
    for row in published_corrected():
        assert round(row["p_corrected"], 2) == expected[row["task"]], row
    print("PASS  published Welch with Nadeau-Bengio inflation:", {r["task"]: round(r["p_corrected"], 2) for r in published_corrected()})

    same = corrected_paired_t([0.8, 0.7, 0.9], [0.8, 0.7, 0.9], 0.25)
    assert same["p_two_sided"] == 1.0
    plain = corrected_paired_t([0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.25, 0.45, 0.5, 0.55], 0.0)
    from scipy import stats

    reference = stats.ttest_rel([0.2, 0.25, 0.45, 0.5, 0.55], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(plain["p_two_sided"] - reference.pvalue) < 1e-12
    print("PASS  corrected t with ratio 0 equals the plain paired t-test")

    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    print("PASS  Holm adjustment")

    rng = np.random.default_rng(1)
    labels = ["a", "b", "c"]
    records = [
        {"instance_id": i, "study_id": i // 4, "true_name": labels[i % 3], "pred_name": labels[rng.integers(3)]}
        for i in range(120)
    ]
    other = [{**row, "pred_name": row["true_name"] if i % 2 else row["pred_name"]} for i, row in enumerate(records)]
    first = cluster_bootstrap(records, other, labels, n_bootstrap=300, seed=7)
    second = cluster_bootstrap(records, other, labels, n_bootstrap=300, seed=7)
    assert first == second
    assert first["pooled_delta"] > 0 and first["ci_excludes_zero"]
    print("PASS  cluster bootstrap is deterministic for a fixed seed and detects a clear difference")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    self_test()
