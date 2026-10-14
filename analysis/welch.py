"""Welch's t-test, STL vs MTL, on the per-fold task macro-F1.

Two forms are reported:

* ``published`` - what the SBQS paper and the TCC report. For each task the
  best STL model (highest mean macro-F1 over the encoders) is compared with the
  best MTL model, two-sided, from the fold means and the population SDs
  (ddof=0) shown in the tables. This is the only reading that reproduces the
  published p-values (0.76 / 0.22 / 0.36 / 0.59, see ``published_check``),
  even though the paper states the one-sided hypothesis "MTL outperforms STL".
* ``per_encoder`` - STL vs MTL of the same encoder on the five fold values,
  both one-sided (H1: MTL > STL, as stated in the paper) and two-sided.

alpha = 0.05 in both.
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.05

# Mean and SD of the best STL / MTL model per task in the published tables,
# used to check that ``welch_from_stats`` reproduces the published p-values.
PUBLISHED_BEST = {
    "CT-Abstract": ((81.48, 1.93), (81.79, 1.06), 0.76),
    "CT-Introduction": ((89.84, 1.30), (88.80, 1.15), 0.22),
    "CT-Planning": ((65.95, 1.86), (64.91, 1.50), 0.36),
    "CT-Conclusion": ((98.31, 1.17), (97.89, 1.21), 0.59),
}


def welch_from_values(stl: list[float], mtl: list[float]) -> dict:
    from scipy import stats

    if len(stl) < 2 or len(mtl) < 2:
        return {"t": None, "p_two_sided": None, "p_mtl_greater": None}
    two = stats.ttest_ind(mtl, stl, equal_var=False)
    one = stats.ttest_ind(mtl, stl, equal_var=False, alternative="greater")
    return {"t": float(two.statistic), "p_two_sided": float(two.pvalue), "p_mtl_greater": float(one.pvalue)}


def welch_from_stats(mean_stl: float, sd_stl: float, mean_mtl: float, sd_mtl: float, n: int = 5) -> dict:
    """Two-sided Welch test from summary statistics, as the published one."""
    from scipy import stats

    result = stats.ttest_ind_from_stats(mean_mtl, sd_mtl, n, mean_stl, sd_stl, n, equal_var=False)
    return {"t": float(result.statistic), "p_two_sided": float(result.pvalue)}


def published_check() -> list[dict]:
    """Recompute the published p-values from the published table values."""
    rows = []
    for task, ((m_stl, s_stl), (m_mtl, s_mtl), reported) in PUBLISHED_BEST.items():
        test = welch_from_stats(m_stl, s_stl, m_mtl, s_mtl)
        rows.append({"task": task, "p_recomputed": round(test["p_two_sided"], 3), "p_reported": reported})
    return rows


def _summary(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def per_encoder(f1: dict[tuple[str, str, str], list[float]], encoders: list[str], tasks: list[str]) -> list[dict]:
    """``f1[(mode, encoder, task)]`` holds the fold macro-F1 values."""
    rows = []
    for encoder in encoders:
        for task in tasks:
            stl = f1.get(("stl", encoder, task), [])
            mtl = f1.get(("mtl", encoder, task), [])
            if not stl or not mtl:
                continue
            (m_s, sd_s), (m_m, sd_m) = _summary(stl), _summary(mtl)
            test = welch_from_values(stl, mtl)
            rows.append(
                {
                    "encoder": encoder,
                    "task": task,
                    "stl_macro_f1": m_s,
                    "stl_sd": sd_s,
                    "mtl_macro_f1": m_m,
                    "mtl_sd": sd_m,
                    "delta": m_m - m_s,
                    "transfer": "positive" if m_m > m_s else "negative",
                    "t": test["t"],
                    "p_mtl_greater": test["p_mtl_greater"],
                    "p_two_sided": test["p_two_sided"],
                    "reject_h0_mtl_greater": bool(test["p_mtl_greater"] is not None and test["p_mtl_greater"] < ALPHA),
                }
            )
    return rows


def published_form(f1: dict[tuple[str, str, str], list[float]], encoders: list[str], tasks: list[str]) -> list[dict]:
    """Best STL vs best MTL per task, two-sided, from means and population SDs."""
    rows = []
    for task in tasks:
        best = {}
        for mode in ("stl", "mtl"):
            candidates = [
                (_summary(f1[(mode, encoder, task)]), encoder)
                for encoder in encoders
                if f1.get((mode, encoder, task))
            ]
            if candidates:
                best[mode] = max(candidates, key=lambda item: item[0][0])
        if len(best) < 2:
            continue
        ((m_s, sd_s), enc_s), ((m_m, sd_m), enc_m) = best["stl"], best["mtl"]
        n = min(len(f1[("stl", enc_s, task)]), len(f1[("mtl", enc_m, task)]))
        from_stats = welch_from_stats(m_s, sd_s, m_m, sd_m, n)
        from_values = welch_from_values(f1[("stl", enc_s, task)], f1[("mtl", enc_m, task)])
        rows.append(
            {
                "task": task,
                "best_stl": enc_s,
                "stl_macro_f1": m_s,
                "stl_sd": sd_s,
                "best_mtl": enc_m,
                "mtl_macro_f1": m_m,
                "mtl_sd": sd_m,
                "delta": m_m - m_s,
                "transfer": "positive" if m_m > m_s else "negative",
                "p_two_sided_published_form": from_stats["p_two_sided"],
                "p_two_sided_from_folds": from_values["p_two_sided"],
                "reject_h0": bool(from_stats["p_two_sided"] < ALPHA),
            }
        )
    return rows
