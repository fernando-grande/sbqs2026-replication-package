"""Reproduction criterion, fixed before training (see REPRODUCTION.md).

Compares the re-implementation with the published Macro-F1 (Tables 2 and 5 of
the SBQS paper, identical in the TCC). Values are percentages. run_eval.py
applies it and writes reports/reproduction_verdict.csv.

Tolerance T = max(2 pp, published SD).

(a) per task and mode, the best encoder's mean Macro-F1 is within T of the
    published best (T uses the SD of the published best). Secondary, reported
    only: share of encoder x task x mode cells within their own T.
(b) per task, with d = best MTL - best STL and tol = max(2 pp, larger SD of the
    two published best models): if |d_published| > tol, the direction of d must
    match; otherwise |d_reimplemented| <= tol ("tie reproduced").
(c) Welch, published form (best STL vs best MTL, two-sided): p >= 0.05.

Levels, in this order:
    incomplete             one of the 40 cells (2 modes x 5 encoders x 4 tasks)
                           or a Welch p is missing: no verdict yet
    not reproduced         c fails in any task, or b fails in most tasks
    reproduced             a, b and c pass in every task
    reproduced qualitatively   b and c pass in every task, a fails somewhere
    partial reproduction   any other case

    python analysis/reproduction_check.py     # runs the self-test
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ("CT-Abstract", "CT-Introduction", "CT-Planning", "CT-Conclusion")
ENCODERS = ("bert", "scibert", "distilbert", "roberta", "deberta")

# Published Macro-F1, mean and SD over 5 folds, in %.
PUBLISHED = {
    "stl": {
        "bert": ((78.93, 1.31), (88.22, 1.63), (63.47, 1.10), (97.42, 1.14)),
        "scibert": ((80.12, 0.48), (89.84, 1.30), (65.95, 1.86), (97.33, 1.25)),
        "distilbert": ((77.77, 0.95), (89.45, 0.89), (63.96, 1.45), (97.66, 1.44)),
        "roberta": ((81.48, 1.93), (88.30, 1.44), (64.67, 2.09), (98.31, 1.17)),
        "deberta": ((80.11, 1.65), (87.71, 1.53), (64.12, 0.96), (97.05, 1.96)),
    },
    "mtl": {
        "bert": ((79.56, 0.95), (87.74, 1.56), (62.74, 1.12), (97.89, 1.21)),
        "scibert": ((81.79, 1.06), (88.80, 1.15), (64.91, 1.50), (97.32, 1.73)),
        "distilbert": ((78.58, 1.25), (88.23, 0.86), (63.41, 1.27), (96.60, 2.12)),
        "roberta": ((79.86, 2.86), (88.11, 1.57), (64.67, 1.56), (97.31, 0.71)),
        "deberta": ((79.22, 2.57), (87.44, 3.69), (63.29, 1.57), (96.38, 2.16)),
    },
}
PUBLISHED_WELCH_P = {"CT-Abstract": 0.76, "CT-Introduction": 0.22, "CT-Planning": 0.36, "CT-Conclusion": 0.59}
MIN_TOLERANCE = 2.0
ALPHA = 0.05


def published_cells() -> dict[tuple[str, str, str], tuple[float, float]]:
    return {
        (mode, encoder, task): values[i]
        for mode, table in PUBLISHED.items()
        for encoder, values in table.items()
        for i, task in enumerate(TASKS)
    }


def _best(cells: dict, mode: str, task: str) -> tuple[str, float, float] | None:
    candidates = [(enc, *cells[(mode, enc, task)]) for enc in ENCODERS if (mode, enc, task) in cells]
    return max(candidates, key=lambda c: c[1]) if candidates else None


def verdict(
    reimpl: dict[tuple[str, str, str], tuple[float, float]],
    welch_p: dict[str, float],
    not_counted: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """``reimpl[(mode, encoder, task)] = (mean %, sd %)``; ``welch_p[task]``.

    ``not_counted`` lists runs that exist but were rejected (unfinished, wrong
    number of folds, different signature); they make the verdict incomplete.
    """
    pub = published_cells()
    missing = list(not_counted or [])
    missing += sorted(f"{m}/{e}/{t}: no counted run" for (m, e, t) in pub if (m, e, t) not in reimpl)
    missing += sorted(f"welch/{t}: no p-value" for t in TASKS if welch_p.get(t) is None)
    if missing:
        return "incomplete", [{"task": "all", "missing": "; ".join(dict.fromkeys(missing))}]
    rows = []
    for task in TASKS:
        row = {"task": task}
        a_pass = True
        for mode in ("stl", "mtl"):
            p_best, r_best = _best(pub, mode, task), _best(reimpl, mode, task)
            tol = max(MIN_TOLERANCE, p_best[2])
            ok = r_best is not None and abs(r_best[1] - p_best[1]) <= tol
            row.update(
                {
                    f"{mode}_published_best": f"{p_best[0]} {p_best[1]:.2f}",
                    f"{mode}_reimpl_best": f"{r_best[0]} {r_best[1]:.2f}" if r_best else "",
                    f"{mode}_tolerance": tol,
                    f"a_{mode}": ok,
                }
            )
            a_pass &= ok
        row["a"] = a_pass

        p_stl, p_mtl = _best(pub, "stl", task), _best(pub, "mtl", task)
        r_stl, r_mtl = _best(reimpl, "stl", task), _best(reimpl, "mtl", task)
        tol_b = max(MIN_TOLERANCE, p_stl[2], p_mtl[2])
        d_pub = p_mtl[1] - p_stl[1]
        d_re = (r_mtl[1] - r_stl[1]) if r_stl and r_mtl else None
        if abs(d_pub) > tol_b:
            b_rule = "direction"
            b_pass = d_re is not None and (d_re > 0) == (d_pub > 0)
        else:
            b_rule = "tie"
            b_pass = d_re is not None and abs(d_re) <= tol_b
        row.update({"delta_published": d_pub, "delta_reimpl": d_re, "b_tolerance": tol_b, "b_rule": b_rule, "b": b_pass})

        p_value = welch_p.get(task)
        row.update({"welch_p": p_value, "c": p_value is not None and p_value >= ALPHA})

        cells = [(mode, enc) for mode in ("stl", "mtl") for enc in ENCODERS]
        within = [
            abs(reimpl[(m, e, task)][0] - pub[(m, e, task)][0]) <= max(MIN_TOLERANCE, pub[(m, e, task)][1])
            for m, e in cells
            if (m, e, task) in reimpl
        ]
        row["cells_within_tolerance"] = f"{sum(within)}/{len(within)}"
        rows.append(row)

    b_failures = sum(not r["b"] for r in rows)
    if any(not r["c"] for r in rows) or b_failures > len(rows) / 2:
        level = "not reproduced"
    elif all(r["a"] and r["b"] and r["c"] for r in rows):
        level = "reproduced"
    elif all(r["b"] and r["c"] for r in rows):
        level = "reproduced qualitatively"
    else:
        level = "partial reproduction"
    return level, rows


def self_test() -> None:
    """Positive and negative cases; raises AssertionError on a wrong verdict."""
    pub = published_cells()
    cases = []

    cases.append(("published values", dict(pub), dict(PUBLISHED_WELCH_P), "reproduced"))

    lower = {k: (v[0] - 5, v[1]) for k, v in pub.items()}
    cases.append(("F1 5 pp lower everywhere", lower, dict(PUBLISHED_WELCH_P), "reproduced qualitatively"))

    flipped = dict(lower)
    for enc in ENCODERS:  # MTL 5 pp worse again in one task: |delta| > tolerance
        mean, sd = flipped[("mtl", enc, "CT-Planning")]
        flipped[("mtl", enc, "CT-Planning")] = (mean - 5, sd)
    cases.append(("F1 5 pp lower, one task breaks b", flipped, dict(PUBLISHED_WELCH_P), "partial reproduction"))

    significant = dict(PUBLISHED_WELCH_P, **{"CT-Introduction": 0.01})
    cases.append(("Welch p < 0.05 in one task", dict(pub), significant, "not reproduced"))

    partial_run = dict(pub)
    del partial_run[("stl", "deberta", "CT-Abstract")]
    cases.append(("one encoder x task x mode missing", partial_run, dict(PUBLISHED_WELCH_P), "incomplete"))

    cases.append(("all cells present but one run rejected", dict(pub), dict(PUBLISHED_WELCH_P), "incomplete"))

    for name, values, p_values, expected in cases:
        rejected = ["stl/bert/CT-Abstract: 3/5 folds"] if name.startswith("all cells present") else None
        level, _ = verdict(values, p_values, rejected)
        assert level == expected, f"{name}: got {level!r}, expected {expected!r}"
        print(f"PASS  {name}: {level}")


if __name__ == "__main__":
    self_test()
