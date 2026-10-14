"""Stage 07 - training views and cross-validation folds.

Emits three things:

``instances.csv``
    The long-format table consumed by the training code, carrying ``study_id``
    and ``origin`` alongside the text and label.

``by_class/<label>.csv``
    One file per class, in the ``id,section,text`` layout the existing scripts
    under ``project/`` expect, extended with ``study_id`` and ``origin``.

``folds.csv``
    The five cross-validation folds, materialised. Writing folds to disk rather
    than recomputing them inside the training loop means every model in the
    comparison sees byte-identical splits, and a reader can verify the leakage
    claim without running anything.

Folds follow the "stratified group 5-fold" of the paper: groups are whole
studies (a study owns all of its excerpts and all of their paraphrases, so no
paraphrase can sit in a different fold from its original), and studies are
assigned so that every class is spread across the folds as evenly as possible.
Stratification uses the original excerpts only.

The algorithm is the one of scikit-learn's ``StratifiedGroupKFold`` (greedy
assignment of each group to the fold that minimises the spread of per-class
proportions), written out here with a fixed seed so the folds do not depend on
the installed library version. Studies holding the rarest classes are placed
first, which is what lets Impact (six originals) and Limitations reach as many
folds as possible. ``reports/07_fold_coverage.csv`` lists, per fold and class,
how many original and synthetic excerpts the test side holds.

Note for the training code: ``project/data_loader.py`` currently asserts that
every class file has the same row count and uses the row index as the grouping
key. Both assumptions break here, since ``Limitations`` and ``Impact`` stay
below the task target and the grouping key must be ``study_id``.
"""
import random
from collections import Counter, defaultdict

from . import config, io_utils, schema

STAGE = "07-build-training-sets"

FIELDS = ["id", "study_id", "task", "label", "text", "origin"]
FOLD_FIELDS = ["fold", "study_id", "split"]


def run(n_folds: int = config.N_FOLDS) -> dict:
    config.ensure_dirs()

    rows = io_utils.read_csv(config.FINAL_DIR / "instances.csv")

    instances = [
        {
            "id": position,
            "study_id": int(row["study_id"]),
            "task": row["task"],
            "label": row["label"],
            "text": row["text"],
            "origin": row["origin"],
        }
        for position, row in enumerate(rows)
    ]

    io_utils.write_csv(config.TRAINING_DIR / "instances.csv", FIELDS, instances)

    by_class = defaultdict(list)
    for row in instances:
        by_class[row["label"]].append(row)
    for label, group in by_class.items():
        io_utils.write_csv(
            config.TRAINING_DIR / "by_class" / f"{label}.csv",
            ["id", "section", "text", "study_id", "origin"],
            [
                {
                    "id": position,
                    "section": label,
                    "text": row["text"],
                    "study_id": row["study_id"],
                    "origin": row["origin"],
                }
                for position, row in enumerate(group)
            ],
        )

    folds = _stratified_group_k_fold(instances, n_folds, config.SEED)
    fold_rows = [
        {"fold": fold, "study_id": study_id, "split": "test"}
        for fold, studies in folds.items()
        for study_id in sorted(studies)
    ]
    io_utils.write_csv(config.TRAINING_DIR / "folds.csv", FOLD_FIELDS, fold_rows)

    per_fold = {
        fold: sum(1 for row in instances if row["study_id"] in studies)
        for fold, studies in folds.items()
    }

    io_utils.banner(STAGE, f"instances: {len(instances)}")
    io_utils.banner(STAGE, f"classes written: {len(by_class)}")
    io_utils.banner(STAGE, f"studies per test fold: { {f: len(s) for f, s in folds.items()} }")
    io_utils.banner(STAGE, f"instances per test fold: {per_fold}")

    coverage, missing = _coverage(instances, folds)
    io_utils.write_csv(
        config.REPORTS / "07_fold_coverage.csv",
        ["fold", "task", "label", "test_original", "test_synthetic"],
        coverage,
    )
    if missing:
        io_utils.banner(STAGE, f"WARNING classes with no original excerpt in some test fold: {missing}")
    else:
        io_utils.banner(STAGE, "every class has original excerpts in every test fold")

    return {"instances": len(instances), "folds": n_folds}


def _stratified_group_k_fold(instances: list[dict], n_folds: int, seed: int) -> dict[int, set[int]]:
    """Assign whole studies to folds, spreading every class across the folds."""
    originals = [row for row in instances if row["origin"] == "original"]
    classes = sorted({f"{row['task']}|{row['label']}" for row in originals})
    index = {name: i for i, name in enumerate(classes)}
    totals = Counter(f"{row['task']}|{row['label']}" for row in originals)

    per_study: dict[int, list[int]] = defaultdict(lambda: [0] * len(classes))
    size: Counter = Counter()
    for row in instances:
        size[row["study_id"]] += 1
    for row in originals:
        per_study[row["study_id"]][index[f"{row['task']}|{row['label']}"]] += 1
    for study in size:
        per_study[study]  # studies with synthetic rows only still need a fold

    studies = sorted(per_study)
    random.Random(seed).shuffle(studies)
    # Rarest class first, then larger studies: the scarce classes are placed
    # while every fold is still free to take them.
    def rarity(study: int) -> tuple:
        present = [totals[classes[i]] for i, n in enumerate(per_study[study]) if n]
        return (min(present) if present else 10**9, -size[study])

    studies.sort(key=rarity)

    fold_counts = [[0] * len(classes) for _ in range(n_folds)]
    fold_size = [0] * n_folds
    assignment: dict[int, set[int]] = {fold: set() for fold in range(n_folds)}

    def spread(counts: list[list[int]]) -> float:
        total = 0.0
        for c, name in enumerate(classes):
            shares = [counts[f][c] / totals[name] for f in range(n_folds)]
            mean = sum(shares) / n_folds
            total += (sum((x - mean) ** 2 for x in shares) / n_folds) ** 0.5
        return total / len(classes)

    for study in studies:
        vector = per_study[study]
        best = None
        for fold in range(n_folds):
            trial = [row[:] for row in fold_counts]
            trial[fold] = [a + b for a, b in zip(trial[fold], vector)]
            key = (round(spread(trial), 12), fold_size[fold], fold)
            if best is None or key < best[0]:
                best = (key, fold)
        fold = best[1]
        fold_counts[fold] = [a + b for a, b in zip(fold_counts[fold], vector)]
        fold_size[fold] += size[study]
        assignment[fold].add(study)
    return assignment


def _coverage(instances: list[dict], folds: dict[int, set[int]]) -> tuple[list[dict], list[str]]:
    study_fold = {study: fold for fold, studies in folds.items() for study in studies}
    counts: Counter = Counter()
    for row in instances:
        counts[(study_fold[row["study_id"]], row["task"], row["label"], row["origin"])] += 1
    rows, missing = [], set()
    for fold in sorted(folds):
        for task in schema.TASKS:
            for label in schema.LABELS_BY_TASK[task]:
                original = counts[(fold, task, label, "original")]
                rows.append(
                    {
                        "fold": fold,
                        "task": task,
                        "label": label,
                        "test_original": original,
                        "test_synthetic": counts[(fold, task, label, "synthetic")],
                    }
                )
                if original == 0:
                    missing.add(label)
    return rows, sorted(missing)


if __name__ == "__main__":
    run()
