"""Stage 06 - balancing.

Implements the balancing step of the study.

The rule inferred from the published distribution, and reproduced here:

1. For each task, the target size is the number of **original** excerpts in the
   task's largest class. Abstract targets its ``Objective`` count, Introduction
   its ``Context`` count, and so on.
2. Every class keeps all of its originals, then receives synthetic paraphrases
   until it reaches the target.
3. A class that cannot reach the target keeps whatever the 1:6 ratio allows.
   ``Limitations`` and ``Impact`` are the two classes in this situation: six
   paraphrases of eleven and of six originals cannot span the gap to roughly
   two hundred.

Step 3 is why the phrase "the 1:6 ratio balanced the dataset" overstates the
outcome. The imbalance is reduced by a large factor but not removed.

Paraphrase selection is a seeded round-robin over parent excerpts rather than a
flat random sample. Taking paraphrases breadth-first keeps a handful of
prolific source studies from dominating a small class, which matters because
folds are grouped by study: a class whose synthetic mass came from three papers
would lose most of its training signal whenever those papers fall in the test
fold.
"""
import random
from collections import defaultdict

from . import config, io_utils, schema, stats

STAGE = "06-balance"

FIELDS = [
    "instance_id",
    "study_id",
    "task",
    "label",
    "text",
    "origin",
    "parent_instance_id",
    "generator",
]


def run(seed: int = config.SEED) -> dict:
    config.ensure_dirs()
    rng = random.Random(seed)

    rows = io_utils.read_csv(config.CLEAN_SYNTHETIC_DIR / "instances.csv")

    originals: dict[tuple[str, str], list[dict]] = defaultdict(list)
    synthetic: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        bucket = originals if row["origin"] == "original" else synthetic
        bucket[(row["task"], row["label"])].append(row)

    targets = {}
    for task in schema.TASKS:
        sizes = [
            len(originals[(task, label)]) for label in schema.LABELS_BY_TASK[task]
        ]
        targets[task] = max(sizes) if sizes else 0

    selected: list[dict] = []
    summary = []

    for task in schema.TASKS:
        target = targets[task]
        for label in schema.LABELS_BY_TASK[task]:
            key = (task, label)
            base = sorted(originals[key], key=lambda r: r["instance_id"])
            selected.extend(base)

            needed = max(0, target - len(base))
            pool = _round_robin(synthetic[key], rng)
            taken = pool[:needed]
            selected.extend(taken)

            summary.append(
                {
                    "task": task,
                    "label": label,
                    "display_name": schema.DISPLAY_NAMES[label],
                    "target": target,
                    "originals": len(base),
                    "synthetic_available": len(synthetic[key]),
                    "synthetic_used": len(taken),
                    "final": len(base) + len(taken),
                    "reached_target": "yes" if len(base) + len(taken) >= target else "no",
                }
            )

    target_file = config.FINAL_DIR / "instances.csv"
    io_utils.write_csv(target_file, FIELDS, selected)

    for task in schema.TASKS:
        io_utils.write_csv(
            config.FINAL_DIR / f"{task}.csv",
            FIELDS,
            [r for r in selected if r["task"] == task],
        )

    io_utils.write_csv(
        config.REPORTS / "06_balancing_summary.csv",
        [
            "task",
            "label",
            "display_name",
            "target",
            "originals",
            "synthetic_available",
            "synthetic_used",
            "final",
            "reached_target",
        ],
        summary,
    )

    before = stats.distribution([r for r in rows if r["origin"] == "original"])
    after = stats.write_table(config.REPORTS / "stats_final.csv", selected)
    report = stats.render(after, "Distribution after augmentation and balancing")
    (config.REPORTS / "stats_final.txt").write_text(report, encoding="utf-8")
    (config.REPORTS / "table_distribution.tex").write_text(
        stats.latex_table(before, after), encoding="utf-8"
    )

    short = [r for r in summary if r["reached_target"] == "no"]
    io_utils.banner(STAGE, f"targets per task: {targets}")
    io_utils.banner(STAGE, f"classes below target: {[r['label'] for r in short]}")
    io_utils.banner(STAGE, f"instances written: {len(selected)} -> {target_file.name}")
    print(report)

    return {"instances": len(selected), "targets": targets}


def _round_robin(pool: list[dict], rng: random.Random) -> list[dict]:
    """Interleave paraphrases so consecutive picks come from different parents.

    ``pool`` is grouped by parent excerpt, each group is shuffled with the
    seeded generator, and the groups are then drained one item at a time. The
    result is a deterministic ordering that spreads the selection across as many
    source studies as possible.
    """
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_parent[row["parent_instance_id"]].append(row)

    groups = []
    for parent in sorted(by_parent):
        group = sorted(by_parent[parent], key=lambda r: r["instance_id"])
        rng.shuffle(group)
        groups.append(group)
    rng.shuffle(groups)

    ordered = []
    depth = 0
    while True:
        layer = [group[depth] for group in groups if depth < len(group)]
        if not layer:
            break
        ordered.extend(layer)
        depth += 1
    return ordered


if __name__ == "__main__":
    run()
