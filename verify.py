"""Integrity checks on the dataset (the versioned one or one rebuilt by run_dataset.py)."""
import sys
from collections import Counter

from pipeline import config, io_utils, schema

HARD_FAILURES = 0

# Hashes of the versioned training tables (REPRODUCTION.md). Update them only
# together with a deliberate, documented rebuild of the dataset.
EXPECTED_SHA256 = {
    "instances.csv": "664e1fed540bc24f62795c8e7f3ca56e446eceff80660932c28575bf41cf2655",
    "folds.csv": "19a29d882ec900c2cb5df6888df9491b5ff55a05e14a642c266499e1baa485f5",
}


def check(name: str, ok: bool, detail: str = "", hard: bool = True) -> None:
    global HARD_FAILURES
    status = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not ok and hard:
        HARD_FAILURES += 1


def main() -> int:
    folds = io_utils.read_csv(config.TRAINING_DIR / "folds.csv")
    training = io_utils.read_csv(config.TRAINING_DIR / "instances.csv")

    test_fold = {int(row["study_id"]): int(row["fold"]) for row in folds}

    occurrences = Counter(int(row["study_id"]) for row in folds)
    repeated = [study for study, count in occurrences.items() if count > 1]
    studies = {int(row["study_id"]) for row in training}
    unassigned = sorted(studies - set(test_fold))
    check(
        "every study in exactly one test fold",
        not repeated and not unassigned,
        f"repeated={repeated} unassigned={unassigned}",
    )

    # The parent of each paraphrase is only in data/.../06_final, which is
    # produced by run_dataset.py and not versioned. On a training machine that
    # only has the versioned 07_training tables these two checks are skipped;
    # the hash check below guarantees the tables are the verified ones.
    final_path = config.FINAL_DIR / "instances.csv"
    if final_path.exists():
        final = io_utils.read_csv(final_path)
        by_id = {row["instance_id"]: row for row in final}
        mismatched = [
            row["instance_id"]
            for row in final
            if row["origin"] == "synthetic"
            and row["parent_instance_id"] in by_id
            and by_id[row["parent_instance_id"]]["study_id"] != row["study_id"]
        ]
        check("synthetic rows inherit the study of their parent", not mismatched, f"{len(mismatched)} mismatched")
        split_violations = [
            row["instance_id"]
            for row in final
            if row["origin"] == "synthetic"
            and row["parent_instance_id"] in by_id
            and test_fold[int(row["study_id"])] != test_fold[int(by_id[row["parent_instance_id"]]["study_id"])]
        ]
        check("paraphrase stays in the same fold as its parent", not split_violations, f"{len(split_violations)} violations")
    else:
        print("[SKIP] paraphrase/parent checks: 06_final not present (run_dataset.py output, not versioned)")

    check("training table exists", (config.TRAINING_DIR / "instances.csv").exists())

    # Every class needs original excerpts in every test fold.
    fold_of = {int(row["study_id"]): int(row["fold"]) for row in folds}
    present = {(fold_of[int(r["study_id"])], r["task"], r["label"]) for r in training if r["origin"] == "original"}
    missing = sorted(
        f"fold {fold}: {label}"
        for fold in sorted(set(fold_of.values()))
        for task in schema.TASKS
        for label in schema.LABELS_BY_TASK[task]
        if (fold, task, label) not in present
    )
    check("every class has original excerpts in every test fold", not missing, ", ".join(missing[:10]))

    # Hashes identify the exact dataset a run was trained on (run_info.json
    # stores the same values).
    for name in EXPECTED_SHA256:
        digest = io_utils.content_sha256(config.TRAINING_DIR / name)
        relative = (config.TRAINING_DIR / name).relative_to(config.PACKAGE_ROOT).as_posix()
        print(f"[INFO] sha256 (LF-normalised) {relative} = {digest}")
        check(f"{relative} is the published dataset (REPRODUCTION.md)", digest == EXPECTED_SHA256[name], f"expected {EXPECTED_SHA256[name]}")
    # Every published run must have been trained with the code of this
    # package and on these tables.
    code = io_utils.code_sha256(config.PACKAGE_ROOT)
    print(f"[INFO] code fingerprint (pipeline/io_utils.code_sha256) = {code}")
    runs = sorted(config.METRICS_DIR.rglob("run_info.json"))
    if runs:
        import json

        infos = [json.loads(path.read_text(encoding="utf-8")) for path in runs]
        other_code = [str(p.parent.relative_to(config.METRICS_DIR)) for p, i in zip(runs, infos) if i.get("environment", {}).get("code_sha256") != code]
        data = (EXPECTED_SHA256["instances.csv"], EXPECTED_SHA256["folds.csv"])
        other_data = [
            str(p.parent.relative_to(config.METRICS_DIR))
            for p, i in zip(runs, infos)
            if (i["signature"]["instances_csv_sha256"], i["signature"]["folds_csv_sha256"]) != data
        ]
        check(f"all {len(runs)} runs in metrics/ were trained with this code", not other_code, ", ".join(other_code[:5]))
        check(f"all {len(runs)} runs in metrics/ were trained on these tables", not other_data, ", ".join(other_data[:5]))
    else:
        print("[INFO] no runs in metrics/ yet")
    return 1 if HARD_FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
