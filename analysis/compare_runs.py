"""Compare a retraining with the published run of the same configuration.

    python analysis/compare_runs.py <published run dir> <retrained run dir>

e.g., after retraining into a copy of the package:

    python analysis/compare_runs.py metrics/stl/scibert/CT-Planning ../copy/metrics/stl/scibert/CT-Planning

Reports, per fold: whether the prediction files are byte-identical, how many
predicted labels differ and the Macro-F1 per task of both runs; then whether
the run signatures, environments and per-epoch losses match (REPRODUCTION.md,
"Retraining from a clean clone"). Exit code 0 only when every fold is
byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd




def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _macro_f1(df: pd.DataFrame) -> dict:
    """Macro-F1 (%) per task over the classes with support (as in analysis/metrics.py)."""
    out = {}
    for task, rows in df.groupby("task"):
        f1s = []
        for label in sorted(set(rows["true_name"])):
            tp = int(((rows["true_name"] == label) & (rows["pred_name"] == label)).sum())
            fp = int(((rows["true_name"] != label) & (rows["pred_name"] == label)).sum())
            fn = int(((rows["true_name"] == label) & (rows["pred_name"] != label)).sum())
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        out[task] = round(100 * sum(f1s) / len(f1s), 2)
    return out


def compare(published: Path, retrained: Path) -> bool:
    folds = sorted(published.glob("fold*_predictions.csv"))
    if not folds:
        raise SystemExit(f"no fold*_predictions.csv in {published}")
    all_identical = True
    for path in folds:
        other = retrained / path.name
        fold = int(path.stem.split("_")[0].removeprefix("fold"))
        if not other.exists():
            print(f"fold {fold}: MISSING in {retrained}")
            all_identical = False
            continue
        identical = _sha256(path) == _sha256(other)
        all_identical &= identical
        line = f"fold {fold}: {'byte-identical' if identical else 'DIFFERENT'}"
        if not identical:
            a, b = pd.read_csv(path), pd.read_csv(other)
            if len(a) != len(b) or list(a.columns) != list(b.columns):
                line += f" (shape {a.shape} vs {b.shape})"
            else:
                diffs = int((a["pred_name"] != b["pred_name"]).sum())
                line += f" ({diffs} predicted labels differ out of {len(a)} rows)"
        print(line)
        if not identical:
            print(f"    published Macro-F1: {_macro_f1(pd.read_csv(path))}")
            print(f"    retrained Macro-F1: {_macro_f1(pd.read_csv(other))}")
    for name in ("run_info.json",):
        a = json.loads((published / name).read_text(encoding="utf-8"))
        b = json.loads((retrained / name).read_text(encoding="utf-8"))
        same_sig = a.get("signature") == b.get("signature")
        print(f"signature: {'same' if same_sig else 'DIFFERENT'}")
        env_a, env_b = a.get("environment", {}), b.get("environment", {})
        for key in ("code_sha256", "torch", "transformers", "numpy", "pandas", "cuda", "cudnn", "gpu"):
            if env_a.get(key) != env_b.get(key):
                print(f"environment differs: {key} {env_a.get(key)} vs {env_b.get(key)}")
        if a.get("epoch_losses") or b.get("epoch_losses"):
            same = a.get("epoch_losses") == b.get("epoch_losses")
            print(f"per-epoch losses: {'identical' if same else 'DIFFERENT'}")
    print("RESULT:", "byte-identical predictions in every fold" if all_identical else "differences found (see above)")
    return all_identical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("published", type=Path)
    parser.add_argument("retrained", type=Path)
    args = parser.parse_args()
    sys.exit(0 if compare(args.published, args.retrained) else 1)


if __name__ == "__main__":
    main()
