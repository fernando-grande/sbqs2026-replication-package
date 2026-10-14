"""Train STL or MTL classifiers on the official dataset.

    python run_train.py --mode stl --encoder scibert
    python run_train.py --mode mtl --encoder scibert
    python run_train.py --mode stl --encoder bert --task CT-Planning
    python run_train.py --mode both --encoder all                      # the whole study
    python run_train.py --mode mtl --encoder distilbert --epochs 1 --folds 1   # smoke test

Results go to metrics/<mode>/<encoder>[/<task>]/ (metrics_smoke/ for smoke
tests):

    fold{k}_predictions.csv     every test prediction (with origin of the item)
    fold{k}_metrics.json, folds.json, summary.json
                                metrics of each fold and their mean and SD
    run_info.json               commit, versions, GPU, seeds, attention, memory,
                                data hashes

A run is skipped only when its run_info.json says "completed" AND its
signature (model, revision, attention, determinism, gradient checkpointing,
hyperparameters, dataset hashes) equals the current one; anything else is
retrained. --force always retrains.

Reproducibility: every fold starts from seed SEED + fold (Python, numpy,
torch, CUDA and the DataLoader shuffling); cuDNN is deterministic and torch
deterministic algorithms are enforced (an operation without a deterministic
implementation raises an error; --allow-nondeterministic turns that into a
warning and is recorded). Attention is "eager" for every encoder (config).
"""
from __future__ import annotations

import os

# Must be set before torch initialises CUDA: deterministic cuBLAS workspace,
# and expandable segments to limit allocator fragmentation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.metrics import aggregate, fold_metrics
from pipeline import config, io_utils, run_signature, schema
from train.data import TextDataset, class_weights, iter_folds, label_maps, load_instances
from train.loop import evaluate, make_loader, make_scheduler, train_epoch
from train.models import MultiHeadClassifier, SingleTaskClassifier

MIB = 1024 * 1024


def set_seed(seed: int, strict: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def section_classes():
    return {task: list(labels) for task, labels in schema.LABELS_BY_TASK.items()}


def environment() -> dict:
    import transformers

    cuda = torch.cuda.is_available()
    return {
        # Fingerprint of the package code (pipeline/io_utils.code_sha256);
        # verify.py checks it against the code of the package.
        "code_sha256": io_utils.code_sha256(ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


def _free_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _memory(device: str) -> dict:
    if device != "cuda":
        return {}
    return {
        "max_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
        "allocated_mib": round(torch.cuda.memory_allocated() / MIB, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1),
    }


def run(
    mode: str,
    encoder_key: str,
    task_filter: str | None = None,
    epochs: int | None = None,
    max_folds: int | None = None,
    force: bool = False,
    strict: bool = True,
) -> Path:
    config.ensure_dirs()

    if encoder_key not in config.ENCODERS:
        raise SystemExit(f"unknown encoder {encoder_key}. Choose from {list(config.ENCODERS)}")
    encoder = config.ENCODERS[encoder_key]
    model_name, revision = encoder["name"], encoder["revision"]
    checkpointing = bool(encoder.get("gradient_checkpointing", False))
    attn = config.ATTN_IMPLEMENTATION
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = section_classes()
    if task_filter:
        if task_filter not in classes:
            raise SystemExit(f"unknown task {task_filter}")
        classes = {task_filter: classes[task_filter]}

    instances_path = config.TRAINING_DIR / "instances.csv"
    folds_path = config.TRAINING_DIR / "folds.csv"
    if not instances_path.exists() or not folds_path.exists():
        raise SystemExit(f"Training tables missing ({config.TRAINING_DIR}). They are versioned: pull the repository.")

    hp = dict(config.TRAINING)
    smoke = epochs is not None or max_folds is not None
    if epochs is not None:
        hp["epochs"] = epochs

    signature = run_signature.current(encoder_key, strict, hp)

    out_dir = config.metrics_dir(smoke) / mode / encoder_key
    if task_filter:
        out_dir = out_dir / task_filter
    info_path = out_dir / "run_info.json"
    if not force and not smoke and info_path.exists():
        previous = json.loads(info_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed" and previous.get("signature") == signature:
            print(f"skip {out_dir} (already trained with the same signature; use --force to retrain)")
            return out_dir
        print(f"retrain {out_dir} (previous run incomplete or trained with a different signature)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("fold*_predictions.csv"):
        stale.unlink()

    df = load_instances(instances_path, section_classes())
    if task_filter:
        df = df[df["task"] == task_filter]
    label2id, _ = label_maps(classes)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    info = {
        "status": "running",
        "mode": mode,
        "encoder": encoder_key,
        "task": task_filter,
        "signature": signature,
        "seed_rule": "SEED + fold, reset at the start of every fold",
        "smoke_test": smoke,
        "instances": int(len(df)),
        "environment": environment(),
        "folds_done": [],
        "memory": {},
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(
        f"device: {device}"
        + (f" ({info['environment']['gpu']})" if device == "cuda" else " - no GPU found")
        + f" | attention {attn} | deterministic {'strict' if strict else 'warn_only'}"
        + f" | gradient checkpointing {'on' if checkpointing else 'off'}"
    )

    fold_results: list[dict] = []
    for fold, train_df, test_df in iter_folds(df, folds_path):
        if max_folds is not None and fold >= max_folds:
            break
        seed = config.SEED + fold
        set_seed(seed, strict)
        _free_memory(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"\n===== {mode.upper()} {encoder_key}{' ' + task_filter if task_filter else ''} fold {fold + 1} (seed {seed}) =====")
        print(f"train {len(train_df)} / test {len(test_df)}")
        train_set = TextDataset(train_df, tokenizer, label2id, hp["max_length"])
        test_set = TextDataset(test_df, tokenizer, label2id, hp["max_length"])
        train_loader = make_loader(train_set, hp["batch_size"], shuffle=True, seed=seed)
        test_loader = make_loader(test_set, hp["batch_size"], shuffle=False)

        options = {"revision": revision, "attn_implementation": attn, "gradient_checkpointing": checkpointing}
        if mode == "stl":
            task = next(iter(classes))
            model = SingleTaskClassifier(model_name, len(classes[task]), hp["dropout"], **options).to(device)
            weights = class_weights(train_df, task, label2id, device)
            criteria = {task: torch.nn.CrossEntropyLoss(weight=weights)}
        else:
            model = MultiHeadClassifier(model_name, classes, hp["dropout"], **options).to(device)
            criteria = {task: torch.nn.CrossEntropyLoss() for task in classes}

        optimizer = AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
        steps = hp["epochs"] * max(len(train_loader), 1)
        scheduler = make_scheduler(optimizer, steps, hp["warmup_ratio"])

        losses = []
        for epoch in range(hp["epochs"]):
            loss = train_epoch(model, train_loader, optimizer, scheduler, device, mode, hp["max_grad_norm"], criteria)
            losses.append(loss)
            print(f"epoch {epoch + 1}/{hp['epochs']} loss={loss:.6f}")

        records = evaluate(model, test_loader, device, mode, classes, label2id)
        for row in records:
            row["fold"] = fold + 1
        pd.DataFrame(records).to_csv(out_dir / f"fold{fold + 1}_predictions.csv", index=False)

        result = fold_metrics(records, classes)
        result["fold"] = fold + 1
        fold_results.append(result)
        (out_dir / f"fold{fold + 1}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for task_name, task_metrics in result["tasks"].items():
            print(
                f"{task_name}: macro_f1={task_metrics['f1']:.4f} "
                f"acc={task_metrics['accuracy']:.4f} (n={task_metrics['n']})"
            )

        memory = _memory(device)
        weights = None
        del model, optimizer, scheduler, criteria, weights, train_loader, test_loader, train_set, test_set
        _free_memory(device)
        if device == "cuda":
            memory["allocated_after_cleanup_mib"] = round(torch.cuda.memory_allocated() / MIB, 1)
            print(
                f"memory fold {fold + 1}: peak allocated {memory['max_allocated_mib']} MiB, "
                f"peak reserved {memory['max_reserved_mib']} MiB, "
                f"allocated after cleanup {memory['allocated_after_cleanup_mib']} MiB"
            )
        info["memory"][str(fold + 1)] = memory
        info.setdefault("epoch_losses", {})[str(fold + 1)] = losses
        info["folds_done"].append(fold + 1)
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    (out_dir / "folds.json").write_text(json.dumps(fold_results, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(aggregate(fold_results), indent=2), encoding="utf-8")

    info["status"] = "completed"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    _free_memory(device)
    print(f"wrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["stl", "mtl", "both"], required=True)
    parser.add_argument("--encoder", choices=list(config.ENCODERS) + ["all"], default="scibert")
    parser.add_argument(
        "--task",
        default=None,
        help="STL only: train a specialist for one task (default: all four, one after another)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override the epochs (smoke tests only)")
    parser.add_argument("--folds", type=int, default=None, help="run only the first N folds (smoke tests only)")
    parser.add_argument("--force", action="store_true", help="retrain runs that already have results")
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="warn instead of failing on operations without a deterministic implementation (recorded in run_info)",
    )
    args = parser.parse_args()

    if args.task and args.mode != "stl":
        parser.error("--task trains a single-task (STL) model; use it with --mode stl")

    encoders = list(config.ENCODERS) if args.encoder == "all" else [args.encoder]
    modes = ["stl", "mtl"] if args.mode == "both" else [args.mode]
    options = {
        "epochs": args.epochs,
        "max_folds": args.folds,
        "force": args.force,
        "strict": not args.allow_nondeterministic,
    }
    for encoder in encoders:
        for mode in modes:
            if mode == "stl" and not args.task:
                for task in schema.TASKS:
                    run("stl", encoder, task_filter=task, **options)
            else:
                run(mode, encoder, task_filter=args.task, **options)


if __name__ == "__main__":
    main()
