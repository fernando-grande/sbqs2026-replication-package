"""Signature of a training run, shared by training and evaluation.

A run's results are only reused (training resume) or counted (evaluation)
when the signature stored in its run_info.json equals the current one. Kept
free of torch so that run_eval.py can check it without the training stack.
"""
from __future__ import annotations

from . import config, io_utils


def current(
    encoder_key: str,
    strict: bool = True,
    hyperparameters: dict | None = None,
) -> dict:
    encoder = config.ENCODERS[encoder_key]
    signature = {
        "model_name": encoder["name"],
        "model_revision": encoder["revision"],
        "attn_implementation": config.ATTN_IMPLEMENTATION,
        "deterministic": "strict" if strict else "warn_only",
        "gradient_checkpointing": bool(encoder.get("gradient_checkpointing", False)),
        "hyperparameters": dict(hyperparameters or config.TRAINING),
        "seed_base": config.SEED,
        "instances_csv_sha256": io_utils.content_sha256(config.TRAINING_DIR / "instances.csv"),
        "folds_csv_sha256": io_utils.content_sha256(config.TRAINING_DIR / "folds.csv"),
    }
    return signature


def differences(stored: dict | None, expected: dict) -> list[str]:
    """Human-readable fields where a stored signature differs from the expected one."""
    if not stored:
        return ["no signature (run made before signatures, e.g. SDPA attention)"]
    out = []
    for key, value in expected.items():
        if stored.get(key) != value:
            if key in ("attn_implementation", "deterministic"):
                out.append(f"{key}={stored.get(key)}")
            else:
                out.append(key)
    return out
