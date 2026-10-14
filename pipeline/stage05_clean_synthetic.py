"""Stage 05 - post-processing of generated text.

Generalises ``experiments-data-aug/limpa.py``, which handled one hard-coded file
per run and only stripped the ``1. `` to ``6. `` numbering.

Original rows pass through untouched. Synthetic rows are:

* stripped of the list marker the model prepends ("1. ", "2) ", "- ", bullets);
* discarded when empty, when shorter than half the parent excerpt, or when they
  merely echo the parent verbatim, since such rows add no lexical diversity and
  inflate the class count without enriching the feature space.
"""
from . import config, io_utils, textnorm

STAGE = "05-clean-synthetic"

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


def check_paraphrase(raw: str, parent: str, siblings: set[str]) -> tuple[str, str]:
    """Return (cleaned text, drop reason); the reason is "" when the row is kept.

    ``siblings`` holds the casefolded paraphrases already kept for the same
    parent. Stage 04 uses the same rule to know how many usable paraphrases a
    class will end up with.
    """
    text = textnorm.strip_enumeration(textnorm.normalise(raw))
    if not text:
        return text, "empty after cleaning"
    if text.casefold() == parent.casefold():
        return text, "identical to parent"
    if parent and len(text) < len(parent) / 2:
        return text, "truncated relative to parent"
    if text.casefold() in siblings:
        return text, "duplicate of a sibling paraphrase"
    return text, ""


def usable_paraphrases(parent: str, paraphrases: list[str]) -> list[str]:
    """The paraphrases of one excerpt that survive this stage, cleaned."""
    siblings: set[str] = set()
    kept = []
    for raw in paraphrases:
        text, reason = check_paraphrase(raw, parent, siblings)
        if not reason:
            siblings.add(text.casefold())
            kept.append(text)
    return kept


def run() -> dict:
    config.ensure_dirs()

    rows = io_utils.read_csv(config.AUGMENTED_DIR / "instances.csv")
    parents = {r["instance_id"]: r["text"] for r in rows if r["origin"] == "original"}

    kept: list[dict] = []
    dropped: list[dict] = []
    seen_per_parent: dict[str, set[str]] = {}

    for row in rows:
        if row["origin"] == "original":
            kept.append(row)
            continue

        parent = parents.get(row["parent_instance_id"], "")
        siblings = seen_per_parent.setdefault(row["parent_instance_id"], set())
        text, reason = check_paraphrase(row["text"], parent, siblings)
        if reason:
            dropped.append({**row, "drop_reason": reason})
            continue
        siblings.add(text.casefold())

        kept.append({**row, "text": text})

    target = config.CLEAN_SYNTHETIC_DIR / "instances.csv"
    io_utils.write_csv(target, FIELDS, kept)
    io_utils.write_csv(
        config.REPORTS / "05_dropped_synthetic.csv", FIELDS + ["drop_reason"], dropped
    )

    synthetic = sum(1 for r in kept if r["origin"] == "synthetic")
    io_utils.banner(STAGE, f"synthetic rows discarded: {len(dropped)}")
    io_utils.banner(STAGE, f"synthetic rows kept: {synthetic}")
    io_utils.banner(STAGE, f"rows written: {len(kept)} -> {target.name}")

    return {"kept": len(kept), "dropped": len(dropped)}


if __name__ == "__main__":
    run()
