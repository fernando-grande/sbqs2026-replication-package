"""Stage 04 - LLM-based data augmentation (AugGPT style).

Requests ``config.PARAPHRASES_PER_EXCERPT`` semantic paraphrases per original
excerpt from Gemini (``config.GEMINI_MODEL``), in batches of
``config.AUGMENTATION_BATCH_SIZE``. The prompt is the one of the original
``google-api/api.py``, reproduced verbatim (AugGPT, Dai et al.: "You are a
helpful assistant that rephrase text and make sentence smoothly"), and the
model's default sampling temperature is kept, as in that script.

Only what the balancing needs is generated. Each task is balanced towards its
largest class (paper, Section 3.2.3); for every class below that target the
stage asks for the smallest number of paraphrases per excerpt that covers the
deficit, never more than six (the AugGPT setting), and only for as many
excerpts as needed. Excerpts are picked in a seeded order spread across
studies. If the cleaning rule of stage 05 would drop a paraphrase, a few more
excerpts are paraphrased until the deficit is covered. Classes with too few
originals (Limitations, Impact) stop at seven times their size.

**Every response is cached.** ``cache/gemini_paraphrases.jsonl`` is keyed by
the model, the prompt version and the excerpt, so re-running the stage costs
nothing, produces the same output, and survives an interrupted run. Each record
also stores which provider answered (``served_by``).

**The document group travels with the paraphrase.** Each synthetic row keeps
the ``study_id`` of its parent. Stage 07 partitions folds by ``study_id``, so
without this a paraphrase could land in the test fold while its original sat in
the training fold, which is the leakage the paper claims to prevent.

Offline mode (``--offline``) fills the table with clearly marked placeholder
text instead of calling the API, so that stages 05 to 07 can be exercised
end-to-end without a key. Placeholders are tagged ``offline-placeholder`` in
the ``generator`` column and must never be used for training.
"""
import argparse
import hashlib
import math
import random
import re
import time
from collections import defaultdict

from . import config, io_utils, llm, schema
from .stage05_clean_synthetic import usable_paraphrases

STAGE = "04-augment"

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

# Verbatim from google-api/api.py, the script behind the published data.
PROMPT_VERSION = "auggpt-v1"
PROMPT = """
You are a helpful assistant that rephrase text and make sentence smoothly.
For each original phrase given below, generate {n} rephrased answers that are semantically similar.
Strictly follow the following output format:

[Original phrase X]: <repeat the original phrase here>
[Rephrased versions]:
<rephrased answer 1>
<rephrased answer 2>
...

Example:
[Original phrase 1]: Consistency is the key to achieving meaningful progress.
[Rephrased versions]:
Steady effort is essential for reaching real progress.
True progress comes from being consistent over time.
Maintaining consistency is what unlocks lasting improvement.
...

Here is the batch of phrases to be rephrased:
{batch}
"""

_BLOCK = re.compile(r"\[Original phrase \d+\]:")
_VERSIONS = re.compile(r"\[Rephrased versions\]:")


def cache_key(text: str) -> str:
    payload = f"{config.GEMINI_MODEL}|{PROMPT_VERSION}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_cache() -> dict[str, list[str]]:
    cache = {}
    for record in io_utils.read_jsonl(config.AUGMENTATION_CACHE):
        cache[record["key"]] = record["paraphrases"]
    return cache


def parse_response(response: str, batch: list[str]) -> dict[str, list[str]]:
    """Map the model output back onto the batch, relying on block order.

    Matching by position rather than by repeating the original text is
    deliberate: the model frequently normalises whitespace or punctuation when
    echoing the phrase, so exact string matching loses whole blocks.
    """
    results: dict[str, list[str]] = {}
    blocks = [b for b in _BLOCK.split(response) if b.strip()]
    for position, block in enumerate(blocks):
        if position >= len(batch):
            break
        parts = _VERSIONS.split(block)
        if len(parts) != 2:
            continue
        versions = [
            line.strip()
            for line in parts[1].splitlines()
            if line.strip() and line.strip() != "..."
        ]
        results[batch[position]] = versions
    return results


def call_gemini(backend, batch: list[str], n: int, max_retries: int = 5) -> tuple[dict[str, list[str]], str]:
    listing = "".join(f"[Original phrase {i + 1}]: {t}\n" for i, t in enumerate(batch))
    prompt = PROMPT.format(n=n, batch=listing)

    for attempt in range(max_retries):
        try:
            text, served_by, _ = backend.generate(prompt)
            return parse_response(text, batch), served_by
        except Exception as error:  # noqa: BLE001 - surface any API failure
            code = getattr(error, "code", None)
            if isinstance(code, int) and 400 <= code < 500 and code != 429:
                raise SystemExit(f"[{STAGE}] ERROR the API rejected the request: {error}")
            if attempt == max_retries - 1:
                io_utils.banner(STAGE, f"ERROR batch abandoned after {max_retries} tries: {error}")
                return {}, ""
            delay = 5 * (2**attempt)
            io_utils.banner(STAGE, f"retry in {delay}s ({attempt + 1}/{max_retries}): {error}")
            time.sleep(delay)
    return {}, ""


def spread_order(rows: list[dict], rng: random.Random) -> list[dict]:
    """Excerpts interleaved across studies, in a seeded order.

    Taking excerpts breadth-first over studies keeps a class's synthetic rows
    from coming out of a handful of papers, which matters because folds are
    grouped by study.
    """
    by_study: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["instance_id"]):
        by_study[row["study_id"]].append(row)
    groups = [by_study[study] for study in sorted(by_study)]
    for group in groups:
        rng.shuffle(group)
    rng.shuffle(groups)
    ordered = []
    depth = 0
    while True:
        layer = [group[depth] for group in groups if depth < len(group)]
        if not layer:
            break
        ordered.extend(layer)
        depth += 1
    # Distinct texts first: some classes repeat an excerpt verbatim across
    # studies (Experiment Design uses a closed vocabulary), and paraphrasing
    # the same sentence many times adds little diversity.
    seen: set[str] = set()
    first, repeats = [], []
    for row in ordered:
        key = row["text"].casefold()
        (repeats if key in seen else first).append(row)
        seen.add(key)
    return first + repeats


def plan_augmentation(instances: list[dict]) -> list[dict]:
    """Per class: deficit to the task target and paraphrases per excerpt.

    The task target is the size of the task's largest class (paper, 3.2.3).
    ``k`` is the smallest number of paraphrases per excerpt that can cover the
    deficit, capped at ``config.PARAPHRASES_PER_EXCERPT`` (six, as in AugGPT).
    Classes with too few originals (e.g. Limitations) stay below the target.
    """
    by_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in instances:
        by_class[(row["task"], row["label"])].append(row)
    targets = {task: max(len(rows) for (t, _), rows in by_class.items() if t == task) for task, _ in by_class}
    plans = []
    for task in schema.TASKS:
        for label in schema.LABELS_BY_TASK[task]:
            rows = by_class.get((task, label), [])
            deficit = targets.get(task, 0) - len(rows)
            if deficit <= 0 or not rows:
                continue
            k = min(config.PARAPHRASES_PER_EXCERPT, math.ceil(deficit / len(rows)))
            plans.append({"task": task, "label": label, "rows": rows, "target": targets[task], "deficit": deficit, "k": k})
    return plans


def run(
    offline: bool = False,
    limit: int | None = None,
    provider: str = config.GROUNDING_PROVIDER,
    sleep: float = 1.0,
) -> dict:
    config.ensure_dirs()
    rng = random.Random(config.SEED)

    instances = io_utils.read_csv(config.CLEAN_DIR / "instances.csv")
    plans = plan_augmentation(instances)
    cache = load_cache()
    io_utils.banner(STAGE, f"cache holds {len(cache)} excerpts")

    backend = None
    if not offline:
        if not llm.has_api_key(provider):
            raise SystemExit(
                f"[{STAGE}] no API key: set OPENROUTER_API_KEY or GOOGLE_API_KEY in .env, "
                "or run with --offline to exercise the remaining stages with placeholder text."
            )
        backend = llm.make_backend(
            provider,
            temperature=None,  # model default, as in google-api/api.py
            json_mode=False,
            max_tokens=config.GROUNDING_MAX_OUTPUT_TOKENS,
        )
        io_utils.banner(STAGE, f"model: {config.GEMINI_MODEL} via {llm.resolve_provider(provider)}")

    generated = 0
    failed = 0

    def ensure(requests: dict[str, int]) -> None:
        """Make sure each text has at least the requested number of paraphrases."""
        # Grouped by how many MORE paraphrases a text needs. New paraphrases
        # are appended to the cached ones, never replace them, so a text used
        # by another corpus keeps the same first paraphrases (and that
        # dataset stays byte-identical).
        by_extra: dict[int, list[str]] = defaultdict(list)
        for text, n in requests.items():
            have = len(cache.get(cache_key(text), []))
            if have < n:
                by_extra[n - have].append(text)
        for extra, texts in sorted(by_extra.items()):
            generate(texts, extra)

    def generate(texts: list[str], k: int) -> None:
        nonlocal generated, failed
        missing = [{"text": text} for text in texts]
        if offline:
            for row in missing:
                key = cache_key(row["text"])
                have = cache.get(key, [])
                cache[key] = have + [
                    f"[OFFLINE PLACEHOLDER {len(have) + i + 1}] {row['text']}" for i in range(k)
                ]
            return
        for start in range(0, len(missing), config.AUGMENTATION_BATCH_SIZE):
            if limit is not None and generated >= limit:
                return
            batch_rows = missing[start : start + config.AUGMENTATION_BATCH_SIZE]
            batch = [row["text"] for row in batch_rows]
            produced, served_by = call_gemini(backend, batch, k)
            for row in batch_rows:
                paraphrases = produced.get(row["text"], [])
                if not paraphrases:
                    failed += 1  # not cached: the next run asks again
                    continue
                key = cache_key(row["text"])
                paraphrases = cache.get(key, []) + paraphrases
                cache[key] = paraphrases
                io_utils.append_jsonl(
                    config.AUGMENTATION_CACHE,
                    {
                        "key": key,
                        "text": row["text"],
                        "paraphrases": paraphrases,
                        "requested": len(paraphrases),
                        "model": config.GEMINI_MODEL,
                        "prompt_version": PROMPT_VERSION,
                        "served_by": served_by,
                    },
                )
            generated += len(batch_rows)
            io_utils.banner(STAGE, f"generated {generated} excerpts so far")
            time.sleep(sleep)

    # instance_id -> (first, count): the slice of its text's paraphrases it
    # uses. Repeated texts get consecutive slices, so two excerpts with the
    # same wording never share a paraphrase.
    chosen: dict[str, tuple[int, int]] = {}
    slots: dict[str, int] = defaultdict(int)
    report = []
    for plan in plans:
        order = spread_order(plan["rows"], rng)
        k, deficit = plan["k"], plan["deficit"]
        usable, used = 0, 0
        # Take just enough excerpts; if cleaning (stage 05) would drop some
        # paraphrases, take a few more excerpts until the deficit is covered.
        while usable < deficit and used < len(order):
            take = order[used : used + math.ceil((deficit - usable) / k)]
            used += len(take)
            requests: dict[str, int] = {}
            for row in take:
                first = slots[row["text"]]
                slots[row["text"]] += k
                chosen[row["instance_id"]] = (first, k)
                requests[row["text"]] = max(requests.get(row["text"], 0), first + k)
            ensure(requests)
            for row in take:
                first, count = chosen[row["instance_id"]]
                paraphrases = cache.get(cache_key(row["text"]), [])[first : first + count]
                if paraphrases:
                    usable += len(usable_paraphrases(row["text"], paraphrases))
                else:
                    chosen.pop(row["instance_id"])
        report.append(
            {
                "task": plan["task"],
                "label": plan["label"],
                "originals": len(plan["rows"]),
                "target": plan["target"],
                "deficit": deficit,
                "paraphrases_per_excerpt": k,
                "excerpts_paraphrased": sum(1 for r in plan["rows"] if r["instance_id"] in chosen),
                "usable_paraphrases": usable,
                "final": len(plan["rows"]) + min(usable, deficit),
            }
        )
        io_utils.banner(
            STAGE,
            f"{plan['label']}: {len(plan['rows'])} originals, deficit {deficit}, "
            f"k={k}, excerpts {report[-1]['excerpts_paraphrased']}, usable {usable}",
        )

    if failed:
        io_utils.banner(STAGE, f"WARNING {failed} excerpts got no paraphrases; re-run to retry them")
    io_utils.write_csv(config.REPORTS / "04_augmentation_plan.csv", list(report[0]) if report else [], report)

    generator = "offline-placeholder" if offline else config.GEMINI_MODEL
    out = []
    synthetic = 0
    for row in instances:
        out.append(
            {
                "instance_id": row["instance_id"],
                "study_id": row["study_id"],
                "task": row["task"],
                "label": row["label"],
                "text": row["text"],
                "origin": "original",
                "parent_instance_id": "",
                "generator": "",
            }
        )
        first, count = chosen.get(row["instance_id"], (0, 0))
        for position, paraphrase in enumerate(cache.get(cache_key(row["text"]), [])[first : first + count]):
            out.append(
                {
                    "instance_id": f"{row['instance_id']}s{position:02d}",
                    "study_id": row["study_id"],
                    "task": row["task"],
                    "label": row["label"],
                    "text": paraphrase,
                    "origin": "synthetic",
                    "parent_instance_id": row["instance_id"],
                    "generator": generator,
                }
            )
            synthetic += 1

    target = config.AUGMENTED_DIR / "instances.csv"
    io_utils.write_csv(target, FIELDS, out)

    io_utils.banner(STAGE, f"originals: {len(instances)}  synthetic: {synthetic}  (API excerpts this run: {generated})")
    io_utils.banner(STAGE, f"rows written: {len(out)} -> {target.name}")
    if offline:
        io_utils.banner(STAGE, "WARNING offline placeholders are not usable for training")

    return {"originals": len(instances), "synthetic": synthetic}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the API and emit marked placeholder paraphrases",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after sending N excerpts to the API (for testing)"
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "openrouter", "google"),
        default=config.GROUNDING_PROVIDER,
        help="route to the Gemini model (default auto: OpenRouter when OPENROUTER_API_KEY is set)",
    )
    args = parser.parse_args()
    run(offline=args.offline, limit=args.limit, provider=args.provider)


if __name__ == "__main__":
    main()
