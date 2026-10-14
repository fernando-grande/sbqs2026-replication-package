"""Extractive grounding of the spreadsheet excerpts with Google Gemini.

The annotators of the systematic mapping often recorded interpretative
summaries instead of the literal sentences of the paper. As described in the
SBQS 2026 paper (Section 3.2.2), each populated spreadsheet value is treated as
a semantic query: the model receives the full PDF text of the study and must
return, for every field, the verbatim span that expresses the same content.

One request per study, carrying every populated field at once, as in the
original ``extrai_jsons/api.py``. Differences from that script:

* 22 of the 23 classes are grounded, not only the planning ones;
* fields that the annotators left empty are NOT sent, so the model never
  invents an instance for them and the class distribution is preserved;
* ``Experiment Design`` is not grounded: it keeps the annotators' closed
  vocabulary, as the original prompt required (see ``NOT_GROUNDED``);
* answers are cached, and every span is checked against the PDF text
  (``extract/fields.py``) before it is used.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from pipeline import config, io_utils, llm, schema
from pipeline.llm import ApiError, has_api_key, load_env_file, resolve_provider  # noqa: F401 - re-exported

PROMPT = """You are a senior research assistant in empirical Software Engineering.
You are grounding the protocol of a controlled experiment reported in a paper.

You receive:
1. [GUIDE]: a JSON object. Each key is a field of the reporting guideline of
   Jedlitschka et al. (2008); each value is what a human annotator recorded for
   that field. The value may be an exact quote, a paraphrase or a summary.
2. [ARTICLE]: the full text of the paper, extracted from its PDF.

For every key in [GUIDE], find in [ARTICLE] the passage that expresses the same
content as the annotator's value and copy it VERBATIM.

Rules:
- Copy text exactly as it appears in [ARTICLE]: same words, same order. Do not
  paraphrase, summarise, translate, correct or complete it. You may only join
  words split by a line break or a hyphen at the end of a line.
- If the value is already a literal quote of [ARTICLE], return it unchanged.
- If the content is spread over several passages, return each one verbatim and
  join them with " ... ".
- Keep each span as short as possible while covering the annotator's value.
- If no passage of [ARTICLE] expresses that content, return "" for that key.
- Return every key of [GUIDE], and no other key.

Field definitions:
{definitions}

[GUIDE]
{guide}

[ARTICLE]
{article}

Answer with a single JSON object mapping each key of [GUIDE] to the copied span.
"""

DEFINITIONS = {
    "Background": "Abstract - context and motivation of the study.",
    "Objective": "Abstract - the objective of the study.",
    "Methods": "Abstract - the research method used.",
    "Results": "Abstract - the main results.",
    "Limitations": "Abstract - the limitations of the study.",
    "Conclusions": "Abstract - the conclusions of the study.",
    "ProblemStatement": "Introduction - the problem the study addresses.",
    "ResearchObjectives": "Introduction - the research objective (e.g. in GQM form).",
    "Context": "Introduction - the context in which the study takes place.",
    "Goals": "Experiment planning - the goals of the experiment.",
    "ExperimentalUnits": "Experiment planning - the experimental units / participants and how they were sampled.",
    "ExperimentalMaterial": "Experiment planning - the objects and material used in the experiment.",
    "Tasks": "Experiment planning - the tasks performed by the participants.",
    "Hypotheses": "Experiment planning - the null and alternative hypotheses.",
    "Parameters": "Experiment planning - the parameters kept fixed during the experiment.",
    "IndependentVariables": "Experiment planning - the independent variables (factors and treatments).",
    "DependentVariables": "Experiment planning - the dependent variables (response variables and their measures).",
    "Procedure": "Experiment planning - the procedure followed to run the experiment.",
    "AnalysisProcedure": "Experiment planning - the data analysis procedure and statistical tests.",
    "Summary": "Conclusion - the summary of the findings.",
    "Impact": "Conclusion - the impact of the findings.",
    "FutureWork": "Conclusion - the future work.",
}


def build_prompt(guide: dict[str, str], article: str) -> str:
    definitions = "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in guide)
    return PROMPT.format(
        definitions=definitions,
        guide=json.dumps(guide, ensure_ascii=False, indent=2),
        article=article,
    )


def cache_key(study_id: str, guide: dict[str, str], article: str) -> str:
    payload = json.dumps(
        {
            "model": config.GEMINI_MODEL,
            "prompt": config.GROUNDING_PROMPT_VERSION,
            "study_id": str(study_id),
            "guide": guide,
            "article": hashlib.sha256(article.encode("utf-8")).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_cache() -> dict[str, dict]:
    return {record["key"]: record for record in io_utils.read_jsonl(config.GROUNDING_CACHE)}


def parse_response(text: str, labels: list[str]) -> dict[str, str]:
    """Read the JSON answer; tolerate a markdown fence around it."""
    body = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL)
    if fenced:
        body = fenced.group(1).strip()
    # strict=False: spans copied from the PDF may keep raw line breaks and tabs
    # inside the JSON strings, which the default parser rejects.
    data = json.loads(body, strict=False)
    if not isinstance(data, dict):
        raise ValueError("answer is not a JSON object")
    spans = {}
    for label in labels:
        value = data.get(label, "")
        if isinstance(value, list):
            value = " ... ".join(str(part) for part in value)
        spans[label] = str(value or "").strip()
    return spans


class GroundingConfigError(RuntimeError):
    """The API rejected the request itself (key, model, quota tier...)."""


def salvage_fields(text: str, labels: list[str]) -> dict[str, str]:
    """Fields whose JSON string value is complete in a truncated answer."""
    from json.decoder import scanstring

    spans = {}
    for label in labels:
        match = re.search(r'"%s"\s*:\s*"' % re.escape(label), text)
        if not match:
            continue
        try:
            value, _ = scanstring(text, match.end(), False)
        except ValueError:
            continue  # the string never closes: this is where the answer was cut
        spans[label] = value.strip()
    return spans


class Grounder:
    """Calls Gemini once per study, with a persistent cache and retries.

    Gemini occasionally degenerates while copying a passage that is followed by
    a list in the PDF: it emits line breaks until the output limit and the JSON
    never closes. Such an answer is not thrown away. The fields completed
    before the loop are kept, the field being written is dropped (it keeps the
    spreadsheet text), and the fields after it are asked again in a new call.
    """

    def __init__(self, sleep: float = config.GROUNDING_SLEEP, provider: str = config.GROUNDING_PROVIDER) -> None:
        self.provider = resolve_provider(provider)
        self._backend = llm.make_backend(
            self.provider,
            temperature=0,
            json_mode=True,
            max_tokens=config.GROUNDING_MAX_OUTPUT_TOKENS,
        )
        self._sleep = sleep
        self._last_call = 0.0
        self.cache = load_cache()
        self.calls = 0
        # When set, only cached answers are returned and the API is not called.
        self.offline = False

    def _generate(self, guide: dict[str, str], article: str) -> tuple[dict[str, str], list[str], str, list[str]]:
        """Spans for ``guide``; recovers from answers truncated by a loop."""
        labels = list(guide)
        text, served_by, truncated = self._backend.generate(build_prompt(guide, article))
        if not truncated:
            return parse_response(text, labels), [text], served_by, []

        spans = salvage_fields(text, labels)
        pending = [label for label in labels if label not in spans]
        if not pending:
            return spans, [text], served_by, []
        # The first unfinished field is the one the model looped on.
        dropped, rest = [pending[0]], pending[1:]
        texts = [text]
        if rest:
            self.calls += 1
            more, more_texts, _, more_dropped = self._generate({label: guide[label] for label in rest}, article)
            spans.update(more)
            texts += more_texts
            dropped += more_dropped
        return {label: spans.get(label, "") for label in labels}, texts, served_by, dropped

    def ground(self, study_id: str, guide: dict[str, str], article: str) -> tuple[dict[str, str], str]:
        """Return (spans by label, origin), origin being "cache", "api" or "error: ..."."""
        if not guide:
            return {}, "empty"
        key = cache_key(study_id, guide, article)
        if key in self.cache:
            return self.cache[key]["spans"], "cache"
        if self.offline:
            return {}, "pending"

        labels = list(guide)
        error = ""
        for attempt in range(1, config.GROUNDING_MAX_ATTEMPTS + 1):
            wait = self._sleep - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.monotonic()
                self.calls += 1
                spans, texts, served_by, dropped = self._generate(guide, article)
            except Exception as exc:  # noqa: BLE001 - any API or parsing failure is retried
                error = f"{type(exc).__name__}: {exc}"
                code = getattr(exc, "code", None)
                if isinstance(code, int) and 400 <= code < 500 and code != 429:
                    # Bad key, unknown model, bad request: retrying cannot help.
                    raise GroundingConfigError(error) from exc
                if code == 429 and "PerDay" in str(exc):
                    # Daily quota: nothing will succeed until it resets. The
                    # cache keeps what was done, so the next run resumes here.
                    raise GroundingConfigError("daily request quota exhausted: " + error[:300]) from exc
                io_utils.banner("extract", f"study {study_id}: attempt {attempt} failed ({error[:160]})")
                if isinstance(exc, ValueError) and attempt >= 2:
                    # A malformed answer at temperature 0 tends to repeat;
                    # leave the study for the next run instead of paying again.
                    break
                time.sleep(min(60, 10 * attempt))
                continue

            record = {
                "key": key,
                "study_id": str(study_id),
                "model": config.GEMINI_MODEL,
                "served_by": served_by,
                "prompt_version": config.GROUNDING_PROMPT_VERSION,
                "labels": labels,
                "spans": spans,
                "raw": texts[0] if len(texts) == 1 else texts,
            }
            if dropped:
                record["dropped_fields"] = dropped
                io_utils.banner("extract", f"study {study_id}: answer looped on {', '.join(dropped)}; kept the spreadsheet text there")
            io_utils.append_jsonl(config.GROUNDING_CACHE, record)
            self.cache[key] = record
            return spans, "api"
        # Failures are not cached, so the next run tries this study again.
        return {}, f"error: {error}"


# Classes whose spreadsheet value is a category, not an excerpt. The annotators
# recorded Experiment Design with a closed vocabulary ("One factor with two
# treatments", ...), and the original grounding prompt (extrai_jsons/api.py,
# rule 5) kept that vocabulary instead of copying text from the paper.
NOT_GROUNDED = frozenset({"ExperimentDesign"})


def guide_for(instances: list[dict]) -> dict[str, str]:
    """Spreadsheet value per label for one study, in schema order."""
    by_label = {row["label"]: row["text"] for row in instances}
    return {
        label: by_label[label]
        for label in schema.ALL_LABELS
        if label in by_label and label not in NOT_GROUNDED
    }
