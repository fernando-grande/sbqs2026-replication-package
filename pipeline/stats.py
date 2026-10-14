"""Per-class distribution tables shared by stages 03 and 06."""
from collections import Counter
from pathlib import Path

from . import io_utils, schema

FIELDS = ["task", "label", "display_name", "count"]


def distribution(instances: list[dict]) -> Counter:
    return Counter((row["task"], row["label"]) for row in instances)


def write_table(path: Path, instances: list[dict]) -> Counter:
    counts = distribution(instances)
    rows = []
    for task in schema.TASKS:
        for label in schema.LABELS_BY_TASK[task]:
            rows.append(
                {
                    "task": task,
                    "label": label,
                    "display_name": schema.DISPLAY_NAMES[label],
                    "count": counts.get((task, label), 0),
                }
            )
    io_utils.write_csv(path, FIELDS, rows)
    return counts


def render(counts: Counter, title: str) -> str:
    """Format a distribution as a fixed-width table for the console and reports."""
    lines = [title, "-" * len(title)]
    total = 0
    for task in schema.TASKS:
        lines.append(f"{task}")
        subtotal = 0
        for label in schema.LABELS_BY_TASK[task]:
            count = counts.get((task, label), 0)
            subtotal += count
            lines.append(f"  {schema.DISPLAY_NAMES[label]:<24}{count:>6}")
        lines.append(f"  {'subtotal':<24}{subtotal:>6}")
        total += subtotal
    lines.append(f"{'TOTAL':<26}{total:>6}")
    return "\n".join(lines)


def latex_table(before: Counter, after: Counter) -> str:
    """Emit the per-class distribution table used in the paper."""
    head = [
        r"\begin{table}[t]",
        r"\caption{Per-class distribution before and after LLM-based augmentation.}",
        r"\label{tab:distribution}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Task & Class & Before & After \\",
        r"\midrule",
    ]
    body = []
    total_before = total_after = 0
    for position, task in enumerate(schema.TASKS):
        labels = schema.LABELS_BY_TASK[task]
        if position:
            body.append(r"\midrule")
        body.append(rf"\multirow{{{len(labels)}}}{{*}}{{{task}}}")
        for label in labels:
            b = before.get((task, label), 0)
            a = after.get((task, label), 0)
            total_before += b
            total_after += a
            body.append(
                rf" & {schema.DISPLAY_NAMES[label]} & {b} & {a} \\"
            )
    tail = [
        r"\midrule",
        rf"\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Total}}}} & "
        rf"\textbf{{{total_before:,}}} & \textbf{{{total_after:,}}} \\".replace(",", "{,}"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(head + body + tail)
