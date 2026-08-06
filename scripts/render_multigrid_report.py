"""Render a multigrid evaluation JSON report as readable Markdown."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCORE_LABELS = {
    "token_accuracy": "Token accuracy",
    "exact_match": "Exact match",
    "loss": "Loss",
}

DISPLAY_NAMES = {
    "gru": "GRU",
    "ssm": "SSM",
    "d_ff": "FFN width",
}


def _display_name(value: object) -> str:
    normalized = str(value).strip().lower()
    return DISPLAY_NAMES.get(normalized, normalized.replace("_", " ").title())


def _markdown(value: object) -> str:
    """Escape a value for use in a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _ordered_values(
    preferred: Iterable[object] | None,
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[object]:
    values: list[object] = []
    seen: set[object] = set()
    for value in preferred or ():
        if value not in seen:
            values.append(value)
            seen.add(value)
    for row in rows:
        value = row.get(key)
        if value is not None and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def _format_score(value: float, metric: str, *, bold: bool = False) -> str:
    rendered = f"{value:.2%}" if metric != "loss" else f"{value:.4f}"
    return f"**{rendered}**" if bold else rendered


def _best_value(values: Iterable[float], metric: str) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return min(finite) if metric == "loss" else max(finite)


def _is_best(value: float, best: float | None) -> bool:
    return best is not None and math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12)


def _table(headers: Sequence[object], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(_markdown(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown(value) for value in row) + " |"
        for row in rows
    )
    return lines


def _score_matrix(
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[object],
    mechanisms: Sequence[object],
    metric: str,
) -> list[str]:
    scores = {
        (row.get("task"), row.get("mechanism")): float(row[metric])
        for row in rows
        if row.get(metric) is not None
    }
    matrix: list[list[str]] = []
    for task in tasks:
        available = [
            scores[(task, mechanism)]
            for mechanism in mechanisms
            if (task, mechanism) in scores
        ]
        best = _best_value(available, metric)
        cells = [_display_name(task)]
        for mechanism in mechanisms:
            value = scores.get((task, mechanism))
            cells.append(
                "—"
                if value is None
                else _format_score(value, metric, bold=_is_best(value, best))
            )
        matrix.append(cells)
    return _table(
        ["Test", *(_display_name(value) for value in mechanisms)], matrix
    )


def _mean_rows(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, object], list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            grouped[(row.get("task"), row.get("mechanism"))].append(
                float(row[metric])
            )
    return [
        {"task": task, "mechanism": mechanism, metric: sum(values) / len(values)}
        for (task, mechanism), values in grouped.items()
    ]


def render_report(report: Mapping[str, Any], metric: str) -> str:
    """Return a Markdown rendering of an evaluation report."""
    if metric not in SCORE_LABELS:
        raise ValueError(f"unsupported score metric: {metric}")

    config = report.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("report 'config' must be an object")
    accuracy = report.get("accuracy", [])
    if not isinstance(accuracy, list) or not all(
        isinstance(row, Mapping) for row in accuracy
    ):
        raise ValueError("report 'accuracy' must be a list of objects")
    if not accuracy:
        raise ValueError("report contains no accuracy results")

    tasks = _ordered_values(config.get("tasks"), accuracy, "task")
    mechanisms = _ordered_values(
        config.get("mechanisms"), accuracy, "mechanism"
    )
    capacities = _ordered_values(config.get("capacities"), accuracy, "capacity")
    label = SCORE_LABELS[metric]
    lines = [
        "# Multigrid Evaluation Results",
        "",
        f"Score shown: **{label}**. The best result in each test row is bold.",
        "",
        "## Evaluation setup",
        "",
    ]
    setup_rows = [
        ("Tests", len(tasks)),
        ("Mechanisms", len(mechanisms)),
        ("Capacities", ", ".join(str(value) for value in capacities)),
    ]
    for key, title in (
        ("train_capacity", "Training capacity"),
        ("train_steps", "Training steps"),
        ("batch_size", "Batch size"),
        ("eval_batches", "Evaluation batches"),
        ("seed", "Seed"),
    ):
        if key in config:
            setup_rows.append((title, config[key]))
    lines.extend(_table(("Setting", "Value"), setup_rows))

    lines.extend(("", f"## Mean {label.lower()} across capacities", ""))
    lines.extend(
        _score_matrix(
            _mean_rows(accuracy, metric), tasks, mechanisms, metric
        )
    )

    train_capacity = config.get("train_capacity")
    lines.extend(("", "## Scores by capacity", ""))
    for index, capacity in enumerate(capacities):
        capacity_rows = [row for row in accuracy if row.get("capacity") == capacity]
        regime = next(
            (row.get("regime") for row in capacity_rows if row.get("regime")),
            None,
        )
        if regime is None and isinstance(capacity, (int, float)) and isinstance(
            train_capacity, (int, float)
        ):
            regime = "interpolation" if capacity <= train_capacity else "extrapolation"
        suffix = f" — {_display_name(regime)}" if regime else ""
        lines.extend((f"### Capacity {capacity}{suffix}", ""))
        lines.extend(
            _score_matrix(capacity_rows, tasks, mechanisms, metric)
        )
        if index != len(capacities) - 1:
            lines.append("")

    parameters = report.get("parameters", [])
    if isinstance(parameters, list) and parameters:
        lines.extend(("", "## Mechanisms", ""))
        parameter_rows = []
        for row in parameters:
            if not isinstance(row, Mapping):
                continue
            delta = row.get("delta_vs_multigrid")
            parameter_rows.append(
                (
                    _display_name(row.get("name", row.get("mechanism", ""))),
                    f"{int(row['parameters']):,}"
                    if row.get("parameters") is not None
                    else "—",
                    f"{int(delta):+,}" if delta is not None else "—",
                    f"{int(row['d_ff']):,}" if row.get("d_ff") is not None else "—",
                )
            )
        lines.extend(
            _table(
                ("Mechanism", "Parameters", "Δ vs. multigrid", "FFN width"),
                parameter_rows,
            )
        )

    runtime = report.get("runtime", [])
    if isinstance(runtime, list) and runtime:
        lines.extend(("", "## Runtime", ""))
        runtime_rows = []
        for row in runtime:
            if not isinstance(row, Mapping):
                continue
            peak = row.get("peak_memory_bytes")
            runtime_rows.append(
                (
                    _display_name(row.get("mechanism", "")),
                    row.get("sequence_length", "—"),
                    f"{float(row['prefill_ms']):.2f}"
                    if row.get("prefill_ms") is not None
                    else "—",
                    f"{float(row['prefill_tokens_per_second']):,.1f}"
                    if row.get("prefill_tokens_per_second") is not None
                    else "—",
                    f"{float(row['uncached_decode_ms_per_token']):.2f}"
                    if row.get("uncached_decode_ms_per_token") is not None
                    else "—",
                    f"{float(peak) / (1024**2):.1f}" if peak is not None else "—",
                )
            )
        lines.extend(
            _table(
                (
                    "Mechanism",
                    "Sequence length",
                    "Prefill (ms)",
                    "Prefill (tokens/s)",
                    "Decode (ms/token)",
                    "Peak memory (MiB)",
                ),
                runtime_rows,
            )
        )

    notes = report.get("notes", {})
    if isinstance(notes, Mapping) and notes:
        lines.extend(("", "## Notes", ""))
        lines.extend(f"- {_markdown(value)}" for value in notes.values())

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render evaluate_multigrid report.json results as Markdown."
    )
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=Path("checkpoints/multigrid-evaluation-v3/report.json"),
        help="input report.json path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: REPORT directory/report.md)",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(SCORE_LABELS),
        default="token_accuracy",
        help="score rendered in the matrices (default: token_accuracy)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or args.report.with_name("report.md")
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError("report root must be a JSON object")
        markdown = render_report(report, args.metric)
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"error: {error}") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
