from clinical_codes.evaluation.metrics import MetricsSummary


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _overall_top3_recall(summary: MetricsSummary) -> str:
    all_none = all(qt.top3_recall is None for qt in summary.by_type.values())
    return "n/a" if all_none else f"{summary.top3_recall:.2f}"


def format_markdown(summary: MetricsSummary) -> str:
    lines: list[str] = []

    # Overall
    lines.append("## Overall\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total queries | {summary.n_total} |")
    lines.append(f"| Errors | {summary.n_errors} |")
    lines.append(f"| System-selection F1 | {summary.system_selection_f1:.2f} |")
    lines.append(f"| Top-3 recall | {_overall_top3_recall(summary)} |")
    lines.append(f"| Must-include hit rate | {summary.must_include_hit_rate:.2f} |")
    lines.append(f"| Mean iterations | {summary.mean_iterations:.2f} |")
    lines.append(f"| Mean API calls | {summary.mean_api_calls:.2f} |")
    lines.append("")

    # By query type
    lines.append("## By query type\n")
    lines.append("| Type | N | System F1 | Top-3 recall | Must-include | Mean iter | Mean API calls |")
    lines.append("|---|---|---|---|---|---|---|")
    for qt in summary.by_type.values():
        lines.append(
            f"| {qt.query_type} | {qt.n} | {qt.system_selection_f1:.2f} | "
            f"{_fmt(qt.top3_recall)} | {_fmt(qt.must_include_hit_rate)} | "
            f"{qt.mean_iterations:.2f} | {qt.mean_api_calls:.2f} |"
        )
    lines.append("")

    # Failures
    failures = [qm for qm in summary.per_query if qm.system_f1 < 1.0 or qm.error is not None]
    lines.append("## Failures (system_f1 < 1.0 or error)\n")
    if not failures:
        lines.append("*(none)*")
    else:
        lines.append("| Query ID | Query | Type | System F1 | Error |")
        lines.append("|---|---|---|---|---|")
        for qm in failures:
            error_str = qm.error if qm.error is not None else "—"
            lines.append(
                f"| {qm.query_id} | {qm.query} | {qm.query_type} | "
                f"{qm.system_f1:.2f} | {error_str} |"
            )

    return "\n".join(lines)
