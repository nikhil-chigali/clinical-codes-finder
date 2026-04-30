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

    return "\n".join(lines)
