from pathlib import Path

from clinical_codes.evaluation.metrics import MetricsSummary


def format_markdown(summary: MetricsSummary) -> str:
    lines: list[str] = []

    lines.append("## Overall\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total queries | {summary.n_total} |")
    lines.append(f"| Errors | {summary.n_errors} |")
    lines.append(f"| System-selection F1 | {summary.system_selection_f1:.2f} |")
    lines.append(f"| Top-3 recall | {summary.top3_recall:.2f} |")
    lines.append(f"| Must-include hit rate | {summary.must_include_hit_rate:.2f} |")
    lines.append(f"| Mean iterations | {summary.mean_iterations:.2f} |")
    lines.append(f"| Mean API calls | {summary.mean_api_calls:.2f} |")

    return "\n".join(lines)
