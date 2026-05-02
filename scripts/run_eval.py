from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from clinical_codes.cli.display import render_error
from clinical_codes.config import settings
from clinical_codes.evaluation.metrics import compute_metrics
from clinical_codes.evaluation.reporter import write_report
from clinical_codes.evaluation.runner import run_query
from clinical_codes.evaluation.schema import GoldSet

app = typer.Typer(add_completion=False)


@app.command()
def run(
    gold: Annotated[Path, typer.Option(help="Path to gold set JSON file")],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to write results")
    ] = Path("results"),
) -> None:
    err_console = Console(stderr=True)
    console = Console()

    if not settings.anthropic_api_key:
        render_error(err_console, "ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        raise typer.Exit(code=1)

    if not gold.exists():
        render_error(err_console, f"Gold file not found: {gold}")
        raise typer.Exit(code=1)

    try:
        data = json.loads(gold.read_text(encoding="utf-8"))
        gold_set = GoldSet.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        render_error(err_console, f"Invalid gold file: {e}")
        raise typer.Exit(code=1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    console.print(Rule(f"Eval run {run_id} — {len(gold_set.queries)} queries"))

    results = []
    for gq in gold_set.queries:
        result = run_query(gq)
        if result.error:
            status = f"[red]ERROR: {result.error}[/red]"
        else:
            status = f"[green]{result.latency_s:.1f}s[/green]"
        console.print(f"  {gq.id} ({gq.query_type}): {status}")
        results.append(result)

    summary = compute_metrics(results, gold_set.queries)
    json_path, md_path = write_report(summary, run_id, output_dir)

    console.print(Rule("Results"))

    overall = Table(show_header=True, header_style="bold")
    overall.add_column("Metric")
    overall.add_column("Value", justify="right")
    overall.add_row("Total queries", str(summary.n_total))
    overall.add_row("Errors", str(summary.n_errors))
    overall.add_row("System-selection F1", f"{summary.system_selection_f1:.2f}")
    overall.add_row("Top-3 recall", f"{summary.top3_recall:.2f}")
    overall.add_row("Must-include hit rate", f"{summary.must_include_hit_rate:.2f}")
    overall.add_row("Mean iterations", f"{summary.mean_iterations:.2f}")
    overall.add_row("Mean API calls", f"{summary.mean_api_calls:.2f}")
    console.print(overall)

    by_type = Table(show_header=True, header_style="bold")
    by_type.add_column("Type")
    by_type.add_column("N", justify="right")
    by_type.add_column("System F1", justify="right")
    by_type.add_column("Top-3 recall", justify="right")
    by_type.add_column("Must-include", justify="right")
    by_type.add_column("Mean iter", justify="right")
    by_type.add_column("Mean API calls", justify="right")
    for qt in summary.by_type.values():
        top3 = "n/a" if qt.top3_recall is None else f"{qt.top3_recall:.2f}"
        mi = "n/a" if qt.must_include_hit_rate is None else f"{qt.must_include_hit_rate:.2f}"
        by_type.add_row(
            qt.query_type, str(qt.n),
            f"{qt.system_selection_f1:.2f}", top3, mi,
            f"{qt.mean_iterations:.2f}", f"{qt.mean_api_calls:.2f}",
        )
    console.print(by_type)

    console.print(f"\nWrote: [bold]{json_path}[/bold]")
    console.print(f"       [bold]{md_path}[/bold]")


if __name__ == "__main__":
    app()
