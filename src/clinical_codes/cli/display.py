from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.table import Table

from clinical_codes.schemas import CodeResult, SystemName

_NODE_LABELS: dict[str, str] = {
    "planner": "Searching...",
    "executor": "Evaluating...",
    "evaluator": "Consolidating...",
    "consolidator": "Summarizing...",
}


def update_status(status: Status, node_name: str) -> None:
    status.update(_NODE_LABELS.get(node_name, f"{node_name}..."))


def render_results(
    console: Console,
    consolidated: dict[SystemName, list[CodeResult]],
    search_terms: dict[SystemName, str],
    verbose: bool,
) -> None:
    console.print(Rule("Results"))
    if not consolidated:
        console.print("[dim]No results[/dim]")
        return

    for system, results in consolidated.items():
        term = search_terms.get(system, "")
        console.print(f"\n[bold]{system.value}[/bold]  [dim]searched: \"{term}\"[/dim]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Code", style="cyan")
        table.add_column("Display")
        if verbose:
            table.add_column("Score", style="dim")
        for r in results:
            row = [r.code, r.display]
            if verbose:
                row.append(f"{r.score:.2f}")
            table.add_row(*row)
        console.print(table)


def render_error(console: Console, message: str, tb: str | None = None) -> None:
    body = message
    if tb:
        body += f"\n\n[dim]{escape(tb)}[/dim]"
    console.print(Panel(body, title="[red]Error[/red]", border_style="red"))
