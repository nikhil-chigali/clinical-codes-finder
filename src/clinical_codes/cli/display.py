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
    "re_ranker": "Summarizing...",
}


def update_status(status: Status, node_name: str) -> None:
    status.update(_NODE_LABELS.get(node_name, f"{node_name}..."))


def render_results(
    console: Console,
    consolidated: list[CodeResult],
    search_terms: dict[SystemName, str],
    verbose: bool,
) -> None:
    console.print(Rule("Results"))
    if not consolidated:
        console.print("[dim]No results[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim")
    table.add_column("System")
    table.add_column("Code", style="cyan")
    table.add_column("Display")
    table.add_column("Searched as", style="dim")
    if verbose:
        table.add_column("Score", style="dim")

    for i, r in enumerate(consolidated, 1):
        term = search_terms.get(r.system, "")
        row = [str(i), r.system.value, r.code, r.display, term]
        if verbose:
            row.append(f"{r.score:.2f}")
        table.add_row(*row)

    console.print(table)


def render_error(console: Console, message: str, tb: str | None = None) -> None:
    body = message
    if tb:
        body += f"\n\n[dim]{escape(tb)}[/dim]"
    console.print(Panel(body, title="[red]Error[/red]", border_style="red"))
