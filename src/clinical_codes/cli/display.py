from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from clinical_codes.schemas import CodeResult, SystemName


def render_results(
    console: Console,
    consolidated: dict[SystemName, list[CodeResult]],
    verbose: bool,
) -> None:
    console.print(Rule("Results"))
    if not consolidated:
        console.print("[dim]No results[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("System")
    table.add_column("Code")
    table.add_column("Display")
    if verbose:
        table.add_column("Score")

    for system, results in consolidated.items():
        for result in results:
            row = [system.value, result.code, result.display]
            if verbose:
                row.append(f"{result.score:.2f}")
            table.add_row(*row)

    console.print(table)
