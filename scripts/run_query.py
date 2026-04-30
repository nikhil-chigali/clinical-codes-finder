from __future__ import annotations

import asyncio
import json
import traceback
from typing import Annotated

import typer
from rich.console import Console
from rich.rule import Rule

from clinical_codes.cli.display import render_error, render_results, update_status
from clinical_codes.config import settings
from clinical_codes.graph.builder import build_graph, make_initial_state
from clinical_codes.schemas import CodeResult, SystemName

app = typer.Typer(add_completion=False)

_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


@app.command()
def run(
    query: Annotated[str, typer.Argument(help="Clinical term to look up")],
    output: Annotated[
        str, typer.Option(help="Output format: table or json")
    ] = "table",
    verbose: Annotated[
        bool, typer.Option(help="Show scores, iterations, tracebacks")
    ] = False,
) -> None:
    if output not in ("table", "json"):
        err_console = Console(stderr=True)
        render_error(err_console, f"Invalid --output value '{output}'. Must be 'table' or 'json'.")
        raise typer.Exit(code=1)

    json_mode = output == "json"
    console = Console(stderr=json_mode)

    if not settings.anthropic_api_key:
        render_error(console, "ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        raise typer.Exit(code=1)

    asyncio.run(_run_async(query, console, json_mode, verbose))


async def _run_async(
    query: str,
    console: Console,
    json_mode: bool,
    verbose: bool,
) -> None:
    graph = _get_graph()
    initial_state = make_initial_state(query)

    consolidated: dict[SystemName, list[CodeResult]] = {}
    attempt_history: list = []
    summary = ""
    summary_started = False

    try:
        with console.status("Planning...") as status:
            async for event in graph.astream(
                initial_state,
                stream_mode=["updates", "messages"],
            ):
                mode, data = event
                if mode == "updates":
                    node_name = next(iter(data))
                    node_data = data[node_name]
                    if isinstance(node_data, dict):
                        if "consolidated" in node_data:
                            consolidated = node_data["consolidated"]
                        if "attempt_history" in node_data:
                            attempt_history.extend(node_data["attempt_history"])
                        if "summary" in node_data:
                            summary = node_data["summary"]
                    if not summary_started:
                        update_status(status, node_name)
                elif mode == "messages" and not json_mode:
                    chunk, meta = data
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if meta.get("langgraph_node") == "summarizer" and content:
                        if not summary_started:
                            status.stop()
                            console.print(Rule("Summary"))
                            summary_started = True
                        console.out(content, end="")
    except Exception:
        tb = traceback.format_exc() if verbose else None
        render_error(console, "An error occurred while running the query.", tb=tb)
        raise typer.Exit(code=1)

    if summary_started:
        console.print("\n")

    if json_mode:
        result = {
            "query": query,
            "summary": summary,
            "results": {
                system.value: [
                    {"code": r.code, "display": r.display, "score": r.score}
                    for r in results
                ]
                for system, results in consolidated.items()
            },
        }
        print(json.dumps(result, indent=2))
        return

    render_results(console, consolidated, verbose=verbose)

    if verbose and attempt_history:
        console.print(Rule("Iterations"))
        for attempt in attempt_history:
            systems = ", ".join(s.value for s in attempt.planner_output.selected_systems)
            console.print(f"  Iteration {attempt.iteration}: {systems}")


if __name__ == "__main__":
    app()
