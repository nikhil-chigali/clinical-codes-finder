import io

from rich.console import Console

from clinical_codes.schemas import CodeResult, SystemName


def _make_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, highlight=False), buf


def _make_result(
    system: SystemName, code: str, display: str, score: float
) -> CodeResult:
    return CodeResult(system=system, code=code, display=display, score=score, raw={})


# ── render_results ────────────────────────────────────────────────────────────


def test_render_results_shows_system_and_code() -> None:
    from clinical_codes.cli.display import render_results

    console, buf = _make_console()
    consolidated = {
        SystemName.RXNORM: [
            _make_result(SystemName.RXNORM, "860975", "metFORMIN 500 MG Tablet", 1.0)
        ]
    }
    render_results(console, consolidated, verbose=False)
    output = buf.getvalue()
    assert "860975" in output
    assert "RXNORM" in output


def test_render_results_empty_shows_no_results() -> None:
    from clinical_codes.cli.display import render_results

    console, buf = _make_console()
    render_results(console, {}, verbose=False)
    output = buf.getvalue()
    assert "No results" in output


def test_render_results_verbose_shows_score_column() -> None:
    from clinical_codes.cli.display import render_results

    console, buf = _make_console()
    consolidated = {
        SystemName.RXNORM: [
            _make_result(SystemName.RXNORM, "860975", "metFORMIN 500 MG Tablet", 0.75)
        ]
    }
    render_results(console, consolidated, verbose=True)
    output = buf.getvalue()
    assert "Score" in output
    assert "0.75" in output
