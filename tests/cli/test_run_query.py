from typer.testing import CliRunner

from scripts.run_query import app

runner = CliRunner()


def test_invalid_output_flag_exits_1() -> None:
    result = runner.invoke(app, ["diabetes", "--output", "xml"])
    assert result.exit_code == 1


def test_missing_api_key_exits_1(monkeypatch) -> None:
    monkeypatch.setattr("scripts.run_query.settings.anthropic_api_key", "")
    result = runner.invoke(app, ["diabetes"])
    assert result.exit_code == 1
