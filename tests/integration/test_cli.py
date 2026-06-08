from pathlib import Path

from typer.testing import CliRunner

from ktabforge.cli import app


def test_validate_config_cli_accepts_competition_example():
    repo_root = Path(__file__).resolve().parents[2]
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "validate-config",
            "--config",
            str(repo_root / "configs" / "competition.example.yaml"),
            "--schema",
            str(repo_root / "configs" / "schemas" / "competition.schema.json"),
        ],
    )

    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()

