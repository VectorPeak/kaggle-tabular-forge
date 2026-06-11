import sys
import types
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


def test_eda_scan_cli_invokes_runner_and_prints_status_fields(tmp_path):
    config_path = tmp_path / "eda.yaml"
    config_path.write_text("eda: {}\n", encoding="utf-8")

    calls: list[Path] = []

    def fake_run_eda_scan_from_config(path: Path):
        calls.append(path)
        return types.SimpleNamespace(
            status="completed",
            eda_id="p06-eda-smoke",
            artifact_dir=Path("artifacts") / "eda" / "churn_tiny" / "p06-eda-smoke",
        )

    fake_package = types.ModuleType("ktabforge.eda")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    fake_runner = types.ModuleType("ktabforge.eda.runner")
    fake_runner.run_eda_scan_from_config = fake_run_eda_scan_from_config

    old_package = sys.modules.get("ktabforge.eda")
    old_runner = sys.modules.get("ktabforge.eda.runner")
    sys.modules["ktabforge.eda"] = fake_package
    sys.modules["ktabforge.eda.runner"] = fake_runner
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["eda", "scan", "--config", str(config_path)])
    finally:
        if old_package is None:
            sys.modules.pop("ktabforge.eda", None)
        else:
            sys.modules["ktabforge.eda"] = old_package
        if old_runner is None:
            sys.modules.pop("ktabforge.eda.runner", None)
        else:
            sys.modules["ktabforge.eda.runner"] = old_runner

    assert result.exit_code == 0, result.stdout
    assert calls == [config_path]
    assert "completed" in result.stdout
    assert "p06-eda-smoke" in result.stdout
    assert "artifacts" in result.stdout
