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


def test_proposal_validate_cli_invokes_runner_and_prints_status_fields(tmp_path):
    config_path = tmp_path / "proposal.yaml"
    config_path.write_text("proposal: {}\n", encoding="utf-8")

    calls: list[Path] = []

    def fake_validate_proposal_config(path: Path):
        calls.append(path)
        return types.SimpleNamespace(
            status="valid",
            proposal_id="p06-freq-count",
            artifact_dir=Path("artifacts") / "proposals" / "churn_tiny" / "p06-freq-count",
        )

    fake_package = types.ModuleType("ktabforge.proposals")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    fake_runner = types.ModuleType("ktabforge.proposals.runner")
    fake_runner.validate_proposal_config = fake_validate_proposal_config

    old_package = sys.modules.get("ktabforge.proposals")
    old_runner = sys.modules.get("ktabforge.proposals.runner")
    sys.modules["ktabforge.proposals"] = fake_package
    sys.modules["ktabforge.proposals.runner"] = fake_runner
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["proposal", "validate", "--config", str(config_path)])
    finally:
        if old_package is None:
            sys.modules.pop("ktabforge.proposals", None)
        else:
            sys.modules["ktabforge.proposals"] = old_package
        if old_runner is None:
            sys.modules.pop("ktabforge.proposals.runner", None)
        else:
            sys.modules["ktabforge.proposals.runner"] = old_runner

    assert result.exit_code == 0, result.stdout
    assert calls == [config_path]
    assert "valid" in result.stdout
    assert "p06-freq-count" in result.stdout
    assert "artifacts" in result.stdout


def test_proposal_register_cli_invokes_runner_and_prints_status_fields(tmp_path):
    config_path = tmp_path / "proposal.yaml"
    config_path.write_text("proposal: {}\n", encoding="utf-8")

    calls: list[Path] = []

    def fake_register_proposal_from_config(path: Path):
        calls.append(path)
        return types.SimpleNamespace(
            status="registered",
            proposal_id="p06-freq-count",
            artifact_dir=Path("artifacts") / "proposals" / "churn_tiny" / "p06-freq-count",
        )

    fake_package = types.ModuleType("ktabforge.proposals")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    fake_runner = types.ModuleType("ktabforge.proposals.runner")
    fake_runner.register_proposal_from_config = fake_register_proposal_from_config

    old_package = sys.modules.get("ktabforge.proposals")
    old_runner = sys.modules.get("ktabforge.proposals.runner")
    sys.modules["ktabforge.proposals"] = fake_package
    sys.modules["ktabforge.proposals.runner"] = fake_runner
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["proposal", "register", "--config", str(config_path)])
    finally:
        if old_package is None:
            sys.modules.pop("ktabforge.proposals", None)
        else:
            sys.modules["ktabforge.proposals"] = old_package
        if old_runner is None:
            sys.modules.pop("ktabforge.proposals.runner", None)
        else:
            sys.modules["ktabforge.proposals.runner"] = old_runner

    assert result.exit_code == 0, result.stdout
    assert calls == [config_path]
    assert "registered" in result.stdout
    assert "p06-freq-count" in result.stdout
    assert "artifacts" in result.stdout


def test_feature_build_cli_invokes_runner_and_prints_status_fields(tmp_path):
    config_path = tmp_path / "feature_build.yaml"
    config_path.write_text("feature_build: {}\n", encoding="utf-8")

    calls: list[Path] = []

    def fake_run_feature_build_from_config(path: Path):
        calls.append(path)
        return types.SimpleNamespace(
            status="completed",
            feature_build_id="p06-feature-build",
            artifact_dir=Path("artifacts") / "features" / "churn_tiny" / "p06-feature-build",
        )

    fake_features = types.ModuleType("ktabforge.features")
    fake_features.__path__ = []  # type: ignore[attr-defined]
    fake_pipeline = types.ModuleType("ktabforge.features.pipeline")
    fake_pipeline.run_feature_build_from_config = fake_run_feature_build_from_config

    old_features = sys.modules.get("ktabforge.features")
    old_pipeline = sys.modules.get("ktabforge.features.pipeline")
    sys.modules["ktabforge.features"] = fake_features
    sys.modules["ktabforge.features.pipeline"] = fake_pipeline
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["feature", "build", "--config", str(config_path)])
    finally:
        if old_features is None:
            sys.modules.pop("ktabforge.features", None)
        else:
            sys.modules["ktabforge.features"] = old_features
        if old_pipeline is None:
            sys.modules.pop("ktabforge.features.pipeline", None)
        else:
            sys.modules["ktabforge.features.pipeline"] = old_pipeline

    assert result.exit_code == 0, result.stdout
    assert calls == [config_path]
    assert "completed" in result.stdout
    assert "p06-feature-build" in result.stdout
    assert "artifacts" in result.stdout
