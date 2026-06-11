from pathlib import Path
from typing import Annotated

import typer

from ktabforge.config.schema import validate_config_file

app = typer.Typer(help="Evidence-first tools for Kaggle tabular workflows.")

ConfigOption = Annotated[
    Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)
]
SchemaOption = Annotated[
    Path, typer.Option("--schema", exists=True, file_okay=True, dir_okay=False)
]
DataDirOption = Annotated[Path, typer.Option("--data-dir", file_okay=False, dir_okay=True)]
ArtifactRootOption = Annotated[
    Path, typer.Option("--artifact-root", file_okay=False, dir_okay=True)
]
CompetitionOption = Annotated[str, typer.Option("--competition")]
ExperimentIdOption = Annotated[str, typer.Option("--experiment-id")]
TargetOption = Annotated[str, typer.Option("--target")]
IdColumnOption = Annotated[str, typer.Option("--id-column")]
SplitsOption = Annotated[int, typer.Option("--n-splits")]
SeedOption = Annotated[int, typer.Option("--seed")]
TopNOption = Annotated[int, typer.Option("--top-n")]
MaxRunsOption = Annotated[int | None, typer.Option("--max-runs")]
DryRunOption = Annotated[bool, typer.Option("--dry-run")]


@app.command("validate-config")
def validate_config(
    config: ConfigOption,
    schema: SchemaOption,
) -> None:
    """Validate a YAML config file against a JSON Schema file."""
    result = validate_config_file(config, schema)
    if result.valid:
        typer.echo("valid")
        return

    for error in result.errors:
        typer.echo(error, err=True)
    raise typer.Exit(code=1)


@app.command("smoke")
def smoke(
    data_dir: DataDirOption,
    artifact_root: ArtifactRootOption,
    competition: CompetitionOption,
    experiment_id: ExperimentIdOption,
    target: TargetOption,
    id_column: IdColumnOption,
    n_splits: SplitsOption = 3,
    seed: SeedOption = 42,
) -> None:
    """Run a smoke evidence pipeline."""
    from ktabforge.pipeline.evidence import run_smoke_evidence

    run_smoke_evidence(
        data_dir=data_dir,
        artifact_root=artifact_root,
        competition=competition,
        experiment_id=experiment_id,
        target=target,
        id_column=id_column,
        n_splits=n_splits,
        seed=seed,
    )
    typer.echo("smoke complete")


@app.command("run")
def run(config: ConfigOption) -> None:
    """Run an experiment from a YAML config file."""
    from ktabforge.pipeline.runner import run_experiment_from_config

    try:
        result = run_experiment_from_config(config)
    except (FileExistsError, TypeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    status = getattr(result, "status", None) or "completed"
    typer.echo(f"{status}")


@app.command("compare")
def compare(
    artifact_root: ArtifactRootOption,
    competition: CompetitionOption,
    top_n: TopNOption = 20,
) -> None:
    """Compare completed experiments for a competition."""
    from ktabforge.reports.compare import compare_experiments

    frame = compare_experiments(
        artifact_root=artifact_root,
        competition=competition,
        top_n=top_n,
    )
    if frame.empty:
        typer.echo("no completed experiments with OOF evidence found")
        return

    typer.echo(frame.to_csv(index=False).rstrip())


@app.command("ensemble")
def ensemble(config: ConfigOption) -> None:
    """Run an OOF-backed ensemble from a YAML config file."""
    from ktabforge.ensembles.runner import run_ensemble_from_config

    try:
        result = run_ensemble_from_config(config)
    except (FileExistsError, TypeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(result.status)


@app.command("factory")
def factory(
    config: ConfigOption,
    dry_run: DryRunOption = False,
    max_runs: MaxRunsOption = None,
) -> None:
    """Run or plan a matrix-driven candidate experiment factory."""
    from ktabforge.factory.runner import run_factory_from_config

    try:
        result = run_factory_from_config(
            config,
            dry_run=dry_run,
            max_runs=max_runs,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"status: {result.status} planned: {result.planned_count} "
        f"completed: {result.completed_count} failed: {result.failed_count} "
        f"report: {result.report_path}"
    )
    if dry_run and result.plan_path.exists():
        import pandas as pd

        plan = pd.read_csv(result.plan_path)
        if "experiment_id" in plan.columns:
            typer.echo("planned_experiments: " + ", ".join(plan["experiment_id"].astype(str)))


@app.command("stack-preflight")
def stack_preflight(config: ConfigOption) -> None:
    """Prepare stack matrices and selection artifacts without training a stacker."""
    from ktabforge.stacking.runner import run_stacking_preflight

    try:
        result = run_stacking_preflight(config)
    except (FileExistsError, TypeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"{result.status} accepted: {result.accepted_count} rejected: "
        f"{result.rejected_count} report: {result.selection_report_path}"
    )
