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

    result = run_experiment_from_config(config)
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
