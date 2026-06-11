from __future__ import annotations

from pathlib import Path

import pandas as pd

from ktabforge.metrics.scoring import metric_higher_is_better


def write_candidate_report(
    path: str | Path,
    *,
    factory_id: str,
    metric_name: str,
    summary: pd.DataFrame,
    top_n: int,
) -> Path:
    output_path = Path(path)
    completed = summary[summary["status"] == "completed"].copy()
    if "oof_score" in completed.columns:
        completed["oof_score"] = pd.to_numeric(completed["oof_score"], errors="coerce")
        completed = completed.sort_values(
            "oof_score",
            ascending=not metric_higher_is_better(metric_name),
            na_position="last",
        )
    top = completed.head(top_n)
    failed = summary[summary["status"] == "failed"].copy()

    body = [
        "# Candidate Factory Report",
        "",
        "## Summary",
        "",
        f"- Factory: {factory_id}",
        f"- Completed: {len(completed)}",
        f"- Failed: {len(failed)}",
        f"- Metric: {metric_name}",
        "",
        "## Top Candidates",
        "",
        "| rank | experiment_id | model_family | preset | feature_set | seed | "
        "oof_score | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        body.append(
            "| {rank} | {experiment_id} | {model_family} | {model_preset} | "
            "{feature_set} | {seed} | {oof_score} | {status} |".format(
                rank=rank,
                experiment_id=_markdown_cell(row.get("experiment_id", "")),
                model_family=_markdown_cell(row.get("model_family", "")),
                model_preset=_markdown_cell(row.get("model_preset", "")),
                feature_set=_markdown_cell(row.get("feature_set", "")),
                seed=_markdown_cell(row.get("seed", "")),
                oof_score=_markdown_cell(row.get("oof_score", "")),
                status=_markdown_cell(row.get("status", "")),
            )
        )

    body.extend(
        [
            "",
            "## Failures",
            "",
            "| experiment_id | config_hash | config_path | error_type | error_message |",
        ]
    )
    body.append("| --- | --- | --- | --- | --- |")
    for _, row in failed.iterrows():
        body.append(
            f"| {_markdown_cell(row.get('experiment_id', ''))} | "
            f"{_markdown_cell(row.get('config_hash', ''))} | "
            f"{_markdown_cell(row.get('config_path', ''))} | "
            f"{_markdown_cell(row.get('error_type', ''))} | "
            f"{_markdown_cell(row.get('error_message', ''))} |"
        )

    body.extend(["", "## Next Step", "", "Recommended ensemble candidate_ids:"])
    for experiment_id in top["experiment_id"].tolist() if not top.empty else []:
        body.append(f"- {experiment_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return output_path


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\r\n", "\n").replace("\n", "<br>").replace("|", "\\|")
