from __future__ import annotations

from typing import Any


def render_eda_summary(
    *,
    eda_id: str,
    competition: str,
    focus: list[str],
    profile: dict[str, Any],
    watchlist: list[dict[str, Any]],
    feature_backlog: list[dict[str, Any]],
) -> str:
    target_distribution = profile.get("target_distribution", {})
    target_lines = "\n".join(
        f"- `{label}`: {count}" for label, count in sorted(target_distribution.items())
    )
    watchlist_lines = "\n".join(
        f"- `{item['severity']}` `{item['rule']}`: {item['message']}" for item in watchlist
    )
    backlog_lines = "\n".join(
        f"- `{item['feature_family']}`: {', '.join(item['source_columns'])}"
        for item in feature_backlog
    )
    focus_value = ", ".join(focus) if focus else "none"

    return (
        f"# EDA Summary\n\n"
        f"- `eda_id`: `{eda_id}`\n"
        f"- `competition`: `{competition}`\n"
        f"- `focus`: {focus_value}\n"
        f"- `train_rows`: {profile['row_count']['train']}\n"
        f"- `test_rows`: {profile['row_count']['test']}\n"
        f"- `feature_count`: {len(profile.get('feature_columns', []))}\n"
        f"- `numeric_features`: {len(profile.get('numeric_columns', []))}\n"
        f"- `categorical_features`: {len(profile.get('categorical_columns', []))}\n\n"
        f"## Target Distribution\n\n"
        f"{target_lines or '- unavailable'}\n\n"
        f"## Leakage Watchlist\n\n"
        f"{watchlist_lines or '- none'}\n\n"
        f"## Feature Backlog Seeds\n\n"
        f"{backlog_lines or '- none'}\n"
    )
