from pathlib import Path

from ktabforge.config.schema import validate_config_file


def test_competition_example_matches_schema():
    repo_root = Path(__file__).resolve().parents[2]

    result = validate_config_file(
        repo_root / "configs" / "competition.example.yaml",
        repo_root / "configs" / "schemas" / "competition.schema.json",
    )

    assert result.valid is True
    assert result.errors == []

