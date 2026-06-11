import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ktabforge.config.loader import load_yaml_file


@dataclass(frozen=True)
class ConfigValidationResult:
    valid: bool
    errors: list[str]


def _format_error_path(path_parts: list[Any]) -> str:
    if not path_parts:
        return "<root>"
    return ".".join(str(part) for part in path_parts)


def bundled_schema_path(schema_name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "schemas" / schema_name


def validate_loaded_config(config: Any, schema_path: Path) -> ConfigValidationResult:
    with Path(schema_path).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda error: list(error.path))
    messages = [
        f"{_format_error_path(list(error.path))}: {error.message}"
        for error in errors
    ]
    return ConfigValidationResult(valid=not messages, errors=messages)


def require_loaded_config_valid(config: Any, schema_path: Path) -> None:
    result = validate_loaded_config(config, schema_path)
    if result.valid:
        return
    raise ValueError("; ".join(result.errors))


def validate_config_file(config_path: Path, schema_path: Path) -> ConfigValidationResult:
    config = load_yaml_file(Path(config_path))
    return validate_loaded_config(config, Path(schema_path))
