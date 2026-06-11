from __future__ import annotations

import re

_SAFE_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def safe_path_segment(value: str, *, field: str) -> str:
    if "/" in value or "\\" in value or ".." in value or not _SAFE_PATH_SEGMENT.match(value):
        raise ValueError(f"Unsafe {field}: {value}")
    return value
