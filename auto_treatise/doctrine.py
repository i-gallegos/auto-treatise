"""Per-doctrine paths and settings under ``doctrines/<doctrine>/``.

Paths are relative to the repository root, so the CLIs must be run from there.
"""

import json
from pathlib import Path

import tomli as tomllib

def load_config(doctrine: str) -> dict:
    with open(Path("doctrines") / doctrine / "config.toml", "rb") as f:
        return tomllib.load(f)


def load_split(doctrine: str, split_file: str) -> dict[str, list[int]]:
    with open(Path("doctrines") / doctrine / split_file) as f:
        return json.load(f)


def db_path(doctrine: str) -> str:
    return str(Path("doctrines") / doctrine / "data" / "citing_cases_clean.db")


def analysis_root(doctrine: str, test: bool) -> Path:
    root = Path("doctrines") / doctrine / "analysis"
    return root / "test" if test else root
