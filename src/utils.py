"""Shared utilities: constants, config loading, and small IO helpers."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

# Canonical label space used throughout the project.
LABELS: List[str] = ["positive", "negative", "uncertain", "not_mentioned"]

# Canonical pathology keys (also used as column prefixes: <pathology>_gold).
PATHOLOGIES: List[str] = [
    "pneumonia",
    "pleural_effusion",
    "pneumothorax",
    "cardiomegaly",
]


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
