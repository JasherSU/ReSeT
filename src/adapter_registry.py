"""Load and validate the single adapter registry."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent
REGISTRY_FILE = ROOT / "adapters.json"
REQUIRED_FIELDS = {
    "repository",
    "revision",
    "profile",
    "patches",
    "overlay",
    "tracker_module",
    "tracker_class",
    "parameter",
    "runner",
}


def relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"{field} must be a repository-relative path: {value}")
    return path


def load_registry() -> Dict[str, Dict[str, Any]]:
    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not registry:
        raise RuntimeError("Adapter registry must be a non-empty object")
    for name, entry in registry.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise RuntimeError(f"Invalid adapter name: {name}")
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise RuntimeError(f"Adapter {name} is missing fields: {sorted(missing)}")
        if not str(entry["repository"]).startswith("https://"):
            raise RuntimeError(f"Adapter {name} must use an HTTPS repository")
        if not re.fullmatch(r"[0-9a-f]{40}", str(entry["revision"])):
            raise RuntimeError(f"Adapter {name} must pin a full commit hash")
        relative_path(str(entry["overlay"]), f"{name}.overlay")
        relative_path(str(entry["runner"]), f"{name}.runner")
        profile = relative_path(str(entry["profile"]), f"{name}.profile")
        if len(profile.parts) != 1:
            raise RuntimeError(f"Adapter {name} profile must be a profile name")
        if not isinstance(entry["patches"], list):
            raise RuntimeError(f"Adapter {name} patches must be a list")
        for index, patch in enumerate(entry["patches"]):
            relative_path(str(patch), f"{name}.patches[{index}]")
    return registry
