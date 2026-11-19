#!/usr/bin/env python3
"""Apply one ReSeT overlay to its pinned upstream tracker checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from src.adapter_registry import load_registry, relative_path

ROOT = Path(__file__).resolve().parent / "src"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def copy_tree(source_root: Path, destination_root: Path) -> int:
    """Copy a source tree while preserving its relative layout."""
    copied = 0
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        if "__pycache__" in relative.parts or source.suffix in {".pyc", ".pyo"}:
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    args = parser.parse_args()

    registry = load_registry()
    if args.tracker not in registry:
        raise SystemExit(f"Unknown tracker: {args.tracker}")
    spec = registry[args.tracker]

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git checkout: {repo}")

    expected = spec["revision"]
    actual = git(repo, "rev-parse", "HEAD")
    if actual != expected and not args.allow_revision_mismatch:
        raise SystemExit(f"Revision mismatch: expected {expected}, found {actual}")

    status = git(repo, "status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("Checkout is not clean; use a fresh clone or pass --allow-dirty explicitly")

    for patch_value in spec["patches"]:
        patch_path = ROOT / relative_path(str(patch_value), f"{args.tracker}.patch")
        if not patch_path.is_file():
            raise SystemExit(f"Missing compatibility patch: {patch_path}")
        subprocess.run(
            ["git", "-C", str(repo), "apply", "--check", str(patch_path)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "apply", str(patch_path)],
            check=True,
        )

    profile_name = str(spec["profile"])
    profile = ROOT / "reset_core" / "profiles" / profile_name
    if not profile.is_dir():
        raise SystemExit(f"Missing shared core profile: {profile}")
    core_destination = repo / "lib" / "test" / "tracker" / "reset_core"
    copied = copy_tree(profile, core_destination)
    for shared_name in ("__init__.py", "adapter_api.py", "adapter_contract.py"):
        shutil.copy2(ROOT / "reset_core" / shared_name, core_destination / shared_name)
        copied += 1
    calibration = ROOT / "reset_core" / "calibration"
    if calibration.is_dir():
        copied += copy_tree(calibration, core_destination / "calibration")

    overlay = ROOT / relative_path(str(spec["overlay"]), f"{args.tracker}.overlay")
    if not overlay.is_dir():
        raise SystemExit(f"Missing tracker overlay: {overlay}")
    copied += copy_tree(overlay, repo)

    print(
        f"Applied {copied} files to {repo} "
        f"(shared core profile: {profile_name}; compatibility patches: {len(spec['patches'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
