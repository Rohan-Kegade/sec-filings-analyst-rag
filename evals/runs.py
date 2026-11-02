"""Saved eval runs, shared by every eval (one folder per eval under evals/runs/)."""

import json
import subprocess
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save(result: dict, kind: str) -> Path:
    """Write a run to evals/runs/<kind>/<timestamp>.json."""
    folder = RUNS_DIR / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result['timestamp'].replace(':', '-')}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def save_report(markdown: str, run_path: Path) -> Path:
    """Write a readable Markdown report next to a saved run (same name, .md)."""
    path = run_path.with_suffix(".md")
    path.write_text(markdown, encoding="utf-8")
    return path


def list_runs(kind: str) -> list[Path]:
    return sorted((RUNS_DIR / kind).glob("*.json"))


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
