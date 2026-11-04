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


def md_cell(text) -> str:
    """Make text safe for one Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def rejected_items_md(metric: str, scored: dict) -> list[str]:
    """Report lines for one scored metric: the claims/statements the judge rejected, then the
    borderline ones, as tables. Falls back to the judge's reason when there are no details.

    `scored` is one metric's entry from score_case: {"score", "reason", "error", "details"?}.
    """
    details = scored.get("details") or {}
    faithfulness = metric == "faithfulness"
    key = "unsupported" if faithfulness else "irrelevant"
    label = "Claims not supported" if faithfulness else "Statements off the question"
    noun = "claim" if faithfulness else "statement"
    lines = []
    for items, title in [(details.get(key, []), f"{label} (lower the score)"),
                         (details.get("borderline", []),
                          "Borderline: partly supported, not counted against it")]:  # fmt: skip
        if items:
            lines += ["", f"{metric} - {title}:", "", f"| {noun} | judge's reason |", "|---|---|"]
            lines += [f"| {md_cell(c['text'] or '(not matched to a claim)')} "
                      f"| {md_cell(c['reason'] or '')} |" for c in items]  # fmt: skip
    if not details and scored.get("reason") and (scored.get("score") or 0) < 1:
        lines += ["", f"{metric}: {md_cell(scored['reason'])}"]
    return lines


def save_report(markdown: str, run_path: Path) -> Path:
    """Write a readable Markdown report next to a saved run (same name, .md)."""
    path = run_path.with_suffix(".md")
    path.write_text(markdown, encoding="utf-8")
    return path


def list_runs(kind: str) -> list[Path]:
    return sorted((RUNS_DIR / kind).glob("*.json"))


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
