"""Shared naming rules for experiment artifacts.

The trainer and the notebooks must derive artifact paths from a W&B run name in
exactly the same way. Previously the trainer sanitized the run name for the
filesystem while the notebook used the raw name, so any run name containing a
character outside ``[A-Za-z0-9_-]`` (for example ``beta-0.1``) produced a plot
the notebook could not find.
"""

from pathlib import Path

# Characters kept verbatim. Everything else, including ".", becomes "-", which
# matches the file names already written by earlier runs.
ALLOWED_CHARACTERS = "-_"


def safe_run_name(run_name: str) -> str:
    """Return the filesystem-safe form of ``run_name`` used for artifacts."""
    cleaned = "".join(
        character if character.isalnum() or character in ALLOWED_CHARACTERS else "-"
        for character in run_name
    )
    return cleaned.strip("-") or "run"


def comparison_plot_path(run_name: str, results_dir: Path | str = "results") -> Path:
    """Return the path of the solution-comparison figure for ``run_name``."""
    return Path(results_dir) / f"{safe_run_name(run_name)}_comparison.png"
