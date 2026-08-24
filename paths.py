"""Shared project paths — portable across machines."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
MODELS = ROOT / "models"
FIGURES = DOCS / "figures"


def resolve_data_dir() -> Path:
    """Locate C-MAPSS train/test/RUL text files."""
    candidates = [
        ROOT / "data",
        ROOT,
        Path(r"C:\kaggle\input\cmapss-jet-engine-simulated-data"),
        Path("/kaggle/input/cmapss-jet-engine-simulated-data"),
    ]
    for d in candidates:
        if d.is_dir() and list(d.glob("train_FD*.txt")):
            return d
    raise FileNotFoundError(
        "Could not find C-MAPSS data (train_FD*.txt). "
        "Place files under ./data/ or the project root."
    )


DATA_DIR = resolve_data_dir()
