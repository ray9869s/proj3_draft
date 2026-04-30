"""Placeholder entry point for the Week 2 3D demo."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.week2_3d import main


if __name__ == "__main__":
    # TODO: Replace with the actual Week 2 demo runner.
    main()
