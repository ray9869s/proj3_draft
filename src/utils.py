"""
Utility functions for plotting and file management.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_trajectory_plot(
    results: List[Dict],
    labels: Sequence[str],
    save_path: str | Path,
    title: str = "2D Air-Jet Sorting Trajectories",
    target_region: Optional[tuple[float, float]] = None,
) -> None:
    """
    Save a 2D trajectory plot.

    Parameters
    ----------
    results:
        List of trajectory result dictionaries from simulate_trajectory_2d.
    labels:
        Labels for each trajectory.
    save_path:
        Output image path.
    target_region:
        Optional tuple (x_min, x_max) for landing target region.
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    plt.figure(figsize=(8, 5))

    for result, label in zip(results, labels):
        pos = result["position"]
        plt.plot(pos[:, 0], pos[:, 1], label=label)

        landing_x = result.get("landing_position")
        if landing_x is not None:
            plt.scatter([landing_x], [0.0], s=30)

    if target_region is not None:
        x_min, x_max = target_region
        plt.axvspan(x_min, x_max, alpha=0.2, label="target region")

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_sensitivity_plot(
    x_values: Sequence[float],
    landing_positions: Sequence[float],
    xlabel: str,
    save_path: str | Path,
    title: str,
) -> None:
    """Save landing-position sensitivity plot."""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    x_values = np.asarray(x_values)
    landing_positions = np.asarray(landing_positions)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_values, landing_positions, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("landing x-position [m]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_landing_summary(results: List[Dict], labels: Sequence[str]) -> None:
    """Print landing position and success/failure for each simulation case."""
    for result, label in zip(results, labels):
        landing_x = result.get("landing_position")
        landing_t = result.get("landing_time")
        success = result.get("success")

        if landing_x is None:
            print(f"{label}: object did not land within t_max")
            continue

        if success is None:
            print(f"{label}: landing x = {landing_x:.4f} m, landing time = {landing_t:.4f} s")
        else:
            status = "SUCCESS" if success else "FAIL"
            print(
                f"{label}: landing x = {landing_x:.4f} m, "
                f"landing time = {landing_t:.4f} s, {status}"
            )
