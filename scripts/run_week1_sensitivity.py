"""
Run Week 1 sensitivity analysis.

Execute from the project root:

    python scripts/run_week1_sensitivity.py
"""

from pathlib import Path
import sys
from dataclasses import replace

import numpy as np

# Allow importing from src/ when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.week1_2d import (  # noqa: E402
    Object2D,
    Jet2D,
    Simulation2D,
    InitialCondition2D,
    TargetRegion2D,
    simulate_trajectory_2d,
)
from src.utils import save_sensitivity_plot  # noqa: E402


def run_single_case(
    obj: Object2D,
    jet: Jet2D,
    sim: Simulation2D,
    initial: InitialCondition2D,
    target: TargetRegion2D,
) -> float:
    """Run one simulation and return landing x-position."""
    result = simulate_trajectory_2d(
        obj=obj,
        jet=jet,
        sim=sim,
        initial=initial,
        target=target,
    )

    landing_x = result["landing_position"]

    if landing_x is None:
        return np.nan

    return landing_x


def main() -> None:
    base_obj = Object2D(
        mass=0.02,
        area=5.0e-4,
        drag_coefficient=1.0,
    )

    base_jet = Jet2D(
        force=(0.0, 0.03),
        t_on=0.35,
        duration=0.12,
    )

    sim = Simulation2D(
        dt=0.001,
        t_max=3.0,
        gravity=9.81,
        air_density=1.225,
    )

    initial = InitialCondition2D(
        position=(0.0, 0.5),
        velocity=(1.0, 0.0),
    )

    target = TargetRegion2D(
        x_min=1.0,
        x_max=1.5,
    )

    output_dir = PROJECT_ROOT / "results" / "week1" / "figures"

    sensitivity_ranges = {
        "mass": np.linspace(0.005, 0.05, 10),
        "drag_coefficient": np.linspace(0.2, 2.0, 10),
        "jet_force_y": np.linspace(0.0, 0.08, 10),
        "jet_t_on": np.linspace(0.0, 0.8, 10),
    }

    variation_summary = {}

    # 1. Mass sweep
    values = sensitivity_ranges["mass"]
    landing_positions = []

    for mass in values:
        obj = replace(base_obj, mass=float(mass))
        landing_x = run_single_case(obj, base_jet, sim, initial, target)
        landing_positions.append(landing_x)

    save_sensitivity_plot(
        x_values=values,
        landing_positions=landing_positions,
        xlabel="mass [kg]",
        save_path=output_dir / "sensitivity_mass.png",
        title="Landing Position Sensitivity to Mass",
    )

    variation_summary["mass"] = np.nanmax(landing_positions) - np.nanmin(landing_positions)

    # 2. Drag coefficient sweep
    values = sensitivity_ranges["drag_coefficient"]
    landing_positions = []

    for cd in values:
        obj = replace(base_obj, drag_coefficient=float(cd))
        landing_x = run_single_case(obj, base_jet, sim, initial, target)
        landing_positions.append(landing_x)

    save_sensitivity_plot(
        x_values=values,
        landing_positions=landing_positions,
        xlabel="drag coefficient Cd [-]",
        save_path=output_dir / "sensitivity_drag_coefficient.png",
        title="Landing Position Sensitivity to Drag Coefficient",
    )

    variation_summary["drag_coefficient"] = np.nanmax(landing_positions) - np.nanmin(landing_positions)

    # 3. Jet force sweep
    values = sensitivity_ranges["jet_force_y"]
    landing_positions = []

    for fy in values:
        jet = replace(base_jet, force=(0.0, float(fy)))
        landing_x = run_single_case(base_obj, jet, sim, initial, target)
        landing_positions.append(landing_x)

    save_sensitivity_plot(
        x_values=values,
        landing_positions=landing_positions,
        xlabel="jet force Fy [N]",
        save_path=output_dir / "sensitivity_jet_force.png",
        title="Landing Position Sensitivity to Jet Force",
    )

    variation_summary["jet_force_y"] = np.nanmax(landing_positions) - np.nanmin(landing_positions)

    # 4. Jet timing sweep
    values = sensitivity_ranges["jet_t_on"]
    landing_positions = []

    for t_on in values:
        jet = replace(base_jet, t_on=float(t_on))
        landing_x = run_single_case(base_obj, jet, sim, initial, target)
        landing_positions.append(landing_x)

    save_sensitivity_plot(
        x_values=values,
        landing_positions=landing_positions,
        xlabel="jet activation time t_on [s]",
        save_path=output_dir / "sensitivity_jet_timing.png",
        title="Landing Position Sensitivity to Jet Timing",
    )

    variation_summary["jet_t_on"] = np.nanmax(landing_positions) - np.nanmin(landing_positions)

    print("Week 1 sensitivity analysis completed.\n")
    print("Landing-position variation by parameter:")

    for name, variation in variation_summary.items():
        print(f"- {name}: {variation:.4f} m")

    most_sensitive = max(variation_summary, key=variation_summary.get)
    print(f"\nMost sensitive parameter in this sweep: {most_sensitive}")
    print(f"\nSaved plots to: {output_dir}")


if __name__ == "__main__":
    main()
