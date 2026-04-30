"""
Run basic Week 1 2D trajectory simulations.

Execute from the project root:

    python scripts/run_week1_basic.py
"""

from pathlib import Path
import sys

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
from src.utils import save_trajectory_plot, print_landing_summary  # noqa: E402


def main() -> None:
    obj = Object2D(
        mass=0.02,
        area=5.0e-4,
        drag_coefficient=1.0,
    )

    jet = Jet2D(
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

    target = TargetRegion2D(
        x_min=1.0,
        x_max=1.5,
    )

    initial_conditions = [
        InitialCondition2D(position=(0.0, 0.5), velocity=(0.8, 0.0)),
        InitialCondition2D(position=(0.0, 0.5), velocity=(1.0, 0.0)),
        InitialCondition2D(position=(0.0, 0.5), velocity=(1.2, 0.0)),
        InitialCondition2D(position=(0.0, 0.5), velocity=(1.4, 0.0)),
    ]

    labels = [
        "vx0 = 0.8 m/s",
        "vx0 = 1.0 m/s",
        "vx0 = 1.2 m/s",
        "vx0 = 1.4 m/s",
    ]

    results = []

    for initial in initial_conditions:
        result = simulate_trajectory_2d(
            obj=obj,
            jet=jet,
            sim=sim,
            initial=initial,
            target=target,
        )
        results.append(result)

    save_path = PROJECT_ROOT / "results" / "week1" / "figures" / "week1_basic_trajectories.png"

    save_trajectory_plot(
        results=results,
        labels=labels,
        save_path=save_path,
        title="Week 1 Basic 2D Trajectories",
        target_region=(target.x_min, target.x_max),
    )

    print_landing_summary(results, labels)
    print(f"\nSaved trajectory plot to: {save_path}")


if __name__ == "__main__":
    main()
