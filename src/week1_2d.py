"""
Week 1: 2D point-mass air-jet sorting simulator.

This module implements:
- gravity
- quadratic drag
- finite-duration air-jet force
- explicit Euler time integration
- landing detection
- target-region classification
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class Object2D:
    """Physical properties of a 2D point-mass object."""
    mass: float = 0.02                 # kg
    area: float = 5.0e-4               # m^2
    drag_coefficient: float = 1.0      # dimensionless


@dataclass
class Jet2D:
    """Finite-duration air-jet force."""
    force: Tuple[float, float] = (0.0, 0.03)  # N, (Fx, Fy)
    t_on: float = 0.5                         # s
    duration: float = 0.1                     # s


@dataclass
class Simulation2D:
    """Numerical simulation parameters."""
    dt: float = 0.001                 # s
    t_max: float = 5.0                # s
    gravity: float = 9.81             # m/s^2
    air_density: float = 1.225        # kg/m^3


@dataclass
class InitialCondition2D:
    """Initial position and velocity."""
    position: Tuple[float, float] = (0.0, 0.5)  # m
    velocity: Tuple[float, float] = (1.0, 0.0)  # m/s


@dataclass
class TargetRegion2D:
    """
    Target landing region.

    For Week 1, we define the target using x-position only.
    The object succeeds if x_min <= x_landing <= x_max.
    """
    x_min: float = 1.5
    x_max: float = 2.5


def compute_gravity_force(obj: Object2D, sim: Simulation2D) -> np.ndarray:
    """Return gravity force vector."""
    return np.array([0.0, -obj.mass * sim.gravity], dtype=float)


def compute_drag_force(
    velocity: np.ndarray,
    obj: Object2D,
    sim: Simulation2D,
) -> np.ndarray:
    """
    Return quadratic drag force.

    F_drag = -0.5 * rho * Cd * A * |v| * v
    """
    speed = np.linalg.norm(velocity)

    if speed < 1.0e-12:
        return np.zeros(2, dtype=float)

    return -0.5 * sim.air_density * obj.drag_coefficient * obj.area * speed * velocity


def compute_jet_force(t: float, jet: Jet2D) -> np.ndarray:
    """
    Return air-jet force.

    The jet is active only during:
    t_on <= t <= t_on + duration
    """
    if jet.t_on <= t <= jet.t_on + jet.duration:
        return np.array(jet.force, dtype=float)

    return np.zeros(2, dtype=float)


def simulate_trajectory_2d(
    obj: Object2D,
    jet: Jet2D,
    sim: Simulation2D,
    initial: InitialCondition2D,
    target: Optional[TargetRegion2D] = None,
) -> Dict[str, Any]:
    """
    Simulate 2D object trajectory until landing or t_max.

    Returns
    -------
    result : dict
        time: array, shape (N,)
        position: array, shape (N, 2)
        velocity: array, shape (N, 2)
        force_gravity: array, shape (N, 2)
        force_drag: array, shape (N, 2)
        force_jet: array, shape (N, 2)
        landing_position: float or None
        landing_time: float or None
        success: bool or None
    """
    position = np.array(initial.position, dtype=float)
    velocity = np.array(initial.velocity, dtype=float)

    time_history = []
    position_history = []
    velocity_history = []
    gravity_history = []
    drag_history = []
    jet_history = []

    landing_position = None
    landing_time = None

    n_steps = int(sim.t_max / sim.dt) + 1

    for step in range(n_steps):
        t = step * sim.dt

        # Store current state
        time_history.append(t)
        position_history.append(position.copy())
        velocity_history.append(velocity.copy())

        # Compute forces
        f_gravity = compute_gravity_force(obj, sim)
        f_drag = compute_drag_force(velocity, obj, sim)
        f_jet = compute_jet_force(t, jet)

        gravity_history.append(f_gravity.copy())
        drag_history.append(f_drag.copy())
        jet_history.append(f_jet.copy())

        # Landing detection
        if position[1] <= 0.0 and step > 0:
            landing_position = position[0]
            landing_time = t
            break

        # Explicit Euler integration
        total_force = f_gravity + f_drag + f_jet
        acceleration = total_force / obj.mass

        velocity = velocity + acceleration * sim.dt
        position = position + velocity * sim.dt

    time_array = np.array(time_history)
    position_array = np.array(position_history)
    velocity_array = np.array(velocity_history)

    success = None
    if target is not None and landing_position is not None:
        success = target.x_min <= landing_position <= target.x_max

    return {
        "time": time_array,
        "position": position_array,
        "velocity": velocity_array,
        "force_gravity": np.array(gravity_history),
        "force_drag": np.array(drag_history),
        "force_jet": np.array(jet_history),
        "landing_position": landing_position,
        "landing_time": landing_time,
        "success": success,
        "object": obj,
        "jet": jet,
        "simulation": sim,
        "initial": initial,
        "target": target,
    }
