"""
Week 2: 3D rigid-body air-jet sorting simulator.

Coordinate convention:
    x: conveyor belt direction
    y: conveyor belt width direction / air-jet nozzle position direction
    z: vertical direction

Main modeling assumptions:
    - The object is a rigid body represented by discrete surface points.
    - The air jet exists only in a finite x-zone:
          x_start <= x <= x_start + x_width
    - Inside that x-zone, the jet is strongest at the center and smoothly weakens
      toward the zone boundaries using a raised-cosine window.
    - In the y-z plane, the jet has a Gaussian profile.
    - Local surface normals are used to weight the force received by each surface point.
    - Landing is detected at the first contact between the lowest surface point and
      the landing plane.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Object3D:
    """3D rigid-body object represented by surface points."""
    name: str
    object_type: str
    mass: float
    drag_coefficient: float
    surface_points_body: np.ndarray
    surface_normals_body: np.ndarray
    area_weights: np.ndarray
    inertia_body: np.ndarray
    size_x: float
    size_y: float
    size_z: float
    rod_length: Optional[float] = None
    rod_radius: Optional[float] = None


@dataclass
class Jet3D:
    """
    Finite-region air-jet model.

    x-direction profile:
        x_start <= x <= x_start + x_width

        A raised-cosine profile is used:
            x_profile = 0 at x_start
            x_profile = 1 at the center
            x_profile = 0 at x_start + x_width

    y-z profile:
        Gaussian centered at (y_center, z_center)

    Jet direction:
        angle_deg is measured from +x toward +z.
    """
    umax: float = 25.0
    x_start: float = 0.00
    x_width: float = 0.05
    y_center: float = 0.0
    z_center: float = 0.2
    sigma: float = 0.08
    angle_deg: float = 45.0
    t_on: float = 0.15
    duration: float = 0.15
    noise_std: float = 0.0


@dataclass
class Simulation3D:
    dt: float = 0.001
    t_max: float = 3.0
    gravity: float = 9.81
    air_density: float = 1.225
    landing_z: float = 0.0


@dataclass
class InitialCondition3D:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.2)
    velocity: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TargetRegion3D:
    x_min: float = 0.30
    x_max: float = 0.80


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm_q = np.linalg.norm(q)
    if norm_q < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm_q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 x q2. Quaternion format: [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    w, x, y, z = q

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_degrees_to_quaternion(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> Tuple[float, float, float, float]:
    """
    Convert roll-pitch-yaw angles in degrees to quaternion [w, x, y, z].

    Roll  : rotation around x-axis
    Pitch : rotation around y-axis
    Yaw   : rotation around z-axis
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = normalize_quaternion(np.array([w, x, y, z], dtype=float))
    return tuple(float(v) for v in q)


def update_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
    dqdt = 0.5 * quaternion_multiply(q, omega_quat)
    return normalize_quaternion(q + dqdt * dt)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    safe_norms = np.where(norms < 1.0e-12, 1.0, norms)
    return vectors / safe_norms[:, None]


def compute_box_inertia(mass: float, lx: float, ly: float, lz: float) -> np.ndarray:
    ixx = (1.0 / 12.0) * mass * (ly ** 2 + lz ** 2)
    iyy = (1.0 / 12.0) * mass * (lx ** 2 + lz ** 2)
    izz = (1.0 / 12.0) * mass * (lx ** 2 + ly ** 2)
    return np.diag([ixx, iyy, izz])


def compute_cylinder_inertia_x_axis(mass: float, length: float, radius: float) -> np.ndarray:
    """
    Solid cylinder inertia tensor aligned with body x-axis.

    Ixx = 1/2 m r^2
    Iyy = Izz = 1/12 m (3r^2 + L^2)
    """
    ixx = 0.5 * mass * radius ** 2
    iyy = (1.0 / 12.0) * mass * (3.0 * radius ** 2 + length ** 2)
    izz = iyy
    return np.diag([ixx, iyy, izz])


def create_surface_grid_plate(
    lx: float,
    ly: float,
    lz: float,
    nx: int = 7,
    ny: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-lx / 2.0, lx / 2.0, nx)
    ys = np.linspace(-ly / 2.0, ly / 2.0, ny)

    points = []
    normals = []

    for z, normal_z in [(-lz / 2.0, -1.0), (lz / 2.0, 1.0)]:
        for x in xs:
            for y in ys:
                points.append([x, y, z])
                normals.append([0.0, 0.0, normal_z])

    points = np.array(points, dtype=float)
    normals = np.array(normals, dtype=float)

    total_area = 2.0 * lx * ly
    area_weights = np.ones(len(points), dtype=float) * (total_area / len(points))
    return points, normals, area_weights


def create_surface_grid_rod(
    length: float,
    radius: float,
    n_length: int = 17,
    n_theta: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create representative surface points and normals for a cylindrical rod
    aligned with body x-axis.
    """
    xs = np.linspace(-length / 2.0, length / 2.0, n_length)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    points = []
    normals = []

    # Side surface
    for x in xs:
        for theta in thetas:
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points.append([x, y, z])
            normals.append([0.0, np.cos(theta), np.sin(theta)])

    # End caps
    cap_radii = np.linspace(0.0, radius, 5)
    for x, normal_x in [(-length / 2.0, -1.0), (length / 2.0, 1.0)]:
        for rr in cap_radii:
            for theta in thetas:
                y = rr * np.cos(theta)
                z = rr * np.sin(theta)
                points.append([x, y, z])
                normals.append([normal_x, 0.0, 0.0])

    points = np.array(points, dtype=float)
    normals = np.array(normals, dtype=float)

    side_area = 2.0 * np.pi * radius * length
    cap_area = 2.0 * np.pi * radius ** 2
    total_area = side_area + cap_area
    area_weights = np.ones(len(points), dtype=float) * (total_area / len(points))

    return points, normals, area_weights


def create_irregular_flake_points(
    lx: float,
    ly: float,
    lz: float,
    n_points: int = 120,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create representative surface points and approximate outward normals
    for an irregular flake.

    This is not a true mesh. Normals are approximated by radial directions
    from the body-frame center to each point.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-lx / 2.0, lx / 2.0, size=n_points)
    y = rng.uniform(-ly / 2.0, ly / 2.0, size=n_points)
    z = rng.uniform(-lz / 2.0, lz / 2.0, size=n_points)

    x = x * rng.uniform(0.75, 1.15, size=n_points)
    y = y * rng.uniform(0.75, 1.15, size=n_points)
    z = z * rng.uniform(0.60, 1.40, size=n_points)

    points = np.column_stack([x, y, z])
    normals = normalize_vectors(points.copy())

    # For points very close to center, default normal is +z.
    near_center = np.linalg.norm(points, axis=1) < 1.0e-12
    normals[near_center] = np.array([0.0, 0.0, 1.0], dtype=float)

    approx_area = 2.0 * lx * ly
    area_weights = np.ones(n_points, dtype=float) * (approx_area / n_points)
    return points, normals, area_weights


def create_object_3d(
    object_type: str = "plate",
    mass: float = 0.05,
    size_x: float = 0.10,
    size_y: float = 0.10,
    size_z: float = 0.01,
    drag_coefficient: float = 1.0,
    rod_length: Optional[float] = None,
    rod_radius: Optional[float] = None,
    seed: int = 1,
) -> Object3D:
    object_type = object_type.lower()

    if object_type == "plate":
        points, normals, areas = create_surface_grid_plate(
            lx=size_x,
            ly=size_y,
            lz=size_z,
            nx=7,
            ny=7,
        )
        inertia = compute_box_inertia(mass, size_x, size_y, size_z)

        return Object3D(
            name="thin plate",
            object_type="plate",
            mass=mass,
            drag_coefficient=drag_coefficient,
            surface_points_body=points,
            surface_normals_body=normals,
            area_weights=areas,
            inertia_body=inertia,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
        )

    if object_type == "rod":
        if rod_length is None:
            rod_length = size_x
        if rod_radius is None:
            rod_radius = max(size_y, size_z) / 2.0

        size_x = rod_length
        size_y = 2.0 * rod_radius
        size_z = 2.0 * rod_radius

        points, normals, areas = create_surface_grid_rod(
            length=rod_length,
            radius=rod_radius,
            n_length=17,
            n_theta=16,
        )
        inertia = compute_cylinder_inertia_x_axis(mass, rod_length, rod_radius)

        return Object3D(
            name="cylindrical rod",
            object_type="rod",
            mass=mass,
            drag_coefficient=drag_coefficient,
            surface_points_body=points,
            surface_normals_body=normals,
            area_weights=areas,
            inertia_body=inertia,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            rod_length=rod_length,
            rod_radius=rod_radius,
        )

    if object_type == "irregular":
        points, normals, areas = create_irregular_flake_points(
            lx=size_x,
            ly=size_y,
            lz=size_z,
            n_points=120,
            seed=seed,
        )
        inertia = compute_box_inertia(mass, size_x, size_y, size_z)

        return Object3D(
            name="irregular flake",
            object_type="irregular",
            mass=mass,
            drag_coefficient=drag_coefficient,
            surface_points_body=points,
            surface_normals_body=normals,
            area_weights=areas,
            inertia_body=inertia,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
        )

    raise ValueError(f"Unknown object_type: {object_type}")


def transform_surface_points(
    position: np.ndarray,
    quaternion: np.ndarray,
    points_body: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    r_vectors = points_body @ rotation_matrix.T
    points_world = position[None, :] + r_vectors
    return points_world, r_vectors, rotation_matrix


def transform_normals(
    quaternion: np.ndarray,
    normals_body: np.ndarray,
) -> np.ndarray:
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    normals_world = normals_body @ rotation_matrix.T
    return normalize_vectors(normals_world)


def get_jet_direction(jet: Jet3D) -> np.ndarray:
    theta = np.deg2rad(jet.angle_deg)
    direction = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=float)
    norm_direction = np.linalg.norm(direction)

    if norm_direction < 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    return direction / norm_direction


def smooth_x_window(points_x: np.ndarray, x_start: float, x_width: float) -> np.ndarray:
    """
    Raised-cosine x-profile.

    x_profile = 0 at x_start
    x_profile = 1 at x_start + x_width/2
    x_profile = 0 at x_start + x_width

    Outside the interval, x_profile = 0.
    """
    width = max(x_width, 1.0e-9)
    s = (points_x - x_start) / width

    profile = np.zeros_like(points_x, dtype=float)
    inside = (s >= 0.0) & (s <= 1.0)
    profile[inside] = 0.5 * (1.0 - np.cos(2.0 * np.pi * s[inside]))

    return profile


def gaussian_jet_velocity(
    points_world: np.ndarray,
    t: float,
    jet: Jet3D,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    n_points = points_world.shape[0]
    velocities = np.zeros((n_points, 3), dtype=float)

    if not (jet.t_on <= t <= jet.t_on + jet.duration):
        return velocities

    x = points_world[:, 0]
    y = points_world[:, 1]
    z = points_world[:, 2]

    x_profile = smooth_x_window(
        points_x=x,
        x_start=jet.x_start,
        x_width=jet.x_width,
    )

    yz_sigma = max(jet.sigma, 1.0e-6)
    yz_profile = np.exp(
        -((y - jet.y_center) ** 2 + (z - jet.z_center) ** 2)
        / (2.0 * yz_sigma ** 2)
    )

    profile = x_profile * yz_profile

    noise_factor = 1.0
    if jet.noise_std > 0.0 and rng is not None:
        noise_factor = 1.0 + rng.normal(0.0, jet.noise_std)

    jet_direction = get_jet_direction(jet)
    u_mag = jet.umax * noise_factor * profile
    velocities = u_mag[:, None] * jet_direction[None, :]
    return velocities


def compute_local_surface_velocity(
    velocity_com: np.ndarray,
    angular_velocity: np.ndarray,
    r_vectors: np.ndarray,
) -> np.ndarray:
    rotational_velocity = np.cross(angular_velocity[None, :], r_vectors)
    return velocity_com[None, :] + rotational_velocity


def compute_jet_forces_and_torque(
    obj: Object3D,
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    angular_velocity: np.ndarray,
    t: float,
    jet: Jet3D,
    sim: Simulation3D,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    points_world, r_vectors, rotation_matrix = transform_surface_points(
        position=position,
        quaternion=quaternion,
        points_body=obj.surface_points_body,
    )

    normals_world = transform_normals(
        quaternion=quaternion,
        normals_body=obj.surface_normals_body,
    )

    u_jet = gaussian_jet_velocity(
        points_world=points_world,
        t=t,
        jet=jet,
        rng=rng,
    )

    v_surface = compute_local_surface_velocity(
        velocity_com=velocity,
        angular_velocity=angular_velocity,
        r_vectors=r_vectors,
    )

    u_rel = u_jet - v_surface
    speed_rel = np.linalg.norm(u_rel, axis=1)

    u_rel_hat = np.zeros_like(u_rel)
    active = speed_rel > 1.0e-12
    u_rel_hat[active] = u_rel[active] / speed_rel[active, None]

    # Surface-normal weighting:
    # Surfaces more aligned with the local relative flow receive stronger force.
    # This is still simplified, but it is more physical than treating all points equally.
    exposure = np.abs(np.sum(u_rel_hat * normals_world, axis=1))

    local_forces = (
        0.5
        * sim.air_density
        * obj.drag_coefficient
        * obj.area_weights[:, None]
        * exposure[:, None]
        * speed_rel[:, None]
        * u_rel
    )

    total_force = np.sum(local_forces, axis=0)
    local_torques = np.cross(r_vectors, local_forces)
    total_torque = np.sum(local_torques, axis=0)

    return {
        "points_world": points_world,
        "r_vectors": r_vectors,
        "normals_world": normals_world,
        "rotation_matrix": rotation_matrix,
        "u_jet": u_jet,
        "v_surface": v_surface,
        "u_rel": u_rel,
        "local_forces": local_forces,
        "total_force": total_force,
        "local_torques": local_torques,
        "total_torque": total_torque,
    }


def compute_body_drag_force(
    velocity: np.ndarray,
    obj: Object3D,
    sim: Simulation3D,
    reference_area: float,
) -> np.ndarray:
    """
    Simplified whole-body drag.

    This does not yet include orientation-dependent projected area.
    """
    speed = np.linalg.norm(velocity)

    if speed < 1.0e-12:
        return np.zeros(3, dtype=float)

    return (
        -0.5
        * sim.air_density
        * obj.drag_coefficient
        * reference_area
        * speed
        * velocity
    )


def simulate_rigid_body_3d(
    obj: Object3D,
    jet: Jet3D,
    sim: Simulation3D,
    initial: InitialCondition3D,
    target: Optional[TargetRegion3D] = None,
    reference_area: Optional[float] = None,
    seed: int = 1,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    position = np.array(initial.position, dtype=float)
    velocity = np.array(initial.velocity, dtype=float)
    quaternion = normalize_quaternion(np.array(initial.quaternion, dtype=float))
    angular_velocity = np.array(initial.angular_velocity, dtype=float)

    if reference_area is None:
        reference_area = float(np.sum(obj.area_weights) / 2.0)

    time_history = []
    position_history = []
    velocity_history = []
    quaternion_history = []
    angular_velocity_history = []

    force_gravity_history = []
    force_drag_history = []
    force_jet_history = []
    force_total_history = []
    torque_jet_history = []

    jet_impulse = np.zeros(3, dtype=float)
    angular_impulse = np.zeros(3, dtype=float)

    max_angular_speed = 0.0
    landing_position = None
    landing_time = None

    n_steps = int(sim.t_max / sim.dt) + 1

    for step in range(n_steps):
        t = step * sim.dt

        force_gravity = np.array([0.0, 0.0, -obj.mass * sim.gravity], dtype=float)

        jet_data = compute_jet_forces_and_torque(
            obj=obj,
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
            t=t,
            jet=jet,
            sim=sim,
            rng=rng,
        )

        force_jet = jet_data["total_force"]
        torque_jet = jet_data["total_torque"]

        force_drag = compute_body_drag_force(
            velocity=velocity,
            obj=obj,
            sim=sim,
            reference_area=reference_area,
        )

        force_total = force_gravity + force_drag + force_jet

        time_history.append(t)
        position_history.append(position.copy())
        velocity_history.append(velocity.copy())
        quaternion_history.append(quaternion.copy())
        angular_velocity_history.append(angular_velocity.copy())

        force_gravity_history.append(force_gravity.copy())
        force_drag_history.append(force_drag.copy())
        force_jet_history.append(force_jet.copy())
        force_total_history.append(force_total.copy())
        torque_jet_history.append(torque_jet.copy())

        jet_impulse += force_jet * sim.dt
        angular_impulse += torque_jet * sim.dt

        angular_speed = np.linalg.norm(angular_velocity)
        max_angular_speed = max(max_angular_speed, angular_speed)

        points_world_for_landing, _, _ = transform_surface_points(
            position=position,
            quaternion=quaternion,
            points_body=obj.surface_points_body,
        )

        lowest_surface_z = np.min(points_world_for_landing[:, 2])

        if lowest_surface_z <= sim.landing_z and step > 0:
            landing_position = position.copy()
            landing_time = t
            break

        acceleration = force_total / obj.mass

        velocity = velocity + acceleration * sim.dt
        position = position + velocity * sim.dt

        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        inertia_world = rotation_matrix @ obj.inertia_body @ rotation_matrix.T
        inertia_world_inv = np.linalg.pinv(inertia_world)

        gyroscopic_term = np.cross(
            angular_velocity,
            inertia_world @ angular_velocity,
        )

        angular_acceleration = inertia_world_inv @ (torque_jet - gyroscopic_term)

        angular_velocity = angular_velocity + angular_acceleration * sim.dt
        quaternion = update_quaternion(quaternion, angular_velocity, sim.dt)

    time_array = np.array(time_history)
    position_array = np.array(position_history)
    velocity_array = np.array(velocity_history)
    quaternion_array = np.array(quaternion_history)
    angular_velocity_array = np.array(angular_velocity_history)

    has_landed = landing_position is not None

    success = None
    if target is not None and has_landed:
        success = target.x_min <= landing_position[0] <= target.x_max

    final_points_world, _, _ = transform_surface_points(
        position=position_array[-1],
        quaternion=quaternion_array[-1],
        points_body=obj.surface_points_body,
    )

    return {
        "time": time_array,
        "position": position_array,
        "velocity": velocity_array,
        "quaternion": quaternion_array,
        "angular_velocity": angular_velocity_array,
        "force_gravity": np.array(force_gravity_history),
        "force_drag": np.array(force_drag_history),
        "force_jet": np.array(force_jet_history),
        "force_total": np.array(force_total_history),
        "torque_jet": np.array(torque_jet_history),
        "landing_position": landing_position,
        "landing_time": landing_time,
        "has_landed": has_landed,
        "final_time": time_array[-1],
        "final_position": position_array[-1],
        "success": success,
        "jet_impulse": jet_impulse,
        "angular_impulse": angular_impulse,
        "max_angular_speed": max_angular_speed,
        "final_points_world": final_points_world,
        "object": obj,
        "jet": jet,
        "simulation": sim,
        "initial": initial,
        "target": target,
        "reference_area": reference_area,
    }


def compute_hit_offset(initial_position: Tuple[float, float, float], jet: Jet3D) -> float:
    x_com = initial_position[0]
    y_com = initial_position[1]
    z_com = initial_position[2]

    x_end = jet.x_start + max(jet.x_width, 0.0)

    if jet.x_start <= x_com <= x_end:
        dx = 0.0
    elif x_com < jet.x_start:
        dx = jet.x_start - x_com
    else:
        dx = x_com - x_end

    return float(
        np.sqrt(
            dx ** 2
            + (y_com - jet.y_center) ** 2
            + (z_com - jet.z_center) ** 2
        )
    )
