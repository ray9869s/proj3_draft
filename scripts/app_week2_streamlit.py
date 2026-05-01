"""
Interactive Week 2 3D Rigid-Body Air-Jet Simulator using Streamlit.

Run from the project root:

    streamlit run scripts/app_week2_streamlit.py
"""

from pathlib import Path
import sys
import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.week2_3d import (  # noqa: E402
    Jet3D,
    Simulation3D,
    InitialCondition3D,
    TargetRegion3D,
    create_object_3d,
    simulate_rigid_body_3d,
    compute_hit_offset,
    transform_surface_points,
    euler_degrees_to_quaternion,
)


# ---------------------------------------------------------------------
# Default values and reset utilities
# ---------------------------------------------------------------------

DEFAULTS = {
    # Object
    "object_type": "plate",
    "mass": 0.050,
    "drag_coefficient": 1.0,
    "size_x": 0.10,
    "size_y": 0.10,
    "size_z": 0.01,
    "rod_length": 0.15,
    "rod_radius": 0.015,

    # Initial motion
    "x0": 0.0,
    "y0": 0.0,
    "z0": 0.20,
    "vc": 1.0,
    "vy_initial": 0.0,
    "vz_initial": 0.0,

    # Initial orientation
    "roll0": 0.0,
    "pitch0": 0.0,
    "yaw0": 0.0,

    # Initial angular velocity
    "omega_x0": 0.0,
    "omega_y0": 0.0,
    "omega_z0": 0.0,

    # Air jet
    "umax": 25.0,
    "jet_x_start": 0.00,
    "jet_x_width": 0.05,
    "jet_y_center": 0.0,
    "jet_z_center": 0.20,
    "sigma": 0.08,
    "jet_angle_deg": 45.0,
    "jet_t_on": 0.15,
    "jet_duration": 0.15,
    "noise_std": 0.00,

    # Simulation
    "dt": 0.001,
    "t_max": 2.0,
    "gravity": 9.81,
    "air_density": 1.225,
    "landing_z": 0.0,

    # Target
    "target_x_min": 0.30,
    "target_x_max": 0.80,

    # Plot axes
    "use_fixed_axes": True,
    "x_plot_min": -0.10,
    "x_plot_max": 1.50,
    "y_plot_min": -0.50,
    "y_plot_max": 0.50,
    "z_plot_min": 0.00,
    "z_plot_max": 0.60,

    # Options
    "seed": 1,
    "show_surface_points": False,

    # Animation
    "animation_max_frames": 80,
    "animation_fps": 12,
    "animation_dpi": 100,
}


SECTION_KEYS = {
    "object": [
        "object_type",
        "mass",
        "drag_coefficient",
        "size_x",
        "size_y",
        "size_z",
        "rod_length",
        "rod_radius",
    ],
    "initial_motion": [
        "x0",
        "y0",
        "z0",
        "vc",
        "vy_initial",
        "vz_initial",
    ],
    "initial_orientation": [
        "roll0",
        "pitch0",
        "yaw0",
    ],
    "initial_angular_velocity": [
        "omega_x0",
        "omega_y0",
        "omega_z0",
    ],
    "air_jet": [
        "umax",
        "jet_x_start",
        "jet_x_width",
        "jet_y_center",
        "jet_z_center",
        "sigma",
        "jet_angle_deg",
        "jet_t_on",
        "jet_duration",
        "noise_std",
    ],
    "simulation": [
        "dt",
        "t_max",
        "gravity",
        "air_density",
        "landing_z",
    ],
    "target": [
        "target_x_min",
        "target_x_max",
    ],
    "plot_axes": [
        "use_fixed_axes",
        "x_plot_min",
        "x_plot_max",
        "y_plot_min",
        "y_plot_max",
        "z_plot_min",
        "z_plot_max",
    ],
    "options": [
        "seed",
        "show_surface_points",
        "animation_max_frames",
        "animation_fps",
        "animation_dpi",
    ],
}


def initialize_session_defaults():
    """Initialize Streamlit session state with default values."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_keys(keys):
    """Reset selected session-state keys to their default values."""
    for key in keys:
        st.session_state[key] = DEFAULTS[key]


def reset_all():
    """Reset all user-controlled parameters and clear stored results."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    for key in ["last_result", "last_parameter_json", "last_csv_data", "last_gif_bytes"]:
        if key in st.session_state:
            del st.session_state[key]


initialize_session_defaults()


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Week 2 3D Rigid-Body Air-Jet Simulator",
    layout="wide",
)

st.title("Week 2: Interactive 3D Rigid-Body Air-Jet Simulator")

st.write(
    """
    This app simulates a 3D rigid object moving in the conveyor direction and being hit by
    a finite-duration air jet. The object is represented by surface points, so the jet
    can generate both total force and torque.
    """
)

st.info(
    "Coordinate convention: x = conveyor belt direction, y = belt width / jet nozzle position direction, "
    "z = vertical direction. The target is defined by landing x-position. "
    "The finite air-jet zone uses a smooth x-profile and a Gaussian y-z profile."
)

main_run_button = st.button(
    "Run 3D Simulation",
    type="primary",
    key="main_run_button",
)


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("Run")

sidebar_run_button = st.sidebar.button(
    "Run 3D Simulation",
    type="primary",
    key="sidebar_run_button",
)

st.sidebar.button(
    "Reset All Parameters",
    key="reset_all_parameters_button",
    on_click=reset_all,
)

st.sidebar.caption(
    "Adjust the parameters below, then click Run. "
    "Section reset buttons return only that section to its default values."
)

st.sidebar.divider()


# ---------------------------------------------------------------------
# Object parameters
# ---------------------------------------------------------------------

st.sidebar.header("Object Parameters")

st.sidebar.button(
    "Reset Object Parameters",
    key="reset_object_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["object"],),
)

object_type = st.sidebar.selectbox(
    "Object type",
    options=["plate", "rod", "irregular"],
    index=["plate", "rod", "irregular"].index(st.session_state["object_type"]),
    key="object_type",
)

mass = st.sidebar.slider(
    "Mass m [kg]",
    min_value=0.005,
    max_value=0.300,
    step=0.005,
    format="%.3f",
    key="mass",
)

drag_coefficient = st.sidebar.slider(
    "Effective drag coefficient Cd [-]",
    min_value=0.1,
    max_value=3.0,
    step=0.05,
    key="drag_coefficient",
)

rod_length = None
rod_radius = None

if object_type == "rod":
    rod_length = st.sidebar.slider(
        "Rod length [m]",
        min_value=0.02,
        max_value=0.50,
        step=0.005,
        format="%.3f",
        key="rod_length",
    )

    rod_radius = st.sidebar.slider(
        "Rod radius [m]",
        min_value=0.002,
        max_value=0.10,
        step=0.001,
        format="%.3f",
        key="rod_radius",
    )

    size_x = rod_length
    size_y = 2.0 * rod_radius
    size_z = 2.0 * rod_radius

    st.sidebar.caption(
        "For rod, the cylinder axis is aligned with the body x-axis. "
        "The app internally uses size_x = length and size_y = size_z = 2*radius."
    )

else:
    size_x = st.sidebar.slider(
        "Object size in x [m]",
        min_value=0.02,
        max_value=0.30,
        step=0.005,
        format="%.3f",
        key="size_x",
    )

    size_y = st.sidebar.slider(
        "Object size in y [m]",
        min_value=0.005,
        max_value=0.30,
        step=0.005,
        format="%.3f",
        key="size_y",
    )

    size_z = st.sidebar.slider(
        "Object size in z [m]",
        min_value=0.002,
        max_value=0.10,
        step=0.002,
        format="%.3f",
        key="size_z",
    )


# ---------------------------------------------------------------------
# Initial motion
# ---------------------------------------------------------------------

st.sidebar.header("Initial Motion")

st.sidebar.button(
    "Reset Initial Motion",
    key="reset_initial_motion_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["initial_motion"],),
)

x0 = st.sidebar.slider(
    "Initial COM x [m]",
    min_value=-0.5,
    max_value=1.5,
    step=0.01,
    key="x0",
)

y0 = st.sidebar.slider(
    "Initial COM y [m]",
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    key="y0",
)

z0 = st.sidebar.slider(
    "Initial COM z [m]",
    min_value=0.02,
    max_value=1.0,
    step=0.01,
    key="z0",
)

vc = st.sidebar.slider(
    "Conveyor speed vx [m/s]",
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    key="vc",
)

vy_initial = st.sidebar.slider(
    "Initial belt-width velocity vy [m/s]",
    min_value=-2.0,
    max_value=2.0,
    step=0.05,
    key="vy_initial",
)

vz_initial = st.sidebar.slider(
    "Initial vertical velocity vz [m/s]",
    min_value=-2.0,
    max_value=2.0,
    step=0.05,
    key="vz_initial",
)


# ---------------------------------------------------------------------
# Initial orientation
# ---------------------------------------------------------------------

st.sidebar.header("Initial Orientation")

st.sidebar.button(
    "Reset Initial Orientation",
    key="reset_initial_orientation_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["initial_orientation"],),
)

roll0 = st.sidebar.slider(
    "Initial roll angle [deg]",
    min_value=-180.0,
    max_value=180.0,
    step=1.0,
    key="roll0",
)

pitch0 = st.sidebar.slider(
    "Initial pitch angle [deg]",
    min_value=-180.0,
    max_value=180.0,
    step=1.0,
    key="pitch0",
)

yaw0 = st.sidebar.slider(
    "Initial yaw angle [deg]",
    min_value=-180.0,
    max_value=180.0,
    step=1.0,
    key="yaw0",
)

st.sidebar.caption(
    "Roll = rotation around x, pitch = rotation around y, yaw = rotation around z."
)


# ---------------------------------------------------------------------
# Initial angular velocity
# ---------------------------------------------------------------------

st.sidebar.header("Initial Angular Velocity")

st.sidebar.button(
    "Reset Initial Angular Velocity",
    key="reset_initial_angular_velocity_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["initial_angular_velocity"],),
)

omega_x0 = st.sidebar.slider(
    "Initial omega_x [rad/s]",
    min_value=-50.0,
    max_value=50.0,
    step=1.0,
    key="omega_x0",
)

omega_y0 = st.sidebar.slider(
    "Initial omega_y [rad/s]",
    min_value=-50.0,
    max_value=50.0,
    step=1.0,
    key="omega_y0",
)

omega_z0 = st.sidebar.slider(
    "Initial omega_z [rad/s]",
    min_value=-50.0,
    max_value=50.0,
    step=1.0,
    key="omega_z0",
)


# ---------------------------------------------------------------------
# Air jet
# ---------------------------------------------------------------------

st.sidebar.header("Finite Air-Jet Zone")

st.sidebar.button(
    "Reset Air-Jet Parameters",
    key="reset_air_jet_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["air_jet"],),
)

umax = st.sidebar.slider(
    "Jet maximum velocity Umax [m/s]",
    min_value=0.0,
    max_value=80.0,
    step=1.0,
    key="umax",
)

jet_x_start = st.sidebar.slider(
    "Jet start x_start near conveyor end [m]",
    min_value=-0.5,
    max_value=2.0,
    step=0.01,
    key="jet_x_start",
)

jet_x_width = st.sidebar.slider(
    "Jet thickness x_width [m]",
    min_value=0.005,
    max_value=0.50,
    step=0.005,
    format="%.3f",
    key="jet_x_width",
)

st.sidebar.caption(
    "The jet exists from x_start to x_start + x_width. "
    "It is strongest at the center of this interval and smoothly weakens toward the boundaries."
)

jet_y_center = st.sidebar.slider(
    "Jet center yj [m]",
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    key="jet_y_center",
)

jet_z_center = st.sidebar.slider(
    "Jet center zj [m]",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="jet_z_center",
)

sigma = st.sidebar.slider(
    "Jet width sigma in y-z plane [m]",
    min_value=0.01,
    max_value=0.30,
    step=0.005,
    format="%.3f",
    key="sigma",
)

jet_angle_deg = st.sidebar.slider(
    "Jet angle relative to +x [deg]",
    min_value=-30.0,
    max_value=120.0,
    step=1.0,
    key="jet_angle_deg",
)

st.sidebar.caption(
    "Angle guide: 0 deg = +x direction, 45 deg = forward/upward, 90 deg = +z direction."
)

jet_t_on = st.sidebar.slider(
    "Jet activation time t_on [s]",
    min_value=0.0,
    max_value=2.0,
    step=0.01,
    key="jet_t_on",
)

jet_duration = st.sidebar.slider(
    "Jet duration dt_jet [s]",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="jet_duration",
)

noise_std = st.sidebar.slider(
    "Jet noise std [-]",
    min_value=0.0,
    max_value=0.50,
    step=0.01,
    key="noise_std",
)


# ---------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------

st.sidebar.header("Simulation Parameters")

st.sidebar.button(
    "Reset Simulation Parameters",
    key="reset_simulation_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["simulation"],),
)

dt = st.sidebar.slider(
    "Time step dt [s]",
    min_value=0.0005,
    max_value=0.01,
    step=0.0005,
    format="%.4f",
    key="dt",
)

t_max = st.sidebar.slider(
    "Maximum simulation time [s]",
    min_value=0.5,
    max_value=5.0,
    step=0.1,
    key="t_max",
)

gravity = st.sidebar.slider(
    "Gravity g [m/s2]",
    min_value=0.0,
    max_value=20.0,
    step=0.01,
    key="gravity",
)

air_density = st.sidebar.slider(
    "Air density rho [kg/m3]",
    min_value=0.0,
    max_value=2.0,
    step=0.005,
    key="air_density",
)

landing_z = st.sidebar.slider(
    "Landing plane z [m]",
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    key="landing_z",
)


# ---------------------------------------------------------------------
# Target region
# ---------------------------------------------------------------------

st.sidebar.header("Target Region")

st.sidebar.button(
    "Reset Target Region",
    key="reset_target_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["target"],),
)

target_x_min = st.sidebar.slider(
    "Target x_min [m]",
    min_value=-1.0,
    max_value=5.0,
    step=0.05,
    key="target_x_min",
)

target_x_max = st.sidebar.slider(
    "Target x_max [m]",
    min_value=-1.0,
    max_value=5.0,
    step=0.05,
    key="target_x_max",
)

if target_x_max < target_x_min:
    st.sidebar.warning("Target x_max should be larger than x_min.")


# ---------------------------------------------------------------------
# Plot axis limits
# ---------------------------------------------------------------------

st.sidebar.header("Plot Axis Limits")

st.sidebar.button(
    "Reset Plot Axis Limits",
    key="reset_plot_axes_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["plot_axes"],),
)

use_fixed_axes = st.sidebar.checkbox(
    "Use fixed plot axes",
    key="use_fixed_axes",
)

x_plot_min = st.sidebar.number_input(
    "Plot x_min [m]",
    step=0.05,
    key="x_plot_min",
)

x_plot_max = st.sidebar.number_input(
    "Plot x_max [m]",
    step=0.05,
    key="x_plot_max",
)

y_plot_min = st.sidebar.number_input(
    "Plot y_min [m]",
    step=0.05,
    key="y_plot_min",
)

y_plot_max = st.sidebar.number_input(
    "Plot y_max [m]",
    step=0.05,
    key="y_plot_max",
)

z_plot_min = st.sidebar.number_input(
    "Plot z_min [m]",
    step=0.05,
    key="z_plot_min",
)

z_plot_max = st.sidebar.number_input(
    "Plot z_max [m]",
    step=0.05,
    key="z_plot_max",
)

axis_limits = None

if use_fixed_axes:
    axis_limits = {
        "x": (x_plot_min, x_plot_max),
        "y": (y_plot_min, y_plot_max),
        "z": (z_plot_min, z_plot_max),
    }

    if x_plot_max <= x_plot_min:
        st.sidebar.warning("Plot x_max should be larger than x_min.")
    if y_plot_max <= y_plot_min:
        st.sidebar.warning("Plot y_max should be larger than y_min.")
    if z_plot_max <= z_plot_min:
        st.sidebar.warning("Plot z_max should be larger than z_min.")


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------

st.sidebar.header("Options")

st.sidebar.button(
    "Reset Options",
    key="reset_options_button",
    on_click=reset_keys,
    args=(SECTION_KEYS["options"],),
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    step=1,
    key="seed",
)

show_surface_points = st.sidebar.checkbox(
    "Show surface points",
    key="show_surface_points",
)

st.sidebar.subheader("Animation Export")

animation_max_frames = st.sidebar.slider(
    "Maximum GIF frames",
    min_value=20,
    max_value=160,
    step=10,
    key="animation_max_frames",
)

animation_fps = st.sidebar.slider(
    "GIF FPS",
    min_value=5,
    max_value=30,
    step=1,
    key="animation_fps",
)

animation_dpi = st.sidebar.slider(
    "GIF DPI",
    min_value=60,
    max_value=160,
    step=10,
    key="animation_dpi",
)

run_button = main_run_button or sidebar_run_button


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def result_to_dataframe(result):
    time = result["time"]
    position = result["position"]
    velocity = result["velocity"]
    angular_velocity = result["angular_velocity"]
    force_jet = result["force_jet"]
    force_drag = result["force_drag"]
    force_gravity = result["force_gravity"]
    force_total = result["force_total"]
    torque_jet = result["torque_jet"]

    data = {
        "time_s": time,
        "x_m": position[:, 0],
        "y_m": position[:, 1],
        "z_m": position[:, 2],
        "vx_m_per_s": velocity[:, 0],
        "vy_m_per_s": velocity[:, 1],
        "vz_m_per_s": velocity[:, 2],
        "omega_x_rad_per_s": angular_velocity[:, 0],
        "omega_y_rad_per_s": angular_velocity[:, 1],
        "omega_z_rad_per_s": angular_velocity[:, 2],
        "F_jet_x_N": force_jet[:, 0],
        "F_jet_y_N": force_jet[:, 1],
        "F_jet_z_N": force_jet[:, 2],
        "F_drag_x_N": force_drag[:, 0],
        "F_drag_y_N": force_drag[:, 1],
        "F_drag_z_N": force_drag[:, 2],
        "F_gravity_x_N": force_gravity[:, 0],
        "F_gravity_y_N": force_gravity[:, 1],
        "F_gravity_z_N": force_gravity[:, 2],
        "F_total_x_N": force_total[:, 0],
        "F_total_y_N": force_total[:, 1],
        "F_total_z_N": force_total[:, 2],
        "tau_jet_x_Nm": torque_jet[:, 0],
        "tau_jet_y_Nm": torque_jet[:, 1],
        "tau_jet_z_Nm": torque_jet[:, 2],
    }

    return pd.DataFrame(data)


def make_parameter_dict(hit_offset):
    object_dict = {
        "type": object_type,
        "mass_kg": mass,
        "drag_coefficient": drag_coefficient,
        "size_x_m": size_x,
        "size_y_m": size_y,
        "size_z_m": size_z,
    }

    if object_type == "rod":
        object_dict["rod_length_m"] = rod_length
        object_dict["rod_radius_m"] = rod_radius

    return {
        "object": object_dict,
        "initial_condition": {
            "position_m": [x0, y0, z0],
            "velocity_m_per_s": [vc, vy_initial, vz_initial],
            "orientation_deg": {
                "roll": roll0,
                "pitch": pitch0,
                "yaw": yaw0,
            },
            "angular_velocity_rad_per_s": [omega_x0, omega_y0, omega_z0],
        },
        "jet": {
            "umax_m_per_s": umax,
            "x_start_m": jet_x_start,
            "x_width_m": jet_x_width,
            "x_end_m": jet_x_start + jet_x_width,
            "x_profile": "raised_cosine",
            "y_center_m": jet_y_center,
            "z_center_m": jet_z_center,
            "sigma_yz_m": sigma,
            "angle_deg": jet_angle_deg,
            "t_on_s": jet_t_on,
            "duration_s": jet_duration,
            "noise_std": noise_std,
            "initial_offset_from_jet_region_m": hit_offset,
        },
        "simulation": {
            "dt_s": dt,
            "t_max_s": t_max,
            "gravity_m_per_s2": gravity,
            "air_density_kg_per_m3": air_density,
            "landing_z_m": landing_z,
        },
        "target_region": {
            "x_min_m": target_x_min,
            "x_max_m": target_x_max,
        },
        "plot_axis_limits": {
            "use_fixed_axes": use_fixed_axes,
            "x": [x_plot_min, x_plot_max],
            "y": [y_plot_min, y_plot_max],
            "z": [z_plot_min, z_plot_max],
        },
        "animation": {
            "max_frames": animation_max_frames,
            "fps": animation_fps,
            "dpi": animation_dpi,
        },
        "coordinate_convention": {
            "x": "conveyor belt direction",
            "y": "belt width / jet nozzle position direction",
            "z": "vertical direction",
        },
    }


def make_body_box_vertices_from_points(points_body):
    x_min, y_min, z_min = np.min(points_body, axis=0)
    x_max, y_max, z_max = np.max(points_body, axis=0)

    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=float,
    )

    return vertices


def make_box_faces(vertices_world):
    return [
        [vertices_world[0], vertices_world[1], vertices_world[2], vertices_world[3]],
        [vertices_world[4], vertices_world[5], vertices_world[6], vertices_world[7]],
        [vertices_world[0], vertices_world[1], vertices_world[5], vertices_world[4]],
        [vertices_world[1], vertices_world[2], vertices_world[6], vertices_world[5]],
        [vertices_world[2], vertices_world[3], vertices_world[7], vertices_world[6]],
        [vertices_world[3], vertices_world[0], vertices_world[4], vertices_world[7]],
    ]


def add_body_box_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.25,
    facecolor="tab:blue",
    edgecolor="black",
):
    vertices_body = make_body_box_vertices_from_points(obj.surface_points_body)

    vertices_world, _, _ = transform_surface_points(
        position=position,
        quaternion=quaternion,
        points_body=vertices_body,
    )

    faces = make_box_faces(vertices_world)

    body = Poly3DCollection(
        faces,
        alpha=alpha,
        linewidths=1.0,
        edgecolors=edgecolor,
        facecolors=facecolor,
    )

    ax.add_collection3d(body)
    return ax


def add_irregular_points_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.5,
    color="tab:purple",
):
    points_world, _, _ = transform_surface_points(
        position=np.asarray(position, dtype=float),
        quaternion=np.asarray(quaternion, dtype=float),
        points_body=obj.surface_points_body,
    )

    ax.scatter(
        points_world[:, 0],
        points_world[:, 1],
        points_world[:, 2],
        s=14,
        alpha=alpha,
        color=color,
    )

    return ax


def create_cylinder_mesh_body(length, radius, n_length=12, n_theta=24):
    xs = np.linspace(-length / 2.0, length / 2.0, n_length)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)

    side_points = []
    for i in range(n_length):
        row = []
        for theta in thetas:
            row.append([xs[i], radius * np.cos(theta), radius * np.sin(theta)])
        side_points.append(row)

    side_points = np.array(side_points, dtype=float)

    left_cap = []
    right_cap = []
    for theta in thetas:
        left_cap.append([-length / 2.0, radius * np.cos(theta), radius * np.sin(theta)])
        right_cap.append([length / 2.0, radius * np.cos(theta), radius * np.sin(theta)])

    left_cap = np.array(left_cap, dtype=float)
    right_cap = np.array(right_cap, dtype=float)

    return side_points, left_cap, right_cap


def transform_body_mesh_points(position, quaternion, points_body):
    original_shape = points_body.shape
    flat = points_body.reshape(-1, 3)

    points_world, _, _ = transform_surface_points(
        position=np.asarray(position, dtype=float),
        quaternion=np.asarray(quaternion, dtype=float),
        points_body=flat,
    )

    return points_world.reshape(original_shape)


def add_rod_cylinder_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.35,
    facecolor="tab:purple",
    edgecolor="black",
):
    length = obj.rod_length
    radius = obj.rod_radius

    if length is None or radius is None:
        return add_body_box_to_axis(
            ax=ax,
            obj=obj,
            position=position,
            quaternion=quaternion,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    side_body, left_cap_body, right_cap_body = create_cylinder_mesh_body(
        length=length,
        radius=radius,
        n_length=12,
        n_theta=24,
    )

    side_world = transform_body_mesh_points(position, quaternion, side_body)
    left_cap_world = transform_body_mesh_points(position, quaternion, left_cap_body)
    right_cap_world = transform_body_mesh_points(position, quaternion, right_cap_body)

    X = side_world[:, :, 0]
    Y = side_world[:, :, 1]
    Z = side_world[:, :, 2]

    ax.plot_surface(
        X,
        Y,
        Z,
        alpha=alpha,
        linewidth=0.4,
        edgecolor=edgecolor,
        color=facecolor,
        shade=True,
    )

    left_center = np.array([[-length / 2.0, 0.0, 0.0]], dtype=float)
    right_center = np.array([[length / 2.0, 0.0, 0.0]], dtype=float)
    left_center_world = transform_body_mesh_points(position, quaternion, left_center)[0]
    right_center_world = transform_body_mesh_points(position, quaternion, right_center)[0]

    left_faces = []
    right_faces = []
    for i in range(len(left_cap_world) - 1):
        left_faces.append([left_center_world, left_cap_world[i], left_cap_world[i + 1]])
        right_faces.append([right_center_world, right_cap_world[i], right_cap_world[i + 1]])

    cap_collection = Poly3DCollection(
        left_faces + right_faces,
        alpha=alpha,
        linewidths=0.4,
        edgecolors=edgecolor,
        facecolors=facecolor,
    )
    ax.add_collection3d(cap_collection)

    return ax


def add_body_geometry_to_axis(
    ax,
    obj,
    position,
    quaternion,
    alpha=0.35,
    facecolor="tab:purple",
    edgecolor="black",
):
    if obj.object_type == "rod":
        return add_rod_cylinder_to_axis(
            ax=ax,
            obj=obj,
            position=position,
            quaternion=quaternion,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    if obj.object_type == "irregular":
        return add_irregular_points_to_axis(
            ax=ax,
            obj=obj,
            position=position,
            quaternion=quaternion,
            alpha=max(alpha, 0.45),
            color=facecolor,
        )

    return add_body_box_to_axis(
        ax=ax,
        obj=obj,
        position=position,
        quaternion=quaternion,
        alpha=alpha,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )


def add_target_region_to_3d_axis(
    ax,
    target,
    landing_z,
    axis_limits=None,
    facecolor="lightskyblue",
    edgecolor="tab:blue",
    alpha=0.22,
):
    if axis_limits is not None:
        y_min, y_max = axis_limits["y"]
    else:
        y_min, y_max = -0.50, 0.50

    x_min = target.x_min
    x_max = target.x_max
    z = landing_z

    vertices = [
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z],
    ]

    target_face = Poly3DCollection(
        [vertices],
        alpha=alpha,
        linewidths=1.0,
        edgecolors=edgecolor,
        facecolors=facecolor,
    )

    ax.add_collection3d(target_face)
    return ax


def add_jet_x_zone_to_3d_axis(
    ax,
    jet,
    axis_limits=None,
    landing_z=0.0,
    facecolor="gold",
    edgecolor="darkorange",
    alpha=0.16,
):
    if axis_limits is not None:
        y_min, y_max = axis_limits["y"]
    else:
        y_min, y_max = -0.50, 0.50

    x_min = jet.x_start
    x_max = jet.x_start + jet.x_width
    z = landing_z + 0.002

    vertices = [
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z],
    ]

    jet_zone_face = Poly3DCollection(
        [vertices],
        alpha=alpha,
        linewidths=1.0,
        edgecolors=edgecolor,
        facecolors=facecolor,
    )

    ax.add_collection3d(jet_zone_face)
    return ax


def apply_3d_axis_limits(ax, axis_limits):
    if axis_limits is not None:
        x_min, x_max = axis_limits["x"]
        y_min, y_max = axis_limits["y"]
        z_min, z_max = axis_limits["z"]
    else:
        x_min, x_max = -0.10, 1.50
        y_min, y_max = -0.50, 0.50
        z_min, z_max = 0.00, 0.60

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    x_range = max(x_max - x_min, 1.0e-6)
    y_range = max(y_max - y_min, 1.0e-6)
    z_range = max(z_max - z_min, 1.0e-6)

    ax.set_box_aspect((x_range, y_range, z_range))


def plot_3d_trajectory(result, show_points=True, axis_limits=None):
    position = result["position"]
    quaternion = result["quaternion"]
    obj = result["object"]
    jet = result["jet"]

    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    trajectory_color = "tab:blue"
    start_color = "tab:blue"
    landing_color = "tab:orange"
    target_facecolor = "lightskyblue"
    target_edgecolor = "tab:blue"
    jet_zone_facecolor = "gold"
    jet_zone_edgecolor = "darkorange"
    initial_body_color = "tab:red"
    final_body_color = "tab:purple"
    surface_point_color = "tab:green"

    ax.plot(x, y, z, linewidth=2, color=trajectory_color, label="COM trajectory")

    ax.scatter(
        x[0],
        y[0],
        z[0],
        s=60,
        marker="o",
        color=start_color,
        label="start",
    )

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(
            landing_position[0],
            landing_position[1],
            landing_position[2],
            s=80,
            marker="x",
            color=landing_color,
            linewidths=2.5,
            label="landing",
        )

    target = result.get("target")
    sim = result.get("simulation")

    if target is not None and sim is not None:
        add_target_region_to_3d_axis(
            ax=ax,
            target=target,
            landing_z=sim.landing_z,
            axis_limits=axis_limits,
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
        )

        add_jet_x_zone_to_3d_axis(
            ax=ax,
            jet=jet,
            axis_limits=axis_limits,
            landing_z=sim.landing_z,
            facecolor=jet_zone_facecolor,
            edgecolor=jet_zone_edgecolor,
            alpha=0.16,
        )

    add_body_geometry_to_axis(
        ax=ax,
        obj=obj,
        position=position[0],
        quaternion=quaternion[0],
        alpha=0.18,
        facecolor=initial_body_color,
        edgecolor="black",
    )

    add_body_geometry_to_axis(
        ax=ax,
        obj=obj,
        position=position[-1],
        quaternion=quaternion[-1],
        alpha=0.35,
        facecolor=final_body_color,
        edgecolor="black",
    )

    if show_points:
        points = result["final_points_world"]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=8,
            alpha=0.6,
            color=surface_point_color,
            label="final surface points",
        )

    ax.set_xlabel("x: conveyor direction [m]")
    ax.set_ylabel("y: belt width / jet position [m]")
    ax.set_zlabel("z: vertical direction [m]")

    if result.get("has_landed", False):
        ax.set_title("3D COM Trajectory, Object Orientation, Target, and Jet Zone")
    else:
        ax.set_title("3D COM Trajectory and Final Simulated State, Not Landed")

    apply_3d_axis_limits(ax, axis_limits)

    legend_handles = [
        Line2D([0], [0], color=trajectory_color, linewidth=2, label="COM trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=start_color,
            markeredgecolor=start_color,
            markersize=8,
            label="start",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=landing_color,
            markersize=9,
            markeredgewidth=2.5,
            linestyle="None",
            label="landing",
        ),
        Patch(
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
            label="target landing region",
        ),
        Patch(
            facecolor=jet_zone_facecolor,
            edgecolor=jet_zone_edgecolor,
            alpha=0.16,
            label="finite air-jet x-zone",
        ),
        Patch(
            facecolor=initial_body_color,
            edgecolor="black",
            alpha=0.18,
            label="initial body",
        ),
        Patch(
            facecolor=final_body_color,
            edgecolor="black",
            alpha=0.35,
            label="final / last simulated body",
        ),
    ]

    if show_points:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=surface_point_color,
                markeredgecolor=surface_point_color,
                markersize=5,
                linestyle="None",
                label="final surface points",
            )
        )

    ax.legend(handles=legend_handles)

    return fig


def draw_animation_frame(ax, result, frame_index, axis_limits=None, show_points=False):
    """Draw one animation frame on a 3D axis."""
    position = result["position"]
    quaternion = result["quaternion"]
    obj = result["object"]
    jet = result["jet"]
    target = result.get("target")
    sim = result.get("simulation")
    time = result["time"]

    current_time = float(time[frame_index])

    # Jet valve ON/OFF status based on time condition
    jet_is_on = jet.t_on <= current_time <= jet.t_on + jet.duration

    trajectory_color = "tab:blue"
    start_color = "tab:blue"
    current_color = "tab:orange"
    target_facecolor = "lightskyblue"
    target_edgecolor = "tab:blue"

    if jet_is_on:
        jet_zone_facecolor = "limegreen"
        jet_zone_edgecolor = "green"
        jet_zone_alpha = 0.28
        jet_status_text = "● JET ON"
        jet_status_color = "green"
    else:
        jet_zone_facecolor = "lightcoral"
        jet_zone_edgecolor = "red"
        jet_zone_alpha = 0.18
        jet_status_text = "● JET OFF"
        jet_status_color = "red"

    current_body_color = "tab:purple"
    surface_point_color = "tab:green"

    ax.clear()

    # Give extra space for the legend outside the plot.
    fig = ax.figure
    fig.subplots_adjust(
        left=0.03,
        right=0.72,
        bottom=0.08,
        top=0.90,
    )

    x = position[: frame_index + 1, 0]
    y = position[: frame_index + 1, 1]
    z = position[: frame_index + 1, 2]

    ax.plot(
        x,
        y,
        z,
        linewidth=2,
        color=trajectory_color,
        label="COM trajectory",
    )

    ax.scatter(
        position[0, 0],
        position[0, 1],
        position[0, 2],
        s=45,
        marker="o",
        color=start_color,
        label="start",
    )

    ax.scatter(
        position[frame_index, 0],
        position[frame_index, 1],
        position[frame_index, 2],
        s=60,
        marker="o",
        color=current_color,
        label="current COM",
    )

    if target is not None and sim is not None:
        add_target_region_to_3d_axis(
            ax=ax,
            target=target,
            landing_z=sim.landing_z,
            axis_limits=axis_limits,
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
        )

        add_jet_x_zone_to_3d_axis(
            ax=ax,
            jet=jet,
            axis_limits=axis_limits,
            landing_z=sim.landing_z,
            facecolor=jet_zone_facecolor,
            edgecolor=jet_zone_edgecolor,
            alpha=jet_zone_alpha,
        )

    add_body_geometry_to_axis(
        ax=ax,
        obj=obj,
        position=position[frame_index],
        quaternion=quaternion[frame_index],
        alpha=0.40,
        facecolor=current_body_color,
        edgecolor="black",
    )

    if show_points:
        points_world, _, _ = transform_surface_points(
            position=position[frame_index],
            quaternion=quaternion[frame_index],
            points_body=obj.surface_points_body,
        )

        ax.scatter(
            points_world[:, 0],
            points_world[:, 1],
            points_world[:, 2],
            s=7,
            alpha=0.55,
            color=surface_point_color,
            label="surface points",
        )

    # Title and axis labels
    ax.set_title(
        f"3D Rigid-Body Motion, t = {current_time:.3f} s",
        fontsize=12,
        pad=14,
    )

    ax.set_xlabel(
        "x: conveyor direction [m]",
        fontsize=9,
        labelpad=8,
    )

    ax.set_ylabel(
        "y: belt width / jet position [m]",
        fontsize=9,
        labelpad=12,
    )

    ax.set_zlabel(
        "z: vertical direction [m]",
        fontsize=9,
        labelpad=10,
    )

    ax.tick_params(axis="both", which="major", labelsize=8, pad=2)

    # A fixed view angle helps reduce y/z label overlap.
    ax.view_init(elev=24, azim=-58)

    apply_3d_axis_limits(ax, axis_limits)

    # -----------------------------------------------------------------
    # Compact jet ON/OFF status overlay
    # ON  = green
    # OFF = red
    # -----------------------------------------------------------------

    ax.text2D(
        0.015,
        0.965,
        jet_status_text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=jet_status_color,
        bbox={
            "facecolor": "white",
            "edgecolor": jet_status_color,
            "boxstyle": "round,pad=0.18",
            "alpha": 0.86,
        },
    )

    ax.text2D(
        0.015,
        0.920,
        f"t_on = {jet.t_on:.3f} s | duration = {jet.duration:.3f} s",
        transform=ax.transAxes,
        fontsize=7,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "lightgray",
            "boxstyle": "round,pad=0.16",
            "alpha": 0.72,
        },
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=trajectory_color,
            linewidth=2,
            label="COM trajectory",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=start_color,
            markeredgecolor=start_color,
            markersize=6,
            label="start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=current_color,
            markeredgecolor=current_color,
            markersize=7,
            label="current COM",
        ),
        Patch(
            facecolor=target_facecolor,
            edgecolor=target_edgecolor,
            alpha=0.22,
            label="target landing region",
        ),
        Patch(
            facecolor=jet_zone_facecolor,
            edgecolor=jet_zone_edgecolor,
            alpha=jet_zone_alpha,
            label=f"air-jet x-zone ({'ON' if jet_is_on else 'OFF'})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=jet_status_color,
            markeredgecolor=jet_status_color,
            markersize=7,
            linestyle="None",
            label=f"jet status: {'ON' if jet_is_on else 'OFF'}",
        ),
        Patch(
            facecolor=current_body_color,
            edgecolor="black",
            alpha=0.40,
            label="current body",
        ),
    ]

    if show_points:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=surface_point_color,
                markeredgecolor=surface_point_color,
                markersize=4,
                linestyle="None",
                label="surface points",
            )
        )

    # Put legend outside the 3D axes to avoid covering the trajectory.
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.88,
    )


def create_3d_animation_gif(
    result,
    axis_limits,
    max_frames,
    fps,
    dpi,
    show_points=False,
):
    """
    Create GIF animation bytes from a simulation result.

    This function only creates the GIF in memory for preview/download.
    It does NOT save the GIF permanently to the project folder.
    """
    position = result["position"]
    n_steps = len(position)

    if n_steps <= 1:
        raise ValueError("Not enough trajectory points to create an animation.")

    n_frames = int(min(max_frames, n_steps))
    frame_indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)
    frame_indices = np.unique(frame_indices)

    fig = plt.figure(figsize=(9.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index):
        draw_animation_frame(
            ax=ax,
            result=result,
            frame_index=int(frame_index),
            axis_limits=axis_limits,
            show_points=show_points,
        )
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / max(fps, 1),
        blit=False,
    )

    # Use a temporary file only to generate GIF bytes.
    # The final GIF is not saved to the project folder here.
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    writer = PillowWriter(fps=fps)
    animation.save(temp_path, writer=writer, dpi=dpi)

    plt.close(fig)

    gif_bytes = temp_path.read_bytes()

    try:
        temp_path.unlink()
    except OSError:
        pass

    return gif_bytes


def save_gif_to_project_folder(gif_bytes, filename="week2_3d_animation.gif"):
    """
    Save GIF bytes to the project results folder only when the user clicks Save.
    """
    output_dir = PROJECT_ROOT / "results" / "week2" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    output_path.write_bytes(gif_bytes)

    return output_path


def plot_xy_landing_map(result, target, axis_limits=None):
    position = result["position"]
    jet = result["jet"]

    x = position[:, 0]
    y = position[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(x, y, linewidth=2, label="COM path in x-y")
    ax.scatter(x[0], y[0], s=60, marker="o", label="start")

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(
            landing_position[0],
            landing_position[1],
            s=80,
            marker="x",
            label="landing",
        )

    ax.axvspan(
        target.x_min,
        target.x_max,
        alpha=0.22,
        color="lightskyblue",
        label="target x region",
    )
    ax.axvline(target.x_min, linestyle="--", linewidth=1.0)
    ax.axvline(target.x_max, linestyle="--", linewidth=1.0)

    ax.axvspan(
        jet.x_start,
        jet.x_start + jet.x_width,
        alpha=0.16,
        color="gold",
        label="finite jet x-zone",
    )

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x"])
        ax.set_ylim(axis_limits["y"])

    ax.set_xlabel("x conveyor position [m]")
    ax.set_ylabel("y belt width position [m]")
    ax.set_title("x-y Projection: Target Region and Finite Jet Zone")
    ax.grid(True)
    ax.legend()

    return fig


def plot_yz_projection_with_jet(result, jet, axis_limits=None):
    position = result["position"]
    y = position[:, 1]
    z = position[:, 2]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(y, z, linewidth=2, label="COM path in y-z")
    ax.scatter(y[0], z[0], s=60, marker="o", label="start")

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(
            landing_position[1],
            landing_position[2],
            s=80,
            marker="x",
            label="landing",
        )

    circle = plt.Circle(
        (jet.y_center, jet.z_center),
        jet.sigma,
        fill=False,
        linestyle="--",
        label="jet y-z width sigma",
    )
    ax.add_patch(circle)

    ax.scatter(
        jet.y_center,
        jet.z_center,
        s=80,
        marker="+",
        label="jet center in y-z",
    )

    if axis_limits is not None:
        ax.set_xlim(axis_limits["y"])
        ax.set_ylim(axis_limits["z"])

    ax.set_xlabel("y belt width / jet position [m]")
    ax.set_ylabel("z vertical position [m]")
    ax.set_title("y-z Projection and Jet Cross-Section")
    ax.grid(True)
    ax.legend()

    return fig


def plot_xz_projection(result, target, axis_limits=None):
    position = result["position"]
    jet = result["jet"]

    x = position[:, 0]
    z = position[:, 2]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(x, z, linewidth=2, label="COM path in x-z")
    ax.scatter(x[0], z[0], s=60, marker="o", label="start")

    landing_position = result["landing_position"]
    if landing_position is not None:
        ax.scatter(
            landing_position[0],
            landing_position[2],
            s=80,
            marker="x",
            label="landing",
        )

    ax.axvspan(
        target.x_min,
        target.x_max,
        alpha=0.22,
        color="lightskyblue",
        label="target x region",
    )
    ax.axvline(target.x_min, linestyle="--", linewidth=1.0)
    ax.axvline(target.x_max, linestyle="--", linewidth=1.0)

    ax.axvspan(
        jet.x_start,
        jet.x_start + jet.x_width,
        alpha=0.16,
        color="gold",
        label="finite jet x-zone",
    )
    ax.axhline(
        jet.z_center,
        linestyle=":",
        linewidth=1.0,
        label="jet z center",
    )

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x"])
        ax.set_ylim(axis_limits["z"])

    ax.set_xlabel("x conveyor position [m]")
    ax.set_ylabel("z vertical position [m]")
    ax.set_title("x-z Projection: Target Region and Finite Jet Zone")
    ax.grid(True)
    ax.legend()

    return fig


def plot_time_history(result, key, ylabel, title, labels):
    time = result["time"]
    values = result[key]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, label in enumerate(labels):
        ax.plot(time, values[:, i], label=label)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    return fig


def plot_magnitude_history(result, key, ylabel, title):
    time = result["time"]
    values = result[key]
    magnitude = np.linalg.norm(values, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, magnitude, linewidth=2)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    return fig


def plot_total_force_history(result):
    time = result["time"]
    force_total = result["force_total"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, force_total[:, 0], label="Ftotal,x")
    ax.plot(time, force_total[:, 1], label="Ftotal,y")
    ax.plot(time, force_total[:, 2], label="Ftotal,z")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("total force [N]")
    ax.set_title("Total Force = Jet + Drag + Gravity")
    ax.grid(True)
    ax.legend()

    return fig


def plot_force_breakdown_z(result):
    time = result["time"]
    force_jet = result["force_jet"]
    force_drag = result["force_drag"]
    force_gravity = result["force_gravity"]
    force_total = result["force_total"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, force_jet[:, 2], label="Fjet,z")
    ax.plot(time, force_drag[:, 2], label="Fdrag,z")
    ax.plot(time, force_gravity[:, 2], label="Fgravity,z")
    ax.plot(time, force_total[:, 2], label="Ftotal,z", linewidth=2)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("z-force [N]")
    ax.set_title("Z-Force Breakdown")
    ax.grid(True)
    ax.legend()

    return fig


# ---------------------------------------------------------------------
# Run simulation and store result
# ---------------------------------------------------------------------

if run_button:
    obj = create_object_3d(
        object_type=object_type,
        mass=mass,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        drag_coefficient=drag_coefficient,
        rod_length=rod_length,
        rod_radius=rod_radius,
        seed=seed,
    )

    jet = Jet3D(
        umax=umax,
        x_start=jet_x_start,
        x_width=jet_x_width,
        y_center=jet_y_center,
        z_center=jet_z_center,
        sigma=sigma,
        angle_deg=jet_angle_deg,
        t_on=jet_t_on,
        duration=jet_duration,
        noise_std=noise_std,
    )

    sim = Simulation3D(
        dt=dt,
        t_max=t_max,
        gravity=gravity,
        air_density=air_density,
        landing_z=landing_z,
    )

    initial_quaternion = euler_degrees_to_quaternion(
        roll_deg=roll0,
        pitch_deg=pitch0,
        yaw_deg=yaw0,
    )

    initial = InitialCondition3D(
        position=(x0, y0, z0),
        velocity=(vc, vy_initial, vz_initial),
        quaternion=initial_quaternion,
        angular_velocity=(omega_x0, omega_y0, omega_z0),
    )

    target = TargetRegion3D(
        x_min=target_x_min,
        x_max=target_x_max,
    )

    hit_offset = compute_hit_offset(initial.position, jet)

    result = simulate_rigid_body_3d(
        obj=obj,
        jet=jet,
        sim=sim,
        initial=initial,
        target=target,
        seed=seed,
    )

    df = result_to_dataframe(result)
    csv_data = df.to_csv(index=False).encode("utf-8")
    parameter_dict = make_parameter_dict(hit_offset)
    parameter_json = json.dumps(parameter_dict, indent=2)

    st.session_state["last_result"] = result
    st.session_state["last_hit_offset"] = hit_offset
    st.session_state["last_dataframe"] = df
    st.session_state["last_csv_data"] = csv_data
    st.session_state["last_parameter_json"] = parameter_json

    if "last_gif_bytes" in st.session_state:
        del st.session_state["last_gif_bytes"]


# ---------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    hit_offset = st.session_state["last_hit_offset"]
    df = st.session_state["last_dataframe"]
    csv_data = st.session_state["last_csv_data"]
    parameter_json = st.session_state["last_parameter_json"]

    landing_position = result["landing_position"]
    landing_time = result["landing_time"]
    has_landed = result["has_landed"]
    final_position = result["final_position"]
    final_time = result["final_time"]
    success = result["success"]

    obj = result["object"]
    jet = result["jet"]
    target = result["target"]

    col1, col2 = st.columns([2, 1])

    with col1:
        fig3d = plot_3d_trajectory(
            result,
            show_points=show_surface_points,
            axis_limits=axis_limits,
        )
        st.pyplot(fig3d)

    with col2:
        st.subheader("Simulation Result")

        if not has_landed:
            st.warning("NOT LANDED: The object did not reach the landing plane within t_max.")

            st.metric("Final simulated time [s]", f"{final_time:.4f}")
            st.metric("Final x [m]", f"{final_position[0]:.4f}")
            st.metric("Final y [m]", f"{final_position[1]:.4f}")
            st.metric("Final z [m]", f"{final_position[2]:.4f}")

            st.info(
                "Landing time and landing position are not defined for this run. "
                "Increase t_max, reduce upward jet force, reduce jet duration, "
                "or adjust the jet angle if you want the object to land."
            )

        else:
            st.metric("Landing x [m]", f"{landing_position[0]:.4f}")
            st.metric("Landing y [m]", f"{landing_position[1]:.4f}")
            st.metric("Landing z [m]", f"{landing_position[2]:.4f}")
            st.metric("Landing time [s]", f"{landing_time:.4f}")

            if success:
                st.success("SUCCESS: landing x is inside target region.")
            else:
                st.error("FAIL: landing x missed the target region.")

        st.metric("Initial offset from jet region [m]", f"{hit_offset:.4f}")
        st.metric("Max angular speed [rad/s]", f"{result['max_angular_speed']:.4f}")
        st.metric("Linear impulse |J| [N s]", f"{np.linalg.norm(result['jet_impulse']):.4e}")
        st.metric("Angular impulse |L| [N m s]", f"{np.linalg.norm(result['angular_impulse']):.4e}")

        if has_landed and landing_time is not None and jet.t_on > landing_time:
            st.warning(
                "Jet activates after landing. Decrease t_on, increase initial height, "
                "or increase conveyor speed."
            )

        if hit_offset > 3.0 * jet.sigma:
            st.warning(
                "The initial COM is far from the jet region relative to the y-z jet width. "
                "The jet may barely interact with the object."
            )

        st.subheader("Object Summary")

        object_summary = {
            "object name": obj.name,
            "object type": obj.object_type,
            "number of surface points": int(obj.surface_points_body.shape[0]),
            "total area weight [m2]": float(np.sum(obj.area_weights)),
            "mass [kg]": obj.mass,
            "Cd [-]": obj.drag_coefficient,
            "size_x [m]": obj.size_x,
            "size_y [m]": obj.size_y,
            "size_z [m]": obj.size_z,
        }

        if obj.object_type == "rod":
            object_summary["rod length [m]"] = obj.rod_length
            object_summary["rod radius [m]"] = obj.rod_radius

        st.write(object_summary)

        st.download_button(
            label="Download parameters as JSON",
            data=parameter_json,
            file_name="week2_parameters.json",
            mime="application/json",
        )

    st.subheader("Animation Export")

    st.write(
        """
        Generate a GIF preview of the 3D rigid-body motion first.  
        The GIF will be shown on this page before saving.  
        Click **Save GIF to Project Folder** only if you want to save it locally.
        """
    )

    col_anim_1, col_anim_2 = st.columns([1, 2])

    with col_anim_1:
        generate_gif_button = st.button(
            "Generate GIF Preview",
            type="primary",
            key="generate_3d_gif_preview_button",
        )

    with col_anim_2:
        st.caption(
            f"Current settings: max frames = {animation_max_frames}, "
            f"FPS = {animation_fps}, DPI = {animation_dpi}"
        )

    if generate_gif_button:
        try:
            with st.spinner("Generating GIF preview. This may take a moment..."):
                gif_bytes = create_3d_animation_gif(
                    result=result,
                    axis_limits=axis_limits,
                    max_frames=animation_max_frames,
                    fps=animation_fps,
                    dpi=animation_dpi,
                    show_points=show_surface_points,
                )

            st.session_state["last_gif_bytes"] = gif_bytes

            if "last_gif_path" in st.session_state:
                del st.session_state["last_gif_path"]

            st.success("GIF preview generated. Review it below before saving.")

        except Exception as exc:
            st.error(f"Failed to generate GIF preview: {exc}")

    if "last_gif_bytes" in st.session_state:
        st.subheader("GIF Preview")

        st.image(
            st.session_state["last_gif_bytes"],
            caption="3D rigid-body motion preview",
        )

        col_save_1, col_save_2 = st.columns(2)

        with col_save_1:
            st.download_button(
                label="Download GIF",
                data=st.session_state["last_gif_bytes"],
                file_name="week2_3d_animation.gif",
                mime="image/gif",
            )

        with col_save_2:
            save_gif_button = st.button(
                "Save GIF to Project Folder",
                key="save_3d_gif_to_project_folder_button",
            )

        if save_gif_button:
            try:
                gif_path = save_gif_to_project_folder(
                    st.session_state["last_gif_bytes"],
                    filename="week2_3d_animation.gif",
                )

                st.session_state["last_gif_path"] = str(gif_path)
                st.success(f"GIF saved to: {gif_path}")

            except Exception as exc:
                st.error(f"Failed to save GIF: {exc}")

    if "last_gif_path" in st.session_state:
        st.info(f"Last saved GIF path: {st.session_state['last_gif_path']}")

    st.subheader("2D Projections")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_xy = plot_xy_landing_map(
            result,
            target,
            axis_limits=axis_limits,
        )
        st.pyplot(fig_xy)

    with col_b:
        fig_xz = plot_xz_projection(
            result,
            target,
            axis_limits=axis_limits,
        )
        st.pyplot(fig_xz)

    st.subheader("Jet Cross-Section View")

    fig_yz = plot_yz_projection_with_jet(
        result,
        jet,
        axis_limits=axis_limits,
    )
    st.pyplot(fig_yz)

    st.subheader("Force, Torque, and Angular Velocity")

    col_c, col_d = st.columns(2)

    with col_c:
        force_fig = plot_time_history(
            result=result,
            key="force_jet",
            ylabel="jet force only [N]",
            title="Jet Force Only",
            labels=["Fjet,x", "Fjet,y", "Fjet,z"],
        )
        st.pyplot(force_fig)

    with col_d:
        total_force_fig = plot_total_force_history(result)
        st.pyplot(total_force_fig)

    col_e, col_f = st.columns(2)

    with col_e:
        torque_fig = plot_time_history(
            result=result,
            key="torque_jet",
            ylabel="jet torque [N m]",
            title="Jet Torque",
            labels=["tau_x", "tau_y", "tau_z"],
        )
        st.pyplot(torque_fig)

    with col_f:
        omega_fig = plot_time_history(
            result=result,
            key="angular_velocity",
            ylabel="angular velocity [rad/s]",
            title="Angular Velocity",
            labels=["omega_x", "omega_y", "omega_z"],
        )
        st.pyplot(omega_fig)

    col_g, col_h = st.columns(2)

    with col_g:
        force_breakdown_fig = plot_force_breakdown_z(result)
        st.pyplot(force_breakdown_fig)

    with col_h:
        omega_mag_fig = plot_magnitude_history(
            result=result,
            key="angular_velocity",
            ylabel="|omega| [rad/s]",
            title="Angular Speed Magnitude",
        )
        st.pyplot(omega_mag_fig)

    st.subheader("Trajectory Data")

    st.dataframe(df.head(30), use_container_width=True)

    st.download_button(
        label="Download trajectory data as CSV",
        data=csv_data,
        file_name="week2_trajectory.csv",
        mime="text/csv",
    )


else:
    st.info("Adjust the sliders in the sidebar and click **Run 3D Simulation**.")

    st.subheader("Recommended Starting Values")

    st.write(
        {
            "object type": "plate",
            "mass [kg]": 0.050,
            "plate/irregular size [m]": [0.10, 0.10, 0.01],
            "rod length/radius [m]": [0.15, 0.015],
            "initial COM position [m]": [0.0, 0.0, 0.20],
            "initial COM velocity [m/s]": [1.0, 0.0, 0.0],
            "initial orientation [roll, pitch, yaw] [deg]": [0.0, 0.0, 0.0],
            "jet Umax [m/s]": 25.0,
            "jet x-zone [m]": [0.00, 0.05],
            "jet x-profile": "raised cosine, strongest at x-zone center",
            "jet center [y, z] [m]": [0.0, 0.20],
            "jet y-z sigma [m]": 0.08,
            "jet angle [deg]": 45.0,
            "jet t_on [s]": 0.15,
            "jet duration [s]": 0.15,
            "target x region [m]": [0.30, 0.80],
            "fixed plot axes": {
                "x": [-0.10, 1.50],
                "y": [-0.50, 0.50],
                "z": [0.00, 0.60],
            },
            "animation": {
                "max frames": 80,
                "fps": 12,
                "dpi": 100,
            },
        }
    )
