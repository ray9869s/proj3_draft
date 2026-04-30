"""
Interactive Week 1 2D Air-Jet Simulator using Streamlit.

Run from the project root:

    streamlit run scripts/app_week1_streamlit.py
"""

from pathlib import Path
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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


st.set_page_config(
    page_title="Week 1 Air-Jet 2D Simulator",
    layout="wide",
)

st.title("Week 1: Interactive 2D Air-Jet Sorting Simulator")

st.write(
    """
    This app simulates the 2D motion of a plastic object under gravity,
    aerodynamic drag, and a finite-duration air-jet force.
    Use the sliders to change the physical parameters, then run the simulation.
    """
)

st.info(
    "Coordinate convention: x is the horizontal travel direction and y is the vertical direction. "
    "Therefore, Fx directly changes the horizontal motion, while Fy acts upward/downward and "
    "mainly changes the flight time."
)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("Object Parameters")

mass = st.sidebar.slider(
    "Mass m [kg]",
    min_value=0.005,
    max_value=0.300,
    value=0.050,
    step=0.005,
    format="%.3f",
)

area = st.sidebar.slider(
    "Projected area A [m²]",
    min_value=1.0e-4,
    max_value=3.0e-2,
    value=1.0e-2,
    step=1.0e-4,
    format="%.4f",
)

area_cm2 = area * 10000.0
st.sidebar.caption(f"Projected area = {area_cm2:.1f} cm²")

drag_coefficient = st.sidebar.slider(
    "Drag coefficient Cd [-]",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.05,
)

st.sidebar.header("Initial Condition")

x0 = st.sidebar.slider(
    "Initial x-position [m]",
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.05,
)

y0 = st.sidebar.slider(
    "Initial y-position [m]",
    min_value=0.05,
    max_value=3.0,
    value=1.0,
    step=0.05,
)

vx0 = st.sidebar.slider(
    "Initial x-velocity [m/s]",
    min_value=0.0,
    max_value=5.0,
    value=1.5,
    step=0.05,
)

vy0 = st.sidebar.slider(
    "Initial y-velocity [m/s]",
    min_value=-2.0,
    max_value=5.0,
    value=0.0,
    step=0.05,
)

st.sidebar.header("Air-Jet Parameters")

jet_fx = st.sidebar.slider(
    "Jet force Fx [N]",
    min_value=-0.5,
    max_value=0.5,
    value=0.00,
    step=0.01,
    format="%.2f",
)

jet_fy = st.sidebar.slider(
    "Jet force Fy [N]",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
    format="%.2f",
)

jet_t_on = st.sidebar.slider(
    "Jet activation time t_on [s]",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.01,
)

jet_duration = st.sidebar.slider(
    "Jet duration Δt [s]",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01,
)

st.sidebar.header("Simulation Parameters")

dt = st.sidebar.slider(
    "Time step dt [s]",
    min_value=0.0005,
    max_value=0.02,
    value=0.001,
    step=0.0005,
    format="%.4f",
)

t_max = st.sidebar.slider(
    "Maximum simulation time [s]",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.1,
)

gravity = st.sidebar.slider(
    "Gravity g [m/s²]",
    min_value=0.0,
    max_value=20.0,
    value=9.81,
    step=0.01,
)

air_density = st.sidebar.slider(
    "Air density ρ [kg/m³]",
    min_value=0.0,
    max_value=2.0,
    value=1.225,
    step=0.005,
)

st.sidebar.header("Target Region")

target_x_min = st.sidebar.slider(
    "Target x_min [m]",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.05,
)

target_x_max = st.sidebar.slider(
    "Target x_max [m]",
    min_value=0.0,
    max_value=6.0,
    value=2.0,
    step=0.05,
)

if target_x_max < target_x_min:
    st.sidebar.warning("Target x_max should be larger than x_min.")

st.sidebar.header("Plot Options")

show_force_component = st.sidebar.radio(
    "Force history component",
    options=["y-component", "x-component", "both"],
    index=0,
)

run_button = st.sidebar.button("Run Simulation", type="primary")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def result_to_dataframe(result):
    """Convert simulation result dictionary to a pandas DataFrame."""
    time = result["time"]
    position = result["position"]
    velocity = result["velocity"]
    force_gravity = result["force_gravity"]
    force_drag = result["force_drag"]
    force_jet = result["force_jet"]

    data = {
        "time_s": time,
        "x_m": position[:, 0],
        "y_m": position[:, 1],
        "vx_m_per_s": velocity[:, 0],
        "vy_m_per_s": velocity[:, 1],
        "F_gravity_x_N": force_gravity[:, 0],
        "F_gravity_y_N": force_gravity[:, 1],
        "F_drag_x_N": force_drag[:, 0],
        "F_drag_y_N": force_drag[:, 1],
        "F_jet_x_N": force_jet[:, 0],
        "F_jet_y_N": force_jet[:, 1],
    }

    return pd.DataFrame(data)


def make_parameter_dict():
    """Create a dictionary containing the current parameter values."""
    return {
        "object": {
            "mass_kg": mass,
            "projected_area_m2": area,
            "projected_area_cm2": area_cm2,
            "drag_coefficient": drag_coefficient,
        },
        "initial_condition": {
            "x0_m": x0,
            "y0_m": y0,
            "vx0_m_per_s": vx0,
            "vy0_m_per_s": vy0,
        },
        "air_jet": {
            "Fx_N": jet_fx,
            "Fy_N": jet_fy,
            "t_on_s": jet_t_on,
            "duration_s": jet_duration,
        },
        "simulation": {
            "dt_s": dt,
            "t_max_s": t_max,
            "gravity_m_per_s2": gravity,
            "air_density_kg_per_m3": air_density,
        },
        "target_region": {
            "x_min_m": target_x_min,
            "x_max_m": target_x_max,
        },
        "coordinate_convention": {
            "x": "horizontal travel direction",
            "y": "vertical direction",
            "Fx": "horizontal jet force",
            "Fy": "vertical jet force",
        },
    }


def plot_trajectory(result, target_region):
    """Plot 2D trajectory, target region, landing point, and jet-active points."""
    position = result["position"]
    force_jet = result["force_jet"]

    x = position[:, 0]
    y = position[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, y, linewidth=2, label="trajectory")

    # Initial point
    ax.scatter(x[0], y[0], s=60, marker="o", label="start")

    # Landing point
    landing_x = result["landing_position"]
    if landing_x is not None:
        ax.scatter(landing_x, 0.0, s=80, marker="x", label="landing")

    # Target region
    ax.axvspan(
        target_region.x_min,
        target_region.x_max,
        alpha=0.2,
        label="target region",
    )

    # Ground
    ax.axhline(0.0, linewidth=1.0)

    # Jet active region marker along trajectory
    jet_magnitude = np.linalg.norm(force_jet, axis=1)
    active_indices = np.where(jet_magnitude > 0.0)[0]

    if len(active_indices) > 0:
        ax.scatter(
            x[active_indices],
            y[active_indices],
            s=8,
            label="jet active",
        )

    # Reasonable view limits
    x_max_plot = max(np.max(x), target_region.x_max, 0.5)
    y_max_plot = max(np.max(y), 0.5)

    ax.set_xlim(left=min(-0.05, np.min(x) - 0.1), right=x_max_plot + 0.3)
    ax.set_ylim(bottom=min(-0.05, np.min(y) - 0.05), top=y_max_plot + 0.3)

    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.set_title("Object Trajectory")
    ax.grid(True)
    ax.legend()

    return fig


def plot_force_history(result, component_choice):
    """Plot force history."""
    time = result["time"]
    f_gravity = result["force_gravity"]
    f_drag = result["force_drag"]
    f_jet = result["force_jet"]

    fig, ax = plt.subplots(figsize=(8, 4))

    if component_choice == "y-component":
        ax.plot(time, f_gravity[:, 1], label="gravity Fy")
        ax.plot(time, f_drag[:, 1], label="drag Fy")
        ax.plot(time, f_jet[:, 1], label="jet Fy")
        ax.set_ylabel("force y-component [N]")

    elif component_choice == "x-component":
        ax.plot(time, f_gravity[:, 0], label="gravity Fx")
        ax.plot(time, f_drag[:, 0], label="drag Fx")
        ax.plot(time, f_jet[:, 0], label="jet Fx")
        ax.set_ylabel("force x-component [N]")

    else:
        ax.plot(time, f_gravity[:, 0], label="gravity Fx")
        ax.plot(time, f_gravity[:, 1], label="gravity Fy")
        ax.plot(time, f_drag[:, 0], label="drag Fx")
        ax.plot(time, f_drag[:, 1], label="drag Fy")
        ax.plot(time, f_jet[:, 0], label="jet Fx")
        ax.plot(time, f_jet[:, 1], label="jet Fy")
        ax.set_ylabel("force [N]")

    ax.set_xlabel("time [s]")
    ax.set_title("Force History")
    ax.grid(True)
    ax.legend()

    return fig


def plot_velocity_history(result):
    """Plot velocity history."""
    time = result["time"]
    velocity = result["velocity"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, velocity[:, 0], label="vx")
    ax.plot(time, velocity[:, 1], label="vy")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.set_title("Velocity History")
    ax.grid(True)
    ax.legend()

    return fig


# ---------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------

if run_button:
    obj = Object2D(
        mass=mass,
        area=area,
        drag_coefficient=drag_coefficient,
    )

    jet = Jet2D(
        force=(jet_fx, jet_fy),
        t_on=jet_t_on,
        duration=jet_duration,
    )

    sim = Simulation2D(
        dt=dt,
        t_max=t_max,
        gravity=gravity,
        air_density=air_density,
    )

    initial = InitialCondition2D(
        position=(x0, y0),
        velocity=(vx0, vy0),
    )

    target = TargetRegion2D(
        x_min=target_x_min,
        x_max=target_x_max,
    )

    result = simulate_trajectory_2d(
        obj=obj,
        jet=jet,
        sim=sim,
        initial=initial,
        target=target,
    )

    landing_x = result["landing_position"]
    landing_time = result["landing_time"]
    success = result["success"]

    force_jet = result["force_jet"]
    jet_active_steps = int(np.sum(np.linalg.norm(force_jet, axis=1) > 0.0))
    actual_jet_active_time = jet_active_steps * dt

    weight = mass * gravity
    jet_force_magnitude = float(np.linalg.norm(np.array([jet_fx, jet_fy])))
    jet_to_weight_ratio = jet_force_magnitude / weight if weight > 0.0 else np.nan

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plot_trajectory(result, target)
        st.pyplot(fig)

    with col2:
        st.subheader("Simulation Result")

        if landing_x is None:
            st.warning("The object did not land within the simulation time.")
        else:
            st.metric("Landing x-position [m]", f"{landing_x:.4f}")
            st.metric("Landing time [s]", f"{landing_time:.4f}")

            if success:
                st.success("SUCCESS: Object landed in the target region.")
            else:
                st.error("FAIL: Object missed the target region.")

        st.metric("Actual jet active time [s]", f"{actual_jet_active_time:.4f}")
        st.metric("Jet force / weight [-]", f"{jet_to_weight_ratio:.3f}")

        if landing_time is not None and jet_t_on > landing_time:
            st.warning(
                "Jet activates after the object has already landed. "
                "Decrease t_on or increase the initial height/velocity."
            )

        if actual_jet_active_time <= 0.0:
            st.warning(
                "The air jet was never active during the simulated flight. "
                "Try decreasing t_on or increasing t_max."
            )

        if target_x_max < target_x_min:
            st.warning(
                "The target region is invalid because x_max is smaller than x_min."
            )

        if jet_to_weight_ratio > 2.0:
            st.warning(
                "The jet force is more than twice the object weight. "
                "The result may be useful for sensitivity testing, but it may be physically aggressive."
            )

        st.subheader("Current Parameters")

        st.write(
            {
                "mass [kg]": mass,
                "weight [N]": weight,
                "area [m²]": area,
                "area [cm²]": area_cm2,
                "Cd [-]": drag_coefficient,
                "initial position [m]": (x0, y0),
                "initial velocity [m/s]": (vx0, vy0),
                "jet force [N]": (jet_fx, jet_fy),
                "jet force magnitude [N]": jet_force_magnitude,
                "jet t_on [s]": jet_t_on,
                "jet duration [s]": jet_duration,
            }
        )

        parameter_dict = make_parameter_dict()
        parameter_json = json.dumps(parameter_dict, indent=2)

        st.download_button(
            label="Download parameters as JSON",
            data=parameter_json,
            file_name="week1_parameters.json",
            mime="application/json",
        )

    st.subheader("Force History")

    force_fig = plot_force_history(result, show_force_component)
    st.pyplot(force_fig)

    st.subheader("Velocity History")

    velocity_fig = plot_velocity_history(result)
    st.pyplot(velocity_fig)

    st.subheader("Trajectory Data")

    df = result_to_dataframe(result)
    st.dataframe(df.head(20), use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download trajectory data as CSV",
        data=csv_data,
        file_name="week1_trajectory.csv",
        mime="text/csv",
    )

else:
    st.info("Adjust the sliders in the sidebar and click **Run Simulation**.")

    st.subheader("Recommended Starting Values")

    st.write(
        {
            "mass [kg]": 0.050,
            "projected area [m²]": 0.0100,
            "projected area [cm²]": 100.0,
            "Cd [-]": 1.0,
            "initial position [m]": (0.0, 1.0),
            "initial velocity [m/s]": (1.5, 0.0),
            "jet force [N]": (0.0, 0.10),
            "jet activation time [s]": 0.10,
            "jet duration [s]": 0.20,
            "target region [m]": (1.0, 2.0),
        }
    )
