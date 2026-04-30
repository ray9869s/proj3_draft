# Air-Jet Reinforcement Learning Project

This project develops a Week-based simulation framework for air-jet assisted sorting.

The current implementation includes:

1. Week 1: 2D point-mass air-jet trajectory simulation
2. Week 2: 3D rigid-body air-jet trajectory simulation
3. Later stage: reinforcement-learning environment and controller design

The final goal is to build a simulation-based reinforcement-learning workflow that can determine suitable air-jet control parameters for sorting objects into a target landing region.

---

## 1. Project Layout

```text
project3/
├── README.md
├── requirements.txt
├── reports/
│   ├── week1/
│   ├── week2/
│   └── final/
├── results/
│   ├── week1/
│   │   ├── data/
│   │   ├── figures/
│   │   └── videos/
│   ├── week2/
│   │   ├── data/
│   │   ├── figures/
│   │   └── videos/
│   └── rl/
│       ├── figures/
│       ├── logs/
│       ├── models/
│       └── videos/
├── scripts/
│   ├── app_week1_streamlit.py
│   ├── app_week2_streamlit.py
│   ├── run_week1_basic.py
│   ├── run_week1_sensitivity.py
│   ├── run_week2_analysis.py
│   ├── run_week2_demo.py
│   └── train_rl.py
└── src/
    ├── __init__.py
    ├── rl_env.py
    ├── utils.py
    ├── week1_2d.py
    └── week2_3d.py
```

### Directory Description

| Directory | Purpose |
|---|---|
| `src/` | Core simulation models, utility functions, and reinforcement-learning environment code |
| `scripts/` | Runnable scripts and Streamlit applications |
| `results/` | Generated figures, videos, logs, models, and data files |
| `reports/` | Week-based notes, draft reports, and final project materials |

---

## 2. Environment Setup and Running the Project

This project is designed to be run from the project root directory.

Before running any command, move into the project folder:

```bash
cd project3
```

If your project folder has a different name, replace `project3` with the actual folder name.

---

## 2.1 macOS Setup

### Step 1: Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 2: Activate the virtual environment

```bash
source .venv/bin/activate
```

After activation, your terminal should show something like:

```text
(.venv) user@computer project3 %
```

### Step 3: Install required packages

```bash
pip install -r requirements.txt
```

### Step 4: Run the Week 1 Streamlit app

```bash
streamlit run scripts/app_week1_streamlit.py
```

### Step 5: Run the Week 2 Streamlit app

```bash
streamlit run scripts/app_week2_streamlit.py
```

### Step 6: Stop the Streamlit app

In the terminal where Streamlit is running, press:

```text
Ctrl + C
```

---

## 2.2 Windows Setup

The following commands assume that Python is already installed and available from the terminal.

Use either Command Prompt or PowerShell.

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

If `python` does not work, try:

```powershell
py -m venv .venv
```

### Step 2: Activate the virtual environment

For PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

For Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

After activation, your terminal should show something like:

```text
(.venv) C:\path\to\project3>
```

If PowerShell blocks activation because of execution policy, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install required packages

```powershell
pip install -r requirements.txt
```

### Step 4: Run the Week 1 Streamlit app

```powershell
streamlit run scripts/app_week1_streamlit.py
```

### Step 5: Run the Week 2 Streamlit app

```powershell
streamlit run scripts/app_week2_streamlit.py
```

### Step 6: Stop the Streamlit app

In the terminal where Streamlit is running, press:

```text
Ctrl + C
```

---

## 2.3 Linux Setup

### Step 1: Create a virtual environment

```bash
python3 -m venv .venv
```

If `venv` is not installed, install it first.

For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3-venv
```

Then create the virtual environment again:

```bash
python3 -m venv .venv
```

### Step 2: Activate the virtual environment

```bash
source .venv/bin/activate
```

After activation, your terminal should show something like:

```text
(.venv) user@computer:~/project3$
```

### Step 3: Install required packages

```bash
pip install -r requirements.txt
```

### Step 4: Run the Week 1 Streamlit app

```bash
streamlit run scripts/app_week1_streamlit.py
```

### Step 5: Run the Week 2 Streamlit app

```bash
streamlit run scripts/app_week2_streamlit.py
```

### Step 6: Stop the Streamlit app

In the terminal where Streamlit is running, press:

```text
Ctrl + C
```

---

## 2.4 Common Run Commands

Run Week 1 interactive simulator:

```bash
streamlit run scripts/app_week1_streamlit.py
```

Run Week 2 interactive simulator:

```bash
streamlit run scripts/app_week2_streamlit.py
```

Run the basic Week 1 script:

```bash
python scripts/run_week1_basic.py
```

Run the Week 1 sensitivity script:

```bash
python scripts/run_week1_sensitivity.py
```

Run the Week 2 demo script:

```bash
python scripts/run_week2_demo.py
```

Run the Week 2 analysis script:

```bash
python scripts/run_week2_analysis.py
```

---

## 3. GitHub Usage Guide

This section explains how to upload the project to GitHub and how team members can collaborate.

---

## 3.1 First-Time GitHub Upload

These steps should be done by the person who first creates the GitHub repository.

### Step 1: Check `.gitignore`

Before uploading the project, make sure `.gitignore` excludes virtual environments, cache files, generated results, model checkpoints, and large files.

Important files and folders that should usually not be uploaded:

```text
.venv/
__pycache__/
.DS_Store
results/**/*.csv
results/**/*.png
results/**/*.mp4
results/rl/models/
```

### Step 2: Initialize Git

From the project root:

```bash
git init
```

Check the current file status:

```bash
git status
```

Make sure `.venv/` is not listed.

### Step 3: Add files

```bash
git add .
```

Check the staged files:

```bash
git status
```

If `.venv/`, cache files, or large result files appear, stop and update `.gitignore` before committing.

### Step 4: Create the first commit

```bash
git commit -m "Initial air-jet simulation project"
```

### Step 5: Create a new GitHub repository

On GitHub:

```text
GitHub → New repository
```

Recommended settings:

```text
Repository name: air-jet-rl-project
Visibility: Private
Initialize with README: No
Add .gitignore: No
Choose a license: No
```

Do not initialize with a README because this project already has one.

### Step 6: Connect local repository to GitHub

Replace `USERNAME` and `REPOSITORY_NAME` with your GitHub username and repository name.

```bash
git remote add origin https://github.com/USERNAME/REPOSITORY_NAME.git
git branch -M main
git push -u origin main
```

Example:

```bash
git remote add origin https://github.com/myname/air-jet-rl-project.git
git branch -M main
git push -u origin main
```

---

## 3.2 Adding Team Members

If the repository is private, team members need access permission.

On GitHub:

```text
Repository page → Settings → Collaborators → Add people
```

Invite team members using their GitHub usernames or email addresses.

Each team member must accept the invitation before cloning the repository.

---

## 3.3 Team Member Setup

Team members should clone the repository and set up their own virtual environment.

### Step 1: Clone the repository

```bash
git clone https://github.com/USERNAME/REPOSITORY_NAME.git
```

Example:

```bash
git clone https://github.com/myname/air-jet-rl-project.git
```

### Step 2: Move into the project folder

```bash
cd REPOSITORY_NAME
```

Example:

```bash
cd air-jet-rl-project
```

### Step 3: Create and activate a virtual environment

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 4: Install packages

```bash
pip install -r requirements.txt
```

### Step 5: Run the app

```bash
streamlit run scripts/app_week2_streamlit.py
```

---

## 3.4 Basic Git Workflow

Use the following workflow when making changes.

### Step 1: Check current status

```bash
git status
```

### Step 2: Pull the latest version

Before editing, get the newest version from GitHub:

```bash
git pull
```

### Step 3: Edit files

Modify code, README, reports, or scripts.

### Step 4: Check changes

```bash
git status
```

To see detailed changes:

```bash
git diff
```

### Step 5: Add changed files

Add all changed files:

```bash
git add .
```

Or add a specific file:

```bash
git add scripts/app_week2_streamlit.py
```

### Step 6: Commit changes

```bash
git commit -m "Describe the change briefly"
```

Example:

```bash
git commit -m "Add reset buttons to Week 2 Streamlit app"
```

### Step 7: Push changes

```bash
git push
```

---

## 3.5 Recommended Branch Workflow

For team collaboration, avoid making all changes directly on `main`.

Create a new branch for each feature or task.

### Create a new branch

```bash
git checkout -b feature/week2-ui
```

### Work on files, then commit

```bash
git add .
git commit -m "Improve Week 2 UI"
```

### Push the branch to GitHub

```bash
git push -u origin feature/week2-ui
```

### Open a Pull Request

On GitHub:

```text
Repository page → Pull requests → New pull request
```

Review the changes and merge into `main` after approval.

### Switch back to main

```bash
git checkout main
```

### Pull the updated main branch

```bash
git pull
```

---

## 3.6 Suggested Branch Names

Use clear branch names.

```text
feature/week1-simulator
feature/week2-3d-model
feature/week2-ui
feature/rl-env
feature/report
fix/week2-landing-time
fix/streamlit-reset
docs/readme-update
```

---

## 3.7 Suggested Commit Message Style

Use short and clear commit messages.

Examples:

```text
Add Week 2 Streamlit simulator
Fix landing time for not-landed cases
Add rod cylinder visualization
Update README setup instructions
Add section reset buttons
Refactor air jet x-profile
```

---

## 3.8 Files That Should Not Be Committed

Do not commit:

```text
.venv/
__pycache__/
.DS_Store
large result files
temporary test files
private API keys
Streamlit secrets
model checkpoints unless intentionally needed
```

Generated result files should usually remain local unless the team explicitly decides to share them.

---

## 4. Week 1: 2D Air-Jet Simulator

Week 1 implements a simplified 2D air-jet sorting model.

The object is treated as a point mass moving in the `x-z` plane under gravity, aerodynamic drag, and finite-duration air-jet force.

---

## 4.1 Coordinate System

The Week 1 coordinate convention is:

| Axis | Meaning |
|---|---|
| `x` | Horizontal conveyor direction |
| `z` or `y` in 2D plots | Vertical direction, depending on plotting convention |

The main quantity of interest is the landing position along the conveyor direction.

```text
target_x_min <= landing_x <= target_x_max
```

---

## 4.2 Physical Model

The object is modeled using translational motion only.

The forces considered are:

| Force | Description |
|---|---|
| Gravity | Constant downward force |
| Aerodynamic drag | Drag force opposite to the object velocity |
| Air-jet force | User-defined force applied for a finite time interval |

The air jet is active during:

```text
t_on <= t <= t_on + duration
```

The Week 1 model does not include object rotation, distributed surface force, or torque.

---

## 4.3 Main Inputs

| Parameter | Meaning |
|---|---|
| `mass` | Object mass |
| `projected area` | Effective area used for drag calculation |
| `drag coefficient` | Effective drag coefficient |
| `initial position` | Initial object position |
| `initial velocity` | Initial object velocity |
| `jet force` | Force vector applied by the air jet |
| `jet activation time` | Time when the air jet starts |
| `jet duration` | Time interval during which the jet remains active |
| `target region` | Acceptable landing range |

---

## 4.4 Week 1 Purpose

The Week 1 simulator is mainly used to:

- understand the basic effect of air-jet force on landing position
- test sensitivity to mass, projected area, drag coefficient, jet timing, and jet strength
- provide a simple baseline before moving to the 3D rigid-body model
- build intuition for later reinforcement-learning control variables

---

## 4.5 Week 1 Limitations

The Week 1 model is intentionally simple.

Current limitations include:

- no object rotation
- no torque
- no distributed surface force
- no 3D motion
- no orientation-dependent drag
- no post-impact bouncing, sliding, or rolling

Despite these limitations, the Week 1 model is useful for quickly testing how basic air-jet parameters affect landing behavior.

---

## 5. Week 2: 3D Rigid-Body Air-Jet Simulator

Week 2 extends the model from 2D point-mass motion to 3D rigid-body motion.

The object is represented by discrete surface points, allowing the simulator to compute both translational force and rotational torque.

This simulator is intended as a baseline physical environment before building a reinforcement-learning controller.

---

## 5.1 Coordinate System

The Week 2 coordinate convention is:

| Axis | Meaning |
|---|---|
| `x` | Conveyor belt direction |
| `y` | Belt width direction / air-jet nozzle position direction |
| `z` | Vertical direction |

The target is defined by the landing position along the `x` direction.

```text
target_x_min <= landing_x <= target_x_max
```

---

## 5.2 Object Models

The simulator currently supports three object types:

1. `plate`
2. `rod`
3. `irregular`

### Plate

A plate is modeled as a thin rectangular object.

It is represented by surface points on its top and bottom faces.

This option is useful for modeling:

- thin plastic flakes
- flat objects
- film-like pieces

The user specifies:

| Parameter | Meaning |
|---|---|
| `size_x` | Object length in the conveyor direction |
| `size_y` | Object width in the belt-width direction |
| `size_z` | Object thickness |

### Rod

A rod is modeled as a cylindrical object aligned with the body-frame `x` axis.

The user specifies:

| Parameter | Meaning |
|---|---|
| `rod_length` | Cylinder length |
| `rod_radius` | Cylinder radius |

The rod is visualized as a cylinder in the 3D plot.

Its inertia tensor is approximated using the solid-cylinder formula.

```text
I_xx = 1/2 m r^2
I_yy = I_zz = 1/12 m (3r^2 + L^2)
```

### Irregular

An irregular object is represented by a random point cloud inside the given object dimensions.

It is intended to approximate an asymmetric plastic flake or non-uniform particle.

Unlike plate and rod objects, the irregular object is visualized as point cloud data rather than as a bounding box. This makes its asymmetric geometry easier to inspect.

---

## 5.3 Air-Jet Model

The air jet is modeled as a finite spatial region near the end of the conveyor belt.

The user specifies:

| Parameter | Meaning |
|---|---|
| `x_start` | Start position of the air-jet region |
| `x_width` | Thickness of the air-jet region along the conveyor direction |
| `y_center` | Center of the jet in the belt-width direction |
| `z_center` | Center of the jet in the vertical direction |
| `sigma` | Gaussian width in the `y-z` plane |
| `angle_deg` | Jet direction angle relative to the `+x` direction |
| `t_on` | Jet activation time |
| `duration` | Jet activation duration |
| `umax` | Maximum jet velocity |

### Finite X-Zone

The air jet exists only in the finite x-zone:

```text
x_start <= x <= x_start + x_width
```

Instead of using a hard on/off profile in the `x` direction, the simulator uses a raised-cosine profile.

The x-profile is:

- zero at `x_start`
- maximum at the center of the x-zone
- zero at `x_start + x_width`

This is intended to better approximate a real air jet, which is strongest near the center of the jet region and weaker near the boundaries.

### Gaussian Y-Z Profile

In the `y-z` plane, the jet uses a Gaussian profile:

```text
exp(-((y - y_center)^2 + (z - z_center)^2) / (2 sigma^2))
```

This means that the jet is strongest near `(y_center, z_center)` and becomes weaker as the object surface moves away from that center.

### Jet Direction

The jet direction is controlled by `angle_deg`.

| Angle | Approximate Direction |
|---|---|
| `0 deg` | Along `+x` |
| `45 deg` | Forward and upward |
| `90 deg` | Along `+z` |

The direction vector is defined in the `x-z` plane.

```text
e_jet = [cos(theta), 0, sin(theta)]
```

---

## 5.4 Distributed Surface Force Model

The object is represented by discrete surface points.

Each surface point has:

- a body-frame position
- an approximate surface normal
- an area weight

At each time step, the simulator computes the local relative velocity between the jet and the moving surface point.

```text
u_rel = u_jet - v_surface
```

The local force is weighted by the alignment between the relative flow direction and the local surface normal.

This is still a simplified aerodynamic model, but it is more physical than applying the same force to all surface points regardless of surface orientation.

---

## 5.5 Rotation and Torque

The simulator computes local jet forces at surface points and sums them to obtain:

- total jet force
- total jet torque

The torque is computed from:

```text
tau = r x F
```

where `r` is the vector from the center of mass to each surface point.

The rigid-body orientation is updated using quaternions.

---

## 5.6 Initial Orientation

The initial object orientation can be controlled by:

| Parameter | Meaning |
|---|---|
| `roll` | Initial rotation around the `x` axis |
| `pitch` | Initial rotation around the `y` axis |
| `yaw` | Initial rotation around the `z` axis |

These Euler angles are converted into the initial quaternion used by the rigid-body simulator.

This is useful because the landing trajectory of a plate, rod, or irregular object may strongly depend on its initial orientation.

---

## 5.7 Landing Criterion

The simulation stops when the lowest surface point first reaches the landing plane:

```text
lowest_surface_z <= landing_z
```

The model predicts the first-contact landing position.

It does not model what happens after contact, such as:

- bouncing
- sliding
- rolling
- frictional contact
- post-impact rotation

This choice is intentional because the current goal is to predict the first landing location for sorting.

---

## 5.8 Outputs and Visualization

The Streamlit app provides:

| Output | Description |
|---|---|
| 3D trajectory plot | Center-of-mass trajectory, initial/final object geometry, target region, and jet x-zone |
| x-y projection | Top-view trajectory and target x-region |
| x-z projection | Side-view trajectory and vertical motion |
| y-z projection | Jet cross-section view |
| Jet force plot | Jet-only force components |
| Total force plot | Gravity + drag + jet force components |
| Torque plot | Jet-induced torque components |
| Angular velocity plot | Rigid-body angular velocity |
| Data table | Time histories of position, velocity, force, torque, and angular velocity |
| JSON export | Simulation parameters |
| CSV export | Trajectory and force data |

---

## 5.9 Current Limitations

This model is designed as a simplified baseline simulator, not as a high-fidelity CFD solver.

Current limitations include:

- no detailed CFD flow field
- no post-impact collision dynamics
- no bounce, sliding, or friction after landing
- simplified whole-body drag force
- no orientation-dependent projected-area drag model
- approximate surface normals for irregular objects
- explicit time integration, which may require time-step sensitivity checks for very strong jets
- simplified air-jet velocity field rather than a measured or CFD-derived flow field

Despite these limitations, the model is useful for exploring how object mass, shape, initial orientation, jet timing, jet angle, and jet position affect landing behavior.

---

## 6. Planned Reinforcement-Learning Stage

After the Week 1 and Week 2 simulators are validated, the next stage is to define a reinforcement-learning environment.

A future RL environment may include the following components.

### State

Possible state variables include:

- object type
- object mass
- object size or rod radius/length
- initial position
- initial velocity
- initial angular velocity
- initial orientation
- target range

### Action

Possible action variables include:

- jet activation time
- jet duration
- jet angle
- jet maximum velocity

### Reward

A simple reward function may include:

- positive reward for landing inside the target region
- penalty for distance from the target center
- penalty for excessive jet usage
- penalty if the object does not land within the simulation time

The goal of the RL stage is to learn a control policy that chooses air-jet parameters to sort objects into the desired target region.

---

## 7. Suggested Validation Tests

Before training an RL controller, the simulator should be checked using simple sanity tests.

### Week 1 Tests

- Set jet force to zero and verify basic projectile-like motion under gravity and drag.
- Increase jet force and check whether the landing position changes in the expected direction.
- Vary object mass and verify that heavier objects are less affected by the same jet force.
- Vary projected area and drag coefficient to check drag sensitivity.

### Week 2 Tests

- Set `umax = 0` and verify that jet force and jet torque are nearly zero.
- Vary `jet_angle_deg` and verify that the force direction changes.
- Move `y_center` and `z_center` away from the object and verify that jet force decreases.
- Increase mass and verify that the object becomes less sensitive to the same jet.
- Change initial roll, pitch, and yaw and verify that torque and landing behavior change.
- Compare plate, rod, and irregular objects under the same jet settings.
- Reduce the time step and check whether the landing result remains similar.

---

## 8. Summary

This project currently provides a two-stage simulation workflow:

1. Week 1 builds intuition using a simple 2D point-mass model.
2. Week 2 extends the problem to 3D rigid-body motion with distributed surface force and torque.

The current 3D simulator is not intended to be a high-fidelity aerodynamic solver.

Instead, it provides a controllable and interpretable baseline model that can later be converted into a reinforcement-learning environment.