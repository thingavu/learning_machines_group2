# Group 2 - Learning Machines Course Solution

This repository contains the solution by Group 2 for the Learning Machines course assignments. The project is based on the `full_project_setup` template, with added and modified files that implement solutions for the course tasks.

---

## Key Features

### 1. Code for Assignment 0
- **`a0_actions.py`**:
  - Located in `catkin_ws/src/learning_machines/src/learning_machines/`.
  - Implements the solution for Assignment 0:
    - The robot moves forward until it detects an obstacle (e.g., a wall).
    - It then moves backward, turns right, and moves forward again for a short distance.
    - During the experiment, infrared sensor readings are continuously collected and stored in `.xlsx` files.

- **Modified `__init__.py`**:
  - Located in the same directory as `a0_actions.py`.
  - Modified to execute the functionality implemented in `a0_actions.py`.

### 2. Experiment Results for Assignment 0
- **Results Directory**:
  - The `results/` directory contains the experimental outputs:
    - `hardware_results/`: results from experiments conducted using hardware.
    - `simulation_results/`: results from experiments conducted in simulation.
  - Data is stored in `.xlsx` files, with each file corresponding to a specific experiment and including sensor readings, timestamps, and other relevant details.

### 3. Expanded Dependencies
- **`requirements.txt`**:
  - Contains all the necessary Python packages and libraries required to run the project.
