# Group 2 - Learning Machines Course Solution

This repository contains the solution by Group 2 for the Learning Machines course assignments. The project is based on the `full_project_setup` template, with added and modified files that implement solutions for the course tasks.

## Key Features

### 1. Code for Assignment 0
- **`a0_actions.py`**:
  - Located in `catkin_ws/src/learning_machines/src/learning_machines/`.
  - Implements the solution for Assignment 0:
    - The robot moves forward until it detects an obstacle (e.g., a wall).
    - It then moves backward, turns right, and moves forward again for a short distance.
    - During the experiment, infrared sensor readings are continuously collected and stored in `.xlsx` files.

### 2. Code for Assignment 1
- **`a1_avoid_environment.py`**:
  - Located in `catkin_ws/src/learning_machines/src/learning_machines/`.
  - Implements the solution for Assignment 1:
    - Train a robot to navigate an environment without collisions using the Deep Q-Network (DQN) reinforcement learning algorithm.
    - The robot's goal is to maximize its travel distance by selecting actions that allow it to move forward towards the “safest” space, avoiding obstacles.
    - The `arena_obstacles.ttt` scene is used to train the model.
    - The `arena_obstacles_validation.ttt` scene is used to validate the trained model's performance.
    - During all the experiments, infrared sensor readings, states, actions and rewards are continuously collected and stored in `.csv` files.

### 3. Code for Assignment 2
- **`a2_food.py`**:
  - Located in `catkin_ws/src/learning_machines/src/learning_machines/`.
  - Implements the solution for Assignment 2:
    - Train a robot to find, approach and touch (eat) green items (food) using the Deep Q-Network (DQN) reinforcement learning algorithm.
    - The `arena_approach.ttt` scene is used to train the model.
    - The `arena_approach_validation.ttt` scene, which randomizes the location of items, is used to validate the trained model's performance.
    - During all the experiments, infrared sensor readings, states, actions and rewards are continuously collected and stored in `.csv` files.
- **`process_images.py`**:
  - Located in `catkin_ws/src/learning_machines/src/learning_machines/`.
  - Provides functionality for detecting green objects in images, calculating their properties, and annotating the images for debugging and analysis.

### 4. Experiment Results for all Assignments
- **Results Directory**:
  - The `results/` directory contains the experimental outputs:
    - `a0_hardware_results/`: results from experiments conducted using hardware for assignment 0.
    - `a0_simulation_results/`: results from experiments conducted in simulation for assignment 0.
    - `a1_results/`: results from train, test, validation experiments conducted in simulation for assignment 1, along with the trained model.
    - `a2_results/`: results from train, test, validation experiments conducted in simulation for assignment 2, along with the trained model.

### 5. Expanded Dependencies
- **`requirements.txt`**:
  - Contains all the necessary Python packages and libraries required to run the project.
