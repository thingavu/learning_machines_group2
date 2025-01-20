# wall /obstacle avoiding robot. drive around as much as possible and avoid obstacles
import os

import gym
from gym import spaces
import numpy as np
from robobo_interface import IRobobo
from stable_baselines3 import DQN
from data_files import RESULTS_DIR
import re
import time
import pandas as pd

COLLISION_THRESHOLD = 120
DANGER_THRESHOLD = 80  # Getting very close to obstacle

SAFE_THRESHOLD = 60  # Good distance for navigation
TIMESTEPS = 5000


def get_state(obs):
    if any(sensor > 100 for sensor in obs):
        s0 = "collision"
    elif any(sensor > DANGER_THRESHOLD for sensor in obs):
        s0 = "close"
    elif all(sensor < SAFE_THRESHOLD for sensor in obs):
        s0 = "safe"
    else:
        s0 = "tight"

    front_obstacle = obs[0] > DANGER_THRESHOLD or obs[1] > DANGER_THRESHOLD or obs[2] > DANGER_THRESHOLD
    if front_obstacle:
        # Get readings from sensors (L, R, C, RR, LL)
        left_side_danger = max(obs[0], obs[4])  # LL and L sensors
        right_side_danger = max(obs[1], obs[3])

        if right_side_danger > left_side_danger:
            s1 = "FOR"
        else:
            s1 = "FOL"
    else:
        s1 = "NFO"

    return [s0, s1]


def map_observations(obs):
    mapping = {"FrontL": obs[0],
               "FrontR": obs[1],
               "FrontC": obs[2],
               "FrontRR": obs[3],
               "FrontLL": obs[4]}
    return mapping


class RoboboRLEnvironment(gym.Env):
    def __init__(self, robot: IRobobo, experiment_number, mode, max_steps=100):
        super(RoboboRLEnvironment, self).__init__()
        self.robot = robot
        self.experiment_number = experiment_number
        self.sensor_readings = []
        self.start_time = time.time()
        self.mode = mode

        # Define action space (3 discrete actions: forward, turn left, turn right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (5 IR sensors)
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(5,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = max_steps
        self.prev_sensor_readings = np.zeros(5)
        self.prev_action = None

    def get_current_time(self):
        return round((time.time() - self.start_time) * 1000, 0)  # in milliseconds

    def reset(self):
        """Reset the environment"""
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()

        if hasattr(self.robot, 'play_simulation'):
            self.robot.play_simulation()

        self.current_step = 0

        # Get initial observation
        obs = self.get_observation()
        self.prev_sensor_readings = obs
        return obs

    def step(self, action):
        """Execute action and return new state"""
        initial_observation = self.get_observation()
        # Execute action
        if action == 0:  # Move Forward
            self.robot.move_blocking(40, 40, 300)  # (40, 40, 300)
        elif action == 1:  # Turn Left
            self.robot.move_blocking(-20, 20, 300)  # (-20, 20, 300)
        else:  # Turn Right
            self.robot.move_blocking(20, -20, 300)  # (20, -20, 300)
        # Get new observation
        new_observation = self.get_observation()

        decoded_action = "forward" if action == 0 else "left" if action == 1 else "right"
        # Calculate reward
        reward = self._calculate_reward(new_observation, action)

        self.sensor_readings, saving_result = record_data(self.sensor_readings, self.get_current_time(),
                                                          new_observation, action=decoded_action,
                                                          state=get_state(new_observation), reward=reward,
                                                          experiment_number=self.experiment_number, mode=self.mode)
        if saving_result == "Stuck":
            for _ in range(4):
                self.robot.move_blocking(-20, 20, 300)

        # Check if episode is done
        done = self._check_done(new_observation)

        # Update previous readings
        self.prev_sensor_readings = new_observation
        self.current_step += 1

        return new_observation, reward, done, {}

    def _calculate_reward(self, obs, action):
        """Calculate the reward"""
        reward = 0

        # Heavy penalty for potential collisions (readings > 100)
        if any(sensor > 100 for sensor in obs):
            reward -= 10
            return reward  # Immediate return for collision

        # Penalty for getting too close to obstacles
        if any(sensor > DANGER_THRESHOLD for sensor in obs):
            reward -= 5

        # Reward for moving forward while maintaining safe distance
        if action == 0:  # Forward action
            # Check if path ahead is relatively clear
            if all(sensor < SAFE_THRESHOLD for sensor in obs):
                reward += 5  # Good forward movement
            else:
                reward += 0.5  # Smaller reward if moving forward in tighter spaces

        # Handling turning actions with conditional rewards
        elif action in [1, 2]:  # Turning actions
            # Check if there's an obstacle ahead (center sensors likely indicating front view)
            front_obstacle = obs[0] > DANGER_THRESHOLD or obs[1] > DANGER_THRESHOLD or obs[2] > DANGER_THRESHOLD

            if front_obstacle:
                # Get readings from sensors (L, R, C, RR, LL)
                left_side_danger = max(obs[0], obs[4])  # LL and L sensors
                right_side_danger = max(obs[1], obs[3])  # R and RR sensors

                # Reward turning if it directs the robot away from the greater danger
                if action == 1 and right_side_danger > left_side_danger:  # FOR
                    reward += 2
                elif action == 2 and left_side_danger > right_side_danger:  # FOL
                    reward += 2
                else:
                    reward -= 0.5  # Small penalty if turning in a suboptimal direction
            else:
                # Penalize turning when no immediate obstacle is detected ahead
                reward -= 1

        # Reward for maintaining good exploration distance (safe distance from all obstacles)
        if all(sensor < SAFE_THRESHOLD for sensor in obs):
            reward += 2

        return reward

    def _check_done(self, obs):
        """Check if episode should end"""
        if self.current_step % 200 == 0:
            print(f"Episode {self.current_step}/{self.max_steps}")
        # End if collision
        if any(sensor > COLLISION_THRESHOLD for sensor in obs):
            if self.mode == "test":
                s0, s1 = get_state(self.get_observation())
                self.robot.move_blocking(-40, -40, 300)
                if s1 == "FOR":
                    self.robot.move_blocking(-20, 20, 300)
                    self.robot.move_blocking(-20, 20, 300)
                elif s1 == "FOL":
                    self.robot.move_blocking(20, -20, 300)
                    self.robot.move_blocking(20, -20, 300)
                return False
            else:
                return True

        # End if max steps reached
        if self.current_step >= self.max_steps:
            return True

        # End if simulation stopped
        if hasattr(self.robot, 'is_stopped') and self.robot.is_stopped():
            return True

        return False

    def get_observation(self):
        """Get current observation"""
        ir_sensors = self.robot.read_irs()
        # Use front sensors
        front_sensors = [ir_sensors[i] for i in [2, 3, 4, 5, 7]]  # using 5 front sensors
        return np.array(front_sensors, dtype=np.float32)

    def close(self):
        """Clean up"""
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()


def record_data(sensor_readings, timestamp, obs, action, state, reward,
                experiment_number, mode):
    # print(f"Data recorded at {timestamp}: {obs}")
    sensor_readings.append((timestamp, obs, action, state, reward))
    saving_result = store_sensor_data_in_csv(sensor_readings, experiment_number, mode)
    return [], saving_result


def store_sensor_data_in_csv(sensor_readings, experiment_number, mode):
    """Store sensor data in a CSV file, including time, sensor values, and experiment details.
    If the file already exists, append the new data to it.
    """
    # Create a DataFrame to store sensor readings along with additional info
    data = {
        "Time": [],
        "Mode": [],
        "FrontLL": [],
        "FrontL": [],
        "FrontC": [],
        "FrontR": [],
        "FrontRR": [],
        "State0": [],
        "State1": [],
        "Action": [],
        "Reward": []}

    for reading in sensor_readings:
        data["Time"].append(reading[0])
        data["Mode"].append(mode)
        data["FrontLL"].append(reading[1][4])
        data["FrontL"].append(reading[1][0])
        data["FrontC"].append(reading[1][2])
        data["FrontR"].append(reading[1][1])
        data["FrontRR"].append(reading[1][3])
        data["Action"].append(reading[2])
        data["State0"].append(reading[3][0])
        data["State1"].append(reading[3][1])
        data["Reward"].append(reading[4])

    new_data = pd.DataFrame(data)

    # CSV file path
    csv_path = RESULTS_DIR / f"{mode}_experiment_{experiment_number}.csv"

    # Append new data if the file already exists
    if csv_path.exists():
        existing_data = pd.read_csv(csv_path)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        combined_data = new_data

    # Save the data to the CSV file
    combined_data.to_csv(csv_path, index=False)
    # print(f"Sensor data saved to: {csv_path}")

    if mode == "test":
        two_last_actions = combined_data.tail(2)["Action"].values
        if "right" in two_last_actions and "left" in two_last_actions:
            print("Stuck")
            return "Stuck"
    return "Saved"


def train_model(rob: IRobobo, experiment_number):
    # Create environment
    env = RoboboRLEnvironment(rob, experiment_number, mode="train")

    # Create the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.01,  # Much lower learning rate for stability
        buffer_size=5000,
        learning_starts=200,
        batch_size=32,  # Smaller batch size
        exploration_initial_eps=0.6,  # Higher initial exploration
        exploration_final_eps=0.1,
        exploration_fraction=0.8,  # Slower exploration decay
        target_update_interval=250)  # More frequent target updates

    # Train the model
    try:
        print("Starting training...")
        model.learn(total_timesteps=TIMESTEPS)
        print("Training completed!")

        # Save the model
        # create filename
        filename = os.path.join(RESULTS_DIR, "robobo_dqn_model.pkl")
        model.save(filename)
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        env.close()


def test_model(rob: IRobobo, experiment_number):
    # Create environment
    env = RoboboRLEnvironment(rob, experiment_number, mode="test", max_steps=TIMESTEPS)

    filename = os.path.join(RESULTS_DIR, "robobo_dqn_model.pkl")

    # Load the trained model
    model = DQN.load(filename)

    # Test the model
    try:
        print("Starting testing...")
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Testing completed! Total reward: {total_reward}")

    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        env.close()


def run_all_actions(rob: IRobobo, mode):
    pause = 3
    print("Starting in...")
    for i in reversed(range(pause)):
        print(i + 1)
        time.sleep(1)

    experiment_number = 1
    experiment_files = list(RESULTS_DIR.glob(f"{mode}_experiment_*.csv"))
    if experiment_files:
        experiment_files.sort()
        latest_file = experiment_files[-1]
        match = re.search(r"experiment_(\d+)", latest_file.name)  # Extract the number after "experiment_"
        experiment_number = int(match.group(1)) + 1

    # If it's a simulation robot, start the simulation
    if hasattr(rob, 'play_simulation'):
        rob.play_simulation()

    try:
        if mode == "train":
            # Create and train the model
            print("Starting training phase...")
            train_model(rob, experiment_number=experiment_number)
        else:
            print("\nStarting testing phase...")
            test_model(rob, experiment_number=experiment_number)

    except KeyboardInterrupt:
        print("\nStopping robot...")
    finally:
        # If it's a simulation robot, stop the simulation
        if hasattr(rob, 'stop_simulation'):
            rob.stop_simulation()
