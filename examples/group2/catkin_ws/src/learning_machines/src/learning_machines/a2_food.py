import os
import csv
import gym
from gym import spaces
import numpy as np
import cv2
from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo
from stable_baselines3 import DQN
from data_files import RESULTS_DIR
import re
import time

# Import nearest green object detection from our feature module
from .process_images import detect_nearest_green_object

TIMESTEPS = 5000
N_TESTS = 1
TEST_MODE = "test"  # "validation"


class RoboboItemCollectionEnvironment(gym.Env):
    def __init__(self, robot: IRobobo, experiment_number, mode, max_steps=100):

        super(RoboboItemCollectionEnvironment, self).__init__()
        self.robot = robot
        self.experiment_number = experiment_number
        self.sensor_readings = []
        self.start_time = time.time()
        self.mode = mode
        self.nr_food_collected = 0
        self.state = []

        # Configure robot-specific parameters
        if isinstance(self.robot, SimulationRobobo):
            self.wheel_speed = {
                "forward": (70, 70),
                "backward": (-50, -50),
                "left": (-30, 30),
                "right": (30, -30)
            }
            self.movement_duration = 250
            self.turn_duration = 250
            self.thresholds = {"collision": 120, "food_collection": 53}  # "danger": 80, "safe": 60,
        elif isinstance(self.robot, HardwareRobobo):
            self.wheel_speed = {
                "forward": (80, 80),  # 80
                "backward": (-80, -80),  # 80
                "left": (-70, 70),  # 70
                "right": (70, -70)  # 70
            }
            self.movement_duration = 380  # 380
            self.turn_duration = 120  # 120
            self.thresholds = {"collision": 120, "food_collection": 53}  # "danger": 70, "safe": 30,
        else:
            raise ValueError("Unknown robot type. Must be SimulationRobobo or HardwareRobobo.")

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1e6, shape=(12,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = max_steps
        self.prev_food = 0
        self.prev_green_pixel_count = 0

        log_path = os.path.join(RESULTS_DIR, f'{mode}_robot_item_collection_log_{experiment_number}.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        header = [
            'step',
            'BackL', 'BackR', 'FrontL', 'FrontR', 'FrontC', 'FrontRR', 'BackC', 'FrontLL',
            'green_pixel_count', 'nearest_area', 'centroid_x', 'centroid_y',
            'action', 'reward', 'distance_to_lower_center',
            'food_collected', 'distance_delta', 'target_in_sight', 'stuck', 'oscillatory_action',
            'nr_food_collected']
        self.csv_writer.writerow(header)

        self.collision_delay_counter = 0  # Counter to track persistent collisions
        self.collision_check_delay = 3  # Number of steps to confirm a wall collision

    def get_nr_food_collected(self):
        if isinstance(self.robot, SimulationRobobo):
            current_food = self.robot.get_nr_food_collected()
        else:
            current_food = self.nr_food_collected
        return current_food

    def reset(self):
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()
        if hasattr(self.robot, 'play_simulation'):
            self.robot.play_simulation()

        if hasattr(self.robot, 'set_phone_tilt_blocking'):
            self.robot.set_phone_tilt_blocking(tilt_position=100, tilt_speed=100)

        self.current_step = 0
        self.prev_food = self.get_nr_food_collected()
        self.prev_green_pixel_count = 0
        self.nr_food_collected = 0

        obs = self.get_observation()
        # Initialize distance tracking after first observation
        initial_distance = self.last_features['distance_to_lower_center']
        self.prev_distance = initial_distance if initial_distance is not None else float('inf')
        self.prev_area = np.log1p(self.last_features['nearest_object_area'] or 0)
        return obs

    def step(self, action):
        # print("pan: ", self.robot.read_phone_pan())  # 0
        # print("tilt: ", self.robot.read_phone_tilt())  # 106
        initial_obs = self.get_observation()
        if action == 0:
            self.robot.move_blocking(*self.wheel_speed["forward"], self.movement_duration)
        elif action == 1:
            self.robot.move_blocking(*self.wheel_speed["left"], self.turn_duration)
        elif action == 2:
            self.robot.move_blocking(*self.wheel_speed["right"], self.turn_duration)
        elif action == 3:
            self.robot.move_blocking(*self.wheel_speed["backward"], self.movement_duration)
        new_obs = self.get_observation()
        decoded_action = "forward" if action == 0 else "left" if action == 1 else "right" if action == 2 else "backward"

        reward = self._calculate_reward(initial_obs, new_obs, action)
        done = self._check_done()

        log_row = (
                [self.current_step] +
                new_obs.tolist() +
                [decoded_action, reward, self.last_features['distance_to_lower_center']] +
                self.state)
        if self.csv_writer:
            self.csv_writer.writerow(log_row)

        self.current_step += 1
        return new_obs, reward, done, {}

    def get_state(self, initial_obs, new_obs, action):
        action = int(action)

        # Food collected
        initial_front_sensors = [initial_obs[i] for i in [2, 3, 4, 5, 7]]
        initial_area = initial_obs[9]
        target_was_in_sight = initial_area > 0
        custom_collection_check = any(
            sensor > self.thresholds["food_collection"] for sensor in initial_front_sensors) and target_was_in_sight
        if isinstance(self.robot, SimulationRobobo):
            current_food = self.robot.get_nr_food_collected()
            if self.mode == "train":
                food_collected = current_food - self.prev_food > 0 and custom_collection_check
            else:
                food_collected = current_food - self.prev_food > 0
            nr_food_collected = current_food - self.prev_food
            self.prev_food = current_food
        else:
            nr_food_collected = 1
            food_collected = custom_collection_check

        # Change in nearest green object distance
        current_distance = self.last_features['distance_to_lower_center']
        current_distance = current_distance if current_distance is not None else float('inf')
        distance_delta = self.prev_distance - current_distance
        if not np.isfinite(distance_delta):
            distance_delta = 0
        self.prev_distance = current_distance

        # Target in sight
        current_area = np.log1p(self.last_features['nearest_object_area'] or 0)
        target_in_sight = current_area > 0
        front_sensors = [new_obs[i] for i in [2, 3, 4, 5, 7]]

        # Stuck
        stuck = any(sensor > self.thresholds["collision"] for sensor in front_sensors) and not target_in_sight

        # Oscillatory behavior
        oscillatory_action = False
        if hasattr(self, 'last_action') and action != self.last_action:
            if (self.last_action, action) in [(1, 2), (2, 1), (0, 3), (3, 0)]:  # Oscillating between left and right
                oscillatory_action = True
        self.last_action = action

        self.state = [food_collected, distance_delta, target_in_sight, stuck, oscillatory_action, nr_food_collected]
        return self.state

    def _calculate_reward(self, initial_obs, new_obs, action):
        reward = 0
        action = int(action)

        food_collected, distance_delta, target_in_sight, stuck, oscillatory_action, nr_food = self.get_state(initial_obs,
                                                                                                             new_obs,
                                                                                                             action)

        if food_collected:
            for _ in range(nr_food):
                self.nr_food_collected += 1
                # reward += 50
            reward += 50
            print(f"\n*** Food collected! {self.nr_food_collected}/7 ***")

        # Additional reward from change in nearest green object distance
        reward += 0.01 * distance_delta

        # Reward or penalize based on whether the target is in sight
        if target_in_sight:
            reward += 10
            if action == 0:  # Moving forward
                reward += 5
            elif action in [1, 2]:  # Turning
                reward += 2
            elif action == 3:  # Reversing
                reward -= 5
        else:
            reward -= 5
            if action in [1, 2]:
                reward += 3
            elif action == 0:
                reward -= 5
            elif action == 3:
                reward -= 2

        # Penalize oscillatory behavior
        if oscillatory_action:
            reward -= 10

        # Reward escaping from walls
        if stuck:
            if action == 3:
                reward += 20
            elif action in [1, 2]:
                reward += 5
            elif action == 0:
                reward -= 20

        if action == 3 and not stuck:
            reward -= 1

        return reward

    def _check_done(self):
        if isinstance(self.robot, HardwareRobobo):
            return False
        elif self.get_nr_food_collected() >= 7:
            return True
        # Read current IR sensor readings
        obs = self.get_observation()
        front_sensors = [obs[i] for i in [2, 3, 4, 5, 7]]

        # Check if any sensor indicates a collision
        target_in_sight = self.state[2]
        collision_detected = any(sensor > self.thresholds["collision"] for sensor in front_sensors) and not target_in_sight

        if collision_detected:
            # Increment the delay counter if a collision is detected
            self.collision_delay_counter += 1
        else:
            # Reset the delay counter if no collision is detected
            self.collision_delay_counter = 0
        if self.collision_delay_counter >= self.collision_check_delay:
            print("Collision occured")
        # Confirm wall collision if the delay counter exceeds the threshold
        return self.collision_delay_counter >= self.collision_check_delay

    def get_observation(self):
        ir_sensors = self.robot.read_irs()
        if isinstance(self.robot, HardwareRobobo):
            ir_sensors[0] = 0.0
        image = self.robot.read_image_front()
        green_pixel_count = 0
        nearest_area = 0
        centroid_x = -1
        centroid_y = -1

        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            area_threshold = 500 if isinstance(self.robot, SimulationRobobo) else 200
            features = detect_nearest_green_object(image, area_threshold=area_threshold)
            if isinstance(self.robot, HardwareRobobo):
                features['green_pixel_count'] = features['green_pixel_count']*100 if features['green_pixel_count'] is not None else None
                features['nearest_object_area'] = features['nearest_object_area']*100 if features['nearest_object_area'] is not None else None
                features['distance_to_lower_center'] = features['distance_to_lower_center']*1 if features['distance_to_lower_center'] is not None else None
            self.last_features = features
            green_pixel_count = features['green_pixel_count']
            nearest_area = features['nearest_object_area']
            cx, cy = features['nearest_object_centroid']
            if cx is not None and cy is not None:
                centroid_x = cx
                centroid_y = cy

            if self.current_step % 10 == 0:
                image_path = os.path.join(RESULTS_DIR, f"step_{self.current_step}.jpg")
                # cv2.imwrite(image_path, image)

        extended_observation = ir_sensors + [green_pixel_count, nearest_area, centroid_x, centroid_y]
        return np.array(extended_observation, dtype=np.float32)

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()


def train_model(rob: IRobobo, experiment_number):
    # Create environment
    env = RoboboItemCollectionEnvironment(rob, experiment_number, mode="train")

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
        filename = os.path.join(RESULTS_DIR, "robobo_food_dqn_model.pkl")
        model.save(filename)
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        env.close()


def test_model(rob: IRobobo, experiment_number):
    env = RoboboItemCollectionEnvironment(rob, experiment_number, mode=TEST_MODE, max_steps=TIMESTEPS)
    filename = os.path.join(RESULTS_DIR, "robobo_food_dqn_model.pkl")
    model = DQN.load(filename)

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
    experiment_files = list(RESULTS_DIR.glob(f"{mode}_robot_item_collection_log_*.csv"))
    if experiment_files:
        experiment_files.sort()
        latest_file = experiment_files[-1]
        match = re.search(r"log_(\d+)", latest_file.name)  # Extract the number after "experiment_"
        experiment_number = int(match.group(1)) + 1

    # If it's a simulation robot, start the simulation
    if hasattr(rob, 'play_simulation'):
        rob.play_simulation()
    # else:
    #     rob.set_phone_pan_blocking(0, 50)

    try:
        if mode == "train":
            # Create and train the model
            print("Starting training phase...")
            train_model(rob, experiment_number=experiment_number)
        else:
            for _ in range(N_TESTS):
                print("\nStarting testing phase...")
                test_model(rob, experiment_number=experiment_number)
                experiment_number += 1

    except KeyboardInterrupt:
        print("\nStopping robot...")
    finally:
        # If it's a simulation robot, stop the simulation
        if hasattr(rob, 'stop_simulation'):
            rob.stop_simulation()
