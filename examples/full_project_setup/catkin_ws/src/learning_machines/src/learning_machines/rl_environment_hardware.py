import gym
from gym import spaces
import numpy as np
import os
import csv
import cv2
from .process_images import detect_nearest_object
from robobo_interface import HardwareRobobo

class RoboboRLHardwareEnvironment(gym.Env):

    def __init__(self):
        super().__init__()

        # Initialize robot and action/observation spaces
        super(RoboboRLHardwareEnvironment, self).__init__()
        self.robot = HardwareRobobo(camera=True)
        self.action_space = spaces.Discrete(3)  # Forward, Left, Right

        # 8 IR sensors + 2 distances + alignment + 2 areas + 2 binary flags = 15 total
        self.observation_space = spaces.Box(
            low=np.array([-1] * 15),
            high=np.array([1] * 15),
            dtype=np.float32
        )

        # Core parameters
        self.max_steps = 200
        self.phase = 1
        self.current_step = 0
        self.carrying_food = False

        # Action mappings
        self.actions = {
            0: (60, 60),  # Forward
            1: (-10, 10),  # Left
            2: (20, -20),  # Right
            3: (-50, -50) # Backward
        }

        # Thresholds
        self.thresholds = {
            'collision': 500,
            'mounting_distance': 225,
            'alignment': 0.8
        }

        print("Environment initialized with observation space shape:", self.observation_space.shape)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        log_dir = os.path.join(os.path.expanduser("~"), "robobo_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, 'training_log.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['step', 'phase', 'action', 'reward', 'red_distance',
                                'green_distance', 'alignment', 'collision', 'done_reason'])

    def reset(self):
        print("\n--- Starting New Episode ---")
        # if self.robot.is_running():
        #     self.robot.stop_simulation()
        # self.robot.play_simulation()
        # self.robot.set_phone_tilt_blocking(250, 100)

        self.carrying_food = False
        self.phase = 1
        self.current_step = 0
        obs = self._get_observation()
        print(f"Reset complete. Initial observation shape: {obs.shape}")
        return obs

    def step(self, action):
        print(f"\nStep {self.current_step}, Phase {self.phase}, Action {action}")

        # Execute action
        left_speed, right_speed = self.actions[int(action)]
        self.robot.move_blocking(left_speed, right_speed, 300)

        # Get new state and features
        obs, features = self._get_observation_and_features()
        reward = self._calculate_reward(obs, action, features)
        done, info = self._check_termination(obs)

        # Log step
        self._log_step(obs, action, reward, info.get('done_reason', ''))

        print(f"Reward: {reward:.2f}")
        print(f"Done: {done}, Reason: {info.get('done_reason', 'ongoing')}")

        self.current_step += 1
        return obs, reward, done, info

    def _get_observation_and_features(self):
        ir_readings = self.robot.read_irs()
        image = self.robot.read_image_front()

        # First handle None case, then always rotate
        if image is None:
            image = np.zeros((512, 512, 3))
        image = cv2.rotate(image, cv2.ROTATE_180)

        red_features = detect_nearest_object(image, "red")
        green_features = detect_nearest_object(image, "green")

        # carrying_food = (red_features['distance_to_lower_center'] is not None and
        #                  red_features['distance_to_lower_center'] < self.thresholds['mounting_distance'])
        if red_features['distance_to_lower_center'] and self.carrying_food == False:
            print(red_features['distance_to_lower_center'])
            carrying_food = (any(x > self.thresholds['collision'] / 1000 for x in ir_readings[:8]) \
                            and red_features['distance_to_lower_center'] < self.thresholds['mounting_distance'])
            self.carrying_food = True if carrying_food else False
            if self.carrying_food:
                print("Carrying food because:")
                print(red_features['distance_to_lower_center'])
                print(self.thresholds['mounting_distance'])
        
        obs = np.array([
            *[x / 1000 for x in ir_readings[:8]],  # Normalized IR readings
            red_features['distance_to_lower_center'] / 1000 if red_features[
                                                                   'distance_to_lower_center'] is not None else -1,
            green_features['distance_to_lower_center'] / 1000 if green_features[
                                                                     'distance_to_lower_center'] is not None else -1,
            self._calculate_alignment(red_features if self.phase == 1 else green_features),
            (red_features['nearest_object_area'] / (512 * 512) * 50) * 10 if red_features['nearest_object_area'] is not None else 0,
            (green_features['nearest_object_area'] / (512 * 512) * 50) * 10 if green_features[
                                                                       'nearest_object_area'] is not None else 0,
            0,
            float(self.carrying_food)
        ], dtype=np.float32)

        print(f"IR readings: {obs[:8]}")
        print(f"Red distance: {obs[8]:.3f}, Green distance: {obs[9]:.3f}")
        print(f"Alignment: {obs[10]:.3f}")
        print(f"Object areas: Red={obs[11]:.5f}, Green={obs[12]:.5f}")
        print(f"Has food: {bool(obs[13])}, Carrying food: {bool(obs[14])}")

        features = {
            'red': red_features,
            'green': green_features
        }

        return obs, features

    def _get_observation(self):
        obs, _ = self._get_observation_and_features()
        return obs

    def _calculate_reward(self, obs, action, features):
        # Base step penalty
        reward = -0.1

        # Early return on collision
        if self._detect_collision(obs):
            print("Collision detected!")
            if action == 3:
                return 5
            else:
                return -20.0

        # Get target distance based on current phase
        target_distance = obs[8] if self.phase == 1 else obs[9]
        current_features = features['red'] if self.phase == 1 else features['green']

        if target_distance >= 0:  # Object detected
            if current_features['nearest_object_centroid'][0] is not None:
                # Calculate rewards
                alignment_reward = self.calculate_linear_alignment_reward(current_features['nearest_object_centroid'][0])
                distance_reward = (1 - target_distance) * 5
                reward += distance_reward + alignment_reward

                print(f"Distance reward: {distance_reward:.2f}, "
                      f"Alignment reward: {alignment_reward:.2f}")

            # Phase completion rewards
            if self.carrying_food:
                print("Phase 1 completed! Moving to phase 2")
                reward += 50.0
                self.phase = 2
            elif self.phase == 2 and obs[13]:  # Has food
                print("Task completed successfully!")
                reward += 100.0

        else:  # no target detected:
            # Store last action if not already tracking
            if not hasattr(self, 'last_actions'):
                self.last_actions = []

            # Keep track of last 3 actions
            self.last_actions.append(action)
            if len(self.last_actions) > 3:
                self.last_actions.pop(0)

            # Encourage rotation (actions 1 and 2) over forward movement (action 0)
            if action == 0:  # Forward movement when no target
                reward -= 0.5
                print("Penalty for moving forward without target: -0.5")

            # Penalize oscillating behavior (left-right-left or right-left-right)
            if len(self.last_actions) == 3:
                if (self.last_actions == [1, 2, 1] or
                        self.last_actions == [2, 1, 2]):
                    reward -= 1.0
                    print("Penalty for oscillating behavior: -1.0")

            # Small reward for consistent rotation direction
            if len(self.last_actions) >= 2:
                if self.last_actions[-1] == self.last_actions[-2] and action in [1, 2]:
                    reward += 1
                    print("Reward for consistent rotation: +0.2")

            print(f"No target detected. Action history: {self.last_actions}")

        return reward

    def calculate_linear_alignment_reward(self, x_position, max_reward=5, center_x=256, width=256):
        if x_position is None:
            return 0.0
        return max(max_reward * (1 - abs(x_position - center_x) / width), 0)

    def _calculate_alignment(self, features):
        return self.calculate_linear_alignment_reward(
            features['nearest_object_centroid'][0],
            max_reward=1.0  # Using 1.0 for normalized alignment value
        )

    def _detect_collision(self, obs):
        # at least 3 IR sensors detect collision and at least one of the front sensors is triggered
        if len([x for x in obs[:8] if x > self.thresholds['collision'] / 1000]) >= 3 and \
            (obs[8]<0 or obs[9]<0):
            return True

    def _check_termination(self, obs):
        if self._detect_collision(obs):
            return True, {'done_reason': 'collision'}
        if self.current_step >= self.max_steps:
            return True, {'done_reason': 'max_steps'}
        # if self.phase == 2 and obs[13]:
        #     return True, {'done_reason': 'success'}
        return False, {}

    def _log_step(self, obs, action, reward, done_reason):
        self.csv_writer.writerow([self.current_step, self.phase, action, reward,
                                obs[8], obs[9], obs[10], self._detect_collision(obs),
                                done_reason])

    def close(self):
        print("Closing environment and cleaning up resources")
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if self.robot.is_running():
            self.robot.stop_simulation()