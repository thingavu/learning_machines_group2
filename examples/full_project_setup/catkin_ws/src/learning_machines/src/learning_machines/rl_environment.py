import gym
from gym import spaces
import numpy as np
from robobo_interface import SimulationRobobo
from data_files import RESULTS_DIR
import os
import csv
import cv2
from .process_images import detect_nearest_green_object
from collections import deque
import atexit

# Define constants for actions
MOVE_DURATION_MS = 300
OSCILLATION_PENALTY = -10
MAX_REWARD = 100  # Example value for normalization
ALTERNATING_TURN_PENALTY = -15  # Penalty for repetitive alternating turns
ALTERNATING_TURN_WINDOW = 4  # Number of recent actions to check for alternation

# Define color thresholds for white detection in HSV
WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 30, 255])

class RoboboRLEnvironment(gym.Env):
    """
    Custom Gym environment for Robobo simulation with enhanced obstacle detection
    based solely on white walls and penalization for repetitive alternating turns.
    """

    def __init__(self, save_images=False, max_steps=100):
        super(RoboboRLEnvironment, self).__init__()
        self.robot = SimulationRobobo()
        self.save_images = save_images
        self.max_steps = max_steps

        self.wheel_speed = {
            "forward": (50, 50),
            "backward": (-50, -50),
            "left": (-30, 30),
            "right": (30, -30)
        }
        self.thresholds = {"collision": 200, "danger": 80, "safe": 40}

        # Assuming robot.read_irs() returns 8 sensors
        self.num_ir_sensors = 8
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([-1]*12),
            high=np.array([1e6]*12),
            dtype=np.float32
        )

        self.current_step = 0
        self.prev_food = 0
        self.prev_green_pixel_count = 0

        log_path = os.path.join(RESULTS_DIR, 'robot_item_collection_log_ver2.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        header = [
            'step',
            'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8',
            'green_pixel_count', 'nearest_area', 'centroid_x', 'centroid_y',
            'action', 'reward', 'distance_to_lower_center'
        ]
        self.csv_writer.writerow(header)

        self.collision_delay_counter = 0
        self.collision_check_delay = 5
        self.action_history = deque(maxlen=ALTERNATING_TURN_WINDOW)

        # Initialize last_features with default values
        self.last_features = {
            'green_pixel_count': 0,
            'nearest_object_area': 0,
            'nearest_object_centroid': (None, None),
            'distance_to_lower_center': float('inf')
        }

        atexit.register(self.close)  # Ensure resources are cleaned up

    def reset(self):
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()
        if hasattr(self.robot, 'play_simulation'):
            self.robot.play_simulation()

        if hasattr(self.robot, 'set_phone_tilt_blocking'):
            self.robot.set_phone_tilt_blocking(tilt_position=109, tilt_speed=100)

        self.current_step = 0
        self.prev_food = self.robot.get_nr_food_collected()
        self.prev_green_pixel_count = 0

        obs = self.get_observation()
        initial_distance = self.last_features['distance_to_lower_center']
        self.prev_distance = initial_distance if isinstance(initial_distance, (int, float)) else float('inf')
        self.prev_area = np.log1p(self.last_features.get('nearest_object_area', 0))
        self.prev_potential = -self.prev_distance  # For potential-based reward shaping
        self.prev_target_in_sight = self.prev_area > 0
        return obs

    def step(self, action):
        if action == 0:
            self.robot.move_blocking(*self.wheel_speed["forward"], MOVE_DURATION_MS)
        elif action == 1:
            self.robot.move_blocking(*self.wheel_speed["left"], MOVE_DURATION_MS)
        elif action == 2:
            self.robot.move_blocking(*self.wheel_speed["right"], MOVE_DURATION_MS)
        elif action == 3:
            self.robot.move_blocking(*self.wheel_speed["backward"], MOVE_DURATION_MS)
        else:
            raise ValueError(f"Invalid action: {action}")

        obs = self.get_observation()
        reward = self._calculate_reward(obs, action)
        done = self._check_done()

        distance = self.last_features.get('distance_to_lower_center', float('inf'))
        log_row = (
            [self.current_step] +
            obs.tolist() +
            [action, reward, distance]
        )
        if self.csv_writer:
            self.csv_writer.writerow(log_row)

        self.current_step += 1
        return obs, reward, done, {}

    def _calculate_reward(self, obs, action):
        reward = 0

        # Reward from food collection
        current_food = self.robot.get_nr_food_collected()
        if len(self.action_history) > 0:
            if self.action_history[-1] != 3:
                reward += 50 * (current_food - self.prev_food) if (self.prev_target_in_sight) else 10

        self.prev_food = current_food

        # Reward from distance change
        current_area = np.log1p(self.last_features.get('nearest_object_area', 0))
        current_distance = self.last_features.get('distance_to_lower_center', float('inf'))
        current_distance = current_distance if current_distance else float('inf')
        distance_delta = self.prev_distance - current_distance

        reward += 0.01 * distance_delta
        self.prev_distance = current_distance if isinstance(current_distance, (int, float)) else self.prev_distance

        # Environment context based on white wall detection
        target_in_sight = current_area > 0
        ir_sensors = self.get_observation()[1:9]
        stuck = all(sensor > self.thresholds["collision"] for sensor in ir_sensors)

        # Reward or penalize based on target visibility
        if target_in_sight:
            reward += 20
            if action == 0:
                reward += 10
            elif action in [1, 2]:
                reward += 3
            elif action == 3:
                reward -= 5
        else:
            reward -= 5
            if action in [0, 1, 2]:
                reward += 2
            elif action == 3:
                reward -= 2

        # Track action history for oscillation detection
        self.action_history.append(action)
        # Penalize oscillatory behavior: repeated alternating turns
        if self._is_repeated_alternating_turns():
            reward -= 10  # Apply negative penalty

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

        # Potential-based reward shaping
        potential = -current_distance if isinstance(current_distance, (int, float)) else -self.prev_distance
        shaped_reward = reward + (potential - self.prev_potential)
        self.prev_potential = potential

        # Normalize reward
        shaped_reward = np.clip(shaped_reward / MAX_REWARD, -1, 1)

        print(f"REWARD: {shaped_reward}, Target in sight: {target_in_sight}, Distance: {current_distance}")
        return shaped_reward

    def _is_obstacle_detected(self):
        """
        Determine if a white wall is detected as an obstacle based on image processing.
        """
        # Detect white areas in the image to identify white walls
        image = self.robot.read_image_front()
        if image is not None:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
            white_pixel_count = cv2.countNonZero(mask)
            print(white_pixel_count)
            # Define a threshold for white pixel count to consider as obstacle
            WHITE_OBSTACLE_THRESHOLD = 4000
            if white_pixel_count > WHITE_OBSTACLE_THRESHOLD:
                return True
        return False

    def _is_repeated_alternating_turns(self):
        """
        Check if the action history contains repeated alternating turns (Left ↔ Right).
        """
        if len(self.action_history) < ALTERNATING_TURN_WINDOW:
            return False

        # Extract the last ALTERNATING_TURN_WINDOW actions
        recent_actions = list(self.action_history)[-ALTERNATING_TURN_WINDOW:]

        # Define the alternating pattern: Left, Right, Left, Right or Right, Left, Right, Left also back and front
        pattern1 = [1, 2] * (ALTERNATING_TURN_WINDOW // 2)
        pattern2 = [2, 1] * (ALTERNATING_TURN_WINDOW // 2)
        pattern3 = [0, 3] * (ALTERNATING_TURN_WINDOW // 2)
        pattern4 = [3, 0] * (ALTERNATING_TURN_WINDOW // 2)

        if recent_actions == pattern1 or recent_actions == pattern2 or recent_actions == pattern3 or recent_actions == pattern4:
            return True
        return False

    def _check_done(self):
        ir_sensors = self.get_observation()
        ir_sensors = [ir_sensors[x] for x in range(len(ir_sensors)) if x in [2, 3, 4, 5, 7]]

        # white_wall_detected = self._is_obstacle_detected()
        collision_detected = any(sensor > self.thresholds['collision'] for sensor in ir_sensors)

        current_food = self.robot.get_nr_food_collected()
        green_block_collision = current_food > self.prev_food
        if green_block_collision:
            self.prev_food = current_food
            self.collision_delay_counter = 0
            return False  # Do not terminate

        if current_food >= 7:
            return True  # Terminate

        if collision_detected and not green_block_collision:
            self.collision_delay_counter += 1
        else:
            self.collision_delay_counter = 0

        if self.collision_delay_counter >= self.collision_check_delay:
            return True  # Terminate

        if self.current_step >= self.max_steps:
            return True  # Terminate

        return False

    def get_observation(self):
        ir_sensors = self.robot.read_irs()
        if ir_sensors is None or len(ir_sensors) < self.num_ir_sensors:
            ir_sensors = [0.0] * self.num_ir_sensors  # Default values or handle accordingly

        image = self.robot.read_image_front()
        green_pixel_count = 0
        nearest_area = 0
        centroid_x = -1
        centroid_y = -1

        if image is not None:
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            features = detect_nearest_green_object(image)
            self.last_features = features
            green_pixel_count = features.get('green_pixel_count', 0)
            nearest_area = features.get('nearest_object_area', 0)
            centroid = features.get('nearest_object_centroid', (None, None))
            cx, cy = centroid
            centroid_x = cx if cx is not None else -1
            centroid_y = cy if cy is not None else -1

            # Ensure 'distance_to_lower_center' is a float
            distance = features.get('distance_to_lower_center')
            if distance is None:
                distance = 1000
            self.last_features['distance_to_lower_center'] = distance

            if self.save_images and self.current_step % 10 == 0:
                image_path = os.path.join(RESULTS_DIR, f"step_{self.current_step}.jpg")
                cv2.imwrite(image_path, image)
        else:
            # Handle missing image data
            self.last_features = {
                'green_pixel_count': 0,
                'nearest_object_area': 0,
                'nearest_object_centroid': (None, None),
                'distance_to_lower_center': float('inf')
            }

        extended_observation = ir_sensors + [green_pixel_count, nearest_area, centroid_x, centroid_y]
        return np.array(extended_observation, dtype=np.float32)

    def close(self):
        if self.log_file:
            # Flush remaining logs
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()