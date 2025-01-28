import gym
from gym import spaces
import numpy as np
from robobo_interface import SimulationRobobo
from data_files import RESULTS_DIR
import os
import csv
import cv2

# Import nearest green object detection from our feature module
from .process_images import detect_nearest_green_object

COLLISION_THRESHOLD = 400

class RoboboRLEnvironment(gym.Env):  # Subclass from gym.Env
    def __init__(self):
        super(RoboboRLEnvironment, self).__init__()
        self.robot = SimulationRobobo()

        self.wheel_speed = {
                "forward": (50, 50),
                "backward": (-50, -50),
                "left": (-30, 30),
                "right": (30, -30)
            }
        self.thresholds = {"collision": 120, "danger": 80, "safe": 40}

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1e6, shape=(12,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = 100
        self.prev_food = 0
        self.prev_green_pixel_count = 0

        log_path = os.path.join(RESULTS_DIR, 'robot_item_collection_log.csv')
        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        header = [
            'step',
            'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5',
            'green_pixel_count', 'nearest_area', 'centroid_x', 'centroid_y',
            'action', 'reward', 'distance_to_lower_center'
        ]
        self.csv_writer.writerow(header)

        self.collision_delay_counter = 0  # Counter to track persistent collisions
        self.collision_check_delay = 3   # Number of steps to confirm a wall collision

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
        # Initialize distance tracking after first observation
        initial_distance = self.last_features['distance_to_lower_center']
        self.prev_distance = initial_distance if initial_distance is not None else float('inf')
        self.prev_area = np.log1p(self.last_features['nearest_object_area'] or 0)
        return obs

    def step(self, action):
        if action == 0:
            self.robot.move_blocking(*self.wheel_speed["forward"], 300)
        elif action == 1:
            self.robot.move_blocking(*self.wheel_speed["left"], 300)
        elif action == 2:
            self.robot.move_blocking(*self.wheel_speed["right"], 300)
        elif action == 3:
            self.robot.move_blocking(*self.wheel_speed["backward"], 300)

        obs = self.get_observation()
        reward = self._calculate_reward(obs, action)
        done = self._check_done()

        log_row = (
            [self.current_step] +
            obs.tolist() +
            [action, reward, self.last_features['distance_to_lower_center']]
        )
        if self.csv_writer:
            self.csv_writer.writerow(log_row)

        self.current_step += 1
        return obs, reward, done, {}
    
    def _calculate_reward(self, obs, action):
        reward = 0

        # Ensure action is an integer
        action = int(action)

        # Reward from food collection
        current_food = self.robot.get_nr_food_collected()
        reward += 50 * (current_food - self.prev_food)
        self.prev_food = current_food

        # Additional reward from change in nearest green object area
        current_area = np.log1p(self.last_features['nearest_object_area'] or 0)
        current_distance = self.last_features['distance_to_lower_center']
        current_distance = current_distance if current_distance is not None else float('inf')
        distance_delta = self.prev_distance - current_distance

        if not np.isfinite(distance_delta):
            distance_delta = 0
        reward += 0.01 * distance_delta
        self.prev_distance = current_distance

        # Determine environment context
        target_in_sight = current_area > 0
        stuck = all(sensor > self.thresholds["collision"] for sensor in obs[:5])

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
        if hasattr(self, 'last_action') and action != self.last_action:
            if (self.last_action, action) in [(1, 2), (2, 1)]:  # Oscillating between left and right
                reward -= 10
        self.last_action = action

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

        print("REWARD:", reward, "Target in sight:", target_in_sight, "Distance to lower center:", current_distance)
        return reward
    
    def _check_done(self):
        # Read current IR sensor readings
        ir_sensors = self.get_observation()[:5]
        
        # Check if any sensor indicates a collision
        collision_detected = any(sensor > COLLISION_THRESHOLD for sensor in ir_sensors)

        # Check if a green block was collected
        current_food = self.robot.get_nr_food_collected()
        green_block_collision = current_food > self.prev_food
        if green_block_collision:
            self.prev_food = current_food  # Update food count
            self.collision_delay_counter = 0  # Reset the counter on green block collision
            return False  # Do not reset for green block collisions

        # Check if collected food equals 7
        if current_food >= 7:
            return True  # Reset simulation when 7 or more green blocks are collected

        if collision_detected:
            # Increment the delay counter if a collision is detected
            self.collision_delay_counter += 1
        else:
            # Reset the delay counter if no collision is detected
            self.collision_delay_counter = 0

        # Confirm wall collision if the delay counter exceeds the threshold
        return self.collision_delay_counter >= self.collision_check_delay

    def get_observation(self):
        ir_sensors = self.robot.read_irs()
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
            green_pixel_count = features['green_pixel_count']
            nearest_area = features['nearest_object_area']
            cx, cy = features['nearest_object_centroid']
            if cx is not None and cy is not None:
                centroid_x = cx
                centroid_y = cy

            if self.current_step % 10 == 0:
                image_path = os.path.join(RESULTS_DIR, f"step_{self.current_step}.jpg")
                # cv2.imwrite(image_path, image)

        front_sensors = [ir_sensors[i] for i in [2, 3, 4, 5, 7]]
        # extended_observation = front_sensors + [green_pixel_count, nearest_area, centroid_x, centroid_y]
        extended_observation = ir_sensors + [green_pixel_count, nearest_area, centroid_x, centroid_y]
        return np.array(extended_observation, dtype=np.float32)

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()



# TASK 1

# COLLISION_THRESHOLD = 120

# class RoboboRLEnvironment(gym.Env):  # Subclass from gym.Env
#     def __init__(self):
#         super(RoboboRLEnvironment, self).__init__()
#         self.robot = SimulationRobobo()

#         # Define action space (3 discrete actions: forward, turn left, turn right)
#         self.action_space = spaces.Discrete(3)

#         # Define observation space (5 IR sensors)
#         self.observation_space = spaces.Box(
#             low=0, high=1000, shape=(5,), dtype=np.float32
#         )

#         self.current_step = 0
#         self.max_steps = 100
#         self.prev_sensor_readings = np.zeros(5)
#         self.prev_action = None

#     def reset(self):
#         """Reset the environment"""
#         if hasattr(self.robot, 'is_running') and self.robot.is_running():
#             self.robot.stop_simulation()

#         if hasattr(self.robot, 'play_simulation'):
#             self.robot.play_simulation()

#         self.current_step = 0

#         # Get initial observation
#         obs = self.get_observation()
#         self.prev_sensor_readings = obs
#         return obs

#     def step(self, action):
#         """Execute action and return new state"""
#         # Execute action
#         if action == 0:  # Move Forward
#             self.robot.move_blocking(40, 40, 300)
#         elif action == 1:  # Turn Left
#             self.robot.move_blocking(-20, 20, 300)
#         else:  # Turn Right
#             self.robot.move_blocking(20, -20, 300)

#         # Get new observation
#         obs = self.get_observation()

#         # Calculate reward
#         reward = self._calculate_reward(obs, action)

#         # Check if episode is done
#         done = self._check_done(obs)

#         # Update previous readings
#         self.prev_sensor_readings = obs
#         self.current_step += 1

#         return obs, reward, done, {}

#     def _calculate_reward(self, obs, action):
#         """Calculate the reward"""
#         reward = 0

#         # Define thresholds
#         DANGER_THRESHOLD = 80  # Getting very close to obstacle
#         SAFE_THRESHOLD = 40  # Good distance for navigation

#         # Heavy penalty for potential collisions (readings > 100)
#         if any(sensor > 100 for sensor in obs):
#             reward -= 10
#             return reward  # Immediate return for collision

#         # Penalty for getting too close to obstacles
#         if any(sensor > DANGER_THRESHOLD for sensor in obs):
#             reward -= 5

#         # Reward for moving forward while maintaining safe distance
#         if action == 0:  # Forward action
#             # Check if path ahead is relatively clear
#             if all(sensor < SAFE_THRESHOLD for sensor in obs):
#                 reward += 5  # Good forward movement
#             else:
#                 reward += 0.5  # Smaller reward if moving forward in tighter spaces

#         # Handling turning actions with conditional rewards
#         elif action in [1, 2]:  # Turning actions
#             # Check if there's an obstacle ahead (center sensors likely indicating front view)
#             front_obstacle = obs[1] > DANGER_THRESHOLD or obs[2] > DANGER_THRESHOLD or obs[3] > DANGER_THRESHOLD

#             if front_obstacle:
#                 # Get readings from sensors (LL, L, C, R, RR)
#                 left_side_danger = max(obs[0], obs[1])  # LL and L sensors
#                 right_side_danger = max(obs[3], obs[4])  # R and RR sensors

#                 # Reward turning if it directs the robot away from the greater danger
#                 if action == 1 and right_side_danger > left_side_danger:
#                     reward += 2
#                 elif action == 2 and left_side_danger > right_side_danger:
#                     reward += 2
#                 else:
#                     reward -= 0.5  # Small penalty if turning in a suboptimal direction
#             else:
#                 # Penalize turning when no immediate obstacle is detected ahead
#                 reward -= 1

#         # Reward for maintaining good exploration distance (safe distance from all obstacles)
#         if all(sensor < SAFE_THRESHOLD for sensor in obs):
#             reward += 2

#         # Encourage movement by penalizing staying still (NOT SURE ABOUT THS ONE)
#         if np.array_equal(self.prev_sensor_readings, obs):
#             reward -= 1

#         return reward

#     def _check_done(self, obs):
#         """Check if episode should end"""
#         # End if collision
#         if any(sensor > COLLISION_THRESHOLD for sensor in obs):
#             return True

#         # End if max steps reached
#         # if self.current_step >= self.max_steps:
#         #     return True

#         # End if simulation stopped
#         if hasattr(self.robot, 'is_stopped') and self.robot.is_stopped():
#             return True

#         return False

#     def get_observation(self):
#         """Get current observation"""
#         ir_sensors = self.robot.read_irs()
#         # Use front sensors
#         front_sensors = [ir_sensors[i] for i in [2, 3, 4, 5, 7]] # using 5 front sensors
#         return np.array(front_sensors, dtype=np.float32)

#     def close(self):
#         """Clean up"""
#         if hasattr(self.robot, 'is_running') and self.robot.is_running():
#             self.robot.stop_simulation()

