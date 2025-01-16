import gym
from gym import spaces
import numpy as np
from robobo_interface import SimulationRobobo

COLLISION_THRESHOLD = 120

class RoboboRLEnvironment(gym.Env):  # Subclass from gym.Env
    def __init__(self):
        super(RoboboRLEnvironment, self).__init__()
        self.robot = SimulationRobobo()

        # Define action space (3 discrete actions: forward, turn left, turn right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (5 IR sensors)
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(5,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = 100
        self.prev_sensor_readings = np.zeros(5)
        self.prev_action = None

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
        # Execute action
        if action == 0:  # Move Forward
            self.robot.move_blocking(40, 40, 300)
        elif action == 1:  # Turn Left
            self.robot.move_blocking(-20, 20, 300)
        else:  # Turn Right
            self.robot.move_blocking(20, -20, 300)

        # Get new observation
        obs = self.get_observation()

        # Calculate reward
        reward = self._calculate_reward(obs, action)

        # Check if episode is done
        done = self._check_done(obs)

        # Update previous readings
        self.prev_sensor_readings = obs
        self.current_step += 1

        return obs, reward, done, {}

    def _calculate_reward(self, obs, action):
        """Calculate the reward"""
        reward = 0

        # Define thresholds
        DANGER_THRESHOLD = 80  # Getting very close to obstacle
        SAFE_THRESHOLD = 40  # Good distance for navigation

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
            front_obstacle = obs[1] > DANGER_THRESHOLD or obs[2] > DANGER_THRESHOLD or obs[3] > DANGER_THRESHOLD

            if front_obstacle:
                # Get readings from sensors (LL, L, C, R, RR)
                left_side_danger = max(obs[0], obs[1])  # LL and L sensors
                right_side_danger = max(obs[3], obs[4])  # R and RR sensors

                # Reward turning if it directs the robot away from the greater danger
                if action == 1 and right_side_danger > left_side_danger:
                    reward += 2
                elif action == 2 and left_side_danger > right_side_danger:
                    reward += 2
                else:
                    reward -= 0.5  # Small penalty if turning in a suboptimal direction
            else:
                # Penalize turning when no immediate obstacle is detected ahead
                reward -= 1

        # Reward for maintaining good exploration distance (safe distance from all obstacles)
        if all(sensor < SAFE_THRESHOLD for sensor in obs):
            reward += 2

        # Encourage movement by penalizing staying still (NOT SURE ABOUT THS ONE)
        if np.array_equal(self.prev_sensor_readings, obs):
            reward -= 1

        return reward

    def _check_done(self, obs):
        """Check if episode should end"""
        # End if collision
        if any(sensor > COLLISION_THRESHOLD for sensor in obs):
            return True

        # End if max steps reached
        # if self.current_step >= self.max_steps:
        #     return True

        # End if simulation stopped
        if hasattr(self.robot, 'is_stopped') and self.robot.is_stopped():
            return True

        return False

    def get_observation(self):
        """Get current observation"""
        ir_sensors = self.robot.read_irs()
        # Use front sensors
        front_sensors = [ir_sensors[i] for i in [2, 3, 4, 5, 7]] # using 5 front sensors
        return np.array(front_sensors, dtype=np.float32)

    def close(self):
        """Clean up"""
        if hasattr(self.robot, 'is_running') and self.robot.is_running():
            self.robot.stop_simulation()