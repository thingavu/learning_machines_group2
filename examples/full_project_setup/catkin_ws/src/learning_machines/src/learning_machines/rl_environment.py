import gym
from gym import spaces
import numpy as np
from robobo_interface import SimulationRobobo

COLLISION_THRESHOLD = 100  # IR sensor threshold for collision

class RoboboRLEnvironment(gym.Env):  # Subclass from gym.Env
    def __init__(self):
        super(RoboboRLEnvironment, self).__init__()
        self.robot = SimulationRobobo()

        # Define action space (3 discrete actions: forward, turn left, turn right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (5 IR sensors with range 0 to 1000)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)

        self.previous_blocks_collected = 0
        self.total_blocks = 7  # Total number of blocks to collect

        self.current_step = 0
        self.max_steps = 100  # Maximum number of steps per episode

        self.prev_sensor_readings = [0, 0, 0, 0, 0]


    def reset(self):
        """
        Reset the environment and return the initial observation.
        """
        self.robot.stop_simulation()
        self.robot.play_simulation()
        self.robot.reset_wheels()

        # Confirm the simulation is running
        if not self.robot.is_running():
            raise RuntimeError("Simulation failed to start after reset.")
        
        # Record the initial position
        self.initial_position = self.robot.get_position()

        # Reset counters
        self.previous_blocks_collected = 0
        self.current_step = 0

        # Get the initial observation
        return self.get_observation()

    def step(self, action):
        """
        Execute an action and return observation, reward, done, and info.
        """
        # Execute the action
        if action == 0:  # Move Forward
            self.robot.move_blocking(100, 100, 500)
        elif action == 1:  # Turn Left
            self.robot.move_blocking(-50, 50, 500)
        elif action == 2:  # Turn Right
            self.robot.move_blocking(50, -50, 500)

        # Get the updated observation
        obs = self.get_observation()

        # Calculate reward and check if the episode is done
        reward = self._calculate_reward(obs, action)
        done = self._check_done(obs)
        self.prev_sensor_readings = obs

        # Update the current step
        self.current_step += 1

        return obs, reward, done, {}


    def _calculate_reward(self, obs, action):
        """
        Reward the robot for avoiding collisions and penalize it for hitting obstacles or walls.
        """
        reward = 0

        avg_curr_sensor_reading = sum(obs) / len(obs)
        avg_prev_sensor_reading = sum(self.prev_sensor_readings) / len(self.prev_sensor_readings)

        # Penalize for moving closer to blocks
        if avg_curr_sensor_reading < avg_prev_sensor_reading:
            reward -= 1
        # Reward for moving away from blocks
        elif avg_curr_sensor_reading > avg_prev_sensor_reading:
            reward += 1

        

        # Penalize collisions (obstacle or wall)
        if any(sensor > COLLISION_THRESHOLD for sensor in obs):  # IR sensor threshold for collision
            reward -= 10
        # Reward for moving forward without collision
        elif action == 0:  # Move Forward
            reward += 3
        # elif all(sensor < COLLISION_THRESHOLD for sensor in obs):
        #     reward += 1  # Reward for moving safely

        return reward


    def _check_done(self, obs):
        """
        End the episode if:
        - The simulation stops.
        - The robot collides with an obstacle or wall.
        - The maximum steps or time limit is reached.
        """

        # End the episode if the simulation stops
        if self.robot.is_stopped():
            print("Simulation stopped. Ending episode.")
            return True
    
        # # End the episode on collision
        # if any(sensor > COLLISION_THRESHOLD for sensor in obs):  # Collision threshold
        #     return True

        # Optional: End the episode after a predefined number of steps
        if self.current_step >= self.max_steps:
            return True

        return False
    
    def get_observation(self):
        """
        Retrieve only the front sensor readings for the observation.
        """
        ir_sensors = self.robot.read_irs()  # Read all IR sensors
        front_sensors = [ir_sensors[i] for i in [2, 3, 4, 5, 7]]  # Focus on front sensors
        return np.array(front_sensors, dtype=np.float32)

    def render(self, mode="human"):
        """Optionally render the environment."""
        pass  # No visualization for now

    def close(self):
        """Clean up resources."""
        self.robot.stop_simulation()
