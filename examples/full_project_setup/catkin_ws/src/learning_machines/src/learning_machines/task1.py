# wall /obstacle avoiding robot. drive around as much as possible and avoid obstacles
import os

import gym
from gym import spaces
import numpy as np
from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo
from stable_baselines3 import DQN
from data_files import RESULTS_DIR


class RoboboRLEnvironment(gym.Env):
    def __init__(self, robot):
        super(RoboboRLEnvironment, self).__init__()
        self.robot = robot

        # Check mode and set parameters accordingly
        if isinstance(self.robot, SimulationRobobo):
            self.wheel_speed = {
                "forward": (40, 40),
                "left": (-20, 20),
                "right": (20, -20),
            }
            self.thresholds = {
                "collision": 120,
                "danger": 80,
                "safe": 40,
            }
        elif isinstance(self.robot, HardwareRobobo):
            self.wheel_speed = {
                "forward": (40, 40),
                "left": (-20, 20),
                "right": (20, -20),
            }
            self.thresholds = {
                "collision": 100,
                "danger": 70,
                "safe": 30,
            }
        else:
            raise ValueError("Unknown robot type. Must be SimulationRobobo or HardwareRobobo.")

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
        if hasattr(self.robot, 'is_running') and self.robot.is_running() and isinstance(self.robot, SimulationRobobo):
            self.robot.stop_simulation()

        if hasattr(self.robot, 'play_simulation') and isinstance(self.robot, SimulationRobobo):
            self.robot.play_simulation()

        self.current_step = 0

        # Get initial observation
        obs = self.get_observation()
        self.prev_sensor_readings = obs
        return obs

    def step(self, action):
        """Execute action and return new state"""
        # Execute action using dynamic wheel speeds
        if action == 0:  # Move Forward
            self.robot.move_blocking(*self.wheel_speed["forward"], 300)
        elif action == 1:  # Turn Left
            self.robot.move_blocking(*self.wheel_speed["left"], 300)
        else:  # Turn Right
            self.robot.move_blocking(*self.wheel_speed["right"], 300)

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

        # Dynamic thresholds
        collision_threshold = self.thresholds["collision"]
        danger_threshold = self.thresholds["danger"]
        safe_threshold = self.thresholds["safe"]

        # Heavy penalty for potential collisions
        if any(sensor > collision_threshold for sensor in obs):
            reward -= 10
            return reward  # Immediate return for collision

        # Penalty for getting too close to obstacles
        if any(sensor > danger_threshold for sensor in obs):
            reward -= 5

        # Reward for moving forward while maintaining safe distance
        if action == 0:  # Forward action
            # Check if path ahead is relatively clear
            if all(sensor < safe_threshold for sensor in obs):
                reward += 5  # Good forward movement
            else:
                reward += 0.5  # Smaller reward if moving forward in tighter spaces

        # Handling turning actions with conditional rewards
        elif action in [1, 2]:  # Turning actions
            front_obstacle = obs[1] > danger_threshold or obs[2] > danger_threshold or obs[3] > danger_threshold
            if front_obstacle:
                left_side_danger = max(obs[0], obs[1])  # LL and L sensors
                right_side_danger = max(obs[3], obs[4])  # R and RR sensors
                if action == 1 and right_side_danger > left_side_danger:
                    reward += 2
                elif action == 2 and left_side_danger > right_side_danger:
                    reward += 2
                else:
                    reward -= 0.5
            else:
                reward -= 1

        if all(sensor < safe_threshold for sensor in obs):
            reward += 2

        if np.array_equal(self.prev_sensor_readings, obs):
            reward -= 1

        return reward

    def _check_done(self, obs):
        """Check if episode should end"""
        collision_threshold = self.thresholds["collision"]

        if any(sensor > collision_threshold for sensor in obs):
            return True

        if self.current_step >= self.max_steps:
            return True

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
        if hasattr(self.robot, 'is_running') and self.robot.is_running() and isinstance(self.robot, SimulationRobobo):
            self.robot.stop_simulation()

def train_model(rob: IRobobo):
    # Create environment
    env = RoboboRLEnvironment(rob)

    # Create the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.01,  # Much lower learning rate for stability
        buffer_size=10000,
        learning_starts=200,
        batch_size=32,  # Smaller batch size
        exploration_initial_eps=0.6,  # Higher initial exploration
        exploration_final_eps=0.1,
        exploration_fraction=0.8,  # Slower exploration decay
        target_update_interval=250  # More frequent target updates
    )

    # Train the model
    try:
        print("Starting training...")
        #model.learn(total_timesteps=10000)
        model.learn(total_timesteps=50)
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


def test_model(rob: IRobobo):
    # Create environment
    env = RoboboRLEnvironment(rob)

    # Load the trained model
    model = DQN.load(f"results/robobo_dqn_model")
    print("Model loaded successfully!")

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

# def test_rl_model():
#     env = RoboboRLEnvironment()
#     model = DQN.load(f"{RESULTS_DIR}/robobo_dqn_model")
#     print("Testing the model...")
#     obs = env.reset()
#     for _ in range(1000):
#         action, _ = model.predict(obs, deterministic=False)
#         obs, reward, done, _ = env.step(action)
#         if done:
#             print("Collision detected. Resetting environment...")
#             obs = env.reset()
#     env.close()


def run_all_actions(rob: IRobobo, test_mode=False):
    # If it's a simulation robot, start the simulation
    if hasattr(rob, 'play_simulation'):
        rob.play_simulation()

    try:
        # Create and train the model
        if not test_mode:
            print("Starting training phase...")
            train_model(rob)

        if test_mode:
            print("\nStarting testing phase...")
            test_model(rob)

    except KeyboardInterrupt:
        print("\nStopping robot...")
    finally:
        # If it's a simulation robot, stop the simulation
        if hasattr(rob, 'stop_simulation'):
            rob.stop_simulation()