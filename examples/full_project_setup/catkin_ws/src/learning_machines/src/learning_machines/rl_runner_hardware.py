from stable_baselines3 import DQN
from .rl_environment_hardware import RoboboRLHardwareEnvironment
from data_files import RESULTS_DIR

def test_hardware():
    env = RoboboRLHardwareEnvironment()
    model = DQN.load(f"{RESULTS_DIR}/models/task3_ver2")
    print("Testing the model on hardware...")
    obs = env._get_observation()
    #it should run infinitely until the user stops it
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _ = env.step(action)