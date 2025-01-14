from stable_baselines3 import DQN
from .rl_environment import RoboboRLEnvironment
from data_files import RESULTS_DIR

def train_rl_model():
    env = RoboboRLEnvironment()
    model = DQN(
        "MlpPolicy", 
        env, verbose=1, 
        learning_rate=1e-3, 
        buffer_size=10000, 
        batch_size=32, 
        exploration_initial_eps = 0.9,
        exploration_fraction=1, 
        exploration_final_eps=0.05
    )
    print("Training the model...")
    model.learn(total_timesteps=100)
    model.save(f"{RESULTS_DIR}/robobo_dqn_model")
    print("Model saved to results/robobo_dqn_model.zip")
    env.close()

def test_rl_model():
    env = RoboboRLEnvironment()
    model = DQN.load("models/robobo_dqn_model")
    print("Testing the model...")
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if done:
            print("Collision detected. Resetting environment...")
            obs = env.reset()
    env.close()
