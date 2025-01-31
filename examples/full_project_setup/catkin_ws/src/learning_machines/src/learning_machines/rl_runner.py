from stable_baselines3 import DQN
from .rl_environment import RoboboRLEnvironment
from data_files import RESULTS_DIR

def train_rl_model():
    env = RoboboRLEnvironment()
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.01,
        buffer_size=10000,
        learning_starts=200,
        batch_size=32,
        exploration_initial_eps=0.4,
        exploration_final_eps=0.08,
        exploration_fraction=0.8,
        target_update_interval=250
    )
    try:
        print("Training the model...")
        model.learn(total_timesteps=7000)
        model.save(f"{RESULTS_DIR}/models/task3")
        print("Model saved to results/models/task3.zip")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        env.close()

def test_rl_model():
    env = RoboboRLEnvironment()
    model = DQN.load(f"{RESULTS_DIR}/models/task3_final")
    print("Testing the model...")
    obs = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _ = env.step(action)
        if done:
            print("Collision detected. Resetting environment...")
            obs = env.reset()
    env.close()
