# train_sim.py
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from traffic_env import TrafficEnv


def make_env():
    def _thunk():
        env = TrafficEnv(
            max_count_per_lane=200,
            clear_per_step=5,
            arrival_low=0,
            arrival_high=3,
            max_steps=300,
            penalty_waiting=0.1,
            seed=42,
        )
        return Monitor(env)
    return _thunk


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # single-env is fine here; wrap to VecEnv for SB3
    env = DummyVecEnv([make_env()])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=8_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        device=device,
    )

    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save("models/traffic_dqn")

    print("✅ saved: models/traffic_dqn.zip")
