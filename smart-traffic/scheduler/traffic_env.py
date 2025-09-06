# traffic_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrafficEnv(gym.Env):
    """
    Simple 4-lane traffic signal simulator for RL training.
    Train here first; later you will replace arrivals with YOLO counts in live_control.py.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_count_per_lane: int = 200,
        clear_per_step: int = 5,
        arrival_low: int = 0,
        arrival_high: int = 3,
        max_steps: int = 300,
        penalty_waiting: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=max_count_per_lane, shape=(4,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)

        self.MAX_COUNT = max_count_per_lane
        self.CLEAR_PER_STEP = clear_per_step
        self.ARRIVAL_LOW = arrival_low
        self.ARRIVAL_HIGH = arrival_high
        self.MAX_STEPS = max_steps
        self.PENALTY = penalty_waiting

        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.int32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state[:] = 0
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action: int):
        self.steps += 1

        # vehicles cleared on chosen lane
        cleared = int(min(self.state[action], self.CLEAR_PER_STEP))
        self.state[action] -= cleared

        # arrivals (simulation)
        arrivals = self.rng.integers(self.ARRIVAL_LOW, self.ARRIVAL_HIGH + 1, size=4)
        self.state = np.minimum(self.state + arrivals, self.MAX_COUNT)

        # reward: clear more, wait less
        reward = float(cleared) - self.PENALTY * float(self.state.sum())

        terminated = False
        truncated = self.steps >= self.MAX_STEPS
        info = {"cleared": cleared, "arrivals": arrivals}
        return self.state.copy(), reward, terminated, truncated, info
