import numpy as np


class MultiArmedBandit:
    def __init__(self, n_arms: int = 10):
        self.n_arms = n_arms
        self.true_means = np.random.randn(n_arms)

    def pull(self, arm: int) -> float:
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Arm index {arm} out of range [0, {self.n_arms - 1}]")
        return np.random.randn() + self.true_means[arm]

    def get_optimal_arm(self) -> int:
        return int(np.argmax(self.true_means))

    def get_optimal_reward(self) -> float:
        return float(np.max(self.true_means))

    def reset(self):
        self.true_means = np.random.randn(self.n_arms)
