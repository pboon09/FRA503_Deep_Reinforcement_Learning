import numpy as np
from collections import defaultdict
import torch
import matplotlib
import matplotlib.pyplot as plt

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class BaseAlgorithm:
    """
    Base class for all function approximation-based RL algorithms.

    Args:
        num_of_action (int): Number of discrete actions available.
        action_range (list): [action_min, action_max] for continuous scaling.
        learning_rate (float): Learning rate (stored as self.lr).
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step decay applied to epsilon.
        final_epsilon (float): Floor value for epsilon.
        discount_factor (float): Discount factor γ for future rewards.
    """

    def __init__(
        self,
        num_of_action: int = None,
        action_range: list = [None, None],
        learning_rate: float = None,
        initial_epsilon: float = None,
        epsilon_decay: float = None,
        final_epsilon: float = None,
        discount_factor: float = None,
    ):
        self.lr              = learning_rate
        self.discount_factor = discount_factor
        self.epsilon         = initial_epsilon
        self.epsilon_decay   = epsilon_decay
        self.final_epsilon   = final_epsilon
        self.num_of_action   = num_of_action
        self.action_range    = action_range   # [action_min, action_max]
        self.training_error  = []

        # ===== Matplotlib / plotting (shared by all subclasses) ===== #
        self.episode_durations = []
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()

    def scale_action(self, action: int) -> torch.Tensor:
        """
        Map a discrete action index [0, n-1] to a continuous value in
        [action_min, action_max].

        Args:
            action (int): Discrete action index in [0, num_of_action - 1].

        Returns:
            torch.Tensor: Scaled continuous action tensor.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate by ``epsilon_decay``, floored at
        ``final_epsilon``.

        Call once per environment step during training.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Visualisation                                                      #
    # ------------------------------------------------------------------ #

    # Modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        """
        Plot episode durations with a 100-episode running average.

        Args:
            timestep (int | None): Episode length to record. Pass None to
                                   redraw without adding a new data point.
            show_result (bool): If True titles the plot 'Result',
                                otherwise 'Training...'.
        """
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if self.is_ipython:
            from IPython import display
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #