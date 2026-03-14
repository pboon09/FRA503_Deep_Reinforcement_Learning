from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from RL_Algorithm.RL_base_function import BaseAlgorithm


# ============================================================ #
# ==================== Policy Network ======================== #
# ============================================================ #

class MC_REINFORCE_network(nn.Module):
    """
    Policy network for the MC REINFORCE algorithm.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons per layer.
        n_actions (int): Number of output values.
                         Discrete  → number of action choices.
                         Continuous → dimension of the action vector.
        dropout (float): Dropout rate for regularization.
        action_type (str): ``'discrete'`` or ``'continuous'``.
    """

    def __init__(
        self,
        n_observations: int,
        hidden_size: int,
        n_actions: int,
        dropout: float,
        action_type: str = "discrete",
    ):
        super(MC_REINFORCE_network, self).__init__()

        assert action_type in ("discrete", "continuous"), \
            f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        self.action_type = action_type

        # ===== Shared MLP body ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Learnable log_std (continuous only) ===== #
        # Initialise to 0 so std starts at exp(0) = 1.0
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Discrete  : returns logits of shape ``(batch, n_actions)``.
        Continuous: returns action mean of shape ``(batch, n_actions)``.
                    Use ``self.log_std`` separately to build the Normal distribution.

        Args:
            x (Tensor): Input state tensor of shape ``(batch, n_observations)``.

        Returns:
            Tensor: Logits (discrete) or action mean (continuous).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #


# ============================================================ #
# =================== MC REINFORCE Agent ===================== #
# ============================================================ #

class MC_REINFORCE(BaseAlgorithm):
    """
    Monte-Carlo REINFORCE policy gradient algorithm supporting both
    discrete and continuous action spaces.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
                             Ignored for discrete.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout rate.
        action_type (str): ``'discrete'`` or ``'continuous'``.
        learning_rate (float): AdamW learning rate.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            dropout: float = None,
            action_type: str = None,
            learning_rate: float = None,
            discount_factor: float = None,
    ) -> None:

        assert action_type in ("discrete", "continuous"), \
            f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.action_type = action_type
        self.LR          = learning_rate

        self.policy_net = MC_REINFORCE_network(
            n_observations, hidden_dim, num_of_action, dropout, action_type
        ).to(device)
        self.optimizer  = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device     = device
        self.steps_done = 0
        pass
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Distribution helpers                                                 #
    # ------------------------------------------------------------------ #

    def _get_distribution(self, obs: torch.Tensor):
        """
        Build the action distribution from the current observation.

        Args:
            obs (Tensor): State tensor of shape ``(batch, obs_dim)``.

        Returns:
            torch.distributions.Distribution: Categorical or Normal distribution.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def _sample_action(self, dist) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the distribution and compute its log-probability.

        Args:
            dist: A ``Categorical`` or ``Normal`` distribution object.

        Returns:
            Tuple[Tensor, Tensor]:
                - action  : Discrete: shape ``(batch, 1)``.
                            Continuous: shape ``(batch, action_dim)``.
                - log_prob: Shape ``(batch,)``.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def calculate_stepwise_returns(self, rewards: list) -> torch.Tensor:
        """
        Compute normalised discounted returns G_t for each timestep.

        Args:
            rewards (list): Rewards collected in the episode.

        Returns:
            Tensor: Normalised return tensor of shape ``(T,)``.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def generate_trajectory(self, env):
        """
        Run one full episode and collect the trajectory.

        Args:
            env: The environment.

        Returns:
            Tuple:
                - episode_return (float)
                - stepwise_returns (Tensor): shape ``(T,)``
                - log_prob_actions (Tensor): shape ``(T,)``
                - trajectory (list): ``[(state, action, reward), ...]``
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def calculate_loss(
        self,
        stepwise_returns: torch.Tensor,
        log_prob_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy gradient loss.

        Args:
            stepwise_returns (Tensor): shape ``(T,)``.
            log_prob_actions (Tensor): shape ``(T,)``.

        Returns:
            Tensor: Scalar loss.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def update_policy(
        self,
        stepwise_returns: torch.Tensor,
        log_prob_actions: torch.Tensor,
    ) -> float:
        """
        Backpropagate the REINFORCE loss and update the policy network.

        Args:
            stepwise_returns (Tensor): shape ``(T,)``.
            log_prob_actions (Tensor): shape ``(T,)``.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def learn(self, env, num_agents: int = 1):
        """
        Train the agent for one episode.

        Args:
            env: The environment.
            num_agents (int): Number of parallel agents (>1 for vectorised envs).

        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        self.policy_net.train()

        # ========= put your code here ========= #
        pass
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save policy network weights to disk.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., ``'reinforce_cartpole.pth'``).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load policy network weights from disk.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., ``'reinforce_cartpole.pth'``).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #