from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from storage.off_policy import OffPolicyAlgorithm


class TD3_Actor(nn.Module):
    """
    Deterministic actor network for TD3.

    Args:
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        n_actions (int): Action space dimension.
    """

    def __init__(self, n_observations: int, hidden_dim: int, n_actions: int):
        super(TD3_Actor, self).__init__()
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tensor: Deterministic action in [-1, 1] (scale externally).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #


class TD3_Critic(nn.Module):
    """
    Q-value network for TD3.
    Args:
        n_observations (int): Observation space dimension.
        n_actions (int): Action space dimension.
        hidden_dim (int): Hidden layer width.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int):
        super(TD3_Critic, self).__init__()

        # ===== Q1 network ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Q2 network (independent weights) ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Compute Q1 and Q2 values for a (state, action) pair.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both of shape (batch, 1).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Return only Q1 — used for the actor update.

        The actor maximises Q1 only (not min(Q1, Q2)) because at policy
        update time we want the gradient signal from one network, not a
        min operation which would break the gradient flow.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tensor: Q1 value of shape (batch, 1).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3).

    Args:
        device: Torch device.
        num_of_action (int): Action space dimension.
        action_range (list): [min, max] for action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        learning_rate (float): Learning rate for both actor and critics.
        tau (float): Polyak soft-update coefficient.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
        exploration_noise (float): Std of Gaussian noise added during interaction.
        target_noise (float): Std of smoothing noise added to target actions.
        target_noise_clip (float): Clip range for target smoothing noise.
        policy_update_freq (int): Critic steps between each actor update.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            learning_rate: float = None,
            tau: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
            exploration_noise: float = None,
            target_noise: float = None,
            target_noise_clip: float = None,
            policy_update_freq: int = None,
    ) -> None:

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.actor        = TD3_Actor(n_observations, hidden_dim, num_of_action).to(device)
        self.actor_target = TD3_Actor(n_observations, hidden_dim, num_of_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = TD3_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target = TD3_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.device             = device
        self.tau                = tau
        self.exploration_noise  = exploration_noise
        self.target_noise       = target_noise
        self.target_noise_clip  = target_noise_clip
        self.policy_update_freq = policy_update_freq
        self.total_steps        = 0   # counts critic updates to trigger delayed actor update
        pass
        # ====================================== #

        # OffPolicyAlgorithm.__init__ creates self.memory = ReplayBuffer(buffer_size, batch_size)
        super(TD3, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state: torch.Tensor, add_noise: bool = True):
        """
        Select a deterministic action with optional Gaussian exploration noise.

        Args:
            state (Tensor): Current state.
            add_noise (bool): Add exploration noise during training.
                              Set False for evaluation / play.

        Returns:
            Tensor: Action tensor of shape (action_dim,).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Compute critic and actor loss for one mini-batch.

        Args:
            states (Tensor): Batch of current states.
            actions (Tensor): Batch of actions taken.
            rewards (Tensor): Batch of rewards.
            next_states (Tensor): Batch of next states.
            dones (Tensor): Batch of terminal flags.

        Returns:
            Tuple[Tensor, Tensor | None]: (critic_loss, actor_loss or None)
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack into TD3-ready tensors.

        Returns:
            Tuple or None:
                - states (Tensor)
                - actions (Tensor)
                - rewards (Tensor)
                - next_states (Tensor)
                - dones (Tensor)
        """
        # ========= put your code here ========= #
        batch = super().generate_sample()
        if batch is None:
            return None
        # ====================================== #

    def update_policy(self):
        """
        Perform one critic update and (if scheduled) one actor update.

        Returns:
            float | None: Critic loss value, or None if buffer not ready.
        """
        sample = self.generate_sample()
        if sample is None:
            return None

        states, actions, rewards, next_states, dones = sample
        critic_loss, actor_loss = self.calculate_loss(
            states, actions, rewards, next_states, dones
        )
        # ========= put your code here ========= #
        pass
        # ====================================== #

        self.total_steps += 1

    def update_target_networks(self):
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def learn(self, env, num_agents: int = 1, max_steps: int = 1000):
        """
        Train the agent for one episode (single env) or fixed-length run
        (parallel envs).

        Args:
            env: The Isaac Lab environment.
            num_agents (int): Number of parallel environments.
            max_steps (int): Steps per episode (single) or total steps (parallel).

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor and critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'td3_cartpole.pth').
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor and critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'td3_cartpole.pth').
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #