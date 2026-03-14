from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from storage.off_policy import OffPolicyAlgorithm


class SAC_Actor(nn.Module):
    """
    Stochastic actor network for SAC using the reparameterisation trick.

    Args:
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        n_actions (int): Action space dimension.
        log_std_min (float): Lower bound for log standard deviation.
        log_std_max (float): Upper bound for log standard deviation.
    """

    def __init__(
        self,
        n_observations: int,
        hidden_dim: int,
        n_actions: int,
        log_std_min: float = None,
        log_std_max: float = None,
    ):
        super(SAC_Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # ========= put your code here ========= #
        pass
        # ====================================== #

    def forward(self, state: torch.Tensor):
        """
        Compute mean and log_std of the Gaussian policy.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]: (mean, log_std) both shape (batch, n_actions).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def sample(self, state: torch.Tensor):
        """
        Sample an action using the reparameterisation trick and compute
        the corrected log-probability.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]:
                - action   : Squashed action in (-1, 1), shape (batch, n_actions).
                - log_prob : Corrected log π(a|s),       shape (batch,).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #


class SAC_Critic(nn.Module):
    """
    Twin Q-value network for SAC.

    SAC uses two critics (same as TD3) to reduce overestimation.
    The soft Bellman target uses ``min(Q1, Q2) − α · log π(a'|s')``.

    Args:
        n_observations (int): Observation space dimension.
        n_actions (int): Action space dimension.
        hidden_dim (int): Hidden layer width.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int):
        super(SAC_Critic, self).__init__()

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
        Compute both Q-values.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both shape (batch, 1).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) — off-policy, maximum entropy actor-critic.

    Args:
        device: Torch device.
        num_of_action (int): Action space dimension.
        action_range (list): [min, max] for action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        learning_rate (float): Learning rate for actor and critics.
        alpha_lr (float): Learning rate for automatic temperature tuning.
        tau (float): Polyak soft-update coefficient.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
        init_alpha (float): Initial temperature α.
        auto_alpha (bool): Enable automatic α tuning.
        target_entropy (float | None): Target entropy for auto-tuning.
                                       Defaults to −action_dim if None.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            learning_rate: float = None,
            alpha_lr: float = None,
            tau: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
            init_alpha: float = None,
            auto_alpha: bool = None,
            target_entropy: float | None = None,
    ) -> None:

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.actor        = SAC_Actor(n_observations, hidden_dim, num_of_action).to(device)
        self.critic       = SAC_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target = SAC_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.device    = device
        self.tau       = tau
        self.auto_alpha = auto_alpha

        # ===== Automatic temperature tuning ===== #
        # log_alpha is optimised instead of alpha directly to keep alpha > 0.
        # target_entropy is set to -action_dim as a heuristic (Haarnoja et al. 2018).
        self.log_alpha      = torch.tensor(
            [float(init_alpha)], requires_grad=True, device=device
        ).log()
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy if target_entropy is not None \
                              else -float(num_of_action)
        pass
        # ====================================== #

        # OffPolicyAlgorithm.__init__ creates self.memory = ReplayBuffer(buffer_size, batch_size)
        super(SAC, self).__init__(
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

    def select_action(self, state: torch.Tensor, evaluate: bool = False):
        """
        Sample an action from the stochastic policy.

        Training  (evaluate=False): sample from Normal distribution.
        Inference (evaluate=True) : use the mean (deterministic).

        Args:
            state (Tensor): Current state.
            evaluate (bool): True for deterministic inference.

        Returns:
            Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Compute SAC losses for critics, actor, and temperature.

        Args:
            states (Tensor): Batch of current states.
            actions (Tensor): Batch of actions taken.
            rewards (Tensor): Batch of rewards.
            next_states (Tensor): Batch of next states.
            dones (Tensor): Batch of terminal flags.

        Returns:
            Tuple[Tensor, Tensor, Tensor | None]:
                (critic_loss, actor_loss, alpha_loss or None)
        """
        with torch.no_grad():
        # ========= put your code here ========= #
            pass
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack into SAC-ready tensors.

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
        Perform one update step for critics, actor, and temperature.

        Returns:
            float | None: Critic loss, or None if buffer not ready.
        """
        sample = self.generate_sample()
        if sample is None:
            return None

        states, actions, rewards, next_states, dones = sample
        critic_loss, actor_loss, alpha_loss = self.calculate_loss(
            states, actions, rewards, next_states, dones
        )
        # ========= put your code here ========= #
        pass
        # ====================================== #

        self.alpha = self.log_alpha.exp().item()

        self.update_target_networks()

    def update_target_networks(self):
        """
        Overrides the no-op in OffPolicyAlgorithm.
        """
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
            filename (str): File name (e.g., 'sac_cartpole.pth').
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor and critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'sac_cartpole.pth').
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #