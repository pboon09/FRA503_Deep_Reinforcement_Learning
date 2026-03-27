from __future__ import annotations
import os
import numpy as np
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
        self.body = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions),
        )
        # ====================================== #

        # ===== Learnable log_std (continuous only) ===== #
        # Initialise to -0.5 so std starts at exp(-0.5) ≈ 0.61 for focused exploration
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.full((n_actions,), -0.5))

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
        return self.body(x)
        # ====================================== #


# ============================================================ #
# ==================== Value Network ========================= #
# ============================================================ #

class MC_REINFORCE_value_network(nn.Module):
    """
    Value (baseline) network for MC REINFORCE.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons per layer.
    """

    def __init__(self, n_observations: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
            entropy_coef: float = 0.01,
            value_loss_coef: float = 0.5,
    ) -> None:

        assert action_type in ("discrete", "continuous"), \
            f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.action_type     = action_type
        self.LR              = learning_rate
        self.entropy_coef    = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.policy_net = MC_REINFORCE_network(
            n_observations, hidden_dim, num_of_action, dropout, action_type
        ).to(device)
        self.value_net = MC_REINFORCE_value_network(n_observations, hidden_dim).to(device)
        self.optimizer = optim.AdamW(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate,
        )

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
        output = self.policy_net(obs)
        if self.action_type == "discrete":
            return Categorical(logits=output)
        else:
            return Normal(output, self.policy_net.log_std.exp())
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
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.action_type == "continuous":
            log_prob = log_prob.sum(dim=-1)
        else:
            action = action.unsqueeze(-1)
        return action, log_prob
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
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
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
        obs, info = env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)

        rewards = []
        log_probs = []
        trajectory = []
        done = False
        episode_return = 0.0

        while not done:
            dist = self._get_distribution(obs)
            action, log_prob = self._sample_action(dist)

            if self.action_type == "discrete":
                env_action = self.scale_action(action.item())
            else:
                action_min, action_max = self.action_range
                env_action = action.clamp(action_min, action_max)

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            if not isinstance(reward, (int, float)):
                reward_val = reward.item()
            else:
                reward_val = reward

            rewards.append(reward_val)
            log_probs.append(log_prob)
            trajectory.append((obs, action, reward_val))
            episode_return += reward_val

            if not isinstance(next_obs, torch.Tensor):
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            else:
                obs = next_obs.to(self.device)

        stepwise_returns = self.calculate_stepwise_returns(rewards)
        log_prob_actions = torch.stack(log_probs).squeeze()
        return episode_return, stepwise_returns, log_prob_actions, trajectory
        # ====================================== #

    def calculate_loss(
        self,
        stepwise_returns: torch.Tensor,
        log_prob_actions: torch.Tensor,
        states: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy gradient loss with optional value baseline and entropy.

        Args:
            stepwise_returns (Tensor): Normalised discounted returns, shape ``(T,)``.
            log_prob_actions (Tensor): shape ``(T,)``.
            states (Tensor | None): Episode observations ``(T, obs_dim)``.
                When provided, computes a value-function baseline and adds an
                entropy bonus to reduce variance and encourage exploration.

        Returns:
            Tensor: Scalar loss.
        """
        # ========= put your code here ========= #
        if states is not None:
            values = self.value_net(states)
            advantages = stepwise_returns - values.detach()
            value_loss = F.mse_loss(values, stepwise_returns)
            dist = self._get_distribution(states)
            entropy = dist.entropy()
            if self.action_type == "continuous":
                entropy = entropy.sum(-1)
            entropy = entropy.mean()
        else:
            advantages = stepwise_returns
            value_loss = torch.zeros(1, device=self.device).squeeze()
            entropy = torch.zeros(1, device=self.device).squeeze()

        policy_loss = -(log_prob_actions * advantages).mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        return loss
        # ====================================== #

    def update_policy(
        self,
        stepwise_returns: torch.Tensor,
        log_prob_actions: torch.Tensor,
        states: torch.Tensor = None,
    ) -> float:
        """
        Backpropagate the REINFORCE loss and update the policy and value networks.

        Args:
            stepwise_returns (Tensor): shape ``(T,)``.
            log_prob_actions (Tensor): shape ``(T,)``.
            states (Tensor | None): Episode observations for baseline/entropy.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        loss = self.calculate_loss(stepwise_returns, log_prob_actions, states)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()
        # ====================================== #

    def _scale_action_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Vectorised version of scale_action for a batch of discrete action indices.

        Maps each discrete action index in [0, num_of_action - 1] to a
        continuous value in [action_min, action_max].

        Args:
            actions (Tensor): Integer action indices, shape ``(batch, 1)``.

        Returns:
            Tensor: Scaled continuous actions, shape ``(batch, 1)``.
        """
        action_min, action_max = self.action_range
        continuous = action_min + (actions.float() / (self.num_of_action - 1)) * (action_max - action_min)
        return continuous

    def learn(self, env, num_agents: int = 1, max_steps: int = 600):
        """
        Train the agent for one episode (num_agents=1) or one rollout of
        ``max_steps`` across ``num_agents`` parallel environments.

        With parallel envs (num_agents > 1):
        - Collect a fixed rollout of ``max_steps`` from all envs.
        - When an env signals done, compute MC returns for that completed
          episode and accumulate the policy gradient loss.
        - After the rollout, perform a single averaged policy update.

        Args:
            env: The environment.
            num_agents (int): Number of parallel agents (1 for single env,
                              >1 for vectorised Isaac Lab envs).
            max_steps (int): Maximum number of steps per rollout when using
                             parallel envs.  Ignored when ``num_agents == 1``.

        Returns:
            Tuple: (avg_return, loss, episode_lengths)
        """
        self.policy_net.train()

        # ========= put your code here ========= #
        # ----- Single-env path (original behaviour) ----- #
        if num_agents == 1:
            episode_return, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
            loss = self.update_policy(stepwise_returns, log_prob_actions)
            return episode_return, loss, trajectory

        # ----- Parallel-env path ----- #
        obs, _ = env.reset()  # (num_agents, obs_dim)
        if isinstance(obs, dict): obs = obs["policy"]
        obs = obs.to(self.device)

        # Per-env episode storage
        env_log_probs = [[] for _ in range(num_agents)]
        env_rewards   = [[] for _ in range(num_agents)]
        completed_returns = []
        completed_lengths = []
        total_loss = 0.0
        num_updates = 0

        for step in range(max_steps):
            obs_tensor = obs.to(self.device)
            dist = self._get_distribution(obs_tensor)          # batched
            action, log_prob = self._sample_action(dist)       # (N, ...), (N,)

            # Prepare action for the environment
            if self.action_type == "continuous":
                env_action = action.clamp(self.action_range[0], self.action_range[1])
            else:
                env_action = self._scale_action_batch(action)  # (N, 1)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            if isinstance(next_obs, dict): next_obs = next_obs["policy"]
            dones = terminated | truncated  # (N,)

            # Batch CPU transfer once
            rew_cpu = reward.cpu().numpy()
            done_cpu = dones.cpu().numpy()

            # Store per-env data and handle episode boundaries
            for i in range(num_agents):
                env_log_probs[i].append(log_prob[i])
                env_rewards[i].append(float(rew_cpu[i]))

                if done_cpu[i]:
                    # Episode completed for env i
                    if len(env_rewards[i]) > 1:
                        returns = self.calculate_stepwise_returns(env_rewards[i])
                        lp = torch.stack(env_log_probs[i])
                        loss = self.calculate_loss(returns, lp)
                        total_loss += loss
                        num_updates += 1
                        completed_returns.append(sum(env_rewards[i]))
                        completed_lengths.append(len(env_rewards[i]))
                    # Reset storage for this env (Isaac Lab auto-resets)
                    env_log_probs[i] = []
                    env_rewards[i]   = []

            obs = next_obs

        # Single averaged gradient update
        if num_updates > 0:
            avg_loss = total_loss / num_updates
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()

        avg_return = np.mean(completed_returns) if completed_returns else 0.0
        loss_val   = avg_loss.item() if num_updates > 0 else 0.0
        return avg_return, loss_val, completed_lengths
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
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load policy network weights from disk.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., ``'reinforce_cartpole.pth'``).
        """
        # ========= put your code here ========= #
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )
        # ====================================== #