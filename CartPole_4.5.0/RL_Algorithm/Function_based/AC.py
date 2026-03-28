from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from storage.on_policy import OnPolicyAlgorithm
from networks.mlp import MLP


# ============================================================ #
# =================== Actor-Critic Network =================== #
# ============================================================ #

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network supporting continuous and discrete actions.

    ``action_type='continuous'`` → ``Normal(mean, exp(log_std))``, learnable ``self.log_std``
    ``action_type='discrete'``   → ``Categorical(logits)``, no ``self.log_std``

    Args:
        state_dim (int): Observation space dimension.
        action_dim (int): Action vector dim (continuous) or # choices (discrete).
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
    ):
        super().__init__()

        assert action_type in ("continuous", "discrete"), \
            f"action_type must be 'continuous' or 'discrete', got '{action_type}'"

        self.action_type = action_type
        self.action_dim  = action_dim

        self.actor  = MLP(state_dim, action_dim, hidden_dims, activation)
        self.critic = MLP(state_dim, 1,          hidden_dims, activation)

        if self.action_type == "continuous":
            import math
            self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))

        self.distribution: Normal | Categorical | None = None

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def action_mean(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.mean
        return self.distribution.probs

    @property
    def action_std(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.log_std.exp()
        return torch.ones_like(self.distribution.probs)

    @property
    def entropy(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.entropy().sum(dim=-1)
        return self.distribution.entropy()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError("Use act() or evaluate().")

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """
        Build the action distribution from current observations.

        Continuous: ``Normal(mean, std)``
        Discrete  : ``Categorical(logits)``
        """
        # ========= put your code here ========= #
        mean = self.actor(obs)
        if self.action_type == "continuous":
            self.distribution = Normal(mean, self.log_std.exp())
        else:
            self.distribution = Categorical(logits=mean)
        # ====================================== #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample an action.

        Continuous: shape (batch, action_dim).
        Discrete  : shape (batch, 1).
        """
        # ========= put your code here ========= #
        self._update_distribution(obs)
        sample = self.distribution.sample()
        if self.action_type == "continuous":
            return sample
        else:
            return sample.unsqueeze(-1)
        # ====================================== #

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: actor mean (continuous) or argmax (discrete)."""
        # ========= put your code here ========= #
        mean = self.actor(obs)
        if self.action_type == "continuous":
            return mean
        else:
            return mean.argmax(dim=-1)
        # ====================================== #

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimate V(s), shape (batch, 1)."""
        # ========= put your code here ========= #
        return self.critic(obs)
        # ====================================== #

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of given actions under the current distribution.

        Continuous: sum over action dims → shape (batch,).
        Discrete  : scalar log-prob → shape (batch,).
        """
        # ========= put your code here ========= #
        if self.action_type == "continuous":
            return self.distribution.log_prob(actions).sum(dim=-1)
        else:
            return self.distribution.log_prob(actions.squeeze(-1))
        # ====================================== #


# ============================================================ #
# ====================== AC Agent ============================ #
# ============================================================ #

class AC(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C) — on-policy, episodic.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or # choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        learning_rate (float): Adam learning rate.
        discount_factor (float): Discount factor γ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        max_grad_norm (float): Gradient clipping norm.
    """

    def __init__(
        self,
        device=None,
        num_of_action: int = None,
        action_range: list = [None, None],
        n_observations: int = None,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
        learning_rate: float = None,
        discount_factor: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        max_grad_norm: float = None,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)
        # Orthogonal weight initialisation for training stability
        self.policy.actor.init_weights(scales=1.0)
        self.policy.critic.init_weights(scales=1.0)
        # ====================================== #

        self.optimizer       = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.action_type     = action_type
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm

        super(AC, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Trajectory Collection                                                #
    # ------------------------------------------------------------------ #

    def generate_trajectory(self, env) -> tuple:
        """
        Run one full episode and collect the trajectory as lists.

        Args:
            env: The environment.

        Returns:
            Tuple: (episode_return, log_prob_actions, values, rewards, timestep)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)

        log_prob_actions = []
        values = []
        rewards = []
        episode_return = 0.0
        timestep = 0
        done = False

        while not done:
            obs_tensor = obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32, device=self.device)

            action = self.policy.act(obs_tensor)
            value = self.policy.evaluate(obs_tensor)
            log_prob = self.policy.get_actions_log_prob(action)

            if self.action_type == "continuous":
                action_env = action.clamp(self.action_range[0], self.action_range[1])
            else:
                action_env = action

            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated if isinstance(terminated, bool) else (terminated | truncated).any()

            log_prob_actions.append(log_prob)
            values.append(value.squeeze(-1))
            if isinstance(reward, torch.Tensor):
                reward_val = reward.item() if reward.numel() == 1 else reward.sum().item()
                rewards.append(reward.view(-1))
            else:
                reward_val = float(reward)
                rewards.append(torch.tensor([reward_val], dtype=torch.float32, device=self.device))
            episode_return += reward_val
            timestep += 1

            if not isinstance(next_obs, torch.Tensor):
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            else:
                obs = next_obs.to(self.device)

        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values)
        rewards = torch.cat(rewards)

        return episode_return, log_prob_actions, values, rewards, timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Return & Loss                                                        #
    # ------------------------------------------------------------------ #

    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted Monte-Carlo returns G_t = r_t + γ·r_{t+1} + ...

        Args:
            rewards (Tensor): shape (T,).

        Returns:
            Tensor: Normalised return tensor of shape (T,).
        """
        # ========= put your code here ========= #
        T = len(rewards)
        returns = torch.zeros(T, device=rewards.device)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t].item() + self.discount_factor * G
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
        # ====================================== #

    def calculate_loss(self, log_prob_actions, values, returns):
        """
        Compute actor and critic losses.

        Args:
            log_prob_actions (Tensor): shape (T,).
            values (Tensor): shape (T,).
            returns (Tensor): shape (T,).

        Returns:
            Tuple[Tensor, Tensor]: (actor_loss, critic_loss)
        """
        # ========= put your code here ========= #
        advantages = returns - values.detach()
        actor_loss = -(log_prob_actions * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        # Entropy bonus: subtract entropy from loss so maximising entropy reduces total loss
        if self.policy.distribution is not None:
            entropy = self.policy.entropy.mean()
        else:
            entropy = torch.zeros(1, device=self.device).squeeze()
        actor_loss = actor_loss - self.entropy_coef * entropy
        return actor_loss, critic_loss
        # ====================================== #

    def update_policy(self, log_prob_actions, values, returns) -> float:
        """
        Backpropagate and update.

        Returns:
            float: Total combined loss.
        """
        # ========= put your code here ========= #
        actor_loss, critic_loss = self.calculate_loss(log_prob_actions, values, returns)
        total_loss = actor_loss + self.value_loss_coef * critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return total_loss.item()
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(self, env, max_steps: int = 600, num_agents: int = 1) -> tuple:
        """
        Train the agent for one episode (num_agents=1) or one rollout of
        ``max_steps`` across ``num_agents`` parallel environments.

        With parallel envs (num_agents > 1):
        - Collect a fixed rollout of ``max_steps`` from all envs.
        - When an env signals done, compute MC returns for that completed
          episode and accumulate actor + critic loss.
        - After the rollout, perform a single averaged gradient update.

        Args:
            env: The environment.
            max_steps (int): Maximum steps per rollout (parallel) or per
                             episode cap (single).
            num_agents (int): Number of parallel agents (1 for single env,
                              >1 for vectorised Isaac Lab envs).

        Returns:
            Tuple: (avg_return, loss, timestep_or_lengths)
        """
        self.policy.train()

        # ========= put your code here ========= #
        # ----- Single-env path (original behaviour) ----- #
        if num_agents == 1:
            episode_return, log_prob_actions, values, rewards, timestep = self.generate_trajectory(env)
            returns = self.compute_returns(rewards)
            loss = self.update_policy(log_prob_actions, values, returns)
            return episode_return, loss, timestep

        # ----- Parallel-env path ----- #
        obs, _ = env.reset()  # (num_agents, obs_dim)
        if isinstance(obs, dict): obs = obs["policy"]
        obs = obs.to(self.device)

        # Per-env episode storage
        env_log_probs    = [[] for _ in range(num_agents)]
        env_values       = [[] for _ in range(num_agents)]
        env_rewards_list = [[] for _ in range(num_agents)]
        completed_returns = []
        completed_lengths = []
        total_loss = 0.0
        num_updates = 0

        for step in range(max_steps):
            obs_tensor = obs.to(self.device)
            action   = self.policy.act(obs_tensor)               # (N, action_dim) or (N, 1)
            value    = self.policy.evaluate(obs_tensor)           # (N, 1)
            log_prob = self.policy.get_actions_log_prob(action)   # (N,)

            # Prepare action for the environment
            if self.action_type == "continuous":
                action_env = action.clamp(self.action_range[0], self.action_range[1])
            else:
                action_env = action

            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            if isinstance(next_obs, dict): next_obs = next_obs["policy"]
            dones = terminated | truncated  # (N,)

            # Batch CPU transfer once
            done_cpu = dones.cpu().numpy()

            # Store per-env data and handle episode boundaries
            for i in range(num_agents):
                env_log_probs[i].append(log_prob[i])
                env_values[i].append(value[i].squeeze())
                env_rewards_list[i].append(reward[i])

                if done_cpu[i]:
                    # Episode completed for env i
                    if len(env_rewards_list[i]) > 1:
                        rewards_t = torch.stack(env_rewards_list[i])
                        returns   = self.compute_returns(rewards_t)
                        lp        = torch.stack(env_log_probs[i])
                        vals      = torch.stack(env_values[i])
                        actor_loss, critic_loss = self.calculate_loss(lp, vals, returns)
                        loss = actor_loss + self.value_loss_coef * critic_loss
                        total_loss += loss
                        num_updates += 1
                        completed_returns.append(rewards_t.sum().item())
                        completed_lengths.append(len(env_rewards_list[i]))
                    # Reset storage for this env (Isaac Lab auto-resets)
                    env_log_probs[i]    = []
                    env_values[i]       = []
                    env_rewards_list[i] = []

            obs = next_obs

        # Single averaged gradient update
        if num_updates > 0:
            avg_loss = total_loss / num_updates
            self.optimizer.zero_grad()
            avg_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        avg_return = np.mean(completed_returns) if completed_returns else 0.0
        avg_length = int(np.mean(completed_lengths)) if completed_lengths else max_steps
        loss_val   = avg_loss.item() if num_updates > 0 else 0.0
        return avg_return, loss_val, avg_length
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Required by OnPolicyAlgorithm interface.
        For episodic AC, delegates to self.policy.act().
        """
        return self.policy.act(obs)

    def process_env_step(self, rewards, dones) -> None:
        """Not used by episodic AC — no RolloutBuffer to write to."""
        pass

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        # ========= put your code here ========= #
        self.policy.eval()
        with torch.no_grad():
            return self.policy.act_inference(obs)
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'ac_cartpole.pth').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'ac_cartpole.pth').
        """
        # ========= put your code here ========= #
        self.policy.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )
        # ====================================== #
