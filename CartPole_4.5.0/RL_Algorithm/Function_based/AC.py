from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

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

        Collects next-state values and done flags so that one-step TD errors
        δ_t = r_t + γ·V(s_{t+1})·(1−done) − V(s_t) can be computed.

        Args:
            env: The environment.

        Returns:
            Tuple: (episode_return, log_prob_actions, values, next_values, rewards, dones, timestep)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)

        log_prob_actions = []
        values = []
        next_values = []
        rewards = []
        dones_list = []
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

            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            else:
                next_obs = next_obs.to(self.device)

            # V(s_{t+1}) for TD error computation
            with torch.no_grad():
                next_value = self.policy.evaluate(next_obs)

            log_prob_actions.append(log_prob)
            values.append(value.squeeze(-1))
            next_values.append(next_value.squeeze(-1))
            if isinstance(reward, torch.Tensor):
                reward_val = reward.item() if reward.numel() == 1 else reward.sum().item()
                rewards.append(reward.view(-1))
            else:
                reward_val = float(reward)
                rewards.append(torch.tensor([reward_val], dtype=torch.float32, device=self.device))
            dones_list.append(torch.tensor([1.0 if done else 0.0], device=self.device))
            episode_return += reward_val
            timestep += 1

            obs = next_obs

        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values)
        next_values = torch.cat(next_values)
        rewards = torch.cat(rewards)
        dones_tensor = torch.cat(dones_list)

        return episode_return, log_prob_actions, values, next_values, rewards, dones_tensor, timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Return & Loss                                                        #
    # ------------------------------------------------------------------ #

    def compute_td_errors(self, rewards: torch.Tensor, values: torch.Tensor,
                           next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute one-step TD errors δ_t = r_t + γ·V(s_{t+1})·(1−done) − V(s_t).

        Args:
            rewards (Tensor): shape (T,).
            values (Tensor): shape (T,).
            next_values (Tensor): shape (T,). V(s_{t+1}) for each step.
            dones (Tensor): shape (T,). 1.0 if episode ended at step t.

        Returns:
            Tensor: TD errors of shape (T,).
        """
        # ========= put your code here ========= #
        td_errors = rewards + self.discount_factor * next_values * (1.0 - dones) - values
        return td_errors
        # ====================================== #

    def calculate_loss(self, log_prob_actions, values, td_errors):
        """
        Compute actor and critic losses using one-step TD error as advantage.

        δ_t = r_t + γ·V(s_{t+1})·(1−done) − V(s_t)  (S&B Ch 13.5)

        Args:
            log_prob_actions (Tensor): shape (T,).
            values (Tensor): shape (T,).
            td_errors (Tensor): shape (T,).

        Returns:
            Tuple[Tensor, Tensor]: (actor_loss, critic_loss)
        """
        # ========= put your code here ========= #
        actor_loss = -(log_prob_actions * td_errors.detach()).mean()
        critic_loss = td_errors.pow(2).mean()
        if self.policy.distribution is not None:
            entropy = self.policy.entropy.mean()
        else:
            entropy = torch.zeros(1, device=self.device).squeeze()
        actor_loss = actor_loss - self.entropy_coef * entropy
        return actor_loss, critic_loss
        # ====================================== #

    def update_policy(self, log_prob_actions, values, td_errors) -> float:
        """
        Backpropagate and update.

        Returns:
            float: Total combined loss.
        """
        # ========= put your code here ========= #
        actor_loss, critic_loss = self.calculate_loss(log_prob_actions, values, td_errors)
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
            episode_return, log_prob_actions, values, next_values, rewards, dones_t, timestep = self.generate_trajectory(env)
            td_errors = self.compute_td_errors(rewards, values.squeeze(), next_values.squeeze(), dones_t)
            loss = self.update_policy(log_prob_actions, values.squeeze(), td_errors)
            return episode_return, loss, timestep

        # ----- Parallel-env path: online per-step TD update (S&B Ch 13.5) ----- #
        obs, _ = env.reset()  # (num_agents, obs_dim)
        if isinstance(obs, dict): obs = obs["policy"]
        obs = obs.to(self.device)

        completed_returns = []
        completed_lengths = []
        total_loss = 0.0

        for step in range(max_steps):
            obs_tensor = obs.to(self.device)
            action   = self.policy.act(obs_tensor)
            value    = self.policy.evaluate(obs_tensor).squeeze(-1)
            log_prob = self.policy.get_actions_log_prob(action)

            if self.action_type == "continuous":
                action_env = action.clamp(self.action_range[0], self.action_range[1])
            else:
                action_env = action

            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            if isinstance(next_obs, dict): next_obs = next_obs["policy"]
            dones = (terminated | truncated).float()

            with torch.no_grad():
                next_value = self.policy.evaluate(next_obs).squeeze(-1)

            # δ_t = r_t + γ·V(s_{t+1})·(1−done) − V(s_t)
            delta = reward + self.discount_factor * next_value * (1.0 - dones) - value
            actor_loss, critic_loss = self.calculate_loss(log_prob, value, delta)
            step_loss = actor_loss + self.value_loss_coef * critic_loss

            self.optimizer.zero_grad()
            step_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += step_loss.item()

            if dones.any():
                done_returns = reward[dones.bool()]
                completed_returns.extend(done_returns.cpu().tolist())
                completed_lengths.extend([1] * int(dones.sum().item()))

            obs = next_obs

        avg_return = np.mean(completed_returns) if completed_returns else 0.0
        avg_length = int(np.mean(completed_lengths)) if completed_lengths else max_steps
        loss_val   = total_loss / max_steps
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
            torch.load(os.path.join(path, filename), map_location=self.device, weights_only=True)
        )
        # ====================================== #
