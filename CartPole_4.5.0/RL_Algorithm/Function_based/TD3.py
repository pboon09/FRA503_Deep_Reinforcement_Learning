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
        self.fc1 = nn.Linear(n_observations, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
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
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
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
        self.q1_fc1 = nn.Linear(n_observations + n_actions, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # ====================================== #

        # ===== Q2 network (independent weights) ===== #
        # ========= put your code here ========= #
        self.q2_fc1 = nn.Linear(n_observations + n_actions, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
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
        sa = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2
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
        sa = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        return q1
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
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state)

            if add_noise:
                noise = torch.randn_like(action) * self.exploration_noise
                action = (action + noise).clamp(-1.0, 1.0)

        # Scale action from [-1, 1] to [action_min, action_max]
        action_min, action_max = self.action_range
        scaled_action = action_min + (action + 1.0) * 0.5 * (action_max - action_min)
        return scaled_action
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
        with torch.no_grad():
            # Target policy smoothing
            next_action = self.actor_target(next_states)
            noise = (torch.randn_like(next_action) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            # Target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = rewards + self.discount_factor * (1 - dones) * torch.min(target_q1, target_q2)

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Delayed actor update
        actor_loss = None
        if self.total_steps % self.policy_update_freq == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

        return critic_loss, actor_loss
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

        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.stack([t.action for t in batch]).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones
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

        # ========= put your code here ========= #
        # --- Critic update ---
        with torch.no_grad():
            next_action = self.actor_target(next_states)
            noise = (torch.randn_like(next_action) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(next_states, next_action)
            target_q = rewards + self.discount_factor * (1 - dones) * torch.min(tq1, tq2)
        cq1, cq2 = self.critic(states, actions)
        critic_loss = F.mse_loss(cq1, target_q) + F.mse_loss(cq2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Delayed actor update (fresh forward pass after critic update) ---
        if self.total_steps % self.policy_update_freq == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.update_target_networks()
        # ====================================== #

        self.total_steps += 1

    def update_target_networks(self):
        # ========= put your code here ========= #
        # Polyak update for critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Polyak update for actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # ====================================== #

    def learn(self, env, num_agents: int = 256, max_steps: int = 200):
        """
        Train the agent for a fixed-length run across parallel vectorized envs.

        Isaac Lab auto-resets envs when done, so we never break early.
        The returned next_obs is already the reset observation for done envs.

        Args:
            env: The Isaac Lab vectorized environment.
            num_agents (int): Number of parallel environments.
            max_steps (int): Total steps to run.

        Returns:
            Tuple[float, int]: (avg_episode_return, avg_episode_length)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        if isinstance(obs, dict): obs = obs["policy"]
        ep_returns = torch.zeros(num_agents, device=self.device)
        ep_lengths = torch.zeros(num_agents, device=self.device)
        completed_returns = []
        completed_lengths = []

        for step in range(max_steps):
            # select_action handles batch input (dim==2 skips unsqueeze)
            scaled_action = self.select_action(obs)  # (num_agents, action_dim)

            next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
            if isinstance(next_obs, dict): next_obs = next_obs["policy"]
            dones = (terminated | truncated).float()

            # Convert scaled action back to raw [-1,1] for replay buffer storage
            action_min, action_max = self.action_range
            raw_action = (scaled_action - action_min) / (action_max - action_min) * 2.0 - 1.0

            # Single batch CPU transfer (avoid per-env .item() GPU syncs)
            rew_cpu = reward.cpu().numpy()
            done_cpu = dones.cpu().numpy()

            # Store all transitions
            for i in range(num_agents):
                self.store_transition(
                    obs[i], raw_action[i], float(rew_cpu[i]),
                    next_obs[i], float(done_cpu[i])
                )

            # Accumulate per-env episode returns and lengths (GPU)
            ep_returns += reward
            ep_lengths += 1

            # Track completed episodes (vectorized)
            done_mask = dones.bool().squeeze()
            if done_mask.any():
                done_idx = done_mask.nonzero(as_tuple=True)[0]
                completed_returns.extend(ep_returns[done_idx].cpu().tolist())
                completed_lengths.extend(ep_lengths[done_idx].cpu().tolist())
                ep_returns[done_mask] = 0.0
                ep_lengths[done_mask] = 0.0

            self.update_policy()
            obs = next_obs

        avg_return = sum(completed_returns) / len(completed_returns) if completed_returns else ep_returns.mean().item()
        avg_length = sum(completed_lengths) / len(completed_lengths) if completed_lengths else max_steps
        return avg_return, int(avg_length)
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
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor and critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'td3_cartpole.pth').
        """
        # ========= put your code here ========= #
        checkpoint = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # ====================================== #