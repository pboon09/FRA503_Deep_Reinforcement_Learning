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

        self.log_std_min = log_std_min if log_std_min is not None else -20
        self.log_std_max = log_std_max if log_std_max is not None else 2

        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, n_actions)
        self.log_std_head = nn.Linear(hidden_dim, n_actions)
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
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
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
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # sum over action dimensions
        return action, log_prob
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
        Compute both Q-values.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both shape (batch, 1).
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
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)

        # Scale action from [-1, 1] to [action_min, action_max]
        action_min, action_max = self.action_range
        scaled_action = action_min + (action + 1.0) * 0.5 * (action_max - action_min)
        return scaled_action
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
        # ========= put your code here ========= #
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            target_q = rewards + self.discount_factor * (1 - dones) * min_target_q

        # Critic loss
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Actor loss
        new_action, new_log_prob = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * new_log_prob.unsqueeze(-1) - min_q_new).mean()

        # Alpha loss
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
        else:
            alpha_loss = None

        return critic_loss, actor_loss, alpha_loss
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

        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.stack([t.action for t in batch]).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones
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
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature)
        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        # ====================================== #

        self.alpha = self.log_alpha.exp().item()

        self.update_target_networks()

    def update_target_networks(self):
        """
        Overrides the no-op in OffPolicyAlgorithm.
        """
        # ========= put your code here ========= #
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
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
        obs, _ = env.reset()  # (num_agents, obs_dim)
        ep_returns = torch.zeros(num_agents, device=self.device)
        ep_lengths = torch.zeros(num_agents, device=self.device)
        completed_returns = []
        completed_lengths = []

        for step in range(max_steps):
            # select_action handles batch input (dim==2 skips unsqueeze)
            scaled_action = self.select_action(obs)  # (num_agents, action_dim)

            next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
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
            filename (str): File name (e.g., 'sac_cartpole.pth').
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
            filename (str): File name (e.g., 'sac_cartpole.pth').
        """
        # ========= put your code here ========= #
        checkpoint = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(self.critic.state_dict())
        # ====================================== #