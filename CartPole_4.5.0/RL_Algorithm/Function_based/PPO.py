from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
from storage.on_policy import OnPolicyAlgorithm
from storage.buffers import RolloutBuffer
from RL_Algorithm.Function_based.AC import ActorCritic


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (PPO) — on-policy, clipped surrogate.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        num_learning_epochs (int): Epochs per PPO update.
        num_mini_batches (int): Mini-batches per epoch.
        clip_param (float): PPO clipping ε.
        gamma (float): Discount factor γ.
        lam (float): GAE lambda λ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        learning_rate (float): Adam learning rate.
        max_grad_norm (float): Gradient clipping norm.
        desired_kl (float): KL target for adaptive LR (0 to disable; use 0 for discrete).
        normalize_advantage_per_mini_batch (bool): Normalise advantages per mini-batch.
        use_clipped_value_loss (bool): Apply clipped value loss.
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
        num_learning_epochs: int = None,
        num_mini_batches: int = None,
        clip_param: float = None,
        gamma: float = None,
        lam: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        learning_rate: float = None,
        max_grad_norm: float = None,
        desired_kl: float = None,
        normalize_advantage_per_mini_batch: bool = False,
        use_clipped_value_loss: bool = True,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===== Build ActorCritic network (imported from AC.py) ===== #
        # ========= put your code here ========= #
        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)
        # ====================================== #

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # ===== PPO hyperparameters ===== #
        self.action_type                        = action_type
        self.clip_param                         = clip_param
        self.num_learning_epochs                = num_learning_epochs
        self.num_mini_batches                   = num_mini_batches
        self.value_loss_coef                    = value_loss_coef
        self.entropy_coef                       = entropy_coef
        self.gamma                              = gamma
        self.lam                                = lam
        self.max_grad_norm                      = max_grad_norm
        self.desired_kl                         = desired_kl
        self.learning_rate                      = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_clipped_value_loss             = use_clipped_value_loss

        super(PPO, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
        )

    # ------------------------------------------------------------------ #
    # Rollout collection                                                   #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for all parallel envs and populate self.transition.

        Args:
            obs (Tensor): shape (num_envs, obs_dim).

        Returns:
            Tensor: Sampled actions.
        """
        # ========= put your code here ========= #
        self.policy._update_distribution(obs)
        actions = self.policy.distribution.sample()
        if self.action_type == "discrete":
            actions = actions.unsqueeze(-1)

        self.transition.observations     = obs
        self.transition.actions          = actions
        self.transition.values           = self.policy.evaluate(obs)
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(actions)
        self.transition.action_mean      = self.policy.action_mean
        self.transition.action_sigma     = self.policy.action_std
        # ====================================== #

        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Write rewards and dones into self.transition, then flush to storage.

        Args:
            rewards (Tensor): shape (num_envs,) or (num_envs, 1).
            dones (Tensor): shape (num_envs,) or (num_envs, 1).
        """
        # ========= put your code here ========= #
        self.transition.rewards = rewards
        self.transition.dones   = dones
        # ====================================== #

        # Flush transition into RolloutBuffer via inherited add_transition()
        self.add_transition()

    # ------------------------------------------------------------------ #
    # Return & Advantage Computation                                       #
    # ------------------------------------------------------------------ #

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """
        Compute GAE returns and advantages over the collected rollout.

        Args:
            last_obs (Tensor): Observation after the final rollout step.
                               Shape: (num_envs, obs_dim).
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            last_values = self.policy.evaluate(last_obs)

        advantage = 0.0
        for step in reversed(range(self.storage.num_transitions_per_env)):
            if step == self.storage.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[step + 1]

            next_is_not_done = 1.0 - self.storage.dones[step].float()

            delta = (
                self.storage.rewards[step]
                + self.gamma * next_values * next_is_not_done
                - self.storage.values[step]
            )

            advantage = delta + self.gamma * self.lam * next_is_not_done * advantage

            self.storage.returns[step]    = advantage + self.storage.values[step]
            self.storage.advantages[step] = advantage

        # Advantages are normalized per mini-batch in update() (rsl_rl standard)
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def update(self) -> dict:
        """
        Perform PPO updates over the collected rollout.

        Returns:
            dict: Mean losses {'value', 'surrogate', 'entropy'}.
        """
        mean_value_loss     = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy        = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            # ========= put your code here ========= #
            # Re-evaluate under current policy
            self.policy._update_distribution(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            entropy_batch = self.policy.entropy

            # Per-mini-batch advantage normalization (rsl_rl standard)
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # KL-adaptive learning rate (continuous only)
            if self.desired_kl is not None and self.desired_kl > 0:
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(self.policy.action_std / old_sigma_batch + 1e-5)
                        + (old_sigma_batch**2 + (old_mu_batch - self.policy.action_mean)**2)
                        / (2.0 * self.policy.action_std**2 + 1e-5)
                        - 0.5,
                        dim=-1,
                    ).mean()
                if kl > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            # Clipped surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = ratio * advantages_batch.squeeze(-1)
            surrogate_clipped = torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            ) * advantages_batch.squeeze(-1)
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

            # Value loss (optionally clipped)
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + torch.clamp(
                    value_batch - target_values_batch,
                    -self.clip_param, self.clip_param,
                )
                value_loss_unclipped = (value_batch - returns_batch).pow(2)
                value_loss_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Total loss
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss     += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy        += entropy_batch.mean().item()
            # ====================================== #

        num_updates          = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy        /= num_updates

        self.storage.clear()   # on-policy: discard rollout after update

        return {
            "value":     mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy":   mean_entropy,
        }

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(
        self,
        env,
        num_envs: int,
        num_transitions_per_env: int,
        max_episodes: int = 10000,
    ) -> None:
        """
        Main PPO parallel training loop.

        Args:
            env: Isaac Lab vectorised environment.
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon per env.
            max_episodes (int): Total number of training rollouts.
        """
        # ========= put your code here ========= #
        if self.action_type == "continuous":
            actions_shape = (self.num_of_action,)
        else:
            actions_shape = (1,)

        self._init_storage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=(4,),
            actions_shape=actions_shape,
            device=self.device,
        )

        self.policy.train()
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"].to(self.device)

        # Per-env episode tracking
        ep_rewards = torch.zeros(num_envs, device=self.device)
        ep_steps = torch.zeros(num_envs, dtype=torch.int, device=self.device)
        total_episodes = 0
        iteration = 0
        global_step = 0
        sum_reward = 0.0
        last_log = 0

        while total_episodes < max_episodes:
            with torch.inference_mode():
                for _ in range(num_transitions_per_env):
                    actions = self.act(obs)
                    obs_dict, rewards, terminated, truncated, _ = env.step(actions)
                    obs = obs_dict["policy"].to(self.device)
                    dones = (terminated | truncated).to(self.device)
                    self.process_env_step(rewards.to(self.device), dones)

                    ep_rewards += rewards.to(self.device).squeeze()
                    ep_steps += 1
                    global_step += num_envs
                    for i in range(num_envs):
                        if dones[i].item():
                            self.episode_log.append({
                                "episode": total_episodes,
                                "global_step": global_step,
                                "ep_return": ep_rewards[i].item(),
                                "ep_length": ep_steps[i].item(),
                            })
                            self.episode_durations.append(ep_steps[i].item())
                            sum_reward += ep_rewards[i].item()
                            ep_rewards[i] = 0.0
                            ep_steps[i] = 0
                            total_episodes += 1

            self.compute_returns(obs)

            self.policy.train()
            losses = self.update()
            iteration += 1

            if total_episodes - last_log >= 100 and total_episodes > 0:
                n_new = total_episodes - last_log
                avg = sum_reward / n_new
                print(
                    f"[PPO] iter {iteration:5d} | ep {total_episodes} | "
                    f"avg_return={avg:.2f} | "
                    f"surr={losses['surrogate']:.4f} | "
                    f"val={losses['value']:.4f} | "
                    f"lr={self.learning_rate:.6f}"
                )
                self.plot_durations(timestep=int(avg))
                sum_reward = 0.0
                last_log = total_episodes
        # ====================================== #


    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        # ========= put your code here ========= #
        self.policy.eval()
        with torch.inference_mode():
            return self.policy.act_inference(obs)
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """Save actor-critic weights."""
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """Load actor-critic weights."""
        # ========= put your code here ========= #
        self.policy.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device, weights_only=True)
        )
        # ====================================== #
