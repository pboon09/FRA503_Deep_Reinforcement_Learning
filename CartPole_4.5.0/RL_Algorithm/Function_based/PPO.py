from __future__ import annotations
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

        Continuous: actions shape (num_envs, action_dim).
        Discrete  : actions shape (num_envs, 1).

        Args:
            obs (Tensor): shape (num_envs, obs_dim).

        Returns:
            Tensor: Sampled actions.
        """
        # ========= put your code here ========= #
        self.policy._update_distribution(obs)
        action = self.policy.act(obs)
        self.transition.observations = obs
        self.transition.actions = action
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(action).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
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
        self.transition.dones = dones
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
        last_value = self.policy.evaluate(last_obs).detach()
        advantage = 0

        for step in reversed(range(self.storage.num_transitions_per_env)):
            if step == self.storage.num_transitions_per_env - 1:
                next_values = last_value
            else:
                next_values = self.storage.values[step + 1]

            delta = (self.storage.rewards[step]
                     + self.gamma * next_values * (1 - self.storage.dones[step])
                     - self.storage.values[step])
            advantage = delta + self.gamma * self.lam * (1 - self.storage.dones[step]) * advantage
            self.storage.advantages[step] = advantage
            self.storage.returns[step] = advantage + self.storage.values[step]

        # Normalize advantages
        self.storage.advantages = (self.storage.advantages - self.storage.advantages.mean()) / (self.storage.advantages.std() + 1e-8)
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def update(self) -> dict:
        """
        Perform PPO updates over the collected rollout.

        Calls ``self.storage.mini_batch_generator()`` which now lives in
        ``RolloutBuffer`` (storage/buffers.py) and yields 8-tuples.

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
            self.policy._update_distribution(obs_batch)
            new_log_probs = self.policy.get_actions_log_prob(actions_batch)
            new_values = self.policy.evaluate(obs_batch)
            entropy = self.policy.entropy

            # Normalize advantages per mini-batch if enabled
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Policy ratio
            ratio = torch.exp(new_log_probs - old_actions_log_prob_batch.squeeze(-1))

            # Clipped surrogate objective
            surrogate1 = ratio * advantages_batch.squeeze(-1)
            surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch.squeeze(-1)
            surrogate_loss = torch.min(surrogate1, surrogate2).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (new_values - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss_unclipped = (new_values - returns_batch).pow(2)
                value_loss_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = (new_values - returns_batch).pow(2).mean()

            # Total loss
            loss = -surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # ====================================== #

        num_updates          = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy        /= num_updates

        # Adaptive learning rate based on KL divergence
        if self.desired_kl is not None and self.desired_kl > 0.0:
            with torch.no_grad():
                obs_all = self.storage.observations.flatten(0, 1)
                self.policy._update_distribution(obs_all)
                new_mu = self.policy.action_mean
                new_sigma = self.policy.action_std
                old_mu = self.storage.mu.flatten(0, 1)
                old_sigma = self.storage.sigma.flatten(0, 1)

                # Approximate KL divergence
                kl = torch.sum(
                    torch.log(new_sigma / old_sigma + 1e-5)
                    + (old_sigma.pow(2) + (old_mu - new_mu).pow(2)) / (2.0 * new_sigma.pow(2))
                    - 0.5,
                    dim=-1,
                )
                kl_mean = kl.mean()

                if kl_mean > 2.0 * self.desired_kl:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

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

        Calls ``_init_storage()`` (from OnPolicyAlgorithm) to create the buffer.

        Continuous: actions_shape = (num_of_action,)
        Discrete  : actions_shape = (1,)

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

        obs, _ = env.reset()
        if isinstance(obs, dict): obs = obs["policy"]
        n_obs = obs.shape[1]
        self._init_storage(num_envs, num_transitions_per_env, (n_obs,), actions_shape, self.device)

        obs, _ = env.reset()
        if isinstance(obs, dict): obs = obs["policy"]

        for episode in range(max_episodes):

            with torch.no_grad():
                for _ in range(num_transitions_per_env):
                    action = self.act(obs)

                    if self.action_type == "continuous":
                        action_clipped = action.clamp(self.action_range[0], self.action_range[1])
                    else:
                        action_clipped = action
                    obs, rewards, dones, truncated, _ = env.step(action_clipped)
                    if isinstance(obs, dict): obs = obs["policy"]

                    dones = dones | truncated
                    self.process_env_step(rewards, dones)

                self.compute_returns(obs)

            self.update()
        # ====================================== #


    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.

        Continuous: actor mean. Discrete: argmax of logits.

        Args:
            obs (Tensor): shape (1, obs_dim) or (obs_dim,).
        """
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
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        # ========= put your code here ========= #
        torch.save(self.policy.state_dict(), f"{path}/{filename}")
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        # ========= put your code here ========= #
        self.policy.load_state_dict(torch.load(f"{path}/{filename}"))
        # ====================================== #