from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from storage.on_policy import OnPolicyAlgorithm
from storage.buffers import RolloutBuffer
from network.mlp import MLP


class ActorCritic_A2C(nn.Module):
    """
    Shared Actor-Critic network for A2C.

    Uses the same backbone as PPO's ActorCritic but kept separate so
    students can clearly see that A2C and PPO share the same network
    architecture — the difference is entirely in the update rule.

    Args:
        state_dim (int): Observation space dimension.
        action_dim (int): Action space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        action_type: str = "continuous",
        init_noise_std: float = 1.0,
    ):
        super(ActorCritic_A2C, self).__init__()

        assert action_type in ("continuous", "discrete"), \
            f"action_type must be 'continuous' or 'discrete', got '{action_type}'"

        self.action_type = action_type

        # ===== Actor and Critic networks ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Learnable log_std for continuous actions ===== #
        if self.action_type == "continuous":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))

        self.distribution = None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.action_type == "continuous":
            return self.distribution.entropy().sum(dim=-1)
        return self.distribution.entropy()

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """
        Build the action distribution from current observations.

        Continuous: ``Normal(mean, std)``
        Discrete  : ``Categorical(logits)``
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample an action from the current distribution.

        Continuous: shape (batch, action_dim).
        Discrete  : shape (batch, 1).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: actor mean (continuous) or argmax (discrete)."""
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimate V(s), shape (batch, 1)."""
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of given actions under the current distribution.

        Continuous: sum over action dims → shape (batch,).
        Discrete  : scalar log-prob      → shape (batch,).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #


class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C) — synchronous on-policy.

    Inherits from ``OnPolicyAlgorithm`` which provides:
        • ``self.storage``      — RolloutBuffer
        • ``self.transition``   — Transition container
        • ``add_transition()``  — flush transition into storage
        • ``_init_storage()``   — allocate buffer before training
        • ``plot_durations()``  — from BaseAlgorithm

    Key difference from PPO
    -----------------------
    A2C uses a **one-step TD advantage** instead of GAE:

        A(s, a) = r + γ · V(s') · (1 − done) − V(s)

    There is **no clipping** of the policy ratio and **no multiple epochs**
    of mini-batch updates — the rollout is used exactly once then discarded.
    This makes A2C simpler and faster per update, but less sample-efficient
    than PPO.

    Key difference from AC (episodic)
    ----------------------------------
    AC (in AC.py) collects a full episode and computes Monte-Carlo returns.
    A2C collects a fixed-length rollout from N parallel envs and computes
    TD advantages, allowing updates before the episode ends.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
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
        num_of_action: int = 1,
        action_range: list = [-3.0, 3.0],
        n_observations: int = 5,
        hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        action_type: str = "continuous",
        init_noise_std: float = 1.0,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy = ActorCritic_A2C(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)
        # ====================================== #

        self.optimizer       = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.action_type     = action_type
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm

        # Experiment with different values and configurations to see how they
        # affect the training process. Remember to document any changes you make
        # and analyze their impact on the agent's performance.

        super(A2C, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Rollout collection (implements OnPolicyAlgorithm interface)          #
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
        pass
        # ====================================== #

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Record rewards and dones into self.transition, then flush to storage.

        Args:
            rewards (Tensor): shape (num_envs,) or (num_envs, 1).
            dones (Tensor): shape (num_envs,) or (num_envs, 1).
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

        self.add_transition()

    # ------------------------------------------------------------------ #
    # Return & Advantage Computation                                       #
    # ------------------------------------------------------------------ #

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """
        Compute one-step TD advantages and returns over the rollout.

        A2C uses the simpler TD advantage instead of GAE:

            δ_t = r_t + γ · V(s_{t+1}) · (1 − done) − V(s_t)

        Unlike PPO which accumulates δ with a λ trace, A2C uses δ directly
        as the advantage without any multi-step lookahead correction.

            A_t = δ_t
            R_t = A_t + V(s_t)

        Args:
            last_obs (Tensor): Observation after the final rollout step,
                               shape (num_envs, obs_dim). Used to bootstrap
                               V(s_{T}) for the last transition.
        """
        # ===== Bootstrap value at end of rollout ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        for step in reversed(range(self.storage.num_transitions_per_env)):

            # ===== TD delta: r + γ·V(s')·(1-done) - V(s) ===== #
            # ========= put your code here ========= #
            pass
            # ====================================== #

            # ===== A2C: advantage = delta (no lambda accumulation) ===== #
            # ========= put your code here ========= #
            pass
            # ====================================== #

            # ===== Return = advantage + V(s) ===== #
            # ========= put your code here ========= #
            pass
            # ====================================== #

        # ===== Normalize advantages ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def update(self) -> dict:
        """
        Perform a single A2C update over the collected rollout.

        Unlike PPO, A2C uses the rollout **once** with no clipping:

            actor_loss  = −mean( log π(a|s) · A(s, a) )
            critic_loss = MSE( V(s), R )
            loss        = actor_loss + value_loss_coef · critic_loss
                                     − entropy_coef   · entropy

        Returns:
            dict: {'value': critic_loss, 'actor': actor_loss, 'entropy': entropy}
        """
        # ===== Flatten rollout tensors ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Recompute log-probs, values, entropy for current policy ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Actor loss: -mean(log_prob · advantage) ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Critic loss: MSE(V(s), returns) ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Total loss → gradient step with grad clipping ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        self.storage.clear()

        return {
            "value":   0.0,   # replace with actual critic_loss.item()
            "actor":   0.0,   # replace with actual actor_loss.item()
            "entropy": 0.0,   # replace with actual entropy.mean().item()
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
        Main A2C parallel training loop.

        Calls ``_init_storage()`` (from OnPolicyAlgorithm) to create the buffer.

        Continuous: actions_shape = (num_of_action,)
        Discrete  : actions_shape = (1,)

        Args:
            env: Isaac Lab vectorised environment.
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon per env.
            max_episodes (int): Total number of training rollouts.
        """
        # ===== Create rollout buffer via inherited _init_storage() ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        # ===== Reset environment ===== #
        # ========= put your code here ========= #
        pass
        # ====================================== #

        for episode in range(max_episodes):

            with torch.inference_mode():
                for _ in range(num_transitions_per_env):

                    # ===== Sample actions ===== #
                    # ========= put your code here ========= #
                    pass
                    # ====================================== #

                    # ===== Step environment ===== #
                    # ========= put your code here ========= #
                    pass
                    # ====================================== #

                    # process_env_step calls add_transition() internally
                    # ========= put your code here ========= #
                    pass
                    # ====================================== #

                # ===== Bootstrap returns ===== #
                # ========= put your code here ========= #
                pass
                # ====================================== #

            # ===== Policy update (calls storage.clear() internally) ===== #
            # ========= put your code here ========= #
            pass
            # ====================================== #

    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.

        Continuous: actor mean. Discrete: argmax of logits.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'a2c_cartpole.pth').
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'a2c_cartpole.pth').
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #