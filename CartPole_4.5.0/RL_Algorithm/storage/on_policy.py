from __future__ import annotations
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm
from storage.buffers import RolloutBuffer


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for on-policy RL algorithms.

    Extends ``BaseAlgorithm`` with a ``RolloutBuffer`` and the associated
    helper methods.  On-policy algorithms collect a fresh rollout under the
    *current* policy, perform one (or more epochs of) updates, then discard
    all data before the next rollout.

    Buffer lifecycle (managed here, called by the algorithm)
    --------------------------------------------------------
    1. ``_init_storage()``    — allocate buffer once before training.
    2. ``add_transition()``   — write one env-step into the buffer.
    3. ``compute_returns()``  — implemented by subclass; fills returns/advantages.
    4. ``update()``           — implemented by subclass; calls mini_batch_generator.
    5. ``storage.clear()``    — reset write pointer; called inside ``update()``.

    Args:
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        learning_rate (float): Optimiser learning rate.
        initial_epsilon (float): Starting epsilon for epsilon-greedy (if used).
        epsilon_decay (float): Epsilon decay rate.
        final_epsilon (float): Minimum epsilon.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
        self,
        num_of_action: int = 1,
        action_range: list = [-3.0, 3.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.99,
    ) -> None:
        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        # Buffer and active transition container — populated by _init_storage()
        self.storage:    RolloutBuffer | None             = None
        self.transition: RolloutBuffer.Transition | None  = None

    # ------------------------------------------------------------------ #
    # Buffer management                                                    #
    # ------------------------------------------------------------------ #

    def _init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple,
        actions_shape: tuple,
        device: str | torch.device,
    ) -> None:
        """
        Args:
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon.
            obs_shape (tuple): Single-observation shape, e.g. ``(5,)``.
            actions_shape (tuple): Single-action shape.
            device: Torch device for buffer tensors.
        """
        self.storage = RolloutBuffer(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=obs_shape,
            actions_shape=actions_shape,
            device=device,
        )
        self.transition = RolloutBuffer.Transition()

    def set_storage(self, storage: RolloutBuffer) -> None:
        """
        Attach a pre-built ``RolloutBuffer`` (alternative to ``_init_storage``).

        Args:
            storage (RolloutBuffer): Pre-allocated rollout buffer.
        """
        self.storage    = storage
        self.transition = RolloutBuffer.Transition()

    def add_transition(self) -> None:
        """
        Flush the current ``self.transition`` into ``self.storage``.

        Call after ``process_env_step()`` has populated ``self.transition``
        with rewards and dones.
        """
        self.storage.add_transition(self.transition)
        self.transition.clear()

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for all parallel environments and populate
        ``self.transition`` with obs, actions, values, log-probs, mu, sigma.

        Must be implemented by the concrete algorithm (PPO / AC).
        """
        raise NotImplementedError

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Record rewards and dones into ``self.transition`` after env.step().

        Must be implemented by the concrete algorithm.
        """
        raise NotImplementedError

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """
        Compute GAE advantages and returns over the current rollout.

        Fills ``self.storage.returns`` and ``self.storage.advantages``.
        Must be implemented by the concrete algorithm.
        """
        raise NotImplementedError

    def update(self) -> dict:
        """
        Perform policy gradient updates over the collected rollout.

        Must call ``self.storage.clear()`` at the end.
        Must be implemented by the concrete algorithm.
        """
        raise NotImplementedError