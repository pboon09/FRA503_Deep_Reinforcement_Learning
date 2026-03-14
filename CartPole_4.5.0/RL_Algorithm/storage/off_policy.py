from __future__ import annotations
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm
from storage.buffers import ReplayBuffer


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for off-policy RL algorithms.

    Extends ``BaseAlgorithm`` with a ``ReplayBuffer`` and the shared
    helper methods for storing and sampling transitions.

    Buffer lifecycle (managed here, called by the algorithm)
    --------------------------------------------------------
    1. ``_init_storage()``       — allocate buffer once before training.
    2. ``store_transition()``    — add one ``(s, a, r, s', done)`` tuple.
    3. ``generate_sample()``     — sample a random mini-batch (returns None
                                   if buffer is not ready yet).
    4. ``update_policy()``       — implemented by subclass; uses the sample.
    5. ``update_target_networks()`` — Polyak soft-update of target network;
                                   implemented by subclass (DQN-style) or
                                   overridden for other off-policy methods.

    Args:
        num_of_action (int): Number of discrete actions (or action dim for
                             continuous off-policy algorithms).
        action_range (list): [min, max] for continuous action scaling.
        learning_rate (float): Optimiser learning rate.
        initial_epsilon (float): Starting epsilon for epsilon-greedy.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum epsilon.
        discount_factor (float): Discount factor γ.
        buffer_size (int): ``ReplayBuffer`` capacity.
        batch_size (int): Mini-batch size for each update.
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.5, 2.5],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 10_000,
        batch_size: int = 64,
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

        self.batch_size  = batch_size
        self.buffer_size = buffer_size

        # ===== Replay buffer ===== #
        self.memory = ReplayBuffer(buffer_size, batch_size)

    # ------------------------------------------------------------------ #
    # Buffer management                                                    #
    # ------------------------------------------------------------------ #

    def _init_storage(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        """
        (Re-)allocate the ``ReplayBuffer`` with explicit sizes.

        Call this if you want to override the sizes set in ``__init__``
        after construction (e.g. from a training script).

        Args:
            buffer_size (int): New buffer capacity.
            batch_size (int): New mini-batch size.
        """
        self.buffer_size = buffer_size
        self.batch_size  = batch_size
        self.memory      = ReplayBuffer(buffer_size, batch_size)

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done: bool,
    ) -> None:
        """
        Store one ``(s, a, r, s', done)`` transition in the replay buffer.

        Args:
            state:      Current state observation.
            action:     Action taken.
            reward:     Scalar reward received.
            next_state: Resulting next state.
            done (bool): True if the episode terminated after this step.
        """
        self.memory.add(state, action, reward, next_state, done)

    def generate_sample(self):
        """
        Sample a random mini-batch from the replay buffer.

        Returns ``None`` (and the caller should skip the update) if the
        buffer does not yet hold enough transitions.

        Returns:
            list[Transition] | None: ``batch_size`` transitions, or None.
        """
        return self.memory.sample()

    # ------------------------------------------------------------------ #
    # Abstract interface — must be implemented by subclasses              #
    # ------------------------------------------------------------------ #

    def select_action(self, state) -> tuple:
        """
        Select an action using the current policy (typically epsilon-greedy).

        Must be implemented by the concrete algorithm.
        """
        raise NotImplementedError

    def update_policy(self) -> float | None:
        """
        Perform one gradient update on the policy network.

        Should call ``generate_sample()`` internally and return the loss,
        or None if the buffer is not ready.

        Must be implemented by the concrete algorithm.
        """
        raise NotImplementedError

    def update_target_networks(self) -> None:
        """
        The default implementation here is a no-op placeholder.  Override
        in DQN (or any other algorithm that maintains a target network).
        """
        pass

    def learn(self, env) -> tuple:
        """
        Run one episode of environment interaction and policy updates.

        Must be implemented by the concrete algorithm.
        """
        raise NotImplementedError