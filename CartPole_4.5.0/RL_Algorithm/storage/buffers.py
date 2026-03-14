from __future__ import annotations
import random
from collections import deque, namedtuple
from collections.abc import Generator

import torch


# ============================================================ #
# ==================== ON-POLICY BUFFER ====================== #
# ============================================================ #

class RolloutBuffer:
    """
    Pre-allocated, fixed-length rollout buffer for on-policy algorithms (PPO).

    Stores one complete rollout of length ``num_transitions_per_env`` steps
    collected from ``num_envs`` parallel environments.  After each PPO update
    the buffer is cleared (``clear()``) and refilled — data is **never** reused
    across updates, which is the defining property of on-policy learning.

    Stored fields per transition
    ----------------------------
    observations       : (T, N, *obs_shape)
    actions            : (T, N, *actions_shape)
    rewards            : (T, N, 1)
    dones              : (T, N, 1)  byte
    values             : (T, N, 1)  V(s_t) from critic
    actions_log_prob   : (T, N, 1)  log π(a_t | s_t)
    mu                 : (T, N, *actions_shape)  distribution mean
    sigma              : (T, N, *actions_shape)  distribution std
    returns            : (T, N, 1)  GAE returns  (filled by compute_returns)
    advantages         : (T, N, 1)  GAE advantages

    where T = num_transitions_per_env, N = num_envs.

    Args:
        num_envs (int): Number of parallel environments.
        num_transitions_per_env (int): Rollout horizon (steps per env per update).
        obs_shape (tuple): Shape of a single observation, e.g. ``(5,)``.
        actions_shape (tuple): Shape of a single action.
                               Continuous: ``(action_dim,)``
                               Discrete  : ``(1,)``
        device (str | torch.device): Torch device for all tensors.
    """

    # ------------------------------------------------------------------ #
    # Inner transition container                                           #
    # ------------------------------------------------------------------ #

    class Transition:
        """
        Temporary container that holds the fields for **one** env-step across
        all parallel environments before they are written into the buffer.

        Populated by ``PPO.act()`` and ``PPO.process_env_step()``, then
        flushed into the buffer by ``RolloutBuffer.add_transition()``.
        """

        def __init__(self) -> None:
            self.observations:      torch.Tensor | None = None
            self.actions:           torch.Tensor | None = None
            self.rewards:           torch.Tensor | None = None
            self.dones:             torch.Tensor | None = None
            self.values:            torch.Tensor | None = None
            self.actions_log_prob:  torch.Tensor | None = None
            self.action_mean:       torch.Tensor | None = None
            self.action_sigma:      torch.Tensor | None = None
            # Note: hidden_states removed — feedforward networks only.
            # Add back if you extend to RNN-based policies.

        def clear(self) -> None:
            """Reset all fields to None (reuse object without re-allocating)."""
            self.__init__()

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple,
        actions_shape: tuple,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device                  = device
        self.num_envs                = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actions_shape           = actions_shape
        self.step                    = 0   # write pointer

        T, N = num_transitions_per_env, num_envs

        # ===== Core transition fields ===== #
        self.observations      = torch.zeros(T, N, *obs_shape,      device=device)
        self.actions           = torch.zeros(T, N, *actions_shape,   device=device)
        self.rewards           = torch.zeros(T, N, 1,                device=device)
        self.dones             = torch.zeros(T, N, 1,                device=device).byte()

        # ===== RL fields ===== #
        self.values            = torch.zeros(T, N, 1,                device=device)
        self.actions_log_prob  = torch.zeros(T, N, 1,                device=device)
        self.mu                = torch.zeros(T, N, *actions_shape,   device=device)
        self.sigma             = torch.zeros(T, N, *actions_shape,   device=device)

        # ===== Computed by compute_returns() ===== #
        self.returns           = torch.zeros(T, N, 1,                device=device)
        self.advantages        = torch.zeros(T, N, 1,                device=device)

    def add_transition(self, transition: "RolloutBuffer.Transition") -> None:
        """
        Copy one ``Transition`` into the buffer at the current write pointer.

        Args:
            transition (Transition): Populated transition container.

        Raises:
            OverflowError: If the buffer is full (call ``clear()`` first).
        """
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "RolloutBuffer overflow — call clear() before adding new transitions."
            )
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self) -> None:
        """Reset the write pointer. Tensor data is overwritten on next rollout."""
        self.step = 0

    # ------------------------------------------------------------------ #
    # Mini-batch generator for PPO updates                                 #
    # ------------------------------------------------------------------ #

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator:
        """
        Yield randomised mini-batches over the full rollout for PPO updates.

        Flattens the (T × N) rollout into a single batch of size
        ``T * N``, shuffles the indices once per call, then slices into
        ``num_mini_batches`` equal chunks.  The same shuffled order is
        used across all ``num_epochs`` epochs.

        Yields an 8-tuple per mini-batch (hidden-state fields removed):

            obs_batch                   (mini_batch_size, *obs_shape)
            actions_batch               (mini_batch_size, *actions_shape)
            target_values_batch         (mini_batch_size, 1)
            advantages_batch            (mini_batch_size, 1)
            returns_batch               (mini_batch_size, 1)
            old_actions_log_prob_batch  (mini_batch_size, 1)
            old_mu_batch                (mini_batch_size, *actions_shape)
            old_sigma_batch             (mini_batch_size, *actions_shape)

        Args:
            num_mini_batches (int): Number of mini-batches per epoch.
            num_epochs (int): Number of times to iterate over the rollout.
        """
        batch_size       = self.num_envs * self.num_transitions_per_env
        mini_batch_size  = batch_size // num_mini_batches
        indices          = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        # Flatten (T, N, ...) → (T*N, ...)
        observations         = self.observations.flatten(0, 1)
        actions              = self.actions.flatten(0, 1)
        values               = self.values.flatten(0, 1)
        returns              = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages           = self.advantages.flatten(0, 1)
        old_mu               = self.mu.flatten(0, 1)
        old_sigma            = self.sigma.flatten(0, 1)

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start     = i * mini_batch_size
                stop      = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield (
                    observations[batch_idx],          # obs_batch
                    actions[batch_idx],               # actions_batch
                    values[batch_idx],                # target_values_batch
                    advantages[batch_idx],            # advantages_batch
                    returns[batch_idx],               # returns_batch
                    old_actions_log_prob[batch_idx],  # old_actions_log_prob_batch
                    old_mu[batch_idx],                # old_mu_batch
                    old_sigma[batch_idx],             # old_sigma_batch
                )


# ============================================================ #
# =================== OFF-POLICY BUFFER ====================== #
# ============================================================ #

# Named tuple representing a single stored transition
_Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """
    Fixed-size FIFO experience replay buffer for off-policy algorithms (DQN).

    Stores ``(state, action, reward, next_state, done)`` transitions from any
    past version of the policy.  Random sampling breaks temporal correlation
    and stabilises neural network training.

    Unlike ``RolloutBuffer``, transitions here are **never discarded after an
    update** — the buffer keeps accumulating until it reaches ``buffer_size``,
    at which point the oldest entries are overwritten.  This is valid for
    off-policy algorithms because the Bellman target is re-evaluated using the
    *current* network at training time, making stale data still useful.

    Args:
        buffer_size (int): Maximum number of transitions to store.
        batch_size (int): Number of transitions returned by ``sample()``.
    """

    def __init__(self, buffer_size: int, batch_size: int = 1) -> None:
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done: bool,
    ) -> None:
        """
        Add one transition to the buffer.

        If the buffer is at capacity the oldest transition is silently dropped.

        Args:
            state:      Current state observation.
            action:     Action taken.
            reward:     Scalar reward received.
            next_state: Resulting next state.
            done (bool): True if the episode terminated after this step.
        """
        self.memory.append(_Transition(state, action, reward, next_state, done))

    def sample(self) -> list[_Transition] | None:
        """
        Sample a random mini-batch from the buffer.

        Returns ``None`` when the buffer contains fewer transitions than
        ``batch_size`` so callers can skip the update safely.

        Returns:
            list[Transition] | None: ``batch_size`` randomly drawn
            Transition named-tuples, or None if not enough data yet.
        """
        if len(self.memory) < self.batch_size:
            return None
        return random.sample(self.memory, self.batch_size)

    def is_ready(self) -> bool:
        """Return True once the buffer holds at least ``batch_size`` entries."""
        return len(self.memory) >= self.batch_size

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.memory)