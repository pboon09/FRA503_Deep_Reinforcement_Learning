# FRA 503: Deep Reinforcement Learning

## Cart Pole [ HW3 ]

After reviewing the updated `Cart-Pole` [instruction](https://github.com/S-Tuchapong/FRA503-Deep-Reinforcement-Learning-for-Robotics/tree/main/CartPole_4.5.0), you are now ready to proceed with the final homework.

Similar to the previous homework, this assignment focuses on the **Stabilizing Cart-Pole Task**, but using function approximation-based RL approaches instead of table-based RL approaches.

Additionally, as in the previous homework, the `CartPole` extension repository includes configurations for the **Swing-up Cart-Pole Task** as an optional resource for students seeking a more challenging task.

### Learning Objectives:
1. Understand how **function approximation** replaces tabular representations and enables learning in continuous state spaces.

2. Understand the distinction between **value-based**, **policy-based**, and **actor-critic** approaches and when each is appropriate.

3. Understand the difference between **on-policy** and **off-policy** learning, and the role of **experience replay**.

4. Understand how **parallel environments** change the training loop structure for different algorithms.

5. Understand the difference between **Monte-Carlo** and **Temporal Difference advantage estimation** and how this trade-off affects bias and variance.

6. [Optional] Understand how the **maximum entropy objective** in SAC reframes exploration as part of the learning goal.

### Part 1: Understanding the Algorithm
In this homework, you have to implement at least 5 different function approximation-based RL algorithms:

- **Linear Q-Learning**

- **Deep Q-Network** (DQN)

- **MC REINFORCE algorithm**

- **Actor-Critic (AC)**

- Choose at least one algorithm from the following Actor–Critic methods:
    - **Advantage Actor-Critic** (A2C)
    - **Twin Delayed DDPG** (TD3)
    - **Proximal Policy Optimization** (PPO)
    - **Soft Actor-Critic** (SAC)

For each of the following algorithms, describe: (a) whether it is value-based, policy-based, or actor-critic; (b) whether the policy is stochastic or deterministic; (c) whether it is on-policy or off-policy; (d) what type of action space it supports (discrete / continuous); and (e) how it balances exploration and exploitation.
 

### Part 2: Setting up `Cart-Pole` Agent.

Similar to the previous homework, you will implement a common components that will be the same in most of the function approximation-based RL in the `RL_base_function.py`.The core components should include, but are not limited to:

#### 2.1 RL Base Class (`RL_base_function.py`)
 
This is the shared root for all algorithms. It should include:
 
**Constructor (`__init__`)** initialising:
- `num_of_action` — number of discrete actions or action dimensions
- `action_range` — `[min, max]` for continuous action scaling
- `learning_rate` — optimiser learning rate
- `initial_epsilon`, `epsilon_decay`, `final_epsilon` — ε-greedy parameters
- `discount_factor` — γ for future reward discounting
 
**Core functions:**
- `scale_action()` — maps a raw network output to the valid action range
- `decay_epsilon()` — decrements epsilon toward `final_epsilon`
- `plot_durations()` — shared matplotlib visualisation of episode lengths
 
> Note: `ReplayBuffer` and the linear weight vector `self.w` are **not** part of `BaseAlgorithm`. They have been moved to the storage layer and `Linear_QN` respectively, so that this class remains a minimal shared interface.
 
---
 
#### 2.2 Storage Buffers (`storage/buffers.py`)
 
This file implements two buffer classes used by different algorithm families.
 
##### RolloutBuffer (on-policy — PPO, A2C)
 
Pre-allocates tensors for a fixed-length parallel rollout of shape `(T, N, ...)` where `T = num_transitions_per_env` and `N = num_envs`.
 
**Constructor (`__init__`)** initialising:
- `observations`, `actions`, `rewards`, `dones` — core transition tensors
- `values`, `actions_log_prob`, `mu`, `sigma` — policy quantities for PPO/A2C update
- `returns`, `advantages` — computed after rollout completion
 
**Inner `Transition` class** — a container for a single step's data, passed to `add_transition()`.
 
**Core functions:**
- `add_transition(transition)` — copies one `Transition` into the buffer at `self.step`, raises `OverflowError` if full
- `clear()` — resets `self.step = 0` so the buffer can be reused
- `mini_batch_generator(num_mini_batches, num_epochs)` — yields randomly shuffled mini-batches as 8-tuples: `(obs, actions, target_values, advantages, returns, old_log_prob, old_mu, old_sigma)`
 
##### ReplayBuffer (off-policy — DQN, TD3, SAC)
 
A FIFO experience buffer using a `deque` with `maxlen = buffer_size`.
 
**Constructor (`__init__`)** initialising:
- `memory` — `deque(maxlen=buffer_size)`
- `batch_size` — number of samples drawn per training step
 
**Core functions:**
- `add(state, action, reward, next_state, done)` — appends a `Transition` namedtuple, automatically discarding the oldest entry when full
- `sample()` — returns a random batch of `Transition` objects, or `None` if the buffer holds fewer than `batch_size` entries
- `is_ready()` — returns `True` when buffer is ready to sample
- `__len__()` — returns the current number of stored transitions
 
---
 
#### 2.3 On-Policy Algorithm Base (`storage/on_policy.py`)
 
`OnPolicyAlgorithm(BaseAlgorithm)` is the shared base for **PPO** and **A2C**. It manages buffer allocation and the transition accumulation loop so that each algorithm only needs to implement the update logic.
 
**Constructor** — calls `super().__init__()` and sets up `self.transition = RolloutBuffer.Transition()`.
 
**Core functions:**
- `_init_storage(num_envs, num_transitions_per_env, obs_shape, actions_shape)` — allocates a new `RolloutBuffer` and stores it as `self.storage`
- `set_storage(storage)` — attaches an externally created buffer (useful when the buffer is shared across objects)
- `add_transition()` — flushes `self.transition` into `self.storage` via `storage.add_transition()`
 
**Abstract interface** (must be overridden by subclasses):
- `act(obs)` — samples action, populates `self.transition`, returns action tensor
- `process_env_step(rewards, dones)` — records reward/done into `self.transition`, then calls `add_transition()`
- `compute_returns(last_obs)` — computes advantages and returns over the completed rollout
- `update()` — performs the gradient update using `self.storage`
 
---
 
#### 2.4 Off-Policy Algorithm Base (`storage/off_policy.py`)
 
`OffPolicyAlgorithm(BaseAlgorithm)` is the shared base for **DQN**, **TD3**, and **SAC**. It creates the replay buffer and provides thin wrappers so subclasses do not access `self.memory` directly.
 
**Constructor** — calls `super().__init__()` and creates `self.memory = ReplayBuffer(buffer_size, batch_size)`.
 
**Core functions:**
- `store_transition(state, action, reward, next_state, done)` — calls `self.memory.add()`
- `generate_sample()` — calls `self.memory.sample()`, returns `None` if buffer not ready
- `update_target_networks()` — no-op placeholder; overridden by each subclass with Polyak soft-update logic
 
---
 
#### 2.5 Network (`network/mlp.py`)
 
A shared `MLP` module used as the backbone for actor and critic networks across multiple algorithms.
 
**Constructor** parameters:
- `input_dim`, `output_dim`, `hidden_dims` (list of layer widths), `activation`
 
**Supported activations:** `relu`, `elu`, `tanh`
 
This module is imported by `AC.py`, `PPO.py`, `A2C.py`, `TD3.py`, and `SAC.py` — implement it before implementing any of those files.
 
---
 
#### 2.6 Algorithm Classes (`RL_Algorithm/Function_based/`)
 
You must implement all eight algorithm files. Each class must inherit from the appropriate base class as shown in the inheritance diagram below:
 
```
BaseAlgorithm  (RL_base_function.py)
├── OnPolicyAlgorithm  (storage/on_policy.py)
│   ├── AC              (MC episodic actor-critic)
│   ├── A2C             (TD synchronous actor-critic)
│   └── PPO             (clipped surrogate + GAE)
├── OffPolicyAlgorithm  (storage/off_policy.py)
│   ├── DQN             (deep Q-network with replay)
│   ├── TD3             (twin delayed deterministic)
│   └── SAC             (soft actor-critic, max entropy)
└── (direct)
    ├── Linear_QN       (linear function approximation)
    └── MC_REINFORCE    (policy gradient, MC returns)
```
 
Each algorithm class must include the following:
 
- **Constructor (`__init__`)** — initialise all networks, optimisers, and algorithm-specific hyperparameters, then call `super().__init__()`
- **`select_action(state)`** — returns an action for the given state following the current policy (ε-greedy, noise injection, or stochastic sampling depending on the algorithm)
- **`calculate_loss(...)`** — computes and returns the loss tensor(s) used for the gradient update
- **`update_policy()`** — samples from the buffer (or receives a trajectory) and performs one gradient step
- **`learn(env, ...)`** — the main training loop; supports both single env (`num_agents=1`) and parallel envs (`num_agents > 1`)
- **`save_model(path, filename)`** — saves network weights to disk
- **`load_model(path, filename)`** — loads network weights from disk


### Part 3: Trainning & Playing to stabilize `Cart-Pole` Agent.

You need to implement the `training loop` in train script and `main()` in the play script (in the *"Can be modified"* area of both files). Additionally, you must collect data, analyze results, and save models for evaluating agent performance.

#### Training the Agent

1. `Stabilizing` Cart-Pole Task

    ```
    python scripts/Function_based/train.py --task Stabilize-Isaac-Cartpole-v0 
    ```

2. `Swing-up` Cart-Pole Task (Optional)
    ```
    python scripts/Function_based/train.py --task SwingUp-Isaac-Cartpole-v0
    ```

#### Playing

1. `Stabilize` Cart-Pole Task

    ```
    python scripts/Function_based/play.py --task Stabilize-Isaac-Cartpole-v0 
    ```

2. `Swing-up` Cart-Pole Task (Optional)
    ```
    python scripts/Function_based/play.py --task SwingUp-Isaac-Cartpole-v0 
    ```

### Part 4: Evaluate `Cart-Pole` Agent performance.

Evaluate every algorithm on the Stabilizing task and report results across two dimensions:

1. **Learning efficiency** — how quickly the agent achieves high episode duration (plot episode duration vs. training episode for all algorithms on the same axes).
2. **Deployment performance** — how well the trained agent performs when run with play.py (use act_inference() / deterministic action selection, not sampling).
