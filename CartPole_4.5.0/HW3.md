# FRA 503: Deep Reinforcement Learning

## Cart Pole [ HW3 ]

After reviewing the updated `Cart-Pole` [instruction](https://github.com/S-Tuchapong/FRA503-Deep-Reinforcement-Learning-for-Robotics/tree/main/CartPole_4.5.0), you are now ready to proceed with the final homework.

Similar to the previous homework, this assignment focuses on the **Stabilizing Cart-Pole Task**, but using function approximation-based RL approaches instead of table-based RL approaches.

Additionally, as in the previous homework, the `CartPole` extension repository includes configurations for the **Swing-up Cart-Pole Task** as an optional resource for students seeking a more challenging task.

### Learning Objectives:
1. Understand how **function approximation** works and how to implement it.

2. Understand how **policy-based RL** works and how to implement it.

3. Understand how advanced RL algorithms balance exploration and exploitation.

4. Be able to differentiate RL algorithms based on stochastic or deterministic policies, as well as value-based, policy-based, or Actor-Critic approaches. 

5. Gain insight into different reinforcement learning algorithms, including Linear Q-Learning, Deep Q-Network (DQN), the REINFORCE algorithm, and the Actor-Critic algorithm. Analyze their strengths and weaknesses.

### Part 1: Understanding the Algorithm
In this homework, you have to implement 4 different function approximation-based RL algorithms:

- **Linear Q-Learning**

- **Deep Q-Network** (DQN)

- **REINFORCE algorithm**

- One algorithm chosen from the following Actor-Critic methods:
    - **Deep Deterministic Policy Gradient** (DDPG)
    - **Advantage Actor-Critic** (A2C)
    - **Proximal Policy Optimization** (PPO)
    - **Soft Actor-Critic** (SAC)

For each algorithm, describe whether it follows a value-based, policy-based, or Actor-Critic approach, specify the type of policy it learns (stochastic or deterministic), identify the type of observation space and action space (discrete or continuous), and explain how each advanced RL method balances exploration and exploitation.
 

### Part 2: Setting up `Cart-Pole` Agent.

Similar to the previous homework, you will implement a common components that will be the same in most of the function approximation-based RL in the `RL_base_function.py`.The core components should include, but are not limited to:

#### 1. RL Base class

This class should include:

- **Constructor `(__init__)`** to initialize the following parameters:

    - **Number of actions**: The total number of discrete actions available to the agent.

    - **Action range**: The minimum and maximum values defining the range of possible actions.

    - **Discretize state weight**: Weighting factor applied when discretizing the state space for learning.

    - **Learning rate**: Determines how quickly the model updates based on new information.

    - **Initial epsilon**: The starting probability of taking a random action in an ε-greedy policy.

    - **Epsilon decay rate**: The rate at which epsilon decreases over time to favor exploitation over exploration.

    - **Final epsilon**: The lowest value epsilon can reach, ensuring some level of exploration remains.

    - **Discount factor**: A coefficient (γ) that determines the importance of future rewards in decision-making.

    - **Buffer size**: Maximum number of experiences the buffer can hold.

    - **Batch size**: Number of experiences to sample per batch.

- **Core Functions**
    - `scale_action()`: scale the action (if it is computed from the sigmoid or softmax function) to the proper length.

    - `decay_epsilon()`: Decreases epsilon over time and returns the updated value.

Additional details about these functions are provided in the class file. You may also implement additional functions for further analysis.

#### 2. Replay Buffer Class

A class use to store state, action, reward, next state, and termination status from each timestep in episode to use as a dataset to train neural networks. This class should include:

- **Constructor `(__init__)`** to initialize the following parameters:
  
    - **memory**: FIFO buffer to store the trajectory within a certain time window.
  
    - **batch_size**: Number of data samples drawn from memory to train the neural network.

- **Core Functions**
  
    - `add()`: Add state, action, reward, next state, and termination status to the FIFO buffer. Discard the oldest data in the buffer
    
    - `sample()`: Sample data from memory to use in the neural network training.
 
  Note that some algorithms may not use all of the data mentioned above to train the neural network.

#### 3. Algorithm folder

This folder should include:

- **Linear Q Learning class**

- **Deep Q-Network class**

- **REINFORCE Class**

- One class chosen from the Part 1.

Each class should **inherit** from the `RL Base class` in `RL_base_function.py` and include:

- A constructor which initializes the same variables as the class it inherits from.

- Superclass Initialization (`super().__init__()`).

- An `update()` function that updates the agent’s learnable parameters and advances the training step.

- A `select_action()` function select the action according to current policy.

- A `learn()` function that train the regression or neural network.


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

You must evaluate the agent's performance in terms of **learning efficiency** (i.e., how well the agent learns to receive higher rewards) and **deployment performance** (i.e., how well the agent performs in the Cart-Pole problem). Analyze and visualize the results to determine:

1. Which algorithm performs best?
2. Why does it perform better than the others?
