# FRA 503: Deep Reinforcement Learning

## Cart Pole [ HW2 ]

After following the `Cart-Pole` [instruction](https://github.com/S-Tuchapong/FRA503-Deep-Reinforcement-Learning-for-Robotics/tree/main/CartPole_4.5.0), you are now ready to proceed with the homework.

In this homework, students will work on the **Stabilizing Cart-Pole Task**, where the goal is to train the agent with learning algorithms (i.e. Q-Learning, Monte-Carlo, Temporal Difference Learning, and Double Q-Learning) to control a cart moving along a frictionless track to keep a pole balanced. The pole starts near an upright position (close to 90° vertical), and the agent must apply the right forces to the cart to prevent it from tipping over. The challenge is to stabilize the system while minimizing unnecessary movement. The episode ends if the pole leans too far or the cart moves too far from the center.

[Stabilize Cart-Pole Task.webm](https://github.com/user-attachments/assets/5b7c8574-0ea9-4757-8248-e50095130b09)

The `CartPole` extension repository also provides the configuration for the **Swing-up Cart-Pole Task** as an additional optional resource for students seeking a more challenging task (only the work on the **Stabilizing Cart-Pole Task** will be graded in homework 2). In the **Swing-up Cart-Pole Task**, the pole starts hanging downward and needs to be swung up to achieve the same posture as in the **Stabilizing Cart-Pole Task**.

[SwingUp Cart-Pole Task.webm](https://github.com/user-attachments/assets/03ce068c-f052-416d-93d3-698c48c11606)

### Learning Objectives:
1. Understand how a reinforcement learning agent learns (i.e., evaluates and improves its policy) in an environment where the true dynamic model is unknown.

2. Gain insight into different reinforcement learning algorithms, including Monte Carlo methods, the SARSA algorithm, Q-learning, and Double Q-learning. Analyze their strengths and weaknesses.

3. Explore approaches to implementing reinforcement learning in real-world scenarios where the state and action spaces are continuous.


### Part 1: Setting up `Cart-Pole` Agent.

For the first part of this homework, you will implement a Cart-Pole agent from scratch, i.e., you must implement the **constructor** and **core functions** of the `RL Base Class`, as well as the **algorithms** in the Algorithm folder. The core components should include, but are not limited to:

#### 1. RL Base class

This class should include:

- **Constructor `(__init__)`** to initialize the following parameters:

    - **Control type**: Enumeration of RL algorithms used for decision-making (i.e. Monte Carlo, Temporal Difference, Q-learning, or Double Q-learning).

    - **Number of actions**: The total number of discrete actions available to the agent.

    - **Action range**: The minimum and maximum values defining the range of possible actions.

    - **Discretize state weight**: Weighting factor applied when discretizing the state space for learning.

    - **Learning rate**: Determines how quickly the model updates based on new information.

    - **Initial epsilon**: The starting probability of taking a random action in an ε-greedy policy.

    - **Epsilon decay rate**: The rate at which epsilon decreases over time to favor exploitation over exploration.

    - **Final epsilon**: The lowest value epsilon can reach, ensuring some level of exploration remains.

    - **Discount factor**: A coefficient (γ) that determines the importance of future rewards in decision-making.

- **Core Functions**
    - `get_discretize_action()`: Returns a discrete action based on the current policy.

    - `mapping_action()`: Converts a discrete action back into a continuous action within the defined action range.

    - `discretize_state()`: Discretizes and scales the state based on observation weights.

    - `decay_epsilon()`: Decreases epsilon over time and returns the updated value.

Additional details about these functions are provided in the class file.

**Note:**
The `RL Base Class` also include two additional functions:

- `save_q_value()` which save model function from Q(s,a) as defaultdict.

- `load_q_value()` which load model function from Q(s,a) as defaultdict.

You may also implement additional functions for further analysis.

#### 2. Algorithm folder

This folder should include:

- **Monte Carlo class**

- **SARSA class**

- **Q-Learning Class**

- **Double Q-Learning Class**

Each class should **inherit** from the `RL Base class` and include:

- A constructor which initializes the same variables as the class it inherits from.

- Superclass Initialization (`super().__init__()`).

- An `update()` function that updates the agent’s learnable parameters and advances the training step.

### Part 2: Trainning & Playing to stabilize `Cart-Pole` Agent.

You need to implement the `training loop` in train script and `main()` in the play script (in the *"Can be modified"* area of both files). Additionally, you must collect data, analyze results, and save models for evaluating agent performance.

#### Training the Agent

1. `Stabilizing` Cart-Pole Task

    ```
    python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0 
    ```

2. `Swing-up` Cart-Pole Task (Optional)
    ```
    python scripts/RL_Algorithm/train.py --task SwingUp-Isaac-Cartpole-v0
    ```

#### Playing

1. `Stabilize` Cart-Pole Task

    ```
    python scripts/RL_Algorithm/play.py --task Stabilize-Isaac-Cartpole-v0 
    ```

2. `Swing-up` Cart-Pole Task (Optional)
    ```
    python scripts/RL_Algorithm/play.py --task SwingUp-Isaac-Cartpole-v0 
    ```

### Part 3: Evaluate `Cart-Pole` Agent performance.

You must evaluate the agent's performance in terms of **learning efficiency** (i.e., how well the agent learns to receive higher rewards) and **deployment performance** (i.e., how well the agent performs in the Cart-Pole problem). Analyze and visualize the results to determine:

1. Which algorithm performs best?
2. Why does it perform better than the others?
3. How do the resolutions of the action space and observation space affect the learning process? Why?

Idea for visulization:
![3D Surface Plot of Q-Values](https://media.githubusercontent.com/media/S-Tuchapong/FRA503-Deep-Reinforcement-Learning-for-Robotics/refs/heads/main/CartPole_4.5.0/HW%20materials/3D%20Surface%20Plot%20of%20Q-Values.png)
