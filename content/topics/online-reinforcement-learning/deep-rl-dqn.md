# Function Approximation and Deep Q-Learning

## When the lookup table does not fit

Imagine you are trying to build a Q-table for a spaceship with a million rooms. Each room has ten possible actions. That is ten million entries in your table. Now imagine the spaceship is actually an Atari game with pixels on the screen — the state space is every possible arrangement of pixels, which is astronomically large. Your lookup table would need more entries than there are atoms in the universe. It simply does not fit.

So we do something radical: instead of storing one value for every state-action pair, we build a **function approximator** — a machine that takes in a state (or a state-action pair) and outputs a value estimate. We *guess* the values using a parameterized model. For simple problems, a linear model works. For complex, high-dimensional problems like game screens or robot sensors, we use a deep neural network. That network takes the raw state as input and outputs Q-values for every action.

## The neural network as a value guesser

The idea is elegant: train a neural network $Q(s,a;\theta)$ to approximate $Q^*(s,a)$. You collect experience by interacting with the environment, and you minimize the TD error — the gap between the network's prediction and the bootstrap target $r + \gamma \max_{a'} Q(s',a';\theta)$.

But then something goes wrong. The network starts dreaming and hallucinating. Values diverge. Training becomes unstable. Why? Three problems conspire against you.

First, the training data is **correlated**. Consecutive transitions in an episode are nearly identical — the game barely changes between one frame and the next. Imagine the network trying to learn from the last game it played, but it keeps replaying the exact same five seconds over and over. That is like studying only page 47 of the textbook. It overfits to recent experience and forgets everything else.

Second, the **targets move**. In supervised learning, the labels are fixed — a cat is always a cat. But in Q-learning, the target depends on the network's own parameters. When you update the weights, the targets change, which changes the loss, which changes the gradients. You are chasing a moving target, and the target moves because you are chasing it.

Third, we are combining function approximation, bootstrapping, and off-policy learning — the **deadly triad** from [TD learning](./td-sarsa-qlearning). Without careful engineering, the system explodes.

## DQN: engineering away the instability

DQN (Deep Q-Network) was the breakthrough that showed you *can* make deep RL work. It solves the three problems above with two key tricks.

**The replay buffer** is the solution to correlated data. Instead of training on transitions in order, you store every transition $(s, a, r, s')$ in a big memory buffer. When it is time to train, you sample a random mini-batch from the buffer. This mixes up the pages — you might get a transition from episode 1, another from episode 50, another from episode 200, all in the same batch. The samples are decorrelated, and the network sees a diverse diet of experience.

The replay buffer also solves **catastrophic forgetting** — the tendency of neural networks to forget old knowledge when trained on new data. Without the buffer, the network would learn to play the current level of the game and completely forget how to play the earlier levels. By replaying old transitions, the network retains its accumulated knowledge.

**The target network** is the solution to moving targets. Instead of using the current network to compute the bootstrap target, you maintain a *separate* copy of the network — the target network — and only update it periodically (say, every 10,000 steps). Between updates, the target is fixed, so the optimization is more like supervised learning. You are still bootstrapping, but the target is stable enough for the gradient descent to make progress without chasing itself in circles.

Additional stabilization tricks include **gradient clipping** (preventing the gradients from becoming too large), **robust loss functions** (like Huber loss instead of squared error), and **Double-Q learning** (using two networks to reduce overestimation of Q-values, where one network selects the best action and the other evaluates it).

## Beyond DQN

DQN opened the floodgates, but it is just the beginning. **Policy gradient methods** directly optimize the policy without learning Q-values at all — they adjust the probability of taking each action based on whether the resulting episode was good or bad. **Actor-critic methods** combine the two ideas: a "critic" estimates values (like DQN), while an "actor" adjusts the policy using the critic's guidance. This gives you lower variance than pure policy gradients and more flexibility than pure value-based methods.

For **continuous control** — robotics, motor control, anything where the action is a real number rather than a discrete choice — variants like DDPG and SAC extend these ideas to continuous action spaces.

[[simulation dqn-stability]]
[[simulation activation-functions]]
[[simulation cartpole-learning-curves]]

## Big Ideas
* The replay buffer is not just an optimization trick — it is the key to breaking temporal correlations that would otherwise prevent the neural network from generalizing.
* The target network is a deliberate act of self-deception: you pretend the target is fixed even though it is not, because that pretense is stable enough for gradient descent to make progress.
* DQN's success on Atari was not about the algorithm being perfect — it was about identifying exactly which instabilities were fatal and engineering specific fixes for each one.
* The gap between value-based and policy-based methods is not just technical. Policy gradients optimize what you care about directly; value methods optimize a proxy. Each approach trades off computation, variance, and credit assignment differently.

## What Comes Next

DQN made deep RL work for episodic tasks with discrete actions. But many continuing tasks never end, and the discount factor $\gamma$ feels artificial when there is no natural episode boundary. A factory robot does not "reset" at the end of a shift — it just keeps running.

The next lesson closes the loop between the online learning perspective that opened this topic and the full RL framework. It introduces the average-reward formulation — where you optimize long-run reward per step rather than discounted cumulative reward — and connects it to regret-based guarantees. Algorithms like UCRL show that the optimism-under-uncertainty principle from bandits carries all the way up to MDPs with unknown dynamics.

## Check Your Understanding
1. Without the replay buffer, consecutive training samples are highly correlated. Why does correlation in training data hurt neural network learning, and why does sampling uniformly from a replay buffer break that correlation?
2. The target network is updated every $C$ steps rather than continuously. What happens to training stability as $C \to 1$ (update every step)? What happens as $C \to \infty$ (never update)? Where is the right trade-off?
3. Double-Q learning uses two networks — one to select the best action and another to evaluate it. What specific bias does standard Q-learning have that Double-Q learning corrects, and why does the standard single-network approach produce that bias?

## Challenge
Implement a minimal DQN on CartPole-v1 (available in Gymnasium). Train two versions: one with a replay buffer and target network (standard DQN) and one without either (naive neural Q-learning). Plot the training curves for both and measure how many episodes each needs to reach a score of 195 averaged over 100 episodes. Then ablate: add back one component at a time (buffer only, then target only) and show empirically which component contributes more to stability on this environment. Write a one-paragraph explanation of what your ablation reveals about the deadly triad.
