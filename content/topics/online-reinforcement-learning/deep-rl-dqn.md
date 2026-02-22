# Function Approximation and Deep Q-Learning

## When the lookup table doesn't fit

Imagine building a Q-table for a spaceship with a million rooms, ten actions each. That's ten million entries. Now imagine the spaceship is an Atari game -- the state space is every possible arrangement of pixels, astronomically large. Your table needs more entries than atoms in the universe. It doesn't fit.

So you do something radical: instead of storing one value per state-action pair, you build a **function approximator**. A machine that takes a state and outputs a value estimate. For simple problems, a linear model. For high-dimensional problems like game screens, a deep neural network that takes raw pixels and outputs Q-values for every action.

## The network starts chasing its own tail

The idea is elegant: train a network $Q(s,a;\theta)$ to approximate $Q^*(s,a)$. Collect experience, minimize the TD error -- the gap between the network's prediction and the bootstrap target $r + \gamma \max_{a'} Q(s',a';\theta)$.

But then the network starts hallucinating. Values diverge. Training blows up. Why?

The data is **correlated**. Consecutive transitions barely change -- the game shifts one frame. The network keeps replaying the same five seconds, overfitting to recent experience and forgetting everything else.

The target is **alive**. In supervised learning the label "cat" never changes. Here, the target $r + \gamma \max_{a'} Q(s',a';\theta)$ moves every time you update $\theta$. The network keeps chasing its own tail -- you're teaching a student while rewriting the answer key.

And you're combining function approximation, bootstrapping, and off-policy learning -- the **[deadly triad](./td-sarsa-qlearning)**.

## DQN: the two heroes

DQN (Deep Q-Network) was the breakthrough that showed deep RL can work. It solves the problems above with two tricks.

**The replay buffer** fixes correlated data. Instead of training on transitions in order, you store every transition $(s, a, r, s')$ in a big memory buffer. At training time, sample a random mini-batch. You might get a transition from episode 1, another from episode 50, another from episode 200 -- all in the same batch. Decorrelated, diverse diet of experience. The buffer also solves **catastrophic forgetting** -- without it, the network learns the current level and forgets everything before.

**The target network** fixes moving targets. Instead of using the current network for the bootstrap target, maintain a *separate* copy and only update it periodically (say, every 10,000 steps). Between updates, the target is fixed, so optimization looks more like supervised learning. You're still bootstrapping, but the target holds still long enough for gradient descent to make progress.

Additional stabilizers: **gradient clipping** (preventing giant gradients), **Huber loss** (robust to outliers), and **Double-Q learning** (using two networks to reduce overestimation -- one selects the best action, the other evaluates it).

## Beyond DQN

DQN opened the floodgates. **Policy gradient methods** skip Q-values entirely and directly adjust action probabilities based on whether episodes went well. **Actor-critic methods** combine both: a critic estimates values while an actor adjusts the policy using the critic's guidance.

[[simulation dqn-stability]]

[[simulation replay-buffer-explorer]]
[[simulation activation-functions]]
[[simulation cartpole-learning-curves]]

## What Comes Next

DQN works for episodic tasks with discrete actions, but many tasks never end, and the discount factor feels artificial when there's no natural episode boundary. The next lesson introduces the average-reward formulation and connects it to regret-based guarantees -- closing the loop between the online learning perspective that opened this topic and the full RL setting.

## Challenge (Optional)

Implement a minimal DQN on CartPole-v1. Train two versions: standard DQN (replay buffer + target network) and naive neural Q-learning (neither). Plot training curves and measure episodes to reach a score of 195 averaged over 100 episodes. Then ablate: add back one component at a time (buffer only, target only). Which contributes more to stability? Write a paragraph explaining what your ablation reveals about the deadly triad.
