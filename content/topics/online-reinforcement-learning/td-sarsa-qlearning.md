# Temporal-Difference Learning, SARSA, and Q-Learning

## Learning without waiting for the end

Monte Carlo said: play the whole game, then learn from the final score. TD learning says: why wait? After every single step, you can update your estimate using the reward you just received and your current guess about the next state's value. You are learning as you go, one step at a time.

This is **bootstrapping** — using your own estimates as part of the learning target. It sounds circular, and in a way it is. But it works, and it works remarkably well.

## TD(0): the one-step update

The simplest TD method updates the state-value estimate after every transition:

$$
V(s_t)\leftarrow V(s_t)+\alpha\left[r_{t+1}+\gamma V(s_{t+1})-V(s_t)\right].
$$

The bracketed term is the **TD error** — the difference between what you expected ($V(s_t)$) and a better estimate based on what actually happened ($r_{t+1}+\gamma V(s_{t+1})$). You nudge your estimate in the direction of this error, scaled by the step size $\alpha$.

Think of it like this: you are walking to a restaurant and you estimated it would take 20 minutes. After 5 minutes, you have walked a quarter of the way and you now estimate 18 minutes total. TD learning says: update your original estimate *right now*, based on the partial information. You do not have to arrive at the restaurant to revise your prediction.

## SARSA: on-policy control

SARSA learns the value of state-action pairs, and it learns about the policy it is actually following. The name comes from the five quantities it uses: $S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$.

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\right].
$$

You are in state $s_t$, you take action $a_t$, you get reward $r_{t+1}$, you land in state $s_{t+1}$, and you take action $a_{t+1}$ (according to your current policy, say epsilon-greedy). Then you update $Q(s_t,a_t)$ toward the reward plus the discounted value of what you *actually did* next.

Because SARSA uses the action you *actually took* in the next state, it learns the value of the policy you are following, including its exploration. If your epsilon-greedy policy occasionally walks off a cliff (because it explores), SARSA learns that the cliff-edge state is dangerous — it bakes in the risk of falling.

## Q-Learning: off-policy control

Q-learning looks almost identical to SARSA, with one critical difference:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma \max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right].
$$

Instead of using the action you *actually took* in the next state, Q-learning uses the *best* action — the $\max$. It does not matter what you actually did next; Q-learning assumes you will act optimally from the next state onward.

This makes Q-learning **off-policy**: it learns about the optimal policy regardless of the behavior policy generating the data. You can explore wildly (random actions, curious detours) and Q-learning still converges to $Q^*$, the optimal action-value function, as long as you visit every state-action pair often enough and your step sizes satisfy the right conditions.

The difference is sharp. Imagine a gridworld with a cliff along one edge. SARSA, following epsilon-greedy, learns a *safe* path that stays away from the cliff — because its own exploration sometimes sends it tumbling off. Q-learning learns the *optimal* path right along the cliff edge — because it evaluates the optimal policy, which never falls. Q-learning finds the faster route, but SARSA finds the route that is actually safe given how you are behaving. Neither answer is wrong; they are answering different questions.

[[simulation sarsa-vs-qlearning]]

## Exploration strategies

You need exploration to ensure you visit enough state-action pairs, but you want to exploit what you have learned to collect reward. Three common approaches:

**Epsilon-greedy** is the simplest — with probability $\epsilon$, take a random action; otherwise, take the greedy action. Easy to implement, easy to tune. But it wastes exploration on clearly bad actions.

**Softmax (Boltzmann) exploration** assigns probabilities proportional to $\exp(Q(s,a)/\tau)$, where $\tau$ is a temperature. High temperature means near-uniform exploration; low temperature means near-greedy. It focuses exploration on actions that look promising rather than wasting pulls on obviously bad ones.

**Optimism and UCB-like bonuses** add exploration bonuses to the Q-values, similar to how UCB1 works in bandits. This connects the RL exploration problem back to the bandit ideas from earlier in the course.

## The deadly triad: three bad roommates

There is a stability problem lurking in TD methods, and it becomes critical when we scale up. Imagine three roommates who are perfectly fine individually but, when you put them together, they burn the house down. The three roommates are:

**Function approximation** — instead of storing one value per state in a table, you use a parameterized function (like a neural network) to generalize across states. Perfectly reasonable on its own.

**Bootstrapping** — using your own estimates as targets, as TD does. Works great in tabular settings.

**Off-policy learning** — learning about one policy from data generated by a different policy. Efficient and flexible.

Each one alone is fine. Any two together are usually manageable. But all three together can cause the value estimates to diverge — to spiral upward or oscillate wildly, never converging. The parameters chase their own tail: the target changes because the parameters changed, which changes the target again, in a runaway feedback loop.

This is the **deadly triad**, and it is the central challenge of modern RL. DQN, which we meet next lesson, is essentially an engineering answer to the question: how do we use all three roommates without burning the house down?

[[simulation concentration-bounds]]
[[simulation stochastic-approximation]]

---

*We now have the core toolkit for model-free RL: TD learning, SARSA, and Q-learning. But these all rely on tables — one entry per state-action pair. What happens when the state space is enormous, or continuous? Next lesson, we bring in neural networks and build DQN, the algorithm that played Atari games at superhuman level.*
