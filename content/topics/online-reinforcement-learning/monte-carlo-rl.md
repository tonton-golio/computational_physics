# Monte Carlo Methods for RL

## Just play the game and write down the score

Monte Carlo is just "play the game until it ends and write down the score." Sounds stupid until you realize that is exactly how you learned to ride a bike — you did not solve differential equations of balance and angular momentum. You got on, you fell off, you got on again, and eventually your brain figured out the pattern from the accumulated experience of many complete attempts.

Monte Carlo methods estimate values from **complete episodes**. No transition model is required. You do not need to know the probabilities of anything. You just need to be able to play the game, observe the rewards, and remember what happened.

## The core idea

After an episode finishes, you look back at every state you visited and compute the **return** — the total discounted reward from that point forward:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}.
$$

Then you estimate the value of each state by averaging the returns you observed whenever you visited that state:

$$
V^\pi(s)\approx \frac{1}{N(s)}\sum_{t:\,s_t=s} G_t.
$$

That is it. Play many episodes, average the returns for each state, and the law of large numbers does the rest. As the number of episodes grows, your estimates converge to the true values.

## Variants: when to count the visit

*Try it yourself: if a state appears three times in one episode, which variant — first-visit or every-visit — uses more data, and which gives cleaner statistical guarantees? Guess before reading.*

**First-visit MC** only uses the return from the *first* time you visit a state in an episode. If you pass through state $s$ three times in one episode, you only record the return from the first visit. This gives you independent samples (across episodes) and clean convergence guarantees.

**Every-visit MC** uses the return from *every* visit to state $s$ within an episode. You get more data per episode, but the samples within an episode are correlated. Both variants converge to the right answer, but their finite-sample properties differ.

## On-policy vs off-policy

**On-policy MC control** learns about the policy it is actually following. You play episodes using an epsilon-greedy policy (mostly exploit, sometimes explore), compute returns, update action-values, and improve the policy. The exploration keeps you from getting stuck, and the updates keep improving the policy. It is simple and stable.

**Off-policy MC** is trickier but more flexible. You collect episodes using one policy (the behavior policy) and use importance sampling to correct the returns so they reflect a *different* policy (the target policy). This is the same importance-sampling trick we saw in [EXP3's loss estimates](./bandits-ucb-exp3) — correcting for the mismatch between what you observed and what you needed. It is useful when you want to learn about the optimal policy while exploring with a safe, exploratory policy. The importance-sampling correction can have high variance, though, so you need many episodes for the estimates to settle down.

## Trade-offs: why MC is both wonderful and frustrating

Monte Carlo methods have **low bias**. The return targets are computed from actual rewards, not from estimated values. There is no bootstrapping, no approximation in the target. What you see is what you get.

But they have **high variance**, especially when episodes are long. A single episode might wander through many states and collect wildly different rewards depending on the random actions taken. Averaging over many episodes tames this variance, but it can take a lot of episodes.

Monte Carlo is also **naturally episodic**. You have to wait until the episode ends to compute returns. This is fine for games that terminate (chess, Go, a round of poker), but it is a problem for continuing tasks that never end (a robot that runs 24/7). For those, you need temporal-difference methods, which is exactly where we are headed next.

[[simulation monte-carlo-convergence]]

[[simulation blackjack-trajectory]]

## Big Ideas
* Monte Carlo is the law of large numbers applied to sequential decisions. No equations, no model — just averaging enough experience until the truth emerges from the noise.
* Low bias and high variance is the fundamental trade-off. Unbiased targets are expensive: you have to wait for the whole episode to finish and then survive the noise of every random decision along the way.
* The on-policy/off-policy distinction is about whether you are learning what you are doing or what you wish you were doing. Importance sampling bridges the gap but amplifies variance as the two policies diverge.
* Monte Carlo is episodic by nature. If your task never ends, Monte Carlo does not directly apply — which is exactly the pressure that motivates temporal-difference learning.

## What Comes Next

Monte Carlo waits until the episode ends — clean and unbiased, but maddeningly slow. Temporal-difference learning asks: why wait? After a single transition, you can bootstrap — update right away using the next state's estimated value. The next lesson develops TD(0), SARSA, and Q-learning: the backbone of modern model-free RL.

## Check Your Understanding
1. Why does first-visit MC produce unbiased estimates of $V^\pi(s)$, while every-visit MC produces biased estimates within an episode? Does every-visit MC still converge to the correct value asymptotically?
2. Monte Carlo has low bias but high variance. Identify two specific sources of variance in Monte Carlo return estimates and explain how each one grows with episode length.
3. Off-policy MC uses importance sampling to correct for the mismatch between the behavior and target policies. What happens to the variance of the importance-weighted return when the behavior policy is very different from the target policy?

## Challenge
Implement first-visit Monte Carlo control on a simple episodic environment (e.g., Blackjack from Sutton and Barto). Compare the learning curves — state-value estimates as a function of episodes — for epsilon-greedy policies with $\epsilon = 0.1$ and $\epsilon = 0.5$. Then implement off-policy MC with a target policy of always hitting below 18 and sticking at 18 or above, using a uniform random behavior policy. How many episodes does off-policy MC need to match the accuracy of on-policy MC? Explain the result in terms of the importance weights.
