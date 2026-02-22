# Monte Carlo Methods for RL

## Just play the game and write down the score

Monte Carlo is "play the game until it ends and write down the score." Sounds stupid. But that's exactly how you learned to ride a bike -- you didn't solve differential equations of balance and angular momentum. You got on, fell off, got on again, and eventually your brain figured out the pattern from accumulated experience.

Monte Carlo methods estimate values from **complete episodes**. No model required. You don't need to know any probabilities. Just play the game, observe rewards, and remember what happened.

## The core idea

After an episode finishes, you look back at every state you visited and compute the **return** -- the total discounted reward from that point forward:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}.
$$

That's the sum of rewards from time $t$ onward, each multiplied by $\gamma^k$ so future rewards count a bit less. Then you estimate the value of each state by averaging the returns you observed:

$$
V^\pi(s)\approx \frac{1}{N(s)}\sum_{t:\,s_t=s} G_t.
$$

That's it. Play many episodes, average the returns for each state, and the law of large numbers does the rest.

## First-visit vs every-visit

**First-visit MC** only uses the return from the *first* time you visit a state in an episode. Three visits to state $s$? You only record the first. This gives independent samples (across episodes) and clean convergence guarantees.

**Every-visit MC** uses the return from *every* visit to $s$ within an episode. More data per episode, but samples within an episode are correlated. Both converge to the right answer.

## On-policy vs off-policy

**On-policy MC** learns about the policy it's actually following. Play episodes with epsilon-greedy (mostly exploit, sometimes explore), compute returns, update action-values, improve the policy. Simple and stable.

**Off-policy MC** collects episodes using one policy (behavior) and importance-samples the returns to learn about a *different* policy (target). Same trick we saw in [EXP3](./bandits-ucb-exp3) -- correcting for the mismatch between what you observed and what you needed. Useful when you want to learn about the optimal policy while exploring safely. But importance-sampling corrections can have high variance, so you need many episodes.

## Trade-offs

Monte Carlo has **low bias**. The return targets come from actual rewards, not estimated values. No bootstrapping, no approximation. What you see is what you get.

But it has **high variance**, especially with long episodes. A single episode can wander through states and collect wildly different rewards depending on random actions. Averaging over episodes tames the variance, but it takes a lot of episodes.

Monte Carlo is also **naturally episodic**. You have to wait until the episode ends to compute returns. Fine for games that terminate (chess, poker), but a problem for continuing tasks that never end. For those, you need temporal-difference methods -- which is exactly where we're headed.

[[simulation monte-carlo-convergence]]

[[simulation blackjack-trajectory]]

## What Comes Next

Monte Carlo waits until the episode ends -- clean and unbiased, but maddeningly slow. Temporal-difference learning asks: why wait? After a single transition, you can bootstrap -- update right away using the next state's estimated value. The next lesson develops TD(0), SARSA, and Q-learning: the backbone of modern model-free RL.

## Challenge (Optional)

Implement first-visit Monte Carlo control on Blackjack. Compare learning curves for $\epsilon = 0.1$ vs $\epsilon = 0.5$. Then implement off-policy MC with a target policy of always hitting below 18 and sticking at 18+, using a uniform random behavior policy. How many episodes does off-policy need to match on-policy accuracy? Explain the result in terms of the importance weights.
