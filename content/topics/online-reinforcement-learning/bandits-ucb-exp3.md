# Stochastic and Adversarial Bandits: UCB1 and EXP3

You are standing in front of $K$ slot machines. Each one has a lever. You get to pull one lever per round, for $T$ rounds total. The machine you pull spits out a reward (or a loss). The machines you did *not* pull stay silent. Your job: collect as much reward as possible.

This is the multi-armed bandit problem, and it is the place where exploration meets exploitation head-on.

## UCB1: optimism in the face of uncertainty

Imagine you have just moved to a new city, and you are trying to find the best restaurant. You try a few places. The Thai place was great — 4.5 stars in your personal rating. The Italian place was decent — 3.8 stars. But there is a sushi restaurant you have only been to once, and it was okay — 3.5 stars on that one visit.

Should you go back to the Thai place? Maybe. But here is the thing: you have been to the Thai place ten times, so you have a pretty good estimate of how good it is. The sushi place, you visited only once. Maybe that one visit was a bad night. Maybe the sushi place is actually a 4.8 if you went back. You do not know because you have not given it enough chances.

UCB1 handles this dilemma with a beautiful principle: **be optimistic about what you do not know**. For each arm $a$, you compute an upper confidence bound — your best estimate of that arm's average reward, plus a bonus for how uncertain you are:

$$
a_t \in \arg\max_a \left[\widehat{\mu}_a(t) + \sqrt{\frac{2\log t}{n_a(t)}}\right].
$$

Here $\widehat{\mu}_a(t)$ is the empirical mean reward of arm $a$ so far, and $n_a(t)$ is how many times you have pulled it. The square root term is the "optimism bonus" — it is big when $n_a(t)$ is small (you have not tried this arm much) and shrinks as you pull the arm more and get a better estimate.

So UCB1 is like the restaurant reviewer who goes back to the good places but *also* keeps trying the underexplored ones, just in case there is a hidden gem. Over time, the bonus terms shrink, and you naturally settle on the best arm.

The regret of UCB1 depends on the gaps between the arms. If arm $a$ has a suboptimality gap $\Delta_a$ (how much worse it is than the best arm), then the gap-dependent regret is $O\!\left(\sum_{a:\Delta_a>0}\frac{\log T}{\Delta_a}\right)$. Arms that are almost as good as the best get pulled more before you can distinguish them. Arms that are clearly worse get eliminated quickly. The gap-free version, which does not assume you know the gaps, gives $O(\sqrt{KT\log T})$.

## EXP3: when the world is adversarial

UCB1 relies on a critical assumption: the rewards come from fixed distributions. But what if the adversary can change the payoffs each round? What if the slot machines are rigged, and someone is rewriting the payout tables every night?

That is the adversarial bandit setting, and UCB1 falls apart. You need EXP3 — the exponential-weight algorithm for exploration and exploitation.

EXP3 is Hedge's cousin, adapted for the bandit world. Remember Hedge? It worked in the full-information setting because you could see all the losses. In the bandit setting, you only see the loss of the arm you pulled. So you have to *estimate* the losses of the arms you did not pull. Here is how:

You maintain weights $w_t(i)$ on each arm, just like Hedge. You build a sampling distribution that mixes exploration (trying everything) with exploitation (favoring high-weight arms). Then you pull an arm, observe the loss, and here comes the clever part — you construct an importance-weighted loss estimate.

You only see the loss of the arm you pulled, but you pretend you can estimate the loss of every arm. For the arm you did pull, you take the observed loss and divide it by how likely you were to pull that arm. It is like correcting for the fact that you only measured one kid's height but want to know the class average. If you measured a kid you were very likely to measure, the correction is small. If you measured a kid you almost never pick, the correction is large — that one measurement carries a lot of weight.

This importance-weighted estimate is unbiased — on average it gets the right answer — but it has high variance. The exploration mixing helps control that variance by ensuring you pull every arm with at least some minimum probability.

The regret of EXP3 scales as

$$
R_T = O(\sqrt{KT\log K}).
$$

The lower bound for adversarial bandits is $\Omega(\sqrt{KT})$, so EXP3 is near-optimal, off by just a $\sqrt{\log K}$ factor. That is a beautifully tight result.

[[simulation multi-armed-bandit]]
[[simulation bandit-regret-comparison]]
[[simulation bernoulli-trials]]

## Big Ideas
* UCB1's "optimism in the face of uncertainty" is not just a heuristic — it is the right Bayesian instinct. Treat unknown arms as potentially great, try them, and let data correct you.
* The gap-dependent regret bound reveals a truth: pulling an arm that is only slightly suboptimal is the expensive part, because you need many samples to distinguish it from the best.
* EXP3's importance-weighted loss estimate is an act of statistical imagination: you never saw the loss of the arm you did not pull, so you construct an unbiased proxy. The noise is the cost of that imagination.
* Stochastic and adversarial bandits require completely different algorithms. Importing UCB1 into an adversarial environment, or using EXP3 in a stochastic one, leaves significant regret on the table.

## What Comes Next

Both UCB1 and EXP3 treat every round as if it were identical — there is no side information that distinguishes one round from the next. But in almost every real application, you do have a clue before you act: a user profile, a patient chart, the current state of a system.

Adding that context transforms the bandit problem into something richer. The next lesson introduces EXP4, which combines the exploration-exploitation logic of EXP3 with a pool of policies that each map contexts to action distributions. The key payoff: whether you have ten policies or a million, the regret only grows logarithmically in the policy count.

## Check Your Understanding
1. UCB1 uses the bonus term $\sqrt{2 \log t / n_a(t)}$. What happens to this term as $n_a(t) \to \infty$? What happens when $n_a(t) = 1$? Interpret both limits intuitively.
2. Explain why EXP3 divides the observed loss by the sampling probability in its importance-weighted estimate. What property does this give the estimate, and why is that property essential?
3. Why does the stochastic bandit setting permit gap-dependent regret bounds of the form $O(\sum_a \log T / \Delta_a)$, while the adversarial setting only permits $O(\sqrt{KT \log K})$? What assumption does the stochastic setting have that the adversarial one lacks?

## Challenge
EXP3 achieves $O(\sqrt{KT \log K})$ regret. A lower bound shows adversarial bandits require $\Omega(\sqrt{KT})$ regret. The gap is $\sqrt{\log K}$. Design an argument (or find one in the literature) for why this gap might or might not be closable. Then implement EXP3 computationally on a synthetic adversarial instance where the adversary cycles through arms, and plot the cumulative regret against the $O(\sqrt{KT \log K})$ upper bound. How tight is the bound in practice?
