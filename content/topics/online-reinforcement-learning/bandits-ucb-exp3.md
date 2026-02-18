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

---

*We now have two powerful bandit algorithms — UCB1 for stochastic worlds and EXP3 for adversarial ones. But both of them treat every round the same. What if you get a clue before each decision — a context that tells you something about the current situation? Next lesson, we add context to the bandit problem with EXP4, and suddenly the problem starts looking like real-world personalization.*
