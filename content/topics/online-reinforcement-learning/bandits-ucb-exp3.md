# Stochastic and Adversarial Bandits: UCB1 and EXP3

You're standing in front of $K$ slot machines. Each one has a lever. You get to pull one lever per round, for $T$ rounds total. The machine you pull spits out a reward. The machines you did *not* pull stay silent. Your job: collect as much reward as possible.

This is the multi-armed bandit problem, and it's where exploration meets exploitation head-on.

## UCB1: optimism in the face of uncertainty

Suppose you just moved to a new city and you're hunting for the best restaurant. You've tried the Thai place ten times -- solid 4.5 stars. The Italian place, decent at 3.8. But the sushi place? You went once, it was okay -- 3.5 on that one visit.

Should you go back to Thai? Maybe. But you've been there ten times, so you know what to expect. The sushi place, you've visited once. Maybe that was a bad night. Maybe sushi is actually a 4.8 if you gave it another shot. You don't know because you haven't given it enough chances.

UCB1 handles this with a beautiful principle: **be optimistic about what you don't know**. For each arm $a$, compute an upper confidence bound -- your best estimate plus a bonus for uncertainty:

$$
a_t \in \arg\max_a \left[\widehat{\mu}_a(t) + \sqrt{\frac{2\log t}{n_a(t)}}\right].
$$

Here $\widehat{\mu}_a(t)$ is arm $a$'s empirical mean reward, and $n_a(t)$ is how many times you've pulled it. The square root term is the "optimism bonus" -- big when $n_a(t)$ is small (you haven't tried this arm much) and shrinking as you pull more and get better estimates.

So UCB1 is like the restaurant reviewer who keeps going back to favorites but *also* keeps trying underexplored places, just in case there's a hidden gem. Over time, the bonuses shrink and you naturally settle on the best arm.

The regret depends on the gaps between arms. If arm $a$ has suboptimality gap $\Delta_a$ (how much worse it is than the best), then

$$
R_T = O\!\left(\sum_{a:\Delta_a>0}\frac{\log T}{\Delta_a}\right).
$$

Arms that are almost as good as the best get pulled more before you can distinguish them. Arms that are clearly worse get eliminated fast. The gap-free version gives $O(\sqrt{KT\log T})$.

## EXP3: when the world is adversarial

UCB1 assumes rewards come from fixed distributions. But what if the adversary rewrites the payout tables every night? UCB1 falls apart. You need EXP3.

EXP3 is Hedge's cousin, adapted for bandits. It's like betting on horses, but you only ever see the result of the horse you actually bet on -- so you have to guess how the others would've done.

You maintain weights $w_t(i)$ on each arm, just like Hedge. You build a sampling distribution mixing exploration (trying everything) with exploitation (favoring high-weight arms). Then you pull an arm, observe the loss, and here's the clever part -- you construct an **importance-weighted loss estimate**.

You only saw one arm's loss, but you pretend you can estimate every arm's loss. For the arm you pulled, you take the observed loss and divide by how likely you were to pull it. If you measured a kid you were very likely to pick, the correction is small. If you measured one you almost never pick, the correction is large -- that one measurement carries a lot of weight.

This importance-weighted estimate is unbiased -- on average it gets the right answer -- but it has high variance. The exploration mixing controls that variance by ensuring every arm gets pulled with at least some minimum probability.

The regret of EXP3:

$$
R_T = O(\sqrt{KT\log K}).
$$

The lower bound for adversarial bandits is $\Omega(\sqrt{KT})$, so EXP3 is near-optimal, off by just a $\sqrt{\log K}$ factor. Beautifully tight.

[[simulation multi-armed-bandit]]
[[simulation bandit-regret-comparison]]
[[simulation bernoulli-trials]]

## What Comes Next

UCB1 and EXP3 treat every round as identical -- no side information distinguishes one round from the next. But in practice, you usually have a clue before you act: a user profile, a patient chart, the current state of a system. The next lesson introduces EXP4, which combines EXP3's logic with a pool of context-dependent policies. The key payoff: whether you have ten policies or a million, the regret grows only logarithmically in the policy count.

## Challenge (Optional)

EXP3 achieves $O(\sqrt{KT \log K})$ regret. The lower bound is $\Omega(\sqrt{KT})$. The gap is $\sqrt{\log K}$. Design an argument for why this gap might or might not be closable. Then implement EXP3 on a synthetic adversarial instance where the adversary cycles through arms, and plot cumulative regret against the upper bound. How tight is the bound in practice?
