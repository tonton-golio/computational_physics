# Follow the Leader and Hedge

## Follow the Leader: the obvious idea that fails

The simplest strategy you could imagine: at each round, look at everything so far and pick the action that would've been best. Formally, FTL chooses

$$
a_t \in \arg\min_{a \in \mathcal{A}} \sum_{s=1}^{t-1}\ell_s(a).
$$

In words: you pick the action with the lowest total loss so far. If Expert 2 has been right most often, you follow Expert 2. Sounds perfectly reasonable.

And when losses are well-behaved -- stable, strongly convex -- it actually works. But against an adversary? Disaster. The adversary just flips the best action back and forth. Round 1, Expert A is best. Round 2, Expert B. Round 3, Expert A again. Follow-the-Leader sounds like the smart thing to do... until the adversary starts messing with you. Then it looks like a drunk staggering between lampposts.

FTL's regret grows *linearly* -- the worst possible outcome. The lesson is profound: you need *stability*. You can't just pick the current leader. You need to hedge your bets.

[[simulation ftl-instability]]

## Hedge: the algorithm that hedges

Here's the fix. Instead of following one expert, follow *all of them*, weighted by how much you trust each one. When an expert is wrong, don't drop them -- just reduce their weight. Gently. Multiplicatively.

Setting: $N$ experts, full-information losses $\ell_t(i)\in[0,1]$.

You start with equal weights: $w_1(i)=1$ for every expert. Each round, you build a probability distribution from the weights:

$$
p_t(i)=\frac{w_t(i)}{\sum_j w_t(j)}.
$$

This just says: the probability of following expert $i$ is proportional to their weight. Then you observe all the losses and update:

$$
w_{t+1}(i)=w_t(i)\exp(-\eta \ell_t(i)).
$$

Think of each expert carrying a backpack of confidence. Every time an expert is wrong, you punch a hole in their backpack so they get a little lighter. The parameter $\eta$ controls how big the hole is. Over time, bad experts deflate while good experts stay heavy.

## The learning rate: the gas pedal

The learning rate $\eta$ is like adjusting the gas pedal on a long road trip. Pedal all the way down (large $\eta$)? You react too aggressively to every mistake and overshoot the exit. Barely tap it (tiny $\eta$)? You never get there in time. The sweet spot turns out to be roughly

$$
\eta \asymp \sqrt{\frac{\log N}{T}}.
$$

That's the square root of the log of how many experts you have, divided by the total time horizon. With this choice, regret scales as

$$
R_T = O(\sqrt{T\log N}).
$$

Notice: the dependence on $N$ is only logarithmic. Got a million experts? $\log N$ is about 14. So whether you have 10 experts or a million, Hedge handles it gracefully. The regret grows with $\sqrt{T}$, which is sublinear -- your average regret per round goes to zero. That's the whole trick.

## The potential function: total confidence in the room

The proof that Hedge works revolves around a beautiful idea. Define the **potential function**:

$$
\Phi_t=\sum_i w_t(i).
$$

This is the total weight in the room -- all the confidence backpacks summed up. At the start, $\Phi_1 = N$.

Here's the key insight: $\Phi_t$ can never go below the weight of the best expert, because the best expert's weight is always part of the sum. By tracking how fast $\Phi_t$ shrinks from above (using the average loss and the curvature of the exponential) and how slowly it shrinks from below (because the best expert holds up), you squeeze the regret between two expressions that both come out to $O(\sqrt{T \log N})$.

Think of measuring the temperature of the room. If the room cools fast, everyone is losing. If it cools slowly, the good experts are holding up. The gap between the two cooling rates is the regret.

[[simulation hedge-weights-regret]]

## What Comes Next

Hedge gives us a complete solution for full-information feedback, but real decisions rarely come with a full report card. The next step is the bandit world, where you only see the outcome of the single action you chose.

This silence forces a genuine dilemma: exploiting what you know versus exploring to reduce uncertainty. The next lesson develops UCB1 -- which bets the world is stochastic and uses optimism -- and EXP3, which assumes the world is adversarial and adapts Hedge's exponential weights with an importance-sampling trick. Two completely different philosophies for the same problem.

## Challenge (Optional)

Hedge assumes you know the time horizon $T$ in advance to set $\eta \asymp \sqrt{(\log N)/T}$. Design a "doubling trick" variant that doesn't require knowing $T$: restart Hedge periodically with geometrically increasing horizon lengths. Show that the overall bound remains $O(\sqrt{T \log N})$ up to constants. How much regret does the doubling trick add?
