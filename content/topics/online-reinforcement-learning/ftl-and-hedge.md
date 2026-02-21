# Follow the Leader and Hedge

## Follow the Leader: the obvious idea that fails

The simplest strategy you could imagine is: at each round, look at everything that has happened so far and pick the action that would have been best. Formally, FTL chooses

$$
a_t \in \arg\min_{a \in \mathcal{A}} \sum_{s=1}^{t-1}\ell_s(a).
$$

In words: you pick the action with the lowest total loss so far. It sounds perfectly reasonable. If expert 2 has been right most often, you follow expert 2.

FTL is simple and parameter-free, and when the losses are well-behaved (think strongly convex or stable), it actually works. But here is the problem: against an adversary, FTL can be a disaster. The adversary just keeps flipping the best action back and forth. Round 1, expert A is best. Round 2, expert B is best. Round 3, expert A again. FTL chases back and forth like a dog chasing two squirrels, and its regret grows *linearly* — which is the worst possible outcome.

[[simulation ftl-instability]]

The lesson from FTL is profound: you need *stability*. You cannot just pick the current leader — you need to hedge your bets.

## Hedge: the algorithm that hedges

Now we build the algorithm that fixes FTL's problem. Imagine you are in a classroom with $N$ students (experts), and each student raises their hand to vote on the answer. At first, you trust everyone equally. But after each round, when you see who was right and who was wrong, you adjust your trust.

Here is the key insight: instead of following one expert, you follow *all of them*, weighted by how much you trust each one. And when an expert is wrong, you do not drop them entirely — you just reduce their weight. Gently. Multiplicatively.

Setting: $N$ experts, full-information losses $\ell_t(i)\in[0,1]$.

You start with equal weights: $w_1(i)=1$ for every expert. At each round, you choose a probability distribution over experts based on their weights:

$$
p_t(i)=\frac{w_t(i)}{\sum_j w_t(j)}.
$$

Then you observe all the losses and update:

$$
w_{t+1}(i)=w_t(i)\exp(-\eta \ell_t(i)).
$$

Think of each expert as carrying a backpack of confidence. Every time an expert is wrong, you punch a hole in their backpack so they get a little lighter. But you do it gently, controlled by the parameter $\eta$. A large loss means a bigger hole, and over time the bad experts' backpacks deflate while the good experts stay heavy.

## The learning rate: adjusting the gas pedal

*Try it yourself: what happens to Hedge weights if $\eta$ is too large? Pause and guess before the math.*

We need a learning rate $\eta$ that works like adjusting the gas pedal on a long road trip. If you keep the pedal all the way down (large $\eta$), you react too aggressively to every mistake — you overshoot the exit. If you barely tap the pedal (tiny $\eta$), you never get there in time. The sweet spot turns out to be roughly

$$
\eta \asymp \sqrt{\frac{\log N}{T}}.
$$

That is the square root of the log of how many experts you have, divided by the total time horizon. With this choice, regret scales as

$$
R_T = O(\sqrt{T\log N}).
$$

Notice that the dependence on $N$ is only logarithmic. If you have a million experts, $\log N$ is about 14. So whether you have 10 experts or a million, Hedge handles it gracefully. The regret grows with $\sqrt{T}$, which is sublinear — your average regret per round goes to zero.

## The potential function: total confidence in the room

The proof that Hedge works revolves around a beautiful idea called the **potential function**. Define

$$
\Phi_t=\sum_i w_t(i).
$$

This is the total weight in the room — the sum of all the confidence backpacks. At the start, $\Phi_1 = N$ because everyone starts with weight 1.

Now think about what happens each round. When you apply the exponential update, the total weight changes. If most experts did well (small losses), the total weight barely drops. If everyone did badly, it drops a lot. The key is that $\Phi_t$ can never go below the weight of the best expert — because the best expert's weight is always part of the sum.

By tracking how fast $\Phi_t$ shrinks from above (using the average loss and a second-order correction from the curvature of the exponential) and how slowly it shrinks from below (because the best expert's weight barely changes), you can sandwich the regret between two expressions that both come out to $O(\sqrt{T \log N})$.

The potential function is like measuring the temperature of the room. If the room is getting cold fast, it means everyone is losing. If it is cooling slowly, the good experts are holding up. The gap between the two rates of cooling is the regret.

[[simulation hedge-weights-regret]]

## Big Ideas
* FTL fails not because greediness is wrong, but because pure greediness has no inertia — a tiny adversarial nudge can whipsaw the entire decision.
* Exponential weights are the right answer to instability: multiplicative updates smooth out the influence of any single bad round, and the logarithm in the regret bound reflects exactly how many experts you are hedging across.
* The potential function proof is not a trick — it is a way of seeing that the total weight in the room is squeezed between two bounds that both yield $O(\sqrt{T \log N})$.
* The learning rate $\eta$ mediates a bias-variance trade-off across time: too large and you overreact to noise; too small and you adapt too slowly to identify the best expert.

## What Comes Next

Hedge gives us a complete solution for the full-information setting, but real decisions rarely come with a full report card after every round. The next step is the bandit world, where you only see the outcome of the single action you chose — the other arms stay silent.

This silence forces a genuine dilemma: exploiting what you already know versus exploring to reduce uncertainty. The next lesson develops two algorithms that navigate this dilemma from opposite starting assumptions. UCB1 bets that the world is stochastic and uses optimism to balance exploration. EXP3 assumes the world is adversarial and adapts Hedge's exponential weights with an importance-sampling trick to handle the missing feedback.

## Check Your Understanding
1. Construct a two-expert example where FTL achieves linear regret. What property of the adversary's loss sequence causes FTL to keep switching?
2. In Hedge, why does the regret scale as $O(\sqrt{T \log N})$ rather than $O(\sqrt{TN})$? What role does the exponential weight update play in achieving logarithmic dependence on $N$?
3. If you set $\eta$ too large in Hedge — say, $\eta = 1$ regardless of $T$ and $N$ — what goes wrong? Describe the failure mode in terms of what happens to the weight distribution.

## Challenge
Hedge assumes you know the time horizon $T$ in advance in order to set $\eta \asymp \sqrt{(\log N)/T}$. Design a "doubling trick" variant of Hedge that does not require knowing $T$ ahead of time: restart Hedge periodically with geometrically increasing horizon lengths. Analyze how much regret the doubling trick adds relative to the oracle-tuned version, and show that the overall bound remains $O(\sqrt{T \log N})$ up to constants.
