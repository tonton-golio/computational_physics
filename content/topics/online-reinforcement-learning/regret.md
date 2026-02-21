# The Notion of Regret

## The weather forecaster story

Imagine you are a weather forecaster, and every morning you have to announce "Rain" or "Sunny." You have access to five colleagues — five experts — who each give you their prediction. You do not know who is good. You do not know the weather patterns. All you can do is listen to them, pick one prediction, and then wait to see what actually happens.

After a thousand mornings, you look back. One of those five experts — let us call her Expert 3 — turned out to be right 820 times. You, following your strategy of mixing and matching their advice, were right 790 times. The gap — those 30 extra mistakes you made compared to Expert 3 — that is your **regret**.

Now here is the beautiful part. You did not know in advance that Expert 3 would be the best. You had to figure it out as you went. And yet your regret was only 30 out of 1000 rounds. That is a tiny fraction. Your average per-round regret was essentially zero. That is what we are after in this course: strategies that guarantee your regret stays small, no matter what the world throws at you.

## The formal definition

For losses, we write external regret over horizon $T$ as

$$
R_T = \sum_{t=1}^{T} \ell_t(a_t) - \min_{a \in \mathcal{A}} \sum_{t=1}^{T} \ell_t(a).
$$

The first sum is your cumulative loss — every bad call you made. The second term is the cumulative loss of the single best action in hindsight. The difference is regret: how much worse you did than the best fixed strategy, looking back with perfect knowledge.

If you plot two curves — your loss growing over time and the best expert's loss growing over time — the shaded area between them is your regret. A good algorithm keeps that shaded area from growing too fast.

[[simulation regret-growth-comparison]]

## Sublinear regret: why it means you are winning

**Sublinear regret** means $R_T = o(T)$. In plain language: your regret grows, but it grows slower than the number of rounds. If you play for a thousand rounds and your regret is only 30, that is 3% per round. If you play for a million rounds and your regret is only 1000, that is 0.1% per round. The longer you play, the closer your average performance gets to the best expert. If you are only a tiny bit worse each day, over a thousand days you still look like a genius.

This is why regret replaces the usual "generalization error" from classical machine learning. You do not need any i.i.d. assumption. You do not need the data to come from a nice distribution. The losses can be chosen by an adversary who is actively trying to mess you up, and you can *still* guarantee sublinear regret. That is a remarkable guarantee.

## Variants of regret

There are several flavors of regret, and it helps to know which one you are working with.

**Expected regret**, $\mathbb{E}[R_T]$, averages over any randomness in your algorithm. This is the most common version in stochastic settings.

**High-probability regret** gives bounds that hold with probability at least $1-\delta$. You pay a small price in the bound (usually a log factor in $1/\delta$), but you get a guarantee that works on any single run, not just on average.

**Pseudo-regret** shows up in stochastic bandits. Instead of comparing against the best fixed arm's *realized* loss sequence, you compare against its *expected* loss. It is a slightly weaker benchmark but often easier to analyze.

**External regret** vs **internal (swap) regret**: external regret compares you against the best single action. Internal regret asks a harder question — for every action $a$ you played, would you have done better by swapping it with some other action $a'$? No-internal-regret algorithms are more powerful and connect to game-theoretic equilibria.

## Adversarial vs stochastic: two different worlds

In **stochastic bandits**, the losses or rewards come from fixed distributions. You are pulling slot-machine levers, and each lever has a true average payout that does not change. The difficulty is that you do not know these averages.

In **adversarial online learning**, the losses can be anything. An adversary picks the loss sequence, possibly after seeing your strategy. No distributional assumptions at all. This is the harder setting, and the guarantees you get are necessarily weaker — but they are also more robust.

The minimax lower bounds tell you the price of feedback. With full information (you see every expert's loss after each round), the best possible regret is $\Omega(\sqrt{T \log N})$ where $N$ is the number of experts. With bandit feedback (you only see the loss of the arm you pulled), it jumps to $\Omega(\sqrt{KT})$ where $K$ is the number of arms. That extra $\sqrt{K}$ is the tax you pay for partial feedback.

## Practical examples

Think of a portfolio manager who invests in stocks every day. At the end of the year, she looks back and compares her cumulative return to the single best stock she could have held all year. Her regret is the difference. A no-regret strategy guarantees she is not much worse than that best-in-hindsight stock, even though she could not predict the market.

Or think of an email system that routes your messages to different folders using a set of rules (experts). After a year of routing, you compare the system's accuracy to the single best routing rule. Regret measures the gap, and a good algorithm makes that gap vanishingly small over time.

## Big Ideas
* Regret is not about making mistakes — it is about making mistakes you could have avoided. The benchmark is always the best fixed strategy in hindsight, not perfection.
* Sublinear regret means your per-round error vanishes over time, even against an adversary who knows your algorithm. You do not need a cooperative world to learn well.
* The price of feedback is real and quantified: full information gives $O(\sqrt{T \log N})$ regret while bandit feedback costs an extra $\sqrt{K}$. That gap is the price of silence.
* External and internal regret measure different kinds of second-guessing. No-internal-regret is strictly harder and connects to game-theoretic equilibria.

## What Comes Next

The regret framework tells us *what to measure*, but nothing yet about *what information* we get to make decisions with. Before we can design any algorithm, we need to be precise about the feedback model: does the world show you every outcome, or only the one you chose?

That question — full information versus bandit versus contextual — turns out to determine almost everything: which algorithms are possible, how fast regret can shrink, and what exploration costs you. The next lesson maps out that landscape so every algorithm we build afterward has a clear place to stand.

## Check Your Understanding
1. A strategy achieves regret $R_T = 50\sqrt{T}$ after $T$ rounds. Is this sublinear? What does it say about the average per-round regret as $T \to \infty$?
2. Explain why comparing yourself to the "best fixed action in hindsight" is a meaningful benchmark even though that action is chosen after seeing all the data.
3. Why does adversarial bandit feedback produce a regret lower bound of $\Omega(\sqrt{KT})$ while full-information feedback only requires $\Omega(\sqrt{T \log N})$? What structural property drives that difference?

## Challenge
Design a scenario — with a specific number of experts and a specific adversary strategy — where Follow the Leader achieves linear regret. Then show analytically that your scenario does in fact produce $R_T = \Theta(T)$. What is the minimum number of experts needed to construct the adversarial sequence, and why?
