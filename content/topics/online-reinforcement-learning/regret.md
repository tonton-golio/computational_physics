# The Notion of Regret

## The weather forecaster story

Imagine you're a weather forecaster. Every morning you announce "Rain" or "Sunny." You've got five colleagues -- five experts -- each giving you their prediction. You don't know who's good. You don't know the weather patterns. All you can do is listen, pick one prediction, and wait to see what actually happens.

After a thousand mornings, you look back. One expert -- let's call her Expert 3 -- was right 820 times. You, mixing and matching their advice, were right 790 times. Those 30 extra mistakes compared to Expert 3? That's your **regret**.

And here's the gorgeous part. You didn't know Expert 3 would be best. You had to figure it out as you went. Yet your regret was only 30 out of 1000 rounds. Your average per-round regret was essentially zero. That's what we're chasing: strategies that keep regret small no matter what the world throws at you.

## External vs internal regret

Before we go further, there are two flavors of regret you should know about, because they ask very different questions.

**External regret** compares you against the best single action in hindsight. "If I'd just stuck with Expert 3 the whole time, how much better off would I be?" That's the definition we'll use most.

**Internal (swap) regret** asks something harder. For every action $a$ you played, would you have done better by swapping it with some other action $a'$? No-internal-regret algorithms are more powerful and connect to game-theoretic equilibria. Think of it this way: external regret says "I wish I'd always picked chocolate." Internal regret says "Every time I picked vanilla, I wish I'd picked chocolate, and every time I picked strawberry, I wish I'd picked vanilla." It's a finer kind of second-guessing.

## The formal definition

Here's how we write external regret over $T$ rounds:

$$
R_T = \sum_{t=1}^{T} \ell_t(a_t) - \min_{a \in \mathcal{A}} \sum_{t=1}^{T} \ell_t(a).
$$

The first sum is your cumulative loss -- every bad call you made. The second term is the cumulative loss of the single best action in hindsight. The difference is regret: how much worse you did than the best fixed strategy, looking back with perfect knowledge.

If you plot two curves -- your loss growing over time and the best expert's loss growing over time -- the shaded area between them is your regret.

[[simulation regret-growth-comparison]]

## Sublinear regret: why it means you're winning

**Sublinear regret** means $R_T = o(T)$. In plain English: your regret grows, but slower than the number of rounds. Play for a thousand rounds with regret 30? That's 3% per round. Play for a million rounds with regret 1000? That's 0.1% per round. The longer you play, the closer your average performance gets to the best expert.

And here's the kicker -- you don't need any i.i.d. assumption. The data doesn't have to come from a nice distribution. The losses can be chosen by an adversary who's actively trying to mess you up, and you can *still* guarantee sublinear regret. That's a remarkable guarantee.

## Variants of regret

A few flavors worth knowing:

**Expected regret**, $\mathbb{E}[R_T]$, averages over randomness in your algorithm. Most common in stochastic settings.

**High-probability regret** holds with probability at least $1-\delta$. You pay a small log factor in $1/\delta$, but you get a guarantee that works on any single run, not just on average.

**Pseudo-regret** shows up in stochastic bandits. Instead of comparing against the best arm's *realized* loss sequence, you compare against its *expected* loss. Slightly weaker benchmark, but often easier to analyze.

## Adversarial vs stochastic: two different worlds

In **stochastic bandits**, losses come from fixed distributions. You're pulling slot-machine levers, each with a true average payout that doesn't change. The difficulty is that you don't know these averages.

In **adversarial online learning**, the losses can be anything. An adversary picks them, possibly after seeing your strategy. No distributional assumptions at all. Harder setting, weaker guarantees -- but more robust.

The minimax lower bounds tell you the price of feedback. With full information (you see every expert's loss each round), the best possible regret is $\Omega(\sqrt{T \log N})$ where $N$ is the number of experts. With bandit feedback (you only see the loss of the arm you pulled), it jumps to $\Omega(\sqrt{KT})$ where $K$ is the number of arms. That extra $\sqrt{K}$? That's the tax you pay for partial feedback. The square-root law is the best you can hope for -- no matter how clever you are.

## Practical examples

Think of a portfolio manager who invests in stocks every day. At year's end, she compares her cumulative return to the single best stock she could have held all year. Her regret is the difference. A no-regret strategy guarantees she's not much worse than that best-in-hindsight stock, even though she couldn't predict the market.

Or think of an email system routing messages to folders using a set of rules (experts). After a year, you compare accuracy to the single best rule. Regret measures the gap, and a good algorithm makes it vanishingly small.

## What Comes Next

The regret framework tells us *what to measure*, but nothing yet about *what information* we get to make decisions with. Before we can design any algorithm, we need to nail down the feedback model: does the world show you every outcome, or only the one you chose?

That question -- full information versus bandit versus contextual -- determines almost everything: which algorithms are possible, how fast regret can shrink, and what exploration costs you. The next lesson maps out that landscape.
