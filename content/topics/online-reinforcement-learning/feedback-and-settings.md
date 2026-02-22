# Forms of Feedback and Problem Settings

The kind of feedback you get after each decision changes *everything* -- the algorithms you can use, the regret you can achieve, the whole difficulty of the problem. Here's the quick version: full information is like getting the answer key after every test. Bandit is like getting only your own score. Contextual is getting the answer key for students who look like you.

Let's unpack that.

## Full-information feedback

You make a decision, and the world reveals *all* the outcomes -- not just for the action you chose, but for every action you *could* have chosen.

At round $t$, you observe the whole loss vector:

$$
(\ell_t(1), \ell_t(2), \ldots, \ell_t(K)).
$$

In plain terms: you bet on a horse race, and after the race you find out the finishing times of *every* horse. You know exactly how much you would've won or lost on each one. That's a huge amount of information.

This is **prediction with expert advice**. You have $N$ experts, each gives you advice, you decide, and then you see how *all* of them would've done. It's the friendliest version of the online learning problem, and it's where algorithms like Hedge shine. The best achievable regret here scales as $O(\sqrt{T \log N})$ -- that square root is the signature of a no-regret algorithm.

## Bandit (partial) feedback

Now the world gets cruel. You pull a slot-machine lever and the machine only tells you how much you lost on *that* pull. The other levers stay silent. You have no idea what would've happened if you'd pulled lever 3 instead of lever 7. That silence is the enemy.

Formally, you only observe:

$$
\ell_t(a_t).
$$

Just one number. Out of potentially hundreds of actions, you get feedback on exactly one. If you want to know anything about the other actions, you have to *try them* -- which means sacrificing rounds where you could've been exploiting what you already know. That's the exploration-exploitation dilemma, and it's the heartbeat of the bandit problem.

The cost shows up directly in the regret bounds. That $O(\sqrt{T \log N})$ from full information blows up to $O(\sqrt{KT})$ for bandits, where $K$ is the number of arms. The extra $\sqrt{K}$ is the price you pay for flying blind on all the actions you didn't try.

## Contextual feedback

Sometimes, before you decide, the world hands you a clue. A context vector $x_t$ arrives -- a patient's medical chart, a user's browsing history, the current weather -- and you use that clue to pick your action. Then you get bandit feedback: you only see the result of the action you chose, given that context.

You choose $a_t$ using $(x_t, \text{history})$ and receive only $\ell_t(a_t)$.

This is the **contextual bandit** setting, and it's arguably the most practical of the three. A doctor sees a patient (context), prescribes a treatment (action), and observes the outcome (feedback). She never finds out what the other treatments would've done for *this* patient. But the context gives her leverage -- patients who look similar might tell her something useful.

## Complications that make life harder

Beyond these three core settings, several factors crank up the difficulty. **Delayed feedback** means you don't find out the outcome for several rounds. **Noisy feedback** means the loss signal is corrupted. **Non-stationary environments** mean the world shifts beneath your feet -- the best action today might not be the best tomorrow. **Adversarially chosen outcomes** mean someone is actively trying to make you lose.

Each variant changes the concentration arguments, the variance of your estimators, and the final regret bounds. But they all build on the same foundation: the type of feedback determines the type of algorithm.

## What Comes Next

Now that we know what feedback looks like, it's time to build something that uses it. The full-information setting is the friendliest place to start, and it's where the deepest algorithmic ideas first appear.

The next lesson introduces Follow the Leader -- which sounds right but fails badly -- and then Hedge, which fixes FTL's instability using exponential weights. Hedge is the blueprint for virtually every no-regret algorithm that follows, so understanding why it works pays dividends throughout the rest of the topic.
