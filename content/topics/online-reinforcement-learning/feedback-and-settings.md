# Forms of Feedback and Problem Settings

The kind of feedback you receive after each decision changes *everything* — the algorithms you can use, the regret you can achieve, and the difficulty of the problem. This lesson is about understanding those different feedback models, because everything else in the course is just different flavors of the same underlying pain.

## Full-information feedback

In the full-information setting, you make a decision and then the world reveals *all* the outcomes — not just for the action you chose, but for every action you *could* have chosen.

At round $t$, you observe the whole loss vector:

$$
(\ell_t(1), \ell_t(2), \ldots, \ell_t(K)).
$$

Think of it this way: you are betting on a horse race. After the race, you find out not only how your horse did, but the finishing times of *every* horse. You know exactly how much you would have won or lost on each one. That is a huge amount of information.

This is the setting we call **prediction with expert advice**. You have $N$ experts, each one gives you advice, you make a decision, and then you see how *all* of them would have done. It is the friendliest version of the online learning problem, and it is where algorithms like Hedge shine. The regret you can achieve here scales as $O(\sqrt{T \log N})$ — the square root is the signature of a no-regret algorithm.

## Bandit (partial) feedback

Now the world gets cruel. You pull the slot-machine lever and the machine only tells you how much you lost on *that* pull. The other levers stay silent. You have no idea what would have happened if you had pulled lever 3 instead of lever 7. That silence is the enemy.

Formally, you only observe:

$$
\ell_t(a_t).
$$

Just one number. Out of potentially hundreds or thousands of actions, you get feedback on exactly one. If you want to know anything about the other actions, you have to *try them*, which means sacrificing rounds where you could have been exploiting what you already know. This is the exploration-exploitation dilemma, and it is the heartbeat of the bandit problem.

The cost of partial feedback shows up directly in the regret bounds. That $O(\sqrt{T \log N})$ from the full-information setting blows up to $O(\sqrt{KT})$ for bandits, where $K$ is the number of arms. The extra $\sqrt{K}$ factor is the price you pay for flying blind on all the actions you did not try. You have to explicitly spend rounds exploring, and every exploration round is a round where you might be losing more than you had to.

## Contextual feedback

Sometimes, before you make your decision, the world hands you a clue. A context vector $x_t$ arrives — think of it as a patient's medical chart, or a user's browsing history, or the current weather conditions — and you use that clue to pick your action. Then you get bandit feedback: you only see the outcome of the action you chose, given that context.

You choose $a_t$ using $(x_t, \text{history})$ and receive only $\ell_t(a_t)$.

This is the **contextual bandit** setting, and it is arguably the most practical of the three. A doctor sees a patient (context), prescribes a treatment (action), and observes the outcome (feedback). She never finds out what the other treatments would have done for *this* patient. But the context gives her leverage — patients who look similar to this one might tell her something useful.

## The spectrum of difficulty

These three settings — full information, bandit, contextual — form a spectrum from easy to hard. In the full-information setting, learning is relatively cheap because you see all the outcomes. In the bandit setting, learning is expensive because you see only one outcome and have to explore. In the contextual setting, you have the bandit's pain but the context's help, and the balance between them determines your regret.

There are additional complications that make the problem harder still. **Delayed feedback** means you do not find out the outcome for several rounds. **Noisy feedback** means the loss signal is corrupted. **Non-stationary environments** mean the world changes beneath your feet — the best action today might not be the best action tomorrow. **Adversarially chosen outcomes** mean someone is actively trying to make you lose.

Each of these variants changes the concentration arguments you can use, the variance of your estimators, and the final regret bounds. But they all build on the same foundation: the type of feedback determines the type of algorithm.

---

*We now understand the three feedback models that govern online learning. Next lesson, we are going to build our first real algorithm — Follow the Leader and its much smarter cousin, Hedge — and see how exponential weights can achieve no-regret guarantees in the full-information setting. That is our first weapon.*
