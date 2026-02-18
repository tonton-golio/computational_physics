# Contextual Bandits and EXP4

## The doctor's dilemma

A doctor sees a patient walk through the door. The patient has a specific medical chart — age, blood pressure, lab results, symptoms. The doctor has three possible treatments. She picks one. The patient either gets better or does not. The doctor never finds out what would have happened under the other two treatments for *this specific patient*.

That is a contextual bandit problem. The medical chart is the **context**. The treatment choice is the **action**. The outcome is the **bandit feedback** — you only see the result of the action you took. But unlike plain bandits, the context gives you leverage. Patients who look similar to this one might tell you something about which treatment works here.

This is arguably the most practically important setting in the entire course. Personalized advertising works this way — the user's profile is the context, the ad shown is the action, and whether they click is the feedback. Adaptive interfaces, recommendation engines, medical decision support, dynamic pricing — all contextual bandits.

## How EXP4 works

EXP4 stands for Exponential-weight algorithm for Exploration and Exploitation using Expert advice. It is EXP3's big sibling, designed for the contextual setting.

You have a pool of experts (or policies), and each expert has an opinion about what to do in any given context. When context $x_t$ arrives, each expert proposes a distribution over actions — "for this patient, I would prescribe Treatment A with 70% probability and Treatment B with 30%." The learner then mixes these expert opinions, weighted by how well each expert has done so far.

Here is the step-by-step: the context $x_t$ arrives. Each expert $e$ proposes an action distribution for this context. You form a mixture distribution over experts, weighted by their accumulated performance. You sample an action from the induced distribution over actions. You observe only the loss of the action you chose. You build an importance-weighted loss estimate (just like EXP3), and you update the expert weights exponentially.

The importance weighting is the same trick as before — you correct for the fact that you only observed one action's loss by dividing by the probability of having chosen that action. This lets you update *all* expert weights, even though you only saw feedback for one action.

## Why the regret bound is magical

EXP4 achieves regret that scales roughly with $\sqrt{T \log |\mathcal{E}|}$, where $\mathcal{E}$ is the expert class (up to constants and the bandit feedback penalty).

Here is why that is remarkable. If you have a million experts — a million possible policies, each one mapping contexts to actions differently — the $\log$ term is only about 14. So you are basically paying the bandit price times $\sqrt{14}$. Whether you compete against ten policies or a million, the penalty is gentle. That is the power of exponential weights: the logarithmic dependence on the number of experts means you can afford to have a huge, rich policy class without blowing up the regret.

Compare this to what would happen if you tried each expert one at a time. With a million experts, you would need a million rounds just to try each one once. EXP4 finds the best expert in far fewer rounds because it updates all of them in parallel using the importance-weighted feedback trick.

[[simulation contextual-bandit-exp4]]

---

*We have now covered the full spectrum of online learning, from full-information to bandits to contextual bandits. But in all these settings, the world had no memory — your action today did not change what happens tomorrow. Next lesson, everything changes. We enter the world of Markov Decision Processes, where your actions shape the future state of the world. That is where reinforcement learning really begins.*
