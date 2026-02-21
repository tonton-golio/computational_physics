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

## Big Ideas
* Context is the difference between one-size-fits-all and actually useful. The same action can be optimal for one patient and harmful for another; the context is what tells the algorithm which is which.
* Logarithmic dependence on the policy class size is the exponential weights miracle again. It means you can define an astronomically large space of possible policies and still compete with the best one at almost no extra cost.
* Importance weighting is the universal bridge between what you observed and what you needed to know. It appears in Monte Carlo estimation, causal inference, and off-policy evaluation — all for the same reason.
* EXP4 reveals that "learning with experts" and "learning with contexts" are the same problem viewed from different angles: experts are just policies, and policies are just functions from contexts to actions.

## From stateless to stateful decisions

So far, every decision we have made is stateless — the context arrives, we act, we see a reward, and the slate is wiped clean. But many real problems have *state*: the action you take now changes the world you will face next. A robot stepping left is now in a different place; a doctor prescribing drug A has changed the patient's chemistry for tomorrow's decision. When the context depends on your own previous actions, you have a Markov Decision Process, and the tools we have built — regret, optimism, importance sampling — will need to be extended to handle this richer structure.

## What Comes Next

That leap is the subject of the next lesson. In a Markov Decision Process, actions reshape the state of the world, and the state determines future rewards. A bad move today can trap you in bad states for many rounds; a good move can unlock better states later. The next lesson introduces MDPs and dynamic programming: the framework for planning when your decisions have consequences that ripple through time.

## Check Your Understanding
1. In EXP4, each expert proposes a distribution over actions (not just a single action). Why is this generality necessary for the contextual setting, and what would go wrong if experts were forced to be deterministic?
2. Why does EXP4's regret scale with $\sqrt{T \log |\mathcal{E}|}$ rather than $\sqrt{T |\mathcal{E}|}$? Trace back to where the logarithm enters the analysis.
3. Compare the contextual bandit problem to supervised classification. What does the contextual bandit setting have that supervised learning does not, and what does it lack?

## Challenge
EXP4 requires the ability to compute the mixture distribution over actions induced by the expert weights. When the expert class $\mathcal{E}$ is a large function class (e.g., all linear classifiers over a $d$-dimensional context), this computation may be expensive. Describe the computational bottleneck precisely, then propose an approximation scheme (such as epsilon-greedy on the highest-weighted expert, or sampling a single expert to follow) and analyze how it changes the regret bound.
