# Contextual Bandits and EXP4

## The doctor's dilemma

A doctor sees a patient walk through the door. The patient has a specific chart -- age, blood pressure, lab results, symptoms. The doctor has three treatments. She picks one. The patient either gets better or doesn't. She never finds out what would've happened under the other two treatments for *this specific patient*.

That's a contextual bandit problem. The medical chart is the **context**. The treatment is the **action**. The outcome is the **bandit feedback** -- you only see the result of the action you took. But unlike plain bandits, the context gives you leverage. Patients who look similar might tell you which treatment works here.

This is arguably the most practically important setting in this entire course. Personalized ads, adaptive interfaces, recommendation engines, dynamic pricing -- all contextual bandits.

## How EXP4 works

EXP4 stands for Exponential-weight algorithm for Exploration and Exploitation using Expert advice. It's EXP3's big sibling, built for the contextual setting.

You have a pool of experts (or policies), and each one has an opinion about what to do in any given context. When context $x_t$ arrives, each expert proposes a distribution over actions -- "for this patient, I'd prescribe Treatment A with 70% probability and Treatment B with 30%." You mix these expert opinions, weighted by past performance.

Here's the play-by-play: context $x_t$ arrives. Each expert proposes an action distribution. You form a mixture over experts, weighted by performance. You sample an action from the induced distribution. You observe only the loss of the action you chose. You build an importance-weighted loss estimate (same trick as EXP3), and you update expert weights exponentially.

The importance weighting corrects for the fact that you only saw one action's loss. This lets you update *all* expert weights, even though you got feedback for just one action.

## Why the regret bound is magical

EXP4 achieves regret scaling roughly as $\sqrt{T \log |\mathcal{E}|}$, where $\mathcal{E}$ is the expert class (up to constants and the bandit feedback penalty).

Here's why that's remarkable. Got a million experts -- a million possible policies, each mapping contexts to actions differently? The $\log$ term is only about 14. Whether you compete against ten policies or a million, the penalty is gentle. That's the power of exponential weights: logarithmic dependence on the number of experts means you can afford a huge, rich policy class without blowing up the regret.

Compare to trying each expert one at a time. With a million experts, you'd need a million rounds just to try each once. EXP4 finds the best expert in far fewer rounds because it updates all of them in parallel using the importance-weighted feedback trick.

[[simulation contextual-bandit-exp4]]

## What Comes Next

So far, every decision we've made is stateless -- context arrives, we act, we see a reward, and the slate is wiped clean. But many real problems have *state*: the action you take now changes the world you face next. When the context depends on your own previous actions, you have a Markov Decision Process, and the tools we've built -- regret, optimism, importance sampling -- will need to be extended to handle this richer structure.

That leap is the subject of the next lesson. In an MDP, actions reshape the state of the world, and the state determines future rewards. A bad move today can trap you in bad states for many rounds; a good move can unlock better ones. The next lesson introduces MDPs and dynamic programming: the framework for planning when your decisions have consequences that ripple through time.

## Challenge (Optional)

EXP4 requires computing the mixture distribution over actions induced by the expert weights. When the expert class $\mathcal{E}$ is a large function class (e.g., all linear classifiers over a $d$-dimensional context), this gets expensive. Describe the computational bottleneck precisely, then propose an approximation scheme (epsilon-greedy on the highest-weighted expert, or sampling a single expert to follow) and analyze how it changes the regret bound.
