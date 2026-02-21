# Online and Reinforcement Learning

## Why are we even here?

Picture two weather forecasters. Every single morning, Forecaster A walks to the microphone and says "Rain." Every single morning, Forecaster B says "Sunny." Neither of them looks out the window. Neither of them checks the data. They just repeat themselves, day after day, for a thousand days.

Now here is the question that launches this entire course: at the end of those thousand days, which forecaster was better? You do not know. You *cannot* know until you look at what actually happened. If it rained 900 out of 1000 days, Forecaster A looks like a genius. If the sun shone 900 days, Forecaster B wins.

But what if *you* could do something smarter? What if, each morning, you looked at the track records of both forecasters — and all their friends — and weighted your prediction toward whoever had been right most often? You would start off clueless, but day by day your cumulative mistakes would grow *slower* than the best single forecaster's mistakes. The gap between your total loss and the best expert's total loss — that gap is called **regret**. And making that gap grow as slowly as possible is the entire game of online learning.

That is the thread running through everything in this module. We start with regret, we learn algorithms that minimize it, we move from simple expert advice to slot machines where you only see one outcome at a time, and then we enter the full world of reinforcement learning where your actions change the world around you. Every lesson builds on the last. Every idea comes back to this: how do you make good decisions, one at a time, when you do not know what is coming next?

## Map of the course

Think of this module as a journey through four territories, each one richer and harder than the last:

**Online Learning** (Lessons 1–3): You meet regret, you learn about different kinds of feedback, and you build your first algorithm (Hedge) that plays the expert-advice game with provably small regret.

**Bandits** (Lessons 4–5): Now the world gets meaner. You only see the outcome of the action you chose — the other options stay silent. You learn UCB1, EXP3, and EXP4, algorithms that balance exploration with exploitation.

**Full Reinforcement Learning** (Lessons 6–8): The world gains memory. Your actions change future states. You learn MDPs, Monte Carlo methods, TD learning, SARSA, and Q-learning — the core toolkit for sequential decision-making.

**Deep RL and Continuing Tasks** (Lessons 9–10): State spaces explode. You cannot keep a lookup table anymore, so you bring in neural networks. Then you ask: what happens when the game never ends?

```
Online Learning ──→ Bandits ──→ Full RL ──→ Deep RL & Continuing Tasks
 (Lessons 1-3)    (Lessons 4-5)  (Lessons 6-8)    (Lessons 9-10)
  Regret            UCB1          MDPs              DQN
  Feedback          EXP3          Monte Carlo       Average Reward
  Hedge             EXP4          TD/SARSA/Q        Online RL
```

[[figure mdp-agent-environment-loop]]

[[figure orl-cartpole-learning-image]]

## Why this topic matters

Online learning and reinforcement learning are not abstract curiosities. They are the mathematics behind any system that must make repeated decisions under uncertainty.

When a spam filter decides whether to block an email, it is playing this game — the adversary is an attacker who adapts. When a doctor chooses between treatments for a patient, she is facing a contextual bandit — each patient is different, and she only sees the outcome of the treatment she prescribed. When a robot navigates a warehouse, it is solving a Markov decision process — its actions change where it ends up next.

These ideas power online advertising, recommendation engines, adaptive routing in networks, automated trading, and robotics. Wherever you see repeated interaction with an uncertain world, you will find the tools from this course.

## Key mathematical ideas

The mathematics in this course revolves around four big themes. First, **regret minimization** — you will see how to define and bound the gap between your performance and the best fixed strategy in hindsight, without any statistical assumptions on the data. Second, **exponential weights and mirror-style updates** — the engine behind algorithms like Hedge that multiply confidence by performance. Third, **exploration-exploitation trade-offs** — the bandit's dilemma, where you must balance trying new actions against sticking with what has worked. Fourth, **Bellman operators and fixed-point reasoning** — the backbone of MDP theory, where the right value is the one that does not change when you apply the update rule.

