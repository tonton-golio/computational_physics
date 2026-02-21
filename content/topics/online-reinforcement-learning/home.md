# Online and Reinforcement Learning

Picture two weather forecasters. Every morning, Forecaster A says "Rain." Every morning, Forecaster B says "Sunny." Neither looks out the window. After a thousand days, which one was better? You cannot know until you check what actually happened.

But what if you could do something smarter? Each morning, you look at both track records and weight your prediction toward whoever has been right most often. You start off clueless, but day by day your cumulative mistakes grow *slower* than the best single forecaster's. The gap between your total loss and the best expert's total loss is called **regret**. Making that gap grow as slowly as possible is the entire game of online learning.

That idea is the thread running through everything here. We start with regret, learn algorithms that minimize it, move to slot machines where you only see one outcome at a time, and then enter the full world of reinforcement learning where your actions change the world around you. How do you make good decisions, one at a time, when you do not know what is coming next?

## Why This Topic Matters

- When a spam filter decides whether to block an email, it is playing this game against an adversary who adapts.
- When a doctor chooses between treatments, she faces a contextual bandit where she only sees the outcome of the treatment she prescribed.
- When a robot navigates a warehouse, it solves a Markov decision process where its actions change where it ends up next.
- These ideas power online advertising, recommendation engines, adaptive routing, automated trading, and robotics.
