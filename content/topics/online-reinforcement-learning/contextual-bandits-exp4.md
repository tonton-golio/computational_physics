# Contextual Bandits and EXP4

In contextual bandits, each round starts with context $x_t$.
The learner chooses action $a_t$ using $x_t$ and receives only bandit feedback for that action.

## Why contextual bandits

- Personalization in ads and recommendations.
- Medical dosing with patient covariates.
- Adaptive interfaces and ranking.

## EXP4 idea

EXP4 aggregates a set of experts (policies), where each expert maps context to an action distribution.

At round $t$:

1. Each expert proposes a distribution over actions for context $x_t$.
2. The learner forms a mixture distribution over experts.
3. Sample an action from the induced action distribution.
4. Build an importance-weighted loss estimate.
5. Update expert weights exponentially.

This achieves regret that scales roughly with $\sqrt{T\log |\mathcal{E}|}$ (up to constants and feedback terms), where $\mathcal{E}$ is the expert class.

[[simulation contextual-bandit-exp4]]
