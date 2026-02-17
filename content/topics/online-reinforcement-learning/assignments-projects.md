# Assignments and Project Ideas

## Empirical assignment track

Implement and compare learning curves and regret for:

- Hedge in expert advice.
- UCB1 and EXP3 in multi-armed bandits.
- SARSA and Q-learning in a gridworld or cliff-walking environment.
- DQN on a compact control benchmark.

Suggested outputs:

- Cumulative reward curves.
- Cumulative regret curves.
- Sensitivity to learning rate and exploration schedule.

## Theoretical assignment track

Derive and present key arguments:

- Regret bound sketch for Hedge.
- Concentration-based arm selection argument for UCB.
- Bellman contraction proof and fixed-point uniqueness.
- On-policy vs off-policy update implications.

## Course project

Choose one applied setting and report reproducible experiments:

- Online advertising or recommendation simulator.
- Inventory/control MDP.
- Custom game or robotics-inspired simulator.
- Gymnasium-style benchmark environment.

Minimum project deliverables:

1. Problem definition and assumptions.
2. Baselines and chosen algorithm(s).
3. Experimental protocol.
4. Regret or return analysis.
5. Failure modes and next improvements.
