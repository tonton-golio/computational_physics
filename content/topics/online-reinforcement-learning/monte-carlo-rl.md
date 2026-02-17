# Monte Carlo Methods for RL

Monte Carlo (MC) methods estimate values from **complete episodes**.
No transition model is required.

## Core idea

For return

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1},
$$

estimate values by sample averages:

$$
V^\pi(s)\approx \frac{1}{N(s)}\sum_{t:\,s_t=s} G_t.
$$

## Variants

- First-visit MC.
- Every-visit MC.
- On-policy MC control.
- Off-policy MC with importance sampling.

## Trade-offs

- Low bias for return targets.
- High variance, especially with long horizons.
- Naturally episodic and simple to implement.

[[simulation monte-carlo-convergence]]
