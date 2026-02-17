# Feedback Loops in Biological Systems

## Why feedback matters

Feedback loops are the fundamental building blocks of regulatory circuits in biology. A gene product that influences its own production creates a **feedback loop**, and the sign of that influence, positive or negative, determines the qualitative behavior of the circuit.

- **Negative feedback**: the gene product represses its own transcription. This promotes homeostasis, speeds response, and reduces noise.
- **Positive feedback**: the gene product activates its own transcription. This generates bistability, memory, and irreversible switching.

## Negative autoregulation

Consider a transcription factor $X$ that represses its own promoter. The ODE model is:

$$
\frac{dX}{dt} = \frac{\beta}{1 + (X/K)^n} - \gamma X,
$$

where $\beta$ is the maximal production rate, $K$ is the repression threshold, $n$ is the Hill coefficient, and $\gamma$ is the degradation rate.

Key properties of negative autoregulation:

- **Faster response time**: the system reaches steady state more quickly than an unregulated gene because high initial production is followed by self-limiting repression.
- **Reduced noise**: fluctuations above the mean are suppressed by increased repression, and vice versa.
- **Robustness**: the steady-state level $X^*$ is less sensitive to parameter variations than for constitutive expression.

The steady state satisfies $\beta / (1 + (X^*/K)^n) = \gamma X^*$, which can be solved graphically as the intersection of the production and degradation curves.

## Positive feedback and bistability

When a transcription factor activates its own production, the model becomes:

$$
\frac{dX}{dt} = \frac{\beta (X/K)^n}{1 + (X/K)^n} + \beta_0 - \gamma X,
$$

where $\beta_0$ is a basal (leak) production rate.

For sufficiently cooperative activation ($n \geq 2$) and appropriate parameter values, the production curve intersects the degradation line at **three fixed points**: two stable (low and high expression) and one unstable. This is **bistability**.

- The system can exist in either the low or high state.
- Transitions between states require a sufficiently large perturbation or a change in parameters.
- Bistability provides **cellular memory**: once a cell commits to a state, it remains there even after the inducing signal is removed.

## The genetic toggle switch

Gardner, Cantor, and Collins (2000) constructed a synthetic bistable circuit from two mutually repressing genes:

$$
\frac{dU}{dt} = \frac{\alpha_1}{1 + V^n} - U, \qquad \frac{dV}{dt} = \frac{\alpha_2}{1 + U^n} - V.
$$

This **toggle switch** has two stable steady states (high-$U$/low-$V$ and low-$U$/high-$V$) separated by an unstable saddle point. The system can be flipped between states by transient pulses of inducer.

[[simulation steady-state-regulation]]

## Oscillations: the repressilator

Elowitz and Leibler (2000) built a synthetic oscillator from three genes in a ring of repression: $A$ represses $B$, $B$ represses $C$, $C$ represses $A$.

$$
\frac{dm_i}{dt} = \frac{\alpha}{1 + p_j^n} + \alpha_0 - m_i, \qquad \frac{dp_i}{dt} = \beta(m_i - p_i),
$$

where $j$ is the upstream repressor of gene $i$.

Sustained oscillations arise when:

- The Hill coefficient $n$ is large enough (strong cooperativity).
- Protein and mRNA lifetimes are well matched.
- The repression is strong ($\alpha \gg \alpha_0$).

The repressilator demonstrates that negative feedback loops arranged in odd-numbered rings can generate oscillations, a principle that also explains circadian clocks and the cell cycle.

[[simulation lotka-volterra]]

## Graphical fixed-point analysis

A powerful technique for analyzing one-dimensional feedback circuits:

1. Plot the **production rate** $f(X)$ as a function of $X$.
2. Plot the **degradation rate** $\gamma X$ as a straight line.
3. Intersections are **fixed points**.
4. Stability is determined by the slopes: if $f'(X^*) < \gamma$ the fixed point is stable; if $f'(X^*) > \gamma$ it is unstable.

For multistable systems, this graphical method reveals the number and location of stable states, the thresholds for switching, and the sensitivity to parameter changes.

## Biological examples

- **Competence in *B. subtilis***: positive feedback on *comK* creates a bistable switch; a small fraction of cells stochastically enter the competent state.
- **Lambda phage lysis-lysogeny decision**: mutual repression between CI and Cro creates a toggle switch determining the fate of the infected cell.
- **p53-Mdm2 oscillations**: negative feedback between the tumor suppressor p53 and its inhibitor Mdm2 generates oscillatory pulses in response to DNA damage.
