# Feedback Loops in Biological Systems

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

In [transcriptional regulation](./transcriptional-regulation) you saw how the Hill function lets a cell turn a gene up or down. That's powerful, but it's a one-way street. Now we close the loop. What happens when a gene product influences its *own* production? This simple act -- feedback -- gives cells memory, decision-making, and the ability to keep time.

## Negative autoregulation: the thermostat

Suppose a transcription factor $X$ represses its own promoter:

> **Key Equation -- Negative Autoregulation**
> $$
> \frac{dX}{dt} = \frac{\beta}{1 + (X/K)^n} - \gamma X
> $$
> The gene product represses its own production: when $X$ is high, the Hill term shrinks, self-correcting toward steady state.

> *When $X$ is low, repression is weak and production runs full blast. As $X$ builds up, it sits on its own promoter and slows itself down. The system self-corrects.*

This self-correction gives negative autoregulation three beautiful properties:

* **Faster response**: production starts at full blast when $X$ is low, then self-limits. Gets to steady state faster than a gene without feedback.
* **Reduced noise**: fluctuations above the mean get pushed back down. Fluctuations below get a boost. It's a thermostat.
* **Robustness**: steady state is less sensitive to parameter variations.

## Finding the steady state graphically

Here's a powerful trick. Draw two curves on the same paper:

1. **Production rate** $f(X) = \beta/(1 + (X/K)^n)$ -- starts high, decreases.
2. **Degradation rate** $\gamma X$ -- a straight line through the origin.
3. **Where they cross is where the system is happy and stays put.**

> *At the crossing, production = degradation. Below it, production wins and $X$ rises. Above it, degradation wins and $X$ falls. The system is pulled toward the crossing like a ball rolling to the bottom of a valley.*

[[simulation steady-state-regulation]]

Start with low $\gamma$ and watch the curves cross at one stable point. Increase $\gamma$ -- the crossing moves. For negative feedback, you always get exactly one crossing. Switch to positive feedback and try to find three crossings. That's bistability.

[[simulation production-degradation-crossings]]

The blue S-curve is production for positive feedback; the red line is degradation. Adjust Hill coefficient and $\gamma$ to watch crossings change from one (monostable) to three (bistable). Green dots = stable; grey = unstable.

## Positive feedback and bistability

Now suppose the TF *activates* its own production:

$$
\frac{dX}{dt} = \frac{\beta (X/K)^n}{1 + (X/K)^n} + \beta_0 - \gamma X,
$$

where $\beta_0$ is a small basal leak.

For sufficiently cooperative activation ($n \geq 2$), the S-shaped production curve can cross the degradation line at **three points**: two stable (low and high expression) and one unstable in between. This is **bistability**.

> *The cell now has **memory**. It can sit in the low or high state indefinitely, even after the signal that pushed it there is gone. That's how a stem cell stays a stem cell and how a phage decides to lysogenize.*

[[simulation bifurcation-diagram]]

Drag the degradation slider slowly and watch stable states appear and disappear. Sweep back and forth to trace the hysteresis loop: the system doesn't jump back at the same point it jumped forward. That irreversibility is the hallmark of bistable switches.

## The genetic toggle switch

Gardner, Cantor, and Collins (2000) built a synthetic bistable circuit from two mutually repressing genes:

$$
\frac{dU}{dt} = \frac{\alpha_1}{1 + V^n} - U, \qquad \frac{dV}{dt} = \frac{\alpha_2}{1 + U^n} - V.
$$

> *Gene $U$ represses $V$, and $V$ represses $U$. It's a molecular arm-wrestling match. One wins, the loser stays repressed.*

Two stable states (high-$U$/low-$V$ and vice versa) separated by an unstable saddle point. Flip it with a transient pulse -- like flipping a light switch.

[[simulation phase-plane-portrait]]

The phase portrait shows nullcline intersections and trajectory flows. In the toggle switch, the same technique reveals basins of attraction: which initial conditions lead to which steady state.

## Oscillations: the repressilator

Elowitz and Leibler (2000) built a biological clock from three genes in a ring of repression: $A$ represses $B$, $B$ represses $C$, $C$ represses $A$.

$$
\frac{dm_i}{dt} = \frac{\alpha}{1 + p_j^n} + \alpha_0 - m_i, \qquad \frac{dp_i}{dt} = \beta(m_i - p_i),
$$

where $j$ is the upstream repressor of gene $i$.

Think of it as **three friends who can never all be happy at once**. When $A$ is high, $B$ is repressed. With $B$ low, $C$ rises. When $C$ gets high, it represses $A$, which frees $B$. The system chases its own tail forever.

> *The deep principle: odd-numbered repression rings oscillate. Even-numbered ones switch. This same logic underlies circadian clocks and cell cycle oscillators across all of life.*

[[simulation repressilator]]

Watch three proteins chase each other. At low $n$ ($n = 2$), traces may converge to a fixed point. Crank $n$ to 4+ and clear oscillations emerge. Adjust repression strength and degradation to tune the period.

## Biological examples

Three vivid stories show these motifs in nature:

* **Competence in *B. subtilis***: positive feedback on *comK* creates a bistable switch -- a rare fraction of cells stochastically flip into competence to grab DNA from the environment.
* **Lambda phage**: mutual repression between CI and Cro creates a toggle -- kill the cell or hide in its genome.
* **p53-Mdm2**: negative feedback generates oscillatory pulses after DNA damage -- the cell "thinks it over" before deciding to die.

## Why does nature do it this way?

Negative feedback provides stability and speed (the thermostat). Positive feedback provides commitment and memory (the switch). Odd-numbered repression rings provide timing (the clock). Together, these are the building blocks of cellular logic.

## Check your understanding

* Why does negative autoregulation speed up response time? (Hint: what happens at $t = 0$ when $X = 0$?)
* Draw production and degradation curves for positive feedback with $n = 4$. How many intersections?
* You add a fourth gene to the repressilator ring. Clock or switch?

## Challenge

Take negative autoregulation with $\beta = 10$, $K = 1$, $n = 2$, $\gamma = 1$, starting from $X(0) = 0$. Does $X$ overshoot before settling? Compare to a no-feedback gene $\dot{X} = \beta_\mathrm{eff} - \gamma X$ with the same steady state. Which reaches steady state faster?

## Big ideas

* Negative feedback = homeostasis, speed, noise reduction. The thermostat.
* Positive feedback = bistability and memory. The commitment switch.
* Odd-numbered repression rings oscillate; even-numbered ones switch. Nature's clock-vs-toggle rule.

## What comes next

Feedback lets the cell talk to itself. But it also needs to listen to the outside world -- and bacterial chemotaxis is the most elegant navigation system in nature.
