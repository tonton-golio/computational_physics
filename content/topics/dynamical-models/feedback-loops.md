# Feedback Loops in Biological Systems

## Where we are headed

Last time you saw how the Hill function lets a cell turn gene expression up or down in response to a signal. That is powerful, but it is still a one-way street: a signal comes in, the gene responds. Today we close the loop. What happens when a gene product influences its *own* production? This simple act — feedback — is what gives cells memory, decision-making ability, and the capacity to oscillate. Switches, clocks, and irreversible commitments all emerge from feedback, and you will see exactly why.

## Why feedback matters

A gene product that influences its own production creates a **feedback loop**, and the sign of that influence determines everything:

- **Negative feedback** (the gene product represses itself): promotes homeostasis, speeds up the response, and reduces noise. The system finds a stable set point and sticks to it.
- **Positive feedback** (the gene product activates itself): generates bistability, memory, and irreversible switching. The system can commit to one of two states and stay there.

## Negative autoregulation

Suppose a transcription factor $X$ represses its own promoter. Using the Hill function from last lesson:

$$
\frac{dX}{dt} = \frac{\beta}{1 + (X/K)^n} - \gamma X,
$$

where $\beta$ is the maximal production rate (the "factory at full power"), $K$ is the repression threshold (the "sensitivity dial"), $n$ is the Hill coefficient (the "sharpness knob"), and $\gamma$ is the degradation rate (the "death rate constable").

> *In words: when $X$ is low, the repression term is small and production runs at nearly full speed. As $X$ builds up, it starts sitting on its own promoter and slowing down its own production. The system self-corrects.*

This self-correction gives negative autoregulation three beautiful properties:

- **Faster response time**: the system overshoots at first (production starts at full blast when $X$ is low), then self-limits. This gets to steady state faster than a gene without feedback.
- **Reduced noise**: if $X$ fluctuates above the mean, the increased repression pushes it back down. If it drops below, repression eases and production speeds up. The feedback acts like a thermostat.
- **Robustness**: the steady-state level is less sensitive to parameter variations. If the degradation rate changes slightly, the feedback compensates.

## Finding the steady state graphically

Here is a powerful trick for analyzing any one-dimensional feedback circuit. Think of it as drawing two curves on the same piece of paper:

1. Draw the **production rate** $f(X) = \beta/(1 + (X/K)^n)$ as a function of $X$. For negative feedback, this curve starts high and decreases.
2. Draw the **degradation rate** $\gamma X$ — a straight line through the origin.
3. **Where they cross is where the system is happy and stays put.** That intersection is the steady state.

> *At the crossing point, production exactly balances degradation — nothing changes. If $X$ is below the crossing point, production exceeds degradation, so $X$ increases. If $X$ is above, degradation wins, so $X$ decreases. The system is pulled toward the crossing like a ball rolling to the bottom of a valley.*

> **Figure: Graphical steady-state construction.** Sigmoidal Hill production curve intersecting a straight degradation line $\gamma P$. Show 1 intersection for negative regulation vs 3 intersections for positive regulation (two stable as solid dots, one unstable as open dot).

To check stability: if the production curve is *below* the degradation line to the right of the crossing (and above to the left), the fixed point is stable. If the slopes tell the opposite story, it is unstable.

[[simulation steady-state-regulation]]

> **Try this**: Start with the degradation rate $\gamma$ low and watch the production and degradation curves cross at one point (stable steady state). Now slowly increase $\gamma$ — the crossing point moves. For negative feedback, you always get exactly one crossing. Switch to positive feedback mode and try the same thing — you should be able to find a parameter range where three crossings appear: two stable and one unstable. That is bistability.

## Positive feedback and bistability

Now suppose the transcription factor *activates* its own production:

$$
\frac{dX}{dt} = \frac{\beta (X/K)^n}{1 + (X/K)^n} + \beta_0 - \gamma X,
$$

where $\beta_0$ is a small basal (leak) production rate — even without feedback, a trickle of $X$ is made.

For sufficiently cooperative activation ($n \geq 2$), the production curve can be S-shaped, and it may cross the degradation line at **three points**: two stable (low and high expression) and one unstable in between. This is **bistability**.

> *Look at the picture: a straight degradation line can cross an S-curve once (monostable) or three times (two stable states + one unstable). The cell now has **memory** — it can sit in the low or high state forever, even after the signal is gone. That is how a stem cell stays a stem cell and how a phage decides to lysogenise.*

A push large enough to cross the unstable middle point flips the switch. The positive feedback maintains itself — once a cell commits to the high state, it remains there even after the inducing signal is removed.

> **Figure: Bifurcation diagram for positive feedback.** Steady-state protein level vs degradation rate $\gamma$, with solid (stable) and dashed (unstable) branches, clearly marking the bistable region between the two saddle-node bifurcations.

## The genetic toggle switch

In a landmark experiment, Gardner, Cantor, and Collins (2000) built a synthetic bistable circuit from two mutually repressing genes:

$$
\frac{dU}{dt} = \frac{\alpha_1}{1 + V^n} - U, \qquad \frac{dV}{dt} = \frac{\alpha_2}{1 + U^n} - V.
$$

> *Gene $U$ represses gene $V$, and gene $V$ represses gene $U$. It is a molecular arm-wrestling match. One of them wins, and the loser stays repressed.*

This toggle switch has two stable steady states (high-$U$/low-$V$ and low-$U$/high-$V$) separated by an unstable saddle point. You can flip it between states by hitting it with a transient pulse of inducer — like flipping a light switch by briefly pushing it.

## Oscillations: the repressilator

Elowitz and Leibler (2000) asked a daring question: can you build a biological clock from scratch? They constructed a synthetic oscillator — the **repressilator** — from three genes arranged in a ring of repression: $A$ represses $B$, $B$ represses $C$, $C$ represses $A$.

$$
\frac{dm_i}{dt} = \frac{\alpha}{1 + p_j^n} + \alpha_0 - m_i, \qquad \frac{dp_i}{dt} = \beta(m_i - p_i),
$$

where $j$ is the upstream repressor of gene $i$.

> **Figure: Repressilator.** Three-gene time series showing oscillating protein concentrations (A, B, C out of phase) alongside a phase portrait showing the limit cycle.

Think of it as **three friends who can never all be happy at once**. When $A$ is high, it represses $B$, so $B$ is low. But with $B$ low, $C$ is free to rise. When $C$ gets high enough, it represses $A$, which releases $B$, and the cycle continues. The system chases its own tail forever.

> *Here is the deep principle: negative feedback loops arranged in odd-numbered rings can generate oscillations. Two mutual repressors give a switch (even number = stable). Three in a ring give a clock (odd number = unstable oscillations). This same principle underlies circadian clocks and cell cycle oscillators across all of life.*

Sustained oscillations require:
- Strong cooperativity (high Hill coefficient $n$).
- Well-matched protein and mRNA lifetimes.
- Strong repression ($\alpha \gg \alpha_0$).

> **Try this**: Use the steady-state regulation simulation above. Increase the Hill coefficient $n$ and watch the oscillations become sharper and more sustained. With $n = 1$, the system may settle to a fixed point. With $n = 4$, you should see clear oscillatory behavior.

## Biological examples

- **Competence in *B. subtilis***: positive feedback on *comK* creates a bistable switch. A small fraction of cells stochastically flip into the competent state and take up DNA from the environment.
- **Lambda phage lysis-lysogeny**: mutual repression between CI and Cro creates a toggle switch that determines whether the virus kills the cell or hides in its genome.
- **p53-Mdm2 oscillations**: negative feedback between the tumor suppressor p53 and its inhibitor Mdm2 generates oscillatory pulses in response to DNA damage — the cell's way of "thinking it over" before deciding to die.

## Why does nature do it this way?

Strip away feedback and you strip away everything that makes a cell smart: memory, decision-making, and timekeeping. Feedback is what turns a collection of chemical reactions into a computing machine. Negative feedback provides stability and speed; positive feedback provides commitment and memory; oscillatory circuits provide timing. Together, these are the fundamental building blocks of cellular logic.

## Check your understanding

- Why does negative autoregulation speed up the response time? (Hint: think about what happens at $t = 0$ when $X$ starts from zero.)
- Draw the production and degradation curves for a positive-feedback gene with $n = 4$. How many intersections do you see?
- The repressilator has three genes in a ring. What would happen with four genes in a ring? Would you get oscillations or a switch?

## Challenge

Take the negative autoregulation equation with $\beta = 10$, $K = 1$, $n = 2$, $\gamma = 1$, and start from $X(0) = 0$. Simulate or sketch the trajectory — does $X$ overshoot the steady state before settling down? Now compare to a gene with no feedback: $\dot{X} = \beta_\mathrm{eff} - \gamma X$ where $\beta_\mathrm{eff}$ is chosen so that the steady state is the same. Which system reaches steady state faster? This is the speed advantage of negative feedback in action.

## Big ideas

- **Negative feedback** provides homeostasis, speed, and noise reduction by self-correcting deviations from the set point.
- **Positive feedback** creates bistability and memory — cells can commit to a state and stay there.
- **Odd-numbered repression rings oscillate; even-numbered ones switch** — this is the fundamental principle behind biological clocks and toggle switches.

## What comes next

Feedback loops let the cell talk to itself. But a cell also needs to listen to the outside world. In the next lesson, we discover how bacteria read their mail — sensing food gradients with a sensitivity that would make an engineer jealous, and resetting their sensors through a trick called adaptation. We are about to meet the most elegant navigation system in nature: bacterial chemotaxis.
