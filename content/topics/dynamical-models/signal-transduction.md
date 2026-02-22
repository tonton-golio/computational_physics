# Signal Transduction

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

Watch this tiny bug. An *E. coli* bacterium -- no brain, no eyes, no map -- swims through a sugar gradient doing calculus with its body. It detects a concentration change of one extra molecule per cell volume per micron, across five orders of magnitude of background. It's the most sensitive navigation system in nature, and you're about to understand exactly how it works.

## What is a signal?

A signal is anything the cell can detect -- food molecules, quorum-sensing signals, hormones, pressure, temperature, light. The signal binds to a **receptor** on the cell surface, triggering a cascade inside. That cascade is signal transduction.

## Why so many steps?

Why not just have the signal flip a gene directly? Because intermediate steps give the cell superpowers: **amplification** (one receptor event activates thousands of downstream molecules), **noise filtering**, **integration** of multiple signals, **memory**, and **adaptation** (responding to *changes* rather than absolute levels).

## Bacterial chemotaxis: nature's GPS

### How bacteria swim

You're a bacterium, ~2 micrometers long, with helical flagella driven by molecular motors.

* Motors spin **counterclockwise**: flagella bundle, you swim straight. That's a **run**.
* One motor flips to **clockwise**: the bundle flies apart, you tumble randomly.

A run lasts about 1 second, then you tumble and pick a new direction. On average, you go nowhere -- a random walk. But here's the trick nature discovered: **if things are getting better (more food), run longer. If things are getting worse, tumble sooner.** This bias turns a random walk into directed climbing.

### The molecular circuit

* **CheA** is a kinase associated with the receptor. Less attractant makes CheA more active.
* **CheY** gets phosphorylated by CheA. CheY-P binds the flagellar motor and triggers tumbling.

> *Less food -> more CheA -> more CheY-P -> more tumbling. More food -> less CheA -> less CheY-P -> longer runs toward the food. Simple and beautiful.*

### The problem of saturation

If tumbling depends on *absolute* concentration, the system saturates at high backgrounds. The bacterium goes blind. How does nature fix this?

### Adaptation: the receptor's memory

The answer is **perfect adaptation**. The cell doesn't respond to absolute concentration -- it responds to *changes*. After a step increase in attractant, tumbling drops immediately (the cell runs), then gradually returns to baseline over about a minute. The cell has adapted.

Think of the receptor as a little spring. Attractant binding relaxes the spring. But then **CheR** slowly adds methyl groups to the receptor, stiffening the spring back. This **methylation** is the receptor's memory. The opposing enzyme **CheB** removes methyl groups, and the balance between CheR and CheB sets the adaptation level.

> **Key Equation -- Adaptation via Methylation**
> $$
> \frac{\mathrm{d} [E_m]}{\mathrm{d} t} = k^R [\mathrm{CheR}] - \frac{k^B B^{*}(l) \, E_m}{K_M^B + E_m}
> $$
> CheR adds methyl groups at a constant rate; CheB removes them depending on receptor activity. The balance resets the baseline, giving perfect adaptation.

[[simulation chemotaxis-adaptation]]

Step up the ligand and watch activity spike then return to baseline while methylation ramps up to compensate. Try different step sizes -- transient response changes, but steady-state activity always returns to the same value. That's perfect adaptation.

With adaptation, the bacterium can detect tiny changes at *any* background level -- nanomolar to millimolar. Like a camera that automatically adjusts its exposure.

## Cell-to-cell communication: lateral inhibition

Now suppose your neighbors produce a signal (**Delta** ligand) that activates your **Notch** receptor. When Notch fires, it *represses* your own Delta production.

> *If my neighbor is loud, I shut up. If I shut up, my other neighbors get louder. The result: a checkerboard pattern.*

[[simulation notch-delta-checkerboard]]

Watch a checkerboard emerge from random initial conditions. With high Hill coefficient, cells make sharp high-or-low decisions. Try $n = 1$ to see the pattern dissolve without cooperativity.

This **lateral inhibition** explains the regular spacing of bristles on a fly, alternating cell fates in the inner ear, and the spacing of hair follicles on your skin.

## Why does nature do it this way?

Cells need to respond to *changes*, not absolute levels. Degradation gives speed. Feedback gives memory. Adaptation gives sensitivity across a huge dynamic range. Signal transduction combines all three.

## Check your understanding

* Why does the bacterium respond to *changes* in concentration? What goes wrong without adaptation?
* Delete CheR. What happens to chemotaxis?
* How does Notch-Delta lateral inhibition create a pattern from a uniform population?

## Challenge

Simplified adaptation model: activity $A = (1 + M) / (1 + L)$, methylation adjusts via $\dot{M} = k_R - k_B \cdot A$. What's the steady-state activity $A^*$? Does it depend on $L$?

*Hint: at steady state, $\dot{M} = 0$ gives $A^* = k_R / k_B$. Notice anything about $L$?*

## Big ideas

* Bacterial chemotaxis achieves extraordinary sensitivity by responding to concentration *changes* through methylation-based adaptation.
* Perfect adaptation means steady-state activity is independent of background concentration -- the cell only notices what's *changing*.
* Lateral inhibition (Notch-Delta) creates spatial patterns through mutual repression between neighbors.

## What comes next

We've been studying individual circuits. But inside a real cell, all these circuits are wired together into networks built from a small toolkit of recurring motifs -- each with its own computational personality.
