# Complex Physics

## Course overview

Complex physics studies systems composed of many interacting components whose collective behavior cannot be predicted from individual parts alone. These systems exhibit **emergent phenomena** such as phase transitions, self-organization, power-law distributions, and fractal geometry.

- Classical physics: few-body systems, exact solutions, linear superposition.
- Complex physics: many-body interactions, statistical descriptions, nonlinear dynamics, universality.

Core approach:

1. Identify the relevant degrees of freedom and their interactions.
2. Use statistical mechanics to connect microscopic rules to macroscopic observables.
3. Recognize universal scaling behavior near critical points.
4. Simulate complex systems computationally when analytical solutions fail.

## The road map

Imagine you are watching a pot of water boil. First we will understand why anything happens at all: why does heat spread evenly, why do molecules share energy the way they do? That is **statistical mechanics**. Then we will ask: how can we let a computer roll the dice for us when the math gets too hard? That is the **Metropolis algorithm**. Next comes the dramatic part: what happens when everything suddenly changes, when a magnet loses its magnetism or a liquid becomes a gas? Those are **phase transitions**, and we will build a simple theory (**mean-field**) to understand them, then solve a toy model exactly with the **transfer matrix**.

But here is where it gets truly beautiful. We will discover that completely different systems, magnets and fluids and gases, all behave *identically* near their critical points. That is **universality and critical phenomena**. The same patterns keep showing up: in **percolation** (when a connected path suddenly spans a random medium), in **self-organized criticality** (when a sandpile tunes itself to the edge of chaos), in **networks** (when the rich get richer and power laws emerge), in **agent-based models** (when dumb local rules produce smart global behavior), and even in **econophysics** (when stock markets crash like avalanches).

By the end of this course, you will see these patterns everywhere. Nature reuses her tricks, and we are going to learn them.

## Why this topic matters

- Phase transitions underpin phenomena from magnetism to superconductivity.
- Percolation theory models fluid flow through porous media, disease spreading, and network resilience.
- Self-organized criticality explains power laws in earthquakes, forest fires, and biological evolution.
- Network science describes the structure of the internet, social systems, and gene regulation.
- Agent-based models capture emergent collective behavior from simple local rules.
- Econophysics applies statistical mechanics to financial markets and wealth distributions.

## Key mathematical ideas

- Partition functions, free energy, and thermodynamic observables.
- Order parameters, critical exponents, and scaling relations.
- Renormalization group and universality classes.
- Power-law distributions and fractal dimensions.
- Monte Carlo sampling and Markov chain methods.
- Graph theory and network metrics.

## Prerequisites

- Probability and statistical reasoning.
- Calculus and linear algebra.
- Basic thermodynamics and classical mechanics.
- Familiarity with Python for computational exercises.

## Recommended reading

- Thurner, Hanel, and Klimek, *Introduction to the Theory of Complex Systems*.
- Sethna, *Statistical Mechanics: Entropy, Order Parameters, and Complexity*.
- Newman, *Networks: An Introduction*.

## Learning trajectory

This module is organized from equilibrium foundations to complex emergent behavior:

- **Statistical Mechanics** — Why does a room full of air molecules share energy so democratically? We start with the rules that govern huge numbers of particles.
- **Metropolis Algorithm** — Let the computer do the hard work of exploring impossible-to-calculate probabilities.
- **Phase Transitions** — The moment everything changes: how a magnet suddenly finds its direction.
- **Mean-Field Results** — A beautifully simple (and sometimes wrong) theory where every spin feels the average of all the others.
- **1D Ising & Transfer Matrix** — One dimension, one exact solution, and a clever matrix trick that makes it all work.
- **Critical Phenomena** — The miracle of universality: why magnets, fluids, and gases all forget their differences at the critical point.
- **Percolation and Fractals** — When random connections suddenly span the world, and the geometry turns out to be fractal.
- **Self-Organized Criticality** — Sandpiles, avalanches, and systems that tune themselves to the edge without anyone turning the knob.
- **Networks** — The rich get richer, power laws appear, and six degrees of separation connect us all.
- **Agent-Based Models** — Dumb local rules, smart global behavior: flocking birds, traffic jams, and the Game of Life.
- **Econophysics** — Stock markets crash like sandpiles, and the same mathematics describes both.

## Visual and Simulation Gallery

[[figure complex-sandpile-image]]

[[figure complex-percolation-video]]
