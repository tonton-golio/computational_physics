# Applied Statistics

## What this course is really about

Statistics isn't about numbers. It's about learning the truth when the world is noisy and you only get to peek once.

Every measurement you'll ever make is contaminated by randomness — your instruments are imperfect, your samples are finite, and the universe just won't sit still. This course hands you the tools to cut through that noise and figure out what's actually going on. Not by memorizing formulas, but by understanding *why* each technique exists and *when* it's the right one to reach for.

- **Classical statistics**: estimation, hypothesis testing, confidence intervals.
- **Applied statistics**: model selection, validation, interpretation, and communication of results.
- **The real skill**: knowing when each method is appropriate and what its assumptions demand.

> **A Bayesian sneak preview.** Most of this course uses frequentist methods — p-values, confidence intervals, maximum likelihood. But there is another way of thinking about probability: as a *degree of belief* that updates when new evidence arrives. That's the Bayesian view, and it's closer to how your brain actually works when it learns from data. We cover it formally in lesson 11, but you'll see hints of it much earlier. When we talk about likelihood in lesson 2, or priors in lesson 12, remember: Bayesian reasoning is quietly running in the background the whole time.

## Course map

Here is the big picture — the forest before the trees.

```
FOUNDATIONS                      COMPARING GROUPS
  1. Introduction ─────────┐       7. ANOVA
  2. Distributions & MLE ──┤       8. Experimental Design
  3. CLT & Error Propagation
  4. Simulation            │     REAL-WORLD STRUCTURES
                           │       9.  Mixed Models
TOOLS FOR NOISY DATA       │      10. Longitudinal Data
  5. Chi-Square Fitting ───┘
  6. Hypothesis Testing          ADVANCED TOOLS
                                  11. Bayesian Statistics
                                  12. Advanced Fitting
                                  13. Machine Learning
```

The arrows between these stages matter as much as the stages themselves. The likelihood function introduced in lesson 2 shows up in chi-square fitting (5), hypothesis testing (6), Bayesian statistics (11), and machine learning (13). The CLT (3) justifies why Gaussian methods work everywhere. Mixed models (9) generalize ANOVA (7) to messy, grouped data. And by lesson 13, you'll realize that every "new" machine learning trick is secretly something you already learned.

## Meet Alex

Throughout this course, you'll follow **Alex**, a physicist who measures things for a living and keeps running into the same problem: the universe refuses to give a clean answer. Alex measures the gravitational acceleration and gets $9.81 \pm 0.03$ m/s$^2$ — but is that uncertainty right? Alex compares two treatments and sees a difference — but is it real or just noise? Every lesson, Alex faces a new puzzle. You'll solve it together.

## Why this topic matters

- Every experimental science relies on statistical reasoning to distinguish signal from noise.
- Choosing the wrong test or violating assumptions leads to false conclusions — and in medicine, engineering, or policy, false conclusions have real consequences.
- Modern datasets demand regression, ANOVA, mixed models, and longitudinal methods.
- Statistical literacy is essential for reading and producing scientific literature.

## Key mathematical ideas

- Probability distributions and likelihood functions — the language of uncertainty.
- Parametric and non-parametric hypothesis tests — asking "is this real?"
- Linear models: regression, ANOVA, and their generalizations — the workhorses.
- Random effects and hierarchical/mixed models — handling messy, grouped data.
- Experimental design: randomization, blocking, and power analysis — getting the data right before you analyze it.

## Prerequisites

- Introductory statistics: variation, estimation, confidence intervals, hypothesis tests.
- One-way ANOVA and simple linear regression.
- Basic familiarity with Python or R.

## Recommended reading

- Draper and Smith, *Applied Regression Analysis*.
- Agresti, *Statistical Methods for the Social Sciences*.

- Course notes and documentation.

## Learning trajectory

This module builds like a chain — each topic relies on what came before and sets up what comes next. The connections between lessons are as important as the lessons themselves.

- **Lesson 1 — Introduction and general concepts**: You have a pile of numbers — what can you learn from them? We start with the tools for summarizing data: means, medians, spread, and correlation. These descriptive tools appear in every later lesson. *Next up: where do those numbers come from?*
- **Lesson 2 — Probability density functions**: The mathematical models behind data — distributions and maximum likelihood estimation. The likelihood concept introduced here becomes the foundation for fitting, testing, and Bayesian analysis. *Next up: why do errors always seem to be Gaussian?*
- **Lesson 3 — CLT and error propagation**: Why Gaussian statistics work, and how uncertainties flow through calculations. Connects distributions to the practical error analysis used everywhere else. *Next up: what happens when the math gets too hard for pen and paper?*
- **Lesson 4 — Simulation methods**: When analytical formulas break down, Monte Carlo methods take over. Now that you can fake the universe a million times, watch what happens when you ask "is this real or just luck?" *Next up: fitting models to real data.*
- **Lesson 5 — Chi-square method**: Fitting models to data with uncertainties. Connects maximum likelihood (from lesson 2) to practical regression and goodness-of-fit assessment. *Next up: formalizing "is there a real effect?"*
- **Lesson 6 — Hypothesis testing**: Is there a real effect, or is it just noise? p-values, confidence intervals, and the danger of Simpson's paradox. *Next up: what if you have more than two groups?*
- **Lesson 7 — ANOVA**: Extending hypothesis testing to compare multiple groups at once, with post-hoc tests and non-parametric alternatives. *Next up: planning experiments so the analysis actually works.*
- **Lesson 8 — Design of experiments**: Investing thought *before* spending money. Randomization, blocking, and power analysis make everything else in this course more powerful. *Next up: what happens when your data has natural groupings?*
- **Lesson 9 — Random effects and mixed models**: Handling clustered and hierarchical data where observations within groups are correlated. Generalizes ANOVA to realistic data structures. *Next up: the same idea, but now the data are collected over time.*
- **Lesson 10 — Longitudinal data and repeated measures**: Extending mixed models to data collected over time, with autocorrelation, growth curves, and missing data. Remember those random effects from lesson 9? Here's why they save you in real life. *Next up: a fundamentally different way of thinking about probability.*
- **Lesson 11 — Bayesian statistics**: An alternative framework where probability represents belief. Connects to maximum likelihood through the likelihood function, and offers a natural way to incorporate prior knowledge. *Next up: handling the full complexity of real experiments.*
- **Lesson 12 — Advanced fitting and calibration**: Multi-component models, systematic uncertainties, and model comparison. Synthesizes likelihood, chi-square, and Bayesian ideas into a complete analysis framework. *Next up: the grand finale.*
- **Lesson 13 — Machine learning and data analysis**: Everything you've learned — priors, regularization, cross-validation, uncertainty — is secretly inside every modern ML algorithm. This lesson connects the dots and shows you where to go from here.

## Glossary of key ideas

Here are the 20 most important concepts in this course, each in one sentence.

1. **Mean**: The balancing point of your data — add everything up, divide by how many.
2. **Standard deviation**: How far a typical measurement wanders from the mean.
3. **PDF**: A curve that tells you how likely each outcome is — the mathematical model behind your data.
4. **Likelihood**: The probability of your data, viewed as a function of the model parameters — flip the question around.
5. **Maximum likelihood estimation**: Find the parameter values that make your observed data least surprising.
6. **Central limit theorem**: Average enough random things and the result is always Gaussian — nature's favorite trick.
7. **Error propagation**: How uncertainties in your inputs become uncertainties in your outputs.
8. **Monte Carlo**: Let the computer run your experiment a million times to see what could happen.
9. **Chi-square**: A single number measuring how far your data sit from your model, weighted by how much you trust each point.
10. **p-value**: The probability of seeing data this extreme if nothing interesting is happening — not the probability your hypothesis is true.
11. **Confidence interval**: A range of plausible values for a parameter, constructed so that 95% of such ranges contain the truth.
12. **ANOVA**: One test that asks "do any of these groups differ?" instead of comparing them pair by pair.
13. **Power**: The probability of detecting a real effect when it exists — plan for it before you start.
14. **Random effect**: A source of variability representing a sample from a larger population, not a specific level you chose.
15. **Mixed model**: A model that handles both fixed effects (what you care about) and random effects (the messy context).
16. **ICC**: How much of the total variability comes from the grouping structure — your diagnostic for whether mixed models matter.
17. **Autocorrelation**: How much a signal remembers its own past.
18. **Bayes' theorem**: A rule for updating beliefs when new evidence arrives — posterior is proportional to likelihood times prior.
19. **Nuisance parameter**: Something you must model but don't care about — a ghost in the machine you have to catch.
20. **Overfitting**: When your model memorizes the noise instead of learning the signal.
