# Dynamical Models in Molecular Biology

## Course overview

This course applies the tools of **physics and mathematics** to understand how living systems regulate themselves at the molecular level.
Unlike classical biochemistry, which catalogs pathways and components, we build quantitative, predictive models of gene expression, regulation, and growth.

- Biology provides the systems: genes, proteins, regulatory circuits, and growing cells.
- Physics provides the framework: differential equations, stochastic processes, and steady-state analysis.
- Mathematics provides the rigor: exact solutions, stability analysis, and parameter estimation.

We focus on simple systems, often bacterial, but seek general rules that apply broadly across biology.

## Why this topic matters

- Gene expression is inherently **stochastic**, and noise shapes cell-fate decisions.
- **Feedback loops** in regulatory networks produce switches, oscillations, and memory.
- **Signal transduction** allows cells to sense and adapt to their environment with remarkable specificity and sensitivity.
- Quantitative **growth laws** connect molecular processes to whole-cell physiology.
- **Mutations** are both the raw material of evolution and a fundamental experimental tool.

## Key mathematical ideas

- Ordinary differential equations for production and degradation kinetics.
- The Hill function as a universal model for cooperative binding.
- Stochastic simulation (Gillespie algorithm) for single-cell noise.
- Fixed-point analysis and bifurcation diagrams for feedback circuits.
- Resource allocation models for bacterial growth.

## Prerequisites

- Basic calculus (derivatives and integrals).
- Introductory probability and statistics.
- Familiarity with basic molecular biology (DNA, RNA, protein).
- No advanced mathematics is required; differential equations are introduced from first principles.

## Recommended reading

- Phillips et al., *Physical Biology of the Cell*.
- Alon, *An Introduction to Systems Biology*.
- Weekly scientific articles from the quantitative biology literature.

## Learning trajectory

This module is organized from molecular-level processes to systems-level behavior:

- Gene expression noise: stochastic transcription and translation.
- Differential equations: modeling production, degradation, and steady state.
- Transcription and translation: coupled differential equations for mRNA and protein, number vs concentration.
- Transcriptional regulation: the Hill function, repression, activation, and sRNA regulation.
- Feedback loops: bistability, oscillations, and the repressilator.
- Gene regulatory networks: motifs, operons, and network architecture.
- Signal transduction: chemotaxis, adaptation, and cell-to-cell communication.
- Mutational analysis: screens, selections, and experimental design.
- Bacterial growth: growth laws, proteome allocation, and resource trade-offs.

## Visual and Simulation Gallery

[[simulation gene-expression-noise]]

[[simulation hill-function]]

[[simulation binomial-poisson-comparison]]
