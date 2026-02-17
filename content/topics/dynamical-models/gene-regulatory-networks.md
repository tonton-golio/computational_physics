# Gene Regulatory Networks

## Types of Regulation

There are two types of regulation:
- **Positive regulation**: Gene A activates/increases gene B, symbolized by $A \rightarrow B$
- **Negative regulation**: Gene A represses/decreases gene B, symbolized by $A \dashv B$

By combining these simple types of regulation, genes are controlled. Often, gene regulatory networks are more complicated than what appears to be necessary at first sight.

## Statistics of Regulatory Functions

As the genome size increases, the percentage of gene regulatory motifs increases.

## Real Networks and How We Understand Gene Regulation

Even the fruit fly has an enormously complex network. What can we learn from these networks?

We focus on subparts, i.e., **motifs** of networks, and investigate how "core" key dynamics behave. This is a reductionist approach: we can say something about particular examples but cannot always make general statements.

## Types of Network Motifs

Several types of motifs have been identified:
- **Positive feedback**: $A \leftrightarrow B$
- **Negative feedback**: $A \rightarrow B \dashv A$
- **Feed-forward loops**: $A \rightarrow B \rightarrow C$, $A \rightarrow C$
- **Single input modules**: $A \rightarrow B$, $A \rightarrow C$, $A \rightarrow D$

## Identifying Positive and Negative Feedback

Find a closed loop and multiply the interaction types as integers: positive regulation ($\rightarrow$) as $+1$, and negative regulation ($\dashv$) as $-1$. For example,

$$
A \rightarrow B \rightarrow C \dashv A
$$

gives $(+1)(+1)(-1) = -1$, so this is a **negative feedback loop**.

## Biological Examples of Positive/Negative Regulation

- Phage lambda repressor (CI)
- Bistability in *comK* expression
- ppGpp signaling by the ribosome
- Cell-to-cell communication

## Biological Examples of Feed-Forward Loops

- **AND logic**: *ara genes* need two activators, CRP and AraC. The network is $\mathrm{CRP} \rightarrow \mathrm{AraC} \rightarrow ara$, $\mathrm{CRP} \rightarrow ara$.
- **OR logic**: Flagella of bacteria.

## Biological Examples of Single Input Modules

- Flagellar genes

## Simplification of Dynamical Equations

Biologically, transcription and translation are two independent processes, so we need to build equations for each. However, from a mathematical viewpoint, it is convenient to approximate the transcription process as part of the translation process. We can write an equation for protein concentration time evolution:

$$
\frac{\mathrm{d}P}{\mathrm{d}t} = \frac{\alpha_\mathrm{P} \alpha_\mathrm{m} (P/K)^H}{\Gamma_\mathrm{H} (1+(P/K)^H)} - \Gamma_\mathrm{p} P
$$

## Obtaining Steady-State Concentration from a Graph

To see the parameter-free typical behavior of this equation, we substitute $1$ for most parameters. The equation becomes:

$$
\frac{\mathrm{d}P}{\mathrm{d}t} = \frac{P^H}{1+P^H} - \Gamma_\mathrm{p}P
$$

By plotting $y = P^H/(1+P^H)$ and $y = \Gamma_\mathrm{p}P$ on the same plane, we can visually inspect the steady-state concentration of $P$.

## Negative Regulation

The same analysis applies to negative regulation:

$$
\frac{\mathrm{d}P}{\mathrm{d}t} = \frac{1}{1+P^H} - \Gamma_\mathrm{p}P
$$

[[simulation hill-function]]
