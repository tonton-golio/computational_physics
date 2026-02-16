# Gene Regulatory Networks

# Week 5 description
- Dec 19: How is gene expression is regulated by networks?

# Type of regulation
There are two type of regulation. 
- Positive regulation (gene A activates/increses gene B, symbolized by $A\rightarrow B$
- Negative regulation (gene A represses/decreases gene B, symbolized by $A\dashv B$
By combining this simple type of regulation, gene are controlled.
Often, gene regulatory networks are more complicated than what appears to be 
necessary at first sight.

# Statistics of regulatory function
As the genome size increases, the percentage of gene regulatory motifs increases.

# Real network and how we understand gene regulation
Even fruit fly has enourmous complex network.
What can we know from this networks?
We focus on subpart i.e. motif of network and investigate how "core"/key dynamics 
behaves.
This is a kind of disappoings. You know we can say something about particular 
example but we never say something general.

# Type of network motif
Several type of motif are reported.
- Positive feed back ($A \leftrightarrow B$)
- Negative feed back ($A \rightarrow B \dashv A$)
- Feed-forward loops ($A \rightarrow B \rightarrow C, A \rightarrow C$)
- Single input modules ($A \rightarrow B, A \rightarrow C, A \rightarrow D$)

# How can we find positive feedback and negative feedback
Find closed loop and multiply the type as integer i.e. positive regulation 
$\rightarrow$ as $+1$, and negative regulation $\dashv$ as $-1$. 
For example, 
$$ 
    A \rightarrow B \rightarrow C \dashv A
$$
is negative feedback loop.

# Biological example of positive/negative regulation
- Phage lambda repressor (CI)
- Bistability in *comK* expression
- ppGpp signaling by the robosome
- Cell-to-cell communication

# Biological example of feed-forward loops
- AND logic. *ara genes* needs two activator, CRP and AraC. The network is $\mathrm{CRP} \rightarrow \mathrm{AraC} \rightarrow ara$, $CRP \leftrightarrow ara$.
- OR logic. Flagella of bacteria.

# Biological example of single input modules
- Flagellar genes

# Simplification of dynamical equation
Biologically transcription and translation is two independet process. So we need to
build equation for each. However mathematical viewpoint, it is convenient to 
approximate transcription process as a part of translation process.
We can write equation for protein concentration time evolution.
$$
    \frac{\mathrm{d}P}{\mathrm{d}t}
    =
    \frac
    {\alpha_\mathrm{P} \alpha_\mathrm{m} (P/K)^H}
    {\Gamma_\mathrm{H} (1+(P/K)^H)}
    - \Gamma_\mathrm{p} P
$$

# Obtaining steady state concentration from graph
To see the parameter-free typical behavior of this equation, we substitute $1$ for 
most of parameters. The equation become
$$
    \frac{\mathrm{d}P}{\mathrm{d}t}
    =
    \frac{P^H}{1+P^H} - \Gamma_\mathrm{p}P
$$
By plotting $y = \frac{P}{1+P}$ and $y=P$ on the same $yP$ plane, we can visually
inspect steady state concentration of $P$.

# How about negative regulation?
Same can be done on negative regulation!
$$
    \frac{\mathrm{d}P}{\mathrm{d}t}
    =
    \frac{1}{1+P^H} - \Gamma_\mathrm{p}P
$$

