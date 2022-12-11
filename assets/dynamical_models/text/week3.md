
# title
Probability for Mutational Analysis

# Week 3 description
- Dec 5: Fundamentals of mutations and probability for mutational analysis

# What causes mutations?
There are two type of mutations.
- Spontaneous (chemical mistakes in DNA replication, $\sim 5000$ times a day in a human cell)
    - Depurination (Adenine or guanine base cleaving)
    - Deamination (Cytosine become uracil)
- Induced (chemical change by UV light and radiation, $\sim 100$ times a day in a human cell)

# Fidelity in DNA replication and gene expression
Accuracy (mistake per bases) of each gene expression process:
DNA replication ($1/10^9$) > transcription ($1/10^6$) > translation ($1/10^4$)

# Proofreading in DNA replication
Base pairing is largely determined by hydrogene bond.
However free energy difference of correct base pair and wrong one is only 
4-13 kJ/mol which is not enough to observe error rate $10^{-9}$.
What kind of additional process is necessary to achive this fidelity?
## Edting by DNA polymerase
DNA polymerase can not only do polemerizing but also editing.
If newly added base is wrong, they remove it.
## Cellular enzyme (strand-directed mismatch repair)
Many cellular enzyme are dedicated to DNA repair. 
Some enzyme removes errors by recruiting DNA polymerase.

# Recombination
During meiosis crossing-over happens and it contribute genetic diversity.

# Mutants
Due to their small size and fast growth rate, we can easily grow a billion bacteria
overnight in 1mL medium.
## Genetic selection
We can only grow mutants. 
For example, We can only grow bacteria with T14 receptors.

# Distribution for mutation
Consider the case we observe the cell division and count the number of mutated cell.

# Binomial distribution
$$
    P_N(k)
    =
    \frac{N!}{(N-k)!k!} p^k (1-p)^{(N-k)}
$$

# Poisson distribution
When we take a limit of $N\rightarrow\infty$ with $m=Np=\mathrm{const.}$, we get 
Poisson distribution.
$$
    P_m(k)
    =
    \frac{m^k}{k!} \exp(-m)
$$

# Binomial vs Poisson
If we use large $N$ in binomial distribution, it approaches the Poisson 
distribution (with $Np=m=const.$).
