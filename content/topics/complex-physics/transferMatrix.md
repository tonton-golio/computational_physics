# 1D Ising Model and Transfer Matrix Method

Think of a one-dimensional chain of spins as a book. Each page can be "up" or "down." The energy cost depends only on whether two neighboring pages match or not. The transfer matrix is a clever way of multiplying the probabilities page by page until you reach the end -- and then the eigenvalues pop out and hand you the exact answer.

This is one of the rare cases in statistical mechanics where we can solve the many-body problem *exactly*. No approximations, no Monte Carlo, no mean-field hand-waving. Just linear algebra.

## Big Ideas

* The transfer matrix converts a statistical mechanics sum over $2^N$ configurations into a linear algebra problem -- the partition function is just $\mathrm{Tr}(T^N)$.
* In the thermodynamic limit, only the largest eigenvalue $\lambda_+$ survives: all thermodynamics comes from a single number.
* A phase transition requires a singularity in the free energy, which requires a singularity in $\lambda_+$. In 1D, $\lambda_+$ is always smooth -- so no phase transition exists at any finite temperature, ever.
* This isn't a failure of the model but a deep truth: in 1D, the entropy of domain walls always beats the energy cost of making them.

## Hamiltonian

The 1D Ising model with periodic boundary conditions ($s_{N+1} = s_1$):
$$
    \mathcal{H}
    =
    -J\sum_{i=1}^{N} s_i s_{i+1} - h \sum_{i=1}^{N} s_i.
$$

## Partition Function

The partition function sums over all $2^N$ configurations. The trick: split the external field equally between each pair of neighbors, so the exponent factorizes into identical "bond" terms:
$$
    Z =
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \prod_{i=1}^N
    \exp
    \left(
        \beta J s_i s_{i+1}
        + \beta h \frac{s_i+s_{i+1}}{2}
    \right).
$$

Each factor depends only on two neighboring spins. That's a matrix element waiting to happen.

## Transfer Matrix

Define the transfer matrix $T$ with elements
$$
    T_{s_i, s_{i+1}}
    =
    \exp
    \left(
        \beta J s_i s_{i+1}
        + \beta h \frac{s_i+s_{i+1}}{2}
    \right).
$$

Written as a $2 \times 2$ matrix:
$$
    T
    =
    \begin{bmatrix}
    e^{\beta(J+h)} & e^{-\beta J} \\
    e^{-\beta J} & e^{\beta(J-h)}
    \end{bmatrix}.
$$

Now here's where the magic happens. Summing over intermediate spin values is exactly matrix multiplication! Summing over $s_2$ gives $(T^2)_{s_1, s_3}$, then over $s_3$ gives $(T^3)_{s_1, s_4}$, and so on:
$$
    Z
    =
    \sum_{s_1 = \pm 1}
    (T^N)_{s_1, s_1}
    =
    \mathrm{Tr}(T^N).
$$

The partition function is the trace of the $N$-th power of the transfer matrix. We've converted a statistical mechanics problem into a linear algebra problem.

## Eigenvalues and the Thermodynamic Limit

Since $T$ is a real symmetric $2 \times 2$ matrix, it has two real eigenvalues $\lambda_+ > \lambda_-$:
$$
    Z = \lambda_+^N + \lambda_-^N, \qquad
    \lambda_{\pm}
    =
    e^{\beta J} \cosh(\beta h)
    \pm
    \sqrt{
        e^{2\beta J} \sinh^2(\beta h)
        + e^{-2\beta J}
    }.
$$
The entire partition function of $2^N$ configurations reduces to two eigenvalues of a $2 \times 2$ matrix. In the thermodynamic limit ($N \to \infty$), $\lambda_-^N / \lambda_+^N \to 0$, and the free energy per spin depends only on the largest eigenvalue:
$$
    f = -\frac{1}{\beta} \ln \lambda_+.
$$

[[simulation transfer-matrix-demo]]

## The Punchline: No Phase Transition in 1D

For $h = 0$, the eigenvalues simplify to
$$
    \lambda_+ = 2\cosh(\beta J), \qquad
    \lambda_- = 2\sinh(\beta J).
$$

Both eigenvalues are positive and smooth functions of temperature for all $T > 0$. Since the free energy $f = -(1/\beta)\ln\lambda_+$ is smooth, there's no singularity. No singularity means no phase transition.

And here's the gorgeous part: both eigenvalues are always positive and smooth -- no place for a sudden jump. That single fact tells you there is no phase transition in one dimension. Nature refuses to freeze in a line.

Why? A single domain wall costs energy $2J$ but gains entropy $k_\mathrm{B} \ln N$ (it can sit at any of $N$ bonds). For large enough $N$, entropy always wins, and domain walls proliferate, destroying any long-range order.

Mean-field theory predicted a 1D phase transition -- and got it wrong. This is a concrete example of why mean-field fails in low dimensions: it ignores the fluctuations that matter most.

## What Comes Next

The transfer matrix gave us an exact result, but it was a small-system trick -- beautiful in 1D, not straightforwardly generalizable. The next question is bigger and more mysterious: why do completely different systems -- magnets, fluids, alloys -- behave *identically* near their critical points? [Critical Phenomena](criticalPhenomena) introduces universality classes and the renormalization group, which explain this miracle by showing that microscopic details become irrelevant at the critical point.

## Check Your Understanding

1. Explain in words why summing over intermediate spin values is equivalent to matrix multiplication.
2. The eigenvalues for $h = 0$ are $\lambda_+ = 2\cosh(\beta J)$ and $\lambda_- = 2\sinh(\beta J)$. Both are positive and analytic for all $T > 0$. Why does this immediately rule out a phase transition?

## Challenge

For the 1D Ising model with $h = 0$, compute the spin-spin correlation function $\langle s_0 s_r \rangle$ using the transfer matrix. You should find $\langle s_0 s_r \rangle = \tanh^r(\beta J)$. Extract the correlation length $\xi$. How does $\xi$ behave as $T \to 0$? Does it diverge? What does this tell you about why there is no phase transition -- even though the spins want to align at low temperature?
