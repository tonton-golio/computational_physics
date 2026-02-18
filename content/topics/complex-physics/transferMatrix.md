# 1D Ising Model and Transfer Matrix Method

Think of a one-dimensional chain of spins as a book. Each page (spin) can be "up" or "down." The energy cost depends only on whether two neighboring pages match or not. The transfer matrix is a clever way of multiplying the probabilities page by page until you reach the end of the book — and then the eigenvalues pop out and hand you the exact answer.

This is one of the rare cases in statistical mechanics where we can solve the many-body problem *exactly*. No approximations, no Monte Carlo, no mean-field hand-waving. Just linear algebra.

## Hamiltonian

The Hamiltonian of the 1D Ising model with periodic boundary conditions ($s_{N+1} = s_1$) is
$$
    \mathcal{H}
    =
    -J\sum_{i=1}^{N} s_i s_{i+1} - h \sum_{i=1}^{N} s_i.
$$

## Partition function

The partition function is a sum over all $2^N$ configurations:
$$
\begin{align*}
    Z
    &=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \exp
    \left(
        \beta J\sum_{i=1}^{N} s_i s_{i+1}
        + \beta h \sum_{i=1}^{N} s_i
    \right).
\end{align*}
$$

The trick is to split the external field term equally between each pair of neighbors:
$$
    \beta h \sum_i s_i = \beta h \sum_i \frac{s_i + s_{i+1}}{2},
$$
so the whole exponent factorizes into a product of identical "bond" terms:
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

Each factor depends only on two neighboring spins. That is a matrix element waiting to happen.

## Transfer matrix

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

Written out explicitly as a $2 \times 2$ matrix:
$$
    T
    =
    \begin{bmatrix}
    e^{\beta(J+h)} & e^{-\beta J} \\
    e^{-\beta J} & e^{\beta(J-h)}
    \end{bmatrix}.
$$

Now here is where the magic happens. The partition function involves summing products of these matrix elements over all intermediate spins — but that is exactly what matrix multiplication does! Summing over $s_2$ gives $(T^2)_{s_1, s_3}$, then summing over $s_3$ gives $(T^3)_{s_1, s_4}$, and so on:
$$
\begin{align*}
    Z
    &=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    T_{s_1, s_2} T_{s_2, s_3} \cdots T_{s_N, s_1}
    \\&=
    \sum_{s_1 = \pm 1}
    (T^N)_{s_1, s_1}
    \\&=
    \mathrm{Tr}(T^N).
\end{align*}
$$

The partition function is the trace of the $N$-th power of the transfer matrix. We have converted the statistical mechanics problem into a linear algebra problem.

## Diagonalization and eigenvalues

Since $T$ is a real symmetric matrix, it is diagonalizable: $T = PDP^{-1}$, where $D$ is the diagonal matrix of eigenvalues. Then
$$
    T^N = PD^N P^{-1},
$$
and using the cyclic property of the trace:
$$
    Z = \mathrm{Tr}(T^N) = \mathrm{Tr}(D^N) = \lambda_1^N + \lambda_2^N.
$$

The entire partition function reduces to the eigenvalues of a $2 \times 2$ matrix. That is the power of the transfer matrix method.

## Finding the eigenvalues

The characteristic equation $\det(T - \lambda I) = 0$ gives:
$$
    \lambda^2
    -
    2\lambda \, e^{\beta J} \cosh(\beta h)
    +
    2 \sinh(2\beta J)
    = 0.
$$

Solving the quadratic:
$$
    \lambda_{\pm}
    =
    e^{\beta J} \cosh(\beta h)
    \pm
    \sqrt{
        e^{2\beta J} \sinh^2(\beta h)
        + e^{-2\beta J}
    }.
$$

Since $\lambda_+ > \lambda_-$ always, in the thermodynamic limit ($N \to \infty$) the smaller eigenvalue becomes negligible:
$$
    Z \approx \lambda_+^N.
$$

The free energy per spin is
$$
    f = -\frac{1}{\beta N} \ln Z = -\frac{1}{\beta} \ln \lambda_+.
$$

## The punchline: no phase transition in 1D

For $h = 0$, the eigenvalues simplify to
$$
    \lambda_+ = e^{\beta J} + e^{-\beta J} = 2\cosh(\beta J), \qquad
    \lambda_- = e^{\beta J} - e^{-\beta J} = 2\sinh(\beta J).
$$

Both eigenvalues are positive and analytic (smooth) functions of temperature for all $T > 0$. Since the free energy $f = -(1/\beta)\ln\lambda_+$ is an analytic function of $T$, there is *no singularity* at any finite temperature. No singularity means no phase transition.

This is the exact confirmation of what we suspected: in one dimension, thermal fluctuations always win. A single domain wall (a place where neighboring spins disagree) costs energy $2J$ but increases entropy by $k_\mathrm{B} \ln N$ (it can be placed at any of $N$ bonds). For large enough $N$, the entropy always beats the energy, and domain walls proliferate, destroying any long-range order.

Mean-field theory predicted a phase transition in 1D — and that prediction is wrong. This is a concrete example of why mean-field theory fails in low dimensions: it ignores the fluctuations that matter most.

> **Key Intuition.** The transfer matrix converts the partition function into a trace of a matrix power: $Z = \mathrm{Tr}(T^N)$. In the thermodynamic limit, only the largest eigenvalue matters. If the largest eigenvalue is analytic (no singularity), there is no phase transition. In 1D, the largest eigenvalue is always smooth — so there is no phase transition at finite temperature. Entropy always wins over energy in one dimension.

> **Challenge.** For the 1D Ising model with $h = 0$, compute the average energy per spin $\langle E \rangle / N = -\partial \ln \lambda_+ / \partial \beta$. You should get $-J\tanh(\beta J)$. Check that this interpolates between $0$ (high $T$) and $-J$ (low $T$). Does it make physical sense?

---

*We solved the one-dimensional case exactly — beautiful, but it has no phase transition. In real life, things are higher-dimensional and messy. So now we zoom out and ask a much deeper question: what do all critical points have in common, regardless of the system? That is the subject of critical phenomena and universality.*
