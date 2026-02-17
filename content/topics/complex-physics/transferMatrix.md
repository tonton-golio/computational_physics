# 1D Ising Model and Transfer Matrix Method

## Hamiltonian
We can obtain partition function of 1D-lattice Ising model can
analytically.
Hamiltonian of 1D-lattice Ising model is
$$
    \mathcal{H}
    =
    -J\sum_{i=1}^{N} s_i s_{i+1} - h \sum_{i=1}^{N} s_i
$$
with periodic boundary condition $s_{N+1} = s_{1}$.
## Partition function
Partition function becomes
$$
\begin{align*}
    Z
    &=
    \sum_{\{s_i\}}
    e^{-\beta \mathcal{H}\left(\{s_i\}\right)}
    \\&=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \exp
    \left(
        \beta J\sum_{i=1}^{N} s_i s_{i+1}
        + \beta h \sum_{i=1}^{N} s_i
    \right)
    \\&=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \exp
    \left[
        \beta J \left( s_1 s_2 + \cdots + s_N s_1 \right)
        + \beta h \left( s_1 + \cdots + s_N \right)
    \right]
    \\&=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \exp
    \left(
        \beta J s_1 s_2
        + \cdots
        + \beta J s_N s_1
        + \beta h \frac{s_1+s_2}{2}
        + \cdots
        + \beta h \frac{s_N+s_1}{2}
    \right)
    \\&=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \exp
    \left(
        \beta J s_1 s_2
        + \beta h \frac{s_1+s_2}{2}
    \right)
    \cdots
    \exp
    \left(
        \beta J s_N s_1
        + \beta h \frac{s_N+s_1}{2}
    \right)
    \\&=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \prod_{i=1}^N
    \exp
    \left(
        \beta J s_i s_{i+1}
        + \beta h \frac{s_i+s_{i+1}}{2}
    \right)
\end{align*}
$$
## Transfer matrix
$$
    T_{s_i, s_{i+1}}
    =
    \exp
    \left(
        \beta J s_i s_{i+1}
        + \beta h \frac{s_i+s_{i+1}}{2}
    \right)
$$
is a element of transfer matrix $T$. Transfer matrix is
$$
\begin{align*}
    T
    &=
    \begin{bmatrix}
    T_{+1, +1} & T_{+1, -1} \\
    T_{-1, +1} & T_{-1, -1}
    \end{bmatrix}
    \\&=
    \begin{bmatrix}
    \exp \left[ \beta(J+h) \right] & \exp \left[ -\beta J \right] \\
    \exp \left[ -\beta J \right] & \exp \left[ \beta(J-h) \right]
    \end{bmatrix}
\end{align*}
$$
From the definition of matrix multiplication, we see
$$
    \left(T^2\right)_{s_i, s_{i+2}}
    =
    \sum_{s_{i+1} = \pm1}
    T_{s_i, s_{i+1}}
    T_{s_{i+1}, s_{i+2}}
$$
Using these let's caliculate partition function!
$$
\begin{align*}
    Z
    &=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \prod_{i=1}^N
    T_{s_i, s_{i+1}}
    \\&=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    T_{s_1, s_2}
    \cdots
    T_{s_N, s_1}
    \\&=
    \sum_{s_1 = \pm 1}
    \sum_{s_3 = \pm 1}
    \cdots
    \sum_{s_N = \pm 1}
    \left(
    \sum_{s_2 = \pm 1}
    T_{s_1, s_2}
    T_{s_2, s_3}
    \right)
    T_{s_3, s_4}
    \cdots
    T_{s_N, s_1}
    \\&=
    \sum_{s_1 = \pm 1}
    \sum_{s_3 = \pm 1}
    \cdots
    \sum_{s_N = \pm 1}
    \left(T^2\right)_{s_1, s_3}
    T_{s_3, s_4}
    \cdots
    T_{s_N, s_1}
    \\&=
    \sum_{s_1 = \pm 1}
    \cdots
    \sum_{s_N = \pm 1}
    \left(
    \sum_{s_3 = \pm 1}
    \left(T^2\right)_{s_1, s_3}
    T_{s_3, s_4}
    \right)
    T_{s_4, s_5}
    \cdots
    T_{s_N, s_1}
    \\&=
    \cdots
    \\&=
    \sum_{s_1 = \pm 1}
    \left(
    \sum_{s_N = \pm 1}
    \left(T^{N-1}\right)_{s_1, s_N}
    T_{s_N, s_1}
    \right)
    \\&=
    \sum_{s_1 = \pm 1}
    \left(T^{N}\right)_{s_1, s_1}
    \\&=
    \mathrm{Tr}
    \left(
    T^{N}
    \right)
\end{align*}
$$
Here $T$ is a real symmetric matrix which is diagonalizable.
$$
    T = PDP^{-1}
$$
where $D$ is a diagonal matrix with eigenvalues $\lambda_1$ and
$\lambda_2$
$$
    D
    =
    \begin{bmatrix}
    \lambda_1 & 0 \\
    0 & \lambda_2
    \end{bmatrix}
$$
and $P$ is a matrix containing eigenvectors.
Thanks to this diagonalization $T^{N}$ becomes simpler.
$$
\begin{align*}
    T^{N}
    &=
    PDP^{-1}PDP^{-1} \cdots PDP^{-1}
    \\&=
    PD^{N}P^{-1}
\end{align*}
$$
Using the property of trace,
$$
\begin{aligned}
    \operatorname{Tr}(A B) &=\sum_i(A B)_{i i} \\
    &=\sum_i \sum_j A_{i j} B_{j i} \\
    &=\sum_j \sum_i B_{j i} A_{i j} \\
    &=\sum_j(B A)_{j j} \\
    &=\operatorname{Tr}(B A)
\end{aligned}
$$
we obtain
$$
\begin{aligned}
    \mathrm{Tr}
    \left(
        PD^{N}P^{-1}
    \right)
    &=
    \mathrm{Tr}
    \left(
        D^{N}P^{-1}P
    \right)
    \\&=
    \mathrm{Tr}
    \left(
        D^{N}
    \right)
\end{aligned}
$$
Returning to partition function
$$
\begin{aligned}
    Z
    &=
    \mathrm{Tr}
    \left(
        T^{N}
    \right)
    \\&=
    \mathrm{Tr}
    \left(
        D^{N}
    \right)
    \\&=
    \lambda_1^N + \lambda_2^N
\end{aligned}
$$
Cool. We need eigenvalue of transfer matrix.
$$
\begin{aligned}
    0
    &=
    \operatorname{det}(T-\lambda)
    \\&=
    \left|\begin{array}{cc}
    \exp\left({\beta J+ \beta h}\right)-\lambda & \exp\left({-\beta J}\right) \\
    \exp\left(-\beta J\right) & \exp\left(\beta J - \beta h\right)-\lambda
    \end{array}\right|
    \\&=
    \left(\exp\left(\beta J+ \beta h\right)-\lambda\right)
    \left(\exp\left(\beta J- \beta h\right)-\lambda\right)
    -\exp\left(-2 \beta J\right)
    \\&=
    \exp\left(2\beta J\right)
    -\lambda
    \left(
        \exp\left(\beta J + \beta h\right)
        +
        \exp\left(\beta J - \beta h\right)
    \right)
    +\lambda^2
    -\exp\left(-2 \beta J\right)
\end{aligned}
$$
We obtaiend quadratic equation of $\lambda$.
$$
\begin{aligned}
    0
    &=
    \lambda^2
    -
    \lambda
    \left[
            \exp\left(\beta J+ \beta h\right)
        +\exp\left(\beta J- \beta h\right)
    \right]
    +
    \left[
    \exp\left(2\beta J\right)
    - \exp\left(-2 \beta J\right)
    \right]
    \\&=
    \lambda^2
    -
    2 \lambda \exp \left(\beta J\right) \cosh \left(\beta h\right)
    +
    2 \sinh \left(2 \beta J\right)
\end{aligned}
$$
This equation has two solutions.
$$
\begin{aligned}
    \lambda
    &=
    \exp \left(\beta J\right) \cosh \left(\beta h\right)
    \pm
    \sqrt{
        \exp \left(2 \beta J\right) \cosh^2 \left(\beta h\right)
        -
        2 \sinh \left(2 \beta J\right)
    }
    \\&=
    \exp \left(\beta J\right) \cosh \left(\beta h\right)
    \pm
    \sqrt{
        \exp \left(2 \beta J\right)
        +
        \exp \left(2 \beta J\right)
        \sinh^2 \left(\beta h\right)
        -
        \left(
            \exp\left(2 \beta J\right)
            +
            \exp\left(-2 \beta J\right)
        \right)
    }
    \\&=
    \exp \left(\beta J\right) \cosh \left(\beta h\right)
    \pm
    \sqrt{
        \exp \left(2 \beta J\right)
        \sinh^2 \left(\beta h\right)
        -
        \exp\left(-2 \beta J\right)
    }
\end{aligned}
$$
