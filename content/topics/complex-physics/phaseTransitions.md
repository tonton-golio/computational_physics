# Phase Transitions

## Ising Model Simulation

The Ising model demonstrates phase transitions in magnetic systems. Watch how the system orders below the critical temperature.

[[simulation ising-model]]

## Mean-field Hamiltonian
By approximating the energy acting on a spin in the 1D Ising model, not as the sum of nearest neighbors but instead the mean of the chain, we may simplify interesting terms. We obtain the mean-field Hamiltonian,

$$
\begin{align*}
    \mathcal{H}_\mathrm{MF}
    &=
    \frac{J N z}{2} m^2
    - \left( J z m + h \right) \sum_i s_i.
\end{align*}
$$
Here $z$ is number of nearest-neighbor spins and division by 2 is
for avoiding overlap.

## Mean-field Hamiltonian (derivation)
#### Hamiltonian
The Hamiltonian of the Ising model is
$$
    \mathcal{H}
    =
    -J \sum_{\left<i j\right>} s_i s_j - h \sum_i s_i.
$$

We cannot caliculate partition function directly except for
1D-lattice case and 2D-lattice case.
However, by approximating Hamiltonian with mean field, we can
analytically obtain partition function.
Let's approxiomate Hamiltonian.
#### Ignoring high-order fluctuation
First, let's replace $s_i$ with mean $\left< s_i \right>$ and
fluctuation from mean $\delta s_i = s_i - \left< s_i \right>$.
$$
\begin{align*}
    s_i
    &=
    \left< s_i \right> + \delta s_i
    \\&=
    \left< s_i \right>
    + \left( s_i - \left< s_i \right> \right)
\end{align*}
$$
Here $\left< s_i \right>$ means
$$
    \left< s_i \right>
    =
    \frac{1}{Z} \sum_{n=1}^{2^N} s_i \exp \left( -\beta E_n \right).
$$
$n$ is index of state
(total number of all state is $2^N$, $N$ is number of spin).
Inside of sum of first term of Hamiltonian becomes
$$
\begin{align*}
    s_i s_j
    &=
    \left(
        \left< s_i \right> + \delta s_i
    \right)
    \left(
        \left< s_j \right> + \delta s_j
    \right)
    \\&=
    \left< s_i \right> \left< s_j \right>
    +
    \left< s_i \right> \delta s_j
    +
    \delta s_i \left< s_j \right>
    +
    \delta s_i \delta s_j
    \\& \approx
    \left< s_i \right> \left< s_j \right>
    +
    \left< s_i \right> \delta s_j
    +
    \delta s_i \left< s_j \right>
    \\&=
    \left< s_i \right> \left< s_j \right>
    +
    \left< s_i \right>
    \left(
        s_j - \left< s_j \right>
    \right)
    +
    \left(
        s_i - \left< s_i \right>
    \right)
    \left< s_i \right>
    \\&=
    \left< s_i \right> \left< s_j \right>
    +
    \left< s_i \right>
    \left(
        s_j - \left< s_j \right>
    \right)
    +
    \left(
        s_i - \left< s_i \right>
    \right)
    \left< s_i \right>
    \\&=
    -\left< s_i \right>^2
    +
    \left< s_i \right>
    \left(
        s_i + s_j
    \right)
\end{align*}
$$
We ignore the fluctuation with second order.
We also used
$$
\left< s_1 \right>
= \left< s_2 \right>
= \cdots
= \left< s_i \right>
= \cdots
= \left< s_N \right>
$$
because all spins are equivalent.

What we need to notice is that magnetization in equilibrium state
is equivalent to mean of spin $\left< s_i \right>$.
$$
\begin{align*}
    m
    &=
    \frac{1}{N} \sum_{i=1}^N \left< s_i \right>
    \\&=
    \frac{1}{N} \left< s_i \right> \sum_{i=1}^N
    \\&=
    \frac{1}{N} \left< s_i \right> N
    \\&=
    \left< s_i \right>
\end{align*}
$$
Thus we can replace $\left< s_i \right>$ with $m$.
$$
    s_i s_j
    \approx
    - m^2 + m(s_i + s_j)
$$
#### Mean-field Hamiltonian
Then, mean-field Hamiltonian $\mathcal{H}_\mathrm{MF}$ beocomes
$$
\begin{align*}
    \mathcal{H}_\mathrm{MF}
    &=
    -J \sum_{\left<i j\right>}
    \left(- m^2 + m(s_i + s_j) \right)
    - h \sum_i s_i
    \\&=
    J m^2 \sum_{\left<i j\right>}
    - J \sum_{\left<i j\right>} m(s_i + s_j)
    - h \sum_i s_i
\end{align*}
$$
Let's think about first term.
$$
\begin{align*}
    J m^2 \sum_{\left<i j\right>}
    &=
    J m^2 \frac{1}{2}
    \sum_{i} \sum_{\left<j\right>}
    \\&=
    J m^2 \frac{1}{2}
    \sum_{i=1}^N z
    \\&=
    \frac{J N z}{2} m^2
\end{align*}
$$
Here $z$ is number of nearest-neighbor spins and division by 2 is
for avoiding overlap.

Move on to second term.
$$
\begin{align*}
    - J \sum_{\left<i j\right>} m(s_i + s_j)
    &=
    - J m
    \left(
        \sum_{\left<i j\right>} s_i + \sum_{\left<i j\right>} s_j
    \right)
    \\&=
    - 2 J m \sum_{\left<i j\right>} s_i
    \\&=
    - 2 J m \frac{1}{2} \sum_{i} s_i \sum_{\left<j\right>}
    \\&=
    - J z m  \sum_{i} s_i
\end{align*}
$$
Finally, mean-field Hamiltonian becomes
$$
\begin{align*}
    \mathcal{H}_\mathrm{MF}
    &=
    \frac{J N z}{2} m^2
    - J z m  \sum_{i} s_i
    - h \sum_i s_i
    \\&=
    \frac{J N z}{2} m^2
    - \left( J z m + h \right) \sum_i s_i
\end{align*}
$$
