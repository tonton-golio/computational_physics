# Coherent States

## Eigenstates of the annihilation operator and minimum uncertainty states
### Definition of coherent state
Coherent states are defined as eigenstate of annihilation operator.
$$
    \hat{a}
    \Ket{\alpha}
    =
    \alpha
    \Ket{\alpha}
$$
$$
    \hat{a}^\dag
    \Bra{\alpha}
    =
    \alpha^*
    \Bra{\alpha}
$$

> **Key Intuition.** A coherent state is defined as an eigenstate of the annihilation operator. Because the annihilation operator is non-Hermitian, the eigenvalue is a complex number, encoding both the amplitude and phase of the field.

Since $\hat{a}$ is non-Hermitian, the eigenvalue **$\alpha$ is complex
number**.
Note that $\hat{n}=\hat{a}^\dag\hat{a}$ is Hermitian.

### Expanding coherent state with number state
Let's expand coherent state with number state.
First, we multiply identity $\sum_{n=0}^\infty \Ket{n}\Bra{n}$ to alpha.
By this operation, we can project coherent state to number state.
$$
\begin{aligned}
    \Ket{\alpha}
    &=
    \sum_{n=0}^\infty
    \Ket{n}
    \Braket{n|\alpha}
    \\&=
    \sum_{n=0}^\infty
    c_n
    \Ket{n}
\end{aligned}
$$
We replaced the $\Braket{n|\alpha}=c_n$.

Apply operator to $\Ket{\alpha}$ resuts to
$$
\begin{aligned}
    \hat{a}
    \Ket{\alpha}
    &=
    \alpha
    \Ket{\alpha}
    \\&=
    \sum_{n=0}^\infty
    \alpha
    c_n
    \Ket{n}
\end{aligned}
$$
This follows directly from the definition.
Multiplying $\hat{a}$ from left gives
$$
\begin{aligned}
    \hat{a}
    \sum_{n=0}^\infty
    c_n
    \Ket{n}
    &=
    \sum_{n=1}^\infty
    \sqrt{n}
    c_n
    \Ket{n-1}
    \\&=
    \sum_{n=0}^\infty
    \sqrt{n+1}
    c_{n+1}
    \Ket{n}
\end{aligned}
$$
Coefficient of number state is equal.
$$
    \sqrt{n+1}
    c_{n+1}
    =
    \alpha
    c_n
$$
Thus, we can decide the $c_n$ recursively.
$$
\begin{aligned}
    c_n
    &=
    \frac
    {\alpha}
    {\sqrt{n}}
    c_{n-1}
    \\&=
    \frac
    {\alpha^2}
    {\sqrt{n(n-1)}}
    c_{n-2}
    \\&=
    \cdots
    \\&=
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    c_{0}
\end{aligned}
$$
We almost complete expanding coherent state with number state.
$$
    \Ket{\alpha}
    =
    c_0
    \sum_{n=0}^\infty
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \Ket{n}
$$
We still need to decide $c_0$. We can do this from normalization condition.
$$
\begin{aligned}
    1
    &=
    \Braket{\alpha|\alpha}
    \\&=
    \left| c_0 \right|^2
    \sum_{n, n^\prime}
    \frac
    {\left| \alpha \right|^{2n}}
    {n!}
    \Braket{n|n^\prime}
    \left| c_0 \right|^2
    \\&=
    \sum_{n=0}^\infty
    \frac
    {\left| \alpha \right|^{2n}}
    {n!}
    \\&=
    \left| c_0 \right|^2
    e^
    {\left| \alpha \right|^{2}}
\end{aligned}
$$
Thus,
$$
    \left| c_0 \right|^2
    =
    e^
    {- \left| \alpha \right|^{2}}
$$
$$
    c_0
    =
    e^
    {- \left| \alpha \right|^{2}/2}
$$
Finally coherent coherent state of number state basis is
$$
    \Ket{\alpha}
    =
    e^
    {- \left| \alpha \right|^{2}/2}
    \sum_{n=0}^\infty
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \Ket{n}
$$

> **Key Intuition.** This expansion shows that a coherent state is a specific superposition of all number states, weighted by a Poissonian distribution. The prefactor ensures normalization, and the complex parameter $\alpha$ fully determines the state.

## Big Ideas

* A coherent state is the eigenstate of the annihilation operator — this defines it cleanly, but it is non-Hermitian so the eigenvalue $\alpha$ must be complex.
* Expanding $|\alpha\rangle$ in the number-state basis reveals Poissonian photon statistics: the probability of finding $n$ photons follows $P_n = e^{-|\alpha|^2}|\alpha|^{2n}/n!$.
* The complex number $\alpha$ encodes both the amplitude ($|\alpha|$ is the square root of mean photon number) and the phase of the field.
* Coherent states are the quantum states that best mimic a classical monochromatic wave — laser light is well described by them.

## What Comes Next

Now that we have the coherent state defined and expanded in the Fock basis, the next lesson calculates what it actually looks like as an electromagnetic field. We will find that the expectation value of the electric field oscillates sinusoidally just like a classical wave, and that the field fluctuations are remarkably independent of how strong the field is.

## Check Your Understanding

1. The annihilation operator $\hat{a}$ is non-Hermitian, meaning its eigenvalues are complex. Why does this make physical sense for a coherent state, given that $\alpha$ encodes both the amplitude and phase of an oscillating field?
2. The expansion of $|\alpha\rangle$ in number states involves all $|n\rangle$ from $n=0$ to infinity. Intuitively, why must a coherent state involve an indefinite photon number, even though the field amplitude is definite?
3. If you double the amplitude $\alpha \to 2\alpha$, the mean photon number increases by a factor of four. Explain why this quadratic relationship between amplitude and intensity is both expected classically and confirmed quantum mechanically.

## Challenge

A number state $|n\rangle$ and a coherent state $|\alpha\rangle$ with $|\alpha|^2 = n$ both have the same mean photon number. Compare their photon number distributions. Show explicitly that the variance of the photon number is $n$ for the coherent state and $0$ for the number state. Then construct a state that has a photon number distribution intermediate between these two extremes — for instance, a superposition $c_0|n\rangle + c_1|\alpha\rangle$ — and discuss what determines the variance of this mixture.
