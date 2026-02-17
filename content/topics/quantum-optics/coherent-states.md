# Coherent States
__Topic 3 keywords__
- Coherent states
- Displacement operator
- Generation of coherent states

# Readings
Ch. 3.1-5

# 3.1 Eigenstates of the annihilation operator and minimum uncertainty states
#### Definition of coherent state
Coherent states are defined as eigenstate of annihilation operator.
$$
    \boxed{
        \hat{a}
        \Ket{\alpha}
        =
        \alpha
        \Ket{\alpha}
    }
$$
$$
    \boxed{
        \hat{a}^\dag
        \Bra{\alpha}
        =
        \alpha^*
        \Bra{\alpha}
    }
$$
As you know that $\hat{a}$ is non-Hermitian so the eigenvalue **$\alpha$ is complex
number**.
By the way $\hat{n}=\hat{a}^\dag\hat{a}$ is Hermitian.

#### Expanding coherent state with number state
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
You know, this is just a definition.
Multiplying $\hat{a}$ from left results
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
\boxed{
    \Ket{\alpha}
    =
    e^
    {- \left| \alpha \right|^{2}/2}
    \sum_{n=0}^\infty
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \Ket{n}
}
$$
