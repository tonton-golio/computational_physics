 __Topic 1 keywords__
- Maxwell equations
- Single mode fields
- Creation/annihilation operators
- Number state
- Field fluctuations

# __Readings__
Ch. 2.1-3

# __Why do we need to quantize?__
- So far, we learned that electron in high-energy state decays back to ground state with the emission of light.
- However, according to the Schrödinger equation $$\hat{H} \Ket{\psi_m} = E_m \Ket{\psi_m}$$, electron is stable in excited state i.e. *electron stays in high energy state*.
- How do the decay? Something goes wrong. Why does spontaneous emission happen?
- The answer is **fluctuations**. 
- The new question is *what kind of fluctuations do they have*?

# __Can classical electromagnetic describe this fluctuations?__
In classical electromagnetic, energy of electron in electric field $E(t)$ is 
$$ e \cdot E(t) $$
Thus, when there is no light, there is no fluctuation.
However, it is known that even without light, there is a fluctuation i.e. quantum
mechanical fluctuation.
We need to use quantum mechanics.

# __2.1 Quantization of Single Mode Field__
#### Motivation
Let's start our quantization of electromagnetic waves.
What are we gonna quantize? Of course, as you know, it is **energy**.
#### Situation
Imagine a three-dimensional box and there are perfectly conducting walls at 
$z=0$ and $z=L$. 
We consider **single mode field** i.e. we only take care one mode.
Boundary conditions are
- $\vec{E}(z=0)=0$
- $\vec{E}(z=L)=0$
#### Energy
Electromagnetic energy is
$$
    H 
    = 
    \frac{1}{2} 
    \int \mathrm{d} V
    \left[
        \epsilon_0 \left| \vec{E}(z, t) \right|^2
        + 
        \frac{1}{\mu_0} \left| \vec{B}(z, t) \right|^2
    \right]^2
$$
#### Electric field
From boundary condition, we can determine the spatial part of electric field.
$$
    \vec{E}(z, t)
    =
    A \cdot q(t) \cdot \sin(kz) \cdot \vec{e_x}
$$
Here, parameters are described as follows.
- $A$ is a constant and we can decide. 
- $q(t)$ is the time dependent part and we will see why we use $q$ as symbol.
- $k=\frac{2\pi}{\lambda}=\frac{\pi}{L}m$ is a wave vector and $m$ is integer ($m=0, \pm1, \pm2, \cdots$).
- $\vec{e_x}$ is an unit vector of $x$-direction (we decide the electric field has component on $x$ direction).
We decide $A$ as follows.
$$
    A
    =
    \sqrt{\frac{2\omega^2}{\epsilon_0 V}}
$$
Here, $\omega=kc$ is frequency.
#### Magnetic field
So, we got electric field. How can we get magnetic field? Yea, we use **Maxwell equations**.
$$
    \vec{\nabla} \times \vec{B}
    =
    \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}
$$
To calculate left-hand side term, we can use the table.
$$
    \vec{\nabla} \times \vec{B}
    =
    \begin{vmatrix}
        \vec{e_x} & \vec{e_y} & \vec{e_z}\\
        \partial_x & \partial_y & \partial_z\\
        B_x & B_y & B_z\\
    \end{vmatrix}
$$
Physicist can typically notice that
we need to take care only $y$ direction of magnetic field..
You know we set electric field on $x$ direction.
Thus, 
$$
    \vec{\nabla} \times \vec{B}
    =
    -\frac{\partial B_y}{\partial z} \vec{e_x}
$$
Gotcha. Right-hand side is easy.
$$
    \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}
    =
    A \cdot \dot{q}(t) \cdot \sin(kz) \cdot \vec{e_x}
$$
Then, Maxwell equation become
$$
    -\frac{\partial B_y}{\partial z} \vec{e_x}
    =
    A \cdot \dot{q}(t) \cdot \sin(kz) \cdot \vec{e_x}
$$
Thus, 
$$
\begin{aligned}
    \vec{B}(z, t)
    &=
    \frac{mu_0 \epsilon_0}{k} \cdot 
    A \cdot \dot{q}(t) \cdot \cos(kz) \cdot \vec{e_y}
    \\&=
    \frac{mu_0 \epsilon_0}{k} \cdot 
    A \cdot p(t) \cdot \cos(kz) \cdot \vec{e_y}
\end{aligned}
$$
Here, we introduced $p(t)=\dot{q}(t)$. Again we gonna see why we use symbol $p$.
#### Finally, let's calculate energy!
We have electric field and magnetic field so we can calculate energy.
However, as you can see the difficulty is $\int \mathrm{d} V$. 
How do we process this?
We introduce new variable cross section $\sigma$.
$$
    V 
    =
    \sigma \cdot L
$$
Energy function become
$$
    H 
    = 
    \frac{1}{2} 
    \sigma
    \int_0^L \mathrm{d} z
    \left[
        \epsilon_0 \left| \vec{E}(z, t) \right|^2
        + 
        \frac{1}{\mu_0} \left| \vec{B}(z, t) \right|^2
    \right]^2
$$
By using the math trick
$$
\begin{aligned}
    \int_0^L \mathrm{d}z \sin^2(kz) 
    &=
    \int_0^L \mathrm{d}z \sin^2\left(\frac{\pi}{L}mz\right) 
    \\&=
    \frac{L}{2}
\end{aligned}
$$
we see the simple form of energy function.
$$
    H
    =
    \frac{1}{2} \omega^2 q^2(t)
    +
    \frac{1}{2} p^2(t)
$$
As you notice this energy function is same as that of unit mass classical 
harmonic oscillator.
And our way of symbolizing make sense.
#### Warning
Can we say $q$ as position of photon and $p$ as momentum of photon? No, we can't.
This is just mathematical formulation and analogy to classical harmonic oscillator.
#### Quantization
Phew. It was a long journey to get energy function. 
Let's go back to quantum world.
To quantize the energy function, we need to use operators.
We introduce three operators $\hat{H}, \hat{q}, \hat{p}$.
$$
    \hat{H}
    =
    \frac{1}{2} \omega^2 \hat{q}^2(t)
    +
    \frac{1}{2} \hat{p}^2(t)
$$
To quantize we use commutation relation to operators $\hat{q}$ and $\hat{p}$.
$$
    \left[
        \hat{q}, \hat{p}
    \right]
    =
    i\hbar
$$
Notice that $\hat{q}$ and $\hat{p}$ are Hermitian i.e. observable.
#### New operators
In addition to these operators, we introduce other two non-Hermitian operators.
Namely, **annihilation operator** $\hat{a}$ 
and **creation operator** $\hat{a}^\dag$ .
$$
    \hat{a} 
    =
    \frac{1}{2\hbar\omega}
    \left(
        \omega \hat{q} 
        + 
        i\hat{p} 
    \right)
$$
$$
    \hat{a}^\dag 
    =
    \frac{1}{2\hbar\omega}
    \left(
        \omega \hat{q} 
        - 
        i\hat{p} 
    \right)
$$
These operators have cool commutation relation.
$$
    \left[
        \hat{a}, \hat{a}^\dag
    \right]
    =
    1
$$
Wow it is $1$. It makes our life easier.
#### Quantized electric field, magnetic fields and Hamiltonian using $\hat{a}$ and $\hat{a}^\dag$
Using new operators, electric and magnetic field are quantized.
$$
    \hat{\vec{E}}(z, t)
    =
    \sqrt{\frac{\hbar \omega}{\epsilon_0 V}} \cdot 
    \left(
        \hat{a} + \hat{a}^\dag
    \right) \cdot 
    \sin(kz) \cdot 
    \vec{e_x}
$$
$$
    \hat{\vec{B}}(z, t)
    =
    -i \frac{\mu_0}{k}
    \sqrt{\frac{\epsilon_0 \hbar \omega^2}{V}} \cdot 
    \left(
        \hat{a} - \hat{a}^\dag
    \right) \cdot 
    \cos(kz) \cdot 
    \vec{e_y}
$$
Hamiltonian become simple form.
$$
    \hat{H}
    =
    \hbar \omega
    \left( 
        \hat{a}^\dag \hat{a}
        + 
        \frac{1}{2}
    \right)
$$
#### Time evolution of EM field
As we see above, the time evolution of EM filed is 
time evolution of $\hat{a}(t)$ and $\hat{a}^\dag(t)$.
To see the time evolutions of these operators, we consider them in Heisenberg
picture i.e. Heisenberg equation.
$$
\begin{aligned}
    \frac{\mathrm{d}\hat{a}(t)}{\mathrm{d}t}
    &=
    \frac{i}{\hbar} 
    \left[
        \hat{H}, \hat{a}
    \right]
    \\&=
    \frac{i}{\hbar} 
    \hbar \omega
    \left[
        \hat{a}^\dag \hat{a}, \hat{a}
    \right]
    \\&=
    -i\omega\hat{a}
\end{aligned}
$$
It depends on time exponential manner.
$$
    \hat{a}(t) = \hat{a}(0)e^{-i\omega t}
$$
By taking complex conjugate, we can obtain the $\hat{a}^\dag(t)$.
$$
    \hat{a}^\dag(t) = \hat{a}^\dag(0)e^{i\omega t}
$$
#### Photon number operator and its eigenstate
We can say 
$$
    \hat{n} 
    =
    \hat{a}^\dag \hat{a}
$$ 
as number operator.
By defining $\Ket{n}$ as energy eigenstate of the single model field 
with energy eigenvalue $E_n$ like
$$
\begin{aligned}
    \hat{H} 
    \Ket{n}
    &=
    \hbar \omega
    \left( 
        \hat{a}^\dag \hat{a}
        + 
        \frac{1}{2}
    \right)
    \Ket{n}
    \\&=
    E_n
    \Ket{n}
\end{aligned}
$$
By multiplying $\hat{a}^\dag$, we see
$$
    \hbar \omega
    \left[ 
        \hat{a}^\dag
        \left(\hat{a}^\dag \hat{a}\right)
        + 
        \frac{1}{2}
        \hat{a}^\dag
    \right]
    \Ket{n}
    =
    E_n
    \hat{a}^\dag
    \Ket{n}
$$
$$
    \hbar \omega
    \left[ 
        \hat{a}^\dag
        \left(\hat{a}\hat{a}^\dag -1 \right)
        + 
        \frac{1}{2}
        \hat{a}^\dag
    \right]
    \Ket{n}
    =
    E_n
    \hat{a}^\dag
    \Ket{n}
$$
$$
    \hbar \omega
        \left(
        \hat{a}^\dag\hat{a}
        + 
        \frac{1}{2}
        \right)
        \hat{a}^\dag
    \Ket{n}
    =
    \left(
    E_n
    +
    \hbar \omega
    \right)
    \hat{a}^\dag
    \Ket{n}
$$
We used the commutation relation 
$\left[ \hat{a}, \hat{a}^\dag \right]=\hat{a}\hat{a}^\dag-\hat{a}^\dag\hat{a}=1$

We can notice that $\hat{a}^\dag \Ket{n}$ is eigenstate of 
eigenvalue $E_n + \hbar\omega$. 
Aha. That's why it is called
creation operator.

Similarly we can show why annihilation operator is so called.
$$
    \hat{H} 
    \hat{a} 
    \Ket{n}
    =
    \left( 
        E_n 
        - 
        \hbar \omega 
    \right)
    \hat{a} 
    \Ket{n}
$$
Unlike to creation operator, annihilation operator has Important feature.
$$
    \hat{a} 
    \Ket{0}
    =
    0
$$
#### Vacuum energy 
$$
    \hat{H} 
    \Ket{0} 
    =
    \frac{1}{2} 
    \hbar \omega
    \Ket{0} 
$$
So $\frac{1}{2} \hbar \omega$ is vacuum energy.
#### Number of photon
When vacuum, there is no light. Thus we can interpret $n$ as number of photon.
$$
    E_n
    =
    \hbar \omega
    \left(
    n
    +
    \frac{1}{2} 
    \right)
$$
#### Eigenvalue of annihilation and creation operators
We know 
$$
    \hat{n}
    \Ket{n}
    =
    n
    \Ket{n}
$$
By using this, we can know the eigenvalue of $\hat{a}$ and $\hat{a}^\dag$
$$
    \hat{a}
    \Ket{n}
    =
    c_n
    \Ket{n-1}
$$
$$
    \Bra{n}
    \hat{a}^\dag
    =
    c_n^*
    \Bra{n-1}
$$
$$
\begin{aligned}
    n 
    &=
    \Braket{n|
    \hat{a}^\dag
    \hat{a}
    |n}
    \\&=
    c_n^* 
    \Bra{n-1}
    \cdot
    c_n
    \Ket{n-1}
    \\&=
    \left| c_n \right|^2
\end{aligned}
$$
Thus 
$$
    c_n 
    =
    \sqrt{n}
$$
Gotcha.
$$
    \hat{a}
    \Ket{n}
    =
    \sqrt{n}
    \Ket{n-1}
$$
Similarly, we can show
$$
    \hat{a}^\dag
    \Ket{n}
    =
    c_n
    \Ket{n+1}
$$
$$
    \Bra{n}
    \hat{a}
    =
    c_n^*
    \Bra{n+1}
$$
$$
\begin{aligned}
    n 
    &=
    \Braket{n|
    \hat{a}^\dag
    \hat{a}
    |n}
    \\&=
    \Braket{n|
    \hat{a}
    \hat{a}^\dag
    -1
    |n}
    \\&=
    c_n^* 
    \Bra{n+1}
    \cdot
    c_n
    \Ket{n+1}
    -1
    \\&=
    \left| c_n \right|^2
    -1
\end{aligned}
$$
$$
    c_n 
    =
    \sqrt{n+1}
$$
$$
    \hat{a}^\dag
    \Ket{n}
    =
    \sqrt{n+1}
    \Ket{n}
$$
#### Properties of eigenstate of number state
- Orthogonal $\Braket{n|n^\prime} = \delta_{nn^\prime}$
- Complete $\sum_{n=0}^\infty\Ket{n}\Bra{n} = 1$

# __2.2 Quantum fluctuations__
Time average of electric field is 
$$
\begin{aligned}
    \left<
        \hat{\vec{E}}
    \right>
    &=
    \Braket{n|\hat{\vec{E}}|n}
    \\&=
    \sqrt{
        \frac{\hbar \omega}{\epsilon_0 V}
    }
    \sin (kz) 
    \Braket{n|\left(\hat{a}+\hat{a}^\dag\right)|n}
    \\&=
    0
\end{aligned}
$$
I see. How about variance?
$$
\begin{aligned}
    \left<
        \hat{\vec{E}}^2
    \right>
    &=
    \Braket{n|\hat{\vec{E}}^2|n}
    \\&=
    \frac{\hbar \omega}{\epsilon_0 V}
    \sin^2 (kz) 
    \Braket{
        n|
        \left(
        \hat{a}^2 + {\hat{a}^\dag}^2 + \hat{a}\hat{a}^\dag + \hat{a}^\dag\hat{a}
        \right)
        |n
    }
    \\&=
    \frac{\hbar \omega}{\epsilon_0 V}
    \sin^2 (kz) 
    \left(
        n+\frac{1}{2}
    \right)
\end{aligned}
$$
Wow! Even in the vacuum state ($n=0$), electric field fluctuates!!

# __2.3 Quadrature Operators__
Similar to $\hat{q}$ and $\hat{p}$ operators, we introduce quadrature operators
$\hat{X_1}$ and $\hat{X_2}$..
$$
    \hat{X_1}
    =
    \frac{\hat{a} + \hat{a}^\dag}{2} 
    \propto
    \hat{q}(t)
$$
$$
    \hat{X_2}
    =
    -i
    \frac{\hat{a} - \hat{a}^\dag}{2} 
    \propto
    \hat{p}(t)
$$
$$
    \left[
        \hat{X_1}, \hat{X_2}
    \right]
    =
    \frac{1}{2}
$$
Notice that quadrature operators are observable and homodyne technique is used for
that.
And these operators does not commute. This is because Heisenberg's uncertainty.

We can rewrite electric field with quadrature operators.
$$
\begin{aligned}
    \hat{E_x}
    &=
    \mathcal{E}_0
    \left[
        \hat{a}e^{-i\omega t}
        +
        \hat{a}^\dag e^{i\omega t}
    \right]
    \sin (kz)
    \\&=
    \mathcal{E}_0
    \sin (kz)
    \left[
        \left(
            \hat{a}
            +
            \hat{a}^\dag
        \right)
        \cos \omega t
        -
        i
        \left(
            \hat{a}
            -
            \hat{a}^\dag
        \right)
        \sin \omega t
    \right]
    \\&=
    2
    \mathcal{E}_0
    \sin (kz)
    \left[
        \hat{X_1}
        \cos \omega t
        +
        \hat{X_2}
        \sin \omega t
    \right]
\end{aligned}
$$
Important staff of quadrature operators is their uncertainty.
$$
    \left<
        \left(
            \Delta \hat{X}_1
        \right)^2
    \right>
    \left<
        \left(
            \Delta \hat{X}_2
        \right)^2
    \right>
    \ge
    \frac{1}{4}
    \left|
        \left<
            \left[
                \hat{X}_1, \hat{X}_2
            \right]
        \right>
    \right|^2
    \ge
    \frac{1}{16}
$$
