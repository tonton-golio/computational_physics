# Quantization of Single Mode Field

## Why do we need to quantize?
* So far, we learned that electron in high-energy state decays back to ground state with the emission of light.
* However, according to the Schrödinger equation $$\hat{H} \Ket{\psi_m} = E_m \Ket{\psi_m}$$, electron is stable in excited state i.e. *electron stays in high energy state*.
* How do the decay? Something goes wrong. Why does spontaneous emission happen?
* The answer is **fluctuations**. 
* The new question is *what kind of fluctuations do they have*?

## Can classical electromagnetism describe these fluctuations?
In classical electromagnetic, energy of electron in electric field $E(t)$ is 
$$ e \cdot E(t) $$
Thus, when there is no light, there is no fluctuation.
However, it is known that even without light, there is a fluctuation i.e. quantum
mechanical fluctuation.
We need to use quantum mechanics.

## Quantization of Single Mode Field
### Motivation
Let's start our quantization of electromagnetic waves.
The quantity we quantize is **energy**.
### Situation
Imagine a three-dimensional box and there are perfectly conducting walls at 
$z=0$ and $z=L$. 
We consider **single mode field** i.e. we only take care one mode.
Boundary conditions are
* $\vec{E}(z=0)=0$
* $\vec{E}(z=L)=0$
### Energy
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
### Electric field
From boundary condition, we can determine the spatial part of electric field.
$$
    \vec{E}(z, t)
    =
    A \cdot q(t) \cdot \sin(kz) \cdot \vec{e_x}
$$
Here, parameters are described as follows.
* $A$ is a constant and we can decide. 
* $q(t)$ is the time dependent part and we will see why we use $q$ as symbol.
* $k=\frac{2\pi}{\lambda}=\frac{\pi}{L}m$ is a wave vector and $m$ is integer ($m=0, \pm1, \pm2, \cdots$).
* $\vec{e_x}$ is an unit vector of $x$-direction (we decide the electric field has component on $x$ direction).
We decide $A$ as follows.
$$
    A
    =
    \sqrt{\frac{2\omega^2}{\epsilon_0 V}}
$$
Here, $\omega=kc$ is frequency.
### Magnetic field
So, we got electric field. How can we get magnetic field? We use **Maxwell equations**.
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
Recall that we set the electric field along the $x$ direction.
Thus, 
$$
    \vec{\nabla} \times \vec{B}
    =
    -\frac{\partial B_y}{\partial z} \vec{e_x}
$$
The right-hand side is straightforward.
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
Here, we introduced $p(t)=\dot{q}(t)$. The reason for this choice of symbol will become clear shortly.
### Finally, let's calculate energy!
We have electric field and magnetic field so we can calculate energy.

[[simulation wigner-number-state]]

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
### Warning
Can we say $q$ as position of photon and $p$ as momentum of photon? No, we can't.
This is just mathematical formulation and analogy to classical harmonic oscillator.
### Quantization
Now we return to the quantum description.
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
### New operators
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
        * 
        i\hat{p} 
    \right)
$$
These operators satisfy the commutation relation
$$
    \left[
        \hat{a}, \hat{a}^\dag
    \right]
    =
    1
$$
The result is simply $1$, which greatly simplifies calculations.
### Quantized electric field, magnetic fields and Hamiltonian using $\hat{a}$ and $\hat{a}^\dag$
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
### Time evolution of EM field
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
### Photon number operator and its eigenstate
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
This is why it is called the
creation operator.

Similarly we can show why annihilation operator is so called.
$$
    \hat{H} 
    \hat{a} 
    \Ket{n}
    =
    \left( 
        E_n 
        * 
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
### Vacuum energy 
$$
    \hat{H} 
    \Ket{0} 
    =
    \frac{1}{2} 
    \hbar \omega
    \Ket{0} 
$$
So $\frac{1}{2} \hbar \omega$ is vacuum energy.
### Number of photon
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
### Eigenvalue of annihilation and creation operators
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
Therefore
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
### Properties of eigenstate of number state
* Orthogonal $\Braket{n|n^\prime} = \delta_{nn^\prime}$
* Complete $\sum_{n=0}^\infty\Ket{n}\Bra{n} = 1$

## Big Ideas

* The electromagnetic field in a cavity has the same mathematical structure as a harmonic oscillator — quantizing the field is just quantizing that oscillator.
* Photons are not little particles flying through space; they are excitation quanta of a field mode, counted by the number operator $\hat{n} = \hat{a}^\dagger \hat{a}$.
* Even with zero photons, the field carries an irreducible energy $\frac{1}{2}\hbar\omega$ — the vacuum is not empty.
* The creation and annihilation operators encode the entire ladder of energy levels, and their commutation relation $[\hat{a}, \hat{a}^\dagger] = 1$ is the seed of everything non-classical that follows.

## What Comes Next

The number states $|n\rangle$ we just built have a strange property: the electric field averages to zero, yet the field still fluctuates. The next lesson makes this precise by computing those fluctuations explicitly and introducing the quadrature operators that let us track both "quadrature components" of the field — the quantum analogs of position and momentum in phase space.

## Check Your Understanding

1. The Hamiltonian of the single-mode field is formally identical to that of a unit-mass harmonic oscillator. What physical quantity plays the role of "position" and what plays the role of "momentum" in this analogy, and why can those variables not be interpreted as the photon's actual position and momentum?
2. The annihilation operator satisfies $\hat{a}|0\rangle = 0$ but $\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$. Why does the ladder stop at the bottom but never at the top?
3. The vacuum energy $\frac{1}{2}\hbar\omega$ per mode is a consequence of the commutation relation $[\hat{a}, \hat{a}^\dagger] = 1$. If you tried to set this commutator to zero (classical limit), what would happen to the vacuum energy and what would that imply physically?

## Challenge

The derivation assumed a one-dimensional cavity of length $L$ with a single mode. Generalize the argument to a three-dimensional cubic cavity and sum the zero-point energies over all allowed modes up to some cutoff wavevector $k_{\max}$. Show that the total vacuum energy density diverges as $k_{\max}^4$. This ultraviolet divergence is real and shows up in the Casimir effect — but think about how you would physically regularize it in a measurable quantity.
