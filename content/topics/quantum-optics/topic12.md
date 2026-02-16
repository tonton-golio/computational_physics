__Topic 12 keywords__
- Perturbation theory 
- Rabi oscillations
- Jaynes-Cummings model

# __Readings__
Ch. 4.3-5 

# __4.3 Interaction of an atom with a quantized field__
#### Dipole approximation
Imagine a situation with electromagnetic field act on one atom.

To begin, Hamiltonian of an electron bound to an atom in vacuum is
$$
    \hat{H}_\mathrm{atom}
    =
    \frac{\hat{p}^2}{2m}
    +
    V(\vec{r})
$$
Here, $V(\vec{r})$ is Coulomb potential.

In presence of external fields, the Hamiltonian become,
$$
    \hat{H}
    =
    \frac{1}{2m}
    \left(
    \hat{p}^2
    +
    e \vec{A}(\vec{r}, t)
    \right)^2
    +
    V(\vec{r})
$$

We can further simplify the equation.
The spatial dependency of electromagnetic field is $e^{i \vec{k} \cdot \vec{r}}$.
By assuming typical light wavelength as $\lambda \sim 500 \mathrm{nm}$, 
and $\vec{r}$ is few Angstroms, 
the magnitude of $\left| \vec{k} \cdot \vec{r} \right|$ is smaller than one.
$$
    \left| \vec{k} \cdot \vec{r} \right|
    \ll
    1
$$
Thus, we can ignore the spatial dependency of electromagnetic field.
$$
    \hat{H}
    =
    \hat{H}_\mathrm{atom}
    -
    \hat{\vec{d}} \cdot \hat{\vec{E}} (t)
$$
This is **dipole approximation**.

#### 2-level emitter in cavity (perturbation theory)
Imagine a 2-level system in cavity. 
We apply single optical mode.
Electric field on atom is 
$$
    \hat{\vec{E}}
    =
    \vec{\mathcal{E}}
    \left( 
        \hat{a} + \hat{a}^\dag
    \right)
$$
We are going to solve Schrödinger equation.
$$
    i\hbar
    \frac{\partial}{\partial t}
    \Ket{\psi(t)}
    =
    \hat{H}
    \Ket{\psi(t)}
$$
We use *ansatz* solution. 
$$
\begin{aligned}
    \Ket{\psi(t)}
    &=
    C_i(t) 
    \cdot
    \Ket{a} 
    \Ket{n}
    \cdot
    e^{-i E_a t/\hbar}
    e^{-i n \omega t}
    +
    C_f(t) 
    \cdot
    \Ket{b} 
    \Ket{n-1}
    \cdot
    e^{-i E_b t/\hbar}
    e^{-i (n-1) \omega t}
    \\&=
    C_i(t) 
    \cdot
    \Ket{i} 
    \cdot
    e^{-i E_a t/\hbar}
    e^{-i n \omega t}
    +
    C_f(t) 
    \cdot
    \Ket{f} 
    \cdot
    e^{-i E_b t/\hbar}
    e^{-i (n-1) \omega t}
\end{aligned}
$$
Here, 
$C_i(t)$ is complex coefficient, 
$\Ket{i}$ is initial state with $a$ state and $n$ photon, 
$e^{-i E_a t/\hbar}$ is time evolution of atom, 
and $e^{-i n \omega t}$ is time evolution of photon.


# __4.4 The Rabi model__

# __4.5 Fully quantum-mechanical model; the Jaynes-Cummings model__
