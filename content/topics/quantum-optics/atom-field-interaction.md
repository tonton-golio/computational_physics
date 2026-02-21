# Atom-Field Interaction
__Topic 12 keywords__
* Perturbation theory 
* Rabi oscillations
* Jaynes-Cummings model

# Interaction of an atom with a quantized field
### Dipole approximation
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

### 2-level emitter in cavity (perturbation theory)
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


# The Rabi model

# Fully quantum-mechanical model; the Jaynes-Cummings model

## Big Ideas

* The dipole approximation — treating the electromagnetic field as spatially uniform over the atom — is justified because the optical wavelength ($\sim 500$ nm) dwarfs the atomic size ($\sim 0.1$ nm), making $|\vec{k}\cdot\vec{r}| \ll 1$.
* The atom-field interaction reduces to $-\hat{\vec{d}}\cdot\hat{\vec{E}}$: the dot product of the atomic dipole moment and the quantized electric field at the atomic position.
* The two-level approximation works when the driving field is close to resonance with one particular atomic transition, so all other levels can be ignored.
* Coupling a two-level atom to a quantized field produces a system where energy is exchanged not just in units of $\hbar\omega$ (photons) but through entangled atom-field states — dressed states — that are neither pure atom nor pure field.

## What Comes Next

The stage is set for the Jaynes-Cummings model, which solves this atom-single-mode-field interaction exactly. The next lesson shows that the quantum nature of the field leaves a distinctive fingerprint in the dynamics: the atom oscillates between excited and ground state at a rate that depends on the square root of the photon number, and a coherent field drives collapses and revivals that no classical model can explain.

## Check Your Understanding

1. The dipole approximation sets $e^{i\vec{k}\cdot\vec{r}} \approx 1$ for the field at the electron's position. Under what physical circumstances would this approximation break down, and what kind of radiation-matter coupling would you need to describe those situations?
2. The Hamiltonian in the dipole approximation is $\hat{H} = \hat{H}_\text{atom} - \hat{\vec{d}}\cdot\hat{\vec{E}}$. The minus sign might seem counterintuitive. Explain physically why an electric dipole aligned with the field has lower energy, using an analogy from classical electrostatics.
3. In the two-level approximation, we keep only the ground state $|g\rangle$ and one excited state $|e\rangle$. The perturbation theory ansatz for the state is a superposition of $|a, n\rangle$ and $|b, n-1\rangle$. What physical process does each term represent, and why is the total excitation number $n + \text{(atomic excitation)}$ approximately conserved?

## Challenge

Write down the full Rabi Hamiltonian for a two-level atom coupled to a single cavity mode, including both counter-rotating terms $\hat{a}^\dagger\hat{\sigma}_+$ and $\hat{a}\hat{\sigma}_-$. Show that the total excitation number is not conserved in the full Rabi model (unlike in the Jaynes-Cummings model). Under what condition on the coupling strength $g$ and frequency $\omega$ does the rotating wave approximation become valid, and what physical effect do the counter-rotating terms produce when the RWA breaks down? (This is the Bloch-Siegert shift.)
