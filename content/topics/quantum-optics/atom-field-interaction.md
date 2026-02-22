# Atom-Field Interaction

## An atom meets a photon

Everything so far has been about light by itself -- field modes, photon statistics, phase-space pictures. Now we bring in an atom and let it talk to the quantized field. This is where quantum optics gets really interesting.

## The dipole approximation

Picture an atom sitting in an electromagnetic field. The full Hamiltonian includes the atom's kinetic and potential energy plus the field interaction:

$$
\hat{H} = \frac{1}{2m}\left(\hat{p} + e\vec{A}(\vec{r}, t)\right)^2 + V(\vec{r})
$$

But here's a crucial simplification. Optical wavelengths are around $\lambda \sim 500$ nm, while the atom is a few Angstroms across. The spatial variation of the field across the atom is negligible -- $|\vec{k}\cdot\vec{r}| \ll 1$ -- so we can treat the field as spatially uniform at the atom's location.

This is the **dipole approximation**, and it simplifies the Hamiltonian to:

$$
\hat{H} = \hat{H}_\text{atom} - \hat{\vec{d}}\cdot\hat{\vec{E}}(t)
$$

The interaction is just the dot product of the atomic dipole moment and the electric field. Clean and elegant.

## Two-level atom in a cavity

Now specialize to a two-level atom -- ground state $\Ket{b}$ and excited state $\Ket{a}$ -- sitting in a single-mode cavity. The cavity field is:

$$
\hat{\vec{E}} = \vec{\mathcal{E}}(\hat{a} + \hat{a}^\dag)
$$

The atom-field state is a superposition:

$$
\Ket{\psi(t)} = C_i(t)\Ket{a}\Ket{n}\,e^{-iE_at/\hbar}\,e^{-in\omega t} + C_f(t)\Ket{b}\Ket{n-1}\,e^{-iE_bt/\hbar}\,e^{-i(n-1)\omega t}
$$

The first term: atom excited, $n$ photons. The second: atom in the ground state, $n-1$ photons (one photon emitted). Energy sloshes back and forth between the atom and the field.

## The rotating wave approximation

The full interaction Hamiltonian contains four terms. Two of them -- $\hat{a}^\dag\hat{\sigma}_+$ (atom absorbs while the field *also* creates a photon) and $\hat{a}\hat{\sigma}_-$ (atom emits while the field *also* destroys a photon) -- are the **counter-rotating terms**. They describe the atom trying to create a photon while already emitting one, or absorbing while already being de-excited. Ridiculous at ordinary light intensities.

These terms oscillate at $2\omega$ and average to nearly zero when $g \ll \omega$. Dropping them is the **rotating wave approximation** (RWA), and it makes the problem exactly solvable. The counter-rotating terms only matter in the ultrastrong coupling regime -- interesting physics, but a story for another day.

## Big Ideas

- The dipole approximation works because optical wavelengths dwarf atomic sizes ($|\vec{k}\cdot\vec{r}| \ll 1$), reducing the interaction to $-\hat{\vec{d}}\cdot\hat{\vec{E}}$.
- The two-level approximation holds when the driving field is near-resonant with one transition, so all other levels can be ignored.
- Coupling a two-level atom to a quantized field produces entangled atom-field states (dressed states) that are neither pure atom nor pure field.

## Check Your Understanding

1. Under what conditions would the dipole approximation break down? What kind of radiation would you need?
2. The perturbation theory ansatz has terms $\Ket{a, n}$ and $\Ket{b, n-1}$. What physical process does each represent? Why is total excitation number approximately conserved?

## Challenge

Write a Python script that constructs the full Rabi Hamiltonian for a two-level atom coupled to a single cavity mode (including counter-rotating terms) and diagonalizes it numerically. Compare the energy spectrum to the Jaynes-Cummings model (without counter-rotating terms) as a function of $g/\omega$. At what coupling strength do the two models diverge significantly? Plot the Bloch-Siegert shift.
