# Quantization of a Single-Mode Field

## The field is a spring

Here's the punchline, delivered early: an electromagnetic field trapped in a box behaves exactly like a harmonic oscillator. A spring. That's it. And once you know it's a spring, you know how to quantize it -- because you already know how to quantize a harmonic oscillator.

But let's earn that punchline. Why bother quantizing the field at all?

## Why quantize?

Picture an electron sitting in an excited state of an atom. The Schrodinger equation says it should stay there forever -- $\hat{H}\Ket{\psi_m} = E_m\Ket{\psi_m}$ is perfectly stable. Yet atoms decay. They emit light spontaneously, with nobody pushing them. Something must be shaking that electron loose.

The answer is **fluctuations** -- quantum fluctuations of the electromagnetic field itself. Classical electromagnetism can't explain them: in classical E&M, no light means no electric field means no fluctuation. But nature disagrees. Even in pitch darkness, the field is jittering. To see why, we need to quantize it.

## Setting up the box

Imagine a box with perfectly conducting walls at $z = 0$ and $z = L$. We'll consider a single mode -- one particular way the field can wiggle inside the box. The boundary conditions force the electric field to vanish at both walls.

The electric field takes the form:

$$
\vec{E}(z, t) = A \cdot q(t) \cdot \sin(kz) \cdot \vec{e_x}
$$

where $k = m\pi/L$ for integer $m$, and we've chosen a normalization constant:

$$
A = \sqrt{\frac{2\omega^2}{\epsilon_0 V}}
$$

Why the symbol $q(t)$? You'll see in a moment.

The magnetic field follows from Maxwell's equations ($\vec{\nabla}\times\vec{B} = \mu_0\epsilon_0\,\partial\vec{E}/\partial t$):

$$
\vec{B}(z, t) = \frac{\mu_0\epsilon_0}{k} A \cdot p(t) \cdot \cos(kz) \cdot \vec{e_y}
$$

where we've defined $p(t) = \dot{q}(t)$. The reason for these symbols is about to become obvious.

## The energy is a harmonic oscillator

[[simulation wigner-number-state]]

Now compute the total electromagnetic energy. With the volume $V = \sigma \cdot L$ and the identity $\int_0^L \sin^2(m\pi z/L)\,dz = L/2$, the energy simplifies beautifully:

$$
H = \frac{1}{2}\omega^2 q^2(t) + \frac{1}{2}p^2(t)
$$

That's the Hamiltonian of a unit-mass harmonic oscillator. The symbol $q$ is playing the role of "position" and $p$ is "momentum." But be careful -- **these are not the position and momentum of a photon**. This is pure mathematical analogy. There is no "position of a photon" in any meaningful sense.

## Climbing the ladder

Now we quantize. Promote $q$ and $p$ to operators with the canonical commutation relation $[\hat{q}, \hat{p}] = i\hbar$, and define the **annihilation** and **creation** operators:

$$
\hat{a} = \frac{1}{\sqrt{2\hbar\omega}}(\omega\hat{q} + i\hat{p}), \qquad \hat{a}^\dag = \frac{1}{\sqrt{2\hbar\omega}}(\omega\hat{q} - i\hat{p})
$$

They satisfy the beautifully simple commutation relation:

$$
[\hat{a}, \hat{a}^\dag] = 1
$$

This single equation -- $[\hat{a}, \hat{a}^\dag] = 1$ -- is the seed of everything non-classical that follows. The Hamiltonian becomes:

$$
\hat{H} = \hbar\omega\left(\hat{a}^\dag\hat{a} + \frac{1}{2}\right)
$$

The **number operator** $\hat{n} = \hat{a}^\dag\hat{a}$ counts photons. Its eigenstates $\Ket{n}$ are the energy levels of the field:

$$
E_n = \hbar\omega\left(n + \frac{1}{2}\right)
$$

The creation operator $\hat{a}^\dag$ adds one photon: $\hat{a}^\dag\Ket{n} = \sqrt{n+1}\Ket{n+1}$. The annihilation operator $\hat{a}$ removes one: $\hat{a}\Ket{n} = \sqrt{n}\Ket{n-1}$. And at the bottom of the ladder:

$$
\hat{a}\Ket{0} = 0
$$

You can't annihilate what isn't there. But even $\Ket{0}$ carries energy -- the vacuum energy $\frac{1}{2}\hbar\omega$. The vacuum is not empty.

## The quantized fields

In terms of ladder operators, the electric and magnetic fields become:

$$
\hat{\vec{E}}(z, t) = \sqrt{\frac{\hbar\omega}{\epsilon_0 V}}\,(\hat{a} + \hat{a}^\dag)\,\sin(kz)\,\vec{e_x}
$$

$$
\hat{\vec{B}}(z, t) = -i\frac{\mu_0}{k}\sqrt{\frac{\epsilon_0\hbar\omega^2}{V}}\,(\hat{a} - \hat{a}^\dag)\,\cos(kz)\,\vec{e_y}
$$

And the time evolution is simple exponential rotation:

$$
\hat{a}(t) = \hat{a}(0)\,e^{-i\omega t}, \qquad \hat{a}^\dag(t) = \hat{a}^\dag(0)\,e^{i\omega t}
$$

## Big Ideas

* The electromagnetic field in a cavity is mathematically identical to a harmonic oscillator. Quantizing the field *is* quantizing that oscillator.
* Photons aren't little particles flying around -- they're excitation quanta of a field mode, counted by $\hat{n} = \hat{a}^\dag\hat{a}$.
* Even with zero photons, the field carries vacuum energy $\frac{1}{2}\hbar\omega$. The vacuum is not empty.
* The commutation relation $[\hat{a}, \hat{a}^\dag] = 1$ generates the entire ladder of energy levels and everything non-classical that follows.

## Check Your Understanding

1. What plays the role of "position" and "momentum" in the harmonic-oscillator analogy, and why can't you interpret them as a photon's actual position and momentum?
2. The ladder stops at the bottom ($\hat{a}\Ket{0} = 0$) but never at the top. Why?
3. If you set $[\hat{a}, \hat{a}^\dag] = 0$ (classical limit), what happens to the vacuum energy? What would that imply physically?

## Challenge

Write a Python script that constructs the matrix representations of $\hat{a}$, $\hat{a}^\dag$, and $\hat{n}$ in a truncated Fock space of dimension $N_{\max}$, and verify numerically that $[\hat{a}, \hat{a}^\dag] = \hat{I}$ (up to truncation errors), that $\hat{H} = \hbar\omega(\hat{n} + \frac{1}{2})$, and that the eigenvalues are $E_n = \hbar\omega(n + \frac{1}{2})$. Explore how the commutation relation breaks down as $n$ approaches $N_{\max}$.
