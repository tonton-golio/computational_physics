# Eigenvalue Algorithms
*The computational engines behind eigenvalue discovery*

> *This lesson has been combined with Eigenvalue Problems into a single, unified lesson. Please see the **Eigenvalue Problems** page for the complete treatment, organized as "simple methods first, fancy methods later."*

---

## Quick Reference

The full eigenvalue lesson covers these algorithms in order of increasing sophistication:

### Part I: Simple Methods (one eigenvalue at a time)
- **Power Method** — multiply and normalize until the dominant eigenvalue wins
- **Inverse Iteration** — flip the spectrum to find eigenvalues near a target
- **Rayleigh Quotient Iteration** — cubic convergence through a brilliant feedback loop

### Part II: Fancy Methods (all eigenvalues at once)
- **QR Algorithm** — the industry standard behind `numpy.linalg.eig`
- **Gershgorin Circle Theorem** — quick eigenvalue bounds without computation
- **Functions of Matrices** — applying functions to eigenvalues to define $e^A$, $\sqrt{A}$, etc.

All content, code, interactive simulations, and challenges are in the unified **Eigenvalue Problems** lesson.
