# Scientific Computing

## Course overview

Scientific computing develops the **numerical methods** needed to solve problems in biology, physics, nanoscience, and chemistry that have no closed-form solution. The emphasis is on deriving algorithms, programming them, and understanding their error behavior.

- Accuracy: how close is the numerical answer to the true solution?
- Efficiency: how does the computational cost scale with problem size?
- Robustness: does the method work reliably across a range of inputs?
- Stability: do small perturbations in input produce small perturbations in output?

## Why this topic matters

- Most differential equations arising in science cannot be solved analytically.
- Linear systems with thousands of unknowns appear in finite-element modeling, data fitting, and network analysis.
- Optimization underlies machine learning, inverse problems, and experimental design.
- Understanding numerical error is essential for trusting computational results.

## Why this order? A mountain trail map

Think of this course as a hike up a mountain. Each stop builds on the last, and the view keeps getting better.

```
                                        ___
                                       /   \    11. PDEs
                                      / PDE  \  "Painting the whole sky"
                                     /________\
                                    /          \    10. ODEs
                                   / Time March \   "Marching through time"
                                  /______________\
                                 /                \    09. FFT
                                /   Wave Music     \   "Hearing the hidden notes"
                               /____________________\
                              /                      \    07-08. Eigenvalues
                             / Secret Personalities   \   "The DNA of matrices"
                            /__________________________\
                           /                            \    06. Optimization
                          /      Smart Hiking            \   "Finding the best trail"
                         /________________________________\
                        /                                  \    04-05. Nonlinear
                       /     Nonlinear Wilderness           \   "Where the wild things are"
                      /______________________________________\
                     /                                        \    03. Least Squares
                    /          Messy Data                       \   "Making peace with noise"
                   /____________________________________________\
                  /                                              \    02. Linear Equations
                 /            Exact Linear                        \   "The tools that always work"
                /__________________________________________________|
               /                                                    \    01. Bounding Errors
    START --> /          Floating-Point Lies                          \   "Your computer is a liar"
             /____________________________________________________________\
```

You start at base camp learning that your computer lies to you (floating-point errors). Then you climb through the world of linear equations where we actually have perfect tools. Higher up, the data gets messy (least squares) and the equations get wild (nonlinear). Optimization teaches you to hike smart. Eigenvalues reveal the secret personalities hidden inside matrices. The FFT lets you hear the music in data. ODE solvers teach you to march through time. And at the summit, you can paint the whole sky with PDE simulations of heat, waves, and fluid flow.

## Learning trajectory

| # | Lesson | Subtitle | Time |
|---|--------|----------|------|
| 01 | **Bounding Errors** | *Why your computer is a liar and how to catch it in the act* | 45 min + 20 min coding |
| 02 | **Linear Equations** | *The one family of problems we can solve perfectly (almost)* | 60 min + 30 min coding |
| 03 | **Linear Least Squares** | *When the data doesn't fit, find the closest thing that does* | 50 min + 25 min coding |
| 04 | **Nonlinear Equations** | *Welcome to the jungle where anything is possible* | 55 min + 25 min coding |
| 05 | **Nonlinear Systems** | *Newton goes to higher dimensions and brings friends* | 50 min + 25 min coding |
| 06 | **Optimization** | *Finding the best answer when "best" is all you've got* | 50 min + 30 min coding |
| 07-08 | **Eigenvalue Problems & Algorithms** | *Discovering the secret personalities hiding inside matrices* | 70 min + 30 min coding |
| 09 | **Fast Fourier Transform** | *The machine that tells you which notes are in a song* | 40 min + 20 min coding |
| 10 | **Initial Value Problems** | *Teaching the computer to march through time without tripping* | 55 min + 25 min coding |
| 11 | **Partial Differential Equations** | *Painting the whole sky: heat, waves, and everything in between* | 60 min + 30 min coding |

## Key mathematical ideas

- Matrix factorizations (LU, QR, SVD) and their role in solving linear systems.
- Iterative methods for nonlinear equations (Newton-Raphson, fixed-point iteration).
- Numerical integration of ODEs (Euler, Runge-Kutta) and stability theory.
- Finite-difference discretization of PDEs.
- The discrete Fourier transform and the FFT algorithm.
- Condition numbers and the propagation of rounding errors.

## Prerequisites

- Programming in Python with NumPy.
- Linear algebra: matrix operations, eigenvalues, vector spaces.
- Calculus: derivatives, integrals, Taylor series.

## Recommended reading

- Heath, *Scientific Computing: An Introductory Survey*.
- Trefethen and Bau, *Numerical Linear Algebra*.
- Press et al., *Numerical Recipes*.

### Further reading (only if you're curious)

- 3Blue1Brown: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — watch this before the course if linear algebra feels rusty.
- 3Blue1Brown: [Differential Equations](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6) — beautiful visual intuition for ODEs and PDEs.
- 3Blue1Brown: [But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) — the single best visual explanation of FFT.
- Steve Brunton's YouTube channel — excellent applied scientific computing lectures.
- Gilbert Strang, *Introduction to Linear Algebra* — the gentlest serious linear algebra textbook.

---

## Cheat Sheet of Big Ideas

*Pin this above your desk.*

| Idea | One sentence | Where |
|------|-------------|-------|
| **Condition number** | Measures how much input errors get amplified in the output. | Lesson 01 |
| **LU factorization** | Split a matrix into lower and upper triangles so you can solve by substitution. | Lesson 02 |
| **QR factorization** | Use mirror reflections (Householder) to orthogonalize, saving your significant digits. | Lesson 03 |
| **Newton's method** | Linearize locally, solve, repeat — doubles your correct digits every step. | Lesson 04 |
| **Broyden's method** | Fake the Jacobian with secant info so you never have to compute it. | Lesson 05 |
| **BFGS** | Build up curvature knowledge as you go — quasi-Newton for optimization. | Lesson 06 |
| **Power method** | Keep multiplying by the matrix; the biggest eigenvalue wins. | Lesson 07 |
| **QR algorithm** | Factorize and reverse-multiply until eigenvalues appear on the diagonal. | Lesson 08 |
| **FFT** | Split even/odd recursively to turn $O(N^2)$ into $O(N \log N)$. | Lesson 09 |
| **RK4** | Four slope samples per step give you fourth-order accuracy — the workhorse of ODE solving. | Lesson 10 |
| **Lax equivalence** | Consistency + stability = convergence. If the recipe is right and nothing explodes, dinner turns out perfect. | Lesson 11 |
