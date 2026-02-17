# Metropolis Algorithm

## Ising Model
Ernst Ising introduced the model of ferromagnetism with a discrete
magnetic momemt $s_i$. He approximated spin (magnetic moment) as constrained to two states
$
s_i = \pm 1,
$
for which $s_i$ positive is denoted as *spin up*. The Ising model is has energy given by
$$
    \mathcal{H}
    =
    -J \sum_{\left<i j
\right>} s_i s_j - h \sum_i s_i
$$
Here, $\left<i j
\right>$ means sum is performed over
nearest-neighbor interaction which means
$\sum_{\left<i j
\right>}
= \frac{1}{2} \sum_i \sum_{\left< j
\right>}$.
Index of second summation $\left< j
\right>$ is nearest neighbor
of spin $i$.
$J>0$ is coupling strength and $h$ is the external magnetic field.
Notice that $J<0$ indicates antiferromagnetism.
In case of $J=0$, there is no interspin interaction.
When $J$ depends on pair of spin, it is spin glass.

Magnetization of Ising model becomes
$$
    M = \sum_{i=1}^N s_i
    \Rightarrow
    m = \frac{1}{N} \sum_{i=1}^N s_i.
$$

E. Ising calculated the partition function of the 1D Ising model. The partition function of the 2D model was aesthetically derived by Lars Onsager. The 3D Ising model remains an open problem.

In the light of the analytical difficulty encountered in high dimensional hilbert spaces, we may use computers simulate systems small systems and assess their statistics.

We can spot the critical temperature of the 2d ising phase-transition, by a divergence in the suceptibility, $\chi$. We approximate susceptibility as the variance in magnetization,
$$
\chi = \left< M^2\right> - \left<M\right>^2
$$

## Monte Carlo method
*Monte Carlo is a subpart of Monaco and it is famous for gambling.According to Wikipedia, Nicholas Metropolis suggested the name.*

This method is used for numerically estimate physical observable.
Statistical mechanical average value of physical observable
$\left<A
\right>$ is
$$
\begin{align*}
    \left<A
\right>
    &=
    \frac{\sum_i A_i \exp \left(-\beta E_i
\right) }{Z}
    \&=
    \frac
    {\sum_i A \exp \left(-\beta E_i
\right)}
    {\sum_i \exp \left(-\beta E_i
\right)}
    \&=
    \sum_i A_i P_i
\end{align*}
$$
This can be approximated by
$$
    \left<A
\right>
    \approx
    \sum_i A_i 	ilde{P}_i
$$
Here $	ilde{P}_i$ is sampled probability distribution.
By approximating statistical mechanical probability distribution
with sampled probabiolity distribution, we can get approximated
physical observable from Monte Carlo method.
## Markov process and master equation
Important assumption of Metropolis algorithm is Markov process
which argue that state change is only depends on previous state.
By using this Markov assumption, one can describe the time
evolution of probability distribution of state $i$ as
master equation.
$$
    \frac{\mathrm{d}P_i}{\mathrm{d}t}
    =
    \sum_j \left( w_{ij}P_j - w_{ji}P_i
\right)
$$
Here $P_i$ is probability of state $i$ and $w_{ij}$ is transition
rate from state $j$ to state $i$.
$w_{ij}P_j$ is flux from state $j$ to state $i$.
First term of master equation shows the incoming probablity flux
from state $j$ to state $i$ and second term shows outgoing
probability from state $i$ to state $j$.

Notice that probability normalization
$$
    \sum_i P_i = 1
$$
constrains probability flux to be conserved as a form of master
equation.

In steady state, probability distribution does not depend on time
i.e. $\frac{\mathrm{d}P_i}{\mathrm{d}t} = 0$.
Then
$$
    \sum_j \left( w_{ij}P_j - w_{ji}P_i
\right)
    = 0
$$
which is equivalent to condition of steady state.
$$
    \sum_j w_{ij}P_j = \sum_j w_{ji}P_i
$$
This argues that, in steady state, total flux coming to state $i$
equals to total flux going out from state $i$.
However, this condition can allow system to have irreversible
state transition (circular state transition) which violate
the concept of thermal equilibrium.
## Detailed balance
By using "detailed balance" as a condition of equilibrium,
one can avoid such irreversible state transition.
$$
    w_{ij}P_j = w_{ji}P_i
$$
This argues that flux from state $j$ to state $i$ is equal to
that of opposite.
This condition does not allow irreversible state transition.
## Metropolis-Hastings algorithm
What Metropolis algorithm do are two things: trying random state
transition then accepting that transition with specific criteria.
How can we set the criteria for sampling state from canonical
ensemble?
We need to restrict transition rate to sample state from canonical
ensemble.
By using detailed balance condition, we can know the form of
transition rate and it provide us a criteria.
Equilibrium probability distribution of canonical ensemble is
$$
    P_i
    =
    \frac
    {\exp \left(-\beta E_i
\right)}
    {Z} .
$$
By combining this with detailed balance condition
$$
    \frac{w_{ij}}{w_{ji}}
    =
    \frac{P_i}{P_j}
    =
    \frac
    { \frac{\exp \left(-\beta E_i
\right)}{Z} }
    { \frac{\exp \left(-\beta E_j
\right)}{Z} }
    =
    \exp \left[-\beta \left(E_i - E_j
\right)
\right]
    =
    \exp \left(-\beta \Delta E_{ij}
\right)
$$
Here $\Delta E_{ij}$ is energy difference from state $j$ to state
$i$.

As I mentioned earlier, first step of Metropolis algorithm is
trying random state transition.
In second step of the algorithm, first caliculate energy difference.
If energy difference is negative, accept the trial transition.
If energy difference is positive, accept with weight
$\exp \left[-\beta \Delta E_{ij}
\right]$.

Below are snapshots of the output of a simulation of the 2d Ising model
using the metropolis algorithm.


If we track paramters through time,
we may be able to spot a phase transition (they're a rare breed).
On the right are plots of the energy and magnetization over time. Below
is susceptibility as obtained the variance of the magnetization,
$\chi = \left< \left< M
\right> - M
\right>$ (:shrug)
s