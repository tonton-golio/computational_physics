
## Microcanonical Ensemble
The central assumption of statistical mechanics is the principle of *equal a priori probabilities*, which argues that all states that share an energy level are equally likely to exist in the system.

With this assumption one can say that the system become particular state
$i$ with probability
$$
    P_i = \frac{1}{\Omega}.
$$
Here $\Omega$ is the total number of (quantum) microstate of the 
system with energy $E$.


According to Ludwig Boltzmann, entropy of system with microcanonical 
ensemble of the system with fixed energy $E$ is expressed as
$$
    S = k_\mathrm{B} \ln \Omega.
$$ 
Here $k_\mathrm{B}$ is Boltzmann's constant.

In this system, temperature $T$ is statistically defined as 
$$ 
\begin{align*}
    \frac{1}{T} 
    &= \frac{\partial S}{\partial E} 
    &= k_\mathrm{B} \frac{\partial}{\partial E} \ln \Omega
\end{align*}
$$


## Canonical Ensemble
#### Temperature of system and reservoir become same in equilibrium
Let's consider subsystem of whole system of microcanonical ensemble. 
For simplicity, we only consider only one subsystem and assuming it 
is small enough comparing to the rest of the system. Let's say this 
small part as 
"system" and rest of enoumous part of the original microcanonical 
system as "reservoir" or "heat bath".
By defining the system's energy as $E_\mathrm{s}$ and reservoir's 
energy as $E_\mathrm{r}$, we consider the energy exchange 
between these two. Because total energy conserved, the sum of two 
energy is constant value.
$$ 
    E_\mathrm{t} = E_\mathrm{s} + E_\mathrm{r} 
$$
Here $E_\mathrm{t}$ is total energy of whole system.

Let's think about number of state.
$\Omega(E_\mathrm{t})$ is the total number of states with energy 
$E_\mathrm{t}$.
$\Omega(E_\mathrm{s}, E_\mathrm{r})$ is the total number of states with 
system has energy $E_\mathrm{s}$ and system has energy $E_\mathrm{r}$.
It can be the product of total number of state of system and reservoir.
$$
    \Omega(E_\mathrm{s}, E_\mathrm{r})
    = \Omega_\mathrm{s}(E_\mathrm{s}) \Omega_\mathrm{r}(E_\mathrm{r})
$$
Entropy of whole system become
$$
    S_\mathrm{t} =  k_\mathrm{B} \ln \Omega(E_\mathrm{s}, E_\mathrm{r}).
$$
Entropy of the system and reservoir become
$$
    S_\mathrm{s} =  k_\mathrm{B} \ln \Omega(E_\mathrm{s}),
$$
$$
    S_\mathrm{r} =  k_\mathrm{B} \ln \Omega(E_\mathrm{r}).
$$
Thus total entropy become sum of system and reservoir.
$$
    S_\mathrm{t} = S_\mathrm{s} + S_\mathrm{r}.
$$

Probability of finding the state system energy $E_\mathrm{s}$ and 
reservoir energy $E_\mathrm{r}$ is
$$
\begin{align*}
    P(E_\mathrm{s}, E_\mathrm{r}) 
    &= \frac{\Omega(E_\mathrm{s}, E_\mathrm{r})}{\Omega(E_\mathrm{t})}
    \&= \frac{\Omega_\mathrm{s}(E_\mathrm{s}) 
    \Omega_\mathrm{r}(E_\mathrm{r})}
    {\Omega(E_\mathrm{t})}
\end{align*}
$$
Most probable state (thermodynamical equilibrium state) satisfies
$$
    \frac{\partial P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
    =0
$$
This condition does not change with logarithmic scale.
$$
    \frac{\partial \ln 
    P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
    =0
$$
Using number of state, this condition can be expressed as 
$$
\begin{align*}
    0 &= 
        \frac{\partial \ln 
        P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
        \&= 
        \frac{\partial}{\partial E_\mathrm{s}} 
        \ln \frac{\Omega_\mathrm{s}(E_\mathrm{s}) 
        \Omega_\mathrm{r}(E_\mathrm{r})}{\Omega(E)}
        \&= 
        \frac{\partial}{\partial E_\mathrm{s}} 
        \ln \Omega_\mathrm{s}(E_\mathrm{s}) 
        +
        \frac{\partial}{\partial E_\mathrm{r}} 
        \frac{\partial E_\mathrm{r}}{\partial E_\mathrm{s}} 
        \ln \Omega_\mathrm{r}(E_\mathrm{r})
        \&= 
        \frac{1}{k_\mathrm{B}}
        \frac{\partial S_\mathrm{s}}{\partial E_\mathrm{s}} 
        -
        \frac{1}{k_\mathrm{B}}
        \frac{\partial S_\mathrm{r}}{\partial E_\mathrm{r}} 
        \&= 
        \frac{T_\mathrm{s}}{k_\mathrm{B}}
        -
        \frac{T_\mathrm{r}}{k_\mathrm{B}}.
\end{align*}
$$
Here $T_\mathrm{s}$ is temperature of system and $T_\mathrm{r}$ is 
temperature of reservoir.
From this result, we can say in equilibrium system's temperature 
and reservoir's temperature is same.
$$
        T_\mathrm{s} = T_\mathrm{r}

$$
If two system and reservoir exchange the energy, they have same 
temperature.

#### Boltzmann distribution 
When system is in state $i$ and has energy $E_i$, reservoir has energy
$E_\mathrm{t} - E_i$, probability of happening this state $i$ is 
$$
\begin{align*}
    P_i 
    &\propto 
        \Omega_\mathrm{r}(E_\mathrm{t} - E_i) 
    \&= 
        \exp 
        \left[ \frac{1}{k_\mathrm{B}} S_r(E_\mathrm{t} - E_i) 
\right]
    \approx 
        \exp \left[ \frac{1}{k_\mathrm{B}} 
        \left(S_\mathrm{r}(E_\mathrm{t}) 
        - \left. 
        \frac{\mathrm{d}S_\mathrm{r}}{\mathrm{d}E} 
        
\right|_{E=E_\mathrm{t}}E_i
\right)
        
\right]
    \&= 
        \exp \left[ \frac{1}{k_\mathrm{B}} 
        \left(S_\mathrm{r}(E_\mathrm{t}) 
        - \frac{E_i}{T_\mathrm{r}}
\right)
        
\right]
    \&\propto 
        \exp \left( 
        -\frac{E_i}{k_\mathrm{B}T_\mathrm{r}} 
        
\right)
\end{align*}
$$
We used Taylor series expansion because $E_\mathrm{t}\gg E_i$.
The term 
$\exp \left(-\frac{E_i}{k_\mathrm{B}T} 
\right)$
called "Boltzmann distribution".
The probability that state $i$ happens is proportional to Boltzmann 
distribution
$P_i \propto \exp \left(-\frac{E_i}{k_\mathrm{B}T} 
\right)$.

#### Partition function, free energy and thermodynamical observable
The partition function is defined as the sum of the Boltzmann factor 
of all states of system.
$$
\begin{align*}
    Z &= \sum_i \exp \left({-\frac{E_i}{k_\mathrm{B}T}} 
\right)
        \&= \sum_i \exp \left({-\beta E_i}
\right).
\end{align*}
$$
Here $\beta = \frac{1}{k_\mathrm{B}T}$.
Thus probability of finding system with state $i$ with energy $E_i$ 
becomes
$$
\begin{align*}
    P_i 
    &=
    \frac
    {\exp \left(-\frac{E_i}{k_\mathrm{B}T} 
\right)}
    {Z} 
    \&=
    \frac
    {\exp \left(-\beta E_i 
\right)}
    {Z} .
\end{align*}
$$

Using the partiton the functions, we are able obtain any thermodynamical
observable. 
A particularly important value we may obtain is the (Helmholtz) free 
energy,
$$
\begin{align*}
    F 
    &= -k_\mathrm{B}T \ln Z 
    \&= -\frac{1}{\beta}\ln Z 
    \&= \left<E
\right> - TS.
\end{align*}
$$
We can also obtain average energy from partition function.
$$
\begin{align*}
    \left<E
\right>
    &=
    \sum_i E_i P_i
    \&=
    \frac{1}{Z} \sum_i E_i \exp \left({-\beta E_i}
\right)
    \&=
    -\frac{1}{Z} \sum_i \frac{\partial}{\partial \beta} 
    \exp \left({-\beta E_i}
\right)
    \&=
    -\frac{1}{Z} \frac{\partial}{\partial \beta} Z
    \&=
    - \frac{\partial}{\partial \beta} \ln Z
\end{align*}
$$
From average energy, we can obtain specific heat.
$$
\begin{align*}
    C 
    &=
    \frac{\partial \left<E
\right>}{\partial T}
    \&= 
    \frac{\partial \left<E
\right>}{\partial \beta}
    \frac{\partial \beta}{\partial T}
    \&=
    -\frac{1}{k_\mathrm{B} T^2}
    \frac{\partial \left<E
\right>}{\partial eta}
    \&=
    \frac{1}{k_\mathrm{B} T^2}
    \frac{\partial^2}{\partial \beta^2} \ln Z
\end{align*}
$$
Specific heat is equal to variace of energy.
$$
\begin{align*}
    k_\mathrm{B} T^2C 
    &=
    \frac{\partial^2}{\partial \beta^2} \ln Z
    \&=
    \frac{\partial}{\partial \beta} 
    \left(
    \frac{\partial}{\partial \beta}
    \ln Z
    
\right)
    \&=
    \frac{\partial}{\partial \beta} 
    \left(
    \frac{1}{Z}
    \frac{\partial Z}{\partial \beta}
    
\right)
    \&=
    -
    \frac{1}{Z^2}
    \left(
    \frac{\partial Z}{\partial \beta}
    
\right)^2
    +
    \frac{1}{Z}
    \frac{\partial^2 Z}{\partial \beta^2}
    \&=
    - \left<E
\right>^2 + \left<E^2
\right> 
    \&=
    \left< \left( E - \left< E 
\right> 
\right)^2 
\right>
    \&=
    \left(\Delta E
\right)^2
\end{align*}
$$
Assuming energy of the system can be approximated by system size 
$\left< E 
\right> \sim N k_\mathrm{B} T$ 
(you know this is good approximation because the energy of 
particle in a box is $E=\frac{3}{2} N k_\mathrm{B}T$), 
specific heat is also 
approximated as $C \sim N k_\mathrm{B}$. 

Thus
$$
    \Delta E \sim N^{1/2}.
$$
Variance of energy scales as square root of system size.

From free energy we can obtain the entropy,         
$$
    S = - \left. \frac{\partial F}{\partial T} 
ight|_H.
$$
From free energy we can also obtain the magnetization,         
$$
    M = - \left.\frac{\partial F}{\partial H} 
ight|_T.
$$
Susceptibility becomes
$$
    \chi_T = - \left.\frac{\partial^2 F}{\partial H^2} 
ight|_T.
$$
Notice that susceptibility is equal to variance of magnetization.
$$
\begin{align*}
    k_\mathrm{B} T\chi_T 
    &= 
    -k_\mathrm{B} T
    \frac{\partial^2 F}{\partial H^2} 
    \&= 
    \frac{1}{\beta^2}
    \frac{\partial}{\partial H} 
    \left(
    \frac{\partial}{\partial H} 
    \ln Z
    
\right)
    \&= 
    \frac{1}{\beta^2}
    \frac{\partial}{\partial H} 
    \left(
    \frac{1}{Z} 
    \frac{\partial Z}{\partial H} 
    
\right)
    \&=
    -
    \frac{1}{\beta^2}
    \frac{1}{Z^2}
    \left(
    \frac{\partial Z}{\partial H}
    
\right)^2
    +
    \frac{1}{\beta^2}
    \frac{1}{Z}
    \frac{\partial^2 Z}{\partial H^2}
    \&=
    - \left<M
\right>^2 + \left<M^2
\right> 
    \&=
    \left< \left( M - \left< M 
\right> 
\right)^2 
\right>
    \&=
    \left(\Delta M
\right)^2
\end{align*}
$$


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
        
## Metropolis algorithm
### Monte Carlo method
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
### Markov process and master equation
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
### Detailed balance
By using "detailed balance" as a condition of equilibrium,
one can avoid such irreversible state transition.
$$ 
    w_{ij}P_j = w_{ji}P_i
$$ 
This argues that flux from state $j$ to state $i$ is equal to 
that of opposite. 
This condition does not allow irreversible state transition.
### Metropolis-Hastings algorithm 
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
$\exp \left[-\beta \Delta E_{ij} 
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