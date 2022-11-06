### Microcanonical Ensemble
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
    \&= k_\mathrm{B} \frac{\partial}{\partial E} \ln \Omega
\end{align*}
$$


### Canonical Ensemble
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
magnetic momemt $s_i$.
He approximated spin (magnetic moment) can be only one of the two 
state.
$$
    s_i = 
    \begin{cases}
    +1 \
    -1
    \end{cases}
$$
$s_i = +1$ represents spin up and $s_i = -1$ represents spin down.

Hamiltonian of Ising model is
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
    M = \sum_{i=1}^N s_i.
$$
or 
$$
    m = \frac{1}{N} \sum_{i=1}^N s_i.
$$
depending on the problem.

Caliculating partition function of 1D-lattice Ising model was 
performed by E. Ising himself, and that of 2D-lattice Ising 
model was aesthetically performed by Lars Onsager. 
However, nobody succeed in caliculating partition function of
3D-lattice Ising model until now as far as I know. 
Thus, generally speaking, caliculating partition function of 
Ising model is one of the hardest problem we human being have.
Nevertheless, we can use computer to numerically caliculate 
thermodynamic observable of canonical ensemble.

## Metropolis algorithm
### Monte Carlo method
Monte Carlo is a subpart of Monaco and it is famous for gambling.
According to Wikipedia, Nicholas Metropolis suggest the name.

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
# Phase transitions & Critical phenomena
## Mean Field Solution to Ising Model
### Hamiltonian and partition function
Hamiltonian and partition function of Ising model are
$$
\mathcal{H} 
= 
-J \sum_{\left<i j
\right>} s_i s_j - h \sum_i s_i
$$
$$
\begin{align*}
Z
&= 
\mathrm{Tr}
\left(
e^{-\beta \mathcal{H}}

\right)
\&= 
\sum_{\{s_i\}} 
e^{-\beta \mathcal{H}\left(\{s_i\}
\right)}
\&= 
\sum_n^{2^N} 
\left\langle n 
\right\vert
e^{-\beta \mathcal{H}}
\left\vert n 
\right
angle
\&= 
\sum_{n}^{2^N} 
e^{-\beta E_n}
\end{align*}
$$
Here $n$ is index of state, $\left\vert n 
\right
angle$ is 
ket state $n$, $N$ is total number of spins and $\{s_i\}$ means
all possible configuration of Ising model.

We cannot caliculate partition function directly except for 
1D-lattice case and 2D-lattice case.
However, by approximating Hamiltonian with mean field, we can 
analytically obtain partition function.
Let's approxiomate Hamiltonian.
### Ignoring high-order fluctuation
First, let's replace $s_i$ with mean $\left< s_i 
\right>$ and 
fluctuation from mean $\delta s_i = s_i - \left< s_i 
\right>$.
$$
\regin{align*}
s_i 
&= 
\left< s_i 
\right> + \delta s_i
\&= 
\left< s_i 
\right> 
+ \left( s_i - \left< s_i 
\right> 
\right)
\end{align*}
$$
Here $\left< s_i 
\right>$ means
$$
\left< s_i 
\right>
=
\frac{1}{Z} \sum_{n=1}^{2^N} s_i \exp \left( -\beta E_n 
\right).
$$
$n$ is index of state 
(total number of all state is $2^N$, $N$ is number of spin). 
Inside of sum of first term of Hamiltonian becomes
$$
\begin{align*}
s_i s_j
&=
\left( 
    \left< s_i 
\right> + \delta s_i

\right) 
\left( 
    \left< s_j 
\right> + \delta s_j

\right) 
\&=
\left< s_i 
\right> \left< s_j 
\right>
+ 
\left< s_i 
\right> \delta s_j
+ 
\delta s_i \left< s_j 
\right> 
+ 
\delta s_i \delta s_j
\& \approx
\left< s_i 
\right> \left< s_j 
\right>
+ 
\left< s_i 
\right> \delta s_j
+ 
\delta s_i \left< s_j 
\right> 
\&=
\left< s_i 
\right> \left< s_j 
\right>
+ 
\left< s_i 
\right> 
\left( 
    s_j - \left< s_j 
\right>

\right) 
+ 
\left( 
    s_i - \left< s_i 
\right>

\right) 
\left< s_i 
\right> 
\&=
\left< s_i 
\right> \left< s_j 
\right>
+ 
\left< s_i 
\right> 
\left( 
    s_j - \left< s_j 
\right>

\right) 
+ 
\left( 
    s_i - \left< s_i 
\right>

\right) 
\left< s_i 
\right> 
\&=
-\left< s_i 
\right>^2
+ 
\left< s_i 
\right> 
\left( 
    s_i + s_j 

\right) 
\end{align*}
$$
We ignore the fluctuation with second order.
We also used 
$$
\left< s_1 \right> 
= \left< s_2 \right> 
= \cdots
= \left< s_i \right> 
= \cdots
= \left< s_N \right>
$$
because all spins are equivalent.

What we need to notice is that magnetization in equilibrium state
is equivalent to mean of spin $\left< s_i 
\right>$.
$$
\begin{align*}
m 
&= 
\frac{1}{N} \sum_{i=1}^N \left< s_i 
\right>
\&= 
\frac{1}{N} \left< s_i 
\right> \sum_{i=1}^N
\&= 
\frac{1}{N} \left< s_i 
\right> N
\&= 
\left< s_i 
\right>
\end{align*}
$$
Thus we can replace $\left< s_i 
\right>$ with $m$.
$$
s_i s_j
\approx 
- m^2 + m(s_i + s_j)
$$
### Mean-field Hamiltonian
Then, mean-field Hamiltonian $\mathcal{H}_\mathrm{MF}$ beocomes
$$
\begin{align*}
\mathcal{H}_\mathrm{MF}
&= 
-J \sum_{\left<i j
\right>} 
\left(- m^2 + m(s_i + s_j) 
\right) 
- h \sum_i s_i
\&= 
J m^2 \sum_{\left<i j
\right>}  
- J \sum_{\left<i j
\right>} m(s_i + s_j)
- h \sum_i s_i
\end{align*}
$$
Let's think about first term.
$$
\begin{align*}
J m^2 \sum_{\left<i j
\right>}
&=
J m^2 \frac{1}{2} 
\sum_{i} \sum_{\left<j
\right>}
\&=
J m^2 \frac{1}{2} 
\sum_{i=1}^N z
\&=
\frac{J N z}{2} m^2 
\end{align*}
$$
Here $z$ is number of nearest-neighbor spins and division by 2 is
for avoiding overlap.

Move on to second term.
$$
\begin{align*}
- J \sum_{\left<i j
\right>} m(s_i + s_j)
&=
- J m 
\left( 
    \sum_{\left<i j
\right>} s_i + \sum_{\left<i j
\right>} s_j

\right)
\&=
- 2 J m \sum_{\left<i j
\right>} s_i
\&=
- 2 J m \frac{1}{2} \sum_{i} s_i \sum_{\left<j
\right>} 
\&=
- J z m  \sum_{i} s_i
\end{align*}
$$
Finally, mean-field Hamiltonian becomes
$$
\begin{align*}
\mathcal{H}_\mathrm{MF}
&= 
\frac{J N z}{2} m^2 
- J z m  \sum_{i} s_i
- h \sum_i s_i
\&= 
\frac{J N z}{2} m^2 
- \left( J z m + h 
\right) \sum_i s_i
\end{align*}
$$
### Mean-field partition function
We shut up and caliculate mean-field partition function.
$$
\begin{align*}
Z_{\mathrm{MF}}
&= 
\sum_{\{s_i\}} 
e^{-\beta \mathcal{H}\left(\{s_i\}
\right)}
\&= 
\sum_{s_1 = \pm 1} \sum_{s_2 = \pm 1} 
\cdots \sum_{s_N = \pm 1}
\exp 
\left[
-\beta 
\mathcal{H} 
\left( \{s_i\} 
\right)

\right]
\&= 
\sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
\exp 
\left[
    - \beta\frac{J N z}{2} m^2 
    + \beta \left( J z m + h 
\right) \sum_i s_i

\right]
\&= 
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
\prod_{i=1}^N
\exp 
\left[
    \beta \left( J z m + h 
\right) s_i

\right]
\&= 
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
\sum_{s_1 = \pm 1} 
\exp 
\left[
    \beta \left( J z m + h 
\right) s_1 

\right]

\right]
\cdots 
\left[
\sum_{s_N = \pm 1}
\exp 
\left[
    \beta \left( J z m + h 
\right) s_N

\right]

\right]
\&= 
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
\sum_{s = \pm 1}
\exp 
\left[
    \beta \left( J z m + h 
\right) s

\right]

\right]^N
\&= 
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    \beta (J z m + h)

\right)

\right]^N
\end{align*}
$$
### Self-consistent equation of magnetization
We can also caliculate statistical mechanically averaged magnetization
$m=\left<s_i
\right>$.
$$
\begin{align*}
m 
&= \left<s_i
\right>
\&=
\frac{1}{Z_{\mathrm{MF}}}
\sum_{s_1 = \pm 1} 
\cdots 
\sum_{s_N = \pm 1}
s_i
\exp 
\left[
-\beta 
\mathcal{H} 
\left( \{s_i\} 
\right)

\right]
\&=
\frac{1}{Z_{\mathrm{MF}}}
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
\sum_{s_1 = \pm 1} 
\exp 
\left[
    \beta \left(J z m + h 
\right) s_1 

\right]

\right]
\cdots 
\left[
\sum_{s_i = \pm 1} 
s_i
\exp 
\left[
    \beta \left(J z m + h 
\right) s_i

\right]

\right]
\cdots 
\left[
\sum_{s_N = \pm 1} 
\exp 
\left[
    \beta \left(J z m + h 
\right) s_N

\right]

\right]
\&=
\frac{1}{Z_{\mathrm{MF}}}
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
2\cosh 
\left(
    \beta J z m + \beta h

\right)

\right]
\cdots 
\left[
2\sinh
\left(
    \beta J z m + \beta h

\right)

\right]
\cdots 
\left[
2\cosh
\left(
    \beta J z m + \beta h

\right)

\right]
\&=
\frac
{
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    \beta J z m + \beta h

\right)

\right]^{N-1}
2\sinh
\left(
    \beta J z m + \beta h

\right)
}
{
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    \beta J z m + \beta h

\right)

\right]^N
}
\&=
    anh
\left(
    \beta J z m + \beta h 

\right)
\end{align*}
$$
When there is no external magnetic field $h=0$, this would be
$$
m = 	anh \left( \beta J z m 
\right)
$$
We got a self-consistent equation of magnetization.
This form is analytically unsolvable but we can obtain $m$ from 
graphical method. 
By independetly plot the functions $y = m$ and 
$y=	anh \left( \beta J z m 
\right)$, we can see $m$ which satisfies 
the self-consistent equation (consider $J$ and $z$ as constant).
- make graph
We clearly see that as the temperature increases, in a particular 
temperature, number of solution become one from three.
This qualitative change is phase transition.
### Critical temperature of mean-field approximation
At which temperature, does this transition happen? 
When the slope of $y=	anh \left( \beta J z m 
\right)$ is same as 
$y=m$ at $m=0$, these two equation has single intersection.
Taylor series of $	anh(x)$ at $x=0$ is $	anh(x) pprox x$.
Near $m=0$ we can say $	anh(\beta Jzm) = \beta Jzm$.
Then slope become
$$
\begin{align*}
\frac{\mathrm{d}}{\mathrm{d}m} 	anh(\beta Jzm)
\approx
\frac{\mathrm{d}}{\mathrm{d}m} \beta Jzm
\&=
\beta Jz
\end{align*}
$$
This is equal to the slope of $y=m$ at $m=0$ which is $1$.
Then critical temperature $T_\mathrm{c}$ sarisfies
$\frac{Jz}{k_\mathrm{B} T_\mathrm{c}} = 1$.
Critical temperature $T_\mathrm{c}$ is 
$$
T_\mathrm{c}
=
\frac{Jz}{k_\mathrm{B}}.
$$
Critical temperature increases with $J$ and $z$ but it does not 
depend on dimension of lattice.
As we know, there is no phase transition in 1D-lattice Ising model.
This results is qualitatively wrong in that case but as the dimension 
become infinity this result become correct.
### Free energy of mean-field approximation
We already have partition function. 
Why don't we get free energy?
$$
\begin{align*}
F_\mathrm{MF} 
&= 
- \frac{1}{\beta} \ln Z_\mathrm{MF}
\&=
- \frac{1}{\beta} \ln  
\left[
\exp 
\left(
    - \beta\frac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    \beta J z m + \beta h

\right)

\right]^N

\right]
\&=
\frac{JNz}{2}m^2
- \frac{N}{\beta} \ln
\left[
    2 \cosh \left(\beta Jzm + \beta h 
\right)

\right]
\end{align*}
$$
Notice that by differentiating free energy with magnetic field, we
can obtain magnetization.
$$
\begin{align*}
m &= \frac{1}{N}M
    \&= 
    -
    \frac{1}{N} 
    \frac{\partial F_\mathrm{MF}}{\partial h}
    \&= 
    \frac{1}{N} 
    \frac{N}{\beta}
    \frac
    {2 \sinh \left(\beta Jzm + \beta h 
\right)}
    {2 \cosh \left(\beta Jzm + \beta h 
\right)}
    \beta
    \&=
    anh \left(\beta Jzm + \beta h 
\right)
\end{align*}
$$
### Critical exponent of mean-field approximation
By introducint dimensionless temperature parameter $	heta$,
$$
\begin{align*}
heta 
&= 
\frac{T}{T_\mathrm{c}}
\&= 
\frac{k_\mathrm{B}T}{Jz}
\&= 
\frac{1}{\beta Jz}
\end{align*}
$$
we can rewrite free energy as dimensionless form.
$$
\begin{align*}
f_\mathrm{MF}
&=
\frac{F_\mathrm{MF}}{JzN}
\&= 
\frac{m^2}{2}
- \frac{1}{\beta Jz} 
\ln
\left[
    2 \cosh \left(\beta Jzm + \beta Jz h \frac{1}{Jz} 
\right)

\right]
\&= 
\frac{m^2}{2}
- 	heta \ln 2
- 	heta 
\ln \cosh 
\left( 
    \frac{1}{	heta} m 
        + \frac{1}{	heta} \frac{h}{Jz} 

\right)
\&= 
\frac{m^2}{2}
- 	heta \ln 2
- 	heta \ln \cosh 
\left( 
    \frac{m + h'}{	heta} 

\right)
\end{align*}
$$
Here $h':=\frac{h}{Jz}$.
To get intuitive idea of this free energy, let's use the Mclaurin 
series expansion 
$\cosh(x) = 1 + \frac{x^2}{2} + \frac{x^4}{24} + \mathcal{O}(x^6)$.
In the case of $h'=0$ and $\frac{m}{	heta} \ll 1$, 
free energy can be expanded as
$$
f_\mathrm{MF}
= 
\frac{m^2}{2}
- 	heta \ln 2
- 	heta \ln 
\left[ 
    1
    +\frac{1}{2}\frac{m^2}{	heta^2}
    +\frac{1}{24}\frac{m^4}{	heta^4}
    + \mathcal{O}(\frac{m^6}{	heta^6})

\right].
$$
Then we use the Mclaurin series expansion 
$\ln(1+x) \approx x - \frac{x^2}{2}$
$$
\begin{align*}
f_\mathrm{MF}
&= 
\frac{m^2}{2}
- 	heta \ln 2
- 	heta 
\left[ 
    (\frac{1}{2}\frac{m^2}{	heta^2} 
    + \frac{1}{24}\frac{m^4}{	heta^4})
    - \frac{1}{2}
    (\frac{1}{2}\frac{m^2}{	heta^2} 
    + \frac{1}{24}\frac{m^4}{	heta^4})^2
    + \mathcal{O}\left(\frac{m^6}{	heta^6}
\right)

\right]
\&= 
\frac{m^2}{2}
- 	heta \ln 2
- 	heta 
\left[ 
    \frac{1}{2}\frac{m^2}{	heta^2} 
    + \frac{1}{24}\frac{m^4}{	heta^4}
    - \frac{1}{2}\frac{1}{4}\frac{m^4}{	heta^4} 
    + \mathcal{O}\left(\frac{m^6}{	heta^6}
\right)

\right]
\&= 
\frac{m^2}{2}
- 	heta \ln 2
- 	heta 
\left[ 
    \frac{1}{2}\frac{m^2}{	heta^2} 
    - \frac{1}{12}\frac{m^4}{	heta^4}
    + \mathcal{O}\left(\frac{m^6}{	heta^6}
\right)

\right]
\&= 
\frac{1}{12}\frac{m^4}{	heta^3}
+ \frac{1}{2}m^2 \left( 1-\frac{1}{	heta} 
\right)
- 	heta \ln 2
+ \mathcal{O}\left(\frac{m^6}{	heta^6}
\right)
\end{align*}
$$
Let's check the extrema of this free energy.
$$
\begin{align*}
0
&=
\frac{\partial f_\mathrm{MF}}{\partial m}
\&=
\frac{1}{3}\frac{m^3}{	heta^3}
+ m \left( 1-\frac{1}{	heta} 
\right)
\&=
m 
\left(
\frac{1}{3}\frac{m^2}{	heta^3}
+ \left( 1-\frac{1}{	heta} 
\right)

\right)
\end{align*}
$$
Except for $m=0$ this has a solution
$$
\begin{align*}
m^2
&=
\left( \frac{1}{	heta} - 1 
\right) \cdot 3	heta^3
\&=
3\left( 1-	heta 
\right) 	heta^2
\&=
-3 t 	heta^2.
\end{align*}
$$
Here we introduced reduced temperature
$$
\begin{align*}
t 
&:= 
\frac{T - T_\mathrm{C}}{T_\mathrm{C}}
\&= 
    heta - 1
\end{align*}
$$
Thus when $t<0$ i.e. $T<T_\mathrm{c}$, there are three local extrema 
i.e. $m=0, \pm \sqrt{3} |t|^{1/2} 	heta$.
When $t>0$ i.e. $T>T_\mathrm{c}$, there is single local extrema $m=0$.
From $m \sim (-t)^\beta \sim (-t)^{1/2}$, exponent of magnetization is
$1/2$.

To check these extrema are maxima or minima, we need to check second 
derivative of free energy.
$$
\begin{align*}
\frac{\partial^2 f_\mathrm{MF}}{\partial m^2}
&=
\frac{m^2}{	heta^3}
+ \left( 1-\frac{1}{	heta} 
\right)
\&=
\frac{m^2}{(1+t)^3}
+ \left( \frac{1+t-1}{1+t} 
\right)
\&=
\frac{m^2}{(1+t)^3}
+ \left( \frac{t}{1+t} 
\right)
\&=
\frac{1}{1+t}
\left[
\frac{m^2}{(1+t)^2} + t

\right]
\end{align*}
$$
Thus when $t<0$ second derivative is positive for nonzero $m$?????
????????????????????

