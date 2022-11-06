### Microcanonical Ensemble
The central assumption of statistical mechanics is the principle of *equal a priori probabilities*, which argues that all states that share an energy level are equally likely to exist in the system.

With this assumption one can say that the system become particular state
$i$ with probability
$$
    P_i = rac{1}{\Omega}.
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
egin{align*}
    rac{1}{T} 
    &= rac{\partial S}{\partial E} 
    \&= k_\mathrm{B} rac{\partial}{\partial E} \ln \Omega
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
egin{align*}
    P(E_\mathrm{s}, E_\mathrm{r}) 
    &= rac{\Omega(E_\mathrm{s}, E_\mathrm{r})}{\Omega(E_\mathrm{t})}
    \&= rac{\Omega_\mathrm{s}(E_\mathrm{s}) 
    \Omega_\mathrm{r}(E_\mathrm{r})}
    {\Omega(E_\mathrm{t})}
\end{align*}
$$
Most probable state (thermodynamical equilibrium state) satisfies
$$
    rac{\partial P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
    =0
$$
This condition does not change with logarithmic scale.
$$
    rac{\partial \ln 
    P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
    =0
$$
Using number of state, this condition can be expressed as 
$$
egin{align*}
    0 &= 
        rac{\partial \ln 
        P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
        \&= 
        rac{\partial}{\partial E_\mathrm{s}} 
        \ln rac{\Omega_\mathrm{s}(E_\mathrm{s}) 
        \Omega_\mathrm{r}(E_\mathrm{r})}{\Omega(E)}
        \&= 
        rac{\partial}{\partial E_\mathrm{s}} 
        \ln \Omega_\mathrm{s}(E_\mathrm{s}) 
        +
        rac{\partial}{\partial E_\mathrm{r}} 
        rac{\partial E_\mathrm{r}}{\partial E_\mathrm{s}} 
        \ln \Omega_\mathrm{r}(E_\mathrm{r})
        \&= 
        rac{1}{k_\mathrm{B}}
        rac{\partial S_\mathrm{s}}{\partial E_\mathrm{s}} 
        -
        rac{1}{k_\mathrm{B}}
        rac{\partial S_\mathrm{r}}{\partial E_\mathrm{r}} 
        \&= 
        rac{T_\mathrm{s}}{k_\mathrm{B}}
        -
        rac{T_\mathrm{r}}{k_\mathrm{B}}.
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
egin{align*}
    P_i 
    &\propto 
        \Omega_\mathrm{r}(E_\mathrm{t} - E_i) 
    \&= 
        \exp 
        \left[ rac{1}{k_\mathrm{B}} S_r(E_\mathrm{t} - E_i) 
\right]
    \&pprox 
        \exp \left[ rac{1}{k_\mathrm{B}} 
        \left(S_\mathrm{r}(E_\mathrm{t}) 
        - \left. 
        rac{\mathrm{d}S_\mathrm{r}}{\mathrm{d}E} 
        
ight|_{E=E_\mathrm{t}}E_i
\right)
        
\right]
    \&= 
        \exp \left[ rac{1}{k_\mathrm{B}} 
        \left(S_\mathrm{r}(E_\mathrm{t}) 
        - rac{E_i}{T_\mathrm{r}}
\right)
        
\right]
    \&\propto 
        \exp \left( 
        -rac{E_i}{k_\mathrm{B}T_\mathrm{r}} 
        
\right)
\end{align*}
$$
We used Taylor series expansion because $E_\mathrm{t}\gg E_i$.
The term 
$\exp \left(-rac{E_i}{k_\mathrm{B}T} 
\right)$
called "Boltzmann distribution".
The probability that state $i$ happens is proportional to Boltzmann 
distribution
$P_i \propto \exp \left(-rac{E_i}{k_\mathrm{B}T} 
\right)$.

#### Partition function, free energy and thermodynamical observable
The partition function is defined as the sum of the Boltzmann factor 
of all states of system.
$$
egin{align*}
    Z &= \sum_i \exp \left({-rac{E_i}{k_\mathrm{B}T}} 
\right)
        \&= \sum_i \exp \left({-eta E_i}
\right).
\end{align*}
$$
Here $eta = rac{1}{k_\mathrm{B}T}$.
Thus probability of finding system with state $i$ with energy $E_i$ 
becomes
$$
egin{align*}
    P_i 
    &=
    rac
    {\exp \left(-rac{E_i}{k_\mathrm{B}T} 
\right)}
    {Z} 
    \&=
    rac
    {\exp \left(-eta E_i 
\right)}
    {Z} .
\end{align*}
$$

Using the partiton the functions, we are able obtain any thermodynamical
observable. 
A particularly important value we may obtain is the (Helmholtz) free 
energy,
$$
egin{align*}
    F 
    &= -k_\mathrm{B}T \ln Z 
    \&= -rac{1}{eta}\ln Z 
    \&= \left<E
\right> - TS.
\end{align*}
$$
We can also obtain average energy from partition function.
$$
egin{align*}
    \left<E
\right>
    &=
    \sum_i E_i P_i
    \&=
    rac{1}{Z} \sum_i E_i \exp \left({-eta E_i}
\right)
    \&=
    -rac{1}{Z} \sum_i rac{\partial}{\partial eta} 
    \exp \left({-eta E_i}
\right)
    \&=
    -rac{1}{Z} rac{\partial}{\partial eta} Z
    \&=
    - rac{\partial}{\partial eta} \ln Z
\end{align*}
$$
From average energy, we can obtain specific heat.
$$
egin{align*}
    C 
    &=
    rac{\partial \left<E
\right>}{\partial T}
    \&= 
    rac{\partial \left<E
\right>}{\partial eta}
    rac{\partial eta}{\partial T}
    \&=
    -rac{1}{k_\mathrm{B} T^2}
    rac{\partial \left<E
\right>}{\partial eta}
    \&=
    rac{1}{k_\mathrm{B} T^2}
    rac{\partial^2}{\partial eta^2} \ln Z
\end{align*}
$$
Specific heat is equal to variace of energy.
$$
egin{align*}
    k_\mathrm{B} T^2C 
    &=
    rac{\partial^2}{\partial eta^2} \ln Z
    \&=
    rac{\partial}{\partial eta} 
    \left(
    rac{\partial}{\partial eta}
    \ln Z
    
\right)
    \&=
    rac{\partial}{\partial eta} 
    \left(
    rac{1}{Z}
    rac{\partial Z}{\partial eta}
    
\right)
    \&=
    -
    rac{1}{Z^2}
    \left(
    rac{\partial Z}{\partial eta}
    
\right)^2
    +
    rac{1}{Z}
    rac{\partial^2 Z}{\partial eta^2}
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
particle in a box is $E=rac{3}{2} N k_\mathrm{B}T$), 
specific heat is also 
approximated as $C \sim N k_\mathrm{B}$. 

Thus
$$
    \Delta E \sim N^{1/2}.
$$
Variance of energy scales as square root of system size.

From free energy we can obtain the entropy,         
$$
    S = - \left. rac{\partial F}{\partial T} 
ight|_H.
$$
From free energy we can also obtain the magnetization,         
$$
    M = - \left.rac{\partial F}{\partial H} 
ight|_T.
$$
Susceptibility becomes
$$
    \chi_T = - \left.rac{\partial^2 F}{\partial H^2} 
ight|_T.
$$
Notice that susceptibility is equal to variance of magnetization.
$$
egin{align*}
    k_\mathrm{B} T\chi_T 
    &= 
    -k_\mathrm{B} T
    rac{\partial^2 F}{\partial H^2} 
    \&= 
    rac{1}{eta^2}
    rac{\partial}{\partial H} 
    \left(
    rac{\partial}{\partial H} 
    \ln Z
    
\right)
    \&= 
    rac{1}{eta^2}
    rac{\partial}{\partial H} 
    \left(
    rac{1}{Z} 
    rac{\partial Z}{\partial H} 
    
\right)
    \&=
    -
    rac{1}{eta^2}
    rac{1}{Z^2}
    \left(
    rac{\partial Z}{\partial H}
    
\right)^2
    +
    rac{1}{eta^2}
    rac{1}{Z}
    rac{\partial^2 Z}{\partial H^2}
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
    egin{cases}
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
= rac{1}{2} \sum_i \sum_{\left< j 
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
    m = rac{1}{N} \sum_{i=1}^N s_i.
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
egin{align*}
    \left<A
\right>
    &=
    rac{\sum_i A_i \exp \left(-eta E_i 
\right) }{Z}
    \&=
    rac
    {\sum_i A \exp \left(-eta E_i 
\right)}
    {\sum_i \exp \left(-eta E_i 
\right)}
    \&=
    \sum_i A_i P_i
\end{align*}
$$
This can be approximated by 
$$
    \left<A
\right>
    pprox 
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
    rac{\mathrm{d}P_i}{\mathrm{d}t}
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
i.e. $rac{\mathrm{d}P_i}{\mathrm{d}t} = 0$.
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
    rac
    {\exp \left(-eta E_i 
\right)}
    {Z} .
$$
By combining this with detailed balance condition
$$
    rac{w_{ij}}{w_{ji}} 
    =
    rac{P_i}{P_j} 
    =
    rac
    { rac{\exp \left(-eta E_i 
\right)}{Z} }
    { rac{\exp \left(-eta E_j 
\right)}{Z} }
    =
    \exp \left[-eta \left(E_i - E_j
\right) 
\right]
    =
    \exp \left(-eta \Delta E_{ij} 
\right)
$$
Here $\Delta E_{ij}$ is energy difference from state $j$ to state
$i$.

As I mentioned earlier, first step of Metropolis algorithm is 
trying random state transition.
In second step of the algorithm, first caliculate energy difference.
If energy difference is negative, accept the trial transition.
If energy difference is positive, accept with weight 
$\exp \left[-eta \Delta E_{ij} 
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
egin{align*}
Z
&= 
\mathrm{Tr}
\left(
e^{-eta \mathcal{H}}

\right)
\&= 
\sum_{\{s_i\}} 
e^{-eta \mathcal{H}\left(\{s_i\}
\right)}
\&= 
\sum_n^{2^N} 
\left\langle n 
ightert
e^{-eta \mathcal{H}}
\leftert n 
ight
angle
\&= 
\sum_{n}^{2^N} 
e^{-eta E_n}
\end{align*}
$$
Here $n$ is index of state, $\leftert n 
ight
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
egin{align*}
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
rac{1}{Z} \sum_{n=1}^{2^N} s_i \exp \left( -eta E_n 
\right).
$$
$n$ is index of state 
(total number of all state is $2^N$, $N$ is number of spin). 
Inside of sum of first term of Hamiltonian becomes
$$
egin{align*}
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
\& pprox
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
egin{align*}
m 
&= 
rac{1}{N} \sum_{i=1}^N \left< s_i 
\right>
\&= 
rac{1}{N} \left< s_i 
\right> \sum_{i=1}^N
\&= 
rac{1}{N} \left< s_i 
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
pprox 
- m^2 + m(s_i + s_j)
$$
### Mean-field Hamiltonian
Then, mean-field Hamiltonian $\mathcal{H}_\mathrm{MF}$ beocomes
$$
egin{align*}
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
egin{align*}
J m^2 \sum_{\left<i j
\right>}
&=
J m^2 rac{1}{2} 
\sum_{i} \sum_{\left<j
\right>}
\&=
J m^2 rac{1}{2} 
\sum_{i=1}^N z
\&=
rac{J N z}{2} m^2 
\end{align*}
$$
Here $z$ is number of nearest-neighbor spins and division by 2 is
for avoiding overlap.

Move on to second term.
$$
egin{align*}
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
- 2 J m rac{1}{2} \sum_{i} s_i \sum_{\left<j
\right>} 
\&=
- J z m  \sum_{i} s_i
\end{align*}
$$
Finally, mean-field Hamiltonian becomes
$$
egin{align*}
\mathcal{H}_\mathrm{MF}
&= 
rac{J N z}{2} m^2 
- J z m  \sum_{i} s_i
- h \sum_i s_i
\&= 
rac{J N z}{2} m^2 
- \left( J z m + h 
\right) \sum_i s_i
\end{align*}
$$
### Mean-field partition function
We shut up and caliculate mean-field partition function.
$$
egin{align*}
Z_{\mathrm{MF}}
&= 
\sum_{\{s_i\}} 
e^{-eta \mathcal{H}\left(\{s_i\}
\right)}
\&= 
\sum_{s_1 = \pm 1} \sum_{s_2 = \pm 1} 
\cdots \sum_{s_N = \pm 1}
\exp 
\left[
-eta 
\mathcal{H} 
\left( \{s_i\} 
\right)

\right]
\&= 
\sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
\exp 
\left[
    - etarac{J N z}{2} m^2 
    + eta \left( J z m + h 
\right) \sum_i s_i

\right]
\&= 
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
\prod_{i=1}^N
\exp 
\left[
    eta \left( J z m + h 
\right) s_i

\right]
\&= 
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
\sum_{s_1 = \pm 1} 
\exp 
\left[
    eta \left( J z m + h 
\right) s_1 

\right]

\right]
\cdots 
\left[
\sum_{s_N = \pm 1}
\exp 
\left[
    eta \left( J z m + h 
\right) s_N

\right]

\right]
\&= 
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
\sum_{s = \pm 1}
\exp 
\left[
    eta \left( J z m + h 
\right) s

\right]

\right]^N
\&= 
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    eta J z m + eta h

\right)

\right]^N
\end{align*}
$$
### Self-consistent equation of magnetization
We can also caliculate statistical mechanically averaged magnetization
$m=\left<s_i
\right>$.
$$
egin{align*}
m 
&= \left<s_i
\right>
\&=
rac{1}{Z_{\mathrm{MF}}}
\sum_{s_1 = \pm 1} 
\cdots 
\sum_{s_N = \pm 1}
s_i
\exp 
\left[
-eta 
\mathcal{H} 
\left( \{s_i\} 
\right)

\right]
\&=
rac{1}{Z_{\mathrm{MF}}}
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
\sum_{s_1 = \pm 1} 
\exp 
\left[
    eta \left(J z m + h 
\right) s_1 

\right]

\right]
\cdots 
\left[
\sum_{s_i = \pm 1} 
s_i
\exp 
\left[
    eta \left(J z m + h 
\right) s_i

\right]

\right]
\cdots 
\left[
\sum_{s_N = \pm 1} 
\exp 
\left[
    eta \left(J z m + h 
\right) s_N

\right]

\right]
\&=
rac{1}{Z_{\mathrm{MF}}}
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
2\cosh 
\left(
    eta J z m + eta h

\right)

\right]
\cdots 
\left[
2\sinh
\left(
    eta J z m + eta h

\right)

\right]
\cdots 
\left[
2\cosh
\left(
    eta J z m + eta h

\right)

\right]
\&=
rac
{
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    eta J z m + eta h

\right)

\right]^{N-1}
2\sinh
\left(
    eta J z m + eta h

\right)
}
{
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    eta J z m + eta h

\right)

\right]^N
}
\&=
    anh
\left(
    eta J z m + eta h 

\right)
\end{align*}
$$
When there is no external magnetic field $h=0$, this would be
$$
m = 	anh \left( eta J z m 
\right)
$$
We got a self-consistent equation of magnetization.
This form is analytically unsolvable but we can obtain $m$ from 
graphical method. 
By independetly plot the functions $y = m$ and 
$y=	anh \left( eta J z m 
\right)$, we can see $m$ which satisfies 
the self-consistent equation (consider $J$ and $z$ as constant).
- make graph
We clearly see that as the temperature increases, in a particular 
temperature, number of solution become one from three.
This qualitative change is phase transition.
### Critical temperature of mean-field approximation
At which temperature, does this transition happen? 
When the slope of $y=	anh \left( eta J z m 
\right)$ is same as 
$y=m$ at $m=0$, these two equation has single intersection.
Taylor series of $	anh(x)$ at $x=0$ is $	anh(x) pprox x$.
Near $m=0$ we can say $	anh(eta Jzm) = eta Jzm$.
Then slope become
$$
egin{align*}
rac{\mathrm{d}}{\mathrm{d}m} 	anh(eta Jzm)
&pprox
rac{\mathrm{d}}{\mathrm{d}m} eta Jzm
\&=
eta Jz
\end{align*}
$$
This is equal to the slope of $y=m$ at $m=0$ which is $1$.
Then critical temperature $T_\mathrm{c}$ sarisfies
$rac{Jz}{k_\mathrm{B} T_\mathrm{c}} = 1$.
Critical temperature $T_\mathrm{c}$ is 
$$
T_\mathrm{c}
=
rac{Jz}{k_\mathrm{B}}.
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
egin{align*}
F_\mathrm{MF} 
&= 
- rac{1}{eta} \ln Z_\mathrm{MF}
\&=
- rac{1}{eta} \ln  
\left[
\exp 
\left(
    - etarac{J N z}{2} m^2 

\right)
\left[
2\cosh
\left(
    eta J z m + eta h

\right)

\right]^N

\right]
\&=
rac{JNz}{2}m^2
- rac{N}{eta} \ln
\left[
    2 \cosh \left(eta Jzm + eta h 
\right)

\right]
\end{align*}
$$
Notice that by differentiating free energy with magnetic field, we
can obtain magnetization.
$$
egin{align*}
m &= rac{1}{N}M
    \&= 
    -
    rac{1}{N} 
    rac{\partial F_\mathrm{MF}}{\partial h}
    \&= 
    rac{1}{N} 
    rac{N}{eta}
    rac
    {2 \sinh \left(eta Jzm + eta h 
\right)}
    {2 \cosh \left(eta Jzm + eta h 
\right)}
    eta
    \&=
    anh \left(eta Jzm + eta h 
\right)
\end{align*}
$$
### Critical exponent of mean-field approximation
By introducint dimensionless temperature parameter $	heta$,
$$
egin{align*}
heta 
&= 
rac{T}{T_\mathrm{c}}
\&= 
rac{k_\mathrm{B}T}{Jz}
\&= 
rac{1}{eta Jz}
\end{align*}
$$
we can rewrite free energy as dimensionless form.
$$
egin{align*}
f_\mathrm{MF}
&=
rac{F_\mathrm{MF}}{JzN}
\&= 
rac{m^2}{2}
- rac{1}{eta Jz} 
\ln
\left[
    2 \cosh \left(eta Jzm + eta Jz h rac{1}{Jz} 
\right)

\right]
\&= 
rac{m^2}{2}
- 	heta \ln 2
- 	heta 
\ln \cosh 
\left( 
    rac{1}{	heta} m 
        + rac{1}{	heta} rac{h}{Jz} 

\right)
\&= 
rac{m^2}{2}
- 	heta \ln 2
- 	heta \ln \cosh 
\left( 
    rac{m + h'}{	heta} 

\right)
\end{align*}
$$
Here $h':=rac{h}{Jz}$.
To get intuitive idea of this free energy, let's use the Mclaurin 
series expansion 
$\cosh(x) = 1 + rac{x^2}{2} + rac{x^4}{24} + \mathcal{O}(x^6)$.
In the case of $h'=0$ and $rac{m}{	heta} \ll 1$, 
free energy can be expanded as
$$
f_\mathrm{MF}
= 
rac{m^2}{2}
- 	heta \ln 2
- 	heta \ln 
\left[ 
    1
    +rac{1}{2}rac{m^2}{	heta^2}
    +rac{1}{24}rac{m^4}{	heta^4}
    + \mathcal{O}(rac{m^6}{	heta^6})

\right].
$$
Then we use the Mclaurin series expansion 
$\ln(1+x) pprox x - rac{x^2}{2}$
$$
egin{align*}
f_\mathrm{MF}
&= 
rac{m^2}{2}
- 	heta \ln 2
- 	heta 
\left[ 
    (rac{1}{2}rac{m^2}{	heta^2} 
    + rac{1}{24}rac{m^4}{	heta^4})
    - rac{1}{2}
    (rac{1}{2}rac{m^2}{	heta^2} 
    + rac{1}{24}rac{m^4}{	heta^4})^2
    + \mathcal{O}\left(rac{m^6}{	heta^6}
\right)

\right]
\&= 
rac{m^2}{2}
- 	heta \ln 2
- 	heta 
\left[ 
    rac{1}{2}rac{m^2}{	heta^2} 
    + rac{1}{24}rac{m^4}{	heta^4}
    - rac{1}{2}rac{1}{4}rac{m^4}{	heta^4} 
    + \mathcal{O}\left(rac{m^6}{	heta^6}
\right)

\right]
\&= 
rac{m^2}{2}
- 	heta \ln 2
- 	heta 
\left[ 
    rac{1}{2}rac{m^2}{	heta^2} 
    - rac{1}{12}rac{m^4}{	heta^4}
    + \mathcal{O}\left(rac{m^6}{	heta^6}
\right)

\right]
\&= 
rac{1}{12}rac{m^4}{	heta^3}
+ rac{1}{2}m^2 \left( 1-rac{1}{	heta} 
\right)
- 	heta \ln 2
+ \mathcal{O}\left(rac{m^6}{	heta^6}
\right)
\end{align*}
$$
Let's check the extrema of this free energy.
$$
egin{align*}
0
&=
rac{\partial f_\mathrm{MF}}{\partial m}
\&=
rac{1}{3}rac{m^3}{	heta^3}
+ m \left( 1-rac{1}{	heta} 
\right)
\&=
m 
\left(
rac{1}{3}rac{m^2}{	heta^3}
+ \left( 1-rac{1}{	heta} 
\right)

\right)
\end{align*}
$$
Except for $m=0$ this has a solution
$$
egin{align*}
m^2
&=
\left( rac{1}{	heta} - 1 
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
egin{align*}
t 
&:= 
rac{T - T_\mathrm{C}}{T_\mathrm{C}}
\&= 
    heta - 1
\end{align*}
$$
Thus when $t<0$ i.e. $T<T_\mathrm{c}$, there are three local extrema 
i.e. $m=0, \pm \sqrt{3} |t|^{1/2} 	heta$.
When $t>0$ i.e. $T>T_\mathrm{c}$, there is single local extrema $m=0$.
From $m \sim (-t)^eta \sim (-t)^{1/2}$, exponent of magnetization is
$1/2$.

To check these extrema are maxima or minima, we need to check second 
derivative of free energy.
$$
egin{align*}
rac{\partial^2 f_\mathrm{MF}}{\partial m^2}
&=
rac{m^2}{	heta^3}
+ \left( 1-rac{1}{	heta} 
\right)
\&=
rac{m^2}{(1+t)^3}
+ \left( rac{1+t-1}{1+t} 
\right)
\&=
rac{m^2}{(1+t)^3}
+ \left( rac{t}{1+t} 
\right)
\&=
rac{1}{1+t}
\left[
rac{m^2}{(1+t)^2} + t

\right]
\end{align*}
$$
Thus when $t<0$ second derivative is positive for nonzero $m$?????
????????????????????

