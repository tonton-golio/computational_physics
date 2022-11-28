
# title
Differential equation in a nutshell

# Week 2 description
- Nov 28: Intuitively understanding differential equation for gene regulation 
(no worries, we do not need to solve difficult differential equation!!).

# Differential equation for creation
Consider the situation that molecules (such as mRNA) are created with the rate $k$ 
(molecules/unit time).
What is the number of molecules after short time $\Delta t$?
Defining $n(t)$ is number of molecules at time $t$, we see the number of molecules
after short time $\Delta t$ become
$$
    n(t+\Delta t) = n(t) + k \Delta t.
$$
Here $k \Delta t$ is number of molecules created in time period $\Delta t$.
Thus we just get the number of molecules after time $\Delta t$ by just summing
up the number of molecules at time $t$ and number of molecules created in time 
period $\Delta t$.

This equation can mathematically equivalent to 
$$
    \frac{n(t+\Delta t) - n(t)}{\Delta t}
    = 
    k.
$$
In the limit of infinitely small time period $\Delta t \rightarrow 0$, left hand 
side of this equation would be described by differentiation.
$$
    \frac{\mathrm{d} n(t)}{\mathrm{d} t}
    = 
    k.
$$
Thus this simple molecules creation process can be discribed by differential 
equation.

Solution become
$$
    n(t)
    =
    n(0) + kt.
$$
Therefore $n(t)$ increases linearly with time.

# Differential equation for degradation
Next consider the another situation that molecules are degraded with rate $\Gamma$
(1/unit time).
Notice the unit of degradation rate is frequency.
Thus number of molecules degraded in time duration $\Delta t$ would be 
$$
\Gamma n(t) \Delta t
$$
Using same way of previous section, we get
$$
    n(t+\Delta t) = n(t) - \Gamma n(t) \Delta t.
$$
$$
    \frac{n(t+\Delta t) - n(t)}{\Delta t}
    = 
    -\Gamma n(t).
$$
$$
    \frac{\mathrm{d} n(t)}{\mathrm{d} t}
    = 
    -\Gamma n(t).
$$
Solution become
$$
    n(t)
    =
    n(0) \exp \left(-\Gamma t\right).
$$
Therefore $n(t)$ decays exponentially with time.

# Differential equation for creation and degradation
Finally we can make differential equation for system with both creation and 
degradation.
By combining previous two section we have a equation for how number of molecules 
evolve with time.
$$
    \frac{\mathrm{d} n(t)}{\mathrm{d} t}
    = 
    k -\Gamma n(t).
$$
Here, as same in the previous two sections, $k$ (molecules/unit time) is 
creation rate and $\Gamma$ (1/unit time) is degradation rate.

If we let the system go for sufficiently long time, the system reach steady state
i.e. number of molecules does not depend on time.
We can get number of molecules in the steady state by setting 
$\frac{\mathrm{d} n(t)}{\mathrm{d} t} = 0$.
$$
    0
    = 
    k -\Gamma n_\mathrm{ss}.
$$
Thus, 
$$
    n_\mathrm{ss}
    = 
    \frac{k}{\Gamma}.
$$

# Differential equation for transcription and translation
Let's consider more biological process which includes both transcription and 
translation.
For transcription, the differential equation becomes
$$
    \frac{\mathrm{d} n_\mathrm{m}(t)}{\mathrm{d} t}
    = 
    k_\mathrm{m} - \Gamma_\mathrm{m} n_\mathrm{m}(t).
$$
That of translation becomes
$$
    \frac{\mathrm{d} n_\mathrm{p}(t)}{\mathrm{d} t}
    = 
    k_\mathrm{p}n_\mathrm{m}(t) - \Gamma_\mathrm{p} n_\mathrm{p}(t).
$$
The parameters are listed below.
- $n_\mathrm{m}(t)$ is number of mRNA molecules at time $t$
- $n_\mathrm{p}(t)$ is number of protein molecules at time $t$
- $k_\mathrm{m}$ is rate of creation of mRNA 
- $k_\mathrm{p}$ is rate of creation of protein
- $\Gamma_\mathrm{m}$ is rate of degradation of mRNA
- $\Gamma_\mathrm{p}$ is rate of degradation of protein
Notice that creation term of protein depends on also number of mRNA molecules 
$n_\mathrm{m}$.

# Number of molecules vs concentration of molecules
So far we consider the number of molecules but concentration (=number/volume) is 
more convinient to treat both theoretically and experimentally.
The unit of concentration would be $\mathrm{M}$ (mole).
By introducing volume of the system $V$, 
concentration of mRNA would be
- $c_\mathrm{m}(t) = n_\mathrm{m}(t)/V$
concentration of protein would be
- $c_\mathrm{p}(t) = n_\mathrm{p}(t)/V$
by keeping the same notation for other rate parameters, the differential equations
become
$$
    \frac{\mathrm{d} c_\mathrm{m}(t)}{\mathrm{d} t}
    = 
    k_\mathrm{m} - \Gamma_\mathrm{m} c_\mathrm{m}(t).
$$
$$
    \frac{\mathrm{d} c_\mathrm{p}(t)}{\mathrm{d} t}
    = 
    k_\mathrm{p}c_\mathrm{m}(t) - \Gamma_\mathrm{p} c_\mathrm{p}(t).
$$

# Transcriptional regulation: Repression
#### Self-repressing gene
Consider self-repressing gene i.e. the number of protein control the rate of 
creation of mRNA. Differential equation of such system become
$$
    \frac{\mathrm{d} c_\mathrm{m}(t)}{\mathrm{d} t}
    = 
    \alpha_\mathrm{m}\left(c_\mathrm{p}(t)\right) 
    - 
    \Gamma_\mathrm{m} c_\mathrm{m}(t).
$$
$$
    \frac{\mathrm{d} c_\mathrm{p}(t)}{\mathrm{d} t}
    = 
    k_\mathrm{p}c_\mathrm{m}(t) - \Gamma_\mathrm{p} c_\mathrm{p}(t).
$$
To show the dependence of creation rate of mRNA $\alpha$, 
I wrote it as a function of $c_\mathrm{p}(t)$.
$\alpha$ should monotonically decreases as the number of protein increases.

How can we decide the form of $\alpha$?

#### Hill function
Consider the binding process of protein (transcription factor, TF) to promotor, 
the binding process should depend on concentration of TF, but unbinding process
is independent from concentration of TF.
Thus we can build equation for time evolution of probability.
$$
\begin{aligned}
    \frac{\mathrm{d} P_\mathrm{free}}{\mathrm{d} t}
    =& 
    - v_\mathrm{bind} c_\mathrm{p}(t) P_\mathrm{free}
    + k_\mathrm{unbind} P_\mathrm{occupied}
    \\=& 
    - v_\mathrm{bind} c_\mathrm{p}(t) P_\mathrm{free}
    + k_\mathrm{unbind} (1-P_\mathrm{free})
\end{aligned}
$$
The parameters are listed below.
- $P_\mathrm{free}$ is the probability that promotor region is free
- $P_\mathrm{occupied} = 1-P_\mathrm{free}$ is the probability that promotor region is occupied
- $v_\mathrm{bind}$ is binding rate of TF
- $k_\mathrm{unbind}$ is unbinding rate of TF
If we assume that binding/unbinding process is much faster that the transcription
process, we can describe the system with the fraction of time that promotor region
is occupied.
In steady state this fraction can be decided by using
$\frac{\mathrm{d} P_\mathrm{free}}{\mathrm{d} t}=0$
$$
    0
    = 
    - v_\mathrm{bind} c_\mathrm{p}(t) P_\mathrm{free}
    + k_\mathrm{unbind} (1-P_\mathrm{free})
$$
$$
\begin{aligned}
    P_\mathrm{free}
    =&
    \frac{k_\mathrm{unbind}}{k_\mathrm{unbind}+v_\mathrm{bind} c_\mathrm{p}(t)}
    \\=&
    \frac
    {1}
    {1 + \frac{v_\mathrm{bind}}{k_\mathrm{unbind}} c_\mathrm{p}(t)}
    \\=&
    \frac
    {1}
    {1 + \left( c_\mathrm{p}(t)/K \right) }
\end{aligned}
$$
Here $K=\frac{v_\mathrm{bind}}{k_\mathrm{unbind}}$ is dissociation constant.
$P_\mathrm{occupied} = 1-P_\mathrm{free}$ would be
$$
\begin{aligned}
    P_\mathrm{occupied} 
    =&
    1-P_\mathrm{free}
    \\=&
    \frac
    {c_\mathrm{p}(t)/K}
    {1 + \left( c_\mathrm{p}(t)/K \right) }
\end{aligned}
$$

#### Hill function for dimers
If TF forms the dimer the equation of probability that promotor region is free 
become
$$
    \frac{\mathrm{d} P_\mathrm{free}}{\mathrm{d} t}
    = 
    - v_\mathrm{bind} c_\mathrm{p}^2(t) P_\mathrm{free}
    + k_\mathrm{unbind} (1-P_\mathrm{free}).
$$
Notice that binding rate become proportional to square of concentration.
By doing the same above, we get
$$
\begin{aligned}
    P_\mathrm{free}
    =&
    \frac{k_\mathrm{unbind}}{k_\mathrm{unbind}+v_\mathrm{bind} c_\mathrm{p}^2(t)}
    \\=&
    \frac
    {1}
    {1 + \frac{v_\mathrm{bind}}{k_\mathrm{unbind}} c_\mathrm{p}^2(t)}
    \\=&
    \frac
    {1}
    {1 + \left( c_\mathrm{p}(t)/K_2 \right)^2 }
\end{aligned}
$$
Here we define $K_2^2=\frac{v_\mathrm{bind}}{k_\mathrm{unbind}}$.

#### Model for self repressing gene
By using Hill function we got above, we can replace 
$\alpha_\mathrm{m}\left(c_\mathrm{p}(t)\right) =
\frac{\alpha_m}{1 + \left( c_\mathrm{p}(t)/K \right) }
$
we can write differential equations for 
self-repressing gene.
$$
    \frac{\mathrm{d} c_\mathrm{m}(t)}{\mathrm{d} t}
    = 
    \frac{\alpha_m}{1 + \left( c_\mathrm{p}(t)/K \right) }
    - 
    \Gamma_\mathrm{m} c_\mathrm{m}(t).
$$
$$
    \frac{\mathrm{d} c_\mathrm{p}(t)}{\mathrm{d} t}
    = 
    k_\mathrm{p}c_\mathrm{m}(t) - \Gamma_\mathrm{p} c_\mathrm{p}(t).
$$
Or more generally, the mRNA regulated by concentration of repressor 
$c_\mathrm{TF}(T)$ would be
$$
    \frac{\mathrm{d} c_\mathrm{m}(t)}{\mathrm{d} t}
    = 
    \frac{\alpha_m}{1 + \left( c_\mathrm{TF}(t)/K \right) }
    - 
    \Gamma_\mathrm{m} c_\mathrm{m}(t).
$$

# Transcriptional regulation: Activation
In similar manner, we can get Hill function for activation i.e. by using
$P_\mathrm{occupied}$ we can describe the creation rate.
Notice that we assume without activator binding, 
the creation rate of mRNA is lower than that of with activator binding.
$$
    \frac{\mathrm{d} c_\mathrm{m}(t)}{\mathrm{d} t}
    = 
    \frac{\alpha_m \left( c_\mathrm{TF}(t)/K \right)}
    {1 + \left( c_\mathrm{TF}(t)/K \right) }
    - 
    \Gamma_\mathrm{m} c_\mathrm{m}(t).
$$

# Transcriptional regulation: sRNA
sRNA binds to mRNA and make complex. 
The complex is easier to be broken down.
If we consider the interaction between mRNA and sRNA, 
the differential equation become
$$
    \frac{\mathrm{d} c_\mathrm{s}(t)}{\mathrm{d} t}
    = 
    \alpha_\mathrm{s} 
    - \Gamma_\mathrm{s} c_\mathrm{s}(t)
    - \delta c_\mathrm{m}(t) c_\mathrm{s}(t)
$$
$$
    \frac{\mathrm{d} c_\mathrm{s}(t)}{\mathrm{d} t}
    = 
    \alpha_\mathrm{m} 
    - \Gamma_\mathrm{m} c_\mathrm{m}(t)
    - \delta c_\mathrm{m}(t) c_\mathrm{s}(t)
$$
Here $c_\mathrm{s}(t)$ is concentration of sRNA and 
$c_\mathrm{m}(t)$ is concentration of mRNA.

Important thing is third term of right-hand side.
