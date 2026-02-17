# Differential Equations in a Nutshell

[[simulation hill-function]]

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

[[simulation steady-state-regulation]]

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
