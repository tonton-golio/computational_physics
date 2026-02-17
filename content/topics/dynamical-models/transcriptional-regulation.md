# Transcriptional Regulation and the Hill Function

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
