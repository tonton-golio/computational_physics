# Differential Equations for Transcription and Translation

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
