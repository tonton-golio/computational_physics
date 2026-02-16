# Continuum Approximation



## __Introduction__
The physics of continuum mechanics, requires that a mass of particles can be 
modelled as continuous matter where the particles are infinitesimal. 
This approximation depends on the degree of precision required, as Benny puts it;

"Whether a given number of molecules is large enough to warrant the use of
a smooth continuum description of matter depends on the desired precision. 
Since matter is never continuous at sufficiently high precision, continuum 
phyiscs is always an approximation. But as long as the fluctuations in 
physical quantities caused by the discreteness of matter are smaller than 
the desired precision, matter may be taken to be continuous. To observe the 
continuity, one must so to speak avoid looking too sharply at material bodies. 
Fontenelle stated in a similar context that; 

"Science originates from curiosity and bad eyesight"."

From Physics of Continuous Matter 2nd edition by Benny Lautrup



## __Density Fluctuations__
The density of a pure gas is given as;
$$
\rho = \frac{N m}{V}
$$
Where $N$ is the number of molecules, $V$ is the volume and $m$ is the mass of the molecules.
From general statistics, it can be shown that the fluctuations in $N$ follows the RMS of the number of molecules and since the density is linear depended on $N$ this gives;
$$
\frac{\Delta \rho}{\rho} = \frac{\Delta N}{N} = \frac{1}{\sqrt{N}}
$$
So if we require a relative precision of $\epsilon = 10^{-3} > \frac{\Delta \rho}{\rho} $ in the density fluctuations there must be $N > \epsilon^{-2}$ molecules. They occupy a volume of $\epsilon^{-2}L_{molecule}^3$, where $L_{molecule}$ is the molecular seperation length. 
At the scale of $L_{molecule}$ the continuum approximation completely breaks down, and to ensure correct approximation within a certain precision the minimum cell size that we consider is given as;
$$
L_{micro} = \epsilon^{-2}L_{molecule}^3=\epsilon^{-2} \left( \frac{M_{molecule}}{\rho N_A} \right)
$$
$$
L_{molecule} =\left( \frac{V}{N} \right)^{1/3} = \left( \frac{M_{molecule}}{\rho N_A} \right)^{1/3} 
$$
Where $L_{micro}^*$ is the sidelength of the cubic cell that satisfies the precision condition, $M_{molecule}$ is the molar mass of the substance.

## __Macroscopic Smoothness__
Another criteria for the continuum approximation is the Macroscopic Smoothness. 
We require the relative change in density between cells to be less than the 
precision $\epsilon$ along any direction.
$$
\left( \frac{\partial \rho}{\partial x} \right) < \frac{\rho}{L_{macro}} =  \epsilon \frac{\rho^2 N_A}{M_{molecule}}\hspace{10mm} \text{where} \hspace{10mm} L_{macro} = \epsilon^{-1} L_{micro}
$$
  
if the above is fulfilled, the change in density can be assumed to vary smooth, and the continuum approximation holds. However, the thickness of interfaces between macroscopic bodies are typically on the order of $L_{molecule}$ and not $L_{macro}$, we instead represent these as *surface discontinuities*.


## __Velocity Fluctuations__
For gases (and in general fluids), we may devide movement into two categories, *bulk motion* of a volume like the winds or *molecular motion* like thermal energy. In continuum mechanics we are often interested in the bulk motion of the continuum and in order to assume the gas as a continuum, we require that the molecular speeds do not dominate the fluctuations.
The average root-mean-square speed of particles in a gas will be
$$
v_{mol}=\sqrt{\frac{3 R_{mol}T}{M_{mol}}}
$$
where $R_{mol}=8.31447 JK^{-1}mol^{-1}$ is the universal molar gas constant and T the absolute temperature. For air at room temperature, this is $\approx 500ms^{-1}$

In order to guarantee a relative precision of $\epsilon$ is the velocity fluctuations, we require $\Delta v/v \lessapprox \epsilon$, implying that the length of the gas volume must be larger than
$$
L_{micro}^*=\left( \frac{v_{mol}}{v} \right)^{2/3}L_{micro}
$$
So if we require $\epsilon=10^{-3}$ precision in velocity, the minimum sidelength of the volume becomes $L_{micro}^*=100L_{micro}$ for air at room temperature.
## __Mean-Free-Path__
To be written.