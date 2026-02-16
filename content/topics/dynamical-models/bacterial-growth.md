# Bacterial Growth Physiology

# Week 4 description
- Dec 12: How bacteria grow?

# Phase of bacterial growth
*E. coli* has been intensively used for reserach of bacterial growth.
Monod(1949) shows that threre are several phase in bacterial growth, and we focus
on exponential growth phase.
In the exponential phase, the bacterial reach a quasi-steady state of growth.

# Difinition of steady-state growth
1) Intrinsic parameters of the cell remain constant.
2) Extrinsic parameters increase exponentially with precisely the same doubling 
time.
3) Growth condition remain constant.

# Bacterial growth las by Jacques Monod (1949)
He found growth rate $\lambda$ depends on concentration of limiting nutrient in 
Michaelis-Menten manner.
$$
    \lambda 
    =
    \lambda_\mathrm{max} 
    \frac{S}{K_S + S}
$$

# Frederick C. Neidhardt (1999)
> For me, encountering the bacterial growth curve was a transforming experience.

# Bacterial biomass is mainly protein
We now know bacterial growth is goverd by complex metabolic network.
However, 55% of total dry wet of bacteria is protein!
We can approximate bacterial growth as mass of protein. 
Differential equation for this.
$$
    \frac{\mathrm{d}}{\mathrm{d}t} M
    =
    \lambda \cdot M
$$
Here, $M$ is total protein mass, $\lambda$ is growth rate.
If the protein systhesis is rate limiting for growth,
$$
    \frac{\mathrm{d}}{\mathrm{d}t} M
    =
    k \cdot N_R^A
$$
Here $k$ is translation rate per ribosome and $N_R^A$ is total number of actively 
translating ribosomes.
Thus we approximate protein production only depends on number of ribosome.
We can roughly divide protein into three fractions.
- Metabolism/transport $\phi_\mathrm{P} = \frac{M_\mathrm{P}}{M}$
- "Extended" ribosomes (protein for ribosomes?) $\phi_\mathrm{R} = \frac{M_\mathrm{R}}{M}$
- Other tasks $\phi_\mathrm{Q} = \frac{M_\mathrm{Q}}{M}$
Here $\phi_\mathrm{x} = \frac{M_\mathrm{x}}{M}$ is fraction of protein molecules
of specific task.
We immediately know that $\phi_\mathrm{P} + \phi_\mathrm{R} + \phi_\mathrm{Q} = 1$.

# Efficient resource allocation
Nutrient influx by $\phi_\mathrm{P}$ should match nutrient usage by 
$\phi_\mathrm{R}$.
But how does the bacterium know when to adjust the handle?
The molecule called ppGpp(Guanosine pentaphosphate) is the molecular signals to 
stop making ribomes. In default state bacteria make a lot of robosome.
Bacause synthesis of the "extended robosome" is regulated mainly by the promotors
for the ribosomal RNA genes.

# Measuring numbers of growth
As a equation describe the bacterial growth, we have
$$
    \frac{\mathrm{d}}{\mathrm{d}t} M
    =
    k \cdot N_R^A
$$
