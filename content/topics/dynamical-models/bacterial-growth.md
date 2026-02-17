# Bacterial Growth Physiology

## Phases of Bacterial Growth

*E. coli* has been intensively used for research on bacterial growth. Monod (1949) showed that there are several phases in bacterial growth, and we focus on the **exponential growth phase**. In the exponential phase, the bacteria reach a quasi-steady state of growth.

## Definition of Steady-State Growth

1. Intrinsic parameters of the cell remain constant.
2. Extrinsic parameters increase exponentially with precisely the same doubling time.
3. Growth conditions remain constant.

## Bacterial Growth Law by Jacques Monod (1949)

He found growth rate $\lambda$ depends on concentration of limiting nutrient in a Michaelis-Menten manner:

$$
\lambda = \lambda_\mathrm{max} \frac{S}{K_S + S}
$$

[[simulation michaelis-menten]]

## Frederick C. Neidhardt (1999)

> For me, encountering the bacterial growth curve was a transforming experience.

## Bacterial Biomass Is Mainly Protein

We now know bacterial growth is governed by a complex metabolic network. However, 55% of the total dry weight of bacteria is protein. We can approximate bacterial growth as mass of protein. The differential equation for this is:

$$
\frac{\mathrm{d}}{\mathrm{d}t} M = \lambda \cdot M
$$

Here, $M$ is total protein mass and $\lambda$ is the growth rate. If protein synthesis is rate limiting for growth:

$$
\frac{\mathrm{d}}{\mathrm{d}t} M = k \cdot N_R^A
$$

Here $k$ is the translation rate per ribosome and $N_R^A$ is the total number of actively translating ribosomes. Thus we approximate protein production as depending only on the number of ribosomes.

We can roughly divide protein into three fractions:
- Metabolism/transport: $\phi_\mathrm{P} = M_\mathrm{P}/M$
- "Extended" ribosomes (ribosomal protein): $\phi_\mathrm{R} = M_\mathrm{R}/M$
- Other tasks: $\phi_\mathrm{Q} = M_\mathrm{Q}/M$

Here $\phi_\mathrm{x} = M_\mathrm{x}/M$ is the fraction of protein molecules dedicated to a specific task. We immediately know that $\phi_\mathrm{P} + \phi_\mathrm{R} + \phi_\mathrm{Q} = 1$.

## Efficient Resource Allocation

Nutrient influx by $\phi_\mathrm{P}$ should match nutrient usage by $\phi_\mathrm{R}$. But how does the bacterium know when to adjust the balance?

The molecule called **ppGpp** (guanosine pentaphosphate) is the molecular signal to stop making ribosomes. In the default state, bacteria make a lot of ribosomes because synthesis of the "extended ribosome" is regulated mainly by the promoters for the ribosomal RNA genes.

## Measuring Numbers of Growth

As an equation describing bacterial growth, we have:

$$
\frac{\mathrm{d}}{\mathrm{d}t} M = k \cdot N_R^A
$$
