# Signal Transduction

## Introduction

A signal is some sort of molecule (e.g. food, quorum-sensing molecules, hormones, ions, or gases) or physical stimulation (e.g. pressure, temperature, or light). The signal binds to a specific receptor, triggering a cascade of events inside the cell. This is the beginning of **signal transduction**.

## Signal Transduction Involves Many Steps

Thanks to intermediate steps, cells can perform:
- Signal amplification
- Signal spreading between cells
- Signal integration
- Noise filtering
- Signal memory
- Adaptation

These capabilities are enabled by protein-protein interactions. Very often, the transduction steps are a series of **phosphorylation** events.

## Bacterial Chemotaxis: A Well-Studied Example

Bacterial chemotaxis is well studied. Bacteria detect gradients of attractant (like food) and move in the desired direction.

They can detect the smallest change of gradients, such as one molecule per cell volume per micron. They can also detect gradients against a high concentration background spanning five orders of magnitude.

Bacteria perform this while undergoing Brownian motion. They swim straight for about 1 second, then reorient randomly by about 90 degrees. How do they move towards food? They use a **biased random walk**.

They have flagella to move. The flagellar motor is a molecular motor, and depending on which way it rotates, bacteria can **run** (counterclockwise) or **tumble** (clockwise). When they run, flagella converge into a single bundle and bacteria move forward. When they tumble, the flagella do not converge.

This change of rotational direction is governed by **CheY** and **CheA**. When CheY is phosphorylated, CheY-P binds the motor and makes it turn clockwise. If a signal (attractant) binds to the receptor, it lowers the kinase activity of CheA, which leads to less tumbling.

## How Tumbling Frequency Depends on Ligand Concentration

If tumbling frequency depends on the absolute concentration of attractant, it will saturate quickly. If tumbling frequency does not depend on concentration at all, bacteria cannot detect the gradient. How do they detect the gradient?

## Chemotaxis Requires Adaptation

Tumbling frequency changes when a change is detected, but if there is no change for a while, it goes back to the default tumbling frequency. With this **adaptation**, cells can be sensitive to small changes in concentration as well as respond to a wide range of concentrations.

## Modeling Chemotaxis

Assumptions:
- Ligand binding immediately changes the fraction of active methylated receptor
- Michaelis-Menten kinetics

Equation for methylation/demethylation:

$$
\frac{\mathrm{d} [E_m]}{\mathrm{d} t} = k^R [\mathrm{CheR}] - \frac{k^B B^{*}(l) E_m}{K_M^B + E_m}
$$

## Signal Transduction in Space

The fruit fly shows spatial patterning during its development. This is accomplished by signal transduction, where morphogen gradients establish positional information across the embryo.

## Cell-to-Cell Communication

**Lateral inhibition** is a process where a cell inhibits the gene expression of its nearest neighbors. This is described by the **Notch-Delta model**, where Notch receptors on one cell are activated by Delta ligands on neighboring cells, leading to differentiation patterns.
