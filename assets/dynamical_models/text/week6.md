
# title
Signal transduction

# Week 5 description
- Jan 4: How cell respond to external stimulation?

# Beginning of signal transduction
Signal is some sort of molecule (e.g. food, quroum, hormone, ion and gas or 
physical stimulation (e.g. pressure, temperature and light)
Signal somehow binds to specific receptor and cascade of event happen.
This is the biginning of signal transduction.

# Signla transduction involves many steps
Thanks to intermediate steps, cell can perform
- signal amplification
- signal spreading between cells
- signal intergration
- noise filtering
- signal memory
- adaptation
These are enabled by protein-protein interaction.
Very often, these transduction steps are series of phosphorylation events.

# A well studied example of sinal transduction
Bacterial chemotaxis is well studied.

Bacterial chemotaxis is that bacteria detect gradients of attractant (like food)
or move disired derecton.

They can detect the smalled change of gradients such as one moelecule per cell
volume per micron.

They can detect the gradients in high concentraion background (like five orders of
magnitude)

They perform this in Brownian motion. They seim straight for 10 sec. and orient 
randomly 90 degree. However how they move towards food? They use biased random 
walk.

They have flagella to move. It is a molecular motor and depending on which way
it rotates, thry can run (counterclockwise) or tumbles (clockwise).
When thry run, flaggelas converge to single fiber and bacteria move forward.
When they tumble, flagellas does not converge.

This change of rotational way is governed by CheY and CheA.
When CheY is phosphorylated, CheY-P binds the motor and makes it trun clockwise.
If signal (attractant) binds to receptor, it lowers the kinase activity of CheA 
which leads to less tumbling.

# How tumbling frequency depends on ligand concentraion
If tmubling frequency depends on absolute concentration of attractant, it will
saturate quickly. 
If tmubling frequency does not depend on concentration of attractant, bacteria
cannot detect the gradient.
How do they detect gradient?

# Chemotaxis requires adaptation
Tumbling grequency change when a change detected, but if there is no change for 
a while, it goes back to default tumbling frequency.
With this adaptation, cell can be sensitive to small change of concentration as 
well as can react to wide range of concentration.

# Modeling chemotaxis
Assumption 
- Ligand binding immediately changes the fraction of active methylated receptor
- Michaelis-Menten
Equation for methylation/demethylation:
$$
    \frac{\mathrm{d} [E_m]}{\mathrm{d} t}
    =
    k^R [\mathrm{CheR}] - \frac{k^B B^{*}(l) E_m}{K_M^B+E_m}
$$

# Signal transduction in space
Fruit fly shows the pattern dering its development. This is done by signal 
transduction.

# Cell-to-cell communication
Lateral inhibition, which is cell inhibits the gene expression of nearest 
neighbor's expression.
This is called Notch-Delta model.
