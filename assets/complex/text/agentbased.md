
### Game of life
In this type of model, we consider autonomous agents following a set of rules. The example which immediately springs to mind, is Conway's *Game of Life*. The rules are simple, we consider each site to have 8 nieghbors. 

* If a site is alive it will stay alive iff either 2 or 3 of its neighbors are alive. 
* If a site is dead, it will spawn iff 3 of its neighbors are alive.

### Game of life 2
In the figure above, I have initialized a *glider*. a glider is a self-replicating configurations which glides in some direction. For the one shown above, it is self-similar every 4 generations. Other configurations show self-similarity on much greater time-scales. 

### simulation with discrete but random changes

Unlike cellular automata, we may have agent based models which act in a non-deterministic (random) fashion.


### Gillespie algorithm, 
*Traditional continuous and deterministic biochemical rate equations do not accurately predict cellular reactions since they rely on bulk reactions that require the interactions of millions of molecules. They are typically modeled as a set of coupled ordinary differential equations. In contrast, the Gillespie algorithm allows a discrete and stochastic simulation of a system with few reactants because every reaction is explicitly simulated. A trajectory corresponding to a single Gillespie simulation represents an exact sample from the probability mass function that is the solution of the master equation.*

### Example of agent based simulation
* spread in epidemics
* post spread on social networks

### Advantages of agent based models
If actors in a model have different attirbutes (i.e. a heterogenous population) it makes sense to use an agent-based approach. This allows us to have super spreaders and things like that depending on what we are simulating.

In the opposite case, i.e., that with a homogenous population, we may derive results statistically.
