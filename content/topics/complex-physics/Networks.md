# Networks

New web pages don't link randomly. They link to pages that are already popular -- Google, Wikipedia, YouTube. A new paper doesn't cite random old papers. It cites the famous ones, the ones everyone else is already citing.

This is the **"rich get richer"** effect, and it produces power laws everywhere. The distribution of web links, citation counts, social connections -- they all follow the same pattern: a few hubs with enormous numbers of connections, and a vast majority with very few.

Same kind of power law we saw in the sandpile and percolation clusters. Nature loves power laws, and networks are one of her favorite places to put them.

## Big Ideas

* Preferential attachment -- new nodes connect to well-connected nodes -- is all you need to generate a power-law degree distribution. The rich get richer, and the math takes care of the rest.
* On a scale-free network the epidemic threshold drops to zero. A few super-spreaders are enough to infect the whole world. Suddenly vaccination strategy changes completely -- target the hubs!
* Small-world networks are the best of both worlds: high local clustering plus short global paths, achieved by just a few long-range shortcuts.
* Scale-free networks are nearly immune to random failure but catastrophically fragile to targeted hub removal.

## Graph Theory Basics

A **network** (graph) is nodes connected by edges. Networks describe systems where relationships matter as much as the components.

Examples: social networks (friendships), the World Wide Web (hyperlinks), biological networks (protein interactions, gene regulation), infrastructure (power grids, the internet).

## Network Metrics

* **Degree** $k_i$: edges connected to node $i$. The **degree distribution** $P(k)$ tells you more about a network's structure than almost anything else.
* **Clustering coefficient**: fraction of a node's neighbors that know each other. High clustering means cliquishness.
* **Betweenness centrality**: fraction of shortest paths through a node. High centrality means bottleneck -- remove it, many paths break.
* **Connectedness**: how many nodes must go before the network disconnects?

The **amplification factor** involves the second moment of the degree distribution:

$$
\langle \mathcal{A} \rangle = \frac{\langle k^2 \rangle}{\langle k \rangle} - 1.
$$

This diverges for scale-free networks with $\gamma \leq 3$, with profound consequences for epidemics and robustness.

## Small-World Networks

Any two people on Earth are connected by roughly six intermediaries. The **Watts-Strogatz model** explains how: start from a regular ring lattice, randomly rewire each edge with probability $p$. Even a tiny $p \sim 0.01$ dramatically reduces average path length while preserving high clustering. A few long-range shortcuts make the whole network small-world.

## Scale-Free Networks

**Scale-free networks** have a power-law degree distribution:

$$
P(k) \sim k^{-\gamma},
$$

typically with $2 < \gamma < 3$. A few hubs have enormously many connections; most nodes have few. No "typical" number of connections.

[[simulation scale-free-network]]

Watch the network grow. New nodes attach preferentially to well-connected hubs, and the power-law distribution emerges naturally.

## Preferential Attachment

The **Barabasi-Albert model** explains how scale-free networks arise. The algorithm:

1. Start with $m_0$ connected nodes.
2. Add new nodes one at a time, each connecting to $m$ existing nodes.
3. Probability of connecting to node $i$: $\Pi(k_i) = k_i / \sum_j k_j$.

This produces $\gamma = 3$. The rich-get-richer mechanism is all you need.

## Dynamics on Networks

* **Epidemic spreading** (SIR/SIS): on a scale-free network the epidemic threshold vanishes in the thermodynamic limit. Even weakly infectious diseases can spread through hubs. This is why super-spreaders matter -- and why vaccination strategy must change completely. Target the hubs!
* **Diffusion and synchronization**: random walks explore hubs quickly; coupled oscillators synchronize more easily on well-connected networks.
* **Information cascades**: one viral post can reach millions through hub structure.

## Robustness and Attacks

Scale-free networks respond very differently to random failures versus targeted attacks:

* **Random failure**: little effect, because most nodes have low degree -- you're unlikely to hit a hub by chance.
* **Targeted attack**: removing hubs rapidly fragments the network.

This **robustness-yet-fragility** has real consequences: the internet is resilient to random router failures but vulnerable to targeted attacks on major hubs. Same principle for disease control (vaccinate super-spreaders) and infrastructure protection.

## What Comes Next

Networks describe the *structure* of interactions, but not what happens on them. [Agent-Based Models](agentbased) puts dynamics back in: place agents on a network, give each one simple local rules, and watch collective patterns emerge -- flocking, traffic jams, epidemic waves. The connection to the Ising model, percolation, and SOC becomes clear when you see that emergence is a single theme running through all of complex physics.

## Check Your Understanding

1. Why does the scale-free distribution lead to a vanishing epidemic threshold, while the Poisson distribution gives a nonzero one?
2. Explain intuitively why rewiring in the Watts-Strogatz model destroys clustering even as average path length drops dramatically.

## Challenge

Model an epidemic on a scale-free network with $P(k) \sim k^{-3}$ using the SIR model. In the mean-field approximation, the epidemic threshold is $\lambda_c = \langle k \rangle / (\langle k^2 \rangle - \langle k \rangle)$. Show that $\lambda_c \to 0$ as $N \to \infty$ for $\gamma = 3$. Then argue: if you can only vaccinate a fraction $f$ of the population, is it better to vaccinate randomly or target hubs? Estimate $f$ needed for each strategy.
