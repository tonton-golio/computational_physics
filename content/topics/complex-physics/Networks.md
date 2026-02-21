# Networks

New web pages do not link randomly. They link to pages that are already popular — Google, Wikipedia, YouTube. A new paper does not cite random old papers. It cites the famous ones, the ones everyone else is already citing. A new actor does not get cast with random other actors. They get cast alongside established stars.

This is the **"rich get richer"** effect, and it produces power laws everywhere. The distribution of web links, citation counts, social connections, airport traffic — they all follow the same pattern: a few hubs with enormous numbers of connections, and a vast majority with very few.

It is the same kind of power law we saw in the sandpile and in percolation clusters. Nature loves power laws, and networks are one of her favorite places to put them.

## Graph theory basics

A **network** (graph) consists of nodes (vertices) connected by edges (links). Networks provide a natural language for describing complex systems where relationships between components matter as much as the components themselves.

Examples of real-world networks:

* **Social networks**: individuals connected by friendships or interactions.
* **World Wide Web**: pages connected by hyperlinks.
* **Biological networks**: proteins connected by interactions, genes by regulatory links.
* **Infrastructure**: power grids, transportation systems, the internet.

## Network metrics

Key quantities that characterize a network:

* **Degree** $k_i$: the number of edges connected to node $i$. The **degree distribution** $P(k)$ describes the probability that a randomly chosen node has degree $k$. This single function tells you more about a network's structure than almost anything else.
* **Clustering coefficient**: measures the fraction of a node's neighbors that are also connected to each other. High clustering means "my friends know each other" — cliquishness.
* **Betweenness centrality**: the fraction of shortest paths between all pairs of nodes that pass through a given node. High centrality means the node is a bottleneck — remove it, and many paths break.
* **Connectedness**: how many nodes must be removed to disconnect the network?
* **Modularity**: the strength of division into communities. Zero modularity means no meaningful partition.

The **amplification factor** involves the second moment of the degree distribution:

$$
\langle \mathcal{A} \rangle = \frac{\langle k^2 \rangle}{\langle k \rangle} - 1.
$$

This quantity is crucial: it diverges for scale-free networks with exponent $\gamma \leq 3$, with profound consequences for epidemic spreading and network robustness.

## Small-world networks

Here is a surprising fact about social networks: any two people on Earth are connected by roughly six intermediaries. That is the **"six degrees of separation"** phenomenon.

**Small-world networks** explain how this is possible by combining two properties:

* **High clustering**: like a regular lattice, neighbors of a node tend to be connected (your friends know each other).
* **Short path lengths**: like a random graph, any two nodes are connected by a short chain (six degrees).

The Watts-Strogatz model starts from a regular ring lattice and randomly rewires each edge with probability $p$. The remarkable finding: even a tiny rewiring probability ($p \sim 0.01$) dramatically reduces the average path length while preserving high clustering. A few long-range shortcuts are enough to make the whole network small-world.

## Scale-free networks

**Scale-free networks** have a degree distribution that follows a power law:

$$
P(k) \sim k^{-\gamma},
$$

typically with $2 < \gamma < 3$. This means a few nodes (hubs) have enormously many connections, while most nodes have few. There is no "typical" number of connections — the distribution is scale-free.

Examples of scale-free networks:

* The World Wide Web ($\gamma \approx 2.1$ for in-degree).
* Protein interaction networks.
* Citation networks.
* Software dependency graphs.

[[simulation scale-free-network]]

Watch the network grow in this simulation. New nodes attach preferentially to well-connected hubs, and the power-law degree distribution emerges naturally. The resulting network looks nothing like a regular grid or a random graph — it has a few giant hubs surrounded by many weakly connected nodes.

## Preferential attachment

The **Barabasi-Albert model** explains how scale-free networks arise through **preferential attachment**: new nodes connect preferentially to existing nodes that already have many connections.

Algorithm:

1. Start with $m_0$ connected nodes.
2. Add new nodes one at a time, each connecting to $m$ existing nodes.
3. The probability of connecting to node $i$ is proportional to its degree: $\Pi(k_i) = k_i / \sum_j k_j$.

This produces a power-law degree distribution with exponent $\gamma = 3$. The rich-get-richer mechanism is all you need — the power law emerges as a mathematical consequence, just as the power law in the sandpile emerges from the self-organized dynamics.

## Dynamics on networks

Networks are not just static structures; processes unfold on them:

* **Epidemic spreading** (SIR/SIS models): on scale-free networks, the epidemic threshold vanishes in the thermodynamic limit. Even weakly infectious diseases can spread through the hubs. This is why super-spreaders matter so much.
* **Diffusion and synchronization**: random walks on networks explore hubs quickly; coupled oscillators synchronize more easily on networks with high connectivity.
* **Information cascades**: content spreads through social networks via threshold mechanisms — one viral post can reach millions through the hub structure.

## Robustness and attacks

Networks respond very differently to random failures versus targeted attacks:

* **Random failure**: removing random nodes has little effect on scale-free networks because most nodes have low degree — you are unlikely to hit a hub by chance.
* **Targeted attack**: removing hubs rapidly fragments the network, dramatically increasing path lengths and disconnecting components.

This **robustness-yet-fragility** property has real consequences: the internet is resilient to random router failures but vulnerable to targeted attacks on major hubs. The same principle applies to disease control (vaccinate the super-spreaders) and infrastructure protection.

## Big Ideas

* Preferential attachment — new nodes connect to well-connected nodes — is all you need to generate a power-law degree distribution; the rich get richer, and the math takes care of the rest.
* The amplification factor $\langle k^2 \rangle / \langle k \rangle - 1$ diverges for scale-free networks with $\gamma \leq 3$, which means epidemic thresholds vanish and even the weakest disease can spread through hubs.
* Small-world networks are the best of both worlds: high local clustering (your friends know each other) combined with short global paths (six degrees of separation), achieved by adding just a few long-range shortcuts.
* Network robustness is deeply asymmetric: scale-free networks are nearly immune to random node failure but catastrophically fragile to targeted hub removal.

## What Comes Next

Networks describe the *structure* of interactions, but not what happens on them. [Agent-Based Models](agentbased) puts the dynamics back in: place autonomous agents on a network (or a spatial grid) and give each one simple local rules — align with your neighbors, run from predators, spread a disease to your contacts. No agent has a global view, yet collective patterns emerge spontaneously: flocking, traffic jams, epidemic waves. The connection to everything we have studied — the Ising model, percolation, SOC — will become clear when we see that emergence is a single theme running through all of complex physics.

## Check Your Understanding

1. A Poisson random graph (Erdős-Rényi model) has an exponential degree distribution: $P(k) \sim e^{-\langle k \rangle} \langle k \rangle^k / k!$. A scale-free network has $P(k) \sim k^{-\gamma}$. Why does the scale-free distribution lead to a finite epidemic threshold vanishing in the limit $N \to \infty$, while the Poisson distribution gives a nonzero threshold?
2. The clustering coefficient of a node measures whether its neighbors are connected to each other. The Watts-Strogatz model has high clustering for small rewiring probability $p$. Explain intuitively why rewiring destroys clustering even when the average path length drops dramatically.
3. Removing hub nodes rapidly fragments a scale-free network. In a network where the hub with degree $k_{\max} \sim \sqrt{N}$ (more precisely, the natural cutoff scales as $k_{\max} \sim N^{1/(\gamma-1)}$, which gives $\sqrt{N}$ for $\gamma = 3$, with logarithmic corrections) is removed, roughly what fraction of the remaining nodes lose their connection to the giant component?

## Challenge

Model the spread of an epidemic on a scale-free network with degree distribution $P(k) \sim k^{-3}$ using the SIR model (susceptible-infected-recovered). In the mean-field approximation on a heterogeneous network, the epidemic threshold is $\lambda_c = \langle k \rangle / (\langle k^2 \rangle - \langle k \rangle)$. Show analytically that $\lambda_c \to 0$ as $N \to \infty$ for $\gamma = 3$ (since $\langle k^2 \rangle$ diverges logarithmically). Then argue: what does this mean for vaccination strategy? If you can only vaccinate a fraction $f$ of the population, is it better to vaccinate randomly or to target hubs? Estimate how much $f$ you need for each strategy to halt the epidemic.
