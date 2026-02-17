# Networks

## Graph theory basics

A **network** (graph) consists of nodes (vertices) connected by edges (links). Networks provide a natural language for describing complex systems where relationships between components matter as much as the components themselves.

Examples of real-world networks:

- **Social networks**: individuals connected by friendships or interactions.
- **World Wide Web**: pages connected by hyperlinks.
- **Biological networks**: proteins connected by interactions, genes by regulatory links.
- **Infrastructure**: power grids, transportation systems, the internet.

## Network metrics

Key quantities that characterize a network:

- **Degree** $k_i$: the number of edges connected to node $i$. The **degree distribution** $P(k)$ describes the probability that a randomly chosen node has degree $k$.
- **Clustering coefficient**: measures the fraction of a node's neighbors that are also connected to each other. High clustering indicates **cliquishness**.
- **Betweenness centrality**: the fraction of shortest paths between all pairs of nodes that pass through a given node. High centrality indicates a hub or bottleneck.
- **Connectedness**: a measure of resilience. How many nodes must be removed to disconnect the network?
- **Modularity**: the strength of division into communities. Zero modularity means no meaningful partition.

The **amplification factor** of a node in an undirected graph is $\mathcal{A} = k - 1$. The network-average amplification factor involves the second moment of the degree distribution:

$$
\langle \mathcal{A} \rangle = \frac{\langle k^2 \rangle}{\langle k \rangle} - 1.
$$

This quantity diverges for scale-free networks with exponent $\gamma \leq 3$, with profound consequences for epidemic spreading and network robustness.

## Small-world networks

**Small-world networks** (Watts and Strogatz, 1998) combine two properties:

- **High clustering**: like a regular lattice, neighbors of a node tend to be connected.
- **Short path lengths**: like a random graph, any two nodes are connected by a short chain.

The Watts-Strogatz model starts from a regular ring lattice and randomly rewires each edge with probability $p$. Even a small rewiring probability ($p \sim 0.01$) dramatically reduces the average path length while preserving high clustering.

The **six degrees of separation** phenomenon (Milgram, 1967) is a manifestation of the small-world property in social networks.

## Scale-free networks

**Scale-free networks** have a degree distribution that follows a power law:

$$
P(k) \sim k^{-\gamma},
$$

typically with $2 < \gamma < 3$. This means a few nodes (hubs) have very many connections, while most nodes have few.

Examples of scale-free networks:

- The World Wide Web ($\gamma \approx 2.1$ for in-degree).
- Protein interaction networks.
- Citation networks.
- Software dependency graphs.

[[simulation scale-free-network]]

## Preferential attachment

The **Barabasi-Albert model** (1999) generates scale-free networks through **preferential attachment**: new nodes connect preferentially to existing nodes that already have many connections ("the rich get richer").

Algorithm:

1. Start with $m_0$ connected nodes.
2. Add new nodes one at a time, each connecting to $m$ existing nodes.
3. The probability of connecting to node $i$ is proportional to its degree: $\Pi(k_i) = k_i / \sum_j k_j$.

This produces a power-law degree distribution with exponent $\gamma = 3$.

## Dynamics on networks

Networks are not just static structures; processes unfold on them:

- **Epidemic spreading** (SIR/SIS models): on scale-free networks, the epidemic threshold vanishes in the thermodynamic limit, meaning even weakly infectious diseases can spread.
- **Diffusion and synchronization**: random walks on networks explore hubs quickly; coupled oscillators synchronize more easily on networks with high connectivity.
- **Information cascades**: content spreads through social networks via threshold mechanisms.

## Robustness and attacks

Networks respond differently to random failures versus targeted attacks:

- **Random failure**: removing random nodes has little effect on scale-free networks because most nodes have low degree.
- **Targeted attack**: removing hubs rapidly fragments the network, dramatically increasing path lengths and disconnecting components.

This **robustness-yet-fragility** property of scale-free networks has implications for infrastructure protection, disease control, and cybersecurity.
