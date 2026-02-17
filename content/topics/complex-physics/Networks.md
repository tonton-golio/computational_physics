# Networks

### Intro

A network is a set of nodes connected by edges. **Networks are everywhere**, examples include:

## Scale-Free Network Generation

Many real-world networks follow a scale-free degree distribution. The Barabási-Albert model generates such networks through preferential attachment.

[[simulation scale-free-network]]

### Social-networks
Nodes represent bots and individuals, and edges mark connection. Edges may carry weight dependent on the frequency of interaction. Some persons are typically central in their network, and these networks tend to have high cliqueishness. These networks are scale-free, some indivs. have all the friends.

### Paper-authors
People tend to cite central people in their publication network. Thus these networks are also usually scale-free.

### Metrics and features of networks
* *cliquishness* -- how clustered are sub-graphs. A clique is a complete subgraph. A complete graph is one which is densely connected.
* *degree* -- How connected is a node.
* *centrality* -- ranking of node, how many of the possible paths through the network pass through this node.
* *connectedness* -- Resilience. How many nodes must be removed for the network to be dis-jointed.
* *cycle* --  a loop of nodes. A network without these is termed acyclic.
* *modularity* -- how strongly is the network divided into modules. An undivided network has modularity zero.


**Amplification factor** is an attribute of a node in a directed network, which captures the ratio of outgoing nodes to incoming nodes;
$$
    \mathcal{A} = \frac{k_\text{out}}{k_\text{in}},
$$
in which $k$ is the degree. We can expand this concept to un-directed graphs, for which
$
    \mathcal{A} = k-1.
$
Rather than considering only a single node, we can determine the the weighted average amplification factor of a network:

$$
    \mathcal{A} = \frac{\int k(k-1)n(k)dk}{\int kn(k)dk} = \frac{\left<k^2\right>}{\left<k\right>}-1
$$

so, degree times amplification factor times occurance, divided by the total degree of the network.

### Scale-free networks
**Scale-free networks** are networks in the degree distribution follows a power law. Examples of scale free networks include:
* WWW
* Software dependency graphs
* Semantic networks


### Analyzing patterns of networks
A characterisitic of a network, which we can look for 


### Algorithms for generating scale-free networks