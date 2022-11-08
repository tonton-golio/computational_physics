
### Intro

A network is a set of nodes connected by edges. **Networks are everywhere**, examples include:

### Social-networks
Nodes represent bots and individuals, and edges mark connection. Edges may carry weight dependent on the frequency of interaction. Some persons are typically central in their network, and these networks tend to have high cliqueishness. These networks are scale-free, some indivs. have all the friends.

### Paper-authors
People tend to cite central people in their publication network. Thus these networks are also usually scale-free.

### Features of networks
* cliqueishness
* degree
* centrality
* connectedness

#### Amplification factor
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