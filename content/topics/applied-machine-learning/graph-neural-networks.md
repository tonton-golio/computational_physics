# Graph Neural Networks

What if your data has friendships?

Not all data fits neatly into a table or a sequence. Molecules are atoms connected by bonds. Social networks are people connected by relationships. Protein structures are amino acids connected in 3D space. In all these cases, the connections carry as much information as the entities themselves. Graph neural networks learn on this structure directly, without forcing relational data into a flat grid.

## Why graphs

Many systems we care about are fundamentally relational:

* **Molecules**: atoms are nodes, bonds are edges. Predicting molecular properties requires understanding the connectivity, not just a list of atom types.
* **Social and communication networks**: who talks to whom shapes information flow, influence, and community structure.
* **Recommendation systems**: users and items form a bipartite graph, and similar users connect to similar items.
* **Physical simulations**: particles interact through local forces, forming interaction graphs.

A graph $G=(V,E)$ has nodes $V$ (entities) and edges $E$ (relationships). Each node can carry a feature vector, and edges can have their own features (bond type, interaction strength, distance).

## Message passing: the gossip network

And here is the gorgeous part. The central idea in GNNs is beautifully simple. Every node is gossiping with its neighbors, then updating its own diary based on what it heard.

A single GNN layer does two things:

1. **Aggregate**: each node collects messages from its neighbors.
2. **Update**: each node updates its own representation using the collected messages and its current state.

$$
m_v^{(l+1)}=\bigoplus_{u\in\mathcal{N}(v)}\phi^{(l)}(h_v^{(l)},h_u^{(l)},e_{uv})
$$
$$
h_v^{(l+1)}=\psi^{(l)}(h_v^{(l)},m_v^{(l+1)})
$$

Node $v$ gathers messages from all its neighbors $u \in \mathcal{N}(v)$ using a message function $\phi$. The aggregation $\bigoplus$ (sum, mean, or max) must be permutation-invariant -- it should not matter what order you read the messages. Then node $v$ updates its hidden state $h_v$ by combining the old state with the aggregated messages.

After $L$ layers, each node's representation encodes information from its $L$-hop neighborhood. One layer captures immediate neighbors; two layers captures neighbors-of-neighbors; and so on.

[[simulation aml-graph-message-passing]]

## GCN normalization

The Graph Convolutional Network (GCN) is one of the simplest and most widely used GNN variants. For adjacency matrix $A$:

$$
\tilde{A}=A+I,\quad
H^{(l+1)}=\sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right)
$$

The key idea: add self-loops ($A+I$, so each node also receives its own message), then normalize by the degree matrix $\tilde{D}$ so that high-degree nodes do not dominate simply because they have more neighbors. The result is a weighted average of neighbor features, followed by a linear transformation and nonlinearity -- a localized convolution on the graph.

## Oversmoothing: when everyone sounds the same

There is a catch with stacking many GNN layers. After enough rounds of gossip, it is like a party where everyone has talked to everyone else ten times. All opinions have blended into one polite average. Every node sounds exactly the same -- useful information has been washed away.

This is **oversmoothing**: node representations converge to indistinguishable vectors as you add layers, because information has diffused uniformly across the graph. In practice, most GNNs work best with 2-4 layers. Beyond that, you need skip connections, normalization, or attention mechanisms to preserve node identity.

## So what did we really learn?

The central insight of GNNs is that structure is information. Two molecules with the same atoms but different connectivity are different molecules -- any model that ignores the connectivity is throwing away the most important part of the data.

Message passing is a beautifully general operation: it subsumes convolution (message passing on a grid), diffusion (message passing on a manifold), and social influence (message passing on a social graph).

Permutation invariance is not optional -- it is a hard constraint. The correct answer must not change because you listed node A before node B in your input file.

And oversmoothing is the depth tax on GNNs: each layer you add expands the information horizon but dilutes node identity. Most practical GNNs are shallower than you would expect.

## Challenge

Implement message passing by hand on a small graph: four nodes, a triangle plus a tail. Initialize node features, pick a simple aggregation (mean), and compute two rounds of message passing on paper. Then add learned weights $W$, define a graph-level property (say, the sum of all final node representations), and write out the full gradient computation to update $W$ by gradient descent. Does your derivation change if you switch from mean to max aggregation? If max is not differentiable everywhere, how would you handle the gradient at ties?
