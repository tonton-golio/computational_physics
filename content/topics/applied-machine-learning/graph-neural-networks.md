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

The central idea in GNNs is beautifully simple. Every node is gossiping with its neighbors, then updating its own diary based on what it heard.

More precisely, a single GNN layer does two things:

1. **Aggregate**: each node collects messages from its neighbors.
2. **Update**: each node updates its own representation using the collected messages and its current state.

$$
m_v^{(l+1)}=\bigoplus_{u\in\mathcal{N}(v)}\phi^{(l)}(h_v^{(l)},h_u^{(l)},e_{uv})
$$
$$
h_v^{(l+1)}=\psi^{(l)}(h_v^{(l)},m_v^{(l+1)})
$$

In words: node $v$ gathers messages from all its neighbors $u \in \mathcal{N}(v)$ using a message function $\phi$. The aggregation $\bigoplus$ (sum, mean, or max) must be permutation-invariant — it should not matter what order you read the messages. Then node $v$ updates its hidden state $h_v$ using an update function $\psi$ that combines the old state with the aggregated messages.

After $L$ layers of message passing, each node's representation encodes information from its $L$-hop neighborhood. One layer captures immediate neighbors; two layers capture neighbors-of-neighbors; and so on.

## GCN normalization

The Graph Convolutional Network (GCN) is one of the simplest and most widely used GNN variants. For adjacency matrix $A$:

$$
\tilde{A}=A+I,\quad
H^{(l+1)}=\sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right)
$$

The key idea: we add self-loops ($A+I$, so each node also receives its own message), then normalize by the degree matrix $\tilde{D}$ so that high-degree nodes do not dominate simply because they have more neighbors. The result is a weighted average of neighbor features, followed by a linear transformation and nonlinearity — essentially a localized convolution on the graph.

## A tiny example you can compute by hand

Consider four nodes in a triangle-plus-tail graph: nodes A, B, C form a triangle, and node D connects only to C.

Initial features: $h_A=[1,0]$, $h_B=[0,1]$, $h_C=[1,1]$, $h_D=[0,0]$.

With a simple mean-aggregation GNN layer (no learned weights, just averaging):

* Node A's neighbors are B, C. Message: mean of $[0,1]$ and $[1,1]$ = $[0.5, 1.0]$.
* Node B's neighbors are A, C. Message: mean of $[1,0]$ and $[1,1]$ = $[1.0, 0.5]$.
* Node C's neighbors are A, B, D. Message: mean of $[1,0]$, $[0,1]$, $[0,0]$ = $[0.33, 0.33]$.
* Node D's neighbor is only C. Message: $[1,1]$.

After one layer, each node knows about its immediate neighbors. After two layers, node D would know about A and B through C — information has propagated across the graph.

## Oversmoothing: when everyone sounds the same

There is a catch with stacking many GNN layers. After enough layers it is like a party where everyone has talked to everyone else ten times. All opinions have blended into one polite average. Every node sounds exactly the same — useful information has been washed away. This is **oversmoothing**: node representations converge to indistinguishable vectors as you add layers, because information has diffused uniformly across the graph. After enough layers it is like every person in a group chat has forwarded every message to everyone else ten times — eventually everyone is saying exactly the same thing.

In practice, most GNNs work best with 2–4 layers. Beyond that, you need architectural tricks (skip connections, normalization, or attention mechanisms) to preserve node identity.

## Interactive simulations

[[simulation aml-graph-message-passing]]

## Practical notes

* The aggregation function must be permutation-invariant — sum, mean, and max are the standard choices.
* Task formulation matters: node-level tasks (classify each node), edge-level tasks (predict whether an edge exists), and graph-level tasks (predict a property of the whole graph) require different readout strategies.
* For graph-level tasks, you need a pooling step that aggregates all node representations into a single graph vector.

## Big Ideas

* The central insight of GNNs is that structure is information. Two molecules with the same atoms but different connectivity are different molecules — and any model that ignores the connectivity is throwing away the most important part of the data.
* Message passing is a beautifully general operation: it subsumes convolution (which is message passing on a grid), diffusion (which is message passing on a manifold), and social influence (which is message passing on a social graph).
* Permutation invariance is not optional — it is a hard constraint. The correct answer must not change because you listed node A before node B in your input file.
* Oversmoothing is the depth tax on GNNs: each layer you add expands the information horizon but dilutes node identity. Most practical GNNs are shallower than you would expect.

## What Comes Next

Graph neural networks handle data where the structure is relational — but not all structure is spatial. Some data only makes sense in order: sentences, time series, sensor streams. Recurrent neural networks, covered next, bend the feedforward graph into a loop, giving the network memory — the ability to carry information from one timestep to the next. The message-passing idea in GNNs and the hidden-state idea in RNNs are both answers to the same question — how do you propagate information across a structured input? — and seeing them side by side sharpens your intuition for both.

Beyond this module, GNNs are an active research frontier: attention-based graph networks, equivariant architectures for physical systems, and graph transformers all build on the message-passing foundation you just learned. If you care about molecules, protein structure, materials science, or any domain where relationships matter as much as quantities, GNNs are where the action is.

## Check your understanding

* Can you explain message passing to a friend using the gossip analogy?
* What visual would you draw to show a friend why too many GNN layers cause oversmoothing?
* What experiment would test whether your GNN benefits from 2 layers versus 4?

## Challenge

Implement message passing by hand on the four-node triangle-plus-tail graph from the worked example, but this time with learned weights. Initialize $W$ randomly, define a graph-level property (for example, the sum of all final node representations), and write out the full gradient computation to update $W$ by gradient descent — deriving the chain rule through the aggregation step yourself. Now generalize: does your derivation change if the aggregation function is max instead of mean? If it is not differentiable everywhere (as in max-pooling), how would you handle the gradient at ties?
