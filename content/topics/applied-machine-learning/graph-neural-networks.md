# Graph Neural Networks

Graphs represent entities and relations directly. Graph neural networks (GNNs) learn on this structure without forcing data into fixed grids.

## Why graphs
Many systems are relational:
- molecules (atoms and bonds),
- social and communication networks,
- recommendation systems,
- physical interaction graphs.

## Message passing view
A standard GNN layer aggregates neighborhood information:
$$
m_v^{(l+1)}=\bigoplus_{u\in\mathcal{N}(v)}\phi^{(l)}(h_v^{(l)},h_u^{(l)},e_{uv})
$$
$$
h_v^{(l+1)}=\psi^{(l)}(h_v^{(l)},m_v^{(l+1)})
$$

Popular architectures include GCN, GraphSAGE, and GAT.

## GCN normalization
For adjacency $A$:
$$
\tilde{A}=A+I,\quad
H^{(l+1)}=\sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right)
$$

## Interactive simulations
[[simulation aml-graph-convolution-intuition]]

[[simulation aml-graph-adjacency-demo]]

[[simulation aml-graph-message-passing]]

## Practical notes
- Aggregation must be permutation-invariant.
- Deeper vanilla GCNs can oversmooth node representations.
- Task formulation matters: node-level, edge-level, and graph-level objectives differ.
