# Recurrent Neural Networks (Extension)

Recurrent neural networks (RNNs) are sequence models that maintain a hidden state over time. They are useful when prediction depends on order and context, such as language, sensor streams, and temporal forecasting.

## Core recurrence
At timestep $t$:
$$
h_t=f(W_{xh}x_t + W_{hh}h_{t-1} + b_h),\quad
y_t=g(W_{hy}h_t+b_y)
$$

## Why gated variants matter
Vanilla RNNs struggle with long dependencies due to vanishing/exploding gradients. LSTM and GRU architectures introduce gates to improve memory and gradient flow.

## Where this topic fits
This page is intentionally lightweight in Applied Machine Learning. For deeper architecture, training details, and sequence modeling workflows, see the broader deep learning topic:
- `/topics/advanced-deep-learning/ann`

## Practical checklist
- Normalize and window sequence data carefully.
- Use chronological validation splits.
- Track both short-horizon and long-horizon error metrics.
