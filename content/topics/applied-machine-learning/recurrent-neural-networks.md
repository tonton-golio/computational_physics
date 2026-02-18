# Recurrent Neural Networks (Extension)

Some data only makes sense in order. The sentence "the cat sat on the mat" means something; "mat the on sat cat the" does not. Temperature readings, stock prices, sensor streams, musical notes — whenever the meaning depends on sequence, we need a model that remembers what came before. That is what recurrent neural networks do.

## Core recurrence

A standard feedforward network processes each input independently, with no memory. An RNN adds a crucial ingredient: a hidden state $h_t$ that carries information from one timestep to the next.

At each timestep $t$:
$$
h_t=f(W_{xh}x_t + W_{hh}h_{t-1} + b_h),\quad
y_t=g(W_{hy}h_t+b_y)
$$

In words: the hidden state $h_t$ is a blend of the current input $x_t$ and the previous hidden state $h_{t-1}$, passed through a nonlinearity. The output $y_t$ is then computed from the hidden state. The same weights $W_{xh}$, $W_{hh}$, and $W_{hy}$ are reused at every timestep — the network "unrolls" through time but shares parameters.

This means an RNN can, in principle, handle sequences of any length. It reads one token at a time, updates its internal state, and carries a compressed summary of everything it has seen so far.

## The vanishing gradient problem

Here is the catch. Imagine a game of telephone where a message passes through 50 people. By the time it reaches the last person, the message is garbled beyond recognition. The same thing happens in an RNN: when we backpropagate through many timesteps, the gradients get multiplied by the same weight matrix over and over. If that matrix has eigenvalues less than 1, the gradients shrink exponentially — they *vanish*. The network forgets early inputs because the error signal from the end of the sequence never makes it back to the beginning.

The opposite can also happen: if eigenvalues are greater than 1, gradients *explode*, and training becomes unstable. Gradient clipping (capping the gradient norm) is a practical fix for explosions, but vanishing gradients require a more fundamental architectural change.

## Gated variants: LSTM and GRU

The solution is to give the network a notebook — a persistent memory cell with explicit gates that control what to remember, what to forget, and what to output.

**LSTM (Long Short-Term Memory)** introduces a cell state $c_t$ that runs alongside the hidden state. Three gates control it: the *forget gate* decides what old information to erase, the *input gate* decides what new information to write, and the *output gate* decides what part of the cell state to expose as the hidden state. Because the cell state can pass through timesteps with only additive updates (no repeated matrix multiplication), gradients flow much more easily over long sequences.

**GRU (Gated Recurrent Unit)** is a simplified variant with two gates instead of three. It merges the cell state and hidden state into one, using a *reset gate* and an *update gate*. GRUs are faster to train and often perform comparably to LSTMs. When in doubt, try both.

## A concrete example

Consider predicting the next word. Given "The cat sat on the ___", a simple RNN can likely predict "mat" — the context is short and recent. But now consider: "The author, who grew up in Paris and studied literature at the Sorbonne before moving to New York where she worked as a journalist for twenty years, finally published her ___". To predict "book" or "novel," the model must remember "author" and "published" across a gap of 30+ words. A vanilla RNN's hidden state would have long forgotten "author" by the time it reaches "her." An LSTM, with its explicit memory cell, can carry that information across the entire sentence.

## Where this topic goes deeper

This page gives you the core intuition for sequence modeling. For full architectural details — bidirectional RNNs, attention mechanisms, the evolution from LSTMs to transformers, and practical sequence modeling workflows — see [Advanced Deep Learning — Sequence Models](/topics/advanced-deep-learning/ann).

## Practical checklist

- Normalize and window your sequence data carefully — sequence models are sensitive to scale and length.
- Always use chronological validation splits, never random shuffling.
- Start with an LSTM or GRU. Switch to a transformer if sequences are long and you have enough data.
- Track both short-horizon and long-horizon error metrics to understand where your model struggles.

## Check your understanding

- Can you explain the vanishing gradient problem using the telephone game analogy?
- What is the one picture in your head that shows how an LSTM gate decides what to remember?
- What experiment would reveal whether your sequence model is actually using long-range context or just predicting from the last few tokens?
