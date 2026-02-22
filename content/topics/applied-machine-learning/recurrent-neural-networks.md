# Recurrent Neural Networks (Extension)

Sequences have an arrow of time. RNNs remember by carrying a little suitcase of state from one step to the next -- until the suitcase gets too heavy and the gradient vanishes. Then we give it gates.

## Core recurrence

A standard feedforward network processes each input independently, with no memory. An RNN adds a crucial ingredient: a hidden state $h_t$ that carries information from one timestep to the next.

At each timestep $t$:
$$
h_t=f(W_{xh}x_t + W_{hh}h_{t-1} + b_h),\quad
y_t=g(W_{hy}h_t+b_y)
$$

The hidden state $h_t$ is a blend of the current input $x_t$ and the previous hidden state $h_{t-1}$, passed through a nonlinearity. The same weights are reused at every timestep -- the network "unrolls" through time but shares parameters. This means an RNN can, in principle, handle sequences of any length, reading one token at a time and carrying a compressed summary of everything it has seen.

## The vanishing gradient problem

Here is the catch. When we backpropagate through many timesteps, the gradients get multiplied by the same weight matrix over and over. If that matrix has eigenvalues less than 1, the gradients shrink exponentially -- they *vanish*. The network forgets early inputs because the error signal from the end of the sequence never makes it back to the beginning.

The opposite can also happen: eigenvalues greater than 1 make gradients *explode*, and training becomes unstable. Gradient clipping handles explosions, but vanishing gradients require a more fundamental fix.

## Gated variants: LSTM and GRU

The solution is to give the network a notebook -- a persistent memory cell with explicit gates that control what to remember, what to forget, and what to output.

**LSTM (Long Short-Term Memory)** introduces a cell state $c_t$ running alongside the hidden state. Three gates control it: the *forget gate* decides what old information to erase, the *input gate* decides what new information to write, and the *output gate* decides what to expose as the hidden state. Because the cell state passes through timesteps with only additive updates (no repeated matrix multiplication), gradients flow much more easily over long sequences.

**GRU (Gated Recurrent Unit)** simplifies to two gates, merging cell state and hidden state into one. Faster to train, often performs comparably. When in doubt, try both.

## So what did we really learn?

Sequential data has an arrow of time, and that arrow matters. An architecture that ignores order is fundamentally misspecified for the problem.

The vanishing gradient problem is not a quirk of bad implementation -- it is a mathematical inevitability when you multiply the same matrix hundreds of times. LSTM gates are a structural solution, not a patch.

And gating is the key idea: instead of forcing all information through a bottleneck, let the network learn *what* to pass. The same idea reappears in attention mechanisms and transformers, which have largely replaced recurrent architectures by letting every token attend to every other token in a single parallel operation.

## Challenge

Design an experiment to isolate how much long-range context an LSTM actually uses. Construct synthetic sequences where the correct prediction at the last position depends on a token at position 1, with 50 irrelevant tokens in between. Train an LSTM and a baseline that can only see the last 5 tokens. Compare accuracy, then vary the gap length and plot how LSTM accuracy degrades. At what gap length does the LSTM stop beating the short-context baseline? What does that tell you about the effective memory horizon of a trained LSTM?
