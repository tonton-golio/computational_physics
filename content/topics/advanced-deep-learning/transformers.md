# Transformers and Attention Mechanisms

## Why attention?

Imagine reading a 500-word paragraph and trying to connect word 1 to word 499. An RNN has to pass that detail through a chain of 498 intermediate steps -- like the world's worst game of telephone. By the time the signal arrives it's been transformed, attenuated, and mixed with everything in between. And because each step depends on the previous one, you can't even parallelize it.

Attention tears up the telephone game: every word gets a direct hotline to every other word and simply asks, "How relevant are you to me right now?" That single idea is why transformers replaced everything else.

## The attention mechanism

You're at a loud party trying to hear your friend across the room. Your brain doesn't process every sound equally -- it automatically turns up the volume on your friend's voice and turns down the background chatter. That's attention: a learned mechanism for focusing on the relevant parts while ignoring the rest.

In a neural network, attention works through three roles: **queries**, **keys**, and **values**. Think of it as a database lookup. The query is "what am I looking for?" The key is "what do I have to offer?" The value is "here's the actual information." Attention computes a soft match between each query and all keys, then returns a weighted combination of values:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V.
$$

The scaling factor $\sqrt{d_k}$ prevents dot products from growing too large, which would push softmax into saturated regions with tiny gradients.

The attention weights form a matrix where each row sums to one. You can picture it as every word voting on which other words matter most to it. The word "it" might vote heavily for "the cat" when trying to determine what "it" refers to.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```
<!--code-toggle-->
```pseudocode
FUNCTION scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = DIMENSION(Q, -1)
    scores = MATMUL(Q, TRANSPOSE(K)) / SQRT(d_k)
    IF mask IS NOT None:
        scores = FILL_WHERE(scores, mask == 0, -INF)
    weights = SOFTMAX(scores, dim=-1)
    RETURN MATMUL(weights, V)
```

## What if we didn't have attention?

Without attention, a recurrent network passes information from word 1 to word 100 through 99 sequential steps. Long-range dependencies become almost impossible to learn. Attention gives every word a direct phone line to every other word -- no intermediaries, no signal loss. And because all the lookups happen in parallel, it's also vastly faster to compute.

## Multi-head attention

A single attention function captures one type of relationship at a time. But language is rich -- words relate syntactically, semantically, positionally, and logically. **Multi-head attention** runs several attention functions in parallel, each looking for a different pattern:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O.
$$

One head might attend to adjacent words. Another focuses on long-range syntax. Another tracks coreference. The network learns which relationships are useful.

## The transformer architecture

The **transformer** replaces recurrence entirely with attention. It changed everything -- modern language models, vision transformers, protein structure prediction.

**Encoder block** (repeated $N$ times):
1. Multi-head self-attention.
2. Add & layer normalization.
3. Position-wise feedforward network.
4. Add & layer normalization.

**Decoder block** (repeated $N$ times):
1. Masked multi-head self-attention (causal mask prevents attending to future tokens -- no cheating).
2. Multi-head cross-attention (attends to encoder output).
3. Position-wise feedforward network.
4. Add & layer normalization at each step.

**Positional encoding** injects sequence order since attention is permutation-invariant. Without it, "dog bites man" and "man bites dog" would look identical:

$$
\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \qquad \text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}).
$$

[[simulation adl-attention-heatmap]]

## The architecture zoo (briefly)

**BERT** uses the encoder stack with masked language modeling -- randomly mask 15% of tokens and predict them from context. Bidirectional attention lets each token see both left and right.

**GPT** uses the decoder stack with causal attention -- predict the next word given all previous words. Autoregressive generation, scaled to enormous size.

**Vision Transformers** (ViT) split images into patches, embed each patch as a "word," and process them through a standard transformer encoder. They proved that CNN-style inductive biases aren't strictly necessary -- with enough data, transformers match or exceed CNN performance on images.

## Scaling laws and emergent abilities

Wait till you see this. Double the size, double the data, and the machine suddenly invents arithmetic it never saw before. Loss scales as a power law in model parameters, dataset size, and compute -- predictably, boringly, like clockwork. But then certain capabilities (chain-of-thought reasoning, translation) appear only above a threshold model size. Small models can't do them at all; large models suddenly can. That's still one of the most startling things anyone has watched.

## Beyond transformers

* **Diffusion models** have surpassed GANs for image generation by iteratively denoising random noise.
* **State space models** (Mamba, S4) offer linear scaling in sequence length, showing strong performance where the quadratic cost of attention becomes prohibitive.

## Big Ideas

* Attention replaces the sequential telephone chain with a direct connection between every pair of positions -- constant-depth path between any two tokens, regardless of distance.
* The query-key-value abstraction is a soft database lookup: retrieve a weighted blend of all records, where the weights are learned from context.
* Positional encodings are a confession that attention is permutation-invariant -- without them, word order vanishes entirely.

## What Comes Next

You've seen how transformers achieve remarkable things by letting every element attend to every other. But these achievements raise deep questions: why do these systems generalize so well when classical theory says they shouldn't? Why does a model with more parameters than training examples sometimes generalize *better*? The final lesson steps back from the architecture zoo and asks the theoretical questions -- what does the loss landscape look like, why does gradient descent find good solutions, and what happens when networks are attacked with imperceptible perturbations?
