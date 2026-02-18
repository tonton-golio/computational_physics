# Transformers and Attention Mechanisms

## Why attention?

Recurrent neural networks (RNNs) process sequences one token at a time, creating two fundamental limitations:
- **Sequential processing**: Each step depends on the previous, preventing parallelization during training.
- **Long-range dependencies**: Information from early tokens must pass through many time steps to reach later ones, causing vanishing gradients and making it difficult to capture distant relationships.

The **attention mechanism** solves both problems in one stroke: it allows every position to directly attend to every other position in a single computation step. No waiting in line. No game of telephone where the signal degrades over distance. Every word can talk to every other word directly.

## The attention mechanism

Here is the core intuition. You are at a loud party trying to hear your friend across the room. Your brain does not process every sound equally — it automatically turns up the volume on your friend's voice and turns down the background chatter. That is attention: a learned mechanism for focusing on the relevant parts of the input while ignoring the rest.

In a neural network, attention works through three roles: **queries**, **keys**, and **values**. Think of it as a database lookup. The **query** is "what am I looking for?" The **key** is "what do I have to offer?" The **value** is "here is the actual information." Attention computes a soft match between each query and all keys, then returns a weighted combination of the corresponding values.

Formally, given queries $Q$, keys $K$, and values $V$, **scaled dot-product attention** computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V,
$$

where $d_k$ is the dimension of the keys. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large, which would push the softmax into saturated regions with tiny gradients.

The attention weights $\text{softmax}(QK^T / \sqrt{d_k})$ form a matrix where each row sums to one. You can picture it as every word in a sentence voting on which other words matter most to it. The word "it" might vote heavily for "the cat" when trying to determine what "it" refers to.

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

## Multi-head attention

A single attention function can only capture one type of relationship at a time. But language is rich: words relate to each other syntactically (subject-verb), semantically (synonyms), positionally (nearby words), and logically (cause-effect). **Multi-head attention** runs several attention functions in parallel, each looking for a different type of pattern:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O,
$$

where each head computes attention on a different linear projection:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V).
$$

One head might learn to attend to adjacent words. Another might focus on long-range syntactic dependencies. Another might track coreference. The network learns which types of relationships are useful.

## The transformer architecture

The **transformer** (Vaswani et al., 2017) replaces recurrence entirely with attention. It is the architecture that changed everything — enabling modern language models, vision transformers, and protein structure prediction.

**Encoder block** (repeated $N$ times):
1. Multi-head self-attention.
2. Add & layer normalization.
3. Position-wise feedforward network.
4. Add & layer normalization.

**Decoder block** (repeated $N$ times):
1. Masked multi-head self-attention (causal mask prevents attending to future tokens — you cannot cheat by looking at the answer).
2. Multi-head cross-attention (attends to encoder output).
3. Position-wise feedforward network.
4. Add & layer normalization at each step.

**Positional encoding** injects sequence order information since attention is permutation-invariant. Without it, the transformer would treat "dog bites man" and "man bites dog" identically. The standard approach uses sine and cosine functions of different frequencies:

$$
\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \qquad \text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}).
$$

[[simulation adl-attention-heatmap]]

## What if we didn't have attention?

Without attention, a recurrent network must pass information from word 1 to word 100 through 99 sequential steps. By the time the signal arrives, it has been transformed, attenuated, and mixed with everything in between. Long-range dependencies become almost impossible to learn. Attention gives every word a direct phone line to every other word — no intermediaries, no signal loss.

## BERT: bidirectional encoder representations

**BERT** (Devlin et al., 2019) uses only the encoder stack with two pre-training objectives:

- **Masked language modeling** (MLM): randomly mask 15% of input tokens and predict them from context. The network must understand language deeply enough to fill in the blanks.
- **Next sentence prediction** (NSP): predict whether two sentences are consecutive.

BERT's bidirectional attention allows each token to attend to both left and right context, unlike autoregressive models that can only look backward. Fine-tuning BERT on downstream tasks (classification, NER, QA) achieved state-of-the-art results across NLP benchmarks.

## GPT: generative pre-trained transformer

**GPT** (Radford et al., 2018) uses only the decoder stack with causal (left-to-right) attention. Pre-training uses autoregressive language modeling — predict the next word given all previous words:

$$
\mathcal{L} = -\sum_{t} \log P(x_t \mid x_1, \ldots, x_{t-1}).
$$

Scaling laws (Kaplan et al., 2020) showed that model performance improves predictably as a power law in model size, dataset size, and compute. This motivated the development of increasingly large models (GPT-2, GPT-3, GPT-4).

## Vision transformers (ViT)

**Vision Transformers** (Dosovitskiy et al., 2021) apply the transformer architecture to images by:

1. Splitting the image into fixed-size patches (e.g., 16x16 pixels).
2. Linearly embedding each patch into a vector — treating each patch as a "word."
3. Adding positional embeddings.
4. Processing the sequence of patch embeddings through a standard transformer encoder.

ViTs demonstrate that the inductive biases of CNNs (translation equivariance, locality) are not strictly necessary. With sufficient data, transformers match or exceed CNN performance on image classification. The patches attend to each other just like words do, and the network learns which spatial relationships matter.

## Scaling laws and emergent abilities

Key empirical findings about transformer scaling:

- **Loss scales as a power law** in model parameters $N$, dataset size $D$, and compute $C$. Double the compute, and you get a predictable improvement.
- **Emergent abilities**: certain capabilities (arithmetic, chain-of-thought reasoning) appear only above a threshold model size. Small models cannot do them at all; large models suddenly can.
- **Compute-optimal training** (Chinchilla scaling): for a fixed compute budget, model size and dataset size should be scaled proportionally. Many early large models were undertrained on too little data.

## Beyond transformers

The field continues to evolve rapidly:
- **Diffusion models** have surpassed GANs for image generation by iteratively denoising random noise, guided by learned score functions.
- **State space models** (Mamba, S4) offer an alternative to attention with linear scaling in sequence length, showing strong performance on long-context tasks where the quadratic cost of attention becomes prohibitive.

## Further reading

- Vaswani, A., et al. (2017). *Attention Is All You Need*. The paper that introduced the transformer architecture and changed the field.
- Jay Alammar, *The Illustrated Transformer*. The single best visual explanation of how transformers work, step by step.
