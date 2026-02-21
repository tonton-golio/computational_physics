# Transformers and Attention Mechanisms

## Why attention?

Imagine reading a 500-word paragraph and trying to connect word 1 to word 499. An RNN has to pass that detail through a chain of 498 intermediate steps — like the world's worst game of telephone. By the time the signal arrives it has been transformed, attenuated, and mixed with everything in between. And because each step depends on the previous one, you cannot even parallelize the computation.

Attention tears up the telephone game: every word gets a direct hotline to every other word and simply asks, "How relevant are you to me right now?" That single idea is why transformers replaced everything else.

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

The **transformer** replaces recurrence entirely with attention. It is the architecture that changed everything — enabling modern language models, vision transformers, and protein structure prediction.

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

**BERT** uses only the encoder stack with two pre-training objectives:

* **Masked language modeling** (MLM): randomly mask 15% of input tokens and predict them from context. The network must understand language deeply enough to fill in the blanks.
* **Next sentence prediction** (NSP): predict whether two sentences are consecutive.

BERT's bidirectional attention allows each token to attend to both left and right context, unlike autoregressive models that can only look backward. Fine-tuning BERT on downstream tasks (classification, NER, QA) achieved state-of-the-art results across NLP benchmarks.

## GPT: generative pre-trained transformer

**GPT** uses only the decoder stack with causal (left-to-right) attention. Pre-training uses autoregressive language modeling — predict the next word given all previous words:

$$
\mathcal{L} = -\sum_{t} \log P(x_t \mid x_1, \ldots, x_{t-1}).
$$

Scaling laws showed that model performance improves predictably as a power law in model size, dataset size, and compute. This motivated the development of increasingly large models (GPT-2, GPT-3, GPT-4).

## Vision transformers (ViT)

**Vision Transformers** (ViT) apply the transformer architecture to images by:

1. Splitting the image into fixed-size patches (e.g., 16x16 pixels).
2. Linearly embedding each patch into a vector — treating each patch as a "word."
3. Adding positional embeddings.
4. Processing the sequence of patch embeddings through a standard transformer encoder.

ViTs demonstrate that the inductive biases of CNNs (translation equivariance, locality) are not strictly necessary. With sufficient data, transformers match or exceed CNN performance on image classification. The patches attend to each other just like words do, and the network learns which spatial relationships matter.

## Scaling laws and emergent abilities

Key empirical findings about transformer scaling:

* **Loss scales as a power law** in model parameters $N$, dataset size $D$, and compute $C$. Double the compute, and you get a predictable improvement.
* **Emergent abilities**: certain capabilities (arithmetic, chain-of-thought reasoning) appear only above a threshold model size. Small models cannot do them at all; large models suddenly can.
* **Compute-optimal training** (Chinchilla scaling): for a fixed compute budget, model size and dataset size should be scaled proportionally. Many early large models were undertrained on too little data.

## Beyond transformers

The field continues to evolve rapidly:
* **Diffusion models** have surpassed GANs for image generation by iteratively denoising random noise, guided by learned score functions.
* **State space models** (Mamba, S4) offer an alternative to attention with linear scaling in sequence length, showing strong performance on long-context tasks where the quadratic cost of attention becomes prohibitive.

## Big Ideas

* Attention replaces the sequential telephone chain of recurrence with a direct connection between every pair of positions — a constant-depth path between any two tokens, regardless of how far apart they are in the sequence.
* The query-key-value abstraction is a soft database lookup: instead of retrieving exactly one record, you retrieve a weighted blend of all records, where the weights are learned from context.
* Positional encodings are a confession that attention itself is permutation-invariant — without them, "dog bites man" and "man bites dog" are identical to the model, so you must inject sequence order by hand.
* Emergent abilities at scale reveal something unsettling about our theories: capabilities that do not exist in small models appear suddenly at a threshold of scale, which means our generalization bounds and loss curves told us almost nothing about what the model would become.

## What Comes Next

You have now seen how transformers achieve remarkable things by letting every element attend to every other. But these achievements raise deep questions: why do these systems generalize so well when classical theory says they should not? Why does a model with more parameters than training examples sometimes generalize *better*? The final lesson steps back from the architecture zoo and asks the theoretical questions: what does the loss landscape actually look like, why does gradient descent find good solutions, and what happens when networks are attacked with imperceptible perturbations? Understanding what deep learning can and cannot guarantee is the foundation for using it responsibly.

## Check Your Understanding

1. Self-attention computes dot products between every pair of positions, making the complexity quadratic in sequence length. For a sequence of 8,000 tokens with 64-dimensional keys, roughly how many floating-point operations does a single attention layer require? Why does this become a bottleneck for very long sequences?
2. BERT uses bidirectional attention (each token can attend to all other tokens), while GPT uses causal attention (each token can only attend to previous tokens). Why is causal masking necessary for language generation but not for classification tasks like sentiment analysis?
3. Vision Transformers split images into 16x16 patches and treat each patch as a "word." What spatial inductive biases does this design abandon compared to a CNN, and what does a ViT need to learn from scratch that a CNN gets for free?

## Challenge

The attention matrix in a transformer has shape (sequence length) x (sequence length) and costs quadratic memory and compute. Design and implement a **sparse attention** scheme that reduces this to near-linear scaling: for example, each token attends to its local neighborhood plus a set of globally accessible "summary" tokens. Train your sparse attention model on a sequence classification task and compare it to full attention. At what sequence length does your sparse scheme start to outperform full attention in wall-clock time while matching it in accuracy? Where does it fail?

