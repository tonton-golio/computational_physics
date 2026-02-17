# Transformers and Attention Mechanisms

## The attention mechanism

The **attention mechanism** allows a model to focus on different parts of the input when producing each element of the output. Given queries $Q$, keys $K$, and values $V$, **scaled dot-product attention** computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V,
$$

where $d_k$ is the dimension of the keys. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large, which would push the softmax into saturated regions with tiny gradients.

The attention weights $\text{softmax}(QK^T / \sqrt{d_k})$ form a matrix where each row sums to one, representing a learned soft alignment between query and key positions.

## Multi-head attention

Rather than computing a single attention function, the **transformer** uses multiple **attention heads** in parallel:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O,
$$

where each head computes attention on a different linear projection:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V).
$$

Different heads can learn to attend to different types of relationships (syntactic, semantic, positional).

## The transformer architecture

The **transformer** (Vaswani et al., 2017) replaces recurrence entirely with attention. The encoder-decoder architecture consists of:

**Encoder block** (repeated $N$ times):
1. Multi-head self-attention.
2. Add & layer normalization.
3. Position-wise feedforward network.
4. Add & layer normalization.

**Decoder block** (repeated $N$ times):
1. Masked multi-head self-attention (causal mask prevents attending to future tokens).
2. Multi-head cross-attention (attends to encoder output).
3. Position-wise feedforward network.
4. Add & layer normalization at each step.

**Positional encoding** injects sequence order information since attention is permutation-invariant:

$$
\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \qquad \text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}).
$$

[[simulation adl-attention-heatmap]]

## BERT: bidirectional encoder representations

**BERT** (Devlin et al., 2019) uses only the encoder stack with two pre-training objectives:

- **Masked language modeling** (MLM): randomly mask 15% of input tokens and predict them from context.
- **Next sentence prediction** (NSP): predict whether two sentences are consecutive.

BERT's bidirectional attention allows each token to attend to both left and right context, unlike autoregressive models. Fine-tuning BERT on downstream tasks (classification, NER, QA) achieved state-of-the-art results across NLP benchmarks.

## GPT: generative pre-trained transformer

**GPT** (Radford et al., 2018) uses only the decoder stack with causal (left-to-right) attention. Pre-training uses autoregressive language modeling:

$$
\mathcal{L} = -\sum_{t} \log P(x_t \mid x_1, \ldots, x_{t-1}).
$$

Scaling laws (Kaplan et al., 2020) showed that model performance improves predictably as a power law in model size, dataset size, and compute. This motivated the development of increasingly large models (GPT-2, GPT-3, GPT-4).

## Vision transformers (ViT)

**Vision Transformers** (Dosovitskiy et al., 2021) apply the transformer architecture to images by:

1. Splitting the image into fixed-size patches (e.g., 16x16 pixels).
2. Linearly embedding each patch into a vector.
3. Adding positional embeddings.
4. Processing the sequence of patch embeddings through a standard transformer encoder.

ViTs demonstrate that the inductive biases of CNNs (translation equivariance, locality) are not strictly necessary. With sufficient data, transformers match or exceed CNN performance on image classification.

[[simulation adl-convolution-demo]]

## Scaling laws and emergent abilities

Key empirical findings about transformer scaling:

- **Loss scales as a power law** in model parameters $N$, dataset size $D$, and compute $C$.
- **Emergent abilities**: certain capabilities (arithmetic, chain-of-thought reasoning) appear only above a threshold model size.
- **Compute-optimal training** (Chinchilla scaling): for a fixed compute budget, model size and dataset size should be scaled proportionally.

These observations have shaped modern training strategies and motivated the development of ever-larger models.
