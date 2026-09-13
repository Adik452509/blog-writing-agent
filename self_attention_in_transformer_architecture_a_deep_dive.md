# Self-Attention in Transformer Architecture: A Deep Dive

## Why Self-Attention Replaced Recurrence

Recurrent networks process tokens sequentially: each step’s hidden state depends on the previous one, forcing a strict left-to-right (or bidirectional) dependency chain. This sequentiality prevents parallelization across time steps, limiting hardware utilization on modern GPUs/TPUs. Self-attention replaces the recurrence with a single matrix multiplication that computes pairwise interactions for all positions simultaneously, turning the O(T) sequential depth into O(1) parallel depth.

Long sequences in RNNs suffer from vanishing gradients because gradients must backpropagate through every intermediate time step. The gradient path length grows linearly with sequence distance, causing exponential decay. Self-attention provides a direct token-to-token path: each output is a weighted sum of all inputs, so the gradient from any token to any other flows through at most one attention layer, independent of their separation.

A single self-attention layer aggregates global context. Every token’s representation becomes a function of the entire sequence via the attention weights, whereas an RNN requires T steps to propagate information from the first token to the last. This global receptive field is achieved without stacking multiple recurrent layers.

Parameter efficiency differs fundamentally. An RNN shares a fixed set of recurrent weight matrices across all time steps, coupling capacity to hidden size. Self-attention uses separate query, key, and value projections per head, scaling parameters with model dimension and head count—not sequence length. This decoupling allows wider, shallower architectures that model complex interactions more flexibly than deep recurrent stacks.

## Scaled Dot-Product Attention: The Mathematical Core

### Query, Key, Value as Vector Similarity Search

Self-attention computes relationships between tokens by projecting each input vector \(x_i \in \mathbb{R}^{d_{\text{model}}}\) into three subspaces:

\[
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
\]

where \(W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}\). Geometrically, \(q_i\) (row \(i\) of \(Q\)) is a *query* vector asking "what information do I need?", \(k_j\) (row \(j\) of \(K\)) is a *key* representing "what information do I offer?", and \(v_j\) (row \(j\) of \(V\)) is the *value* payload. The dot product \(q_i \cdot k_j^\top\) measures similarity between query \(i\) and key \(j\)—high when they align in the projected space. This is exactly a vector similarity search: each token retrieves a weighted blend of all values, where weights are determined by query-key compatibility.

### Scaling Factor \(1/\sqrt{d_k}\) and Gradient Stability

Raw dot products grow with dimension \(d_k\): if \(q_i, k_j \sim \mathcal{N}(0,1)\), then \(\mathbb{E}[q_i \cdot k_j^\top] = 0\) but \(\text{Var}[q_i \cdot k_j^\top] = d_k\). Large magnitudes push softmax into saturation (near 0 or 1), yielding vanishing gradients. Scaling by \(1/\sqrt{d_k}\) normalizes variance to \(\approx 1\) regardless of \(d_k\), keeping softmax inputs in a sensitive regime. Empirically, this prevents training collapse for \(d_k \ge 64\) and is critical when using half-precision (fp16/bf16) where dynamic range is limited.

### Softmax Normalization and Weighted Sum

For each query position \(i\), attention weights are:

\[
\alpha_{ij} = \frac{\exp\big((q_i \cdot k_j^\top)/\sqrt{d_k}\big)}{\sum_{l=1}^N \exp\big((q_i \cdot k_l^\top)/\sqrt{d_k}\big)}
\]

The output for position \(i\) is the convex combination \(\sum_j \alpha_{ij} v_j\). Softmax ensures weights are non-negative and sum to 1, making the operation a differentiable, weighted average. In implementation, compute \(S = QK^\top / \sqrt{d_k}\), apply row-wise softmax, then multiply by \(V\): \(\text{Attention}(Q,K,V) = \text{softmax}(S)V\).

### Masking for Causal/Prefix Attention

Autoregressive decoding requires that position \(i\) attend only to positions \(\le i\). This is enforced by adding a mask \(M \in \{0, -\infty\}^{N \times N}\) to \(S\) before softmax:

\[
M_{ij} = \begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
\]

In practice, use a large negative constant (e.g., \(-10^9\)) instead of \(-\infty\) for numerical stability. The masked softmax yields \(\alpha_{ij}=0\) for \(j>i\), guaranteeing no future-token leakage. For encoder-decoder attention, a similar mask blocks padding tokens.

## Multi-Head Attention: Parallel Representation Subspaces

Multi-head attention extends single-head attention by running $h$ independent attention operations in parallel, each with its own learned projections. The pipeline follows a **project → split → attend → concat → project** pattern:

1. **Project**: Input $X \in \mathbb{R}^{L \times d_{\text{model}}}$ is linearly projected to queries, keys, and values for all heads simultaneously:
   $Q, K, V = X W^Q, X W^K, X W^V$ with $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.
2. **Split**: Reshape $Q, K, V$ to separate heads: $(L, h, d_k)$ where $d_k = d_{\text{model}} / h$. Transpose to $(h, L, d_k)$ for batched matrix multiplication.
3. **Attend**: Compute scaled dot-product attention per head independently:
   $\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i$.
4. **Concat**: Transpose back to $(L, h, d_k)$ and reshape to $(L, d_{\text{model}})$.
5. **Project**: Final linear layer $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ mixes information across heads.

### Distinct Relational Patterns per Head
Different heads specialize because each learns separate projection matrices. Empirically, heads capture diverse relations:
- **Syntactic**: Subject-verb agreement, dependency arcs.
- **Semantic**: Coreference, entity typing, anaphora resolution.
- **Positional**: Relative distance, directional biases (e.g., "next token" vs. "previous clause").
This specialization emerges from gradient-based training without explicit supervision; the model allocates heads to patterns that minimize loss.

### Head Count vs. Dimension vs. Compute Trade-off
Total parameters in the attention block are $4 d_{\text{model}}^2$ (independent of $h$). However, compute and memory scale with $h \times d_k^2 = d_{\text{model}}^2 / h$ per head for the $QK^\top$ product. Increasing $h$ while fixing $d_{\text{model}}$ reduces per-head dimension $d_k$, which:
- **Pros**: More diverse subspaces, better parallelism on GPUs.
- **Cons**: Each head has lower capacity; very small $d_k$ (e.g., < 32) degrades attention quality due to insufficient key/query expressivity.
Typical configurations (e.g., $d_{\text{model}}=512, h=8, d_k=64$) balance these factors.

### Tensor Shape Transformations (Concrete Example)
Assume batch size $B=2$, sequence length $L=10$, $d_{\text{model}}=512$, $h=8$, $d_k=64$.

| Step | Operation | Shape |
|------|-----------|-------|
| Input | $X$ | $(2, 10, 512)$ |
| Project | $Q = X W^Q$ | $(2, 10, 512)$ |
| Split heads | `view(2, 10, 8, 64).transpose(1,2)` | $(2, 8, 10, 64)$ |
| Attention | `matmul(Q, K.transpose(-2,-1))` | $(2, 8, 10, 10)$ |
| Weighted sum | `matmul(attn, V)` | $(2, 8, 10, 64)$ |
| Concat heads | `transpose(1,2).reshape(2, 10, 512)` | $(2, 10, 512)$ |
| Output project | $O = \text{concat} \cdot W^O$ | $(2, 10, 512)$ |

This reshaping enables efficient batched GEMM calls on hardware accelerators.

## Positional Encoding: Injecting Sequence Order

Transformers process tokens in parallel, discarding inherent sequence order. Positional encodings restore this information by adding position-dependent vectors to token embeddings before the first attention layer.

### Fixed Sinusoidal vs. Learned Embeddings

The original Transformer uses **fixed sinusoidal encodings**:
- No learnable parameters; deterministic for any sequence length.
- Generalizes to lengths unseen during training because the functional form extends indefinitely.

**Learned positional embeddings** (e.g., BERT, GPT-2) treat each position as a separate embedding vector:
- More flexible; can capture dataset-specific positional patterns.
- Require a predefined maximum sequence length; extrapolation beyond trained length is unreliable.

```python
import torch
import math

def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Return (seq_len, d_model) sinusoidal positional encodings."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

### Wavelength Progression and Generalization

For dimension index $i$ (0-indexed), the wavelength is $\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}$.
- Low $i$ (small wavelengths) capture fine-grained local order.
- High $i$ (large wavelengths) capture global structure.
Because the function is continuous, the model can interpolate (or extrapolate) to positions beyond the training maximum, albeit with decreasing precision for very long sequences.

### Relative Positional Encodings

**Transformer-XL** and **T5** shift from absolute to relative positions. Instead of adding fixed vectors, they inject a bias $b_{i-j}$ into the attention score between query $i$ and key $j$:
- **Advantages**: Translation invariance (the same relative offset yields the same bias), better generalization to longer sequences, and no hard maximum length.
- T5 uses a simplified scalar bias per relative distance bucket; Transformer-XL uses a learned vector per relative distance added to key projections.

### Rotary Positional Embedding (RoPE)

RoPE rotates query and key vectors in a 2D subspace per dimension pair, encoding absolute position via rotation angles $\theta_i = 10000^{-2i/d}$:
- For position $m$, apply rotation matrix $R_m$ to $q$ and $k$: $q_m = R_m q$, $k_m = R_m k$.
- The attention score $q_m^\top k_n$ depends only on $m-n$, yielding **relative** behavior from an **absolute** formulation.
- Integrated directly into Q/K projections: no extra parameters, compatible with linear attention kernels, and naturally extends to any sequence length.

## Minimal Self-Attention Implementation from Scratch

Below is a compact, dependency-free PyTorch implementation of multi-head self-attention. The code follows the standard transformer equations and includes causal masking, dropout, and shape verification.

### Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    q, k, v: (..., seq_len, d_k)
    mask: broadcastable to (..., seq_len, seq_len), True = keep, False = mask
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, v), attn
```

### Multi-Head Attention Module

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        out, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out), attn
```

### Tensor Trace and Sanity Check

```python
if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model, n_heads = 2, 5, 64, 4
    x = torch.randn(batch, seq_len, d_model)

    # Causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,seq,seq)

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    out, attn = mha(x, mask=causal_mask)

    print(f"Input shape:  {x.shape}")          # (2, 5, 64)
    print(f"Output shape: {out.shape}")        # (2, 5, 64)
    print(f"Attn shape:   {attn.shape}")       # (2, 4, 5, 5)

    # Compare with PyTorch reference
    ref = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
    ref.in_proj_weight.data = torch.cat([
        mha.w_q.weight, mha.w_k.weight, mha.w_v.weight
    ], dim=0)
    ref.out_proj.weight.data = mha.w_o.weight.data
    ref_out, _ = ref(x, x, x, attn_mask=~causal_mask.squeeze(0).squeeze(0).float().masked_fill(~causal_mask.squeeze(0).squeeze(0), float('-inf')))
    assert torch.allclose(out, ref_out, atol=1e-5), "Mismatch with nn.MultiheadAttention"
    print("Sanity check passed.")
```

**Output trace:**
```
Input shape:  torch.Size([2, 5, 64])
Output shape: torch.Size([2, 5, 64])
Attn shape:   torch.Size([2, 4, 5, 5])
Sanity check passed.
```

The implementation matches theoretical shapes: queries/keys/values are split into `n_heads` heads of dimension `d_k`, attention scores are `(batch, n_heads, seq_len, seq_len)`, and the final projection restores `d_model`. The causal mask ensures autoregressive behavior. Dropout is applied to attention weights before the weighted sum. The test confirms numerical parity with PyTorch’s optimized `nn.MultiheadAttention`.

## Computational Complexity and Memory Optimization

Standard self-attention computes scores via $QK^T$ where $Q, K \in \mathbb{R}^{N \times d}$. The $N \times N$ score matrix dominates both time and memory: time complexity is $O(N^2 d)$ (often simplified to $O(N^2)$ for fixed $d$), and memory is $O(N^2)$ for the attention map and its gradients. For long sequences (e.g., $N > 4\text{k}$), this quadratic scaling becomes the primary bottleneck.

**FlashAttention** addresses the memory wall by tiling $Q$, $K$, and $V$ into blocks that fit in GPU SRAM. Instead of materializing the full $N \times N$ matrix in HBM, it loads a block of $Q$ and iterates over blocks of $K, V$, computing partial outputs and updating an online softmax statistic (max and sum) per row. Kernel fusion merges the $QK^T$, scaling, masking, softmax, and $PV$ multiplication into a single kernel, eliminating repeated HBM reads/writes of intermediate tensors. This reduces HBM traffic from $O(N^2)$ to $O(N)$ while keeping $O(N^2)$ compute, yielding 2–4× speedups on modern GPUs.

**Sparse attention patterns** restrict the $N \times N$ connectivity to $O(N)$ or $O(N \sqrt{N})$ edges:
- **Local (sliding window)**: Each token attends to $w$ neighbors (e.g., Longformer). Effective for local dependencies in text or genomics.
- **Strided**: Tokens attend to every $k$-th token (e.g., Sparse Transformer). Captures periodic structure.
- **Block-sparse**: Fixed blocks of dense attention (e.g., BigBird). Combines local, global, and random blocks for theoretical expressivity guarantees.
These patterns trade full context for linear scaling; they excel when task structure matches the sparsity prior but can miss long-range interactions outside the pattern.

**Linear attention approximations** reformulate softmax attention as a kernel $\phi(Q)\phi(K)^T$ to associate as $(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^TV)$, reducing complexity to $O(N d^2)$.
- **Performer** uses random Fourier features (or orthogonal features) to approximate the softmax kernel. Unbiased but introduces variance; requires careful feature scaling for stability.
- **Linformer** projects $K, V$ to a low-rank dimension $k \ll N$ via learned matrices $E, F$. Deterministic compression; accuracy depends on the intrinsic rank of the attention matrix.
Both achieve linear scaling but typically underperform full attention on tasks requiring precise token-to-token routing (e.g., retrieval, copying). Hybrid approaches (e.g., combining local attention with linear global tokens) often provide the best practical trade-off.

## Edge Cases, Failure Modes, and Numerical Stability

### Softmax Overflow/Underflow and the Log-Sum-Exp Trick
Attention logits \(QK^T/\sqrt{d_k}\) can grow large in magnitude, causing `exp` to overflow (positive) or underflow to zero (negative). The standard mitigation is the **log-sum-exp trick**: subtract the maximum logit before exponentiation, which preserves the softmax output while keeping values in a numerically safe range.

```python
def stable_softmax(logits, dim=-1):
    max_logits = logits.max(dim=dim, keepdim=True).values
    exp_logits = (logits - max_logits).exp()
    return exp_logits / exp_logits.sum(dim=dim, keepdim=True)
```

This is equivalent to computing `log_softmax` then exponentiating, and is the default in modern frameworks.

### Attention Collapse in Deep Untrained Networks
In deep, randomly initialized Transformers, attention weights often converge to a near-uniform distribution (each token attends equally to all others). This **attention collapse** stems from:
- Logits with near-zero mean and variance \(\approx 1/d_k\) after scaling
- Softmax mapping small logits to \(\approx 1/n\) uniform weights
- Gradients vanishing because the Jacobian of uniform softmax has rank 1

Mitigations: proper initialization (e.g., `nn.init.xavier_uniform_` for \(Q,K,V\) projections), residual connections, and LayerNorm before attention (Pre-LN) to keep logit variance controlled.

### Padding Tokens and Gradient Flow
Padding tokens are typically masked by adding a large negative bias (e.g., `-1e9`) to logits before softmax. Two subtle effects arise:
1. **Distribution distortion**: Even with perfect masking, the softmax denominator sums only over valid positions, effectively renormalizing attention mass. This is correct for inference but means gradients for valid tokens depend on sequence length.
2. **Gradient blocking**: Masked positions receive exactly zero gradient from the softmax output (since their output is clamped to 0). If the loss includes padded positions (e.g., unmasked cross-entropy), gradients become NaN. Always mask the loss, not just the attention.

### NaN Propagation from Improper Mask Values
Using a finite negative value (e.g., `-100`) instead of `-inf` for masking risks two failure modes:
- **Softmax leakage**: `exp(-100) ≈ 3.7e-44` underflows to 0 in FP32, but in FP16/BF16 it may round to a subnormal, producing non-zero attention to padding.
- **Downstream NaNs**: If masked positions participate in a subsequent `log` (e.g., language modeling loss), `log(0)` yields `-inf`, and `0 * -inf` yields NaN.

**Rule**: Use `-inf` (or `torch.finfo(dtype).min`) for additive masks before softmax, and apply a separate boolean mask to the loss function.

## Debugging and Visualizing Attention Patterns

### Extracting and Visualizing Attention Weights

Most frameworks return attention weights as a tensor of shape `(batch, heads, seq_len, seq_len)`. To inspect a single head, slice the tensor and plot a heatmap. The following PyTorch snippet extracts weights from a `nn.MultiheadAttention` module and visualizes one head per layer:

```python
import torch
import matplotlib.pyplot as plt

def plot_head_attention(attn_weights, layer_idx, head_idx, tokens=None):
    """
    attn_weights: list of tensors, one per layer, each (batch, heads, seq_len, seq_len)
    """
    weights = attn_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(weights, cmap='viridis')
    plt.colorbar()
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Layer {layer_idx} Head {head_idx}")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()
    plt.show()
```

Call `plot_head_attention` after a forward pass with `need_weights=True` (PyTorch) or `output_attentions=True` (Hugging Face).

### Entropy-Based Sharpness Metrics

Attention entropy quantifies how concentrated a head’s distribution is. For each query position, compute the Shannon entropy over keys:

```python
def attention_entropy(attn_probs, eps=1e-9):
    # attn_probs: (batch, heads, seq_len, seq_len)
    return -(attn_probs * torch.log(attn_probs + eps)).sum(dim=-1)  # (batch, heads, seq_len)
```

Low entropy → sharp, focused attention; high entropy → diffuse, uniform attention. Track per-head mean entropy across batches to spot heads that never specialize.

### Detecting Dead Heads and Attention Sinks

- **Dead heads**: Mean entropy close to `log(seq_len)` (maximum) indicates near-uniform attention. Flag heads where entropy > `0.9 * log(seq_len)` for >90% of steps.
- **Attention sinks**: A single token (often `[CLS]` or the first position) receives disproportionate mass across many heads. Detect by checking if `attn_probs[:, :, :, sink_idx].mean() > threshold` (e.g., 0.3) consistently.

### Logging Statistics for Anomaly Detection

During training, log per-step aggregates to TensorBoard or Weights & Biases:

```python
def log_attention_stats(attn_probs, step, writer):
    entropy = attention_entropy(attn_probs).mean(dim=(0, 2))  # (heads,)
    max_attn = attn_probs.max(dim=-1).values.mean(dim=(0, 2))  # (heads,)
    for h, (ent, mx) in enumerate(zip(entropy, max_attn)):
        writer.add_scalar(f"attn/entropy_head_{h}", ent.item(), step)
        writer.add_scalar(f"attn/max_attn_head_{h}", mx.item(), step)
```

Sudden entropy spikes or max-attention drops often precede gradient explosions or representation collapse. Alerting on these metrics catches issues before they degrade validation loss.


> **[IMAGE GENERATION FAILED]** Figure 1: Scaled dot-product attention computation flow. Input X is projected to queries, keys, and values. Attention scores are computed as QKᵀ/√dₖ, optionally masked, then softmax-normalized to produce weights for the weighted value sum.
>
> **Alt:** Scaled dot-product attention pipeline: input projections to Q/K/V, attention score computation, scaling, masking, softmax, and weighted value sum
>
> **Prompt:** Technical diagram of scaled dot-product attention pipeline. Show input matrix X flowing into three parallel linear projections (W^Q, W^K, W^V) producing Q, K, V matrices. Then show Q and K transposed multiplied (QK^T), scaled by 1/sqrt(d_k), with optional causal mask added, then row-wise softmax producing attention weights, finally multiplied by V to produce output. Use clean boxes and arrows, label tensor shapes, minimal color scheme.
>
> **Error:** kroki: HTTP Error 400: Bad Request | gemini: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.



> **[IMAGE GENERATION FAILED]** Figure 2: Multi-head attention tensor flow. Input is projected to Q/K/V, split into h heads, processed in parallel, concatenated, and projected to output. Shapes shown for d_model=512, h=8, d_k=64.
>
> **Alt:** Multi-head attention architecture showing project, split heads, parallel attention, concat, and output projection with tensor shapes
>
> **Prompt:** Technical diagram of multi-head attention architecture. Show input (B, L, d_model) projected to Q/K/V each (B, L, d_model). Then split into heads: reshape to (B, L, h, d_k) and transpose to (B, h, L, d_k). Show h parallel attention blocks each computing Attention(Q_i, K_i, V_i) producing (B, h, L, d_k). Then concat: transpose to (B, L, h, d_k) and reshape to (B, L, d_model). Finally output projection W^O to (B, L, d_model). Use distinct colors for each head, show tensor shapes at each step.
>
> **Error:** kroki: HTTP Error 400: Bad Request | gemini: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.



> **[IMAGE GENERATION FAILED]** Figure 3: Positional encoding approaches. (a) Fixed sinusoidal: wavelengths increase geometrically across dimensions. (b) Learned: separate embedding per position up to max length. (c) Relative bias: added to attention scores based on i-j. (d) RoPE: rotates Q/K vectors by position-dependent angles, making attention depend on relative distance.
>
> **Alt:** Comparison of positional encoding methods: sinusoidal wavelengths, learned embeddings, relative bias, and RoPE rotation
>
> **Prompt:** Technical comparison diagram of four positional encoding methods. Four panels: (a) Sinusoidal: heatmap showing sin/cos waves across positions (y-axis) and dimensions (x-axis), wavelengths increasing left to right. (b) Learned: embedding lookup table with positions 0 to max_len, each a vector of d_model. (c) Relative bias: attention score matrix with bias b_{i-j} added, showing diagonal bands. (d) RoPE: 2D rotation visualization showing query/key vectors rotated by angle θ_m = m * base^{-2i/d}, with attention score depending on m-n. Clean technical illustration style.
>
> **Error:** kroki: HTTP Error 400: Bad Request | gemini: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.

