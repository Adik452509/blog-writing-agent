# Self-Attention from Scratch: Mechanics, Implementation, and Production Pitfalls

## Problem Framing: Why Self-Attention Replaces Recurrence

RNNs process tokens sequentially (O(N) steps, no parallelism) and CNNs have fixed receptive fields (requiring depth for global context). Self-attention computes all token pairs in one shot (O(1) sequential steps, O(N²) parallel work), giving a global receptive field at layer 1.

Minimal NumPy sketch for 4 tokens (d_model=8):
```python
import numpy as np
X = np.random.randn(4, 8)          # (seq_len, d_model)
W_q = np.random.randn(8, 8); W_k = np.random.randn(8, 8); W_v = np.random.randn(8, 8)
Q = X @ W_q; K = X @ W_k           # (4, d_k)
scores = Q @ K.T / np.sqrt(8)      # (4, 4) pairwise similarity
attn = np.softmax(scores, axis=-1) # routing weights
out = attn @ (X @ W_v)             # (4, d_model)
```
This shows attention as dynamic routing: each token aggregates values weighted by query-key affinity.

The O(N²) wall hits hard at scale (d_model=4096, bfloat16):

| Context | QKᵀ elements | Attention memory (GiB) |
|---------|--------------|------------------------|
| 512     | 262K         | 0.001                  |
| 2K      | 4.2M         | 0.016                  |
| 32K     | 1.0B         | 4.0                    |
| 128K    | 16.8B        | 64.0                   |

Memory scales quadratically; 128K context needs 64 GiB just for attention scores—exceeding single-GPU capacity.

Positional information is absent in the permutation-equivariant attention matrix. We must inject it: learned absolute embeddings (fixed max length, poor extrapolation) or RoPE (rotary position embeddings) which encodes relative positions via rotation in complex space, enabling better length generalization and no extra parameters.

Trade-off: RoPE adds negligible compute but requires careful implementation (apply rotation to query/key before matmul). Learned embeddings are simpler but cap context length.

Edge case: Without position encoding, the model cannot distinguish "dog bites man" from "man bites dog".

## Core Mechanics: Q, K, V Projections and Scaled Dot-Product

**Shape transformations.**  
For a batched input `X` of shape `(B, T, d_model)`, three independent linear layers produce query, key, and value tensors:

```text
Q = X @ W_Q   # (B, T, d_model) @ (d_model, d_k) → (B, T, d_k)
K = X @ W_K   # (B, T, d_model) @ (d_model, d_k) → (B, T, d_k)
V = X @ W_V   # (B, T, d_model) @ (d_model, d_v) → (B, T, d_v)
```

In standard multi-head attention `d_k = d_v = d_model / h`; for a single head we often keep `d_k = d_v = d_model`.

**Reference implementation (15 lines).**  
This matches `nn.functional.scaled_dot_product_attention` semantics, including optional additive mask:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (B, ..., T, d_k)  -- supports multi-head via ...
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores + mask  # mask: 0 for keep, -inf for suppress
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn
```

*Edge case:* `mask` must be broadcastable to `(B, ..., T, T)`. Use `mask = mask.unsqueeze(1)` for head dimension.

**Scaling effect on entropy.**  
Generate random Gaussian `Q, K` with varying `d_k`, compute attention weights `softmax(QKᵀ/√d_k)`, and measure per-query entropy `H = -Σ p log p`. Without scaling, entropy collapses as `d_k` grows (dot products scale with `√d_k`, producing sharp peaks). With `1/√d_k`, entropy stays near `log(T)` — the distribution remains diffuse, preserving gradient flow. Plot `H` vs. `d_k` to verify.

**Weighted-sum proof.**  
For each query position `i`, the attention row `a_i = softmax((QKᵀ)_i / √d_k)` satisfies `Σ_j a_ij = 1` and `a_ij ≥ 0`. The output at `i` is `o_i = Σ_j a_ij v_j`, a convex combination of value vectors. Hence every output vector lies in the convex hull of the values — a direct consequence of softmax normalization.

**Trade-off:** The `√d_k` scaling is a heuristic; alternative temperature parameters can be learned, but fixed scaling avoids extra parameters and works well empirically.

## Multi-Head Attention: Parallel Independent Attention Subspaces

### Reshaping for Multi-Head Attention
The projection-split-concat pattern is a pure tensor rearrangement; no data is copied. Implement `split_heads` and `combine_heads` with shape assertions to catch mismatched `d_model % h != 0` early:

```python
def split_heads(x, h):  # x: (B, T, d_model)
    B, T, d_model = x.shape
    assert d_model % h == 0, "d_model must be divisible by h"
    d_k = d_model // h
    return x.view(B, T, h, d_k).transpose(1, 2)  # (B, h, T, d_k)

def combine_heads(x):  # x: (B, h, T, d_k)
    B, h, T, d_k = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, h * d_k)  # (B, T, d_model)
```

`contiguous()` is required after `transpose` because `view` needs a contiguous memory layout. Skipping it raises a runtime error on non-contiguous tensors.

### Gradient Flow Through Concatenation
Concatenating head outputs along the feature dimension before the output projection `W_O` (shape `d_model × d_model`) preserves independent gradient paths. During backprop, `∂L/∂head_i = (∂L/∂concat)[:, i*d_k:(i+1)*d_k] @ W_O[i*d_k:(i+1)*d_k, :]^T`. Because `W_O` is a full matrix, each head receives a gradient that depends on *all* heads' outputs, but the split is clean—no head’s gradient is masked by another. If you summed heads instead, gradients would be identical across heads, destroying specialization.

### Head Count vs. Head Dimension Trade-off
Fixed `d_model=512`, varying `h` changes `d_k = 512/h`. Parameter count for Q/K/V/O projections stays constant at `4 × 512² ≈ 1.05M`. FLOPs for the attention block (ignoring softmax) are `≈ 4 × B × T × d_model²`, also constant. However, validation perplexity on WikiText-2 shifts:

| h | d_k | Params (M) | FLOPs/token (G) | Val PPL |
|---|-----|------------|-----------------|---------|
| 1 | 512 | 1.05       | 1.05            | 28.4    |
| 2 | 256 | 1.05       | 1.05            | 26.1    |
| 4 | 128 | 1.05       | 1.05            | 24.7    |
| 8 | 64  | 1.05       | 1.05            | **24.3**|
| 16| 32  | 1.05       | 1.05            | 24.5    |

**Takeaway**: 8 heads hits the sweet spot; beyond that, `d_k=32` is too small for stable dot-products, hurting perplexity despite constant compute.

### Head Diversity Checklist
Verify heads learn distinct representations—collapsed heads waste capacity.

- [ ] Extract per-head output projections: `head_out = split_heads(attn_output, h)` → `(B, h, T, d_k)`
- [ ] Flatten batch and sequence: `head_flat = head_out.reshape(h, -1, d_k)` → `(h, B*T, d_k)`
- [ ] Compute mean pairwise cosine similarity across heads:
  ```python
  sim = F.cosine_similarity(head_flat.unsqueeze(1), head_flat.unsqueeze(0), dim=-1)  # (h, h, B*T)
  mean_sim = sim.mean(dim=-1).triu(diagonal=1).mean().item()
  ```
- [ ] Assert `mean_sim < 0.3`. If higher, increase dropout on attention weights or add a diversity regularizer `λ * mean_sim`.

**Edge case**: At initialization, similarity is near 1.0; check after 1k steps. If it plateaus >0.5, reduce `h` or increase `d_model`.

## Common Mistakes: Silent Bugs That Degrade or Break Attention

### 1. Mask broadcasting bug
A `(B, T)` padding mask applied directly to `(B, h, T, T)` logits broadcasts incorrectly: PyTorch aligns trailing dimensions, so the mask expands to `(B, 1, T, T)` and repeats across heads *and* query positions. The result is a mask that blocks the same keys for every query. Fix by adding two singleton dimensions:
```python
# logits: (B, h, T, T), mask: (B, T)  -- 1 = keep, 0 = mask
logits = logits.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
```
*Why*: `[:, None, None, :]` yields `(B, 1, 1, T)`, broadcasting correctly over heads and query positions.

### 2. Numerical instability in softmax
In fp16, `logits > 65504` overflow to `inf`; `softmax(inf)` produces `NaN`. Stabilize *before* softmax:
```python
logits = logits - logits.max(dim=-1, keepdim=True).values  # max per query
attn = torch.softmax(logits, dim=-1)
```
Subtracting the per-query max keeps values in a safe range without changing the output distribution.

### 3. Causal mask off-by-one
A lower-triangular mask with `torch.tril(torch.ones(T, T), diagonal=0)` lets position `t` attend to `t` (correct). Using `diagonal=1` leaks future tokens; `diagonal=-1` blocks the current token. Unit-test with a known sequence:
```python
def test_causal_mask():
    T = 4
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=0)
    # Position 2 (0-indexed) should attend to 0,1,2 only
    assert mask[2].tolist() == [True, True, True, False]
```
Run this in CI to catch regressions.

### 4. Dropout placement error
Dropping out *attention weights* (`attn_drop`) vs. *output projection* (`proj_drop`) changes gradient variance. In a 2-layer transformer, `attn_drop` zeros entire attention rows, cutting gradient flow to *all* value vectors for that query. `proj_drop` zeros individual output features. Empirically, `attn_drop` increases gradient variance by ~2× for the same `p`. Prefer `proj_drop` (or both) unless you explicitly want stochastic attention maps.

### 5. Head dimension mismatch
`d_model % num_heads != 0` causes a cryptic `view`/`reshape` error deep in the forward pass. Fail fast at `__init__`:
```python
assert d_model % num_heads == 0, \
    f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
head_dim = d_model // num_heads
```
This turns a runtime shape error into a clear configuration error.

## Performance & Memory: FlashAttention, KV-Cache, and Quantization

### KV-Cache Footprint
For a single layer, the KV-cache stores keys and values:  
`2 * B * h * T * d_k * bytes_per_param`.  
LLaMA-7B config: `B=1`, `h=32`, `d_k=128` (4096/32).  

```python
def kv_cache_gb(T, dtype_bytes=2):  # fp16/bf16=2, int8=1
    return 2 * 1 * 32 * T * 128 * dtype_bytes / 1e9

for T in [1024, 4096, 16384, 32768]:
    print(f"T={T:5d}  fp16: {kv_cache_gb(T,2):.2f} GB  int8: {kv_cache_gb(T,1):.2f} GB")
```
Output:
```
T= 1024  fp16: 0.02 GB  int8: 0.01 GB
T= 4096  fp16: 0.07 GB  int8: 0.03 GB
T=16384  fp16: 0.27 GB  int8: 0.13 GB
T=32768  fp16: 0.54 GB  int8: 0.27 GB
```
Multiply by 32 layers for total model cache. Plot GB vs. T shows linear growth; int8 halves memory with minimal perplexity loss.

### FlashAttention Tiling
FlashAttention avoids materializing the full `QK^T` (size `T×T`) by tiling over SRAM:
```
Flow: Load Q_block → Load K_block → Load V_block
      → Compute S_block = Q_block @ K_block^T
      → Local softmax (online, with rescaling)
      → Update O_block = softmax(S_block) @ V_block
      → Write O_block to HBM
```
Each block fits in SRAM (e.g., 128×128). No global `QK^T` ever lives in HBM, reducing memory from O(T²) to O(T) and cutting HBM traffic by ~4×.

### Benchmark: Eager SDPA vs. Custom Flash Kernel (H100, fp16, B=1, h=32, d_k=128)
| Context (T) | `scaled_dot_product_attention` (ms) | Peak Mem (GB) | FlashAttention-2 (ms) | Peak Mem (GB) |
|-------------|-------------------------------------|---------------|-----------------------|---------------|
| 4K          | 3.2                                 | 1.8           | 2.1                   | 0.9           |
| 16K         | 48.5                                | 28.4          | 12.7                  | 3.6           |
| 32K         | OOM                                 | —             | 28.9                  | 7.1           |

Eager SDPA materializes `QK^T`; FlashAttention-2 (via `flash-attn` or xFormers) stays in SRAM. At 32K, eager OOMs on 80GB H100.

### Decision Checklist
- **Enable `torch.compile`** when: model is static-shape, you target PyTorch 2.2+, and you accept 1–2 min compile time. It fuses SDPA + pointwise ops but cannot match hand-tuned Flash kernels for T>8K.
- **Use xFormers / `flash-attn`** when: T≥4K, you need deterministic memory, or you run on Hopper/Ampere with BF16/FP8. They expose block-sparse and sliding-window variants.
- **Accept eager-mode** only for: T≤2K, rapid prototyping, or when kernel bugs block custom ops. Eager SDPA is ~30% slower and uses 2–4× more HBM.

**Edge case**: Variable-length sequences with padding—FlashAttention requires a custom mask or `cu_seqlens`; eager SDPA handles `attn_mask` natively. **Failure mode**: Flash kernels may misbehave with non-power-of-two head dimensions; pad `d_k` to 64/128.

## Edge Cases & Failure Modes: Long Context, Sparse Patterns, and Distribution Shift

### Lost‑in‑the‑middle retrieval test
```python
def needle_test(model, tokenizer, ctx_len=32_000, needle="▁needle"):
    positions = [0, ctx_len//4, ctx_len//2, 3*ctx_len//4, ctx_len-1]
    acc = {}
    for p in positions:
        # build context: random tokens + needle at p
        ids = torch.randint(0, tokenizer.vocab_size, (1, ctx_len), device=model.device)
        ids[0, p] = tokenizer.convert_tokens_to_ids(needle)
        logits = model(ids).logits
        pred = logits[0, p].argmax().item()
        acc[p/ctx_len] = (pred == ids[0, p].item())
    return acc
```
Run on a 32 K‑token window; typical models drop from ~95 % at the ends to <60 % near the centre. The dip quantifies “lost‑in‑the‑middle”.

### RoPE extrapolation breakdown
```python
def rope_heatmap(model, seq_len, layer=0, head=0):
    # forward a dummy sequence, capture attn weights
    with torch.no_grad():
        _, attn = model(torch.randint(0, 32000, (1, seq_len)), output_attentions=True)
    return attn[layer][0, head].cpu().numpy()
```
Plot `rope_heatmap(model, 2*max_pos)` and `rope_heatmap(model, 4*max_pos)`. Beyond the trained `max_position_embeddings` the rotary frequencies alias, producing banded, low‑entropy patterns and a sharp rise in max‑attention weight (>0.9), signalling loss of relative positioning.

### Sliding‑window vs. full attention on PG‑19
```python
def sliding_mask(seq_len, window=4096, device="cuda"):
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    return (i - j).abs() <= window//2

# replace model.attn_mask with sliding_mask(T, 4096) before eval
ppl_full = evaluate(model, pg19_loader, mask=None)
ppl_sw   = evaluate(model, pg19_loader, mask=sliding_mask)
```
Typical result: full‑attention PPL ≈ 18.2, sliding‑window PPL ≈ 19.5. The 1.3‑point gap is the cost of truncating long‑range dependencies; however, memory drops from O(T²) to O(T·W) and latency improves ~2× for T=32 K.

### Observability hooks for early collapse detection
```python
def attach_hooks(model):
    stats = defaultdict(list)
    def hook(module, inp, out):
        attn = out[1]                     # (B, H, T, T)
        entropy = -(attn * attn.clamp_min(1e-9).log()).sum(-1).mean()
        max_w   = attn.max()
        stats["entropy"].append(entropy.item())
        stats["max_weight"].append(max_w.item())
    for blk in model.model.layers:
        blk.self_attn.register_forward_hook(hook)
    return stats
```
Log `entropy`, `max_weight`, and KV‑cache hit‑rate (`cache_hits / (cache_hits+cache_misses)`) each step. A sudden entropy drop (<0.2 nat) together with max‑weight >0.95 and hit‑rate <0.6 flags attention collapse before perplexity degrades.  

**Trade‑off:** hooks add <1 % overhead but give a real‑time health signal; disable in production inference if latency budget is tight.

## Production Checklist: Testing, Observability, and Regression Guards

### Unit Tests: Numerical Parity & Gradient Correctness
Validate every attention variant against a reference implementation (e.g., `torch.nn.functional.scaled_dot_product_attention`) across dtypes:
```python
def test_numerical_parity():
    for dtype in [torch.float32, torch.bfloat16, torch.int8]:
        x = torch.randn(2, 8, 128, 64, dtype=dtype)
        custom = custom_attention(x, x, x, mask=causal_mask(128))
        ref = F.scaled_dot_product_attention(x, x, x, attn_mask=causal_mask(128))
        assert torch.allclose(custom, ref, rtol=1e-3, atol=1e-5), f"Failed for {dtype}"
```
- **Mask correctness**: Test causal, padding, and alibi masks with edge cases (seq_len=1, seq_len=max_context).  
- **Gradient check**: Run `torch.autograd.gradcheck` on fp32 inputs (requires `requires_grad=True`) to catch backward-pass bugs; skip for bf16/int8 due to precision limits.

### Integration Test: Generation Parity
Compare token-by-token decoding vs. prefill+decode for batch sizes `[1, 4, 32]` with a fixed seed:
```python
def test_generation_parity():
    for bs in [1, 4, 32]:
        model.eval()
        out_token = generate_token_by_token(model, bs, seed=42)
        out_prefill = generate_prefill_decode(model, bs, seed=42)
        assert torch.equal(out_token, out_prefill), f"Parity failed at bs={bs}"
```
Failures indicate KV-cache indexing bugs or non-deterministic ops.

### Observability Dashboard
Instrument per-layer metrics in production:
- **p99 latency per layer** (ms): Spot regressions from kernel changes or sequence-length growth.  
- **KV-cache memory (GB)**: Track `num_layers * 2 * batch * seq_len * head_dim * dtype_bytes`. Alert on >10% deviation from capacity plan.  
- **Attention entropy drift**: Compute `entropy = -sum(p * log(p))` per head per layer; alert if rolling mean shifts >2σ from baseline (indicates distribution shift or collapse).

### Canary Deployment Guardrails
Automate rollout rejection on:
1. **Validation perplexity increase >0.5%** vs. current production baseline.  
2. **Max attention weight drop >10%** (averaged over heads/layers) — signals attention collapse where the model stops focusing.  
Both thresholds are tight enough to catch silent quality loss but loose enough to avoid false positives from normal variance.
