# Self-Attention: From Fundamentals to Million-Token Contexts

## What Self-Attention Actually Computes

The scaled dot-product attention operation is the core of every transformer. Given an input tensor **X** of shape `(batch, seq_len, d_model)`, we first project it to queries, keys, and values:

```python
# PyTorch-style pseudocode
B, T, D = X.shape          # batch, sequence length, model dimension
H, Dh = num_heads, D // num_heads

Q = X @ Wq   # (B, T, D) -> (B, T, D)
K = X @ Wk
V = X @ Wv

# Split heads: (B, T, H, Dh) -> (B, H, T, Dh)
Q = Q.view(B, T, H, Dh).transpose(1, 2)
K = K.view(B, T, H, Dh).transpose(1, 2)
V = V.view(B, T, H, Dh).transpose(1, 2)
```

Attention scores are computed per head:

```
S = Q @ K.transpose(-2, -1)          # (B, H, T, T)
S = S / sqrt(Dh)                     # scaling
A = softmax(S, dim=-1)               # (B, H, T, T)
O = A @ V                            # (B, H, T, Dh)
```

**Why `1/√Dh`?**  
Without scaling, the dot product `QKᵀ` has variance proportional to `Dh`. Large magnitudes push softmax into saturation, yielding near-zero gradients. Dividing by `√Dh` keeps the variance ~1 regardless of head dimension, preserving gradient flow in deep stacks.

### Self-Attention vs. Cross-Attention
- **Self-attention**: `Q = K = V = X` (same source). Used in encoder layers and decoder self-attention.
- **Cross-attention**: `Q` comes from decoder state, `K` and `V` from encoder output. In an encoder-decoder translator, the decoder attends to the full source sentence at each generation step.

### Minimal Numerical Example
| Parameter | Value |
|-----------|-------|
| batch     | 1     |
| seq_len   | 4     |
| d_model   | 8     |
| num_heads | 2     |
| head_dim  | 4     |

Shapes at each step:
1. Input `X`: `(1, 4, 8)`
2. `Q, K, V` after projection: `(1, 4, 8)`
3. After head split + transpose: `(1, 2, 4, 4)`
4. Scores `QKᵀ`: `(1, 2, 4, 4)`
5. After softmax: `(1, 2, 4, 4)`
6. Output `AV`: `(1, 2, 4, 4)`
7. Merge heads (transpose + reshape): `(1, 4, 8)`

This exact tensor flow—projection → head split → batched matmul → merge—is what every optimized kernel (FlashAttention, Ring Attention, USP) must preserve while minimizing HBM traffic.

## Multi-Head Attention: Why Multiple Heads Help

Multi-head attention splits the model dimension `d_model` into `h` independent heads, each with its own learned projection matrices. For a single head `i`, the queries, keys, and values are computed as:

```
Q_i = X W_i^Q,  K_i = X W_i^K,  V_i = X W_i^V
```

where `W_i^Q, W_i^K, W_i^V ∈ ℝ^{d_model × d_k}` and `d_k = d_v = d_model / h`. This forces each head to operate in a lower-dimensional subspace, reducing per-head computation while allowing parallel processing.

Empirical studies (notably the original *Attention Is All You Need* visualizations) reveal that different heads specialize in distinct linguistic patterns:
- **Syntactic heads** attend to subject–verb agreement, dependency relations, and local phrase structure.
- **Semantic heads** capture long-range coreference, topic coherence, and entity tracking.
- **Positional heads** focus on relative distance or absolute position cues.
- **Global heads** aggregate information across the entire sequence.

After computing attention independently per head, the outputs are concatenated and projected back to `d_model`:

```
MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O
```

where `W^O ∈ ℝ^{h·d_v × d_model}`. This final projection mixes information across subspaces, enabling the model to combine diverse relational signals.

**Trade‑off:** Increasing `h` expands representational capacity (more specialized subspaces) but shrinks `d_k`, limiting the expressiveness of each head’s attention scores. Modern LLMs balance this by fixing `d_k ≈ 128` and scaling `h` with model size. For example, LLaMA‑3 uses 32 heads with `d_k = 128` (`d_model = 4096`), while larger variants (70B) increase `h` to 64 while keeping `d_k` constant. This ratio preserves per-head fidelity while providing enough heads for diverse attention patterns.

## The Quadratic Bottleneck: Complexity and Memory Analysis

### FLOP Breakdown

Naive self-attention computes three distinct kernels per layer. For sequence length *n* and head dimension *d*:

1. **QKᵀ**: Batched matrix multiply of *Q* (n×d) and *Kᵀ* (d×n) → **2n²d FLOPs**  
2. **Softmax**: Row-wise exponentiation, reduction, and division on the n×n score matrix → **~3n² FLOPs**  
3. **Attention @ V**: Multiply the n×n probability matrix by *V* (n×d) → **2n²d FLOPs**

Total per head ≈ **4n²d + 3n²**. With *h* heads, multiply by *h*. For a typical configuration (n=128k, d=64, h=32), the dominant 4n²dh term yields **~130 TFLOPs per layer** — a prohibitive cost for long contexts.

### Memory Analysis

The full n×n attention matrix must be materialized for the backward pass. In bfloat16 (2 bytes/element), storage per head is **2n² bytes**. At n=32k with 32 heads, this alone consumes **~8 GB per layer** just for attention scores, before accounting for Q, K, V, or gradients. Activation memory thus scales quadratically, quickly exceeding HBM capacity on a single GPU.

### Training vs. Autoregressive Inference

- **Training**: The entire QKᵀ matrix is instantiated and stored for gradient computation. Both time and memory are O(n²).  
- **Inference**: The KV cache grows **linearly** (O(n)) across decoding steps. However, the **prefill phase** still requires a full quadratic attention over the prompt, creating a latency spike proportional to prompt length².

### The Memory Wall

Modern GPUs deliver ~100 TFLOPs (bf16) but only ~2–3 TB/s HBM bandwidth. Naive attention streams the n×n matrix from HBM multiple times (read Q,K → write scores → read scores,V → write output), making it **bandwidth-bound**. Flash Attention’s IO-aware tiling fuses the three kernels, keeps intermediate blocks in SRAM, and reduces HBM traffic by **2–4×**, turning the memory wall from a hard limit into a tunable parameter.

## Efficient Attention Alternatives: Linear, Sparse, and Hybrid Approaches

Sub-quadratic attention mechanisms fall into three broad families. Kernel-based linear attention methods (e.g., LoLCATs, GateLoop) rewrite the softmax as a kernel feature map to achieve O(n) time and memory. Sparse pattern approaches (e.g., BigBird, Longformer) restrict the attention graph to local windows, global tokens, or random connections, yielding O(n log n) or O(n) complexity depending on the pattern. State-space hybrids (e.g., Mamba, Hymba) blend selective state-space models with attention layers to capture both long-range dependencies and precise retrieval. ([Source](https://medium.com/@dr.teck/efficient-alternatives-to-transformer-self-attention-397851f324ab))

Theoretical complexity differs markedly: full attention scales as O(n²), sparse patterns as O(n log n) (or O(n) with fixed window sizes), and linear/SSM methods as O(n). In practice, constant factors and hardware utilization often dominate; FlashAttention-3 shows that exact O(n²) attention can outperform approximate O(n) kernels at moderate lengths due to better GPU occupancy. ([Source](https://www.together.ai/blog/flashattention-3))

Empirical trade-offs are significant. Linear attention struggles with recall-intensive tasks such as needle-in-a-haystack retrieval because the kernel approximation compresses information lossily. Sparse attention requires careful pattern design—poorly chosen global tokens or window sizes degrade perplexity on long-range dependencies. Hybrids like Hymba aim to combine SSM’s efficient recurrence with attention’s precise lookup, but their training stability and hyperparameter sensitivity remain active research areas. Not found in provided sources.

**Decision framework.**  
- **Sequence length < 4k**: standard or FlashAttention-3 exact attention is simplest and most accurate.  
- **4k–128k, retrieval-heavy**: sparse patterns (Longformer-style sliding window + global tokens) preserve exact attention for key positions.  
- **>128k, generation-dominant**: linear attention or SSM hybrids (Mamba-2, Hymba) reduce memory pressure; pair with Ring Attention or USP for multi-GPU scaling.  
- **Hardware**: on H100/B200, FlashAttention-3’s low-precision GEMMs and async pipelining often beat approximate kernels up to 32k; beyond that, kernel-based linear attention or Ring-parallelized sparse attention become necessary. Not found in provided sources.

## Flash Attention: Kernel Fusion and IO-Aware Tiling

The core insight behind Flash Attention is computing softmax online using running statistics—specifically, maintaining a running maximum and sum per row—so the full $S = QK^T$ matrix never materializes in HBM ([Source](https://dasroot.net/posts/2025/12/flash-attention-longer-context-gpu-acceleration)). Instead of writing $S$ to global memory, the kernel fuses the $QK^T$, scaling, masking, softmax, and weighted-sum steps into a single pass. This eliminates the $O(N^2)$ HBM traffic that dominates standard attention.

The tiling strategy loads blocks of $Q$, $K$, and $V$ from HBM into on-chip SRAM (shared memory). For each query block, the kernel iterates over key/value blocks, computes partial attention scores, updates the running max and sum, and accumulates the output. By keeping the working set in SRAM, Flash Attention reduces HBM reads/writes by roughly 4× compared to a naive implementation ([Source](https://github.com/Dao-AILab/flash-attention)).

FlashAttention-2 introduces better work partitioning across thread blocks and warps, reducing shared-memory pressure and improving occupancy. These changes yield approximately 2× speedup over the original Flash Attention on the same hardware ([Source](https://www.linkedin.com/posts/dmitriikovrizhnukh_flash-attention-is-an-algorithmic-optimization-activity-7395444718605635584-DmDv), [Source](https://github.com/Dao-AILab/flash-attention)).

FlashAttention-3 adds producer-consumer asynchrony and warp specialization: dedicated warps handle GEMM (via tensor cores), while others manage data movement and softmax reduction. It also supports FP8 accumulation, reaching ~1.2 PFLOPS on H100 GPUs ([Source](https://www.together.ai/blog/flashattention-3)).

In PyTorch 2.2+, the optimized kernel is exposed via `torch.nn.functional.scaled_dot_product_attention` with `attn_implementation='flash_attention_2'`. The backend auto-enables when inputs are on CUDA, have supported dtypes (fp16/bf16), and meet alignment constraints (sequence length multiples of 16, head dimension ≤ 128). For other cases, PyTorch falls back to the memory-efficient or math kernels.

## Scaling to Million-Token Contexts: Ring Attention and Sequence Parallelism

Ring Attention distributes the attention computation across GPUs by arranging them in a logical ring. Each GPU holds a block of keys and values (KV) for a segment of the sequence. During the forward pass, a GPU computes its local query–key dot products (Q@K) while simultaneously receiving the next KV block from its neighbor and sending its own KV block onward. This overlap of communication (NVLink/PCIe transfers) with computation (matrix multiplies) hides latency and enables scaling to contexts far beyond single‑GPU memory ([Source](https://github.com/Ascend/MindSpeed-LLM/blob/master/docs/en/pytorch/features/mcore/ring-attention-context-parallel.md)).

Unified Sequence Parallelism (USP) combines two orthogonal parallelism dimensions: **ulysses_degree** for head‑wise (tensor) parallelism and **ring_degree** for sequence (context) parallelism. For example, on an 8‑GPU L20 node one might configure `ulysses=2` (splitting heads across 2 GPUs) and `ring=4` (splitting the sequence across 4 GPUs), yielding a hybrid mesh that balances memory and compute. USP generalizes DeepSpeed‑Ulysses and Ring Attention into a single framework, allowing flexible trade‑offs between head and sequence partitioning ([Source](https://github.com/feifeibear/long-context-attention)).

Recent 2026 benchmarks show a crossover around **500K–1M tokens** where NVMe offloading (e.g., FlashAttention‑3 with SSD paging) becomes less efficient than pure sequence parallelism. Beyond this threshold, Ring‑style context parallelism delivers lower latency and higher throughput. Additionally, **chunked prefill** — splitting the prefill phase into smaller chunks that fit in shared memory — optimizes time‑to‑first‑token (TTFT) for ultra‑long prompts by reducing peak memory pressure and enabling better pipeline overlap ([Source](https://www.spheron.network/blog/ring-attention-tree-attention-sequence-parallelism-gpu-cloud)).

Standard Ring Attention uses a unidirectional ring, leaving half of the NVLink bandwidth idle during KV block transfers. **TokenRing** introduces bidirectional communication: each GPU simultaneously sends and receives KV blocks on two independent links, effectively doubling bandwidth utilization. This reduces the communication step from *O(N)* to *O(N/2)* for *N* GPUs and improves scaling efficiency, especially on fully connected NVLink topologies ([Source](https://arxiv.org/html/2412.20501v1)).

## Practical Implementation Checklist: Debugging, Numerics, and Profiling

### Numerics: FP16 Softmax Stability
Half-precision softmax can overflow/underflow during the `exp` step, producing NaNs or zero gradients. Always accumulate in FP32:

```python
# Manual stable softmax (reference)
def stable_softmax(x, dim=-1):
    x_fp32 = x.float()
    x_max = x_fp32.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_fp32 - x_max)
    return (exp_x / exp_x.sum(dim=dim, keepdim=True)).to(x.dtype)
```

Flash Attention 2/3 implements an **online softmax** that fuses the max/exp/sum reduction in a single pass, keeping intermediates in FP32 without materializing the full `S = QK^T` matrix. Verify numerics by printing high-precision tensors:

```python
torch.set_printoptions(precision=10, sci_mode=False)
print(attn_weights[0, 0, :5, :5])  # inspect first head, first 5x5 block
```

### Masking Bugs: Shape Mismatches
Causal masks must broadcast across batch and heads. The canonical shape is `(1, 1, seq_len, seq_len)`:

```python
seq_len = 2048
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
    diagonal=1
).view(1, 1, seq_len, seq_len)  # [1, 1, L, L]
```

Common pitfalls:
- Using `(batch, heads, seq_len, seq_len)` wastes memory and breaks kernel fusion.
- Forgetting `diagonal=1` (allows attending to current token) or `diagonal=0` (blocks it).
- Mixing additive (`-inf`) vs multiplicative (`0/1`) masks — Flash Attention expects additive.

### Profiling: Kernel Selection Visibility
Use the PyTorch profiler to confirm which attention backend actually runs:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    out = model(input_ids)

# Filter for attention kernels
for evt in prof.key_averages():
    if "attn" in evt.key.lower() or "flash" in evt.key.lower():
        print(evt.key, evt.cuda_time_total / 1000, "ms", evt.cuda_memory_usage / 1e6, "MB")
```

Look for `flash_attn_fwd` / `flash_attn_bwd` (Flash Attention), `efficient_attention_forward` (xFormers), or `aten::_scaled_dot_product_attention` (math fallback). A math fallback on H100/A100 usually indicates misaligned tensor shapes, unsupported `attn_mask` dtype, or `enable_flash=False`.

### KV Cache Management
Pre-allocate contiguous buffers once per sequence length bucket:

```python
# During model init
max_batch = 8
max_seq = 8192
num_heads = 32
head_dim = 128
kv_cache = torch.empty(
    2, max_batch, num_heads, max_seq, head_dim,
    dtype=torch.bfloat16, device="cuda"
).contiguous()  # [2, B, H, L, D] for K and V
```

At inference time:
- Slice `kv_cache[:, :batch, :, :cur_len]` for current step.
- For beam search, expand batch dimension *before* writing; avoid `repeat_interleave` on the cache itself.
- Monitor fragmentation with `torch.cuda.memory_summary()` — look for "segmentation" spikes after many decode steps. If fragmentation exceeds ~15%, consider a periodic `torch.cuda.empty_cache()` or a custom allocator (e.g., `torch.cuda.CUDACachingAllocator`).

## Performance and Cost Trade-offs: Choosing the Right Attention for Your Workload

| Sequence Length | Recommended Approach | Key Optimizations |
|-----------------|----------------------|-------------------|
| < 4k | Standard FlashAttention-2 | Fused kernels, shared-memory tiling |
| 4k–128k | FA-2 + KV cache optimization | Paged KV cache, chunked prefill |
| 128k–1M | Ring Attention / USP | Sequence parallelism, bidirectional ring |
| > 1M | Hybrid (offload + sequence parallel) | CPU/GPU offload, hierarchical parallelism |

**Training cost at 128k context (7B model).** The Spheron 2026 guide reports that FlashAttention-2 on 8×H100 requires ~1,200 H100-hours for a 7B model at 128k context, while Ring Attention with 8-way context parallelism reduces this to ~850 H100-hours by overlapping communication and computation ([Source](https://www.spheron.network/blog/ring-attention-tree-attention-sequence-parallelism-gpu-cloud)). The crossover occurs near 64k tokens: below that, FA-2’s lower overhead wins; above, Ring Attention’s linear scaling dominates.

**Inference considerations.** Chunked prefill splits long prompts into smaller blocks to bound time-to-first-token (TTFT) while maintaining throughput. On H100, a 32k prefill chunked at 4k reduces peak memory by 4× with <5% latency penalty. Speculative decoding interacts with attention kernels by requiring fast single-token attention; FlashAttention-3’s async copy and FP8 support accelerate the draft model’s attention, but the verification step still benefits from FA-2’s exact kernel ([Source](https://www.together.ai/blog/flashattention-3)).

**Variable-length sequences.** Padding to max length in a batch wastes compute and memory. Flash Attention v2.5+ introduces ragged tensor support, allowing per-sequence lengths without padding. Benchmarks on the Dao-AILab repo show 1.3–1.8× speedup for batches with high length variance ([Source](https://github.com/Dao-AILab/flash-attention)). If ragged tensors are unavailable, bucketing sequences by length and padding within buckets is the next best option. Not found in provided sources for exact v2.5+ API details.
