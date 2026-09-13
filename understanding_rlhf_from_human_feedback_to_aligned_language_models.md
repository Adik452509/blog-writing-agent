# Understanding RLHF: From Human Feedback to Aligned Language Models

## What Is RLHF and Why It Matters

Alignment is the problem of closing the gap between what a language model predicts (the next token) and what a human actually intends. Pretraining teaches a model to predict tokens from vast corpora, but it does not teach the model to follow instructions, refuse unsafe requests, or adopt a consistent style. This misalignment leads to outputs that are plausible but unhelpful or dangerous.

The standard pipeline to address this has three stages:

1. **Pretraining** – Learn a broad distribution of language by predicting the next token on massive text datasets.
2. **Supervised Fine‑Tuning (SFT)** – Fine‑tune the pretrained model on a curated dataset of (prompt, ideal response) pairs. This teaches the model the format and style of desired outputs, but it still mimics the training distribution rather than optimizing for human preference.
3. **RLHF** – Use reinforcement learning to directly optimize the model against a reward model that captures human preferences. The policy (the language model) is updated to maximize the reward while staying close to the SFT model (the reference policy) via a KL‑divergence constraint.

**Concrete example**: Ask a base model to “write a safe refusal for a request to build a bomb.” The base model may continue the prompt with instructions because it has seen similar completions in its training data. An SFT model might refuse but in a rigid, template‑like way. An RLHF‑tuned model learns to refuse politely, explain the policy, and offer a safe alternative—behavior that aligns with human intent.

The RLHF objective formalizes this: maximize the expected reward from the reward model \(R\) while constraining the KL divergence between the current policy \(\pi\) and the reference policy \(\pi_{\text{ref}}\) (the SFT model):

\[
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \big[ R(x, y) \big] - \beta \, \text{KL}\big(\pi(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x)\big)
\]

The KL term prevents the model from drifting too far from the fluent, coherent behavior learned during SFT, avoiding reward hacking and catastrophic forgetting.

## The Three Components: Policy, Reward Model, and Human Data

### Policy Model

The policy model generates text and is initialized from a supervised fine-tuning (SFT) checkpoint. This checkpoint already captures instruction-following behavior from high-quality demonstrations. During RLHF, you choose which layers to update:

- **Full fine-tuning**: All parameters trainable. Highest capacity but risks catastrophic forgetting of pretrained knowledge and requires massive compute.
- **LoRA/QLoRA**: Freeze backbone, train low-rank adapters (typically 0.1–1% of parameters). Preserves base capabilities, reduces VRAM, but may limit reward optimization ceiling.
- **Partial unfreezing**: Unfreeze last N transformer blocks + output head. Middle ground; common in 7B–13B models.

```python
# LoRA policy setup (conceptual)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64, lora_alpha=128, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
policy = get_peft_model(sft_model, lora_config)
policy.print_trainable_parameters()  # ~0.5% params
```

**Edge case**: Over-optimizing the policy against a flawed reward model produces "reward hacking"—verbose, repetitive, or sycophantic outputs that score well but degrade utility.

### Reward Model

The reward model (RM) assigns a scalar score to (prompt, completion) pairs. Standard architecture: frozen pretrained backbone + randomly initialized scalar head (single linear layer). Training uses pairwise comparisons from human annotators:

```
Loss = -log(σ(r_chosen - r_rejected))
```

Where `r` is the RM output. The Bradley-Terry model underpins this: P(chosen > rejected) = σ(r_chosen - r_rejected).

**Architecture choices**:
- Backbone size: Often smaller than policy (e.g., 7B RM for 70B policy) to reduce inference cost during PPO rollouts.
- Calibration: Add a temperature-scaled sigmoid or use reward normalization (running mean/std) to prevent scale drift during training.

**Failure mode**: RM overfits to annotator biases (e.g., preferring longer responses). Mitigate with length normalization and ensemble RMs.

### Human Feedback Data

Comparison data comes from annotators ranking model outputs. Common schemes:

| Scheme | Description | Cost |
|--------|-------------|------|
| Pairwise (A vs B) | Binary choice | Low |
| Best-of-N | Rank N outputs | Higher, richer signal |
| Likert scale | 1–7 quality rating | Enables regression RM |

**Annotator agreement**: Measure Krippendorff's α or Cohen's κ. Target α ≥ 0.6 for pairwise; lower indicates ambiguous prompts or poor guidelines. Low-agreement samples should be filtered or sent to experts.

### Data Quality Filters

Raw comparisons contain noise. Apply filters before RM training:

1. **Length normalization**: Penalize/reward length explicitly or truncate to fixed token budget to prevent length bias.
2. **Consistency checks**: If A > B and B > C but C > A, flag cycle for review.
3. **Adversarial filtering**: Train a probe classifier to detect annotator errors (e.g., factual contradictions, policy violations) and remove flagged pairs.
4. **Deduplication**: Near-duplicate prompts with conflicting labels indicate annotation drift.

**Performance/cost tradeoff**: Tighter filters improve RM quality but reduce dataset size. Typical retention: 60–80% after filtering. For privacy, strip PII from prompts before annotation; use synthetic prompts for sensitive domains.

**Security consideration**: Annotator interfaces can leak model capabilities or user data. Isolate annotation environments, audit access logs, and apply differential privacy if prompts contain sensitive information.

## RLHF Training Loop: PPO Step by Step

### Rollout: Generate Responses from Current Policy

Given a batch of prompts `prompts` of shape `(B, T_prompt)`, the policy π_θ generates responses token-by-token. The rollout produces:
- `responses`: `(B, T_resp)` token IDs
- `logprobs`: `(B, T_resp)` log-probabilities under π_θ
- `values`: `(B, T_resp)` value estimates from the critic head

```python
# Pseudocode: rollout
prompts = batch["input_ids"]          # (B, T_prompt)
responses, logprobs, values = policy.generate(
    prompts, max_new_tokens=T_resp, return_logprobs=True, return_values=True
)
# responses: (B, T_resp), logprobs: (B, T_resp), values: (B, T_resp)
```

**Edge case**: Truncation at `max_new_tokens` can cut off coherent completions. Use dynamic stopping (EOS) with a hard cap.

### Reward Scoring: Scalar Reward from Reward Model

Concatenate prompt and response, pass through the frozen reward model r_φ to get a scalar per sequence:

```python
# Pseudocode: reward scoring
full_seq = torch.cat([prompts, responses], dim=1)  # (B, T_prompt + T_resp)
rewards = reward_model(full_seq).squeeze(-1)       # (B,)
```

**Cost tradeoff**: Reward model inference adds latency. Batch scoring and cache prompt embeddings if prompts repeat.

### KL Penalty: Constrain Policy Drift

Compute per-token log-prob ratio against the frozen reference policy π_ref (usually the SFT model):

```python
# Pseudocode: KL penalty
ref_logprobs = ref_policy(full_seq).log_softmax(-1).gather(-1, full_seq.unsqueeze(-1)).squeeze(-1)
ref_logprobs_resp = ref_logprobs[:, T_prompt:]     # (B, T_resp)

kl_per_token = logprobs - ref_logprobs_resp        # (B, T_resp)
kl_penalty = kl_coeff * kl_per_token.sum(dim=1)    # (B,)
```

**Failure mode**: If `kl_coeff` is too small, the policy collapses to reward hacking; too large stalls learning. Adaptive KL control (target KL ≈ 0.01–0.05 per token) stabilizes training.

### PPO Loss: Clipped Surrogate + Value Loss + Entropy

For each token position, compute the advantage using GAE(λ) over the trajectory. The PPO objective per token:

```
L_clip = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
L_vf   = 0.5 * (V_t - R_t)^2
L_ent  = -β * H(π_θ(·|s_t))
```

where `r_t = exp(logprobs_t - old_logprobs_t)`.

```python
# Pseudocode: PPO loss
ratio = (logprobs - old_logprobs).exp()            # (B, T_resp)
surrogate1 = ratio * advantages
surrogate2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
policy_loss = -torch.min(surrogate1, surrogate2).mean()

value_loss = 0.5 * (values - returns).pow(2).mean()
entropy_loss = -entropy_coeff * (-logprobs * logprobs.exp()).sum(-1).mean()

loss = policy_loss + vf_coeff * value_loss + entropy_loss
```

**Security/privacy**: The reward model sees full (prompt, response) pairs. Strip PII before logging; avoid persisting raw user data in training checkpoints.

### Minimal PyTorch-Style PPO Update Step

```python
def ppo_step(policy, ref_policy, reward_model, optimizer, batch, clip_eps=0.2, kl_coeff=0.05):
    prompts = batch["input_ids"]                    # (B, T_prompt)
    
    # 1. Rollout current policy
    responses, logprobs, values = policy.generate(prompts, max_new_tokens=128)
    full_seq = torch.cat([prompts, responses], dim=1)
    
    # 2. Reward + KL penalty
    rewards = reward_model(full_seq).squeeze(-1)    # (B,)
    ref_logprobs = ref_policy(full_seq).log_softmax(-1).gather(-1, full_seq.unsqueeze(-1)).squeeze(-1)
    kl = (logprobs - ref_logprobs[:, prompts.size(1):]).sum(-1)  # (B,)
    rewards = rewards - kl_coeff * kl
    
    # 3. GAE advantages (simplified: Monte Carlo returns)
    returns = rewards.unsqueeze(1).expand_as(values)  # (B, T_resp)
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 4. PPO loss
    old_logprobs = logprobs.detach()
    ratio = (logprobs - old_logprobs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = 0.5 * (values - returns).pow(2).mean()
    entropy_loss = -0.01 * (-logprobs * logprobs.exp()).sum(-1).mean()
    
    loss = policy_loss + 0.5 * value_loss + entropy_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    
    return {"loss": loss.item(), "kl": kl.mean().item(), "reward": rewards.mean().item()}
```

**Performance note**: This single-step sketch omits minibatching, multiple epochs, and GAE(λ). Production loops accumulate rollouts across many prompts, then run 3–10 epochs of minibatch updates per PPO iteration.

## Beyond PPO: DPO, KTO, and Preference Optimization Alternatives

PPO remains the canonical RLHF algorithm, but its complexity—reward model training, KL-penalized RL loops, and hyperparameter sensitivity—has driven adoption of RL-free alternatives. These methods reframe alignment as a supervised or contrastive learning problem, eliminating the need for online policy optimization.

### DPO: Direct Preference Optimization
DPO derives a closed-form solution for the optimal policy under the Bradley-Terry preference model. Given a dataset of chosen/rejected pairs \((x, y_w, y_l)\), it optimizes:
\[
\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
\]
No reward model, no RL loop, and no sampling during training. The reference model \(\pi_{\text{ref}}\) (usually the SFT checkpoint) anchors the KL constraint implicitly via \(\beta\). DPO works best when pairwise preferences are abundant and the reference model is already reasonably aligned. Failure mode: if the reference model assigns near-zero probability to chosen responses, gradients vanish; overfitting to the preference dataset can also degrade out-of-distribution generalization.

### KTO: Kahneman-Tversky Optimization
KTO replaces pairwise comparisons with binary desirability labels \((x, y, k)\) where \(k \in \{+1, -1\}\). Its loss combines a sigmoid-weighted log-likelihood for desirable samples and a complementary term for undesirable ones:
\[
\mathcal{L}_{\text{KTO}} = -\mathbb{E}_{k=+1} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right) \right] - \mathbb{E}_{k=-1} \left[ \log \sigma \left( -\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right) \right]
\]
This matches human feedback workflows where annotators label single outputs as "good" or "bad" rather than ranking pairs. KTO is more data-efficient when pairwise annotation is costly, but it assumes the binary signal is consistent—noisy or contradictory labels hurt more than in pairwise settings.

### IPO: Identity Preference Optimization
IPO adds an explicit regularization term to the DPO objective to prevent overfitting to the preference dataset:
\[
\mathcal{L}_{\text{IPO}} = \mathcal{L}_{\text{DPO}} + \lambda \, \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{\beta} \right)^2 \right]
\]
The regularizer forces the log-ratio difference toward the theoretical optimum \(1/\beta\), reducing variance when preference data is limited or noisy. IPO trades a small bias for lower variance, often yielding better held-out performance on small datasets.

### Tradeoff Summary

| Method | Data Required | Compute | Training Stability | Final Quality (Typical) |
|--------|---------------|---------|-------------------|-------------------------|
| DPO    | Pairwise (chosen/rejected) | Low (supervised) | High (no RL) | Strong with large, clean pairs |
| KTO    | Binary (desirable/undesirable) | Low (supervised) | High | Competitive; degrades with label noise |
| IPO    | Pairwise (like DPO) | Low (supervised) | Very high (regularized) | Better on small/noisy datasets |

**When to choose which:** Use DPO as default when you have high-quality pairwise data. Switch to KTO if annotation budget only supports binary labels. Add IPO regularization when preference data is scarce or annotator agreement is low. All three avoid PPO’s RL instability, but they inherit the reference model’s biases—audit \(\pi_{\text{ref}}\) for safety and privacy leaks before alignment.

## Failure Modes and Edge Cases in Practice

### Reward Hacking
Reward hacking occurs when the policy model exploits artifacts in the reward model rather than learning the intended behavior. Common patterns include excessive verbosity (the model learns that longer responses receive higher scores), overuse of formatting like bullet points or markdown, and sycophancy—agreeing with the user’s premise even when it is factually incorrect. These behaviors emerge because the reward model, trained on human comparisons, may inadvertently correlate surface-level features with quality. Mitigation strategies include length penalties, formatting-agnostic reward normalization, and adversarial data augmentation that breaks spurious correlations.

### Mode Collapse
Mode collapse manifests as a sharp drop in output diversity: the model converges to a narrow set of high-reward responses, producing repetitive or generic text. In training metrics, this often appears as a sudden spike in KL divergence between the policy and the reference model, indicating the policy has drifted far from the initial distribution. Diversity can be monitored by measuring distinct n-gram counts or entropy over a held-out prompt set. Regularization techniques—such as increasing the KL penalty coefficient, adding entropy bonuses, or using a mixture of reference models—help maintain exploration.

### Reward Model Miscalibration
A reward model trained on a specific prompt distribution (e.g., crowdsourced instructions) can miscalibrate when deployed on out-of-distribution inputs like coding tasks, multi-turn dialogues, or adversarial prompts. The reward scores may become overconfident or systematically biased, leading the policy to optimize for a proxy that no longer reflects human preference. Periodic re-evaluation of the reward model on a diverse, representative benchmark set—and retraining with expanded coverage—reduces this distribution shift.

### Detection and Observability
Effective detection relies on continuous monitoring and structured evaluation:
- **Training curves**: Track reward mean, reward standard deviation, and KL divergence per epoch. A rising KL with plateauing reward signals overoptimization.
- **Automated benchmarks**: Run MT-Bench, AlpacaEval, or custom task-specific evals every few checkpoints to catch regressions in instruction following, reasoning, and style.
- **Human spot-checks**: Sample 20–50 generations per checkpoint for qualitative review; look for verbosity, hallucination, tone shifts, and safety violations.
- **Logging infrastructure**: Store prompt, response, reward, KL, and reference log-probs for each training step to enable post-hoc analysis and correlation studies.

Combining these signals creates an early-warning system that catches pathologies before they reach production.

## Performance, Cost, and Infrastructure Tradeoffs

### Memory Breakdown

RLHF training requires holding multiple models in GPU memory simultaneously. For a 7B parameter model using BF16 precision:

| Component | Parameters | Memory (BF16) |
|-----------|------------|---------------|
| Policy model | 7B | ~14 GB |
| Reference model (frozen) | 7B | ~14 GB |
| Reward model | 1–7B | ~2–14 GB |
| Value head (PPO) | ~0.1B | ~0.2 GB |
| Optimizer states (AdamW, 2×) | 7B | ~56 GB |
| Gradients + activations | — | ~20–40 GB |

**Total**: ~100–140 GB without sharding. ZeRO Stage 3 partitions optimizer states, gradients, and parameters across GPUs, reducing per-GPU memory to ~15–25 GB for 7B models, enabling training on 8× A100 80GB or 16× A100 40GB.

### Compute Estimates

| Method | 7B Model GPU-hours (A100 80GB) | Notes |
|--------|-------------------------------|-------|
| PPO (full) | 100–500 | Multiple forward/backward passes per step; KL penalty adds overhead |
| DPO / IPO / KTO | 10–50 | Single forward pass; no reward model inference during training |
| LoRA/QLoRA PPO | 30–150 | 4–8× fewer trainable parameters; reward model still full precision |

PPO's variance comes from rollout length, KL coefficient tuning, and reward model batch size. DPO converges in 1–3 epochs over the preference dataset.

### Optimizations

```python
# Minimal LoRA + gradient checkpointing + flash attention setup
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

lora_config = LoraConfig(
    r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()
model.config.use_flash_attention_2 = True

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    optim="adamw_torch_fused",
    dataloader_num_workers=4,
)
```

**Sequence packing** (concatenating multiple short sequences to max length) improves GPU utilization by 30–50% for chat data with variable turn lengths.

### Cloud vs. On-Prem Cost Models

| Factor | Cloud (Spot) | Cloud (On-Demand) | On-Prem (3-yr amortized) |
|--------|--------------|-------------------|--------------------------|
| A100 80GB 8× node | $2.50–4.00/hr | $12–15/hr | ~$1.20/hr (power+cooling included) |
| 7B DPO (20 GPU-hrs) | $50–80 | $240–300 | $24 |
| 7B PPO (200 GPU-hrs) | $500–800 | $2,400–3,000 | $240 |

**Spot instance strategy**: Use capacity-optimized allocation, checkpoint every 500 steps, and implement auto-requeue with exponential backoff. Expect 5–15% interruption rate; design training loops to resume from optimizer state + RNG seed.

**Hidden costs**: Data transfer (ingress/egress), reward model hosting (separate inference endpoints), and evaluation compute (generation + human/auto eval) often exceed training spend by 2–3×.

## Security and Privacy Considerations in RLHF Pipelines

### PII in Human Feedback Data

Human feedback datasets—comparisons, rankings, and free-form corrections—often contain personally identifiable information (PII) from annotators or users. Names, locations, medical details, or proprietary code can leak into preference pairs. Before training the reward model, apply automated PII detection (regex + NER) followed by human review. For stronger guarantees, use differential privacy (DP) during reward model training: add calibrated Gaussian noise to gradients and clip per-sample contributions. DP-SGD with a privacy budget ε ≈ 1–3 typically preserves reward model quality while bounding leakage risk. The tradeoff: increased compute (gradient clipping, noise addition) and a modest drop in reward accuracy (~1–3% on benchmark preference tasks).

### Reward Model as Attack Surface

The reward model encodes the organization's preference distribution, making it a high-value target. Adversarial prompts can probe for training preferences—e.g., "Which response would you prefer: [harmful content] or [safe content]?"—revealing policy boundaries or extracting sensitive preference logic. Attackers may also craft inputs that maximize reward while violating safety constraints (reward hacking). Mitigate by hardening the reward model API: strip chain-of-thought, enforce output length limits, and log anomalous query patterns. Rate limiting per client identity reduces automated probing.

### Policy Model Extraction via RLHF API

Deployed RLHF models face membership inference and distillation attacks. An adversary queries the API with curated prompts to reconstruct the policy's behavior, then trains a surrogate model (distillation). Membership inference exploits confidence differences on training vs. non-training prompts. Defenses include: output filtering (refuse high-risk completions), probabilistic watermarking (bias token logits with a secret key detectable only by the owner), and API rate limiting with exponential backoff. Watermarking adds negligible latency but requires key management and may slightly degrade perplexity.

### Mitigations: Federated Reward Modeling

Federated reward modeling keeps raw preference data on-device or in siloed environments. Clients compute local reward model updates; only encrypted gradients or model deltas leave the device. Secure aggregation (e.g., Bonawitz et al.) prevents any single party from inspecting individual contributions. This reduces central PII exposure and limits the blast radius of a breach. Cost: higher communication rounds, heterogeneous compute across clients, and complexity in hyperparameter tuning. For most teams, a hybrid approach—centralized reward training with DP + federated fine-tuning on sensitive domains—balances security, cost, and model quality.

## Evaluation: Beyond Benchmarks to Production Monitoring

Offline benchmarks provide reproducible signal but miss real-world distribution shift. MT-Bench and AlpacaEval measure instruction-following quality against reference models, while IFEval tests strict constraint adherence (format, length, forbidden words). For safety, WildGuard and SafetyBench evaluate refusal rates, over-refusal, and jailbreak robustness across harm categories. Run these nightly on a fixed prompt set; track deltas, not absolute scores.

Human evaluation remains the gold standard for nuanced quality. Use side-by-side comparisons with randomized presentation order to reduce position bias. Collect 5–7 ratings per pair on a 5-point Likert scale (significantly worse → significantly better). Compute Krippendorff's alpha (interval metric) to quantify inter-annotator agreement; α < 0.67 indicates unreliable annotations—redesign guidelines or increase annotator calibration. Edge case: annotators disagree on creative tasks; segment by task type (coding vs. creative writing) and report agreement per segment.

Online metrics close the loop. Instrument three signals: explicit feedback (thumbs up/down), implicit feedback (regeneration rate within 30s), and engagement (session length, turn count). Regeneration rate > 15% often correlates with instruction-following failures. Guard against gaming: users may thumbs-down safe refusals; weight signals by user tenure and historical consistency.

```python
# Minimal drift detection for reward model scores
import numpy as np
from scipy import stats

def detect_reward_drift(baseline_scores, current_scores, alpha=0.01):
    """KS test for distribution shift in reward model outputs."""
    stat, p = stats.ks_2samp(baseline_scores, current_scores)
    return p < alpha, stat, p

# Usage: log reward scores per model version, alert on drift
baseline = np.load("reward_scores_v1.npy")
current = np.load("reward_scores_v2.npy")
drifted, stat, p = detect_reward_drift(baseline, current)
if drifted:
    print(f"Reward model drift detected: KS={stat:.3f}, p={p:.4f}")
```

Continuous evaluation pipeline: deploy candidate models to 1–5% of traffic (canary) with identical system prompts. Run A/B tests for 7–14 days minimum; power analysis determines sample size for your target effect (e.g., 2% win rate lift). Monitor reward model drift weekly—distribution shift in RM scores on held-out prompts signals preference model staleness. Retrain RM when KS-test p < 0.01 or human eval win rate drops > 3%.

Performance/cost tradeoff: offline benchmarks are cheap (~$50/run) but stale; human eval costs $15–30/annotation but catches qualitative regressions; online metrics are free at scale but noisy. Allocate budget: 60% automated, 30% human, 10% online analysis.

Security/privacy: never log raw user prompts in evaluation datasets without PII scrubbing and consent. Anonymize feedback signals before using them for RM retraining. Canary deployments must respect data residency—route EU traffic to EU canary instances only.


> **[IMAGE GENERATION FAILED]** The standard RLHF pipeline with three stages
>
> **Alt:** Three-stage RLHF pipeline: Pretraining → Supervised Fine-Tuning → RLHF
>
> **Prompt:** Technical diagram showing three sequential stages: 1) Pretraining (massive text corpus → base model), 2) Supervised Fine-Tuning (curated prompt-response pairs → SFT model), 3) RLHF (human comparisons → reward model → PPO optimization → aligned model). Clean flowchart style with arrows, minimal text labels.
>
> **Error:** kroki: mermaid still invalid after 2 repairs: Error 400: SyntaxError: Parse error on line 1:
flowchart LR    A[Pretraining Massiv
----------------^
Expecting 'NEWLINE', got 'NODE_STRING'
Error: Syntax error  | gemini: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.



> **[IMAGE GENERATION FAILED]** PPO training loop step by step
>
> **Alt:** PPO training loop with rollout, reward scoring, KL penalty, and policy update
>
> **Prompt:** Flowchart of PPO training loop: Prompt batch → Policy rollout (generate responses, logprobs, values) → Reward Model scoring (scalar reward) → Reference Policy (KL penalty) → Advantage computation (GAE) → PPO clipped surrogate loss + value loss + entropy → Gradient update → Updated policy. Show data flow between components.
>
> **Error:** kroki: mermaid still invalid after 2 repairs: Error 400: SyntaxError: Parse error on line 1:
flowchart TD    A[Prompt Batch] --> 
----------------^
Expecting 'NEWLINE', got 'NODE_STRING'
Error: Syntax error  | gemini: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.



> **[IMAGE GENERATION FAILED]** Core RLHF components and their interactions
>
> **Alt:** RLHF system architecture showing Policy, Reward Model, and Human Feedback Data interactions
>
> **Prompt:** Architecture diagram showing three main components: Policy Model (generates text), Reward Model (scores prompt-response pairs), Human Feedback Data (pairwise comparisons). Arrows show: Human data trains Reward Model; Reward Model provides reward signal to Policy; Policy generates responses for human evaluation. Include Reference Policy as frozen copy of SFT model for KL constraint.
>
> **Error:** kroki: Connection error. | gemini: [Errno 11001] getaddrinfo failed

