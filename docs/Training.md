# Training — BehavioralMacroDreadnought

---

## Environment

| Property | Value |
|---|---|
| Platform | Kaggle dual-T4 GPU |
| GPU | 2× NVIDIA T4 (16GB VRAM each) |
| Framework | PyTorch with AMP (Automatic Mixed Precision) |
| Optimizer | AdamW (lr=3×10⁻⁴, weight_decay=1×10⁻⁴) |

---

## Dataset

| Property | Value |
|---|---|
| File | `unified_training_data_v5.csv` |
| Builder | `serializer_v5.py` |
| Classes | 3 — Beginner (0), Intermediate (1), Advanced (2) |
| Sessions per class | 50,000 (stratified sampling) |
| Total sessions | 150,000 |
| Timestamp span | 2018 – 2026 |

### Source Files

**Advanced:**
- `cowrie_15k_pretty.json` — cowrie SSH honeypot
- `Main_system(app_log).json` — web application logs
- `Main_system(docker).jsonl` — docker network logs
- `honeypot_2.json` — second honeypot layer

**Intermediate:**
- 6 files across all three source types
- Includes cowrie-docker (distinct from cowrie)
- Real attacker session history included

**Beginner:**
- `cowrie.json`
- `Main_system(network logs).json`
- `Weak_Attacks.jsonl` — contains unique `severity` field

---

## SessionAwareDataset — Session Grouping

Records were grouped by a composite session key:

```python
session_id = f"{level_int}_{src_ip}_{session}"
```

The session integer resets to 1 for every attacker, and IP alone is not unique across time. The three-part key is what makes each session unique across the full dataset.

Each session was grouped into a window of the **last H=10 commands**. The terminal record (last record in the session) provided the scalar values `time_delta` and `has_custom_tool`.

---

## Hyperparameters

| Parameter | Symbol | Value |
|---|---|---|
| Latent dimension | d_model | 512 |
| Expert hidden dimension | d_hidden | 4096 |
| Attention heads | n_heads | 16 |
| Command sequence length | seq_len | 256 |
| Session history window | H | 10 |
| Transformer layers | L | 3 |
| Routing uniform base | uniform_base | 0.45 |
| Number of experts | E | 3 |
| Output tool classes | num_tools | 5 |
| Learning rate | lr | 3×10⁻⁴ |
| Weight decay | — | 1×10⁻⁴ |
| Micro-batch size | — | 8 |
| Gradient accumulation steps | — | 8 |
| Effective batch size | — | 64 |
| Training epochs | — | 10 (reduced from 15 for Kaggle session safety) |

---

## Loss Function

```python
Loss_total = (
    loss_threat    # MSELoss — continuous threat score head
  + loss_class     # CrossEntropyLoss — skill classification head
  + loss_tools     # BCEWithLogitsLoss — tool detection head (float32)
  + loss_balance   # [K] CV² load balancing, lambda=0.10
  + loss_echo      # [M] echo penalty, lambda=0.05
  + loss_ortho     # [L] orthogonal penalty, lambda=0.01
) / accumulation_steps
```

> **Critical:** Tool detection loss (`loss_tools`) was computed in float32 even during AMP training. BCE on near-zero targets with float16 produces infinity gradients that corrupt the entire training run. This was a real bug encountered mid-training.

---

## Regularization — Implementation Details

### [K] Load Balancing Loss — CV²

```python
mean_usage   = routing_probs.mean(dim=0)
loss_balance = lambda_K * var(mean_usage) / (mean_usage.mean()**2 + eps)
```

Penalizes unequal expert utilization. If Expert 1 receives 80% of traffic, the penalty fires hard and forces the router to rebalance.

### [L] Orthogonal Penalty — Frobenius Norm

```python
# Extract first linear layer weights from each expert
# Normalize each: w_n = F.normalize(w, p=2, dim=1)
# For each pair: penalty += frobenius_norm(w_i_n @ w_j_n.T)
```

Forces expert weight matrices to be geometrically perpendicular. Combined with orthogonal initialization — experts begin already diverged and the penalty maintains that divergence throughout training.

### [M] Echo Penalty — Temporal Variance

```python
temporal_variance = temporal_out.var(dim=1).mean()
loss_echo = lambda_M * (1.0 / (temporal_var + 1e-8))
```

Prevents TemporalStack from collapsing to always reading only the most recent command.

---

## Expert Initialization

All expert first layers were initialized with orthogonal weights:

```python
def init_orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
```

This gave the Orthogonal Penalty [L] a head start — experts begin already diverged from epoch zero.

---

## Gradient Accumulation

```
micro_batch = 8    (GPU memory constraint)
accum_steps = 8    (simulate larger batch)
effective_batch = 64

Every 8 micro-batches:
  loss.backward() called 8 times  →  gradients accumulate
  optimizer.step() called once
  optimizer.zero_grad() called once
```

---

## The Expert Starvation Problem

Before the Uniform-Base Router was implemented, standard Top-1 routing was used. By epoch 3, Expert 1 was receiving ~75% of traffic. Expert 2 and Expert 3 had effectively stopped learning — their weights froze.

Training was restarted with `uniform_base=0.45`, which fixed the problem permanently by guaranteeing a minimum 15% routing floor per expert.

---

## Results

**Overall validation accuracy: 97.654%**

### Confusion Matrix

| | Predicted: Beginner | Predicted: Intermediate | Predicted: Advanced |
|---|---|---|---|
| True: Beginner | 333 | 4 | 0 |
| True: Intermediate | 5 | 320 | 8 |
| True: Advanced | 0 | 7 | 323 |

Zero Beginner→Advanced confusion. Zero Advanced→Beginner confusion.

### Expert Routing on Validation Set

| Skill Class | Expert 1 (GELU) | Expert 2 (Mish) | Expert 3 (SpLR_V2) |
|---|---|---|---|
| Beginner Sessions | 68% | 19% | 13% |
| Intermediate Sessions | 22% | 61% | 17% |
| Advanced Sessions | 14% | 17% | **69%** |

The routing mechanism learned skill-level specialization without being explicitly supervised to do so.

### SpLR_V2 d-Parameter Distribution After Training

| Channel Behavior | Proportion | d Range | Meaning |
|---|---|---|---|
| Positive d | 69% | +0.01 to +0.6 | Signal amplification during high entropy |
| Negative d | 31% | -0.4 to -0.01 | Noise suppression during high entropy |

This distribution was stable across multiple training runs with different random seeds. It was not programmed — it emerged from gradient descent.

---

## Known Training Limitations

**Tool detection class imbalance:**
- `sqlmap` and `custom` → well trained (present in >3% of sessions)
- `hydra`, `metasploit`, `nikto` → undertrained (present in <3% of sessions)
- These three tool flags are unreliable at inference; use the injection scanner's behavioral tool inference instead

**Temporal data gap:**
- The Advanced dataset contains sessions spanning 2018–2026 mixed together
- The model treats all sessions as temporally equivalent
- 2018 attack patterns differ from 2026 patterns; weighting recent sessions more heavily in future retraining would reduce this bias

---

*Mohammad Khalid AL-Biltaji — June 2026*
