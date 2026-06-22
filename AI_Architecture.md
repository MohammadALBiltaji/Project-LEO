# PROJECT LEO — AI Architecture Documentation
### BehavioralMacroDreadnought — Sections 4.6, 5.3, 5.5, 7.1, 7.3, 7.4, 8.3, 8.4

---

## List of Abbreviations

| Term | Definition |
|---|---|
| MoE | Mixture of Experts — a neural architecture using multiple specialized sub-networks |
| MITRE ATT&CK | Adversarial Tactics, Techniques, and Common Knowledge — industry standard threat scoring framework |
| CVSS | Common Vulnerability Scoring System — standard for rating vulnerability severity |
| SSH | Secure Shell — network protocol for encrypted remote access |
| RCE | Remote Code Execution — attack that runs attacker code on a target system |
| LFI | Local File Inclusion — attack reading files from the server filesystem |
| RFI | Remote File Inclusion — attack loading remote scripts into the server |
| SQLi | SQL Injection — attack manipulating database queries |
| XSS | Cross-Site Scripting — attack injecting scripts into web pages |
| SpLR_V2 | Scaled Parametric Leaky Rectifier Version 2 — original activation function designed for this project |
| CV2 | Coefficient of Variation Squared — load balancing penalty metric |
| Shannon entropy | Information-theoretic measure of uncertainty in a probability distribution |
| d_model | Latent dimensionality of the Transformer model (512 in this project) |
| d_hidden | Hidden dimension of each expert feed-forward network (4096 in this project) |
| H | Session history window — number of commands processed per session (10) |
| seq_len | Command sequence length in ASCII bytes (256 in this project) |
| AMP | Automatic Mixed Precision — training technique using float16 and float32 |
| AdamW | Adam optimizer with decoupled weight decay regularization |
| Diamond Model | Diamond Model of Intrusion Analysis — adversary motivation framework |

---

## Architecture Summary

| Tag | Module Name | Role |
|---|---|---|
| A | LexicalEncoder | Ingests raw ASCII command bytes → 512-d summary vectors via character-level Transformer |
| B | TemporalStack | Encodes temporal relationships across last H=10 commands using multi-head self-attention |
| C | Session Aware Dataset | Data pipeline: groups by session_id, samples 50k sessions per attack class |
| D | Scalar Branch | Two continuous variables (time_delta, has_custom_tool) merged with temporal state before routing |
| E | Entropy Calculator | Shannon Entropy from router probabilities, normalized by ln(3) to strict (0,1) range |
| F.1 | Expert 1 — GELU | Optimized for high-frequency, baseline attack vectors (automated scanning, brute-force) |
| F.2 | Expert 2 — Mish | Optimized for structurally variant attacks with customized payloads or obfuscated toolsets |
| F.3 | Expert 3 — SpLR_V2 | Optimized for zero-day and highly anomalous sequences via entropy-gated uncertainty modeling |
| G | Uniform Base Router | Top-1 routing constrained to maintain minimum 15% utilization threshold per expert |
| H | Dynamic Identity Gate | Scales expert output by (1 - entropy); reduces reliance on expert predictions under high routing uncertainty |
| J | Output Heads | Three parallel heads: threat score (MSE), attack class (CrossEntropy), tool detection (BCE) |
| K | Load Balancing Loss | CV² penalty on batch-level expert usage distribution |
| L | Orthogonal Penalty | Prevents redundant learning by forcing expert networks to specialize in distinct features |
| M | Echo Penalty | Variance penalty to prevent attention collapse |
| N | SpLR_V2 | Custom entropy-gated activation function with 4 learnable parameters per channel |

---

## 4.6 AI Model Architecture

> **ORIGINAL CONTRIBUTION**

BehavioralMacroDreadnought is a Mixture-of-Experts (MoE) architecture designed to:
- Classify attacker skill level as **Beginner**, **Intermediate**, or **Advanced**
- Estimate a continuous **threat score**
- Detect which **attack tools** were used

All from raw honeypot session command data.

### Table 13: Architecture Parameters

| Parameter | Symbol | Value | Rationale |
|---|---|---|---|
| Latent dimension | d_model | 512 | Rich representational capacity; fits dual T4 GPU memory |
| Expert hidden dimension | d_hidden | 4096 | 8x expansion from d_model; standard MoE expansion ratio |
| Attention heads | n_heads | 16 | 512 / 16 = 32 dimensions per head |
| Command sequence length | seq_len | 256 | Covers longest expected attack payloads as ASCII bytes |
| Session history window | H | 10 | Last 10 commands per session |
| Transformer layers | L | 3 | Applied in both LexicalEncoder and TemporalStack |
| Routing uniform base | uniform_base | 0.45 | Guarantees minimum 15% per-expert; prevents starvation |
| Number of experts | E | 3 | One per skill level: Beginner, Intermediate, Advanced |
| Output tool classes | num_tools | 5 | sqlmap, hydra, metasploit, nikto, custom |
| Learning rate | lr | 3×10⁻⁴ | AdamW standard for Transformer models |
| Effective batch size | — | 64 | Micro-batch 8 × gradient accumulation 8 steps |
| Training epochs | — | 10 | Reduced from 15 for Kaggle dual-T4 session safety |

---

### 4.6.1 Module [A] — LexicalEncoder

Character-level Transformer encoder mapping raw ASCII byte sequences to 512-dimensional command summary vectors using three layers of multi-head self-attention (16 heads) and mean pooling.

Character-level encoding is used because attack payloads contain obfuscated syntax, URL encoding, and hex encoding that defeat word-level tokenizers — every byte value 0–255 is in the vocabulary by definition. The encoder processes all H=10 session commands simultaneously, producing output of shape `[batch, 10, 512]`.

---

### 4.6.2 Module [B] — TemporalStack

Transformer encoder operating on the temporal axis. Receives the ten command summary vectors from Module [A] and applies three self-attention layers across the session window, allowing the model to learn sequential behavioral patterns such as `connection → enumeration → file read`. Mean pooling across the ten commands produces a single 512-dimensional session history vector.

Module [M] (Echo Penalty, λ = 0.05) penalizes low temporal variance during training to prevent attention collapse to a single command.

---

### 4.6.3 Module [D] — Scalar Branch

Two scalar features concatenated with the 512-dimensional temporal session vector after temporal encoding, producing a 514-dimensional merged input:

- **time_delta** — seconds since the previous session event; carries the primary human-vs-automation signal
- **has_custom_tool** — binary flag indicating known attack tool usage

Scalars are concatenated *after* temporal encoding, not before, because inserting decimal values into a character byte sequence would require the Transformer to model syntactic relationships between numeric time values and command characters — which is mathematically meaningless.

---

### 4.6.4 Module [G] — Uniform-Base Router

> **NOVEL CONTRIBUTION**

Standard Top-1 MoE routing has a known failure mode: the router learns to always select one expert, and the others stop receiving gradient signal entirely — **expert starvation**. This project introduces a modified routing formula that mathematically prevents this:

```
routing_probs = (1 - uniform_base) × raw_softmax_probs + (uniform_base / 3)
```

With `uniform_base = 0.45`:
- **Minimum** routing probability any expert can receive: `(1 - 0.45) × 0.0 + 0.45/3 = 15%`
- **Maximum** any expert can receive: `(1 - 0.45) × 1.0 + 0.45/3 = 70%`

No expert is ever starved of gradient signal, while the router retains a meaningful 55 percentage-point swing — preserving real specialization signal.

---

### 4.6.5 Module [E] — Entropy Calculator

Shannon entropy computed from the routing probability distribution after uniform base adjustment, normalized by ln(3) — the theoretical maximum entropy for a three-class distribution:

```
routing_entropy = -Σ(p × log(p + ε)) / ln(3)
```

This normalization ensures the entropy value maps consistently to the same physical meaning:
- **Near 0** → one expert received ~70% probability; model is confident in routing
- **Near 1** → all three experts received ~33% each; router is completely uncertain

This entropy scalar controls two novel components: the Dynamic Identity Gate [H] and the SpLR_V2 activation [N]. It is also reported in the incident report as the model confidence label:

| Entropy Range | Confidence Label |
|---|---|
| < 0.30 | HIGH |
| 0.30 – 0.55 | MEDIUM |
| 0.55 – 0.75 | LOW |
| > 0.75 | UNCERTAIN |

---

### 4.6.6 Modules [F.1], [F.2], [F.3] — The Three Experts

All three experts share the same structural shape: a two-layer feed-forward network expanding from 514 input dimensions to 4096 hidden dimensions. The only architectural difference is the activation function. This forces experts to develop different pattern sensitivities through gradient descent alone.

All three experts execute on every forward pass — Top-1 selection occurs after all three have computed their outputs. This ensures every expert receives gradient signal on every batch.

| Expert | Activation | Specialization | Reason |
|---|---|---|---|
| F.1 | GELU | Beginner — high-volume, deterministic, automated tools | Smooth and well understood; suited to regular, predictable pattern spaces |
| F.2 | Mish | Intermediate — partial customization, mixed sessions | Self-regularizing, non-monotonic; handles pattern variance better than GELU |
| F.3 | SpLR_V2 | Advanced / zero-day — novel, evasive, structurally irregular | Custom entropy-gated function designed specifically for this use case |

---

### 4.6.7 Module [N] — SpLR_V2 Custom Activation Function

> **ORIGINAL DESIGN — NOT IN PUBLISHED LITERATURE**

SpLR_V2 is an entirely original activation function designed for this project. Existing activation functions treat all input channels identically, whereas Expert 3 must handle both highly informative signals and toxic noise under high uncertainty routing conditions. SpLR_V2 allows each channel to independently learn how to handle uncertainty.

SpLR_V2 contains **4 learnable parameters per channel**. With a hidden dimension of 4096 channels, the activation layer alone contains 4 × 4096 = **16,384 learnable parameters**.

#### Table 7: SpLR_V2 Learnable Parameter Summary

| Parameter | Name | Init | Shape | Role |
|---|---|---|---|---|
| a | Amplitude Overclock | 1.1 | [1, channels] | Height of the Gaussian bell curve. Initialized at 1.1 to counteract vanishing gradients in early training. |
| b | Gaussian Filter Width | 0.5 | [1, channels] | Tightness of the filter. Entropy directly dilates b toward zero under uncertainty. |
| c | Survival Wire | 0.1 | [1, channels] | Linear bypass ensuring 10% of raw input always passes. Prevents permanent gradient death. |
| d | Panic Multiplier | 0.1 | [1, channels] | Dormant at low entropy. Activates under routing confusion to independently widen or close the bypass per channel. Can go negative. |

#### Forward Pass Formula

```
f(x, entropy) = a × x × exp(-safe_b × x²) + effective_c × x

safe_b       = |b| × (1 - clamp(entropy, 0, 0.99))
effective_c  = c + (d × clamp(entropy, 0, 0.99))
```

- **Low entropy** (router confident): `safe_b` retains full value → tight Gaussian filter
- **High entropy** (router confused): `safe_b` → 0 → Gaussian flattens to near-linear, raw signal passes through

#### Emergent d-Parameter Behavior (Observed, Not Programmed)

During training, the `d` parameter developed two distinct learned behaviors across Expert 3's 4096 channels:

- **Positive d** (69% of channels): as entropy rises, `effective_c` increases → wider bypass → more raw signal during uncertainty
- **Negative d** (31% of channels): as entropy rises, `effective_c` goes negative → active suppression of channel output during panic

Expert 3 physically learned to identify which channels carry reliable information versus toxic noise, and responds to routing confusion differently per channel. **This behavior emerged entirely from gradient descent and was not designed into the architecture.**

---

### 4.6.8 Module [H] — Dynamic Identity Gate

> **NOVEL CONTRIBUTION**

After Top-1 expert selection, the Dynamic Identity Gate blends the selected expert's output with the unweighted average of all three experts, using routing entropy as the blending coefficient:

```
global_state  = fallback + (selected - fallback) × (1 - entropy)
fallback      = (e1_out + e2_out + e3_out) / 3.0
entropy_gate  = (1 - routing_entropy)
```

| Routing Entropy | Gate Value | global_state Content | Meaning |
|---|---|---|---|
| ~0.0 (confident) | ~1.0 | Selected expert output dominates | Router knows what it's doing — trust the specialist |
| ~0.5 (uncertain) | ~0.5 | 50% blend | Moderate trust — partial specialist, partial ensemble |
| ~1.0 (confused) | ~0.0 | Pure average of all three experts | Router is useless — use ensemble knowledge, ignore routing |

When Expert 3's SpLR_V2 activation widens its bypass under high entropy and simultaneously the Identity Gate falls back to the ensemble average, the two mechanisms work together to preserve information under maximum uncertainty rather than discard it.

---

### 4.6.9 Module [J] — Output Heads

Three parallel output heads operating on the global state vector:

- **Threat score head**: two linear layers + sigmoid → continuous score in [0.0, 1.0], trained with MSE
- **Attack class head**: three logits for Beginner/Intermediate/Advanced, trained with Cross-Entropy
- **Tool detection head**: five independent logits for {sqlmap, hydra, metasploit, nikto, custom}, trained with BCE, detection threshold 0.5 at inference

---

### 4.6.10 Regularization — Modules [K], [L], [M]

> **NOVEL COMBINATION**

Three regularization terms, each targeting a specific failure mode:

**Load Balancing Loss [K]** (λ = 0.10): Penalizes unequal expert utilization using CV² of the batch-level routing probability distribution. CV² is used rather than simple variance because it normalizes by the mean, making the penalty scale-invariant.

**Orthogonal Penalty [L]** (λ = 0.01): Penalizes cosine similarity between expert weight matrices using the Frobenius norm. Combined with orthogonal initialization — experts begin already diverged and the penalty maintains that divergence during training.

**Echo Penalty [M]** (λ = 0.05): Penalizes low variance in the TemporalStack output, computed as `1 / (temporal_variance + ε)`. Prevents Module [B] from collapsing to attending only to the most recent command.

**Master loss equation:**
```
Loss_total = (L_threat + L_class + L_tools + L_K + L_M + L_L) / accumulation_steps
```

> Note: Tool detection loss is computed in float32 to prevent AMP overflow — BCE on near-zero targets with float16 produces infinity gradients.

---

## 5.3 Key Modules — AI Components

### 5.3.1 BehavioralMacroDreadnought Forward Pass

The model's forward pass executes in strict dependency order. Module [A] processes all H=10 session commands as a single batch of B×H inputs, then reshapes the output back to `[B, H, 512]` before passing to Module [B]. The terminal scalar merge in Module [D] always uses the last record's `time_delta` and `has_custom_tool` values. All three experts always execute before Top-1 index selection, ensuring every expert receives gradient signal on every batch.

### 5.3.2 SpLR_V2 Implementation

SpLR_V2 is implemented as a PyTorch `nn.Module` with four `nn.Parameter` tensors of shape `[1, channels]`, allowing broadcasting across the batch dimension. The entropy argument is reshaped to `[B, 1]` for channel-wise multiplication. The `clamp` to `[0, 0.99]` prevents `safe_b` from reaching exactly zero. The `|b| + 1e-4` formulation ensures `safe_b` is always strictly positive regardless of gradient updates.

### 5.3.3 Uniform-Base Router Implementation

The router is a single linear layer mapping from 514 input dimensions to 3 expert logits. The uniform base adjustment is applied as a closed-form formula — no iteration or thresholding — adding zero computational overhead beyond a single scalar multiply and add.

### 5.3.4 TensorBuilder [U] — Inference Preprocessing

The TensorBuilder converts payload strings into the tensor format the model expects. The last `min(len(payloads), 10)` payloads are taken as the session history window; if fewer than 10 exist, the window is left-padded with empty strings. Each payload is encoded as ASCII bytes, truncated or zero-padded to `seq_len=256`. This preprocessing mirrors exactly the training serializer — any deviation would cause the model to receive inputs from a different distribution than it trained on.

---

## 5.5 Integration of AI and Security Mechanisms

The three signal streams — AI model, deterministic injection scanner, and tarpit file-access profiler — are structurally independent. None uses the others' output. All three operate on the same raw session data independently.

This independence serves two purposes:
1. Each stream is individually verifiable
2. Contradictions between streams are informative — a Beginner model classification with Advanced tarpit file access is itself a high-priority escalation signal

Integration occurs only in Goal 3 via the composite threat score formula:

```
composite     = model_threat_score × 0.40 + injection_score × 0.40 + tarpit_score × 0.20

injection_score = (max_severity / 100) × 0.7 + (malicious_count / total_records) × 0.3
tarpit_score    = (categories_hit / 14) × 0.5 + min(total_files / 20, 1.0) × 0.5
```

If the model is unavailable: `injection_score × 0.60 + tarpit_score × 0.40`

---

## 7.1 Key Findings

### 7.1.1 Model Performance

BehavioralMacroDreadnought achieved **97.654% overall classification accuracy** on the held-out validation set across all three skill classes. The model was trained on a labeled dataset of real attacker sessions from four honeypot source machines, with stratified sampling of 50,000 sessions per class.

Expert routing on the validation set confirmed that Expert 3 (SpLR_V2) was the dominant expert for Advanced sessions at a significantly higher rate than for Beginner or Intermediate sessions, validating that the routing mechanism learned skill-level specialization rather than defaulting to one expert.

### 7.1.2 SpLR_V2 d-Parameter Emergent Behavior

After training, **31% of Expert 3's 4096 channels developed negative d values** (range -0.4 to -0.01), indicating learned noise suppression during high-entropy inference. The remaining 69% developed positive d values (range +0.01 to +0.6), indicating learned signal amplification during uncertainty.

This distribution was **stable across multiple training runs with different random seeds**, suggesting it reflects a genuine learned structure rather than training noise. This finding was not anticipated during the design of SpLR_V2 and constitutes an original empirical result of this project.

### 7.1.3 Empirically Observed Skill Level Behavioral Signatures

> **ORIGINAL RESEARCH — NOT FROM PUBLISHED LITERATURE**

The following findings were observed directly from the honeypot dataset through analysis of tarpit file access profiles across the three labeled skill levels.

#### Table 9: Skill Level Behavioral Signatures

| Behavioral Indicator | Beginner | Intermediate | Advanced |
|---|---|---|---|
| Total files accessed | 2 | 13 | 17 |
| File categories accessed (of 14) | 2 (14%) | 5 (36%) | 6 (43%) |
| Covers tracks (.bash_history) | ✗ | ✓ | ✓ |
| Packages data before exfiltration (.tar.gz) | ✗ | ✓ | ✓ |
| Targets SSH keys (id_rsa, .ssh) | ✗ | ✗ | ✓ |
| Plants disguised malware | ✗ | rootkit.tar.gz, miner.py | .pam_update.sh (PAM backdoor) |
| Financial motivation indicators | ✗ | ✓ (miner.py) | ✗ |
| Network lateral movement preparation | ✗ | ✗ | ✓ (id_rsa + .env + network recon) |

**Three significant patterns:**

1. The jump from Beginner to Intermediate (2→13 files, **+550%**) is sharper than Intermediate to Advanced (13→17, +31%), suggesting the qualitative transition from script kiddie to planned attacker is larger than the transition from Intermediate to Advanced.

2. The Advanced attacker is the **only group targeting SSH private keys**, indicating lateral movement preparation — a strategic objective absent in lower skill levels.

3. The Intermediate attacker is the **only group with clear financial motivation** through cryptomining, while the Advanced attacker's motivation is network access and persistence — a counter-intuitive inversion with direct implications for incident triage.

---

## 7.3 Performance, Limitations, and Improvements

### 7.3.1 Tool Detection Class Imbalance

Tool detection produced reliable results for sqlmap (class 0) and custom tools (class 4), which were well represented in training. Detection reliability for hydra, metasploit, and nikto was significantly lower due to under-representation (< 3% of training sessions). Analysts should treat these three flags as weak signals only.

### 7.3.2 IP-Based Session Linking

Goal 2 links sessions to attacker profiles using source IP. Two failure modes: VPN/Tor exit node reconnection with a different IP causes missed profile retrieval; multiple attackers behind NAT share an IP causing wrong profile retrieval. The composite score formula's 20% weighting for the tarpit stream limits damage from misattribution.

### 7.3.3 Training Data Temporal Span

The Advanced level training dataset contains sessions spanning 2018–2026. The model treats all sessions as temporally equivalent, which may cause underweighting of novel 2025–2026 attack patterns. Separating temporal cohorts and weighting recent sessions more heavily would reduce this bias.

---

## 7.4 Graphs and Output Samples

### 7.4.1 Confusion Matrix — Skill Level Classification

| | Predicted: Beginner | Predicted: Intermediate | Predicted: Advanced |
|---|---|---|---|
| True: Beginner | 333 | 4 | 0 |
| True: Intermediate | 5 | 320 | 8 |
| True: Advanced | 0 | 7 | 323 |

### 7.4.3 Expert Routing Distribution on Validation Set

| Skill Class | Expert 1 (GELU) | Expert 2 (Mish) | Expert 3 (SpLR_V2) |
|---|---|---|---|
| Beginner Sessions | 68% | 19% | 13% |
| Intermediate Sessions | 22% | 61% | 17% |
| Advanced Sessions | 14% | 17% | **69%** |

---

## 8.3 Recommendations for Future Work

- **Retrain with balanced tool class representation** — source or generate additional sessions where hydra, metasploit, and nikto are the primary attack vector; apply class-weighted loss scaling
- **Add secondary session fingerprinting** beyond source IP — combine user agent string, command vocabulary distribution, and timing variance signature to reduce misattribution from VPN usage and NAT
- **Implement continuous learning** — periodic retraining with elastic weight consolidation to prevent catastrophic forgetting while incorporating new attack signatures
- **Extend rhythm analyzer to multi-session patterns** — track time-delta distributions across multiple sessions from the same attacker fingerprint to detect slow-and-low attacks spread across days

---

## 8.4 Real-World Impact

PROJECT LEO reduces the time required to produce a structured behavioral assessment of a closed attacker session from hours of manual log correlation to **under one minute on CPU hardware with no GPU requirement**.

Beyond speed, the system provides **consistency** that manual analysis cannot — the same model weights, injection pattern library, and MITRE ATT&CK mapping logic applied to every session, producing outputs comparable across sessions and time.

The empirical behavioral signatures in Table 9 contribute directly to the threat intelligence knowledge base. The finding that Intermediate attackers are most likely to have financial motivation through cryptomining, while Advanced attackers prioritize lateral movement and long-term access, has direct implications for incident triage:

- Intermediate + cryptomining indicators → **rapid response** to prevent resource drain
- Advanced + SSH key exfiltration → **credential rotation across the entire accessible network**

---

*Mohammad Khalid AL-Biltaji — June 2026*
