# PROJECT LEO
### Behavioral Honeypot Analysis System — BehavioralMacroDreadnought

A full AI pipeline that turns raw honeypot session logs into structured incident reports. Classifies attacker skill level, estimates threat score, detects tools, and maps intent to MITRE ATT\&CK — in under one minute on CPU.

**97.654% validation accuracy** across Beginner / Intermediate / Advanced skill classes.

---

## What It Does

PROJECT LEO fuses three independent analysis streams on every closed attacker session:

| Stream | What it produces |
|---|---|
| **BehavioralMacroDreadnought** (MoE neural model) | Skill class, threat score, tool detection, routing entropy |
| **Injection Scanner** | Per-payload attack type + CVSS-anchored severity score |
| **Tarpit Profiler** | File access intent, MITRE ATT\&CK tactics, motivation label |

All three run independently on the same raw session data. Contradictions between streams are surfaced explicitly in the analyst notes — a Beginner model classification with Advanced tarpit file access is itself a high-priority signal.

---

## Architecture — BehavioralMacroDreadnought

A Mixture-of-Experts Transformer trained on honeypot session command data.

| Tag | Module | Role |
|---|---|---|
| A | LexicalEncoder | Character-level Transformer → 512-d command vectors |
| B | TemporalStack | Self-attention across 10-command session history |
| D | Scalar Branch | time_delta + has_custom_tool merged after temporal encoding |
| E | Entropy Calculator | Shannon entropy normalized by ln(3) → strict (0,1) |
| G | Uniform-Base Router | Top-1 routing with 15% minimum per-expert floor |
| F.1 | Expert 1 — GELU | Beginner specialist |
| F.2 | Expert 2 — Mish | Intermediate specialist |
| F.3 | Expert 3 — SpLR\_V2 | Advanced / zero-day specialist |
| H | Dynamic Identity Gate | Entropy-weighted fallback to ensemble average |
| J | Output Heads | Threat score (MSE) · Attack class (CE) · Tools (BCE) |
| N | SpLR\_V2 | Custom entropy-gated activation — see below |

### Novel Contributions

**Uniform-Base Router** — Prevents expert starvation by guaranteeing ≥15% routing probability per expert while preserving a 55-point specialization range:

```
routing_probs = (1 - uniform_base) × raw_softmax + (uniform_base / 3)
```

**SpLR\_V2** — Original entropy-gated activation function with 4 learnable parameters per channel (16,384 parameters in Expert 3's hidden layer):

```
f(x, entropy) = a × x × exp(-safe_b × x²) + effective_c × x
safe_b        = |b| × (1 - clamp(entropy, 0, 0.99))
effective_c   = c + (d × clamp(entropy, 0, 0.99))
```

The `d` parameter can go negative. After training, 31% of Expert 3's 4096 channels developed negative `d` values — learned noise suppression during high-entropy routing. This behavior was not programmed; it emerged from gradient descent and was stable across multiple random seeds.

**Dynamic Identity Gate** — Blends selected expert output with ensemble average using routing entropy as the coefficient. At entropy ≈ 1.0, the gate falls back entirely to the ensemble average rather than committing to an unreliable Top-1 selection.

---

## Empirical Findings — Attacker Behavioral Signatures

Original findings from the honeypot dataset. Not drawn from published literature.

| Behavioral Indicator | Beginner | Intermediate | Advanced |
|---|---|---|---|
| Total files accessed | 2 | 13 | 17 |
| File categories accessed (of 14) | 2 (14%) | 5 (36%) | 6 (43%) |
| Covers tracks (.bash\_history) | ✗ | ✓ | ✓ |
| Packages data before exfil (.tar.gz) | ✗ | ✓ | ✓ |
| Targets SSH keys (id\_rsa, .ssh) | ✗ | ✗ | ✓ |
| Plants disguised malware | ✗ | rootkit.tar.gz, miner.py | .pam\_update.sh (PAM backdoor) |
| Financial motivation | ✗ | ✓ (cryptomining) | ✗ |
| Lateral movement preparation | ✗ | ✗ | ✓ |

The jump from Beginner to Intermediate (2→13 files, +550%) is sharper than Intermediate to Advanced (13→17, +31%). The Intermediate attacker is the only group with clear financial motivation through cryptomining, while the Advanced attacker's goal is network access and persistence — a counter-intuitive inversion with direct implications for incident triage.

---

## Expert Routing on Validation Set

The routing mechanism learned skill-level specialization:

| Skill Class | Expert 1 (GELU) | Expert 2 (Mish) | Expert 3 (SpLR\_V2) |
|---|---|---|---|
| Beginner Sessions | 68% | 19% | 13% |
| Intermediate Sessions | 22% | 61% | 17% |
| Advanced Sessions | 14% | 17% | **69%** |

---

## Quick Start

```python
from leo_pipeline import LeoReportEngine

engine = LeoReportEngine(
    model_path="macro_dreadnought_best.pth",   # trained weights
    profiles_dir="attacker_profiles/",          # tarpit JSON files
    output_dir="reports/",
)

# From a session file (JSON or JSONL)
report = engine.run_from_file("session.json")
print(report)

# From raw records already in memory
report = engine.run_from_records(list_of_json_dicts)
print(report)
```

Works without PyTorch — if the model weights are unavailable, the injection scanner and tarpit profiler remain fully operational and the threat score formula degrades gracefully to `injection_score × 0.60 + tarpit_score × 0.40`.

### Integration Points

Search `INTEGRATION POINT` in `leo_pipeline.py` to find the four configuration locations:

1. `MODEL_PATH` — path to trained `.pth` weights
2. `PROFILES_DIR` — directory containing the 4 attacker profile JSONs
3. `OUTPUT_DIR` — where reports are saved
4. `run_from_file()` — entry point for live session files

---

## Report Output

Each session produces a structured 8-section incident report:

1. Threat score (0–100) with label: NEGLIGIBLE / LOW / MEDIUM / HIGH / CRITICAL
2. Model output — skill class, routing entropy, dominant expert, interpretation
3. Per-payload injection analysis with CVSS-anchored severity
4. Tarpit file access with MITRE ATT\&CK tactic breakdown
5. Attacker command timeline with inline severity flags
6. Behavioral tool inference
7. Analyst notes — contradictions, rhythm anomalies, multi-vector flags
8. Recommended actions keyed to threat level

---

## Threat Score Fusion

```
composite = model_threat_score × 0.40 + injection_score × 0.40 + tarpit_score × 0.20

injection_score = (max_severity / 100) × 0.7 + (malicious_count / total_records) × 0.3
tarpit_score    = (categories_hit / 14) × 0.5 + min(total_files / 20, 1.0) × 0.5
```

---

## Theoretical Grounding

- Severity scores: CVSS v3.1 (FIRST, 2019)
- Intent mapping: MITRE ATT\&CK (MITRE, 2024)
- Motivation framework: Diamond Model of Intrusion Analysis (Caltagirone et al., 2013)

---

## Project Context

Built as an original research contribution for a final-year graduation project in AI and Data Science. The model, activation function, routing mechanism, and behavioral findings are original work. The architecture descends from MACRO-DREADNOUGHT (computer vision) — the forensic bus pattern and orthogonal weight penalty both originate there.

**Author:** Mohammad Khalid AL-Biltaji — Zarqa University, June 2026
