"""
╔══════════════════════════════════════════════════════════════════════╗
║              PROJECT LEO — FULL UNIFIED PIPELINE                    ║
║          BehavioralMacroDreadnought Inference System                 ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE THIS FILE:
    Search for the tag  # ── INTEGRATION POINT  ──   to find every place
    your team needs to plug in files, logs, or directories.

    There are exactly 4 integration points:
        # ── INTEGRATION POINT 1 ── -1  →  Path to trained model weights (.pth)
        # ── INTEGRATION POINT 1 ── -2  →  Directory containing attacker profile JSONs
        # ── INTEGRATION POINT 1 ── -3  →  Directory to save output reports
        # ── INTEGRATION POINT 1 ── -4  →  The session JSON file or raw records to analyze

PIPELINE ORDER:
    Phase 0  [P]  →  Raw JSON → SessionBatch
    Goal 1   [Q][R][S][T]  →  Injection scan + severity + rhythm
    Model    [A-N]  →  BehavioralMacroDreadnought inference
    Goal 2   [V][X]  →  Tarpit file access + intent profiling
    Goal 3   [Y]  →  Final incident report synthesis

TEAM INTEGRATION:
    The cyber team produces 4 attacker profile JSON files.
    Place them all in the same directory (# ── INTEGRATION POINT ── 2).
    Each file follows this naming convention:
        advance_cowrie_attacker_profile.json
        advance_honeypot_2_attacker_profile.json
        intermediate_cowrie_attacker_profile.json
        script_kiddie_cowrie_attacker_profile.json

    Each record in those files must have:
        src_ip, timestamp, src_machine,
        files_taken (14 categories), total_files_taken

THEORETICAL GROUNDING:
    Severity scores:   CVSS v3.1 (FIRST, 2019)
    Intent mapping:    MITRE ATT&CK (MITRE, 2024)
    Motivation:        Diamond Model (Caltagirone et al., 2013)
    Rhythm analysis:   Behavioral biometrics literature
"""

import os
import re
import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

# ── Optional torch import ──────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[LEO] PyTorch not found. Model inference disabled.")
    print("[LEO] Injection flagging and tarpit analysis still fully operational.")


# ══════════════════════════════════════════════════════════════════════
#
#  # ── INTEGRATION POINT 1 ── 
#  MODEL WEIGHTS PATH
#  Point this to my macro_dreadnought_best.pth file.
#  This is the trained model produced by benchmark_v2.py on Kaggle.
#
MODEL_PATH = "macro_dreadnought_best.pth"
#
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#
#  # ── INTEGRATION POINT  ── 2
#  ATTACKER PROFILE DIRECTORY
#  Point this to the folder containing the 4 JSON files
#  produced by the cyber team's tarpit logging system.
#  Required files:
#      advance_cowrie_attacker_profile.json
#      advance_honeypot_2_attacker_profile.json
#      intermediate_cowrie_attacker_profile.json
#      script_kiddie_cowrie_attacker_profile.json
#
PROFILES_DIR = "attacker_profiles/"
#
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#
#  # ── INTEGRATION POINT  ── 3
#  REPORT OUTPUT DIRECTORY
#  Reports are saved here as plain text files.
#  One file per session: leo_report_{session_id}.txt
#
OUTPUT_DIR = "reports/"
#
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════

SEQ_LEN      = 256
HISTORY_LEN  = 10
NUM_TOOLS    = 5
UNIFORM_BASE = 0.45
DEVICE       = "cpu"   # change to "cuda" if GPU available

EVENT_COMMAND    = "command"
EVENT_CONNECTION = "connection"

CLASS_LABELS  = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
TOOL_NAMES    = ["sqlmap", "hydra", "metasploit", "nikto", "custom"]
TOOL_VOCAB    = ["sqlmap", "hydra", "metasploit", "nikto", "custom"]

EXPERT_LABELS = {
    0: "Expert-1 GELU (Beginner specialist)",
    1: "Expert-2 Mish (Intermediate specialist)",
    2: "Expert-3 SpLR_V2 (Advanced / zero-day specialist)",
}

EXPERT_INTERPRETATIONS = {
    (0, False): "Confident beginner pattern. High-volume deterministic attack. Likely automated tooling.",
    (0, True):  "Expert-1 under uncertainty. Cross-reference injection analysis.",
    (1, False): "Confident intermediate pattern. Some customization. Hybrid manual/automated.",
    (1, True):  "Expert-2 under uncertainty. Mixed signals. Check session rhythm.",
    (2, False): "Expert-3 HIGH CONFIDENCE. Novel/zero-day pattern. SpLR_V2 gate CLOSED — clean advanced signal. Escalate immediately.",
    (2, True):  "Expert-3 HIGH ENTROPY. Gate OPEN — chaos flooding through. LOW confidence. Weight injection and tarpit signals more heavily.",
}


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODULE [N]: SpLR_V2 — Custom Entropy-Gated Activation Function
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class SpLR_V2(nn.Module):
        """
        Custom entropy-gated activation function for Expert 3 [F.3].
        4 learnable parameters per channel:
            a — Amplitude Overclock    (init: 1.1)
            b — Gaussian Filter Width  (init: 0.5)
            c — Survival Wire          (init: 0.1)
            d — Panic Multiplier       (init: 0.1)
        d can go negative — toxic noise channels close during panic.
        """
        def __init__(self, channels):
            super().__init__()
            self.a = nn.Parameter(torch.ones(1, channels) * 1.1)
            self.b = nn.Parameter(torch.ones(1, channels) * 0.5)
            self.c = nn.Parameter(torch.ones(1, channels) * 0.1)
            self.d = nn.Parameter(torch.ones(1, channels) * 0.1)

        def forward(self, x, entropy=None):
            safe_b      = torch.abs(self.b) + 1e-4
            effective_c = self.c
            if entropy is not None:
                clamped     = torch.clamp(entropy, 0, 0.99)
                safe_b      = safe_b * (1.0 - clamped).view(-1, 1)
                effective_c = self.c + (self.d * clamped.view(-1, 1))
            return self.a * x * torch.exp(-safe_b * x.pow(2)) + effective_c * x


    # ══════════════════════════════════════════════════════════════════
    # MODULE [A]: LexicalEncoder — Character-Level ASCII Transformer
    # ══════════════════════════════════════════════════════════════════

    class LexicalEncoder(nn.Module):
        def __init__(self, vocab_size=256, d_model=512,
                     max_len=256, num_heads=16, num_layers=3):
            super().__init__()
            self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=d_model * 4,
                batch_first=True, activation='gelu'
            )
            self.transformer_stack = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, token_tensor):
            emb = self.embedding(token_tensor) + self.pos_encoding
            out = self.transformer_stack(emb)
            return out.mean(dim=1)


    # ══════════════════════════════════════════════════════════════════
    # MODULE [A-N]: BehavioralMacroDreadnought — Full MoE Architecture
    # ══════════════════════════════════════════════════════════════════

    class BehavioralMacroDreadnought(nn.Module):
        """
        Mixture-of-Experts behavioral analysis model.
        Modules: [A] LexicalEncoder, [B] TemporalStack, [D] Scalar Branch,
                 [E] Entropy, [G] Router, [F.1-3] Experts, [H] Identity Gate,
                 [J] Output Heads, [K][L][M] Regularization (training only)
        """
        def __init__(self, d_model=512, seq_len=256, history_len=10,
                     num_heads=16, d_hidden=4096, num_tools=5,
                     num_layers=3, uniform_base=0.45):
            super().__init__()
            self.history_len  = history_len
            self.d_model      = d_model
            self.uniform_base = uniform_base

            # [A] LexicalEncoder
            self.lexical_branch = LexicalEncoder(
                d_model=d_model, max_len=seq_len,
                num_heads=num_heads, num_layers=num_layers
            )
            # [B] TemporalStack
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                batch_first=True, activation='gelu'
            )
            self.temporal_stack = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)

            # [D] Scalar Branch — input_dim = 512 + 2
            self.input_dim = d_model + 2

            # [G] Uniform-Base Router
            self.router = nn.Linear(self.input_dim, 3)

            # [F.1] Expert 1 — GELU (Beginner specialist)
            self.expert_1 = nn.Sequential(
                nn.Linear(self.input_dim, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_hidden)
            )
            # [F.2] Expert 2 — Mish (Intermediate specialist)
            self.expert_2 = nn.Sequential(
                nn.Linear(self.input_dim, d_hidden), nn.Mish(),
                nn.Linear(d_hidden, d_hidden)
            )
            # [F.3] Expert 3 — SpLR_V2 (Advanced / zero-day specialist)
            self.e3_linear1 = nn.Linear(self.input_dim, d_hidden)
            self.e3_splr    = SpLR_V2(channels=d_hidden)  # [N]
            self.e3_linear2 = nn.Linear(d_hidden, d_hidden)

            # [J] Output Heads
            self.head_threat = nn.Sequential(
                nn.Linear(d_hidden, 128), nn.ReLU(),
                nn.Linear(128, 1), nn.Sigmoid()
            )
            self.head_class = nn.Sequential(
                nn.Linear(d_hidden, 128), nn.ReLU(),
                nn.Linear(128, 3)
            )
            self.head_tools = nn.Sequential(
                nn.Linear(d_hidden, 128), nn.ReLU(),
                nn.Linear(128, num_tools)
            )

        def forward(self, scalar_vec, temporal_token_tensor):
            b, h, t = temporal_token_tensor.shape

            # [A] Lexical encoding
            lex_out         = self.lexical_branch(temporal_token_tensor.view(-1, t))
            session_history = lex_out.view(b, h, self.d_model)

            # [B] Temporal encoding
            temporal_out    = self.temporal_stack(session_history)
            final_history   = temporal_out.mean(dim=1)
            temporal_var    = temporal_out.var(dim=1).mean()

            # [D] Scalar merge
            merged = torch.cat([final_history, scalar_vec], dim=1)

            # [G] Uniform-base routing
            raw_probs     = F.softmax(self.router(merged), dim=-1)
            routing_probs = (1.0 - self.uniform_base) * raw_probs + (self.uniform_base / 3.0)

            # [E] Entropy calculation
            raw_entropy     = -(routing_probs * torch.log(routing_probs + 1e-8)).sum(dim=1)
            routing_entropy = raw_entropy / np.log(3)

            # Top-1 selection
            top1_indices = torch.argmax(routing_probs, dim=-1)

            # [F.1][F.2][F.3] All three experts execute
            e1_out    = self.expert_1(merged)
            e2_out    = self.expert_2(merged)
            e3_hidden = self.e3_linear1(merged)
            e3_act    = self.e3_splr(e3_hidden, entropy=routing_entropy)
            e3_out    = self.e3_linear2(e3_act)

            expert_outputs = torch.stack([e1_out, e2_out, e3_out], dim=1)
            selected       = expert_outputs[torch.arange(b).to(merged.device), top1_indices]

            # [H] Dynamic Identity Gate
            entropy_gate  = (1.0 - routing_entropy).unsqueeze(-1)
            safe_fallback = (e1_out + e2_out + e3_out) / 3.0
            global_state  = safe_fallback + (selected - safe_fallback) * entropy_gate

            # [J] Output heads
            threat_score = self.head_threat(global_state)
            class_logits = self.head_class(global_state)
            tool_logits  = self.head_tools(global_state)

            return (
                global_state, routing_probs, threat_score,
                class_logits, tool_logits,
                routing_entropy, top1_indices, temporal_var
            )


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODULE [P]: LogRouter — Phase 0 Live Session Serializer
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SerializedRecord:
    session_id:        str
    timestamp:         str
    src_machine:       str
    log_tag:           str
    payload:           str
    time_delta:        float
    has_custom_tool:   int
    tool_0:            int
    tool_1:            int
    tool_2:            int
    tool_3:            int
    tool_4:            int
    event_type:        str
    raw_record:        dict
    is_advance_attack: bool


@dataclass
class SessionBatch:
    session_id:    str
    src_ip:        str
    src_machine:   str
    records:       List[SerializedRecord]
    time_deltas:   List[float]
    record_count:  int
    command_count: int

    def get_payloads(self) -> List[str]:
        return [r.payload for r in self.records]

    def get_commands_only(self) -> List[SerializedRecord]:
        return [r for r in self.records if r.event_type == EVENT_COMMAND]

    def to_model_input(self) -> dict:
        last = self.records[-1]
        return {
            "payloads":        self.get_payloads(),
            "time_delta":      last.time_delta,
            "has_custom_tool": last.has_custom_tool,
            "session_id":      self.session_id,
        }


def _detect_log_tag(record: dict) -> str:
    src = record.get("src_machine", "").lower().strip()
    if src in ("cowrie", "cowrie-docker"):
        return "[COWRIE]"
    if src == "honeypot_2":
        return "[HONEYPOT]"
    if src == "main_system":
        return "[NETWORK]" if "log_src" in record else "[APP]"
    return "[UNKNOWN]"


def _detect_event_type(record: dict, log_tag: str) -> str:
    if log_tag in ("[COWRIE]", "[HONEYPOT]"):
        return EVENT_CONNECTION if not str(record.get("command", "")).strip() else EVENT_COMMAND
    return EVENT_COMMAND


def _extract_scalars(record: dict) -> dict:
    try:
        time_delta = float(record.get("time_delta", 0.0))
    except (ValueError, TypeError):
        time_delta = 0.0
    attack_tool     = str(record.get("attack_tool", "unknown")).lower().strip()
    has_custom_tool = 0 if attack_tool in ("unknown", "none", "") else 1
    return {"time_delta": time_delta, "has_custom_tool": has_custom_tool}


def _extract_tool_labels(record: dict) -> dict:
    attack_tool = str(record.get("attack_tool", "unknown")).lower().strip()
    labels = {}
    for i, tool in enumerate(TOOL_VOCAB):
        if tool == "custom":
            labels[f"tool_{i}"] = 1 if (
                attack_tool not in ("unknown", "none", "") and
                not any(t in attack_tool for t in TOOL_VOCAB[:-1])
            ) else 0
        else:
            labels[f"tool_{i}"] = 1 if tool in attack_tool else 0
    return labels


def _extract_post_params(record: dict) -> str:
    params = record.get("parameters", {})
    if not params:
        return ""
    post = params.get("POST", {})
    if not post or not isinstance(post, dict):
        return ""
    pairs = [f"{k}={str(v).strip()}" for k, v in post.items() if v is not None]
    return ("POST:" + " ".join(pairs)) if pairs else ""


def _build_payload(record: dict, log_tag: str, event_type: str) -> str:
    if log_tag in ("[COWRIE]", "[HONEYPOT]"):
        if event_type == EVENT_CONNECTION:
            return f"{log_tag} CMD:connection_attempt"
        command = str(record.get("command", "")).strip()
        user    = str(record.get("user", "")).strip()
        if user and log_tag == "[HONEYPOT]":
            return f"{log_tag} CMD:{command} USER:{user}"
        return f"{log_tag} CMD:{command}"

    if log_tag == "[NETWORK]":
        method      = str(record.get("method", "")).strip()
        uri         = str(record.get("request_uri", "")).strip()
        attack_type = str(record.get("attack_type", "")).strip().lower()
        parts = []
        if method and uri:
            parts.append(f"{method} {uri}")
        if attack_type not in ("normal", "none", "unknown", ""):
            parts.append(f"ATTACK:{attack_type}")
        is_upload = record.get("is_fileUpload", False)
        if is_upload is True or str(is_upload).lower() == "true":
            parts.append("FILEUPLOADED:unknown")
        return f"{log_tag} {' '.join(parts)}".strip()

    if log_tag == "[APP]":
        method      = str(record.get("method", "")).strip()
        uri         = str(record.get("request_uri", "")).strip()
        event       = str(record.get("event", "")).strip().lower()
        attack_type = str(record.get("attack_type", "")).strip().lower()
        username    = str(record.get("username", "")).strip()
        parts = []
        if method and uri:
            parts.append(f"{method} {uri}")
        if event not in ("normal", "none", "unknown", ""):
            parts.append(f"EVENT:{event}")
        if attack_type not in ("normal", "none", "unknown", ""):
            parts.append(f"ATTACK:{attack_type}")
        if username not in ("unknown", "none", ""):
            parts.append(f"USER:{username}")
        post_str = _extract_post_params(record)
        if post_str:
            parts.append(post_str)
        return f"{log_tag} {' '.join(parts)}".strip()

    command = str(record.get("command", "")).strip()
    uri     = str(record.get("request_uri", "")).strip()
    return f"{log_tag} {command or uri or 'unknown_event'}"


def assemble_session(raw_records: List[dict]) -> Optional[SessionBatch]:
    """Phase 0 entry point — converts raw JSON list to SessionBatch."""
    if not raw_records:
        return None
    try:
        sorted_records = sorted(raw_records, key=lambda r: r.get("timestamp", ""))
    except Exception:
        sorted_records = raw_records

    serialized    = []
    time_deltas   = []
    command_count = 0

    for record in sorted_records:
        log_tag    = _detect_log_tag(record)
        event_type = _detect_event_type(record, log_tag)
        scalars    = _extract_scalars(record)
        tools      = _extract_tool_labels(record)
        payload    = _build_payload(record, log_tag, event_type)
        src_ip     = str(record.get("src_ip", "unknown")).strip()
        session    = str(record.get("session", "0")).strip()
        session_id = f"{src_ip}_{session}"
        adv_val    = record.get("is_Advance_Attack", False)
        is_advance = adv_val if isinstance(adv_val, bool) else str(adv_val).lower() == "true"

        sr = SerializedRecord(
            session_id=session_id, timestamp=str(record.get("timestamp", "")),
            src_machine=str(record.get("src_machine", "unknown")),
            log_tag=log_tag, payload=payload,
            time_delta=scalars["time_delta"],
            has_custom_tool=scalars["has_custom_tool"],
            tool_0=tools["tool_0"], tool_1=tools["tool_1"],
            tool_2=tools["tool_2"], tool_3=tools["tool_3"],
            tool_4=tools["tool_4"], event_type=event_type,
            raw_record=record, is_advance_attack=is_advance,
        )
        serialized.append(sr)
        time_deltas.append(scalars["time_delta"])
        if event_type == EVENT_COMMAND:
            command_count += 1

    if not serialized:
        return None

    first = serialized[0]
    return SessionBatch(
        session_id=first.session_id,
        src_ip=str(sorted_records[0].get("src_ip", "unknown")),
        src_machine=str(sorted_records[0].get("src_machine", "unknown")),
        records=serialized, time_deltas=time_deltas,
        record_count=len(serialized), command_count=command_count,
    )


def load_session_from_file(filepath: str) -> Optional[SessionBatch]:
    """Phase 0 — load from JSON or JSONL file on disk."""
    records = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
                if isinstance(data, list):
                    records = [r for r in data if isinstance(r, dict)]
                elif isinstance(data, dict):
                    for key in ("root", "data", "records", "logs"):
                        if key in data and isinstance(data[key], list):
                            records = data[key]
                            break
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj.get("root", obj))
                        elif isinstance(obj, list):
                            records.extend([r for r in obj if isinstance(r, dict)])
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(f"[Phase0] File not found: {filepath}")
        return None
    except Exception as e:
        print(f"[Phase0] Read error: {e}")
        return None
    return assemble_session(records)


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODULE [Q]: PatternScanner — Attack Signatures
# MODULE [R]: SeverityEngine — CVSS-Anchored Scoring
# MODULE [S]: LabelAuditor   — Team Label Agreement
# MODULE [T]: RhythmAnalyzer — Human vs Automation
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

# [R] Base scores — serializer threat weights * 100, validated against CVSS v3.1
BASE_SCORES = {
    "reconnaissance": 30, "bruteforce": 40, "xss": 50,
    "sqli": 60, "lfi": 60, "rfi": 70, "traversal": 70,
    "rce": 90, "command_injection": 90,
}

# [R] Multipliers
MULTIPLIERS = {
    "chaining": 8, "evasion": 10, "destructive": 15, "reverse_shell": 20,
}

# [T] Rhythm thresholds — low std catches fake-slow tools regardless of mean
RHYTHM_THRESHOLDS = {
    "automation_mean_ceiling": 0.1,
    "automation_std_ceiling":  0.3,
    "human_mean_floor":        2.0,
    "human_std_floor":         1.5,
    "min_commands":            3,
}

# [Q] Shell pattern set
SHELL_PATTERNS = {
    "command_injection": [
        (r"[;|&`]\s*(id|whoami|uname|hostname|ifconfig|ip\s+a)", "Chained recon command"),
        (r"[;|&`]\s*cat\s+/",                                    "Chained file read"),
        (r"(bash\s+-[ic]|sh\s+-[ic]|/bin/sh|/bin/bash)",        "Shell spawn attempt"),
        (r"(crontab\s+-|/etc/cron\.|at\s+now)",                 "Persistence via scheduler"),
        (r"(chmod\s+[0-7]{3,4}|chown\s+root)",                  "Permission manipulation"),
        (r"(rm\s+-rf\s+/|shred\s+|dd\s+if=)",                   "Destructive disk command"),
        (r"(iptables|ufw)\s+.*(delete|drop|flush|-F\b)",        "Firewall manipulation"),
    ],
    "rce": [
        (r"(mkfifo|mknod)\s+/",                               "Named pipe creation"),
        (r"python[23]?\s+-c\s+['\"]import\s+socket",          "Python reverse shell"),
        (r"perl\s+-e\s+['\"]use\s+Socket",                    "Perl reverse shell"),
        (r"(wget|curl)\s+https?://[^\s]+\.(sh|py|pl|elf|exe)","Remote executable download"),
        (r"chmod\s+\+x.*&&|&&.*chmod\s+\+x",                  "Download and execute chain"),
    ],
    "reconnaissance": [
        (r"\b(whoami|id|uname\s+-a|hostname|ifconfig|ip\s+a)\b","System enumeration"),
        (r"\b(nmap|masscan|netstat|ss\s+-|arp\s+-)\b",          "Network reconnaissance"),
        (r"\b(ps\s+aux|w\b|who\b|last\b|lastlog)\b",            "Process/user enumeration"),
        (r"\b(ls\s+-la|find\s+/|locate\s+|which\s+)\b",         "Filesystem exploration"),
        (r"cat\s+(/etc/passwd|/etc/shadow|/etc/hosts|/proc/)",  "Sensitive file read"),
    ],
    "lfi": [
        (r"(\.\.\/){2,}",                                "Directory traversal sequence"),
        (r"(\.\.%2f){2,}|(\.\.%5c){2,}",                "URL-encoded traversal"),
        (r"/etc/(passwd|shadow|hosts|crontab|sudoers)",  "Direct system file access"),
        (r"/proc/(self|[0-9]+)/(environ|cmdline|maps)", "Proc filesystem read"),
    ],
    "traversal": [
        (r"(c:\\\\windows\\\\|c:/windows/)",   "Windows path traversal"),
        (r"\.(env|config|cfg|ini|bak|old|swp)$","Config/backup file access"),
        (r"/var/(log|www|run|lib)/",           "System data directory access"),
    ],
}

# [Q] Web pattern set
WEB_PATTERNS = {
    "sqli": [
        (r"(?i)union\b.{0,30}\bselect\b",                          "UNION SELECT exfiltration"),
        (r"(?i)(\bor\b|\band\b)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+","Boolean tautology"),
        (r"(?i)(sleep\s*\(\s*\d+|benchmark\s*\()",                 "Time-based blind SQLi"),
        (r"(?i)(drop|truncate|delete)\s+\btable\b",               "Destructive DDL injection"),
        (r"(--\s*$|;\s*--|#\s*$)",                                 "SQL comment termination"),
        (r"(?i)(information_schema|sys\.tables)",                  "Schema enumeration"),
        (r"(?i)(load_file\s*\(|into\s+outfile)",                   "File read/write via SQL"),
    ],
    "xss": [
        (r"(?i)<script[^>]*>",                          "Script tag injection"),
        (r"(?i)(onerror|onload|onclick|onmouseover)\s=","Event handler injection"),
        (r"(?i)javascript\s*:",                          "JavaScript URI scheme"),
        (r"(?i)(document\.cookie|window\.location)",    "Cookie/redirect exfiltration"),
        (r"(?i)%3cscript|%3e.*%3c/script",              "URL-encoded script tag"),
    ],
    "lfi": [
        (r"(\.\.\/){2,}",                         "Directory traversal in URI"),
        (r"(?i)(php://filter|php://input)",        "PHP stream wrapper"),
        (r"(?i)/etc/(passwd|shadow|hosts)",        "Sensitive file via URI"),
    ],
    "rfi": [
        (r"(?i)https?://[a-z0-9./-]+\.(sh|php|pl|py|exe|elf)","Remote executable fetch"),
        (r"(?i)(include|require)(_once)?\s*\(\s*['\"]https?:", "PHP remote include"),
        (r"(?i)\?.*=(https?://|ftp://)",                       "URL parameter remote include"),
    ],
    "traversal": [
        (r"(\.\.\/){2,}",                              "Path traversal in URI"),
        (r"(\.\.%2f){2,}|(\.\.%5c){2,}",              "URL-encoded path traversal"),
        (r"(?i)\.(env|config|cfg|ini|bak|old|swp)(\?|$)","Config/backup file request"),
    ],
    "bruteforce": [
        (r"EVENT:login_attempt",                                           "Login attempt event"),
        (r"(?i)POST:.*username=(admin|root|test|guest|administrator)",     "Default credential attempt"),
        (r"(?i)POST:.*password=(\d{4,8}|password|123456|qwerty)",         "Common password attempt"),
    ],
    "reconnaissance": [
        (r"(?i)(robots\.txt|sitemap\.xml|\.git/|\.svn/)", "Web recon file access"),
        (r"(?i)/(admin|administrator|wp-admin|phpmyadmin)","Admin panel probe"),
        (r"(?i)\.(bak|old|backup|swp|~)(\?|$)",           "Backup file probe"),
    ],
}

# [R] Multiplier patterns
MULTIPLIER_PATTERNS = {
    "reverse_shell": [
        (r"(mkfifo|mknod)\s+/",       "Named pipe"),
        (r"python[23]?\s+-c.*socket", "Python socket"),
        (r"perl\s+-e.*Socket",        "Perl socket"),
        (r"bash\s+-i\s+>&?",          "Bash redirect shell"),
        (r"/dev/tcp/",                "Bash TCP redirect"),
    ],
    "evasion": [
        (r"(\.\.%2f|%2e%2e|%252e)",   "URL encoding evasion"),
        (r"0x[0-9a-fA-F]{4,}",        "Hex encoding"),
        (r"base64\s+-d|base64_decode", "Base64 encoding"),
        (r"\\x[0-9a-fA-F]{2}",        "Hex escape evasion"),
    ],
    "destructive": [
        (r"rm\s+-rf\s+/",         "Recursive delete root"),
        (r"(shred|wipe)\s+",      "Secure delete"),
        (r"dd\s+if=.*of=",        "Disk overwrite"),
        (r"(?i)DROP\s+TABLE",     "SQL table drop"),
        (r"(?i)TRUNCATE\s+TABLE", "SQL table truncate"),
    ],
}

# Compile all patterns once at import
def _compile(pd):
    return {k: [(re.compile(p), d) for p, d in v] for k, v in pd.items()}

SHELL_C      = _compile(SHELL_PATTERNS)
WEB_C        = _compile(WEB_PATTERNS)
MULTIPLIER_C = _compile(MULTIPLIER_PATTERNS)


@dataclass
class MatchDetail:
    attack_type:  str
    base_score:   int
    description:  str
    matched_text: str


@dataclass
class InjectionResult:
    raw_payload:        str
    session_id:         str
    time_delta:         float
    source_machine:     str
    scan_type:          str
    event_type:         str
    timestamp:          str
    attack_types_found: List[str]
    matches:            List[MatchDetail] = field(default_factory=list)
    label_agreement:    str = "no_label"
    severity_score:     int = 0
    session_rhythm:     str = "unknown"

    @property
    def is_malicious(self) -> bool:
        return len(self.attack_types_found) > 0

    @property
    def is_connection_event(self) -> bool:
        return self.event_type == EVENT_CONNECTION


def _scan_patterns(payload: str, scan_type: str) -> List[MatchDetail]:
    pattern_set = SHELL_C if scan_type == "shell" else WEB_C
    matches, seen = [], set()
    for attack_type, compiled_list in pattern_set.items():
        for compiled_re, description in compiled_list:
            m = compiled_re.search(payload)
            if m:
                key = (attack_type, description)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(MatchDetail(
                    attack_type=attack_type,
                    base_score=BASE_SCORES.get(attack_type, 30),
                    description=description,
                    matched_text=m.group(0)[:120],
                ))
    matches.sort(key=lambda x: x.base_score, reverse=True)
    return matches


def _calculate_severity(matches: List[MatchDetail], payload: str) -> int:
    if not matches:
        return 0
    score = matches[0].base_score
    unique_types = list(dict.fromkeys(m.attack_type for m in matches))
    if len(unique_types) > 1:
        score += MULTIPLIERS["chaining"] * (len(unique_types) - 1)
    for condition, compiled_list in MULTIPLIER_C.items():
        for compiled_re, _ in compiled_list:
            if compiled_re.search(payload):
                score += MULTIPLIERS[condition]
                break
    return max(1, min(100, score))


def _check_label_agreement(payload: str, attack_types_found: List[str]) -> str:
    m = re.search(r"ATTACK:(\w+)", payload)
    if not m:
        return "no_label"
    team_label = m.group(1).lower().strip()
    if team_label in ("none", "normal", "unknown"):
        return "no_label"
    return "agree" if team_label in attack_types_found else "disagree"


def analyze_rhythm(time_deltas: List[float]) -> str:
    """[T] Module — classify session timing as automated/human/hybrid/unknown."""
    if len(time_deltas) < RHYTHM_THRESHOLDS["min_commands"]:
        return "unknown"
    deltas = np.array(time_deltas, dtype=float)
    mean, std = float(deltas.mean()), float(deltas.std())
    if mean < RHYTHM_THRESHOLDS["automation_mean_ceiling"]:
        return "automated"
    if std < RHYTHM_THRESHOLDS["automation_std_ceiling"]:
        return "automated"
    if mean >= RHYTHM_THRESHOLDS["human_mean_floor"] and std >= RHYTHM_THRESHOLDS["human_std_floor"]:
        return "human"
    return "hybrid"


def scan_payload(payload: str, scan_type: str, session_id: str,
                 time_delta: float, timestamp: str, src_machine: str,
                 log_tag: str, event_type: str) -> InjectionResult:
    """[Q][R][S] — scan one payload and return InjectionResult."""
    if event_type == EVENT_CONNECTION:
        return InjectionResult(
            raw_payload=payload, session_id=session_id, time_delta=time_delta,
            source_machine=src_machine, scan_type=scan_type, event_type=event_type,
            timestamp=timestamp, attack_types_found=[], matches=[],
            label_agreement="no_label", severity_score=0, session_rhythm="unknown",
        )
    matches            = _scan_patterns(payload, scan_type)
    attack_types_found = list(dict.fromkeys(m.attack_type for m in matches))
    severity_score     = _calculate_severity(matches, payload)
    label_agreement    = _check_label_agreement(payload, attack_types_found)
    return InjectionResult(
        raw_payload=payload, session_id=session_id, time_delta=time_delta,
        source_machine=src_machine, scan_type=scan_type, event_type=event_type,
        timestamp=timestamp, attack_types_found=attack_types_found,
        matches=matches, label_agreement=label_agreement,
        severity_score=severity_score, session_rhythm="unknown",
    )


def scan_session(batch: SessionBatch) -> List[InjectionResult]:
    """[Q][R][S][T] — scan full session, inject rhythm into all results."""
    results = []
    for record in batch.records:
        scan_type = "shell" if record.log_tag in ("[COWRIE]", "[HONEYPOT]") else "web"
        result = scan_payload(
            payload=record.payload, scan_type=scan_type,
            session_id=record.session_id, time_delta=record.time_delta,
            timestamp=record.timestamp, src_machine=record.src_machine,
            log_tag=record.log_tag, event_type=record.event_type,
        )
        results.append(result)
    rhythm = analyze_rhythm(batch.time_deltas)
    for r in results:
        r.session_rhythm = rhythm
    return results


def session_summary(results: List[InjectionResult]) -> dict:
    """Aggregate Goal 1 results for Goal 3 consumption."""
    malicious = [r for r in results if r.is_malicious]
    max_sev   = max((r.severity_score for r in results), default=0)
    all_types = list(dict.fromkeys(t for r in results for t in r.attack_types_found))
    rhythm    = results[0].session_rhythm if results else "unknown"
    agreements = {"agree": 0, "disagree": 0, "no_label": 0}
    for r in results:
        agreements[r.label_agreement] = agreements.get(r.label_agreement, 0) + 1
    critical = [r.raw_payload for r in results if r.severity_score >= 80]
    return {
        "total_records":     len(results),
        "malicious_count":   len(malicious),
        "clean_count":       len(results) - len(malicious),
        "max_severity":      max_sev,
        "attack_types":      all_types,
        "session_rhythm":    rhythm,
        "label_agreements":  agreements,
        "critical_payloads": critical,
    }


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODULE [V]: TarpitReader — Load Attacker Profile Files
# MODULE [X]: IntentProfiler — MITRE ATT&CK + Diamond Model
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

FILE_CATEGORIES = [
    "credentials", "Malware / Dropped Tools", "Data Exfiltration",
    "Network Reconnaissance", "System / OS Files", "Infrastructure",
    "SQL / Database", "Logs", "Financial", "Operational Notes",
    "Scripts / Tools", "Personal Data", "Tokens / API Keys", "Media / Documents",
]

# [X] MITRE ATT&CK weights — MITRE Corporation (2024)
INTENT_WEIGHTS = {
    "credentials":              {"credential_access": 0.7, "persistence": 0.2, "exfiltration": 0.1},
    "Malware / Dropped Tools":  {"persistence": 0.4, "impact": 0.2, "defense_evasion": 0.2, "financial": 0.2},
    "Data Exfiltration":        {"exfiltration": 0.9, "impact": 0.1},
    "Network Reconnaissance":   {"discovery": 0.8, "persistence": 0.2},
    "System / OS Files":        {"discovery": 0.6, "credential_access": 0.2, "defense_evasion": 0.2},
    "Infrastructure":           {"discovery": 0.7, "impact": 0.3},
    "SQL / Database":           {"collection": 0.5, "exfiltration": 0.4, "impact": 0.1},
    "Logs":                     {"defense_evasion": 0.8, "discovery": 0.2},
    "Financial":                {"impact": 0.7, "exfiltration": 0.3},
    "Operational Notes":        {"collection": 0.6, "discovery": 0.4},
    "Scripts / Tools":          {"persistence": 0.5, "credential_access": 0.3, "defense_evasion": 0.2},
    "Personal Data":            {"exfiltration": 0.6, "impact": 0.4},
    "Tokens / API Keys":        {"credential_access": 0.6, "persistence": 0.3, "exfiltration": 0.1},
    "Media / Documents":        {"exfiltration": 0.7, "collection": 0.3},
}

TACTIC_LABELS = {
    "credential_access": "Credential Access (TA0006)",
    "persistence":       "Persistence (TA0003)",
    "discovery":         "Discovery (TA0007)",
    "defense_evasion":   "Defense Evasion (TA0005)",
    "collection":        "Collection (TA0009)",
    "exfiltration":      "Exfiltration (TA0010)",
    "impact":            "Impact (TA0040)",
    "financial":         "Impact / Financial (TA0040)",
}

CONFIDENCE_THRESHOLDS = {
    "HIGH":   {"min_categories": 4, "min_files": 10},
    "MEDIUM": {"min_categories": 2, "min_files": 4},
}

FILENAME_INSIGHTS = {
    "id_rsa":           ("lateral_movement",   "SSH private key — enables access to other systems"),
    ".ssh":             ("lateral_movement",   "SSH directory — keys for remote access"),
    ".env":             ("lateral_movement",   "Environment file — service credentials"),
    ".pam_update.sh":  ("persistence",        "PAM module disguised as system update — backdoor"),
    "rootkit.tar.gz":  ("persistence",        "Rootkit archive — long-term hidden access"),
    "miner.py":         ("financial",          "Cryptomining script — resource exploitation"),
    ".all_loot.tar.gz":("exfiltration",       "Packaged archive — systematic bulk exfiltration"),
    "exfil.tar.gz":    ("exfiltration",       "Exfiltration archive — data staged for theft"),
    ".bash_history":   ("defense_evasion",    "Command history — covering tracks"),
    ".emp_roster":     ("discovery",          "Employee roster — mapping human targets"),
    "/proc/net/arp":   ("discovery",          "ARP table — mapping live network hosts"),
    "/etc/passwd":     ("credential_access",  "User database — account enumeration"),
    "/etc/shadow":     ("credential_access",  "Password hashes — offline cracking target"),
    "passwords.txt":   ("credential_access",  "Plain text password file — opportunistic grab"),
    "/proc/cpuinfo":   ("financial",          "CPU info — verifying mining capability"),
}


@dataclass
class TarpitProfile:
    src_ip:           str
    timestamp:        str
    src_machine:      str
    files_taken:      Dict[str, List[str]]
    total_files:      int
    categories_hit:   List[str]
    intent_profile:   Dict[str, float]
    primary_intent:   str
    motivation:       dict
    skill_indicators: dict
    confidence:       str


def _compute_intent(files_taken, categories_hit, total_files):
    if total_files == 0:
        return {}
    raw: Dict[str, float] = {}
    for cat in categories_hit:
        w = len(files_taken.get(cat, [])) / total_files
        for tactic, tw in INTENT_WEIGHTS.get(cat, {}).items():
            raw[tactic] = raw.get(tactic, 0.0) + w * tw
    total = sum(raw.values())
    if total == 0:
        return {}
    return {t: round((s / total) * 100, 1) for t, s in sorted(raw.items(), key=lambda x: x[1], reverse=True)}


def _motivation(files_taken, categories_hit, total_files):
    all_files = [f.lower() for fl in files_taken.values() for f in fl]
    if (any(f in all_files for f in ["id_rsa", ".ssh", ".env", ".live_creds"]) and
            "Network Reconnaissance" in categories_hit):
        return {"label": "APT — Lateral Movement",
                "conclusion": "Attacker used this system as a pivot point to reach other network targets.",
                "explanation": "SSH private keys and environment credentials combined with network reconnaissance indicate lateral movement preparation (MITRE ATT&CK TA0008).",
                "mitre_ref": ["TA0006", "TA0007", "TA0008"]}
    if any(f in all_files for f in ["miner.py", "xmrig", "minerd"]):
        return {"label": "Financial — Cryptomining",
                "conclusion": "Attacker deployed mining software for financial gain.",
                "explanation": "Mining scripts with CPU verification indicate resource exploitation (MITRE ATT&CK TA0040).",
                "mitre_ref": ["TA0003", "TA0040"]}
    if (any(f in all_files for f in ["rootkit.tar.gz", ".pam_update.sh", "sys_updates"]) and
            "Logs" in categories_hit):
        return {"label": "Persistence & Stealth",
                "conclusion": "Attacker prioritized long-term hidden access.",
                "explanation": "Rootkit/backdoor deployment combined with log access indicates persistent foothold intent (MITRE ATT&CK TA0003, TA0005).",
                "mitre_ref": ["TA0003", "TA0005"]}
    if len({"Personal Data", "Financial", "SQL / Database", "Data Exfiltration"} & set(categories_hit)) >= 2:
        return {"label": "Financial — Data Exfiltration",
                "conclusion": "Attacker targeted monetizable data for sale or extortion.",
                "explanation": "Access pattern focuses on sellable data consistent with data broker activity (MITRE ATT&CK TA0010).",
                "mitre_ref": ["TA0009", "TA0010"]}
    if total_files <= 3 or len(categories_hit) <= 2:
        return {"label": "Opportunistic",
                "conclusion": "No clear strategic goal — grabbed obvious targets without a plan.",
                "explanation": "Low file count and category diversity indicates selection by filename recognition (Diamond Model: opportunistic adversary).",
                "mitre_ref": []}
    return {"label": "General Reconnaissance",
            "conclusion": "Attacker systematically mapped the system without a single dominant goal.",
            "explanation": "Broad coverage suggests intelligence gathering for a future targeted attack (MITRE ATT&CK TA0007).",
            "mitre_ref": ["TA0007"]}


def _skill_indicators(files_taken, categories_hit, total_files):
    all_files = [f.lower() for fl in files_taken.values() for f in fl]
    div_score = round((len(categories_hit) / len(FILE_CATEGORIES)) * 100, 1)
    return {
        "covered_tracks":      {"present": ".bash_history" in all_files, "explanation": "Accessed bash history — covering tracks"},
        "packaged_exfil":      {"present": any(".tar.gz" in f or ".zip" in f for f in all_files), "explanation": "Created archive — systematic bulk exfiltration"},
        "targeted_ssh":        {"present": any(f in all_files for f in ["id_rsa", ".ssh"]), "explanation": "Targeted SSH keys — lateral movement preparation"},
        "planted_malware":     {"present": bool(files_taken.get("Malware / Dropped Tools")), "explanation": "Dropped malware — persistence planned"},
        "financial_motivation":{"present": "Financial" in categories_hit or any(f in all_files for f in ["miner.py", "xmrig"]), "explanation": "Financial data or mining tools — monetization motive"},
        "category_diversity":  {"score": div_score, "raw": f"{len(categories_hit)}/{len(FILE_CATEGORIES)}", "explanation": "Proportion of file categories accessed"},
        "total_files":         {"count": total_files, "explanation": "Total files accessed"},
    }


def _confidence(categories_hit, total_files):
    n, f = len(categories_hit), total_files
    if n >= CONFIDENCE_THRESHOLDS["HIGH"]["min_categories"] and f >= CONFIDENCE_THRESHOLDS["HIGH"]["min_files"]:
        return "HIGH"
    if n >= CONFIDENCE_THRESHOLDS["MEDIUM"]["min_categories"] and f >= CONFIDENCE_THRESHOLDS["MEDIUM"]["min_files"]:
        return "MEDIUM"
    return "LOW"


def _build_tarpit_profile(record: dict) -> TarpitProfile:
    src_ip      = str(record.get("src_ip", "unknown"))
    files_taken = record.get("files_taken", {})
    total_files = int(record.get("total_files_taken", 0))
    categories_hit = [c for c in FILE_CATEGORIES if files_taken.get(c) and len(files_taken[c]) > 0]
    intent_profile = _compute_intent(files_taken, categories_hit, total_files)
    primary_intent = TACTIC_LABELS.get(max(intent_profile, key=intent_profile.get), "Undetermined") if intent_profile else "Undetermined"
    return TarpitProfile(
        src_ip=src_ip, timestamp=str(record.get("timestamp", "")),
        src_machine=str(record.get("src_machine", "unknown")),
        files_taken=files_taken, total_files=total_files,
        categories_hit=categories_hit, intent_profile=intent_profile,
        primary_intent=primary_intent,
        motivation=_motivation(files_taken, categories_hit, total_files),
        skill_indicators=_skill_indicators(files_taken, categories_hit, total_files),
        confidence=_confidence(categories_hit, total_files),
    )


class TarpitReader:
    """[V] Loads all 4 attacker profile files and indexes by src_ip."""

    PROFILE_FILES = [
        "advance_cowrie_attacker_profile.json",
        "advance_honeypot_2_attacker_profile.json",
        "intermediate_cowrie_attacker_profile.json",
        "script_kiddie_cowrie_attacker_profile.json",
    ]

    def __init__(self, profiles_dir: str):
        self.profiles_dir = profiles_dir
        self._index: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        total = 0
        for fname in self.PROFILE_FILES:
            fpath = os.path.join(self.profiles_dir, fname)
            if not os.path.exists(fpath):
                print(f"[TarpitReader] Not found: {fname}")
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "root" in data:
                    data = data["root"]
                records = data if isinstance(data, list) else [data]
                for r in records:
                    if isinstance(r, dict):
                        ip = r.get("src_ip", "").strip()
                        if ip:
                            self._index[ip] = r
                            total += 1
            except Exception as e:
                print(f"[TarpitReader] Error reading {fname}: {e}")
        print(f"[TarpitReader] Loaded {total} profiles across {len(self._index)} unique IPs")

    def get_profile(self, src_ip: str) -> Optional[TarpitProfile]:
        record = self._index.get(src_ip)
        return _build_tarpit_profile(record) if record else None


def render_tarpit_section(profile: TarpitProfile) -> str:
    lines, h = [], "─" * 65
    mot = profile.motivation
    lines += [
        h, "  TARPIT FILE ACCESS ANALYSIS", h,
        f"  Attacker IP    : {profile.src_ip}",
        f"  Files accessed : {profile.total_files} across {len(profile.categories_hit)} categories",
        f"  Confidence     : {profile.confidence}", "",
        "  PRIMARY GOAL", h,
        f"  {mot['label']}", "",
        f"  Conclusion: {mot['conclusion']}", "",
        f"  Why: {mot['explanation']}", "",
    ]
    if mot.get("mitre_ref"):
        lines.append(f"  MITRE ATT&CK: {', '.join(mot['mitre_ref'])}")
    lines += ["", "  MITRE ATT&CK TACTIC BREAKDOWN", h]
    for tactic, pct in profile.intent_profile.items():
        if pct > 0:
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {tactic:<20} {bar}  {pct:.1f}%")
    lines += ["", "  FILES ACCESSED BY CATEGORY", h]
    for cat in FILE_CATEGORIES:
        fl = profile.files_taken.get(cat, [])
        if fl:
            lines.append(f"  {cat}")
            for fname in fl:
                insight = FILENAME_INSIGHTS.get(fname.lower(), (None, None))
                lines.append(f"    → {fname:<30} {insight[1] if insight[1] else ''}")
            lines.append("")
    lines += ["  SKILL INDICATORS", h]
    si = profile.skill_indicators
    for key in ["covered_tracks", "packaged_exfil", "targeted_ssh", "planted_malware", "financial_motivation"]:
        ind = si.get(key, {})
        mark = "✓" if ind.get("present") else "✗"
        lines.append(f"  {mark}  {ind.get('explanation', '')}")
    cd = si.get("category_diversity", {})
    tf = si.get("total_files", {})
    lines += [
        "", "  BEHAVIORAL SCORES", h,
        f"  Category diversity : {cd.get('raw','N/A')} ({cd.get('score',0)}%)",
        f"  Total files taken  : {tf.get('count',0)}",
        f"  Primary intent     : {profile.primary_intent}", "",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODULE [U]: TensorBuilder — Preprocessing for Model Inference
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

def preprocess_session(payloads: List[str], time_delta: float = 0.0,
                       has_custom_tool: int = 0):
    if not TORCH_AVAILABLE:
        return None, None
    history = list(payloads[-HISTORY_LEN:])
    while len(history) < HISTORY_LEN:
        history.insert(0, "")
    token_rows = []
    for cmd in history:
        b_arr = list(str(cmd).encode("ascii", errors="ignore"))[:SEQ_LEN]
        b_arr += [0] * (SEQ_LEN - len(b_arr))
        token_rows.append(b_arr)
    token_tensor  = torch.tensor([token_rows], dtype=torch.long)
    scalar_tensor = torch.tensor([[time_delta, float(has_custom_tool)]], dtype=torch.float32)
    return scalar_tensor, token_tensor


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODEL LOADER AND INFERENCE RUNNER
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelOutput:
    available:             bool
    threat_score:          float
    attack_class:          int
    attack_class_label:    str
    tool_flags:            List[bool]
    tool_names_detected:   List[str]
    routing_probs:         List[float]
    routing_entropy:       float
    dominant_expert:       int
    dominant_expert_label: str
    expert_interpretation: str
    confidence_label:      str


def _confidence_from_entropy(entropy: float) -> str:
    if entropy < 0.30:  return "HIGH"
    if entropy < 0.55:  return "MEDIUM"
    if entropy < 0.75:  return "LOW"
    return "UNCERTAIN"


def load_model(model_path: str, device: str = "cpu"):
    if not TORCH_AVAILABLE:
        return None
    if not os.path.exists(model_path):
        print(f"[Model] Not found: {model_path}")
        return None
    print(f"[Model] Loading from {model_path}...")
    model = BehavioralMacroDreadnought(
        seq_len=SEQ_LEN, history_len=HISTORY_LEN,
        num_tools=NUM_TOOLS, uniform_base=UNIFORM_BASE,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("[Model] Loaded successfully.")
    return model


def run_inference(model, payloads: List[str], time_delta: float = 0.0,
                  has_custom_tool: int = 0, device: str = "cpu") -> ModelOutput:
    if not TORCH_AVAILABLE or model is None:
        return ModelOutput(
            available=False, threat_score=0.0, attack_class=0,
            attack_class_label="Unknown", tool_flags=[False]*5,
            tool_names_detected=[], routing_probs=[0.33, 0.33, 0.33],
            routing_entropy=1.0, dominant_expert=0,
            dominant_expert_label="N/A", expert_interpretation="Model unavailable.",
            confidence_label="N/A",
        )
    scalar_t, token_t = preprocess_session(payloads, time_delta, has_custom_tool)
    scalar_t = scalar_t.to(device)
    token_t  = token_t.to(device)
    with torch.no_grad():
        (_, routing_probs, threat_score, class_logits, tool_logits,
         routing_entropy, top1_idx, _) = model(scalar_t, token_t)
    threat      = float(threat_score[0, 0].item())
    atk_class   = int(torch.argmax(class_logits, dim=-1)[0].item())
    tool_probs  = torch.sigmoid(tool_logits[0]).tolist()
    tool_flags  = [p >= 0.5 for p in tool_probs]
    r_probs     = routing_probs[0].tolist()
    entropy     = float(routing_entropy[0].item())
    dom_expert  = int(top1_idx[0].item())
    high_e      = entropy >= 0.55
    interp      = EXPERT_INTERPRETATIONS.get((dom_expert, high_e),
                  EXPERT_INTERPRETATIONS[(dom_expert, False)])
    return ModelOutput(
        available=True, threat_score=round(threat, 4),
        attack_class=atk_class, attack_class_label=CLASS_LABELS[atk_class],
        tool_flags=tool_flags,
        tool_names_detected=[TOOL_NAMES[i] for i, f in enumerate(tool_flags) if f],
        routing_probs=[round(p, 4) for p in r_probs],
        routing_entropy=round(entropy, 4), dominant_expert=dom_expert,
        dominant_expert_label=EXPERT_LABELS[dom_expert],
        expert_interpretation=interp,
        confidence_label=_confidence_from_entropy(entropy),
    )


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODULE [Y]: ReportSynthesizer — Threat Score Fusion + Report Renderer
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

THREAT_LABELS = {85: "CRITICAL", 65: "HIGH", 40: "MEDIUM", 15: "LOW", 0: "NEGLIGIBLE"}


def fuse_threat_score(model_output: ModelOutput, injection_summary: dict,
                      tarpit_profile: Optional[TarpitProfile]) -> Tuple[int, str]:
    max_sev   = injection_summary.get("max_severity", 0)
    total     = max(injection_summary["total_records"], 1)
    mal_ratio = injection_summary["malicious_count"] / total
    inj_score = (max_sev / 100.0) * 0.7 + mal_ratio * 0.3
    if tarpit_profile and tarpit_profile.total_files > 0:
        tarpit_score = (len(tarpit_profile.categories_hit) / 14.0) * 0.5 + \
                       min(tarpit_profile.total_files / 20.0, 1.0) * 0.5
    else:
        tarpit_score = 0.0
    if model_output.available:
        composite = model_output.threat_score * 0.40 + inj_score * 0.40 + tarpit_score * 0.20
    else:
        composite = inj_score * 0.60 + tarpit_score * 0.40
    score_100 = max(0, min(100, int(round(composite * 100))))
    label = "NEGLIGIBLE"
    for threshold, lbl in sorted(THREAT_LABELS.items(), reverse=True):
        if score_100 >= threshold:
            label = lbl
            break
    return score_100, label


def infer_tools_from_behavior(injection_results: List[InjectionResult]) -> List[str]:
    all_types    = [t for r in injection_results for t in r.attack_types_found]
    all_payloads = " ".join(r.raw_payload for r in injection_results)
    inferred = []
    if all_types.count("sqli") >= 2:
        inferred.append(f"sqlmap (likely) — {all_types.count('sqli')} SQLi patterns")
    login_hits = sum(1 for r in injection_results if "login" in r.raw_payload.lower() and r.is_malicious)
    if login_hits >= 3:
        inferred.append(f"hydra/medusa (likely) — {login_hits} login attempts")
    if all_types.count("rce") >= 1 and all_types.count("command_injection") >= 1:
        inferred.append("metasploit (possible) — RCE + CMDi combination")
    if all_types.count("lfi") >= 1 and all_types.count("xss") >= 1:
        inferred.append("nikto/web scanner (likely) — LFI+XSS fingerprint")
    has_known = any(t in all_payloads.lower() for t in ["sqlmap", "hydra", "metasploit", "nikto"])
    if not has_known and len(set(all_types)) >= 3:
        inferred.append("custom tool (likely) — high diversity, no standard signatures")
    return inferred or ["Manual attack — no automated tool signature detected"]


def render_report(session_id, attacker_ip, payloads, injection_results,
                  injection_summary, model_output, tarpit_profile,
                  threat_score, threat_label, inferred_tools) -> str:
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    H, h = "═" * 70, "─" * 70
    lines = []

    lines += [H, "  PROJECT LEO — ANALYTICAL INCIDENT REPORT", H,
              f"  Session ID   : {session_id}",
              f"  Attacker IP  : {attacker_ip}",
              f"  Report Time  : {now}",
              f"  Payloads     : {len(payloads)} records in session",
              H, ""]

    bar = "█" * (threat_score // 5) + "░" * (20 - threat_score // 5)
    lines += ["  SECTION 1 — THREAT SCORE", h,
              f"  {bar}  {threat_score}/100  [{threat_label}]", ""]

    lines += ["  SECTION 2 — BEHAVIORAL MODEL OUTPUT", h]
    if model_output.available:
        lines += [
            f"  Attack class     : {model_output.attack_class_label} (class {model_output.attack_class})",
            f"  Confidence       : {model_output.confidence_label} (entropy={model_output.routing_entropy:.4f})",
            f"  Dominant expert  : {model_output.dominant_expert_label}",
            f"  Expert routing   : E1 {model_output.routing_probs[0]*100:.1f}%  E2 {model_output.routing_probs[1]*100:.1f}%  E3 {model_output.routing_probs[2]*100:.1f}%",
            "", f"  Interpretation: {model_output.expert_interpretation}", "",
            "  Tools detected (model):",
        ]
        if model_output.tool_names_detected:
            for t in model_output.tool_names_detected:
                rel = "HIGH reliability" if t in ("sqlmap", "custom") else "LOW reliability — undertrained"
                lines.append(f"    ✓ {t:<15} ({rel})")
        else:
            lines.append("    No specific tools flagged by model")
    else:
        lines.append("  Model unavailable. Injection and tarpit signals used exclusively.")
    lines.append("")

    lines += ["  SECTION 3 — INJECTION ANALYSIS", h,
              f"  Total records    : {injection_summary['total_records']}",
              f"  Malicious        : {injection_summary['malicious_count']}",
              f"  Max severity     : {injection_summary['max_severity']}/100",
              f"  Attack types     : {', '.join(injection_summary['attack_types']) or 'none'}",
              f"  Session rhythm   : {injection_summary['session_rhythm']}",
              f"  Label agreement  : {injection_summary['label_agreements']}", "",
              "  Per-payload breakdown:"]
    for r in injection_results:
        if r.is_malicious:
            types = ", ".join(r.attack_types_found)
            lines.append(f"    [{r.severity_score:>3}/100] {r.raw_payload[:55]:<55}  {types}")
        else:
            lines.append(f"    [  CLEAN] {r.raw_payload[:55]}")
    lines.append("")

    lines += ["  SECTION 4 — TARPIT FILE ACCESS ANALYSIS", h]
    if tarpit_profile:
        for line in render_tarpit_section(tarpit_profile).split("\n"):
            lines.append(f"  {line}")
    else:
        lines.append("  No tarpit file access recorded for this session.")
    lines.append("")

    lines += ["  SECTION 5 — ATTACKER COMMAND TIMELINE", h]
    for i, (payload, inj) in enumerate(zip(payloads, injection_results), 1):
        flag = ""
        if inj.is_malicious:
            flag = f"  ← [{inj.severity_score}/100] {'+'.join(inj.attack_types_found)}"
        lines.append(f"  {i:>3}. {payload[:55]:<55}{flag}")
    lines.append("")

    lines += ["  SECTION 6 — BEHAVIORAL TOOL INFERENCE", h]
    for t in inferred_tools:
        lines.append(f"  •  {t}")
    lines.append("")

    lines += ["  SECTION 7 — ANALYST NOTES", h]
    if model_output.available:
        e, entropy = model_output.dominant_expert, model_output.routing_entropy
        if e == 2 and entropy < 0.30:
            lines += ["  ▲ EXPERT-3 HIGH CONFIDENCE:", "    SpLR_V2 gate CLOSED — clean advanced signal. Escalate immediately.", ""]
        elif e == 2 and entropy >= 0.55:
            lines += ["  ▲ EXPERT-3 HIGH ENTROPY:", "    SpLR_V2 gate OPEN — chaos flooding through. LOW confidence.", ""]
        elif e == 0 and entropy < 0.30:
            lines += ["  ◉ EXPERT-1 HIGH CONFIDENCE: Deterministic beginner pattern.", ""]
        elif e == 1 and entropy < 0.30:
            lines += ["  ◉ EXPERT-2 HIGH CONFIDENCE: Intermediate — partial customization.", ""]
    if model_output.available and tarpit_profile and tarpit_profile.total_files >= 10 and model_output.attack_class == 0:
        lines += ["  ⚠ SIGNAL CONTRADICTION: Model says Beginner but tarpit shows advanced file access.", ""]
    rhythm = injection_summary.get("session_rhythm", "unknown")
    if rhythm == "hybrid":
        lines += ["  ▲ HYBRID RHYTHM: Skilled human reading output between commands. Most dangerous profile.", ""]
    elif rhythm == "automated":
        lines += ["  ◉ AUTOMATED RHYTHM: Scripted attack, no human supervision.", ""]
    attack_types = injection_summary.get("attack_types", [])
    if len(attack_types) >= 3:
        lines += [f"  ▲ MULTI-VECTOR: {len(attack_types)} distinct attack types — {', '.join(attack_types)}", ""]
    if tarpit_profile:
        mot = tarpit_profile.motivation
        lines += [f"  ▲ TARPIT CONFIRMS: {mot['label']}", f"    {mot['conclusion']}", ""]
    critical = injection_summary.get("critical_payloads", [])
    if critical:
        lines += [f"  ▲ {len(critical)} CRITICAL PAYLOAD(S) — score >= 80/100", ""]
    lines.append("")

    lines += ["  SECTION 8 — RECOMMENDED ACTIONS", h]
    if threat_score >= 85:
        actions = ["1. IMMEDIATE  Block attacker IP at perimeter firewall.",
                   "2. IMMEDIATE  Preserve all session logs for forensics.",
                   "3. URGENT     Audit all systems attacker may have reached.",
                   "4. URGENT     Rotate all credentials in accessed categories."]
    elif threat_score >= 65:
        actions = ["1. HIGH       Rate-limit or block attacker IP.",
                   "2. HIGH       Review accessed systems for indicators.",
                   "3. MEDIUM     Flag session for security team review."]
    elif threat_score >= 40:
        actions = ["1. MEDIUM     Monitor attacker IP for continued activity.",
                   "2. LOW        Log session for threat intelligence."]
    else:
        actions = ["1. LOW        Continue monitoring. No immediate action required."]
    for a in actions:
        lines.append(f"  {a}")
    lines += ["", H, f"  END OF REPORT — PROJECT LEO — {session_id}", H]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
# MODULE [Y]: LeoReportEngine — MAIN ENTRY POINT
# Wires Phase 0, Goal 1, Goal 2, Model, Goal 3 together.
# ██████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════

class LeoReportEngine:
    """
    Single entry point for the full PROJECT LEO pipeline.

    Usage:
        engine = LeoReportEngine()
        report = engine.run_from_file("session.json")
        print(report)
    """

    def __init__(self, model_path: str = MODEL_PATH,
                 profiles_dir: str = PROFILES_DIR,
                 output_dir: str = OUTPUT_DIR,
                 device: str = DEVICE):
        self.device     = device
        self.output_dir = output_dir

        self.model = load_model(model_path, device)

        self.tarpit_reader = None
        if os.path.exists(profiles_dir):
            self.tarpit_reader = TarpitReader(profiles_dir)
        else:
            print(f"[LeoReportEngine] Profiles dir not found: {profiles_dir}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"[LeoReportEngine] Ready.")
        print(f"  Model:   {'loaded' if self.model else 'disabled'}")
        print(f"  Tarpit:  {'loaded' if self.tarpit_reader else 'disabled'}")
        print(f"  Output:  {output_dir}")

    def run(self, batch: SessionBatch, injection_results: List[InjectionResult]) -> str:
        """Run from already-processed batch and injection results."""
        inj_summary = session_summary(injection_results)
        model_input = batch.to_model_input()
        model_out   = run_inference(
            model=self.model, payloads=model_input["payloads"],
            time_delta=model_input["time_delta"],
            has_custom_tool=model_input["has_custom_tool"],
            device=self.device,
        )
        tarpit_profile = self.tarpit_reader.get_profile(batch.src_ip) if self.tarpit_reader else None
        threat_score, threat_label = fuse_threat_score(model_out, inj_summary, tarpit_profile)
        inferred_tools = infer_tools_from_behavior(injection_results)
        report = render_report(
            session_id=batch.session_id, attacker_ip=batch.src_ip,
            payloads=batch.get_payloads(), injection_results=injection_results,
            injection_summary=inj_summary, model_output=model_out,
            tarpit_profile=tarpit_profile, threat_score=threat_score,
            threat_label=threat_label, inferred_tools=inferred_tools,
        )
        if self.output_dir:
            safe_id  = batch.session_id.replace(".", "_")
            filepath = os.path.join(self.output_dir, f"leo_report_{safe_id}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"[LeoReportEngine] Saved: {filepath}")
        return report

    def run_from_records(self, raw_records: list) -> str:
        """Run from raw JSON records already in memory."""
        batch = assemble_session(raw_records)
        if not batch:
            return "[LeoReportEngine] Empty session — no report generated."
        injection_results = scan_session(batch)
        return self.run(batch, injection_results)

    def run_from_file(self, filepath: str) -> str:
        """
        ══════════════════════════════════════════════════════════════
        # ── INTEGRATION POINT  ── 4
        SESSION FILE INPUT
        Point this at a JSON or JSONL file from the honeypot.
        The file should contain all records for ONE closed session.
        Supported source machines: cowrie, cowrie-docker,
        honeypot_2, main_system (app + docker logs).

        Example:
            report = engine.run_from_file("sessions/attacker_session.json")
        ══════════════════════════════════════════════════════════════
        """
        batch = load_session_from_file(filepath)
        if not batch:
            return f"[LeoReportEngine] Could not read session: {filepath}"
        injection_results = scan_session(batch)
        return self.run(batch, injection_results)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("PROJECT LEO — Full Unified Pipeline")
    print("=" * 70)
    print()
    print("To run against a real session file:")
    print("    engine = LeoReportEngine()")
    print("    report = engine.run_from_file('your_session.json')")
    print("    print(report)")
    print()
    print("To run against raw records in memory:")
    print("    engine = LeoReportEngine()")
    print("    report = engine.run_from_records(list_of_json_dicts)")
    print()
    print(" Integration points — search # ── INTEGRATION POINT ──  to find them:")
    print("    # ── INTEGRATION POINT 1 ── 1  →  MODEL_PATH     (line ~45)")
    print("    # ── INTEGRATION POINT 1 ── 2  →  PROFILES_DIR   (line ~60)")
    print("    # ── INTEGRATION POINT 1 ── 3  →  OUTPUT_DIR     (line ~75)")
    print("    # ── INTEGRATION POINT 1 ── 4  →  run_from_file  (in method docstring)")
    print()

    # Quick smoke test with synthetic records
    DEMO = [
        {"timestamp": "2026-04-15T15:24:25+00:00", "src_machine": "cowrie",
         "src_ip": "10.10.0.1", "session": 1, "command": "",
         "time_delta": 0, "is_Advance_Attack": False, "attack_tool": "unknown"},
        {"timestamp": "2026-04-15T15:24:33+00:00", "src_machine": "cowrie",
         "src_ip": "10.10.0.1", "session": 1, "command": "whoami",
         "time_delta": 7.6, "is_Advance_Attack": False, "attack_tool": "unknown"},
        {"timestamp": "2026-04-15T15:24:37+00:00", "src_machine": "cowrie",
         "src_ip": "10.10.0.1", "session": 1, "command": "cat /etc/passwd",
         "time_delta": 4.0, "is_Advance_Attack": False, "attack_tool": "unknown"},
        {"timestamp": "2026-04-15T15:24:52+00:00", "src_machine": "cowrie",
         "src_ip": "10.10.0.1", "session": 1,
         "command": "python3 -c 'import socket,subprocess'",
         "time_delta": 15.0, "is_Advance_Attack": True, "attack_tool": "unknown"},
        {"timestamp": "2026-04-15T16:09:43+00:00", "src_machine": "main_system",
         "src_ip": "10.10.0.1", "session": 1, "command": "",
         "time_delta": 0.0, "event": "login_attempt", "attack_tool": "sqlmap",
         "attack_type": "sqli", "request_uri": "/login.php", "method": "POST",
         "username": "admin",
         "parameters": {"GET": [], "POST": {"username": "admin' UNION SELECT 1,username,password FROM users--"}}},
    ]

    engine = LeoReportEngine()
    report = engine.run_from_records(DEMO)
    print(report)