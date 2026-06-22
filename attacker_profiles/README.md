# Attacker Profiles

This directory contains the tarpit file-access profiles produced by the honeypot logging system. PROJECT LEO's tarpit profiler (Module [V]) loads these files at startup and indexes them by `src_ip` for real-time session lookup.

---

## Required Files

Place all four files in this directory:

```
attacker_profiles/
├── advance_cowrie_attacker_profile.json
├── advance_honeypot_2_attacker_profile.json
├── intermediate_cowrie_attacker_profile.json
└── script_kiddie_cowrie_attacker_profile.json
```

---

## JSON Format

Each file is a JSON array. Every record must follow this structure:

```json
[
    {
        "src_ip": "10.10.0.1",
        "timestamp": "2026-04-04T16:23:41.098278+03:00",
        "src_machine": "cowrie",
        "files_taken": {
            "credentials": ["passwords.txt"],
            "Malware / Dropped Tools": [],
            "Data Exfiltration": [],
            "Network Reconnaissance": [],
            "System / OS Files": [],
            "Infrastructure": [],
            "SQL / Database": [],
            "Logs": [],
            "Financial": [],
            "Operational Notes": ["todo.txt"],
            "Scripts / Tools": [],
            "Personal Data": [],
            "Tokens / API Keys": [],
            "Media / Documents": []
        },
        "total_files_taken": 2
    }
]
```

---

## Field Definitions

| Field | Type | Description |
|---|---|---|
| `src_ip` | string | Attacker source IP — used as the lookup key |
| `timestamp` | string | ISO 8601 timestamp of the session |
| `src_machine` | string | Honeypot source: `cowrie`, `cowrie-docker`, or `honeypot_2` |
| `files_taken` | object | Files accessed per category — all 14 category keys must be present |
| `total_files_taken` | int | Total count of files accessed across all categories |

---

## The 14 File Categories

All 14 keys must be present in `files_taken` even if empty:

1. `credentials`
2. `Malware / Dropped Tools`
3. `Data Exfiltration`
4. `Network Reconnaissance`
5. `System / OS Files`
6. `Infrastructure`
7. `SQL / Database`
8. `Logs`
9. `Financial`
10. `Operational Notes`
11. `Scripts / Tools`
12. `Personal Data`
13. `Tokens / API Keys`
14. `Media / Documents`

---

## Notes

- Records are indexed by `src_ip` at startup. If two records share the same IP, the last one loaded wins.
- If this directory is missing or empty, the tarpit stream is disabled and the composite threat score formula degrades to `injection_score × 0.60 + tarpit_score × 0.40`.
- Real attacker profile data is not included in this repository.
