#!/usr/bin/env python3
"""AUDIT of CORESIDENCY_L_AXIS C-1: "he2 phase A is 3600/32, not 2048/32".
Enumerates the ACTUAL (input_len, output_len, rate) of every he2 phase-A/B
bench JSONL in the repo, and checks for telemetry."""
import json, glob, collections, os

c = collections.Counter()
for tag in ("he2A_", "he2B_"):
    for f in sorted(glob.glob(tag + "*.jsonl")):
        try:
            e = json.loads(open(f).readline())
        except Exception:
            c[(tag, "UNREADABLE")] += 1; continue
        c[(tag, e.get("random_input_len"), e.get("random_output_len"),
           e.get("dataset_name"), e.get("request_rate"))] += 1
for k, v in sorted(c.items(), key=lambda x: (str(x[0]))):
    L, O = k[1], k[2]
    ratio = (L / O) if isinstance(L, int) and isinstance(O, int) and O else None
    print(f"{k}  n_files={v}" + (f"   L/out={ratio:.1f}" if ratio else ""))
print("\ntelemetry files matching he2*:", glob.glob("he2*telemetry*") or "NONE")
print("PDMUX_TELEMETRY_PATH in he2_bench.sbatch:",
      "PRESENT" if "PDMUX_TELEMETRY_PATH" in open("he2_bench.sbatch").read() else "ABSENT")
