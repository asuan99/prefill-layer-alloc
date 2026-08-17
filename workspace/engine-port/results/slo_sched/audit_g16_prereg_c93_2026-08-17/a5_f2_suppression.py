#!/usr/bin/env python3
"""Audit (i): does F2 ("no new numbers on control failure") actually suppress
the campaign block, and does K11 resolve WITHOUT fallback on the 7-arm grid?

Estimands are firewalled to CONSTANTS, so no decision quantity is computed;
only the SHAPE of the emitted report is inspected.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched")
import g16_analyze as G
HERE = Path(__file__).resolve().parent.parent

CONST = {k: 1.0 for k in (
    "M_ttft", "M_itl", "requests", "duration_s", "throughput_req_s", "goodput_req_s",
    "ttft_pass_pct", "itl_p95_pass_pct", "joint_pass_pct", "band_mass_ttft",
    "band_mass_itl", "empty_itl_requests", "ttft_p50_ms", "ttft_p95_ms",
    "ttft_p99_ms", "token_itl_p50_ms", "token_itl_p95_ms", "token_itl_p99_ms")}
G._headline_estimands = lambda path: dict(CONST)

grid, _ = G.load_campaign_grid(HERE, "HI")
sat = G._saturation_gap(grid, contenders=["d44"])
print("K11 on the real campaign arm set:")
print("   upper_arms:", sat["upper_arms"], "basis:", sat["upper_arm_basis"],
      "(values firewalled; only the SET is checked)")

sys.argv = ["g16_analyze.py", "--campaign", "--controls", "--out",
            "/tmp/claude-100018302/-scratch-ehmoon-whlee/8c2a3fff-74d8-4c85-9fd8-df28017f1fe0/scratchpad/f2_probe.json"]
G.PC_A_TARGETS["HI"]["d44"]["goodput_req_s"] = 999.0      # force a control failure
try:
    G.main()
except SystemExit as exc:
    print("\nF2 test: main() exit code:", exc.code)
rep = json.loads(Path("/tmp/claude-100018302/-scratch-ehmoon-whlee/8c2a3fff-74d8-4c85-9fd8-df28017f1fe0/scratchpad/f2_probe.json").read_text())
camp = rep["campaign"]
print("   campaign block keys:", sorted(camp))
print("   -> numbers on disk under a failed control:",
      "NO (suppressed)" if camp.get("SUPPRESSED") else "YES (LEAK)")
print("   analyzer_sha256 recorded in report (F4):", rep["analyzer_sha256"][:16])
