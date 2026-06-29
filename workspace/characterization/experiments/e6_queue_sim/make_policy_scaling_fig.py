"""Each policy's execution time (full-model decode ITL) vs decode batch and vs sequence length.
Policies: fused / co_schedule / agnostic_protect / layer_aware_protect.
(A) vs batch — fully measured (E5 backends, FullModelLM).
(B) vs sequence length — attn-decode part scaled by the measured E3 O(L) factor (ssm O(1)).
Output: reports/figures/policy_scaling.png
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
for _p in (_CHAR, os.path.abspath(os.path.join(_here, "..", "..", ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from experiments.e6_queue_sim.run_layer_aware import FullModelLM, _POLICY

M, NA, NS = "zamba2_2.7b", 9, 45
COL = {"fused": "#c5221f", "co_schedule": "#9aa0a6", "agnostic_protect": "#e8710a", "layer_aware_protect": "#1a73e8"}
LAB = {"fused": "fused (vLLM)", "co_schedule": "co_schedule", "agnostic_protect": "agnostic", "layer_aware_protect": "layer_aware"}
POLS = ["fused", "co_schedule", "agnostic_protect", "layer_aware_protect"]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

lm = FullModelLM(f"{_CHAR}/results_v2/e5_sim_b8_opt/serving_coexec_full_{M}_a100_sxm4_80gb.csv", NA, NS)


def parts(pol, b):
    """(attn_decode_per_layer, ssm_decode_per_layer, attn_solo_p, ssm_solo_p, attn_solo_d, ssm_solo_d)"""
    be = _POLICY[pol]
    pa, da, spa, sda = lm._get(be["attn"], "attn", b)
    ps, ds, sps, sds = lm._get(be["ssm"], "ssm", b)
    return da, ds, spa, sps, sda, sds


def decode_itl(pol, b, r_attn=1.0):
    da, ds, spa, sps, sda, sds = parts(pol, b)
    if pol == "fused":                      # sp + sd, attn-decode scales with context
        return (NA * spa + NS * sps) + (NA * sda * r_attn + NS * sds)
    return NA * da * r_attn + NS * ds       # decoupled: decode_stream sum

# E3 O(L) factor for attn-decode (median ratio to ctx4096)
e3 = pd.read_csv(f"{_CHAR}/results_v2/e3/decode_sweep_{M}_a100_sxm4_80gb.csv", skiprows=1)
e3.columns = [c.split("__")[0] for c in e3.columns]; e3 = e3[(e3.status == "ok") & (e3.layer_type == "attn")]
SM = sorted(e3.sm_count.unique())[-1]
CTX = [1024, 4096, 16384]
rA = {}
for c in CTX:
    num = e3[(e3.sm_count == SM) & (e3.context_len == c)].set_index("batch").latency_ms
    den = e3[(e3.sm_count == SM) & (e3.context_len == 4096)].set_index("batch").latency_ms
    rA[c] = float((num / den).median())

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8))

# (A) decode ITL vs batch
for pol in POLS:
    y = [decode_itl(pol, b) for b in BATCHES]
    axA.plot(BATCHES, y, "-o", color=COL[pol], lw=2.2, ms=5, label=LAB[pol])
axA.set_xscale("log", base=2); axA.set_yscale("log")
axA.set_xlabel("decode batch"); axA.set_ylabel("full-model decode ITL (ms)")
axA.set_title("(A) policy execution time vs batch (measured)", fontsize=11, fontweight="bold")
axA.legend(fontsize=9); axA.grid(alpha=0.3, which="both")

# (B) decode ITL vs sequence length (batch=64 fixed), attn part scaled by E3 O(L)
b = 64
for pol in POLS:
    y = [decode_itl(pol, b, rA[c]) for c in CTX]
    axB.plot(CTX, y, "-o", color=COL[pol], lw=2.2, ms=6, label=LAB[pol])
axB.set_xscale("log", base=2); axB.set_yscale("log")
axB.set_xlabel("sequence length (KV context)"); axB.set_ylabel(f"full-model decode ITL (ms), batch={b}")
axB.set_title("(B) policy execution time vs seq-len\n(attn-decode O(L) from E3; ssm O(1))", fontsize=11, fontweight="bold")
axB.legend(fontsize=9); axB.grid(alpha=0.3, which="both")
axB.text(0.5, 0.05, "context↑ → attn-decode↑ → fused/co가 더 빨리 악화\n(예약 정책은 attn만 늘어 완만)",
         transform=axB.transAxes, ha="center", fontsize=8, color="#555",
         bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0c060"))

fig.suptitle("Per-policy decode ITL scaling with batch & sequence length (zamba2_2.7b, A100)",
             fontsize=12.5, fontweight="bold", y=1.02)
out = Path(_CHAR).parents[1] / "reports" / "figures"
p = out / "policy_scaling.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print(f"wrote {p}  (r_attn: " + ", ".join(f"{c}:{rA[c]:.2f}" for c in CTX) + ")")
