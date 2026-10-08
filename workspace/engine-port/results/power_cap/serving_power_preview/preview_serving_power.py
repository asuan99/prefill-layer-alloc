#!/usr/bin/env python3
"""Zero-GPU preview: does board power separate by serving-engine state in EXISTING data?  (CPU only)

Data: N3 shape B 3rd run (Job pdmux-n3-capacity-38ed3f4-20261006092813, Nemotron-H, D44 sticky-OFF, PHASE_EVENTS=1 units)
  - power: meta/clk.csv `power.draw` (polled snapshot, ~100 ms update, ~60-175 ms lag -- sensor_bw probe)
  - engine state: decode_iteration events (t_launch, t_sync, bs, prefill_in_flight, decode_sms) cached by
    ../analysis_2026-10-07/pc_itl_throttle.py, mapped to wall clock with its per-unit offset (half-width <= 52 ms).
For each 100 ms power sample (shifted back by LAG to undo sensor lag) the window [t-LAG-0.1, t-LAG] is classified:
  idle        : no decode iteration in the window and none within 1 s
  decode_only : decode iterations present, none with prefill_in_flight
  co_resident : decode iterations present, >= 50 % with prefill_in_flight (prefill on the other partition)
  mixed       : 0 < co fraction < 50 %
  no_decode   : no decode iteration but engine active within 1 s (prefill-only or gaps)
Reports mean power per state, by realized decode split and batch-size bin.  DESCRIPTIVE, n = one Job.
"""
import bisect, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis_2026-10-07"))
import pc_common as C  # noqa: E402

LAGS = [0.0, 0.06, 0.12, 0.18]
cfg = C.VESSL["n3_r3"]
clk = C.parse_clk(cfg["clk"]); ct = [r[0] for r in clk]
itl = json.load(open(HERE.parent / "analysis_2026-10-07" / "out" / "pc_itl_throttle.json"))
cache = HERE.parent / "analysis_2026-10-07" / "cache"
timing = C.read_timing(cfg["out"] / "TIMING.tsv", "unit")


def run(LAG):
    rows = []
    for u, rec in itl["per_unit"].items():
        O = rec["offset"]["O_hat"]
        dec = [json.loads(l) for l in open(cache / f"dec_{u}.jsonl")]
        ev = sorted((float(e["t_launch"]) + O, float(e["t_sync"]) + O, int(e.get("bs") or 0), bool(e.get("prefill_in_flight")),
                     int(e.get("decode_sms") or 0)) for e in dec)
        et = [e[0] for e in ev]
        w = timing[u]
        i, j = bisect.bisect_left(ct, w["a"]), bisect.bisect_right(ct, w["b"])
        for t, sm, pw, tmp, bits in clk[i:j]:
            b1 = t - LAG; a1 = b1 - 0.1
            k0, k1 = bisect.bisect_left(et, a1), bisect.bisect_right(et, b1)
            inwin = ev[k0:k1]
            near = bisect.bisect_right(et, b1 + 1.0) - bisect.bisect_left(et, a1 - 1.0)
            if not inwin:
                state = "no_decode" if near else "idle"; bs = 0; dsm = None
            else:
                co = sum(e[3] for e in inwin) / len(inwin)
                state = "decode_only" if co == 0 else ("co_resident" if co >= 0.5 else "mixed")
                bs = round(st.fmean(e[2] for e in inwin)); dsm = max(set(e[4] for e in inwin), key=[e[4] for e in inwin].count)
            rows.append({"unit": u, "state": state, "power_W": pw, "sm_mhz": sm, "cap": bool(bits & 0x4), "bs": bs, "decode_sms": dsm})
    return rows


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["state"]].append(r)
    out = {}
    for s, v in g.items():
        p = sorted(x["power_W"] for x in v)
        out[s] = {"n": len(v), "mean_W": st.fmean(p), "p10_W": p[len(p) // 10], "p90_W": p[int(.9 * len(p))],
                  "cap_frac": sum(x["cap"] for x in v) / len(v), "sm_mhz_mean": st.fmean(x["sm_mhz"] for x in v),
                  "decode_sms_mode": max(set(x["decode_sms"] for x in v), key=[x["decode_sms"] for x in v].count)}
    return out


res = {}
for L in LAGS:
    rows = run(L); res[f"lag_{int(L*1000)}ms"] = summarize(rows)
best = max(res, key=lambda k: res[k].get("co_resident", {}).get("mean_W", 0) - res[k].get("decode_only", {}).get("mean_W", 0))
rows = run(int(best.split("_")[1][:-2]) / 1000)
by = defaultdict(list)
for r in rows:
    if r["state"] in ("decode_only", "co_resident"):
        by[(r["state"], r["decode_sms"], min(r["bs"], 8))].append(r["power_W"])
res["chosen_lag"] = best
res["by_state_split_bs"] = {f"{s}|dsm={d}|bs={b}": {"n": len(v), "mean_W": st.fmean(v)} for (s, d, b), v in sorted(by.items(), key=str) if len(v) >= 20}
json.dump(res, open(HERE / "preview_serving_power.json", "w"), indent=1)
for L in LAGS:
    k = f"lag_{int(L*1000)}ms"
    print(k, {s: (v["n"], round(v["mean_W"]), round(v["cap_frac"], 2), v["decode_sms_mode"]) for s, v in sorted(res[k].items())})
print("chosen", best); [print(" ", k, v["n"], round(v["mean_W"])) for k, v in res["by_state_split_bs"].items()]

# figure: power distribution per state (chosen lag) + time-series excerpt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YEL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(11, 7.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
ax = fig.add_subplot(gs[0, 0])
order = [("idle", "#9a9893"), ("decode_only", ORANGE), ("mixed", YEL), ("co_resident", BLUE), ("no_decode", "#6b6a66")]
data = [[r["power_W"] for r in rows if r["state"] == s] for s, _ in order]
bp = ax.boxplot([d if d else [0] for d in data], patch_artist=True, widths=0.55, showfliers=False)
for patch, (_, c) in zip(bp["boxes"], order): patch.set_facecolor(c); patch.set_alpha(0.85)
for med in bp["medians"]: med.set_color(INK)
ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([f"{s}\n(n={len(d)})" for (s, _), d in zip(order, data)], fontsize=8.5)
ax.set_ylabel("board power (W), 100 ms sample"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.set_title(f"power by engine state (sensor lag undone: {best.split('_')[1]})", loc="left", fontsize=10, color=INK)
ax2 = fig.add_subplot(gs[0, 1])
ks = [k for k in res["by_state_split_bs"] if "|dsm=108|" in k or "|dsm=44|" in k]
for s, c in (("decode_only", ORANGE), ("co_resident", BLUE)):
    pts = sorted((int(k.split("bs=")[1]), res["by_state_split_bs"][k]["mean_W"]) for k in ks if k.startswith(s))
    if pts: ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=c, lw=2, label=s)
ax2.set_xlabel("decode batch size (mean in window, capped at 8)"); ax2.set_ylabel("mean board power (W)"); ax2.grid(axis="y", color=GRID)
ax2.legend(frameon=False, fontsize=9); ax2.set_title("by decode batch size", loc="left", fontsize=10, color=INK)
ax3 = fig.add_subplot(gs[1, :])
u = "g_r045_2084" if "g_r045_2084" in itl["per_unit"] else next(iter(itl["per_unit"]))
ur = [r for r in rows if r["unit"] == u]
w = timing[u]; i, j = bisect.bisect_left(ct, w["a"]), bisect.bisect_right(ct, w["b"])
first = next((k for k, r in enumerate(ur) if r["state"] != "idle"), 0)
first = max(0, first - 10)
seg = clk[i:j][first:first + 250]; ur = ur[first:first + 250]
colors = {"idle": "#9a9893", "decode_only": ORANGE, "mixed": YEL, "co_resident": BLUE, "no_decode": "#6b6a66"}
t0 = seg[0][0]
ax3.plot([x[0] - t0 for x in seg], [x[2] for x in seg], color="#b9b8b3", lw=1, zorder=1)
for r, x in zip(ur[:len(seg)], seg):
    ax3.scatter(x[0] - t0, x[2], color=colors[r["state"]], s=12, zorder=2)
for s, c in colors.items(): ax3.scatter([], [], color=c, s=20, label=s)
ax3.legend(frameon=False, ncol=5, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax3.grid(axis="y", color=GRID)
ax3.set_xlabel(f"time (s) — unit {u} (offered 0.45 req/s), 25 s from the first non-idle sample"); ax3.set_ylabel("W")
ax3.set_title("100 ms board power colored by engine state\n", loc="left", fontsize=10, color=INK)
fig.suptitle("Serving-engine preview from existing data (N3 shape B, Nemotron-H, D44 sticky-OFF; polled 100 ms power)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); (HERE / "figures").mkdir(exist_ok=True); fig.savefig(HERE / "figures" / "preview_power_by_state.png", dpi=150)
print("wrote", HERE / "figures" / "preview_power_by_state.png")
