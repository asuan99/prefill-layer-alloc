#!/usr/bin/env python
"""cudagraph re-measurement 시각화. 측정 수치는 Probe 1/2/3(+4) 실측(하드코딩, 원자료 cg_probe*.out)."""
import os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

# ---- colors ----
C = {"agn":"#888888","d24":"#1f77b4","d44":"#1f77b4","slo":"#d62728",
     "lacoord":"#9467bd","cgOFF":"#bbbbbb","cgON":"#2ca02c"}

# ================= Probe 1 & 2: decode wall (in2000/o96) =================
# median TPOT (ms) at rate2 / rate4
wall = {
 "plain": {"cgOFF":{"tpot":[62.70,82.41],"p99":[142.2,108.05],"itl":[39.09,40.50]},
           "cgON": {"tpot":[13.51,54.04],"p99":[39.1,91.71],  "itl":[10.26,26.15]}},
 "pdmux": {"cgOFF":{"tpot":[40.97,39.03],"p99":[44.41,43.17],"itl":[37.40,37.87]},
           "cgON": {"tpot":[12.00,18.86],"p99":[26.92,20.03],"itl":[11.82,18.77]}},
}

# ================= Probe 3: goodput@SLO =================
pf = {"rates":[2,3,4],  # prefill-bound in3600/o32
      "agn":[0.989,0.268,0.207], "d24":[2.276,1.839,0.857],
      "slo":[2.130,1.148,0.727], "lacoord":[1.313,0.162,0.027]}
dh = {"rates":[2,3,4,6],  # decode-heavy in2000/o96
      "agn":[2.256,2.878,1.721,0.682], "d44":[2.263,3.311,3.200,1.383],
      "slo":[2.266,3.323,3.626,1.019]}
# transfer failure detail @ prefill-bound (rate2): TTFT(ms), decode ITL(ms), goodput
trans = {"pol":["agn","d24","slo","lacoord"],
         "ttft":[3025,615,774,697], "itl":[17.88,20.71,18.10,20.95],
         "itl_r3":[20.35,31.79,29.23,77.28], "gp":[0.989,2.276,2.130,1.313]}

# ================= Probe 4 (optional): tuned decode-SM sweep =================
def load_probe4():
    outs = sorted(glob.glob(os.path.join(OUT,"cg_probe4_*.out")))
    if not outs: return None
    txt = open(outs[-1]).read()
    # parse "SPLIT=dNN" blocks and their goodput rows from the trailing summary
    import re
    splits=[];
    for m in re.finditer(r"SPLIT=(d\d+)", txt):
        if m.group(1) not in splits: splits.append(m.group(1))
    # goodput rows are printed in order (split x rate) after "goodput@SLO rows"
    tail = txt.split("goodput@SLO rows")[-1] if "goodput@SLO rows" in txt else ""
    gps = [float(x) for x in re.findall(r"gp=([\d.]+)", tail)]
    if not splits or not gps: return None
    rates = [2,3,4]  # prefill-bound
    per = {}; k=0
    for s in splits:
        per[s] = gps[k:k+len(rates)]; k+=len(rates)
    return {"splits":splits,"rates":rates,"gp":per}
p4 = load_probe4()

# ================= FIGURE =================
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":0.3,"axes.axisbelow":True})
fig = plt.figure(figsize=(15, 9))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30)
SLO=60.0

# --- Panel A: decode wall (median TPOT, rate2) ---
axA = fig.add_subplot(gs[0,0])
groups=["plain","pdmux"]; x=range(len(groups)); w=0.36
off=[wall[g]["cgOFF"]["tpot"][0] for g in groups]
on =[wall[g]["cgON"]["tpot"][0]  for g in groups]
axA.bar([i-w/2 for i in x], off, w, label="cudagraph OFF", color=C["cgOFF"])
axA.bar([i+w/2 for i in x], on,  w, label="cudagraph ON",  color=C["cgON"])
for i,v in enumerate(off): axA.text(i-w/2, v+2, f"{v:.0f}", ha="center", fontsize=8)
for i,v in enumerate(on):  axA.text(i+w/2, v+2, f"{v:.0f}", ha="center", fontsize=8)
axA.axhline(SLO, color="k", ls="--", lw=1); axA.text(1.3, SLO+2, "SLO 60ms", fontsize=7)
axA.set_xticks(list(x)); axA.set_xticklabels(groups)
axA.set_ylabel("Median decode TPOT (ms)"); axA.set_title("(A) cudagraph removes decode wall\n(in2000/o96, rate2)")
axA.legend(fontsize=7)

# --- Panel B: pdmux TPOT & ITL vs rate ---
axB = fig.add_subplot(gs[0,1])
rr=[2,4]
axB.plot(rr, wall["pdmux"]["cgOFF"]["tpot"], "o--", color=C["cgOFF"], label="TPOT cgOFF")
axB.plot(rr, wall["pdmux"]["cgON"]["tpot"],  "o-",  color=C["cgON"],  label="TPOT cgON")
axB.plot(rr, wall["pdmux"]["cgOFF"]["itl"],  "s:",  color="#e0a0a0", label="ITL cgOFF", alpha=0.8)
axB.plot(rr, wall["pdmux"]["cgON"]["itl"],   "s-",  color="#7fc97f", label="ITL cgON", alpha=0.8)
axB.axhline(SLO, color="k", ls="--", lw=1)
axB.set_xlabel("request rate"); axB.set_ylabel("ms"); axB.set_xticks(rr)
axB.set_title("(B) pdmux decode latency vs rate\n(cudagraph ON vs OFF)"); axB.legend(fontsize=7)

# --- Panel C: prefill-bound goodput ---
axC = fig.add_subplot(gs[0,2])
for p,ls,mk in [("d24","-","o"),("slo","-","s"),("agn","--","^"),("lacoord","-","D")]:
    axC.plot(pf["rates"], pf[p], ls, marker=mk, color=C[p],
             label={"d24":"tuned-d24","slo":"SLO-v7b","agn":"agnostic","lacoord":"layer-aware*"}[p],
             lw=2 if p in("d24","slo") else 1.5)
axC.set_xlabel("request rate"); axC.set_ylabel("goodput@SLO (req/s)"); axC.set_xticks(pf["rates"])
axC.set_title("(C) PREFILL-bound in3600/o32\n(cudagraph ON)"); axC.legend(fontsize=7)
axC.annotate("layer-aware\ncollapses", xy=(4,0.027), xytext=(3.2,0.9),
             fontsize=7, color=C["lacoord"], arrowprops=dict(arrowstyle="->",color=C["lacoord"]))

# --- Panel D: decode-heavy goodput ---
axD = fig.add_subplot(gs[1,0])
for p,ls,mk in [("slo","-","s"),("d44","-","o"),("agn","--","^")]:
    axD.plot(dh["rates"], dh[p], ls, marker=mk, color=C[p],
             label={"d44":"tuned-d44","slo":"SLO-v7b","agn":"agnostic"}[p],
             lw=2 if p in("d44","slo") else 1.5)
axD.set_xlabel("request rate"); axD.set_ylabel("goodput@SLO (req/s)"); axD.set_xticks(dh["rates"])
axD.set_title("(D) DECODE-heavy in2000/o96\n(cudagraph ON)"); axD.legend(fontsize=7)

# --- Panel E: layer-aware transfer failure (rate2/3, prefill-bound) ---
axE = fig.add_subplot(gs[1,1])
pol=trans["pol"]; xi=range(len(pol)); w=0.38
axE.bar([i-w/2 for i in xi], trans["ttft"], w, color="#4c72b0", label="TTFT (ms) @r2")
axE.set_ylabel("TTFT (ms)  — lower better", color="#4c72b0")
axE.axhline(3000, color="#4c72b0", ls=":", lw=1); axE.text(0,3080,"TTFT SLO 3s",fontsize=6,color="#4c72b0")
axE.set_xticks(list(xi)); axE.set_xticklabels(["agn","d24","slo","LA*"])
axE2=axE.twinx()
axE2.bar([i+w/2 for i in xi], trans["itl_r3"], w, color="#dd8452", label="decode ITL (ms) @r3")
axE2.axhline(SLO, color="#dd8452", ls=":", lw=1); axE2.text(2.4,SLO+3,"ITL SLO 60ms",fontsize=6,color="#dd8452")
axE2.set_ylabel("decode ITL (ms) — lower better", color="#dd8452")
axE.set_title("(E) layer-aware transfer FAILS\nprefill TTFT edge (LA* 697≈d24) but decode ITL blows SLO")

# --- Panel F: Probe 4 tuned sweep (or note) ---
axF = fig.add_subplot(gs[1,2])
if p4:
    smmap={"d08":8,"d16":16,"d24":24,"d34":34,"d44":44,"d54":54}
    sms=[smmap[s] for s in p4["splits"]]
    for ri,r in enumerate(p4["rates"]):
        ys=[p4["gp"][s][ri] if ri<len(p4["gp"][s]) else float("nan") for s in p4["splits"]]
        axF.plot(sms, ys, "o-", label=f"rate {r}")
    axF.set_xlabel("decode SM (fixed split)"); axF.set_ylabel("goodput@SLO (req/s)")
    axF.set_title("(F) tuned decode-SM sweep\n(cudagraph ON, prefill-bound)")
    axF.legend(fontsize=7)
else:
    axF.text(0.5,0.5,"(F) tuned decode-SM sweep\n(Probe 4 running — fills in)",
             ha="center",va="center",fontsize=9,color="#999")
    axF.set_xticks([]); axF.set_yticks([])

fig.suptitle("Zamba2-2.7B PD-mux: cudagraph re-measurement (A100-80GB, sglang v0.5.10, clean async serving)",
             fontsize=12, y=0.99)
p=os.path.join(FIG,"cudagraph_remeasurement.png")
fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)
