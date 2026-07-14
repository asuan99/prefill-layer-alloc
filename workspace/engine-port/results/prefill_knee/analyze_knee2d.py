#!/usr/bin/env python3
"""Parse knee2d_result_L*.txt -> per-(L,B,SM) table + Diff B(B,L) heatmap.

Diff B = attn SM-sensitivity / mamba SM-sensitivity, where
  sensitivity_type(SM) = per_layer_time_type(SM) / per_layer_time_type(108).
Diff B ~ 1  => no reallocation lever (both types equally SM-hungry).
Diff B >> 1 => cheap-type (mamba) surplus SM can be given to attn/decode (lever exists).

Usage:  python analyze_knee2d.py            # scans this dir
        python analyze_knee2d.py DIR        # scans DIR
Outputs: knee2d_table.csv, knee2d_diffB.png  (next to the result files)
"""
import glob, os, re, sys, csv, math

HERE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

LINE = re.compile(
    r"RESULT L=(\d+) B=(\d+) sm=(\w+).*?"
    r"ctxlen=(\d+) n=(\d+)(?: bs=(-?\d+) ntok=(-?\d+))? \|.*?"
    r"per-attn\(9\)=([\d.]+) per-mamba\(54\)=([\d.]+)"
)

def sm_int(s):
    return 108 if s == "full" else int(s)

# rows[(L,B)][SM] = dict(attn=, mamba=, ctxlen=, bs=, ntok=)
rows = {}
files = sorted(glob.glob(os.path.join(HERE, "knee2d_result_L*.txt")))
if not files:
    sys.exit(f"no knee2d_result_L*.txt in {HERE}")
for fn in files:
    for ln in open(fn):
        m = LINE.search(ln)
        if not m:
            continue
        L, B, sm, ctxlen, n, bs, ntok, pa, pm = m.groups()
        L, B, sm = int(L), int(B), sm_int(sm)
        rows.setdefault((L, B), {})[sm] = dict(
            attn=float(pa), mamba=float(pm), ctxlen=int(ctxlen), n=int(n),
            bs=int(bs) if bs is not None else -1,
            ntok=int(ntok) if ntok is not None else -1,
        )

Ls = sorted({L for (L, _) in rows})
Bs = sorted({B for (_, B) in rows})
SMs = sorted({sm for d in rows.values() for sm in d})
lowSM = min(SMs)  # sensitivity reference at the most-starved partition

# ---- CSV: full per-(L,B,SM) table + per-cell Diff B ----
csv_path = os.path.join(HERE, "knee2d_table.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["L", "B", "SM", "bs_obs", "ntok_obs", "ctxlen",
                "per_attn_ms", "per_mamba_ms", "cost_ratio_attn_over_mamba"])
    for (L, B) in sorted(rows):
        for sm in sorted(rows[(L, B)]):
            r = rows[(L, B)][sm]
            cr = r["attn"] / r["mamba"] if r["mamba"] else float("nan")
            w.writerow([L, B, sm, r["bs"], r["ntok"], r["ctxlen"],
                        f"{r['attn']:.4f}", f"{r['mamba']:.4f}", f"{cr:.3f}"])

# ---- Diff B(B,L) at lowSM ----
def diffB(L, B):
    d = rows.get((L, B), {})
    if 108 not in d or lowSM not in d:
        return None
    a_sens = d[lowSM]["attn"] / d[108]["attn"]
    m_sens = d[lowSM]["mamba"] / d[108]["mamba"]
    return a_sens / m_sens if m_sens else None

def mamba_frac(L, B):
    d = rows.get((L, B), {})
    if 108 not in d:
        return None
    r = d[108]
    ta, tm = r["attn"] * 9, r["mamba"] * 54
    return tm / (ta + tm) if (ta + tm) else None

print(f"parsed {len(rows)} (L,B) cells from {len(files)} files; SMs={SMs} lowSM={lowSM}")
print(f"\nDiff B = attn_sens/mamba_sens @ {lowSM}SM  (>1 => lever; ~1 => none)")
print("L\\B      " + "".join(f"{B:>10}" for B in Bs))
for L in Ls:
    cells = []
    for B in Bs:
        db = diffB(L, B)
        cells.append("     n/a" if db is None else f"{db:>10.2f}")
    print(f"{L:<8}" + "".join(cells))
print("\nmamba time-fraction @108SM (gain-ceiling weight):")
print("L\\B      " + "".join(f"{B:>10}" for B in Bs))
for L in Ls:
    cells = []
    for B in Bs:
        mf = mamba_frac(L, B)
        cells.append("     n/a" if mf is None else f"{mf:>10.2f}")
    print(f"{L:<8}" + "".join(cells))

# ---- heatmap ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    M = np.full((len(Bs), len(Ls)), np.nan)
    for i, B in enumerate(Bs):
        for j, L in enumerate(Ls):
            db = diffB(L, B)
            if db is not None:
                M[i, j] = db
    fig, ax = plt.subplots(figsize=(1.6 * len(Ls) + 2, 1.1 * len(Bs) + 2))
    vmax = max(2.0, np.nanmax(M) if np.isfinite(M).any() else 2.0)
    # diverging around 1.0 (=no lever): TwoSlopeNorm centered at 1
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=min(0.8, np.nanmin(M) if np.isfinite(M).any() else 0.8), vcenter=1.0, vmax=vmax)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"L={L}" for L in Ls])
    ax.set_yticks(range(len(Bs))); ax.set_yticklabels([f"B={B}" for B in Bs])
    ax.set_xlabel("context length L"); ax.set_ylabel("batch B (concurrency)")
    ax.set_title(f"prefill Diff B = attn_sens / mamba_sens @ {lowSM}SM\n"
                 "(red>1 = layer-aware lever exists; white=1 = none; ctx3600 single-pt was ~1.17)")
    for i in range(len(Bs)):
        for j in range(len(Ls)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Diff B")
    fig.tight_layout()
    png = os.path.join(HERE, "knee2d_diffB.png")
    fig.savefig(png, dpi=130)
    print(f"\nwrote {csv_path}\nwrote {png}")
except Exception as e:
    print(f"\nwrote {csv_path}\n(heatmap skipped: {e})")
