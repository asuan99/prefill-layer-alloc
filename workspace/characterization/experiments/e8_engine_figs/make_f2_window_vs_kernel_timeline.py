"""F2. window_vs_kernel_timeline
(D) granularity 메커니즘: decode의 per-layer 창(mamba 0.36ms)이 동시 실행되는 prefill
per-layer 커널(9-11ms)보다 훨씬 짧아, mamba 창 경계에서 SM을 재배분하려면 prefill 커널보다
20~30배 빠르게 스위칭해야 함 → 불가. 창이 커널 안에 파묻힘.
값 전부 measured:
  decode  per-layer @full SM: attn 3.14ms · mamba 0.36ms   (r0c/knee_result_835571.txt)
  prefill per-layer @full SM: attn 10.97ms · mamba 9.03ms  (prefill_knee/knee_result_837931.txt)
  레이어 수: attn 9 · mamba 54 (Zamba2-2.7B, knee txt에 명시)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _common import EP, ATT, SSM, read_knee, save

dec = read_knee(f"{EP}/r0c/knee_result_835571.txt")       # decode per-layer
pf  = read_knee(f"{EP}/prefill_knee/knee_result_837931.txt")  # prefill per-layer
i0 = dec["sm"].index(108)  # full-SM 지점
d_attn, d_mamba = dec["attn"][i0], dec["mamba"][i0]        # 3.14, 0.36
p_attn, p_mamba = pf["attn"][i0], pf["mamba"][i0]          # 10.97, 9.03
n_attn, n_mamba = dec["n_attn"], dec["n_mamba"]            # 9, 54

fig, (axT, axB) = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 1], hspace=0.32))

# ---- 상단: decode step 을 layer-type 창으로 (실제 길이 비율). 첫 ~30ms 구간 확대 ----
WIN = 30.0  # ms 표시 구간
def draw_decode(ax):
    t = 0.0; seq = [("mamba", d_mamba)] * n_mamba + [("attn", d_attn)] * n_attn
    for lt, w in seq:
        if t >= WIN: break
        ww = min(w, WIN - t)
        ax.barh(0, ww, left=t, height=0.6, color=(SSM if lt == "mamba" else ATT),
                edgecolor="white", linewidth=0.4, zorder=3)
        t += w
    ax.annotate(f"one mamba-decode window = {d_mamba:.2f} ms\n(the SM we'd want to reclaim)",
                xy=(d_mamba, 0.3), xytext=(4.5, 0.95), fontsize=9, color=SSM, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SSM, lw=1.4))
    ax.set_yticks([0]); ax.set_yticklabels(["decode step\n(54 mamba + 9 attn)"], fontsize=9)
    ax.set_title("F2 · a decode layer-window is buried inside the concurrent prefill kernel  → (D) granularity",
                 fontsize=11.5, fontweight="bold", loc="left")
draw_decode(axT)

# ---- 하단: 동시 실행되는 prefill 커널 한 층(9.03ms mamba) 을 같은 시간축에 ----
def draw_prefill(ax):
    # 한 prefill mamba-layer 커널(9.03ms)이 WIN 구간 안에서 차지하는 폭
    n_full = int(WIN // p_mamba)
    t = 0.0
    for k in range(n_full + 1):
        ww = min(p_mamba, WIN - t)
        if ww <= 0: break
        ax.barh(0, ww, left=t, height=0.6, color=SSM, alpha=0.35,
                edgecolor=SSM, linewidth=1.3, zorder=3)
        t += p_mamba
    ax.annotate(f"one prefill mamba-layer kernel = {p_mamba:.1f} ms\n"
                f"= {p_mamba/d_mamba:.0f}× the mamba-decode window",
                xy=(p_mamba, 0.3), xytext=(p_mamba + 1.5, 0.95), fontsize=9, color="#0b3d91",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#0b3d91", lw=1.4))
    ax.set_yticks([0]); ax.set_yticklabels(["concurrent\nprefill chunk"], fontsize=9)
    ax.set_xlabel("time (ms) — first 30 ms of a step, drawn to scale", fontsize=10)
draw_prefill(axB)

for ax in (axT, axB):
    ax.set_xlim(0, WIN); ax.set_ylim(-0.5, 1.35); ax.grid(axis="x", alpha=0.3, zorder=0)

leg = [Patch(facecolor=SSM, label=f"mamba layer (decode {d_mamba:.2f} / prefill {p_mamba:.1f} ms)"),
       Patch(facecolor=ATT, label=f"attn layer (decode {d_attn:.2f} ms)")]
axT.legend(handles=leg, fontsize=8.5, loc="upper right", ncol=1, framealpha=0.9)
fig.tight_layout()
save(fig, "f2_window_vs_kernel_timeline")
