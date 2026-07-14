"""F7. window_vs_kernel_timeline (개념도 — 비율만 정확)
decode mamba layer-window(~0.36-0.97ms)가 동시 실행 prefill 커널(~2.2ms/층)보다 짧아
SM 재배분이 실효 없음. 수치는 prefill_vs_decode_execution.md §4-6 라벨로 사용(conceptual)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _deck import ATT, SSM, save

MAMBA_WIN, ATTN_WIN, PF_KERNEL, WIN = 0.36, 3.14, 2.2, 24.0   # ms (개념 라벨)
N_MAMBA, N_ATTN = 54, 9

fig, (axT, axB) = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True,
                               gridspec_kw=dict(hspace=0.35))
# 상단: decode step 을 layer 창으로
t = 0.0
for lt, w in [("mamba", MAMBA_WIN)] * N_MAMBA + [("attn", ATTN_WIN)] * N_ATTN:
    if t >= WIN: break
    axT.barh(0, min(w, WIN - t), left=t, height=0.6,
             color=(SSM if lt == "mamba" else ATT), edgecolor="white", linewidth=0.4, zorder=3)
    t += w
axT.annotate(f"one mamba-decode window ≈ {MAMBA_WIN:.2f} ms\n(the SM we'd want to release)",
             xy=(MAMBA_WIN, 0.32), xytext=(3.5, 0.95), fontsize=9, color=SSM, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=SSM, lw=1.4))
axT.set_yticks([0]); axT.set_yticklabels(["decode step\n(54 mamba + 9 attn)"], fontsize=9)
axT.set_title("F7 · a decode layer-window is shorter than the concurrent prefill kernel  (conceptual, to scale)",
              fontsize=11.5, fontweight="bold", loc="left")
# 하단: 동시 prefill 커널 (2.2ms/층)
t = 0.0
while t < WIN:
    axB.barh(0, min(PF_KERNEL, WIN - t), left=t, height=0.6, color=SSM, alpha=0.35,
             edgecolor=SSM, linewidth=1.3, zorder=3)
    t += PF_KERNEL
axB.annotate(f"one prefill layer-kernel ≈ {PF_KERNEL:.1f} ms\n= {PF_KERNEL/MAMBA_WIN:.0f}× the mamba window "
             f"→ can't rebalance SM mid-kernel",
             xy=(PF_KERNEL, 0.32), xytext=(PF_KERNEL + 1.2, 0.95), fontsize=9, color="#0b3d91",
             fontweight="bold", arrowprops=dict(arrowstyle="->", color="#0b3d91", lw=1.4))
axB.set_yticks([0]); axB.set_yticklabels(["concurrent\nprefill chunk"], fontsize=9)
axB.set_xlabel("time (ms) — one step, drawn to scale", fontsize=10)
for ax in (axT, axB):
    ax.set_xlim(0, WIN); ax.set_ylim(-0.5, 1.35); ax.grid(axis="x", alpha=0.3, zorder=0)
axT.legend(handles=[Patch(facecolor=SSM, label=f"mamba (decode {MAMBA_WIN:.2f} / prefill {PF_KERNEL:.1f} ms)"),
                    Patch(facecolor=ATT, label=f"attn (decode {ATTN_WIN:.2f} ms)")],
           fontsize=8.5, loc="upper right", framealpha=0.9)
fig.tight_layout()
save(fig, "f7_window_vs_kernel_timeline")
