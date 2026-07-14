"""F3. decode_vs_prefill_differential_dual  (F4를 좌패널로 흡수)
같은 이론(attn/mamba SM-민감도 differential)이 decode에선 크게(8.7-35×), prefill에선
거의 안(1.2-1.4×) 벌어짐 → 양쪽 다 layer-aware 실패를 정반대 이유로 예측하는 대칭성.
좌: decode per-layer ms vs SM (r0c/knee_result_835571.txt)
우: prefill per-layer ms vs SM (prefill_knee/knee_result_837931.txt)
같은 log y축·같은 x축(SM 108->8). 모든 값 measured."""
import matplotlib.pyplot as plt
from _common import EP, ATT, SSM, read_knee, save

dec = read_knee(f"{EP}/r0c/knee_result_835571.txt")
pf  = read_knee(f"{EP}/prefill_knee/knee_result_837931.txt")

fig, (axD, axP) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

def panel(ax, k, title, sub):
    ax.plot(k["sm"], k["attn"],  "o-", color=ATT, lw=2.4, ms=7, label=f"attn  (×{k['n_attn']})")
    ax.plot(k["sm"], k["mamba"], "s-", color=SSM, lw=2.4, ms=7, label=f"mamba (×{k['n_mamba']})")
    # differential(vertical gap) 주석: 최저 SM(8) 지점 비율
    r_hi = k["attn"][-1] / k["mamba"][-1]
    r_lo = k["attn"][0] / k["mamba"][0]
    ax.annotate(f"attn/mamba\n{r_lo:.1f}×→{r_hi:.0f}×" if r_hi > 3 else f"attn/mamba\n{r_lo:.1f}×→{r_hi:.1f}×",
                xy=(k["sm"][-1], (k["attn"][-1] * k["mamba"][-1]) ** 0.5),
                xytext=(0.5, 0.5), textcoords="axes fraction",
                fontsize=11, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(k["sm"]); ax.set_xticklabels([str(s) for s in k["sm"]])
    ax.invert_xaxis()
    ax.set_xlabel("SMs given to the kernel  (108 → 8)", fontsize=10)
    ax.set_title(f"{title}\n{sub}", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9.5, loc="upper left")

panel(axD, dec, "(left) DECODE per-layer sensitivity",
      "differential is HUGE → lever exists, but window too short → (D) kills it")
panel(axP, pf, "(right) PREFILL per-layer sensitivity",
      "differential is TINY (both compute-bound) → no lever to pull")
axD.set_ylabel("per-layer latency (ms, log)", fontsize=10)

fig.suptitle("F3 · same differential×window theory, opposite failure modes: "
             "decode lever exists but is un-actionable, prefill lever barely exists",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "f3_decode_vs_prefill_differential_dual")
