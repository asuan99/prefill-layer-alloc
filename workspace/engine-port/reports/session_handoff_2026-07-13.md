# Session handoff — 2026-07-13 (cudagraph 재측정 + fidelity-ladder + layer-aware 최종 종결)

이전 핸드오프 [session_handoff_2026-07-07.md](session_handoff_2026-07-07.md)(coordinated layer-aware 반증·SLO-aware track 시작)의 후속.
이 세션은 **(1) SLO-aware v7b 마무리·정리, (2) full-system/engine/sim fidelity-ladder 정립, (3) cudagraph 재측정으로 layer-aware를 최종 종결, (4) 전 결과 시각화**.

---

## 0. 한 줄 상태

**layer-aware(모든 형태: decode-sub-step·prefill-type·span·§14 예약)는 실엔진에서 최종 死.** cudagraph 재측정이
반증을 **강화**함(cudagraph ⊥ sub-step layer-aware). **실전 권고 = agnostic / regime-tuned uniform / SLO-aware(무튜닝 generalist).**
**운영점 = cudagraph-ON**(기존 전 서빙수치는 no-cudagraph 하한이었음). 트리 clean, 8 커밋(`a2440f7`..`e7c6a55`).

---

## 1. 이 세션 커밋 (시간순)

| commit | 내용 |
|---|---|
| `a2440f7` | SLO-aware v7b(layer-span eval) + B mixed-load payoff 데이터 |
| `a1ae349` | policy_comparison §2b 메커니즘·전환기준 테이블 |
| `66533cc` | **fidelity-ladder 보고서** (full-system vs engine vs queue-sim) |
| `cfbb569` | policy_comparison §1 스코프 박스 |
| `77119e0` | **cudagraph 재측정** (Probe 1-3): wall 제거·랭킹 유지·layer-aware 반증 강화 |
| `4161433` | cudagraph raw bench .out |
| `96ea8da` | **per-result-dir 시각화** 11개 + cudagraph figure |
| `e7c6a55` | **Probe 4** tuned decode-SM 스윕 → 예약 트랙 CLOSED + panel F |

---

## 2. 확정된 결론 (이 세션에서 새로/강화)

### 2.1 Bullet/MuxWise vs 내 정책 (taxonomy)
- 둘 다 **layer-agnostic(type-blind)** — 내 SLO-aware·tuned·fused도 그러함. layer-aware(死)만 type-aware.
- 진짜 분류축 = **신호**: 내 agnostic=decode-배치(open); **Bullet=큐 지연(closed) = 내 SLO-aware와 같은 계열**(Bullet=더 좋은 기판 위 SLO-aware); MuxWise=layer-span bubble-free 균형(계산형).
- ⚠️ **표 오류 수정됨**(§2b): agnostic "step 내 전환 無"는 intra-step만 맞고, agnostic은 **매 step decode_bs로 재계산=빈번 전환**. "거의 전환 無"는 tuned/SLO(수렴후) 것.

### 2.2 fidelity-ladder (핵심 프레임) — [system_vs_engine_vs_sim.md](system_vs_engine_vs_sim.md)
- 내 작업 = full serving system이 아니라 **sglang 위 SM-split 정책 레이어 1개**(스케줄러·KV·green-ctx 메커니즘 전부 상속).
- **bracket 논제**: queue-sim=upper bound(낙관, switch-cost/isolation/decode-wall을 0으로 이상화), 내 engine=lower bound(무거운 기판), Bullet-급 full-system=사이.
- 7 차이축: (A)전환비용 (B)cudagraph (C)single-proc/GIL (D)스케줄러 (E)KV (F)신호품질 (G)isolation.

### 2.3 ★cudagraph 재측정 (이 세션 최대 성과) — [results/cudagraph_probe/cudagraph_results.md](../results/cudagraph_probe/cudagraph_results.md)
- **"pdmux+mamba cudagraph 불가"는 오해=수동 `--disable-cuda-graph` flag.** py3.14/torch2.9+hybrid+triton+**pdmux green-ctx**서 캡처·replay 정상(크래시 0).
- **decode wall 실재·제거**: pdmux TPOT rate2 41→12ms. 절대 goodput ~1.5-2×↑. **기존 전 서빙수치=no-cudagraph 하한.**
- **정책 랭킹 유지**(cgON 재확인): prefill-bound=tuned-d24≈SLO≫agn; decode-heavy=SLO≈tuned-d44>agn. **SLO=무튜닝 generalist.**
- ★★**layer-aware 반증 강화**: **cudagraph ⊥ sub-step layer-aware**(sub-step green-ctx 재분할=고정그래프 캡처 불가→coord decode 영구 eager). core는 벽 넘고 layer-aware는 못 넘어 **격차 확대**. lacoord: prefill TTFT 우위 실재(697<<3025)나 decode SLO 붕괴(gp r4 **0.027**).
- ★★**§14 "예약" 트랙 CLOSED (Probe 4)**: tuned decode-SM 스윕 → cudagraph가 최적 split을 **d24→d16**으로 밀어(attn-decode 싸져 16SM 충분, d08=8SM은 TPOT 135ms floor) "예약"의 prefill 이득을 tuned가 흡수. 예약 lever는 순수 decode-side per-layer(독립 prefill lever 無; [prefill_knee] 비용비 1.2-1.4×). **step-fixed d16 gp 0.913 vs sub-step lacoord 0.027(r4)=34× 격차.** **layer-type-aware 마지막 형태도 死.**

### 2.4 SLO-aware (이전 세션 확정, 이 세션 cudagraph서 재확인)
stationary=static 매칭(무튜닝, isolation 오버헤드0)·mixed 1-stress=TIE·**dual-stress=+18% WIN**. cudagraph 하에서도 우위 유지(임계값은 no-cudagraph TPOT에 튜닝됨→재튜닝 여지). 상세 [[slo-aware-scheduling-track]].

---

## 3. 데이터·파일 맵

- **코드(러닝)**: dev tree `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/multiplex/multiplexing_mixin.py` + 미러 `workspace/engine-port/src/multiplex/multiplexing_mixin.py`. env: `PDMUX_SLO_SCHED`/`_EMA`/`_DWELL`, `PDMUX_LA_COORD`(+`_OPT`/`_V4`/`_PF`). 이 세션 코드변경 無(cudagraph는 순전히 launch flag).
- **cudagraph 실험**: `results/cudagraph_probe/` — probe1_plain/probe2_pdmux/probe3_cg/probe4_tuned_sweep.sbatch, cudagraph_results.md, pdmux_d{08,16,34,54}.yml, figures/cudagraph_remeasurement.png, make_plots.py.
- **시각화**: `results/<dir>/<dir>_viz.png` (11개) + 생성기 `results/plot_result_dirs.py`(CSV 자동파싱).
- **보고서**: `reports/` — system_vs_engine_vs_sim.md(fidelity-ladder), policy_comparison.md(§1 스코프·§2b 메커니즘), slo_aware_scheduling_design.md, prefill_vs_decode_execution.md.
- **환경**: venv `/scratch/ehmoon/whlee/sglang_engine_venv`(py3.14, torch2.9.1+cu130), module `conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0`, 파티션 `amd_a100nv_8`(A100-80GB). cudagraph 켜려면 launch에서 `--disable-cuda-graph --disable-piecewise-cuda-graph` **제거**.

---

## 4. 열린 항목 / 다음 유망

1. **per-window cudagraph** (layer-aware 최후 검증) — 창마다 그래프 캡처로 (D) drain만 격리. **권고 미실행**(sglang 커스텀 캡처 大공수·(a)substrate가 drain을 잔차 지배로 지목·payoff 낮음 예상). 실행 시에만 관에 마지막 못.
2. **SLO 컨트롤러 cudagraph 재튜닝** — 임계값(TPOT_SLO/HI/LO)이 no-cudagraph TPOT(~40ms)에 맞춰짐. cudagraph TPOT(~13-40ms)에 재튜닝하면 추가 이득 여지. mixed(B) 재측정도 cudagraph에서 반복 가치.
3. **cudagraph 하 full 정책 재캠페인** (선택) — 이 세션은 nprompt 100·rate 축약. 기존 slo_sched 방법론(nprompt 120·rate 1-6)으로 cgON 재측정하면 리포트 수치 갱신.
4. **보고서 coordinated 개정 미완**(이전 세션 열린항목 B, `sm_policy_report.html`) — 여전히 미완.

---

## 5. 메모리 포인터
[[fidelity-ladder-scope]](이 세션 핵심: 스코프·bracket·cudagraph 판정·예약 CLOSED), [[slo-aware-scheduling-track]], [[engine-port-p0-triage]], [[prefill-layer-alloc-status]](§14 예약 원출처).
