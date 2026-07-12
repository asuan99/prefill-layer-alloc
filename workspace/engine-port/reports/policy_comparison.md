# PD-mux 정책 종합 비교 — SM-allocation vs Scheduler/Controller

작성: 2026-07-09. 이 프로젝트에서 실엔진(sglang v0.5.10, Zamba2-2.7B, A100-80GB, clean async serving)으로
측정한 **모든 PD-mux 정책**을 한자리에 비교한다. 두 패러다임:
- **(A) SM-allocation 정책** — prefill↔decode SM split을 고정/준동적으로 정함 (fused·agnostic·tuned-uniform·agnostic_v2·layer-aware).
- **(B) Scheduler/Controller 정책** — SLO 피드백으로 split을 **동적** 조정 (SLO-aware).

정본 상세: layer-aware=[sm_policy_report.html](sm_policy_report.html)·[prefill_vs_decode_execution.md](prefill_vs_decode_execution.md); SLO-aware=[slo_aware_scheduling_design.md](slo_aware_scheduling_design.md). 원자료 `results/`.

라벨: goodput@SLO(req/s) = `#{req: TTFT≤3s ∧ mean-ITL≤60ms} / duration`. 두 regime: **in3600/o32=prefill-bound**, **in2000/o96=decode-heavy**.

---

## 1. 전체 정책 · 한 줄 판정

| # | 정책 | 패러다임 | split 결정 | 판정 |
|---|---|---|---|---|
| 1 | **fused** | (A) baseline | pdmux 미사용(decode가 prefill에 결합) | 부하시 붕괴(worst baseline) |
| 2 | **agnostic** | (A) 준동적 | decode 배치크기로 split | 견고, 4모델 전부서 fused 이김 |
| 3 | **tuned-uniform** (d24/d44) | (A) 정적 | regime-최적 고정 split | ★**정적 최적** — agnostic도 이김 |
| 4 | **agnostic_v2** | (A) 정적 | 전층 낮은 floor(decode16) | decode 실작업 있으면 최악 |
| 5 | **layer-aware (decode-type)** | (A) sub-step | decode 층타입별 split (R0d/(a)/v4) | ★**반증** — (D) granularity |
| 6 | **layer-aware (prefill-type=PF)** | (A) sub-step | prefill 층타입별 split (PF/fix1/fix2) | ★**최악** — starvation+(D) 양측 |
| 7 | **SLO-aware (controller)** | (B) 동적 | TPOT/큐 피드백으로 split | ★**steady-state=static(isolation 증명)**·regime 자동수렴; 유일 upside 정책; mixed(B) 미검 |

## 2. goodput@SLO 비교표

### in3600/o32 (prefill-bound) — TPOT p50 병기
| rate | fused | agnostic | **d24(tuned)** | agn_v2/d16 | LA:(a)OPT | LA:v4 | PF:fix2 | SLO-v6 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.79 | 1.19 | **1.20**(40) | 1.09 | 0.96 | 0.85 | 0.73 | 1.19 |
| 2 | 0.27 | 2.06 | **2.24**(43) | 1.87 | 0.31 | 0.15 | 0.00 | 1.32† |
| 3 | 0.00 | 0.55 | **1.18** | 1.03 | 0.00 | 0.00 | 0.00 | 0.34† |
| 4 | 0.00 | 0.31 | **0.47** | 0.41 | 0.00 | 0.00 | 0.00 | 0.48 |

†SLO-v6는 d24로 **수렴**하며 **steady-state=static**(isolation 증명); prefill-bound서 startup transient만큼 낮음(back-to-back rate 벤치 아티팩트, 연속부하선 amortize).

### in2000/o96 (decode-heavy)
| rate | agnostic | **d44(tuned)** | LA:(a)OPT | LA:v4 | PF:fix2 | SLO-v6 |
|---|---|---|---|---|---|---|
| 2 | 2.25 | 2.24 | 1.82 | 1.37 | 1.64 | **2.25** |
| 3 | 3.24 | 3.22 | 0.43 | 0.35 | 0.00 | **3.19** |
| 4 | 1.59 | **2.27** | 0.06 | 0.00 | 0.00 | 1.22 |

★**SLO-v6 = decode-heavy(in2000)서 static 매칭**(r2 2.25≈2.24, r3 3.19≈3.22). LA:R0d=coord baseline은 두 regime 모두 rate2+서 goodput 0.
isolation: SLO 코드경로를 d24에 pin → static과 **정확히 동일**(2.318≡2.319) ⇒ **steady-state 오버헤드 0, 메커니즘 검증**.

## 3. 정책별 상세

**1. fused** — pdmux 없이 decode를 prefill forward에 결합. decode TPOT가 prefill에 묶여 부하시 폭발(in3600 r2 88ms→r3 303ms). 실HW서 "PD 분리가 왜 필요한가"의 대조군.

**2. agnostic (pdmux)** — decode 배치크기로 split을 준동적 조정. **모든 하이브리드(4모델, spatial 포함)서 fused 이김**. decode TPOT ~42ms 평탄(병렬, non-gated). 견고한 실전 기본.

**3. tuned-uniform (d24/d44)** — regime-최적 고정 split. **정적 중 최적** — agnostic(auto)도 이김(in3600 d24 r3 +114%, in2000 d44 r4 +43%). ⚠️ **최적 split이 regime마다 다름**(in3600=decode24, in2000=decode44) → 정적은 하나만 선택 가능 = SLO-aware(동적)의 동기.

**4. agnostic_v2** — 전층을 낮은 floor(decode16)로. prefill 환원 최대·민감층 무보호. **decode 실작업 있으면 최악**(NemotronH·Granite 서빙 확증; decode 굶어 TPOT 폭발). decode-light(in3600 out32)선 d16과 유사(1.87).

**5. layer-aware (decode-type)** — decode를 층타입 윈도우로 쪼개 attn 보호/mamba 환원. **4개 실현 전부 반증**: inefficient_v1 698ms · R0d 124ms · (a)OPT 85ms · v4 95ms, 전부 step-level에 패. 근본=**(D) granularity**(sub-step SM 재배분이 짧은 창서 불가; 조율 비용 > 이득). [상세 §07](sm_policy_report.html).

**6. layer-aware (prefill-type = PF)** — split을 prefill 층타입 기준으로. **최악**: mamba-prefill을 34 SM로 환원해 prefill starvation(mamba-prefill은 SM-민감=knee 확증) + decode를 slice해 (D) 양측 부담. fix1(decode∝SM)=2차·무효, fix2(chunk↓)=저부하만 회복(150→52ms)이나 부하 붕괴. **prefill엔 "공짜로 뺄 SM" 없음**([prefill knee](prefill_vs_decode_execution.md)).

**7. SLO-aware (controller, (B))** — `PDMUX_SLO_SCHED`: 측정 TPOT-EMA(스파이크 outlier 제거)·prefill 큐 피드백으로 step-level split을 동적 조정(deadband+TPOT-gated hysteresis). ★**핵심 실증 2가지**: (i) **regime 자동 적응**(단일 정책·무튜닝으로 in3600→d24, in2000→d34–44 수렴); (ii) **steady-state 오버헤드 0 — isolation으로 증명**(SLO 코드경로를 d24에 pin한 결과 static d24와 **정확히 동일** 2.318≡2.319). ⇒ 메커니즘은 옳고 공짜다. **in2000(decode-heavy)선 static 매칭**(r2 2.25≈2.24, r3 3.19≈3.22). in3600(prefill-bound)선 아직 **startup transient**로 static 미달(r2 1.32 vs 2.24) — 컨트롤러 적응 중 prefill 일시 starve→back-to-back rate 벤치서 backlog 잔존(연속부하선 amortize). 컨트롤러 진화: v1 invisible-oscillation→v4 outlier-rejection→v6 neutral-start. **layer-aware와 직교**(층타입 아님, SLO 신호로 step split만=(D)-safe). **전 investigation서 유일하게 upside 있는 정책**(layer-aware 全패 vs 이건 static 매칭+동적 잠재). 진짜 이득 지점 = **mixed/bursty load**(step B; 정적이 평균에 튜닝돼 suboptimal한 곳) — 미검증.

## 4. 핵심 결론

1. **PD 분리(pdmux)는 항상 이득** — agnostic이 fused를 모든 모델서 이김.
2. **정적 최적 = tuned-uniform**(regime별 고정 split). agnostic도 이김.
3. **layer-aware(sub-step, decode·prefill 양측)는 전부 반증** — per-layer-type SM 재배분은 (D) granularity로 step-level을 못 이김. prefill-type(PF)은 starvation까지 겹쳐 최악.
4. **SLO-aware(동적 controller)는 유일하게 남은 유망 방향** — regime 적응은 확증됐고(정적이 못 하는 것), mixed load서 정적을 이길 잠재력. 단 control-law 튜닝 필요.

## 5. 왜 그런가 (통합 mechanism)

- **static 최적성**: 최적 split = prefill(compute-hungry)·decode(memory-satiated)의 **workload-SM 균형점**. tuned-uniform이 이를 잡음.
- **layer-aware 실패**: 그 균형을 **층타입으로 흔들면** (i) sub-step 재배분이 커널 timescale서 불가((D)), (ii) prefill-type은 mamba-prefill을 굶김(compute-bound). = 균형을 깨는 방향.
- **SLO-aware 잠재**: 균형점 자체가 **부하/regime 따라 이동**하는데 정적은 고정 → **동적으로 이동 추적**하면 정적을 이길 수 있음. 특히 mixed load.

## 부록 — 원자료
fused/agnostic/tuned: `results/r0a`,`results/r0c`. layer-aware decode: `results/r0d_coord_la`,`results/a_substrate`,`results/v4_multiwin`. layer-aware prefill(PF): `results/pf_boundary{,_fix1,_fix2}`. prefill knee: `results/prefill_knee`. SLO-aware: `results/slo_sched`.
