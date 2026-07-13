# PD-mux 정책 종합 비교 — SM-allocation vs Scheduler/Controller

작성: 2026-07-09. 이 프로젝트에서 실엔진(sglang v0.5.10, Zamba2-2.7B, A100-80GB, clean async serving)으로
측정한 **모든 PD-mux 정책**을 한자리에 비교한다. 두 패러다임:
- **(A) SM-allocation 정책** — prefill↔decode SM split을 고정/준동적으로 정함 (fused·agnostic·tuned-uniform·agnostic_v2·layer-aware).
- **(B) Scheduler/Controller 정책** — SLO 피드백으로 split을 **동적** 조정 (SLO-aware).

정본 상세: layer-aware=[sm_policy_report.html](sm_policy_report.html)·[prefill_vs_decode_execution.md](prefill_vs_decode_execution.md); SLO-aware=[slo_aware_scheduling_design.md](slo_aware_scheduling_design.md). 원자료 `results/`.

라벨: goodput@SLO(req/s) = `#{req: TTFT≤3s ∧ mean-ITL≤60ms} / duration`. 두 regime: **in3600/o32=prefill-bound**, **in2000/o96=decode-heavy**.

> **⚠️ 스코프 — 이건 full serving system 비교가 아니다.** 여기의 모든 정책은 **sglang v0.5.10 위에 얹은
> "SM-split 결정 로직" 한 레이어**다(스케줄러·KV·PD-mux green-ctx 메커니즘·모델러너는 전부 sglang 상속).
> Bullet/MuxWise는 프로세스 아키텍처+메커니즘+정책+cudagraph를 **공동설계한 full-system**이라 층위가 다르다.
> 그래서 여기 결과는 **특정 기판(green-ctx drain·no-cudagraph·single-process) 위의 정책 비교**이고, 각 결론이
> **substrate-artifact인지 fundamental인지**는 [system_vs_engine_vs_sim.md](system_vs_engine_vs_sim.md)에서 분리한다.
> 특히 **"layer-aware 반증"(#5·#6)은 이 기판 한정 진술** — prefill-side TTFT 기전은 확증·fundamental이고,
> 죽은 건 sub-step decode-type 전환((D))과 no-cudagraph decode wall이 가린 goodput 전환이다.

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
| 7 | **SLO-aware (controller)** | (B) 동적 | TPOT/큐 피드백, **layer-span 평가** | ★**steady-state=static(isolation)**·양 regime 자동수렴(v7b); 유일 upside; type-aware span=無이득; mixed(B) 검증중 |

## 2. goodput@SLO 비교표

### in3600/o32 (prefill-bound) — TPOT p50 병기
| rate | fused | agnostic | **d24(tuned)** | agn_v2/d16 | LA:(a)OPT | LA:v4 | PF:fix2 | **SLO-v7b** |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.79 | 1.19 | **1.20**(40) | 1.09 | 0.96 | 0.85 | 0.73 | 1.19 |
| 2 | 0.27 | 2.06 | **2.24**(43) | 1.87 | 0.31 | 0.15 | 0.00 | **2.11** |
| 3 | 0.00 | 0.55 | **1.18** | 1.03 | 0.00 | 0.00 | 0.00 | 0.70 |
| 4 | 0.00 | 0.31 | **0.47** | 0.41 | 0.00 | 0.00 | 0.00 | 0.47 |

★SLO-v7b(damped layer-span eval)는 d24로 자동수렴·**static 매칭**(r2 2.11≈2.24; isolation: pinned=static 정확 동일). v6(prefill-boundary eval)의 transient(r2 1.32)를 layer-span eval로 해소.

### in2000/o96 (decode-heavy)
| rate | agnostic | **d44(tuned)** | LA:(a)OPT | LA:v4 | PF:fix2 | **SLO-v7b** |
|---|---|---|---|---|---|---|
| 2 | 2.25 | 2.24 | 1.82 | 1.37 | 1.64 | **2.25** |
| 3 | 3.24 | 3.22 | 0.43 | 0.35 | 0.00 | **3.23** |
| 4 | 1.59 | **2.27** | 0.06 | 0.00 | 0.00 | **2.21** |

★**SLO-v7b = 양 regime 모두 static 매칭**(in2000: r3 3.23≈3.22, r4 2.21≈2.27, **d44 자동수렴**; in3600: r2 2.11≈2.24, d24 자동수렴). **단일 정책·무튜닝으로 per-regime 최적 재현.** LA:R0d은 두 regime 모두 rate2+서 goodput 0.
isolation: SLO 코드경로 pin → static과 **정확 동일**(2.318≡2.319)=**steady-state 오버헤드 0**. mechanism: **layer-span 평가+idx-change시만 drain**(agnostic 수렴후 switch 드묾→green-ctx서도 (D) 회피). Bullet(libsmctrl layer-span·cudagraph)의 green-ctx 근사. ⚠️**type-aware span sizing**(prefill span을 attn/ssm 경계서 단축)=**net-negative**(TTFT 2-4×↑, span 3× 짧아져 오버헤드)—layer-type은 span 경계로도 이득 無.

## 2b. 작용 방식 · 전환 기준 (알고리즘 축)

goodput(§2)과 별개로, 각 정책이 **split을 어떻게 결정하고 언제 바꾸나**를 정리한다.

| 정책 | 작용 방식 (알고리즘) | **전환 기준** (무엇이 split을 바꾸나) | granularity | step 내 전환 |
|---|---|---|---|---|
| fused | mixed batch 1-forward, SM 분할 없음 | — (분할 없음) | 없음 | — |
| agnostic | disjoint 파티션 동시실행; `idx=decode_bs·(n-2)//divisor` | **decode 배치크기** (prefill 경계서) | step | 사실상 無 |
| tuned-uniform d24/d44 | 고정 split (manual_divisions) | **없음** — 사전 수동 튜닝 | 고정 | 없음 |
| agnostic_v2 | 전층 decode floor 16 고정 | 없음 | 고정 | 없음 |
| LA decode-type (R0d/(a)/v4) | decode를 층타입 윈도우(~19)로 쪼개 attn보호/mamba환원 | **decode 층타입** (매 윈도우, 강제) | sub-step | **YES ~19/step** |
| LA prefill-type (PF/fix) | prefill을 층타입 윈도우로, decode slice | **prefill 층타입** | sub-step | **YES** |
| type-aware span | prefill span을 type 경계서 단축 | span경계=type경계(split은 SLO가) | span | (SLO) |
| **SLO-aware v7b** | disjoint 동시실행; split을 **측정 latency**로; layer-span 평가·**idx 바뀔 때만 drain** | **측정 TPOT vs SLO + prefill 큐** (closed-loop) | step-level, layer-span 평가 | **NO**(수렴 후 드묾) |

**전환 신호(signal) 축으로 재분류:** 없음(고정)=fused·tuned·agn_v2 · 배치크기(부하 proxy)=agnostic · **층타입(구조 proxy)**=layer-aware 全 · **측정 latency(SLO 직접·closed-loop)**=SLO-aware.

### ★ SLO가 다른 메커니즘과 근본적으로 다른 3가지
1. **유일한 closed-loop** — 나머지는 전부 open-loop(사전 고정, 또는 SLO와 무관한 proxy: agnostic=배치크기, layer-aware=층타입). **실제 SLO를 측정하는 건 SLO-aware뿐**(TPOT-EMA를 재서 SLO 경계에 맞춤) → workload 자동적응(regime 사전지식 불요)·목표(goodput) 직접 최적화·자기교정.
2. **전환 트리거가 구조가 아니라 성능** — layer-aware: 트리거=층타입 경계(구조적)→매 스텝 **강제** ~19회 전환→green-ctx drain→**(D)로 죽음**. SLO-aware: 트리거=측정 latency 임계 교차(성능적)→SLO 위험이 실제 바뀔 때만→수렴 후 **희소**→**(D) 회피**. 같은 green-ctx인데 "언제 바꾸나"가 구조(강제·빈번) vs 성능(필요시·희소)이라 갈림.
3. **split의 *값*만 바꿈, *구조*는 안 건드림** — layer-aware는 스텝 *내부*를 층타입으로 쪼갬(→(D)). SLO-aware는 decode를 통째로 두고(완전 오버랩) **split 값만** latency로 조정 = (D) 함정 원리적 회피. layer-span *평가*(반응성)와 *switching*(비용)을 분리한 게 핵심.

> **한 줄**: layer-aware="구조(층타입)에 반응해 스텝을 쪼갬"→(D) 死 / SLO-aware="성능(측정 latency)에 반응해 스텝-split 값만"→생존·static 격파. 본질=**open-loop proxy vs closed-loop 실측 + 구조-트리거(강제) vs 성능-트리거(희소).**

## 3. 정책별 상세

**1. fused** — pdmux 없이 decode를 prefill forward에 결합. decode TPOT가 prefill에 묶여 부하시 폭발(in3600 r2 88ms→r3 303ms). 실HW서 "PD 분리가 왜 필요한가"의 대조군.

**2. agnostic (pdmux)** — decode 배치크기로 split을 준동적 조정. **모든 하이브리드(4모델, spatial 포함)서 fused 이김**. decode TPOT ~42ms 평탄(병렬, non-gated). 견고한 실전 기본.

**3. tuned-uniform (d24/d44)** — regime-최적 고정 split. **정적 중 최적** — agnostic(auto)도 이김(in3600 d24 r3 +114%, in2000 d44 r4 +43%). ⚠️ **최적 split이 regime마다 다름**(in3600=decode24, in2000=decode44) → 정적은 하나만 선택 가능 = SLO-aware(동적)의 동기.

**4. agnostic_v2** — 전층을 낮은 floor(decode16)로. prefill 환원 최대·민감층 무보호. **decode 실작업 있으면 최악**(NemotronH·Granite 서빙 확증; decode 굶어 TPOT 폭발). decode-light(in3600 out32)선 d16과 유사(1.87).

**5. layer-aware (decode-type)** — decode를 층타입 윈도우로 쪼개 attn 보호/mamba 환원. **4개 실현 전부 반증**: inefficient_v1 698ms · R0d 124ms · (a)OPT 85ms · v4 95ms, 전부 step-level에 패. 근본=**(D) granularity**(sub-step SM 재배분이 짧은 창서 불가; 조율 비용 > 이득). [상세 §07](sm_policy_report.html).

**6. layer-aware (prefill-type = PF)** — split을 prefill 층타입 기준으로. **최악**: mamba-prefill을 34 SM로 환원해 prefill starvation(mamba-prefill은 SM-민감=knee 확증) + decode를 slice해 (D) 양측 부담. fix1(decode∝SM)=2차·무효, fix2(chunk↓)=저부하만 회복(150→52ms)이나 부하 붕괴. **prefill엔 "공짜로 뺄 SM" 없음**([prefill knee](prefill_vs_decode_execution.md)).

**7. SLO-aware (controller, (B))** — `PDMUX_SLO_SCHED`: 측정 TPOT-EMA(outlier 제거)·prefill 큐 피드백으로 split을 **layer-span마다 평가**하고 **idx 바뀔 때만 drain+switch**(agnostic 수렴후 switch 드묾→green-ctx서도 (D) 회피). deadband+TPOT-gated hysteresis+damping(EMA 0.85·dwell). ★**핵심 실증**: (i) **양 regime 자동 적응**(단일 정책·무튜닝: in3600→d24, in2000→**d44** 수렴); (ii) **steady-state 오버헤드 0**(isolation: pin→static과 정확 동일 2.318≡2.319); (iii) **v7b(damped layer-span eval)로 양 regime static 매칭**(in3600 r2 **2.11**≈2.24, in2000 r3 **3.23**≈3.22·r4 **2.21**≈2.27). v6(prefill-boundary eval)의 in3600 transient(1.32)를 layer-span eval이 해소. **컨트롤러 진화**: v1 invisible-oscillation→v4 outlier-rejection→v6 neutral-start→**v7b damped layer-span**. **mechanism = Bullet(libsmctrl layer-span·cudagraph decode·latency signal)의 green-ctx 근사**(MuxWise 기판/granularity + latency 신호). ⚠️**type-aware span sizing 死**: prefill span을 attn/ssm 경계서 단축(homogeneous span)→**net-negative**(TTFT 2-4×↑, Zamba2 type-run ~6층<<budget 18층→span 3× 짧아 오버헤드). layer-type은 span 경계로도 이득 無. **layer-aware와 직교·전 investigation 유일 upside**. 진짜 이득 = **mixed/bursty load**(step B; cross-cell: 단일 static split이 두 regime 최적 불가) — 검증중.

## 4. 핵심 결론

1. **PD 분리(pdmux)는 항상 이득** — agnostic이 fused를 모든 모델서 이김.
2. **정적 최적 = tuned-uniform**(regime별 고정 split). agnostic도 이김.
3. **layer-aware(sub-step, decode·prefill 양측)는 전부 반증** — per-layer-type SM 재배분은 (D) granularity로 step-level을 못 이김. prefill-type(PF)은 starvation까지 겹쳐 최악.
4. ★**SLO-aware(동적 controller) = 유일 upside 정책, B로 payoff 확인** — (i) stationary: static 매칭(무튜닝 per-regime 수렴, isolation 오버헤드0); (ii) mixed 1-stress: best static과 TIE; (iii) **mixed dual-stress(어떤 static도 불가): static WIN +18%**(SLO-v7b combined 1.698 vs d44 1.438, 양 phase 승). **static에 절대 안 짐 + dual-stress서 격파 + 무튜닝 자동적응.** mechanism=Bullet(libsmctrl layer-span·cudagraph·latency)의 green-ctx 근사(layer-span damped 평가·idx-change시만 drain). 상세 `results/slo_sched/B_mixed_results.md`. caveat: green-ctx no-cudagraph 절대값 하한·dual-stress서만 명확 win·임계값 hand-tuned.

## 5. 왜 그런가 (통합 mechanism)

- **static 최적성**: 최적 split = prefill(compute-hungry)·decode(memory-satiated)의 **workload-SM 균형점**. tuned-uniform이 이를 잡음.
- **layer-aware 실패**: 그 균형을 **층타입으로 흔들면** (i) sub-step 재배분이 커널 timescale서 불가((D)), (ii) prefill-type은 mamba-prefill을 굶김(compute-bound). = 균형을 깨는 방향.
- **SLO-aware 잠재**: 균형점 자체가 **부하/regime 따라 이동**하는데 정적은 고정 → **동적으로 이동 추적**하면 정적을 이길 수 있음. 특히 mixed load.

## 부록 — 원자료
fused/agnostic/tuned: `results/r0a`,`results/r0c`. layer-aware decode: `results/r0d_coord_la`,`results/a_substrate`,`results/v4_multiwin`. layer-aware prefill(PF): `results/pf_boundary{,_fix1,_fix2}`. prefill knee: `results/prefill_knee`. SLO-aware: `results/slo_sched`.
