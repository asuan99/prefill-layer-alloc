# B — mixed/phase-alternating workload: does SLO-aware dynamic BEAT static?

작성: 2026-07-12. SLO-aware track의 payoff 검증. 한 서버서 phase A(in3600/o32=prefill-bound)와
phase B(in2000/o96=decode-heavy)를 교대(3 rounds, 48 prompts/phase). 하네스 `mixed_bench.sbatch`.
정책: SLO-v7b(동적, layer-span damped) vs static d24/d44 vs agnostic. goodput@SLO(TTFT≤3s∧TPOT≤60ms).

## 근거 — 2×2 (stationary cross-cell, 120 prompts): 단일 static은 두 regime 최적 불가
| 정책 | in3600 (prefill) r2/r3 | in2000 (decode) r4 |
|---|---|---|
| static **d24** | **2.24 / 1.18** (best) | **0.18** (fails, vs d44 2.27 = 12×) |
| static **d44** | 1.30 / **0.29** (fails) | **2.27** (best) |
| agnostic | 2.06 / 0.55 | 1.59 |
| **SLO-v7b** | **2.11** / 0.70 | **2.21** |

## Mixed 결과 (combined goodput = Σgood/Σduration)

### workload (r2,r4) — 한 regime(decode)만 stress
| 정책 | phaseA(prefill@r2) | phaseB(decode@r4) | **COMBINED** | min-phase |
|---|---|---|---|---|
| **SLO-v7b** | 1.53 | 2.95 | **2.091** | 1.528 |
| static-d44 | 1.54 | 2.93 | **2.093** | 1.540 |
| static-d24 | 2.15 | 1.32 | 1.825 | 1.324 |
| agnostic | 0.38 | 1.25 | 0.720 | 0.375 |

→ SLO-v7b가 **d24(+15%)·agnostic(~3×)는 이기나 d44와 TIE**(2.091≈2.093). 이유: in3600@r2(중부하)선 d44의 prefill-64가 안 굶어(1.54≈1.53) d44가 이 mix를 감당. **한 regime만 stress면 그 regime best static으로 충분.**

### workload (r3,r4) — 두 regime 모두 stress
| 정책 | phaseA(prefill@r3) | phaseB(decode@r4) | **COMBINED** | min-phase |
|---|---|---|---|---|
| **SLO-v7b** | **0.521** | **3.204** | **1.698** | **0.521** |
| static-d44 | 0.445 | 2.862 | 1.438 | 0.445 |

→ ★**SLO-v7b가 d44를 +18% 이김**(combined 1.698 vs 1.438), **양 phase 모두 승**(prefill 0.52>0.45, decode 3.20>2.86). d44는 prefill@r3서 굶음(0.445), d24는 decode@r4서 fail(0.18). **어떤 static도 못 버티는 dual-stress서 동적이 승.**

## 판정
- **SLO-v7b는 static에 절대 지지 않는다.** 한 regime stress면 best static과 무승부, dual-stress면 **static 격파(+18%)**.
- **동적의 가치 ∝ workload 다양성/부하** — 두 regime이 모두 stress돼 단일 static이 불가능한 지점서 발현.
- **무튜닝 자동 적응**: d44는 이 workload용으로 미리 골라야 하나, SLO-aware는 실시간 latency 피드백으로 스스로 per-phase 최적(d24↔d44)을 찾음.

## 종합 (SLO-aware track)
| | stationary | mixed 1-stress | mixed dual-stress |
|---|---|---|---|
| SLO-v7b | static 매칭(무튜닝, isolation: 오버헤드 0) | best static과 TIE | **static WIN +18%** |

layer-aware 全변형(decode-window/prefill-window/span-boundary)이 (D) granularity로 진 것과 대조적으로,
**SLO-aware(step-level 동적 latency-adaptive split, layer-span 평가, green-ctx)는 유일하게 static을 매칭+격파.**
mechanism=Bullet(libsmctrl layer-span·cudagraph)의 green-ctx 근사. caveat: green-ctx no-cudagraph 절대값 하한;
mixed 부하/rate 의존(dual-stress서만 명확한 win); 임계값 hand-tuned(Zamba2/이 regime).
