# (a) Substrate isolation — coord-la with GPU-ordering + pin removal (PDMUX_LA_COORD_OPT)

작성: 2026-07-08. 목적: R0d의 coord-la 124ms TPOT가 **substrate 오버헤드**(윈도우당 CPU
`synchronize` + decode SM-pinning)인지 **구조적**(monolithic prefill이 window 0만 오버랩)인지 분리.

## 변경 (dev-tree, env `PDMUX_LA_COORD_OPT=1`; 기존 `PDMUX_LA_COORD` 경로 불변)
- 윈도우당 `d_s.synchronize()`(CPU 블록 ~19×/step) → **GPU-side `wait_stream`/event 순서화** (스텝당 CPU sync 1회).
- 윈도우 1–18 decode **핀 제거**: prefill 끝난 뒤 decode를 full-SM 정규 스트림(idx3 = (0,108))에서 실행.
- full-SM decode ↔ prefill 꼬리 경합 방지: prefill event로 순서화.
- 정확성: boot_ok=1, SANITY "Paris" 정확(jobs 837520/837521). windowing 수학 불변(스트림/sync만 변경).

## 결과 — goodput@SLO (req/s) & TPOT p50 (ms)

### in3600/o32 (prefill-bound)
| rate | R0d coord(124ms) | **(a) OPT** | agnostic | d24 tuned-uniform |
|---|---|---|---|---|
| 1 | 0.785 (57ms) | **0.957 (51ms)** | 1.194 (41ms) | 1.196 (40ms) |
| 2 | 0.000 (124ms) | **0.311 (85ms)** | 2.063 (43ms) | **2.239 (43ms)** |
| 3 | 0.000 (123ms) | **0.000 (99ms)** | 0.622 (41ms) | **1.179 (41ms)** |
| 4 | 0.000 | 0.000 (99ms) | 0.313 | 0.473 |

### in2000/o96 (decode-heavy)
| rate | R0d coord | **(a) OPT** | agnostic | d44 tuned-uniform |
|---|---|---|---|---|
| 1 | 1.152 (47ms) | **1.174 (42ms)** | 1.17 | 1.17 |
| 2 | 0.803 (62ms) | **1.821 (51ms)** | 2.25 | 2.24 |
| 3 | 0.076 (117ms) | **0.433 (71ms)** | 3.24 | 3.22 |
| 4 | 0.000 (134ms) | **0.063 (105ms)** | 1.59 | 2.27 |

## 판정: substrate가 **약 절반**, 구조적 잔차가 **결정적**

1. **substrate는 실재했다** (R0d 124ms 마그니튜드는 confounded): OPT가
   - in3600 r2 TPOT 124→**85ms**, goodput 0→**0.31** (TPOT 갭의 ~47% 회수).
   - in2000 r2 goodput 0.80→**1.82** (agn까지 갭의 ~70% 회수, decode-heavy서 핀 제거 효과 큼).
   ⇒ **R0d의 "124ms"는 CPU-sync+pinning으로 부풀려진 값**. 사용자의 magnitude 회의는 옳았다.

2. **그러나 sign은 robust**: OPT도 rate≥2(in3600)·rate≥3(in2000)서 agnostic·tuned-uniform에
   **여전히 decisively 패배**. TPOT는 agnostic의 평탄 ~42ms에 **끝내 근접 못 하고** 50→99ms로 부하와 함께 상승.

3. **잔차의 정체 = 구조적, model-independent**: monolithic prefill 청크가 **window 0 하나만 오버랩** →
   나머지 ~48층 decode가 prefill 뒤로 **직렬화**. agnostic/tuned-uniform은 prefill∥decode 전-스텝
   **완전 오버랩**(=max(prefill,decode))이라 구조적으로 우위. 이 잔차는 윈도우 수(19 vs 9)와 무관.

## 세 가지 coord-la 실현이 모두 패배 (granularity (D) 삼중 확인)
| 설계 | prefill 오버랩 | 대가 | in3600 r2 TPOT |
|---|---|---|---|
| inefficient_v1 | 윈도우마다 slice (sim 비전) | 윈도우당 run_batch | 698ms |
| R0d optimized | window 0 통째 | 단일-윈도우 오버랩 | 124ms |
| **(a) OPT** | window 0 통째 + substrate 제거 | 단일-윈도우 직렬화 잔차 | 85ms |

⇒ fine-grained 오버랩을 사면(v1) 스케줄 오버헤드가, 오버헤드를 피하면(R0d/OPT) 오버랩이 죽는다.

## 함의 (다음 실험)
- **NemotronH 이식(윈도우 9 vs 19)**: 잔차가 window-count가 아니라 monolithic-prefill 단일-윈도우 오버랩이라
  **flip 가능성 낮음**. (다만 릴리즈 런이 길어 window0 오버랩 비중 약간↑: 7/52 vs 6/54.)
- **미검증 결정 설계 = version-4**: prefill을 **경량 `forward_split_prefill`로 윈도우마다 slice**(v1의 멀티-윈도우
  오버랩) + **(a)의 GPU-ordering**(v1의 run_batch 오버헤드 제거). sim의 실제 비전을 값싸게 realize하는 유일 경로.
  이게 지면 layer-aware definitively 死; 이기면 regime 발견.
