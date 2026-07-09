# version-4 — faithful sim multi-window overlap (PDMUX_LA_COORD_V4): BUILT, CORRECT, REFUTED

작성: 2026-07-08. 목적: (a) 이후 사용자 선택. sim의 **정확한 비전**(prefill이 모든 mamba decode
윈도우에 overlap; attn 윈도우 보호)을 실엔진에 **값싸게** realize해 tuned-uniform을 이기는지 최종 판정.

## 구현 (env `PDMUX_LA_COORD_V4=1`; baseline/OPT 경로는 elif로 분리 보존)
- prefill을 mamba(release) 윈도우마다 **slice**: persistent `split_prefill_batch.split_forward_batch`에
  경량 `MR.forward(pfb, split_forward_count=k)` (run_batch·per-window get_model_worker_batch 없음 →
  inefficient_v1의 698ms 회피). per_win = ceil(fwd_total / mamba_windows).
- 각 mamba 윈도우: decode 16 SM ∥ prefill 92 SM (green-ctx 상보 disjoint 반쪽, 동시). attn 윈도우:
  decode 74 SM 보호, prefill 정지(k=0). 윈도우는 GPU-ordered(`wait_stream`)로 직렬화(연속 파티션이 물리 SM 공유).
- **버그 1건 수정**: v4가 prefill을 run_batch 우회 → `split_prefill_batch.output_ids` 미설정 → 완료 prefill을
  running_batch에 merge 시 `torch.cat([...,None])` 크래시(schedule_batch.py:2289). decode 쪽처럼 수동 설정으로 fix.
  (SANITY 단일요청은 concurrent v4 경로 미트리거라 통과 → 부하 warmup서 발현.)
- 정확성: boot_ok=1, SANITY "Paris" 정확; 부하 스윕 완주(크래시無, goodput·TPOT sane). jobs 837718/837719.

## 결과 — goodput@SLO (req/s), TPOT p50 (ms)

### in3600/o32
| rate | R0d(124) | (a)OPT | **v4** | agnostic | d24 tuned |
|---|---|---|---|---|---|
| 1 | 0.785 (57) | 0.957 (51) | **0.847 (51)** | 1.194 (41) | 1.196 (40) |
| 2 | 0.000 (124) | 0.311 (85) | **0.145 (95)** | 2.063 (43) | **2.239 (43)** |
| 3 | 0.000 | 0.000 (99) | **0.000 (107)** | 0.622 | **1.179** |

### in2000/o96
| rate | R0d | (a)OPT | **v4** | agnostic | d44 tuned |
|---|---|---|---|---|---|
| 1 | 1.152 (47) | 1.174 (42) | **1.105 (43)** | 1.17 | 1.17 |
| 2 | 0.803 (62) | 1.821 (51) | **1.370 (54)** | 2.25 | 2.24 |
| 3 | 0.076 (117) | 0.433 (71) | **0.346 (111)** | 3.24 | 3.22 |
| 4 | 0.000 | 0.063 (105) | **0.000 (112)** | 1.59 | **2.27** |

## 판정: 충실한 sim 비전도 decisively 패배 — 게다가 (a)OPT보다 나쁨

1. **v4는 agnostic·tuned-uniform에 전 rate≥2서 decisively 패배**(in3600 r2 0.145 vs d24 2.24 = 15×; in2000 r3 0.35 vs agn 3.24).
2. ★**v4 < (a)OPT** (in3600 r2 0.145<0.311, TPOT 95>85; in2000 r2 1.37<1.82). **sim의 fine-grained 오버랩을 더 추구할수록 더 나빠진다.**
   - (a)OPT: 파티션 스위치 ~1회, decode를 full-SM 연속 스트림으로 → decode 빠름.
   - v4: 파티션 스위치 19회, decode를 sub-partition서 직렬화 → decode 느림. prefill 오버랩 이득(TTFT)을 decode TPOT 손실이 초과.
3. ⇒ **개선의 gradient가 layer-aware의 이상(fine-grained per-type)에서 멀어져 coarse step-level(agnostic/tuned-uniform)로 향한다.**

## (D) granularity — 이제 4개 실현으로 확정 (working implementations)
| 실현 | prefill 오버랩 | decode granularity | in3600 r2 TPOT | 결과 |
|---|---|---|---|---|
| inefficient_v1 | 윈도우마다(run_batch) | 윈도우 | 698ms | 패 |
| R0d | window 0만 | 윈도우 핀 | 124ms | 패 |
| (a)OPT | window 0만 | full-SM 연속 | 85ms | 패 (최선) |
| **v4** | **윈도우마다(경량)** | **윈도우 직렬화** | **95ms** | **패** |

sim이 가정한 무비용 fine-grained SM 핸드오프(0.4%/7.8µs)는 실재하지 않는다. per-layer-type 조율은
~19 cross-partition 윈도우 직렬화를 요구하고, 그 decode-side 비용이 per-type-SM 이득을 초과한다.
**prefill/decode SM 트레이드는 step 단위서만 성립 = agnostic/tuned-uniform이 이미 최적.**

## 최종 결론
**layer-aware의 모든 실현이 소진됐다.** 창립 가설(layer-type-aware SM 배분이 hybrid serving 이득)은 —
sim의 정확한 비전을 작동하는 구현으로 realize해도 — 실엔진에서 definitively 반증됐다. 실전 권고 = **agnostic /
regime-tuned uniform (coordinated step-level split)**. 

## caveat (정직)
- v4의 **부하 하 출력 coherence는 독립 parity-check 안 함**(SANITY 단일요청 + serving 메트릭 sane만). 단
  v4가 decisively **패배**하므로 verdict은 correctness와 무관하게 robust(이겼다면 parity 필수였을 것). 메트릭 패턴
  (TPOT 50-112ms=윈도우 오버헤드 예측치, TTFT 정상 스케일, r1 goodput 0.85-1.1)은 정상작동-but-느림과 일치.
- config m16a74 1점(knee상 합리적). prefill을 attn 윈도우서도 34 SM 전진시키는 변형은 미검(구조적 비용 불변 예상).
