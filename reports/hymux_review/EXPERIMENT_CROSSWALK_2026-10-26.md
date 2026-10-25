# 비용 모델 → goodput 실험 대조표 (2026-10-26, main 세션)

> **지위**: 정본 아님. 새 판정 0, GPU 0. 사용자 질문("이 비용 모델이 goodput 향상으로 이어지려면 어떤 실험이 필요한가, 현재 계획 실험의 의의, 아티팩트 문서 도출 실험과의 대조")에 대한 정리.
> **대상 문서**: 원고 "HyMux … (원고 v4)" `claude.ai/artifact/9fRb5GXFeCBYCvr1TtP8Yr` · "HyMux 검증 실험 계획과 유의사항" `claude.ai/artifact/NZHeWGiLPsk9FGkCp6n3k4`(이하 계획 문서). 두 문서 본문은 수정하지 않았고 검수 사항은 문서 댓글로 남겼다(원고 14건, 계획 문서 9건).
> **현재 계획**: 비용 항 PREREG rev3.5 `workspace/engine-port/results/cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md`(미봉인) + 결정 1–30 `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`의 G0·V 계획.

## 1. 비용 모델이 goodput 향상이 되기까지의 사슬

비용 모델이 goodput을 올린다는 주장은 다섯 고리가 모두 닫혀야 성립한다. 한 고리라도 비면 그 위의 수치는 인용하지 않는다.

| 고리 | 질문 | 닫는 실험 | 현재 상태 |
|---|---|---|---|
| ① 항 정확도 | 모델이 T_dec·T_span·ρ·C_mem을 맞히는가 | T1–T4(PREREG rev3.5) | 식은 [유도]로 성립, 상수 측정 0건 |
| ② 항의 hybrid 고유성 | state 항·조성 항이 실제 시간에 보이는가 | 조성 스윕 J-H1/J-H2 + Llama, T1의 b_D(요청당 항) | 계획됨(J-H), 결과 0 |
| ③ 예측 → 설정 | 모델이 고른 설정(D, span 예산, cap)이 oracle 설정에 가까운가 | **V**(조립 검증 + oracle 격차) | PREREG 미작성 |
| ④ 설정 → goodput | 그 설정이 tuned baseline보다 goodput이 높은가 | **G0**(tuned fused·기본 PD-mux·best static, shape C) | PREREG 미작성 |
| ⑤ 동적 이득 | 설정을 런타임에 바꾸는 것이 best static보다 나은가 | HyMux X1·X2·X5(계획 문서) | 구현 0. 정본 HE0는 기존 동적 제어기가 best static을 못 넘었다고 확정 |

**핵심 순서**: ③을 정적 설정만으로 먼저 닫으면 논문 방향 ②(비용 모델 기반 static 설정, 트랙 권고)가 성립한다. ⑤는 ③에서 "부하 구간별 oracle static이 전체 best static과 다르다"(동적 여지)가 확인될 때만 시도할 가치가 있다.

## 2. 현재 계획 실험의 의의

| 실험(Job) | 재는 것 | 비용 모델에 주는 것 | 원고 주장 중 판정하는 것 |
|---|---|---|---|
| **smoke + 하네스층 감사** | 실제 ramp, 셀 손실, 하드캡 | 봉인 가능 여부 | 없음(선행 조건) |
| **T1** J-A1/J-A2 | decode step `S_eng`(D 16–64·108u·108v × bs × 실제 ctx), cudagraph ON | `a_D + b_D·bs + c_D·bs·c` 계수. **a_D = 공유 weight 읽기 → BW(D) 곡선**, b_D = state 항, c_D = KV 항 | §3.2 시간 knee(미측정 행), §5.1 a₀/b₀/c₀ 대 실측, ω = BW/80 가정, §7.5 decode 예측 오차 |
| **T2·T2′·T2-ext** J-A | 단일 prefill TTFT(P=108), decode 1개 동거 시 P=108−D, 장문 98k·124k | `u + v·L + w·L²`, q(초선형 지표), **θ′(P 의존 지수)** | §3.2 "prefill 처리율은 P에 비례"(θ′≈1?), §2.1 이차항 크기, Figure 5(c) 유도 |
| **T3** J-A·J-H2 | 동거 ρ, span σ′, 정지 비율 q_stall(총효과, 전력 한도 포함) | 닫힌 식에 없는 HBM 공유 감속 | §7.5 prefill 잔여 처리율(93–96%/84–90%), §8 전력 |
| **T4** J-B | C_mem 산식 대 실현 동시 수, 거절 사유 몫(KV/슬롯) | 구속 자원 전환(admission 입력) | §3.3 Figure 3 아래, 계획 X8 |
| **조성 스윕** J-H1/J-H2 | a4/a13/a22/a31(같은 골격, dummy) × T1·T2·T3·T4, D24/D44 | **층 종류별 항을 식별하는 유일한 경로**(T1은 전체 모델 시간뿐) | §3.2 층별 SM 민감도(2.7/2.5/2.4×, 현재 근거 0), §7.6 Transformer 대조(a31) |
| **Llama 참조** J-B | 다른 모델 T1·T2 | 참조점(교락 공시 필수) | §7.6(보조) |
| **TP2** J-C | 샤딩 사실, TP2 T1·T2·T4, 장문 | N 항 | §7.8 TP(원고는 Falcon-H1-34B TP2/4 — 계획과 모델이 다름) |
| **V**(미작성) | 조립 예측기의 λ\*_SLO 예측, **oracle 격차** | ③ 고리 | §7.5 "보정된 해석식 ≈ 격자 프로파일", 계획 X1 |
| **G0**(미작성) | 헤드라인 goodput, tuned fused(mixed-chunk, chunk 스윕), shape C, 순수 attention 대조, 전 arm 계측 OFF | ④ 고리의 baseline·워크로드 | §7.1 chunk 스윕 행, §7.2 baseline, 계획 X3 |

## 3. 계획 문서 도출 실험 ↔ 현재 계획

| 계획 문서 | 현재 계획의 대응 | 판정 |
|---|---|---|
| G1 출력 정확성 | 없음(HyMux 미구현) | HyMux 구현 시 신규 |
| G2 분할 실현 | PREREG §1-8 실현 SM 검증(`SM_NOT_REALIZED`) | **이미 있음** |
| G3 거절 사유 계측 | T4 거절 사유 몫 | **이미 있음** |
| G4 프로파일 재현성 | J-A1/J-A2 두 round 계수 차(T1 적합도 iv) | **이미 있음** |
| G5 CUDA Graph 유지 | `ABORT_CG_BS`, cudagraph hit 정황 | **이미 있음** |
| letter 실험 1(시간 knee) | T1 b_D·c_D | **이미 있음** — 판정 문장만 추가 필요 |
| letter 실험 2(예측 오차) | T1 적합도 + V | 부분(V 필요) |
| letter 실험 3(고정 상한 sweep) | T4 | **이미 있음** |
| letter 실험 7(attention 비율별 chunked–PD-mux 격차) | 조성 스윕은 비용 항만, goodput 없음 | **신규**(결정 25: 조성 변형 goodput은 G0 범위) |
| X1 W1 동등 / X2 SLO-feedback 대비 | 없음 | HyMux 구현 후. HE0 trace 재사용 |
| X3 W3 span 효과 | G0 shape C 셀(baseline 쪽) | HyMux arm 신규 |
| X4 단일 층 정지 상한 | T2·T2′(전체 prefill만) | **보강 필요**: 층 종류별 span 시간 |
| X5 W4 regime 교대 + phase oracle | 없음 | **static arm만으로 먼저**(아래 C-5) |
| X6 ablation | 없음 | HyMux 구현 후 |
| X7 admission 붕괴 | 없음(정본 entanglement 사슬 관측만) | HyMux 구현 후 |
| X8 구속 자원 전환 | T4 | **이미 있음** |
| X9 대조 모델 | 조성 스윕 a31 + Llama + G0 순수 attention 대조 | 부분 — a31이 가장 깨끗한 대조 |
| X10 결정 비용 | 없음 | HyMux 구현 후(정본 CM-3·(E)는 "무시 가능") |
| §7.8 Falcon-H1-34B TP2/4, replica | J-C는 Nano TP2만 | **보류 권고** |
| Bullet·DuetServe 재구현 | 없음 | **보류 권고**(이식 충실도 위험, 계획 문서 §3도 같은 경고) |

## 4. 구체화할 실험 목록 (우선순위)

### A. 기존 계획에 판정만 더하기 (GPU 추가 0, 봉인 전 PREREG 문안 추가)
1. **A-1 BW(D) 곡선**: T1 계수 a_D에서 `BW_eff(D) = W / a_D`를 기술값으로 도출. 판정 대상 = 원고 ω = BW/80(80 SM 포화) 가정. 정본 금지 서술("고-SM 평탄화 = HBM 포화")을 넘지 않게 "달성 대역폭"으로만 쓴다.
2. **A-2 시간 knee**: T1의 b_D·c_D로 `c_knee(D) = b_D / c_D`를 내고 유도값 2s/k ≈ 17.5k와 비교(자릿수 판정). 원고 §3.2 "시간에서도 knee" 미측정 행을 닫는다.
3. **A-3 prefill의 P 선형성**: T2′의 θ′를 "θ′ ∈ [0.9, 1.1]이면 `PREFILL_P_LINEAR_SCOPED`"로 라벨화. 원고 §3.2·Figure 5(c)와 opchar "74 SM 이상 효율 하락"을 함께 판정.
4. **A-4 조성 기울기로 층 종류별 항**: J-H1의 P3(ctx 기울기 ∝ attention 층 수)에 더해, 조성 간 T2 회귀 `T = Σ n_X·t_X(L) + c`를 기술값으로 등록. roofline 감사 r1의 해소 실험 (B)와 같다.

### B. 기존 Job에 셀 추가 (소량 GPU)
5. **B-1 층 종류별 span 시간(X4 보강)**: J-A의 T2′ boot에서 `PDMUX_PHASE_EVENTS` span 기록을 L {8k, 32k, 64k} × P 2점으로 수집. span이 여러 층을 덮으면 조성 회귀(A-4)로 분해. 판정: 층 하나의 prefill 시간이 θ_stall을 넘는 L.
6. **B-2 J-H2 D 축**: 층 종류별 decode SM 민감도의 유일한 경로인데 D 2점·공통 셀 1개라 약하다. D 16을 추가할지 사용자 결정(셀 실행 가능성은 smoke 후).

### C. 새 PREREG
7. **C-1 V(oracle 격차)**: T1–T3 적합으로 조립한 예측기가 고른 static (D, cap)과, 같은 trace에서 D 격자 전체를 돌려 고른 oracle static의 goodput 격차. ③ 고리이자 논문 방향 ②의 핵심 증거. 계획 X1의 "oracle 없이 oracle 도달"과 같은 질문을 동적 제어 없이 묻는다.
8. **C-2 G0**: tuned fused(mixed-chunk on, chunk {256…8192} 스윕), 기본 PD-mux, best static, shape C(arXiv-Sum 입력 × 강제 출력 {1k, 4k}), 순수 attention 대조(a31 또는 Qwen2.5-7B), 전 arm 계측 OFF, n ≥ 4, 변화 trace. ④ 고리.
9. **C-3 attention 비율별 chunked–PD-mux 격차**(계획 letter 실험 7): G0 안에서 조성 변형 a4/a13/a22/a31 × {tuned fused, PD-mux}. 결정 25의 "조성 변형 goodput은 G0 범위"를 구체화. 원고 §7.2 "chunked 대비 이득은 attention 비율이 낮을수록 작다"를 판정.
10. **C-4 동적 여지(X5 선행, static arm만)**: W1/W3 phase 교대 trace에서 phase별 oracle static과 전체 best static의 goodput 차. 차가 rep SD 이내면 HyMux(⑤)는 이 기판에서 이득 여지가 없으므로 구현을 보류한다. HE0의 "positioning 증거이지 천장 증거가 아님" 범위 축소와도 맞물린다.

### D. HyMux 구현을 전제로 하는 것 (C-4 통과 후에만)
11. HyMux partitioner·span scheduler·admission 구현 → G1 → X1·X2(HE0 trace, SLO마다 재튜닝) → X3 → X5 → X6·X7 → X10.
12. 보류: Falcon-H1-34B TP2/4, replica 확장, Bullet·DuetServe 재구현, vLLM.

## 5. 이 정리에서 바뀌지 않는 것
- 정본 결론(HE0, layer-type 런타임 정책 死, C2 scoped)은 그대로다.
- 원고 §2.1·§5.1의 닫힌 식과 수치는 `analytic_terms.py`와 일치한다([유도], 2026-10-25 대조).
- GPU Job 제출은 smoke → 하네스층 감사 → 봉인 → 사용자 승인 순서를 따른다.
