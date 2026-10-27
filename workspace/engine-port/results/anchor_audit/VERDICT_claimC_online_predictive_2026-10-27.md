# VERDICT — "C(iteration 단위 온라인 예측형 최소-SM 배분) 미주장" 명제 감사 r1 (2026-10-27)

> claims-auditor, 읽기 전용, GPU 0, 새 성능 판정 0. 같은 대상 감사 1회차.
> 발단: main 세션 권고(사다리 A=V-O static 선택 · B=H-D phase 전환까지만 주장, C 미주장)에 대한
> 사용자 반박("C 미주장 근거를 검증해야 한다, C 없이는 생태계 대비 빈약").
> 짝 문서: 학회 fit 검토 `reports/hymux_review/VENUE_C_positioning_2026-10-27.md`.

## 판정

- 명제 "C를 논문에서 (긍정) 주장하지 않는다": **CONFIRMED** — 단 정확한 상태는 **"C는 미검증, 시험 가능"**.
- "C는 증거로 기각됐다": **REFUTED** (CONSENSUS §3 항목57 HE0 ⇏ 천장 소멸; Claim E "반증 아닌 미검증", `VERDICT_claimE_first_arrow_2026-10-19.md:38-46`).
- 이전 권고 근거 (1) HE0를 C 반대 증거로 쓴 것은 과잉.

## 시험된 컨트롤러

| 컨트롤러 | 기전 | 결과 |
|---|---|---|
| bind / bind+GATE / slo (HE0) | reactive·symmetric, 이벤트마다 | best static 미달(관대 5.4σ CONSENSUS:3686 §1-7, tight ~10σ §1-17), Zamba2-2.7B |
| Step D `PDMUX_SLO_LFF` | 온라인 feedforward(ctx) | n=3 검정력 부족, net win 아님(CONSENSUS:3717 각주) |
| Step F `sat_predict` | 포화 예측 → static 앵커 수렴 | 시험됨, 목표가 static이라 C 아님 |
| R2 `generic`/`hybrid` (Claim E) | offline 프로파일 decode-floor | **GPU 측정 0건**(`src/multiplex/controller.py:75`); epoch 단위(≥4 iter·100 ms), `bucket_changed` 서빙 경로 미구현(CONSENSUS:7368), 재정렬·PAUSE 없음 → 현 상태로는 C 아님 |

## 컨트롤러 무관 반대 증거 (범위: Zamba2·HE0 변화 trace·legacy 하네스, Nano 미측정)

- §1-13: 정책 간 차이 97.3%가 HI phase, HI 최적이 LO에서도 무비용 → regime 간 최적 충돌 부재.
- §1-19: phase-oracle 동적도 best static 대비 +2.1% (n=1~2, 크기 강등).
- §1-17: tight SLO에서 static이 decode SM에 단조 증가(d16 33 < d24 41 < d34 50 < d44 73) → Bullet형 "SLO 만족 최소 decode SM" 목적함수가 손해 방향, risk/reward ≈ 18:1(CONSENSUS:3717).

## 재검토 주장 판정

| 주장 | 판정 |
|---|---|
| HE0에서 진 것은 reactive+symmetric | CONFIRMED ("16–24 진동"은 형태 부정확 — 16/34 이봉, SYNTHESIS §1) |
| 예측형은 시험된 적 없다 | **REFUTED(문구)** → "offline 프로파일 기반 최소-SM(Bullet Alg-1형/Claim E)은 서빙 측정 0건" |
| CM-4/6 약점은 NYS이지 반대 증거 아님 | CONFIRMED |
| T1 sticky-D F2/F3·T2 측정값 존재 | CONFIRMED, 단 C에 불리(분할 decode 외삽 13.6–18.3%, `AUDIT_costterms_result_r1_2026-10-27.md:23`) |
| hybrid 귀속 장애(A4 재매개화 항등식) | CONFIRMED |
| 충실한 baseline 공개 엔진 부재 | CONFIRMED (CONSENSUS §5-14:8752, 항목283:7539) |
| anchor 중심 주장이 원래 C | PLAUSIBLE — 원문은 "확장이 자명하지 않다", 반증 4갈래 모두 결과 |
| 결정 14 전환은 식별 문제로 멈춘 것 | PLAUSIBLE (결정 14 원문 미대조) |
| R2 hybrid가 C arm이 될 수 있다 | PLAUSIBLE(조건부) |

## C를 주장 가능하게 하는 최소 조건

1. 충실한 baseline P_gen+recal 같은 엔진에 구현(대상 모델 재측정 금지 — A4 K2).
2. arm: best static(재튜닝)·tuned fused·reactive(bind+GATE)·P_gen+recal·P_hyb·D\* 고정(다중 division 부팅 비용) — 한 Job·한 노드·telemetry 대칭.
3. hybrid 귀속: 모델 hold-out ≥ 1.
4. 비용 항: 분할 T_dec F1 등급화 · δ(p,d) 크기 · CM-6 대기열 항을 목적함수에 포함(최소-SM 목적은 entanglement 사슬을 못 봄).
5. iteration 단위 결정 실제 배선; 실 trace 재생 1종 + 변화 trace.
6. n≥4·paired CI·≥3%·정본 술어·용량 먼저·cliff 회피(q∈{0.8,1.0})·switch_count/체류/p50-p99 병기.
7. 선행 관문 H-D(PRESENT일 때만 기대값 근거; ABSENT는 C 기각 아님, PLAN_V:88).

confound: sim-vs-serving · untuned fused/허수아비 reactive · small-n · metric cliff · 재스코어 · cudagraph-ON · overhead 오귀속 · 다중 플래그 교락 · 기판(KISTI/Zamba2 vs VESSL/Nano) · 관측자 비대칭.

## 금지 문장
- "온라인 adaptive SM 배분은 hybrid에서 이득이 없다/불가능하다"
- "HE0이 예측형 컨트롤러를 반증했다" · "예측형 컨트롤러는 한 번도 시험되지 않았다"
- "동적 천장이 static과 같다고 확정됐다"
- "hybrid-aware 예측기가 generic보다 낫다" / "generic으로 충분하다"
- "우리 agnostic arm은 MuxWise/Bullet baseline이다"
- §1-19/§1-20 +2.1%·+16%를 caveat 없이 인용

## 허용 문장
- "이 저장소가 시험한 single-worker·SM-split 동적 제어(reactive 계열, n=3 온라인 feedforward)는 Zamba2-2.7B 변화 trace에서 best decode-heavy static을 넘지 못했다. 이는 달성 가능한 천장에 대한 진술이 아니다."
- "offline 프로파일 기반 iteration 단위 최소-SM 배분(Bullet/MuxWise형)의 hybrid 확장은 서빙 측정 0건으로 미검증이다. 판정에는 충실한 generic baseline과 모델 hold-out이 필요하다."
- "HE0 trace에서는 per-phase 최적이 인접해 phase-oracle 여유가 작았다(n=1~2, 크기 미확정). Bullet형 최소-SM 목적함수가 이 regime에서 prefill 쪽으로 움직인다는 점이 C의 위험 요인이다."

## 사다리 위치(권고)
C = 사다리 D단 "미검증 가설 + 시험 설계". 진입 관문 = H-D PRESENT + 조건 1·3.
