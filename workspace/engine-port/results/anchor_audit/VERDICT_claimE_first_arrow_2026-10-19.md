# VERDICT — 2026-10-12 연구 감사 의견 (D)절 "Claim E 증거 사슬이 첫 화살표에서 끊긴다" (claims-auditor 1회차, 2026-10-19)

- 대상: `handoff-report/session_handoff_2026-10-13.md:199-228`, `COSTMODEL.md:34-37`
- 대조: CONSENSUS §1-3(`reports/CONSENSUS.md:3682`)·§1-5(`:3684`)·§6 Claim E·G, 동결 MATRIX:640,
  A4 판정 `workspace/engine-port/results/hybrid_sched_s0/audit_a4_split_2026-10-01/VERDICT.md:21-45`,
  `reports/cost_model_design_space_2026-10-01.md:56-63,76`, opchar 감사 `AUDIT_opchar_2026-10-13.md:8-31`(opchar worktree),
  `workspace/engine-port/results/handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md:30-44,85-97`
- 측정 0, 새 수치 0. 재감사는 이 판정 이후 diff만, 최대 1회 남음.

## 하위 판정

| # | 주장 | 판정 |
|---|---|---|
| 1 | "첫 화살표(a: s_attn(SM)≠s_ssm(SM))가 끊겼다" | (i) 증거 부재 + 일부 (iii) 측정 무효로는 **PLAUSIBLE**, (ii) 반대 증거로는 **REFUTED** — "끊김"이 아니라 "미입증" |
| 2 | "a가 끊기면 Claim E가 죽는다"(a = 필요조건) | **REFUTED** — 우회 경로 α(조성/ctx 기울기 오귀속), β(Q18 CM-5 state 예약) |
| 3 | 단일 모델 내 generic 예측 오류 비교는 항등식, hold-out으로만 (A4 K2) | **CONFIRMED** (정밀화: 탈출구 = 모델 hold-out 또는 대상 모델 ctx 기울기를 재지 않는 이식 P_gen; 실질 1차 운반체는 hold-out) |
| 4 | "hybrid ctx 항은 shape C에서만 의미, shape B ~1%" | **NOT-YET-SUPPORTED** — 인용 금지 N2 PROFILE 위 산술 추정, 방향만 PLAUSIBLE, shape C 효과 크기도 미검증 |
| 5 | "Claim E는 현재 근거가 없다"를 우선순위 전제로 | **좁은 형태로만** — §6 "미검증 핵심 가설" 재진술까지 |

### 1의 근거
- §1-3 WIDE(Diff B≈1): prefill·triton·no-cudagraph micro(#8), Zamba2 한정(Nemotron 이식 금지), n_indep=1 → decode a 반증 불가.
- C2 "모델-무관" 2.36–2.91×: 측정점(B≈9–12, L≈1–1.5k) 트래픽 68–94%가 weight-sweep → 층 종류 동질성 증거 아님(#1 변형, 측정점 외삽).
- opchar: 격리 커널·random weight, 정책 추론 인용 금지(#7·N4). 내용은 오히려 계열별 스케일링이 다를 수 있다는 정황.
- decode per-type 유효 측정 0건(835571·858811 무효, 873783 UNAUDITED) = (iii).
- N2 PROFILE: 전체 모델 T_dec라 층 종류 분해 원리적 불가 + 인용 금지.
- 기판 혼합(#11): (D)는 KISTI·Zamba2/triton 계열(Diff B·C2)과 VESSL·Nemotron(N2) 근거를 한 문단에 묶음.

### 2의 우회 경로
- α: P_gen 정의는 `s_ssm≡s_attn` **및** "m1·bs를 a2·bs·ctx로 흡수(전 층 attention 가정)"(`cost_model_design_space:63`) — 둘째 성분은 a가 성립하지 않아도 틀린다. 단 A4 K2로 같은 모델 재적합 시 D≡0 → 이식 P_gen 또는 hold-out에서만 생존.
- β (Q18): SM 스케일링 무관. 제약 — (1) 등록 Claim E("decode floor")가 아니라 anchor 주장 1 ①의 재정의 → claim 문안 사용자 결정 필요, (2) 허수아비 위험(#2): "KV만 보는 generic"이 Bullet/MuxWise 실제 동작인지 미확인, SGLang 자체가 mamba pool 예약·보고, (3) 구속 전환점 ~52k(cap 48)는 유도값·미실측(X-6), shape A/B(≤8k)에서 자극 안 됨.

### 해소 실험(설계 수준, 미등록)
- α: A11 hold-out — 같은 백엔드(flashinfer; Nemotron-H 계열 + Qwen2.5-7B, gate #83 Zamba2 혼합 금지), 한 Job, cudagraph-ON, P_gen(Transformer 적합→이식) vs P_hyb 예측 오차, D{16,24,34,44,108}×b{1,8,32,48}×c{512,2048,8192}, boot n≥4·짝 CI·효과 ≥3% → 결정 발산 → 5-b 서빙.
- β: admission 거절 사유(KV/cap/slot) 계측 + cap {48, C_mem/2, C_mem} × ctx 52k 횡단 스윕, generic 용량 예측 vs 실현 → 서빙 goodput.
- 4: shape C를 5-a에 편입(A·B·C × static{16,24,34,44,108}, 한 Job, n≥4, 절대 SLO, cudagraph-ON, T_dec 항별 분해 병기).

## 안전 문안
- "Claim E(hybrid 정보 decode floor > generic/static)는 §6 기준 미검증이며 찬성 서빙 증거는 0건이다."
- "layer-type별 decode SM 스케일링 차이(a)는 유효 측정 0건(결함 job 철회, per-type UNAUDITED)으로 미입증이다."
- "단일 모델 내 예측표 발산은 재매개화 항등식이며, hybrid 귀속은 모델 hold-out(또는 충실한 이식 generic)에서만 시험 가능하다(A4 K2)."
- "'generic 예측기로 충분' 가설은 기각되지 않았다(검증되지도 않았다)."

## 금지 문안
"a가 반증됐다" · "s_attn≈s_ssm" · "C2 모델-무관이 층 종류 동질성을 보인다" · "Diff B≈1이 decode에서도 성립" ·
"Claim E가 죽었다/기각됐다" · "generic으로 충분하다" · "hybrid-aware의 독립 기여 없음이 확인됐다" ·
"shape B ctx 항 1%"(수치) · "shape C에서 hybrid 항이 유의하다" · N2 PROFILE 계수(2.51/4.63/1.95×, −14~16%) 일체.

## 우선순위 논의용 요약
Claim E는 §6 기준 미검증이고 찬성 서빙 증거 0건으로 Claim G(부분 지지)와 증거 수준이 비대칭이다. 그러나 (D)의 "첫 화살표 끊김"은
반증이 아니라 부재이며, a는 Claim E의 필요조건도 아니다(α 조성 오귀속 — hold-out/이식 generic 한정, β CM-5 state 예약 — claim 재정의·
허수아비·긴 ctx 한정). 우선순위 전제는 "E는 미검증, 판정엔 모델 hold-out 서빙 실험이 필요하고 비용이 크다"까지만. G를 headline으로
고르는 것은 이 근거로 정당화될 수 있으나 그 선택을 E의 반증으로 서술해서는 안 된다.
