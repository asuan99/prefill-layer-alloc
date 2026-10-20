# VERDICT — PREREG_COSTMODEL_TERMS_rev1 규칙층 감사 1회차 (claims-auditor, 2026-10-20) = **NO-GO**

死因 2건(N2 반전 확인, 둘 다 수리 비용 낮음) · MAJOR 7 · MINOR 6 · OK 2. 2회차는 수리 diff만 본다(상한 2회).

## 死因
1. **T4 고정점(§11-9, BLOCKER)**: C\_mem 산식에 출력 예약 항이 없다. `ignore_eos`이고 radix OFF이면 출력 전량이 예약된다(`sglang_engine_dev_multigpu/python/sglang/srt/managers/schedule_policy.py:647-652,752-753`). 그래서 실현/예측 비가 L=4k 0.51 · 8.8k 0.69 · 16k 0.80 · 64k 0.89로 설계 단계에서 이미 `CMEM_PRED_MISS`로 정해진다(출력을 줄이면 `FILL_INCOMPLETE`). §6-2의 "shape C KV 약 81"도 예약을 반영하면 약 60이다.
   - 수리: 예측식을 `floor(풀/(s+k·(L+예약)))`로 바꾼다. 예약은 ignore\_eos이면 출력 전량, 아니면 min(out, 4096)×new\_token\_ratio다. 원식과 수정식을 둘 다 봉인하고 L별 예측 비를 데이터 전에 표로 낸다. 이 대조는 엔진 회계의 항등식에 가깝다는 점(교훈 9)을 공시하고 "구현 사실 확인" 라벨로 낮춘다. V 예측기의 동시 수 상한과 cap\_V도 같은 식으로 고친다.
2. **V 사다리(BLOCKER, 미등재 A)**: 중심이 arm 예측의 기하평균 ±2 rung이다. 그래서 예측 폭이 1.33–1.44일 때, 25% 이내로 맞춘 arm이 `V_FAIL_OUT_OF_LADDER`로 바뀐다(w=1.44에서 e∈(0.182, 0.223]). SP R9-3("모든 arm 위·아래 각 2 rung")에도 어긋난다.
   - 수리: arm별로 ±2 rung을 둔다.

## MAJOR
- (1) 조립 귀속 불가와 항등식 위험: `V_PASS`에 "V-diag `TERM_DEVIATES` 0개" 조건을 더하고, 아니면 `V_PASS_CANCELLATION`으로 둔다. 귀무 예측기(G\_pred=1 또는 T1 단독)를 V 데이터 전에 봉인하고 그보다 오차가 작아야 PASS로 한다.
- (2) 허용 오차와 Job 효과: 10%는 KISTI 기판, 다른 모델, 미조율 fused에서 빌린 값이다. Job 간 대입은 §0 자기모순이다. J-V 안에 내부 앵커(V arm D와 108v의 T1 4셀 + T2 3L)를 두고, 3%를 넘으면 `V_JOB_SHIFTED` + 보정 예측 2차 병기. R9-6 해상도가 5%를 넘는 셀은 `V_PASS`에 도달할 수 없게 한다.
- (4) T1 외삽: shape C 영역(bs 55–80, c 11–20k)이 격자 볼록껍질 밖이다. (64, 8k)·(64, 12k) 셀을 추가하고 `PRED_EXTRAPOLATED` 처리를 봉인한다.
- (7) Llama·108v는 엔진 `decode_iteration`이 PD-mux mixin에서만 방출된다(`src/multiplex/multiplexing_mixin.py:2245-2270`). 그래서 클라이언트 간격뿐이고 GIL·클라이언트 과부하(confound 7)에 노출된다. 108u에서 엔진 S-per 대 클라이언트 간격을 bs별 3% 이내로 대조하고, 불일치하는 bs는 제외한다. "hybrid 고유"는 원인 문장으로 쓸 수 없고 Nano 1개 대 Llama 1개의 기술 차이까지만이다.
- (8) T3: 중앙값 ρ가 정지를 가리고(SLO는 요청 내부 p95, 프로브 64k 최대 정지 0.8 s), prefill 쪽 감속(span co/solo)이 빠져 있다. iteration 분포 전체, 정지(>5× 중앙값) 횟수, span co/solo를 기록한다. `max(T_dec·ρ, T_span)`의 T\_span 단위(span인지 전체인지)를 봉인한다(N2 RESULT :68-72: decode는 prefill 청크를 기다리지 않음).
- (10) R8-2′: mem 0.82→0.6과 cap=G0 cap→cap\_V를 동시에 바꾼다. SP 문언 개정이 필요하다(R8-2c C=512 prefill 몫 388토큰, 75.8%). O-1 사용자 승인 전에는 봉인 불가.
- (B, 미등재) V arm의 sticky 모드가 미등록이고 `T2_P_NOT_REALIZED`일 때의 대체 규칙이 없다. sticky를 같게 고정하고 실현 SM을 로그로 검증한다. 대체로 bs=1 decode 동거 상태의 T2 측정을 봉인한다.

## MINOR / OK
- MINOR: (3) 상위 2 D 범위 문장 · (6) fused\_tuned 자연 셀 튜닝 편향은 G 문장에만 · (12) 합성 대표성(G7 허용 문안 6까지) · (13) 64k·T2-ext는 프로브 허용 문안 3–4 승계 · (C) cudagraph 패딩(예측기에 ceil(캡처 bs)) · (D) J1 기울기 비교는 0.82 결과라 기술값으로만, 공시 · (E) metric cliff(trace 기반 확률 시뮬레이션, 예측 쪽에도 `CLIFF_ADJACENT`).
- OK: (5) n=2 round(게이트 3은 판정용, 항 맞춤은 기술 적합; V는 "이 Job에서") · (11) Claim E와의 함의 한계 · (F) duration 합산은 조건부(λ\*\_thr 쓰면 R5-1 승계 명시).
- caveat: (G) 전력 상태 차이(항은 저부하, V는 power cap 노출이 높은 부하).

## 결정 14와의 정합
정합이다. 항 → 조립 → 대조 구조이고 Llama는 칸 채우기에만 쓴다(약 24.5 GPU-h). Nano 단일 모델의 비용 모델이다.

## 등록 caveat (문자 그대로)
- "이 비용 모델은 Nemotron-Nano-9B-v2 한 모델의 것이며 'hybrid 서빙 비용 모델' 일반이 아니다."
- "V는 한 Job·두 셀의 예측 일치이며, 허용 오차 10%는 KISTI 타 모델 G에서 빌린 값이다."
- "agnostic 최선은 예측 상위 2 D 중 최선이다."

## 수리 우선순위
① 死因 2건(T4 예약 항, arm별 ±2 rung 사다리) ② 귀무 예측기 + cancellation 라벨 ③ J-V 내부 앵커 ④ sticky·P 실현 대체 규칙 ⑤ 엔진 S-per 대 클라이언트 간격 대조 ⑥ T3 꼬리·span 측 감속 ⑦ bs 64 셀 ⑧ R8-2′ 문언 + O-1 ⑨ MINOR 일괄.
