# 사용자 결정 2026-10-19 — headline 비중·공통 워크로드·claim 확장 (main 세션, 측정 0)

전제: `VERDICT_claimE_first_arrow_2026-10-19.md`(claims-auditor 1회차). 선택지 원문은 같은 세션 대화의 경로 비교표(G / E-a / E-α / E-β / shape C).

1. **(가) headline = Claim G 중심** — "PD-mux 이득의 특성화·일반화"를 뼈대로 둔다. 구체 범위(E의 위치, G의 hybrid 고유성 근거, M3 역할)는 이어서 논의(미정).
2. **(나) shape C(긴 입력 + 긴 출력)를 공통 워크로드로 편입** — 구체값·`max_prefill_tokens`·G0/M3 반영 방식은 미정(G0 PREREG·M3 rev13 몫).
3. **(다) Claim E 문안을 memory/admission 예측(Q18, CM-5)까지 넓힐 필요는 인정** — 확정 문안·허수아비 위험 처리는 (가) 논의 뒤 이어서 결정.

금지: 이 결정을 "Claim E 반증"으로 서술하지 않는다(VERDICT 금지 문안).

## 후속 결정 (같은 날, (가) 범위 — 권고안 채택)
4. **G0에 순수 attention 대조 모델 편입**(예: Qwen2.5-7B, 같은 백엔드) — "G가 hybrid와 transformer에서 다른가"를 1차 비교로. α(hold-out) 재사용 가능. 모델·백엔드 확정은 G0 PREREG 몫.
5. **shape C = G0 두 번째 표준 셀**(arXiv-Sum은 긴 입력·짧은 출력이라 shape C 아님 — 출력 mean 203 [SPEC v0.2 :164]). 데이터셋 후보 OpenThoughts/Mooncake(10-15 P-C 논의), 확정은 G0 PREREG.
6. **M3는 G0 뒤로 연기**, J1(N=1) 결과 분석(CPU)만 먼저. J1 throttle(GPU0 61%)은 G0 계측 공시 항목(F-1)으로 처리.
7. E의 위치 = 기본 "열린 질문 + 시험 설계 공시", β(memory/admission)는 (다) 문안 확정 뒤 cap 특성화 Job(R16-3)에 계측 얹기.

## (다) 결정 (같은 날, 권고안 채택)
8. **Claim E 확장 문안 = 후보2 본문 + 후보1 anchor** (허수아비 점검: venue-strategist, 엔진 층 Q18 = 허수아비, Bullet/MuxWise = 변수 부재).
   - 본문(후보2): "SGLang에서 hybrid의 cap 하나가 admission·decode bs 상한·state 예약을 결합한다. 그 결합의 goodput 비용을 특성화한다."
   - anchor(후보1): "Bullet/MuxWise의 예측기는 메모리 변수 없이 엔진 admission을 상속한다. hybrid로 이식하면 decode 배치 상한이 KV가 아니라 고정 state 슬롯(cap)에 묶이고, ctx < [실측 전환점]에서 구속한다."
   - E(원안, decode floor 예측) = 열린 질문. 후보3(외부 KV 전용 용량 모델 X)은 X 실명 식별 전 금지.
   - 금지: "generic 스케줄러는 mamba state를 무시한다" · "Bullet/MuxWise가 C_mem을 오예측한다" · "~52k에서 구속 전환"(유도값).
   - `handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md` §3-2의 "generic 예측기(KV만 보는 모델)는 hybrid의 C_mem을 과대평가" 문장은 이 결정으로 낡았다(promote 때 정정 포인터).
9. 실험 = cap 특성화 Job(R16-3)에 I-5 거절 사유 계측 + cap×ctx 스윕(설계 초안 `../cap_char/`).

## 결정 8 개정 → 8′ (같은 날, 사용자 질의 후 승인)
경위: (1) SGLang mamba pool은 부팅 시 정적 할당, 슬롯 고갈 = assert(프로세스 사망), radix OFF에서 슬롯 수 = cap → cap≡슬롯 결합은 **SGLang 정적 풀의 구현 결과**(Jenga/vLLM V1 통합 풀이면 소멸). 할당기는 바꾸지 않는다(사용자). (2) I-5 조사(engine-porter): radix OFF에서 슬롯 사유와 cap 사유는 항등 → 거절 사유는 KV / cap(≡슬롯) 2분류. (3) "~52k 경계"는 cap 48 아티팩트지만 long-context 구간 자체는 hybrid 의의가 큰 곳이라 유지(사용자).
- **8′ E′ 문안**: "hybrid의 요청당 메모리는 길이 무관 고정 state s를 가져, 짧은 ctx(< s/k, Nano-9B-v2 ~9k [유도: 0.1435 GB ÷ 16 KiB = 8.8k(GB)–9.4k(GiB), 단위 확인 필요])에서는 s가, 긴 ctx에서는 (작은) KV가 지배한다. 이 비용 구조와 goodput 영향을 양 구간에서 특성화한다. SGLang 정적 풀에서는 cap이 admission·decode bs·state 예약을 함께 정한다(실험 설정 공시)."
- 이전 8의 후보2·후보1 문안은 대체됨("슬롯 상한"은 할당기 종속이라 기여 문장에서 뺀다).
- 실험: cap 블록 {48, C_mem} × 표준 셀 + long-ctx 블록 ctx {32k, 64k, 96k}(+128k 프로브 결과에 따라), cap = C_mem(L), ctx 비례 TTFT SLO. 설계 `../cap_char/DESIGN_cap_longctx_v1_2026-10-19.md`.
- 금지 추가: "~52k가 hybrid 경계" · "9k 경계 실측"(유도값, config 산술).

## 후속 결정 (2026-10-20, 설계 v1.1 §5 권고안 채택)
10. **attention 대조 모델 = Llama-3.1-8B-Instruct**(128k 네이티브, `--attention-backend flashinfer` 명시). 결정 4의 예시 Qwen2.5-7B를 대체. 교락(토크나이저·k 8×·attn 32층 RoPE·gated 라이선스·`ignore_eos` 필수)은 설계 v1.1 §5-2대로 공시.
11. **fused_tuned arm 추가**(cap 블록·long-ctx 블록): G0 튜닝 argmax config(mixed-chunk 포함). 근거 = J1 감사 1번(미조율 분모).
- shape C 데이터셋은 미정(LongProc의 서빙 선례 질의 중).

## 결정 12 (2026-10-20, shape C 재정의 — 사용자 결정)
경위: LongProc은 서빙 선례가 없고, 출력 길이를 특정 과제군에 묶어 연구 범위를 임의로 좁힌다(사용자). 비교 메모 `../cap_char/COMPARE_shapeC_workloads_2026-10-20.md`.
- **LongProc 제외** — 측정에도, 수치 근거로도 쓰지 않는다.
- **입력 원천 = arXiv-Sum(≤16k 표준 셀) + LongBench v1/v2(길이 구간별)**. 구간 표본이 부족하면 InfiniteBench로 보완한다. long-ctx 블록은 합성 고정 길이 대신 **실제 문서를 길이 구간 {32k, 64k, 96k} ±10%로 선별**한다(표본 실측 후 확정).
- **출력 길이 = 강제 출력 요인 {자연(~200), 1k, 4k}**. 근거는 데이터셋이 아니라 비용 구조(생성 중 L\* 횡단, Little 동시 수)다. 블록별 수준은 PREREG에서 GPU-h로 정한다.
- **shape C = (입력 구간 × 출력 길이) 격자의 한 영역**이며 단일 데이터셋이 아니다. 결정 5("shape C = G0 두 번째 표준 셀")는 "arXiv-Sum 입력 × 강제 출력 {1k, 4k}"로 구체화된다.
- 선행: LongBench v1/v2·InfiniteBench 길이 분포 CPU 실측(Nano·Llama 토크나이저).
