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
