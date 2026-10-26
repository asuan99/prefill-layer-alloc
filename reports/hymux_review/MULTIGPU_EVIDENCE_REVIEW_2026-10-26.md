# multi-GPU 증거 검토 — 논문 메인 지표 관점 (2026-10-26, main 세션)

> 지위: 정본 아님, 새 측정 0, 새 판정은 claims-auditor 1회차(읽기 전용)와 venue-strategist 조사의 전사. J2·J4·J8 DP LABELS(G 값)는 열지 않았다(M3_SCOPE 규칙 3).

## 1. 결론
- 현재 multi-GPU **성능 수치 0개**(N>1 G_sym 전부 미정의, PROJECT_STATUS:16, COSTMODEL:26). 인용 가능 = 정확성 게이트(범위 한정)·기판 사실·J1(N=1) 기술값·시간 모형 과소 예보뿐.
- 원고 v4 §7.8 수치 3개(Falcon-H1-34B TP2 1.18×·TP4 1.15×, replica 4개 3.9×, 이득 1.20–1.22× 유지)는 출처 없음. "이득 유지"는 J1 Nano(N=1) 관측(A ≈ 1, B는 PD-mux 불리 부호)과 충돌. 3.9×는 복제 처리율(F-1 항등식). §8 "TP에서 c* 불변"은 산술상 PLAUSIBLE(N ≤ 8에서 KV head 8·mamba head 128·groups 8 나누어떨어짐), 측정 계획은 J-C TP2뿐.
- **M3 S2f는 메인 지표로 불충분**하고, S2f의 진행 조건 ①(DP fused_sym·agnostic boot2 rc 0 완결)이 세 Job 모두 FAIL — 등록 M3 마무리에도 S9f(약 141.8 GPU-h [유도])가 필요.
- 선행 논문: 장치 내 SM 분할 계열(Bullet·DuetServe·Nexus)은 단일 GPU 메인 + TP 확장성 절, top systems 학회 PD 다중화·분리(MuxWise ASPLOS'26·NanoFlow OSDI'25·DistServe OSDI'24·Splitwise ISCA'24)는 70B급 TP8 헤드라인. 가장 가까운 MuxWise는 단일 GPU 이득 1.2×, 큰 이득은 TP8.

## 2. 회수 자료 완결성(값 미개봉; 매핑 정정 J2 = e559c64, J4 = 213df1b, J8 = 3167f74)
| Job | 앵커 N1 | DP boot1·ext | DP boot2 rc 0 | boot2 미완 | TP boot | 셀(완결/부분) |
|---|---|---|---|---|---|---|
| J2 | 2/2 | 4/4·4/4 | fused, agnostic | d44 진행 중, fused_sym 미시작 | 0 | 112 (111/1) |
| J4 | 2/2 | 4/4·4/4 | d44, fused | fused_sym 진행 중, agnostic 미시작 | 0 | 111 (110/1) |
| J8 | 2/2 | 4/4·4/4 | fused, agnostic | fused_sym 진행 중, d44 미시작 | 0 | 119 (118/1) |
회수 시점 sha 대조 OK 기록은 있으나 manifest가 로컬에 없어 재검증 불가, DONE·SHA256SUMS 없음(운영 모델상 "완료" 아님). 9.5 h 정지: 세 Job 11:43Z 동시 시작·21:01–21:05Z 마지막 기록·21:14Z 종료, J1(07:49Z 시작)은 8.60 h 완주 → 경과 시간 상한인지 벽시계 사건인지 구분 불가(원인 미확인).
**정본 갱신 후보(doc-steward)**: CONSENSUS:8799 "telemetry 비대칭은 M3 J1뿐"은 사실과 다름 — J2·J4·J8도 PD-mux arm만 `PDMUX_TELEMETRY_PATH`(J8 DP `env.json` 확인). rev12 자료를 메인에 섞으면 결정 22 위반. `venue_positioning.md` §5 "Bullet ASPLOS'26"은 게재처 미확인(arXiv 2504.19516으로만 인용).

## 3. 인용 가능 문장(claims-auditor)
1. Nano M2g 정확성 게이트 통과(J2 TP2 G2 NOT_TESTED, J4·J8 재실행으로 TP4·TP8 4 arm 포함; 범위 prompt 64·max_new 128·context 16384) — CONFIRMED(범위)
2. TP4·TP8 `TP_NUMERIC_DIVERGENT`(|Δlogprob| p99 0.0656 > 0.05, top-1 불일치 0.25%·0.23%) 병기 의무 — CONFIRMED
3. 메모리 규칙 (iii)이 TP 값을 본 뒤 폐지됐고 원 규칙이면 TP config 전부 `MEMORY_MISMATCH` — 공시 의무
4. 기판: 전 쌍 NV12, J8 NUMA0 = GPU0–3/NUMA1 = GPU4–7, 드라이버 580.105.08, 400 W, 이미지 `1e0d335b…`, Job마다 pod 다름; NUMA 교차 효과 미측정
5. J1(N=1) 감사 허용 문안 5개만, shape B 부호(agnostic λ*_thr < fused_sym)는 PLAUSIBLE·크기 금지
6. "G_sym은 N>1 전부 미정의", "시간 모형이 확장 단계를 빠뜨림(J1 8.60 h 대 5.88 h)"

## 4. 메인 지표로 쓰려면(최소 실험 세트, claims-auditor 제안)
- estimand: **G_SLO(N, mode, shape) = λ*_SLO(PD-mux arm) / λ*_SLO(tuned fused)**, 같은 N·같은 Job, 절대 A′ SLO(TTFT ≤ k·floor_{N=1}(L) ∧ ITL p95 ≤ 50 ms), 조밀 사다리 직접 측정(M3 셀 사후 재채점 금지). GPU당 goodput·N-배율·λ*_thr G_sym은 메인 금지(F-1·처리율 지표).
- arm: tuned fused(사전등록 격자 chunk {2048, 4096, 8192} × mixed-chunk·overlap ON, N마다 별도 seed 선택; piecewise cudagraph OFF 문제 수리 또는 공시) · agnostic · N별로 같은 규칙으로 고른 PD-mux static D · (진단) fused_sym · HyMux는 H-D PRESENT 뒤에만.
- N: TP {1, 2, 4}(가능하면 8), Job마다 N=1 앵커, DP는 한 점 또는 생략.
- 모델: Nano + 순수 attention 대조(Llama-3.1-8B) 같은 N 격자. Falcon-H1-34B는 메인에서 제외(새 M2g·N=1 앵커 불가 가능성).
- 워크로드: shape B·C(+A 유지 또는 공시). n: seed ≥ 4, boot ≥ 2(ABBA), 짝 CI 0 배제 ∧ ≥ 3%. 전 arm 계측 OFF, GPU별 전력·클럭.
- 선행: VESSL Job 이벤트로 9.5 h 원인 조회, Job ≤ 6 h 설계.
- 비용 [유도, 손 비례 환산]: 모델 하나 TP {1, 2, 4} 약 35–56 GPU-h, TP ≤ 8 약 75–120; 두 모델 약 70–112(TP ≤ 4) 또는 150–240(TP ≤ 8).

## 5. 선택지
- (A) 단일 GPU 메인 + 실측 TP 이식성 절(venue-strategist 권고, MLSys·EuroSys/ATC 분석 논문 경로): J-C Nano TP2 비용 항 + §4의 TP {1, 2, 4} 축소판. 9B TP는 기전 프로브로만.
- (B) multi-GPU 메인(OSDI/SOSP/ASPLOS 시스템 경로): N=1 G0가 양의 헤드라인을 먼저 내야 하고, TP가 필요한 hybrid 모델(Falcon-H1-34B·Nemotron-H-47B/56B·Nemotron 3 Nano 30B-A3B — SGLang v0.5.10 동작 전부 미검증) + §4 전체.
- 원고 §7.8은 측정 전까지 삭제 또는 향후 과제.
