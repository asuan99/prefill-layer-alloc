# Motivation 실험: 지금까지와 앞으로 (2026-11-09, main 세션 handoff 산출물)

> 지위: 정리 문서다. 정본이 아니고 새 판정은 0이다. 수치와 판정은 정본(CONSENSUS·COSTMODEL)과 감사 판정서에서 옮겼다.
> 공유 문서: claude.ai Doc "Motivation 측정 공백 점검"(artifact 2wMCeLCGjjRN9WFJjWTXsC). 증거 사슬 그림, 고리별 현황, 측정별 예상 값과 차트 4개가 들어 있다.
> 예상 값 계산: 이 디렉터리의 `predict.py`·`predict.json`. `workspace/engine-port`에서 실행한다(CPU, dstar_model·analytic_terms 사용).
> **다음 세션 과제**: 아래 3절의 각 측정이 **엔진 워크플로우의 어느 지점에서 관측되는가**를 확정한다. 4절은 후보만 적었고 미확정이다.

## 1. 증거 사슬 (anchor 주장이 서려면)

- **고리 1.** hybrid의 비용 구조가 다르다. 식과 정적 사실만 확인됐다. 층 종류별 SM 차이는 무효다(철회).
- **고리 2.** 그 차이가 SM 분할 결정을 바꾼다. 검증 0.
- **고리 3.** baseline 대비 여지가 있고 hybrid에서 다르다. KISTI 8월 결과뿐이다.
- **고리 4.** 정책이 그 여지를 잡는다. 측정 셀 정의가 판정을 막고 있다.

## 2. 지금까지의 실험

| 실험 (시기·기판) | 무엇을 쟀나 | 쓸 수 있는 결론 | 관련 고리 |
|---|---|---|---|
| P1-opint·P1.7 (8월, KISTI A100) | PD-mux 기본 대 미조율 fused, Granite·Zamba2 | 꼬리 SLO goodput +27~+186%. 기전 미확립, 3-플래그 묶음 | 3 |
| layer-type 서빙 (KISTI, Zamba2, no-cudagraph) | 층 경계마다 SM 분할 전환 | 층 경계 SM 전환 계열 死(TPOT 42→124 ms). 시간 맞춤 span·박자·handoff·선점은 미시험 | 4 |
| HE0 (KISTI, Zamba2, 변화 trace) | SLO-aware·binding·gate 동적 대 best static | 동적 < best static(공개 루프 한정) | 4 |
| r0c 층별 decode knee 835571·858811, 본측정 873783 (KISTI) | attention·mamba 층별 decode SM 민감도 | **무효**. 수정 전 계측, full-SM arm이 늘 첫 순서, n=1로 철회. 873783은 GATE N FAIL·미감사 | 1 |
| C2 decode SM 민감도 (7–8B) | decode ITL의 SM16→92 변화 | 2.36–2.91×, 모델 무관(레버 존재만) | 2 |
| opchar (VESSL, Nano, 격리 커널) | 연산자별 green partition SM 스케일링 | 메모리 바운드 연산자는 44 SM에서 ≥86%, GEMM 계단 = wave quantization. 특성화 전용 | 1 |
| 128k 프로브 (VESSL) | 장문 서빙 한계, mem 0.82 대 0.6 | mem 0.6 도입(결정 13)의 근거. chunk −1 상한은 conv1d int32 | 1 |
| N2·N3·E2 (VESSL, 10월 초) | 동거 간섭, 용량 | 동거 감속 존재 확인(ρ, 전력 클럭 교락). 2026-10-15 종료 | 1, 4 |
| M3 J1 (multi-GPU) | N=1만 완주 | N>1 성능 수치 0, G0 뒤로 연기 | — |
| 비용 항 6 Job (10-26, VESSL, Nano) | T1 decode step, T2·T2′ prefill, T3 동거 ρ, T4 메모리, 조성 스윕, TP2 | 감사 r1 PLAUSIBLE 조건부. T1 형태, ρ 1.13–1.19 총효과, 조성 P1·P2·P4 | 1, 2 |
| lockstep 진단 (10-27) | span 시간 대 동거 decode 간격 | 적체 영역에서 별도 정지 항 미관측. 박자 영역은 미측정 | 4 |
| 정책 캠페인 CAP1-R·CAP2·TUNE (10-28~30) | 버스트 용량 λ\*, 동적 튜닝 | λ\*_W1 7.29(버스트), λ\*_W3 0.3375(여유 1요청), TUNE 선택 구성 전환 0(혼잡 가드) | 2, 4 |
| implfix 4 Job (11-01) | 엔진 수정 E1·E2·E3, D\* 사전 예측 채점 | 박자 셀 UNTESTED_REMAINS(하네스 결함), E1+E2 과부하 부호만, D\* E-1…5 HIT·E-6·7·8 MISS | 2, 4 |
| Phase B + 진단 + stock_quick (11-02~04) | SG 기본·vLLM·PD-mux 버스트·지속, SG ITL 정지 진단 | 감사 r1 A1–A11. SG(piecewise OFF) ITL 이봉·간극 약 64 ms, 용량 순서 주장 불가, P-1·P-2 MISS | 3 |
| CPU 작업 (11-07~09) | 미감사 문서 8건 감사, Mamba CUDA graph 코드 검토, 측정별 예상 값 | 감사 4건 GO-with-edits. 두 엔진 모두 decode Mamba는 graph·prefill Mamba는 eager. 예상 값은 아래 3절 | 전체 |

## 3. 앞으로의 실험 (권고 순서: D1–D3 → M1·M2 → M3·M4 → M6)

| ID | 무엇을 | 답하는 고리·질문 | 선행 | 예상 값 (사전 예측, 인용 금지) |
|---|---|---|---|---|
| D1 | baseline 정의 확정(SG piecewise·chunk·mixed, vLLM 기본, PD-mux 묶음 공시) | 고리 3 전제 | — | — |
| D2 | 측정 셀·지표 확정: 용량 정의 하나(권고: 정상상태 SLO 용량), ITL SLO 감도표, 정지 지표 L 격자, 부하 위치 규약 측정 셀 | 고리 3·4 | — | — |
| D3 | 승계 상수 재결정, cap 256 부팅 가능성(mem 0.70·0.75·0.80) | 고리 2·3 | — | cap 256은 mem 0.75에서 평균 문맥 약 2.1k까지(하한, 출력 예약 미반영) |
| M1 | upstream 기본 SG(piecewise ON)와 정지 원인 계측 | 고리 3 baseline | D1, 이미지 결정 | 간극 약 45–64 ms [가설]. Mamba 27층·attention은 eager로 남고 metadata 동기화도 남음 |
| M2a | 철회된 층별 decode job 결함 원문 확인 | M2 입력 | — | CPU |
| M2 | Nano 층 종류별 decode SM 스케일링 재측정(arm 순서 무작위, n≥4) | 고리 1 | M2a | 짧은 문맥에서는 두 곡선이 비슷, 30–44 SM 포화 [가설] |
| M3 | 같은 Job·n≥4, Nano와 attention 대조(Llama-8B 또는 a31) × PD-mux·SG·vLLM, 용량 근처 셀 | 고리 3 | D1, D2, M1 | Nano: 6.3–8.4 req/s에서 PD-mux 1.0→0.70, vLLM ≥0.97, SG 약 0.2(Phase B 사전값). W1에서는 hybrid 고유 여지가 작거나 반대 |
| M4a | hybrid 항 포함/제외 D\*가 갈리는 셀 탐색 | M4 입력 | — | **완료(예비)**: 합성 셀 15개 중 4개, 모두 입력 1k–4k. 실제 W1·W3는 같은 D |
| M4 | 그 셀에서 두 D를 같은 Job·n≥4로 비교 | 고리 2 | M4a, D2, D3 | D44 대 D34 셀 약 +5%. D24 대 D16의 +45~47%는 A6 경성 문턱 의존이라 과대 가능 |
| M5 | L\*·c\* 양쪽 셀(shape C 포함) goodput·구속 자원 | 고리 1→2 | D3 | 메모리 하한상 문맥 약 2.8k 이상에서 Nano decode step < Llama(배치 48) |
| M6 | 정책 여지(V-O, H-D) | 고리 4 | M3 여지 > 3%, D2 | 3% 이하일 가능성 높음(HE0 0.973, implfix 동적 = static) |

## 4. 워크플로우 관측 지점 — 다음 세션에서 확정할 것 (후보)

엔진 단계 이름은 축 지도 §6(코드 실행 흐름)을 따른다: 1 부팅, 2 수신·입장, 3 PD-mux iteration, 4 forward, 5 출력·해제. 관측 수단 후보는 기존 계측(decode_probe I-4·I-6, admission_probe I-5, PHASE_EVENTS, EXTEND 시간 플래그, nsys 커널 레인, 클라이언트 토큰 타임스탬프)이다.

| ID | 관측 지점 후보 | 관측 수단 후보 | 확인할 것 |
|---|---|---|---|
| D3 | 1 부팅(KV·mamba 풀 분할, `max_total_num_tokens`) | 부팅 배너, server_args | 배너 값으로 실현 한계를 정할 수 있는가 |
| M1 | 3 iteration(prefill 발행), 4 forward(Mamba metadata 동기화 `mamba2_metadata.py:201`), 클라이언트 간극 | EXTEND 시간 플래그, host 구간 타이머, 클라이언트 토큰 시각 | host 발행·동기화·launch를 나눌 계측이 이미 있는가, 새 probe가 필요한가 |
| M2 | 4 forward(층별 커널) | nsys `--cuda-graph-trace=node`, opchar green partition | decode graph 안에서 층 종류별 커널 시간을 분리할 수 있는가 |
| M3 | 2 입장(대기·거절), 3 iteration, 5 출력(클라이언트 TTFT·ITL) | admission_probe I-5, 클라이언트 지표, vLLM 쪽 동등 지표 | 엔진 간 같은 정의로 TTFT·ITL을 재는가(Phase B 인용 금지 3) |
| M4 | 3 iteration(분할 선택·체류), 5 출력 | telemetry split·dwell, 클라이언트 지표 | static 비sticky에서 D가 실제로 쓰인 시간 비율 |
| M5 | 1 부팅, 2 입장(KV 대 슬롯 거절 사유), 4 forward decode | admission_probe I-5, decode_probe I-4·I-6 | 구속 자원(슬롯 대 KV)을 요청 단위로 가를 수 있는가 |
| M6 | 3 iteration(컨트롤러 결정, 전환, drain) | switch_count, dwell, PHASE_EVENTS | 전환 drain 대기를 잴 수 있는가(현재 미측정) |

## 5. 열린 사용자 결정 (이 계획에 걸린 것만)

- M1 실행 이미지(py3.12 대 shim) · M3 attention 대조 모델(Llama-8B 대 a31) · M3이 Phase B 세 번째 Job (나)를 대체하는가 · D2(HE0 셀 원칙 a–d 포함) · D3(결정 13 범위) · M6 동결 해제 조건
