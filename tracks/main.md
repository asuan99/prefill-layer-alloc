# TRACK: main — hybrid 서빙 비용 모델 · 비용 항 측정 하네스 · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14: **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU + TP2 항). PD-mux 이득(G)은 검증 대상, "왜 hybrid"(H)는 조성 스윕(결정 24) + Llama 참조점. 수식 기반 = 정본 수식(결정 27). **결정 38: 측정 우선, 데이터 없이 설계·감사를 쌓지 않음.**
## cost model 기여
- 채우는 칸: CM-1(T1)·CM-2(T2·T2′)·CM-4(T3)·CM-5(T4)·CM-6·(확장) TP2·조성. 사전값 = analytic-v1(`reports/cost_model/ANALYTIC_COST_FORMULAS.md`, 계산 `scripts/costmodel/analytic_terms.py`)
- 비용 항 캠페인(10-26): `cap_char/RESULT_costterms_2026-10-26.md` §17(5차 반영) · Phase B(11-02~04): 감사 r1 허용 문안 A1–A11 `stock_baseline/AUDIT_result_phaseB_{claims,recompute}_r1_2026-11-07.md`(9차 반영)
- 이번 세션(11-08~09, CPU, 정본 아님): Mamba2 CUDA graph 코드 대조 `reports/engine_review/2026-11-08_mamba_cudagraph/`(두 엔진 모두 decode Mamba graph·padding 칸 커널 마스킹, prefill Mamba eager split op; SGLang만 Mamba prefill metadata GPU→CPU 동기화) · motivation 공백 정리와 측정별 예상 값 `reports/motivation_gap_2026-11-09/`(M4a 예비: 합성 15셀 중 4셀만 hybrid 항이 D\*를 바꿈, 입력 1k–4k; 메모리 하한상 문맥 약 2.8k 이상에서 Nano decode < Llama) — 사전 예측, 인용 금지
- promote 필요: 아니오 (8차·9차 반영 2026-11-07)
- promote 필요: 예 (감사 — 미감사 문서 r1 4건 GO-with-edits, `reports/audit/2026-11-08_unaudited_r1/`; 각 판정서 허용 문안만, 대상 문서 수정 전에는 인용 금지 목록 적용)
- promote 필요: 아니오 (이번 세션 코드 검토·예상 값은 공시 후보·사전 예측이라 감사 전 반영 안 함)
- promote 필요: 예 (결정 — 2026-11-09 비교 기준선, opchar = 고리 1 보조 근거, 작업 방식: 보고서는 md 금지·HTML, 측정은 번호 대신 보려는 동작 이름으로 부름 → 메모리 반영)
- promote 필요: 예 (감사 — 기준선·측정 셀 정의 시점 r1, 판정 요약 `reports/motivation_gap_2026-11-09/observation_map.html`, 새 판정 등급 없음)
## 현재 상태 (2026-11-09)
- **결론**: 제안 기법 설계 이전에 motivation 증거 사슬(고리 1–4)이 서 있지 않다. 고리 2는 검증 0, 고리 3은 KISTI 8월 결과뿐, 고리 4는 셀 정의가 판정을 막음. 이전 실험 목록 = `reports/motivation_gap_2026-11-09/PRIOR_EXPERIMENTS_AND_PLAN.md`(이력) · 공유 Doc artifact 2wMCeLCGjjRN9WFJjWTXsC
- **관측 지점 코드 확정 완료** = **`reports/motivation_gap_2026-11-09/observation_map.html`**(artifact FzF3JCzNx8KLQ9sfqk7HgM): 측정 7개의 동작 시나리오·계측 상태·옛 번호 대조표. 새 probe 필요 = 도착 시 decode 정지 원인, 층 종류별 decode 시간, 엔진 간 ITL 동등성, 요청별 KV·cap 사유 · 분할 체류 시간·전환 대기는 플래그로 가능
- **비교 기준선 결정(사용자, 2026-11-09)**: SGLang v0.5.10·vLLM 0.31.0 고정, SG chunk·SG mixed·vLLM 모두 기준선, 주 비교 대상 SG mixed. 감사 r1 GO-with-edits(PLAUSIBLE): 사유 "코드베이스가 SGLang을 따름"은 REFUTED(플래그 선택 근거 아님), SG-mix 단독 이득 표기 불가 · 감사 조건(부하점별 최선 기준선 대비, SG-chunked 병기, mixed 2048 추가, 이미지 프로브 선행, 근거 문장 교체) 전부 수용(사용자, 2026-11-09)
- **측정 셀·지표 시점**: 전체 연기는 감사 REFUTED, 항목별 연기 GO-with-edits. SLO는 전체 결과 수집 후 사후 계산(tight·loose 별도 측정 불필요, 사용자 지적). 측정 전 고정 = rate 격자·run 길이·도착 과정·cap·고정 D·SLO 평가 범위·읽기 규칙. 동적 컨트롤러는 게이트 8 예외 · HI 배율·수정 비교 방식은 정책 단계
- **opchar main 병합 완료**(`3fcd52c7`, 인용 정지 오탐 17건 커밋 메시지 공시, 사용자 결정 2026-11-09) · 워크로드 대조·LooGLE 실측·모델별 메모리 교차점 = `reports/motivation_gap_2026-11-09/workload_comparison.html`(artifact 2oUyLhRusZDddR5rbDNeyq)
- **워크로드·모델 범위 결정(사용자, 2026-11-09)**: 주 측정 = 사용자 첫 발화 ShareGPT · openchat_sharegpt4 · arXiv 요약(자연 출력) 단일 분포 셀 + 짧은 요청 rate를 고정하고 LooGLE 5%를 얹는 짝 비교 간섭 셀(부류별 보고, 합산 attainment 주 지표 금지). Mooncake 연기. 상위 용어 = "linear 층"(재귀 상태 층, 합성곱형 LFM2 제외), 주 줄기 Nano-9B-v2(Mamba2), 대조 = Qwen3.5-9B(Gated DeltaNet, L* 약 1.6k) — `reports/motivation_gap_2026-11-09/linear_models.html`(artifact 7PBTQnFiCSybRd85cYgUfB)
- **실험 순서·재반복 방지 결정(사용자, 2026-11-10)**: piecewise 부팅만 motivation 앞, nsys는 층별 decode 측정의 첫 단계, Qwen3.5-9B는 motivation 뒤(Nano 앵커 포함). 실행 전 자동 검사(구성 카드, 완주 리허설, 원자료 스키마) + PREREG 양식(결과별 다음 행동표, 격자 밖 규칙, arm 묶음 명시, 순서 무작위, 측정 계층까지만 결론)을 계측 단계에 포함
- **Phase B 세 번째·네 번째 Job 종결(사용자, 2026-11-10)**: 새 motivation 1단계로 대체. 무패치 대 패치 OFF 동등성 확인은 새 설계에 패치 OFF arm 1개로 흡수, SG 정지 비율 예측 검증은 도착 시 정지 측정이 대신
- **동시 실행 상한과 decode 메모리 트래픽 측정 추가(사용자, 2026-11-10)**: 1단계 앞. 고정 배치 = 문맥 2⁷–2¹⁸ 2배 간격, 배치 1–256(메모리 허용 범위, Nano 75셀·a31 60셀, 비분할 108 SM, chunked 채우기, mem 0.82) + 서빙 mem {0.6, 0.75}(cap 48)·cap {48,128,256}(mem 0.75). 결과의 cap·mem이 이후 측정 고정값. 결정 13 (A)/(B)는 이 측정으로 대체. 256k는 Nano 선언 최대 131,072 초과라 시간 측정 전용
- **attention 대조 모델 결정(사용자, 2026-11-10)**: a31(Nano 골격, 조성 변형)을 주 대조, Llama-8B를 외부 기준 한 점으로
- **"용량" 정의가 넷으로 섞여 있음**: 버스트 λ\*(120요청, 과도 구간 — 7.29가 포화 완료율 5.06–5.34보다 큼), 600 s 지속 안정성 규칙(ITL 없음), 포화 완료율, 모형 λ_cap → D2에서 정상상태 SLO 용량 하나로 정할 것을 권고
- **용어**: 프로젝트 고유 용어 정의를 정리함(대화 2026-11-09). 논문용으로는 문헌 용어로 매핑하고 c\*는 ReplaySSM 인용 처리 필요
- 감사 현황: Phase B 결과 r1 GO-with-edits(교정 `c945e428`, 9차 반영) · 미감사 문서 8건 r1 GO-with-edits(`996a5db7`, 대상 10개 배너) — 남은 정본 반영은 위 promote 줄
- 정책·impl 캠페인, D\*(결정 61), Phase A·B, stock_quick, 결정 43–64i·62: 8·9차 반영 완료. 원문은 PROJECT_STATUS 최상단과 git `tracks/main.md` 이력
- 엔진 수정 브랜치 `impl-defects-e123`(worktree `prefill-layer-alloc-implfix`) main 미병합 · 동결 설계 V-O·H-D(`cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md` rev4)
## 다음 행동
1. **(다음 시작점)** py3.12 SGLang 이미지(Job 밖 빌드, digest 고정) → piecewise ON Nemotron 부팅 + cap·mem 조합 + 문맥 256k 채우기 부팅 확인 프로브. 이어서 재반복 방지 장치(구성 카드 자동 대조, 완주 리허설, 원자료 스키마)와 새 계측 구현 → 사전 사다리 → motivation PREREG. 순서 전체 = `reports/motivation_gap_2026-11-09/roadmap.html`(artifact CUYuUscW3zYzh2SRf3Mkxz)
2. CPU 선행: 철회된 층별 decode job 결함 원문(835571·858811, 873783 GATE N FAIL) 정리 · hybrid 항이 D 선택을 바꾸는 셀 탐색 본판(실제 길이 분포)
3. 아래 결정 뒤 motivation PREREG 초안(감사 2회), 새 probe 구현
4. promote 10차(미감사 문서 감사 r1 허용 문안) — 대상 문서 본문 수정 여부와 함께
## 사용자 결정 대기
- **motivation Job 규모** / Nano·W1만 5 arm × 6 rung × 4 seed × 600 s = 부하 구간 20 h, W3·attention 대조는 별도 / 권고: W1에서 mixed 2048 제외(W3에서만), rung 6개 유지, run 300 s, seed별 Job 블록 = 8 h(블록당 2 h). 600 s 원자료 절단(n=1, 기술값): 용량 아래는 120 s로 충분, 경계 점은 300 s에서 통과·600 s에서 탈락 가능
- **비용 수식 linear 일반화** / FLOPs·decode 항은 Mamba2 전용, s·k 형태만 공통 / 수식 v2로 층별 상태 항 분리(코드·문서·버전 동시, 대조 모델 측정 뒤 권고) · 보류
- **Flex(2610.37432) 위치 확인** / Attention·SSM 서브배치에 SM 단위 분할, 프로젝트의 층 종류 SM 분할과 직접 인접 / venue-strategist 비교 검토(권고) · 보류
- **motivation 잔여 결정** / 실행 이미지(py3.12 권고 대 shim)
- **HE0 셀 부하 재정의** / 동적 arm 혼잡 가드 고정, 원칙 a–d(`policy_tune/MEMO_HE0_cell_load_redefinition_2026-10-30.md`) / 2026-11-09 감사: (a)는 주 용량 규칙과 같은 결정·(c)는 cap 결정·(b)(d)는 정책 단계. 결정 전 정책 판정 셀 Job 제출 금지
- **보고서 후속** / (a) HE0 그림의 SG 환산 막대·동적 예상치 표 삭제(권고) (b) 사후 기준 그림 감사 1회 (c) 정지 지표 PREREG 등록(감사 r1: 보조 지표로만, L 격자 사전 고정·n≥4·같은 Job) (d) 수식 v2(causal 계수·ReplaySSM 인용·2s 커널 종속)
- **통합 정책 측정 후속** / (A) 하네스 PC 창 수정 + 플래그 분리 재측정(권고) (B) 측정 중단 (C) EVAL 셀만
- **논문 방향** / ② + C 사다리(권고) · ② 단독 · ① 특성화(ISPASS/IISWC) — motivation 고리 2·3 결과에 달림
- **HyMux 원고** / 보류 중(구조 검증 감사 r1: 원고 대조로 확정된 미구현은 R4뿐) / 낮춤 · 구현 · 보류
- **multi-GPU 위치** / (A) 단일 GPU 메인 + TP 이식성·DP 라우팅 절(권고) (B) multi-GPU 메인
- **봉인 문서 정오·잔여 정리** / PROPOSAL §4 배너는 달았음 · `handoff-report/claude_web_review_prompt_2026-10-28.md` 커밋 여부 · `/tmp/gate_t_*` 11개 삭제(권고) · 구판 runtime manifest 갱신(권고) · presubmit 차단 2건 범위 한정 OVERRIDE(권고) · roofline η 수식 §6 한 줄(권고) · 진단 PREREG rev2 addendum(`7af52ffb`) 문턱 그대로(권고)
- LongBench v2·InfiniteBench 보류(결정 12′) · 기존 유지: D-10 · radix OFF · 장문 hold-out
## 실행 중·미회수 Job
- 없음(2026-11-09). Phase B 원장 `results/stock_baseline/vessl_jobs.tsv`·`stock_baseline_diag/vessl_jobs.tsv` 전부 회수·sha 검증. Workspace 3개 standby
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안 · gate 감사 r1/r2 · telemetry 비대칭 감사 §3 · PREREG §9-2·§1-9 · RESULT_costterms §17 · Phase B 판정서 인용 금지 · `reports/audit/2026-11-08_unaudited_r1/` 각 판정서 · "슬롯 상한이 hybrid 고유" · M3 판정서 · J1 크기 문장 · AJ-2 J-H1 값 · 이번 세션 예상 값(`predict.json`)은 측정 아님
## 최근 로그
- 2026-11-09(main): 관측 지점 코드 확정·관측 지도 HTML, 기준선 결정·감사 r1, SLO 사후 계산 정리, opchar 병합 시도(인용 정지로 중단)
- 2026-11-07~09(main): handoff 정리·promote 8·9차·Phase B 결과 감사 r1·미감사 8건 감사 r1·Mamba CUDA graph 코드 대조(SGLang·vLLM)·motivation 공백 정리 Doc + 예상 값 차트 4개·용량 정의 4종 정리·용어 정의
- 2026-11-07(main, batch 제약 세션): SGLang cap·슬롯 코드, ReplaySSM 대조, mem 0.6 출처 → 메모 1건
- 2026-10-30~11-02(D\* 세션): D\* 예측식·EVAL 사전 예측 봉인(결정 61), 축 지도 §8b·§6b
- 2026-10-27~11-07(참고 페이지 세션): 스케줄링 정책 개요 페이지 artifact H96kSDZVW7CnfmTKh175Tp(새 판정 0)
