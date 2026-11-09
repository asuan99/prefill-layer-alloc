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
## 현재 상태 (2026-11-09)
- **이번 세션 결론**: 제안 기법 설계 이전에 motivation 증거 사슬(고리 1–4)이 서 있지 않다. 고리 2(hybrid 차이가 SM 분할 결정을 바꾼다)는 검증 0, 고리 3(baseline 대비 여지)은 KISTI 8월 결과뿐, 고리 4는 셀 정의가 판정을 막음. 이전 실험 목록·앞으로의 측정 D1–D3·M1–M6·워크플로우 관측 지점 후보 = **`reports/motivation_gap_2026-11-09/PRIOR_EXPERIMENTS_AND_PLAN.md`** · 공유 Doc "Motivation 측정 공백 점검"(claude.ai artifact 2wMCeLCGjjRN9WFJjWTXsC: 증거 사슬 그림, 예상 값 차트 4개)
- **"용량" 정의가 넷으로 섞여 있음**: 버스트 λ\*(120요청, 과도 구간 — 7.29가 포화 완료율 5.06–5.34보다 큼), 600 s 지속 안정성 규칙(ITL 없음), 포화 완료율, 모형 λ_cap → D2에서 정상상태 SLO 용량 하나로 정할 것을 권고
- **용어**: 프로젝트 고유 용어 정의를 정리함(대화 2026-11-09). 논문용으로는 문헌 용어로 매핑하고 c\*는 ReplaySSM 인용 처리 필요
- 감사 현황: Phase B 결과 r1 GO-with-edits(교정 `c945e428`, 9차 반영) · 미감사 문서 8건 r1 GO-with-edits(`996a5db7`, 대상 10개 배너) — 남은 정본 반영은 위 promote 줄
- 정책·impl 캠페인, D\*(결정 61), Phase A·B, stock_quick, 결정 43–64i·62: 8·9차 반영 완료. 원문은 PROJECT_STATUS 최상단과 git `tracks/main.md` 이력
- 엔진 수정 브랜치 `impl-defects-e123`(worktree `prefill-layer-alloc-implfix`) main 미병합 · 동결 설계 V-O·H-D(`cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md` rev4)
## 다음 행동
1. **(다음 세션 시작점)** `PRIOR_EXPERIMENTS_AND_PLAN.md` §4: 각 motivation 측정(D3·M1–M6)이 엔진 워크플로우(축 지도 §6: 부팅·입장·PD-mux iteration·forward·출력)의 어느 지점에서, 어떤 기존 계측(decode_probe·admission_probe·PHASE_EVENTS·EXTEND 플래그·nsys·클라이언트 시각)으로 관측되는지 코드로 확정. 새 probe가 필요한 칸을 표시
2. CPU 선행: M2a(철회 job 835571·858811 결함 원문, 873783 GATE N FAIL 정리) · M4a 본판(실제 길이 분포로 재탐색)
3. 사용자 결정 D1·D2·D3·M1 이미지·M3 대조 모델 뒤 PREREG 초안(감사 2회)
4. promote 10차(미감사 문서 감사 r1 허용 문안) — 대상 문서 본문 수정 여부와 함께
## 사용자 결정 대기
- **motivation 측정 계획** / 이번 세션 정리로 순서가 정해짐 / D1 baseline 정의 · D2 용량·지표·셀 정의(권고: 정상상태 SLO 용량 + SLO 감도표 + 정지 L 격자) · D3 cap·mem 범위 · M1 이미지(py3.12 대 shim) · M3 대조 모델(Llama-8B 대 a31) · M3가 Phase B 세 번째 Job (나)를 대체하는지(권고: 대체)
- **Phase B 세 번째·네 번째 Job** / 등록 셀 1.82·2.92 req/s는 포화 예보 / (가) 그대로 (나) 용량 근처 rev4 (다) 종결 — 위 M3로 흡수 권고
- **HE0 셀 부하 재정의** / 동적 arm 혼잡 가드 고정, 원칙 a–d(`policy_tune/MEMO_HE0_cell_load_redefinition_2026-10-30.md`; 감사 r1: 3.0 입력 REFUTED, 부하 위치 규약 반전 → 잠정 표기) / 결정 전 정책 판정 셀 Job 제출 금지
- **결정 13 mem 범위** / 0.6이 cap 상향을 막음 / (A) 장문만 0.6, cap 스윕은 mem 0.75 이상(권고, 감사 r1: 0.75는 평균 약 2k 이하에서만 cap 256 수용[유도·하한], 부팅 측정 필요) (B) 0.6 유지
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
- 2026-11-07~09(main): handoff 정리·promote 8·9차·Phase B 결과 감사 r1·미감사 8건 감사 r1·Mamba CUDA graph 코드 대조(SGLang·vLLM)·motivation 공백 정리 Doc + 예상 값 차트 4개·용량 정의 4종 정리·용어 정의
- 2026-11-07(main, batch 제약 세션): SGLang cap·슬롯 코드, ReplaySSM 대조, mem 0.6 출처 → 메모 1건
- 2026-10-30~11-02(D\* 세션): D\* 예측식·EVAL 사전 예측 봉인(결정 61), 축 지도 §8b·§6b
- 2026-10-27~11-07(참고 페이지 세션): 스케줄링 정책 개요 페이지 artifact H96kSDZVW7CnfmTKh175Tp(새 판정 0)
- 2026-11-02~04(main): Phase B 하네스 rev1→3.2, Job 4개 회수·분석, 정오 2건
