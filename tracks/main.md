# TRACK: main — hybrid 서빙 비용 모델 · 비용 항 측정 하네스 · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14: **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU + TP2 항), PD-mux 이득(G) = 검증 대상, "왜 hybrid"(H) = 같은 골격 조성 스윕(결정 24) + Llama 참조점. 수식 기반 = 정본 수식(결정 27). M3는 G0 뒤. **결정 38: 측정 우선, 데이터 없이 설계·감사를 쌓지 않음.**
## cost model 기여
- 채우는 칸: CM-1(T1)·CM-2(T2·T2′)·CM-4(T3)·CM-5(T4)·(확장) TP2·조성. 사전값 = analytic-v1(`reports/cost_model/ANALYTIC_COST_FORMULAS.md`, 계산 `scripts/costmodel/analytic_terms.py`)
- 이번에 알게 된 것(2026-10-26 비용 항 캠페인, 감사 r1 PLAUSIBLE 조건부 → promote 5차 반영): T1 F1은 비분할 P=108만 채택·sticky-D는 F2/F3(두 Job 같은 형) · T2 Nano q NEAR_LINEAR, 장문 124k LONG_SERVED · T3 ρ 1.13–1.19는 동거·400 W 클럭·관측자 비용 총효과 · T4 cap 비구속 점 2개(HELD 1·MISS 1, ratio_A 0.68 예측 → 0.81) · 조성 P1·P2·P4 · TP2 F1·F2·F5 · AJ-1 `BWD_INV_D_MISS_LE64`·`BWD_NO_KNEE_TO_108`, AJ-3 θ′ LINEAR(합성 지수) — 세부·금지 목록 `workspace/engine-port/results/cap_char/RESULT_costterms_2026-10-26.md` §17
- promote 필요: **예 (결정·감사·결과 — 결정 43–58·60·61·59–64i, 정책 캠페인 결과·감사 r1 4건)** · 이전: **아니오 (6차 반영 2026-10-27)** — 결정 39–42·lockstep 진단 감사 r1(P1–P5)·변화 trace 감사 r1·C 감사 r1은 COSTMODEL CM-4, CONSENSUS §1-3/§1-7/§9, PROJECT_STATUS 최상단에 반영(§6 등급 변경 0, 공개 trace 통계 수치 미반영). **감사 필요(미반영)**: ① 구조 검증 보고서 `reports/hymux_review/STRUCTURE_batching_admission_2026-10-27.md`(`7a7a2c08`, 미감사): F1–F13 CONFIRMED(F6 정정: 매 반복 동기화는 decode stream뿐, prefill은 `query()`; F7 index 수는 config 의존; F12 예외 `--enable-prefill-delayer` 미사용) · 원고 미구현 서술 R4·R5·R6·R11, 부분 R7(D_floor 닫힌 식 없음, P=108−D 설정 항등식)·R9 · C6 손익분기 8.8k는 오류 → s/(k_T−k) ≈1,251 토큰 [derived] ② 학회 검토 `reports/hymux_review/VENUE_C_positioning_2026-10-27.md`의 venue_positioning 정정 후보 2건
## 현재 상태 (2026-11-01, 두 세션 공유 — 정책 캠페인 줄 + impl 세션 줄)
- **비용 항 캠페인·lockstep 진단·C 감사**: promote 5·6차 반영 완료(`cap_char/RESULT_costterms_2026-10-26.md`, `anchor_audit/VERDICT_lockstep_diag_claims_r1_2026-10-27.md`, `…/VERDICT_claimC_online_predictive_2026-10-27.md`). 결정 39–42는 6차 반영
- **정책 캠페인(결정 43–58, 60; 원문 = 아래 결과·감사 파일과 git `tracks/main.md` 10-28~30)**: Zamba 배제(46) · static 비sticky(49) · 진단 PREREG 판별량 A 기술값 강등(50) · gate+smoke: 진단 GPU gate D0–D9 PASS, smoke v2 재채점 PASS(`policy_gate_smoke/`) · switch_count = 컨트롤러 결정 변경(53), 양성 대조 PASS · raw 반출 수리(54, m12 원자료 `results/raw_export_20261028/`)
- **CAP1-R** λ*_W1 = 7.29491(버스트, D44·D64 동률) · **CAP2** λ*_W3 = 0.3375(D16, 통과 여유 1요청 — 취약) · **지속성** W1 7.29는 600 s 비지속(★정오 2026-11-02: 완료율 ≈3.0은 `audit_r1_sus.py` 단위 오류, 올바른 포화 완료율 5.06–5.34 — `policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`) · **TUNE** 선택 dyn_a4_w3, FEAS 혼잡 가드로 전환 0(D44 고정) — 결과·감사 r1: `policy_cap1r/`, `policy_cap2/`, `policy_tune/`(각 AUDIT_*_r1, 전부 GO-with-caveats 또는 EVAL-as-HE0 NO-GO). H-D 보류(57), EVAL HE0 라벨 없음(58) → **EVAL 보류·implfix 짝 Job으로 흡수(60)** — impl 세션이 결정 62i로 통합 PREREG에 합침
- 검토 메모(미감사): `policy_tune/MEMO_implfix_impact_on_policy_campaign_2026-10-30.md`(W1 multi-chunk 5.5–9.4%, B1/E3 영향≈0, E1 편차 가능성 최대, W3는 E2 영향 큼), `policy_tune/MEMO_HE0_cell_load_redefinition_2026-10-30.md`(셀 고정 원인은 루프 아님, λ* 루프 의존, 원칙 a–d)
- **결정 61(2026-10-30, 사용자 "진행")**: D\* 예측식 제안 + EVAL 사전 예측 봉인 — `reports/cost_model/PROPOSAL_dstar_model_2026-10-30.md`, 코드 `scripts/costmodel/dstar_model.py`, 봉인 커밋 `113de504`(EVAL 미제출 시점). 사후 재현(CAP 결과 본 뒤 작성): 구속 제약 W1 5/5·W3 4/5, W1 순서 일치, W3 상위 순서 miss(TTFT 항 낙관), W1 크기는 버스트 대 정상상태라 비교 불가. 예측 E-1…E-8(EVAL static 5 arm) — promote 필요: 예 (결정)
- **결정 62(2026-11-01, 사용자 "진행") + 기본 구조 기준선 Phase A(CPU, 기술값)**: 계획 `results/stock_baseline/PLAN_stock_baseline_metrics_2026-11-01.md` §6 권고 채택(vLLM = 제출 시점 최신 stable · 조율 fused·Llama-8B는 2차 · `decode_probe` EXTEND 시간 플래그 기본 OFF 추가) · r 대응량 r_ser/r_mix로 따로 명명(r_span과 합치지 않음). 결과 `stock_baseline/RESULT_stock_baseline_phaseA_2026-11-01.md`: P1-opint 16셀 원문 재현 일치 · M3 J1 fused 행(N=1) · KISTI Granite stock r_ser p50 ≈ 15(in2000, HOLB 관측자 효과 미결정) · Nano r_ser 예보(봉인 전 인용 금지) · 게이트 4 사각(out 64에서 요청당 정지 ≤3회 미검출). 감사 r1 GO-with-edits 7건 반영, result-analyst 재계산 3건 일치 — promote 필요: 예 (결정·결과·감사)
- 방향 재정리(제안, 미결정): 승계 기본값 D=44·cap 48·budget 65536은 Nano에서 재결정 · HyMux 원고는 구현 구조로 낮추고 R1–R3 단일 처치 ablation 가설로 · 비동기 기판 arm 후순위
- 동결 설계: V-O·H-D `cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md` rev4 · DP 라우팅(감사 r2). 사소한 결함: `ct_harness_digest.py check --dir` 상대 경로
- **구현 결함 작업(impl 세션, 2026-10-28~11-01; 안내 `reports/impl_defects/INDEX.md`, 엔진·측정은 브랜치 `impl-defects-e123` = worktree `prefill-layer-alloc-implfix`, main 미병합)**:
  - Phase 0 CPU(`results/impl_defects/PHASE0_CPU_CHECK_2026-10-28.md`) → 루프 의존 범위 감사 r1·r2 → **7차 promote(부분, 2026-10-29) 반영 완료**("layer-type 全형태 死" → 층 경계 SM 전환 계열 死, 시간 맞춤 span·박자·handoff·선점 미시험)
  - 엔진 수정 E1 span budget|time|time_typed · E2 발행 깊이 · E3 handoff · E5 계측(기본 OFF, digest `0a5d9c3c`) · GPU 정확성 gate rev0.1 FAIL(하네스 원인) → rev0.2 UNRESOLVED(E3는 문언상 UNRESOLVED, nocache 경로 미실행; 결과 감사 r1)
  - 통합 정책 측정(EVAL 흡수, PREREG rev1 봉인 `736ec49c`, 4 Job 회수) 결과 `implfix:…/policy_implfix/RESULT_policy_implfix_2026-11-01.md` + 결과 감사 r1(`96793350`): 박자 셀 층 종류 반영 대 무시 **UNTESTED_REMAINS**(+4.4% [−0.6,+9.5]; PC 축약이 warmup·프로브를 섞은 **하네스 결함**) · EVAL 셀 E1+E2 묶음 **ON_BELOW_REF 부호만**(과부하 HI 구간만, LO 동일; 크기·일반화 금지) · 동적 OFF=static(미전환 null) · D\* E-1…5 HIT, E-6·7 MISS, E-8 점추정 MISS · handoff 대 drain ±3% 안(물리 handoff 미관측) · CM-2 γ 1.3–1.7 주장 철회. 그림 8개 `implfix:…/policy_implfix/figures/`
  - 결정(이 세션): 57·58(구 54)·59·60i·61i·62i·63i·64i — 원문은 git `tracks/main.md` 10-29~31. ※ 다른 세션과 번호 충돌(58·60·61) → impl 쪽은 58·접미사 i로 구분, promote 때 재번호 필요 — promote 필요: 예 (결정·감사·결과)
- **설계 결함·반복 패턴 정리(2026-11-01, 사용자 요청 "진행")**: `reports/audit/2026-11-01_design_defects/REPORT.md` — 결함 S1–S12 심각도 순, 반복 패턴 P1–P6 원인(계획/셋업) 분류. 에이전트 규약 개정 적용: experiment-runner(부하 위치·승계 상수·처치 발생 횟수·PREREG 필수 항), result-analyst(게이트 4 차단형·"측정하지 않은 것" 절·cost model 2칸), claims-auditor(confound 12·13·파이프라인 모집단 층·`GO-with-caveats(검출력)`·caveat 상한). 미감사 — promote 필요: 예 (결정·규약)
- **Phase B 착수(2026-11-01 오후)**: PREREG STOCK-BASE rev0 DRAFT 작성(새 규약 첫 적용: 질문 목록 Q1–Q7·검출력 사전 계산·축약 필터 표·승계 상수 공시) · stock 대 PD-mux 그림 2장 `reports/figures/stock_vs_pdmux_2026-11-01/`(N=1 기술값·측정 지도) · vLLM 최신 stable 0.31.0 확인 · engine-porter EXTEND 플래그 작업 중 — promote 필요: 예 (결정)
## 다음 행동
1. (완료 2026-11-01) 통합 정책 측정 4 Job 회수·분석·결과 감사 r1 — 다음은 아래 결정 대기 "통합 정책 측정 후속"과 promote 7차(나머지)
2. HE0 셀 부하 재정의 원칙(a)–(d) 사용자 결정 → 수정 루프 용량 재측정 → OFF·ON 각각 TUNE → 같은 Job 짝 EVAL
3. promote 7차 후보 정리: 결정 43–58·60·60i–64i, 정책 캠페인 결과·감사 r1 4건, 구조 검증(`7a7a2c08`, 감사 필요), 두 메모(감사 필요)
## 사용자 결정 대기
- **HE0 셀 부하 재정의** / 현 셀(HI = 1.5×버스트 λ*_W1)에서 동적 arm이 FEAS 혼잡 가드로 고정(TUNE 4/4) → HE0 판정 불가, 이 원인은 루프 수정으로 안 풀림 / 검토 메모 `policy_tune/MEMO_HE0_cell_load_redefinition_2026-10-30.md` · ★2026-11-01 규약 개정(REPORT 위)이 이 결정에 걸려 있음 — 결정 전까지 정책 판정 셀 Job 제출 금지(승계 상수 규약)
- **HyMux 원고** / 2026-10-27 사용자: **보류(원고 유지)**. 구조 검증 `7a7a2c08`: R4·R5·R6·R11 미구현, R7·R9 부분 — 원고 인용 시 이 대조표가 차단 근거 / (a) 구현 구조로 낮춤 · (b) 구현·측정 · 보류 지속
- **multi-GPU의 논문 내 위치** / 메인 지표 근거 0 / (A) 단일 GPU 메인 + 실측 TP 이식성·DP 라우팅 절(권고) · (B) multi-GPU 메인(G0 양의 결과 + 대형 hybrid 선행) · DP 라우팅 열린 결정 O-D1…O-D5
- **roofline η 정본 반영** / 집계 η≈0.49는 층별 η 아님 / 수식 문서 §6 "실측 대조"에 한 줄만(권고) · 반영 안 함
- **presubmit reachability 차단 2건** / M4R·TC1 rev3 spec 전역 차단(측정 무관) / 범위 한정 OVERRIDE(권고) · 두 spec 폐기
- **로컬 엔진 트리 재동기화** / 2026-10-27 확인: `sglang_engine_dev`는 VESSL 10-20 서빙 manifest(`6015796d…`, 27항목)와 sha256 전부 일치, 차이는 기본 OFF probe 파일·훅뿐. 저장소 untracked `results/runtime_source_manifest.sha256`이 구판(5건 FAILED) / 구판 manifest 갱신·로컬 테스트 실패 원인 재확인(권고) · 유지
- **임시 디렉터리 `/tmp/gate_t_*` 11개** / 테스트 잔여물 / 삭제(권고)
- **논문 방향** / C 감사 r1·학회 검토(2026-10-27): A+B만이면 생태계 대비 약함, C 신규성은 hybrid 귀속뿐 / **② + C 사다리(권고)**: H-D → V-O → C 게이트(H-D 여지 >3% 또는 V-O phase별 oracle 분기, 첫 마일스톤 = 순수 attention에서 예측형이 static을 이기는 재현; 최소 조건 = P_gen+recal·DuetServe판 baseline·hold-out·분할 T_dec F1·δ·CM-6 목적항·iteration 배선) · ② 단독 · ① 특성화(ISPASS/IISWC)
- **진단 PREREG rev2 addendum**(`7af52ffb`, 미봉인) / 관문 문턱 3개·D-4 크기·M1 배치·하네스 감사 1회 / 문턱 그대로·D-4 boot 8·M1은 D-1a 합침·감사 1회(권고) · 조정
- **통합 정책 측정 후속** / 4 Job 결과·감사 r1(아래) — 다음 측정은 새 PREREG 필요 / (A) 하네스 결함(PC 축약 부하 창) 수정 + 박자 셀 검출력 확보 + EVAL 셀 플래그 분리(E1만·E2만·E1+E2, 용량 아래 셀) 재측정(권고) · (B) 결과 promote만 하고 측정 중단 · (C) EVAL 셀만 재측정
- **기준선 Phase B**(2026-11-02): 봉인 전 결정 전부 확정("권고대로 진행": 변화 trace 제거·vLLM 같은 이미지 venv·계획 대체 6건·시간 기준 확인 2단계·용어 "기준 용량"/"부하 창"). PREREG **rev1** `stock_baseline/PREREG_STOCK_BASE_rev1_DRAFT_2026-11-02.md`. 이미지 `sglang-vllm-runtime@a5f9bf0f…` Public 확인. 하네스 rev1 반영(engine-porter 진행 중) → 하네스 감사 1회 → 봉인 → bring-up(A100 용량 대기) → dry-run. ★정오·진단(부록 D)은 그대로 — promote 필요: 예 (결정·정오·진단)
- LongBench v2·InfiniteBench 보류(결정 12′) · 기존 유지: D-10 · radix OFF · 장문 hold-out · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 회수 완료(main): gate+smoke · CAP1 · CAP1-R · CAP2+지속성 · TUNE(job-9t469viisz2i) — 원장 각 `results/policy_*/vessl_jobs.tsv`
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안 · gate 감사 r1/r2 인용 금지 · telemetry 비대칭 감사 §3 · PREREG §9-2·§1-9 금지 · RESULT_costterms §17 promote 금지 목록 · "슬롯 상한이 hybrid 고유" · M3 판정서 문안 · J1 크기 문장 · AJ-2 J-H1 값
## 최근 로그
- 2026-11-01(오후): 그림 8개 설명 · 설계 결함 S1–S12·패턴 P1–P6 정리 · 에이전트 규약 3파일 개정(`reports/audit/2026-11-01_design_defects/`)
- 2026-10-28~11-01(impl 세션): 구현 결함 Phase 0·범위 감사·7차 promote(부분) · 엔진 수정 E1–E3·E5(브랜치) · GPU 정확성 gate 2회 · 통합 정책 측정 4 Job·결과 감사 r1 · 축 지도 §6b(시스템 동작·r 정의)
- 2026-10-28~30(정책 캠페인 세션): 진단·정책 PREREG(감사 r1/r2), gate+smoke·CAP1·CAP1-R·CAP2+지속성·TUNE 6 Job 측정·회수·결과 감사, 결정 43–58·60, raw 반출, implfix 영향·HE0 셀 재정의 메모
- 2026-10-27 오후: catch-up · 외부 비판 문서 대조 · C 감사 r1·학회 검토 · HyMux 구조 대조 · lockstep 정지 진단(CPU) · 방향 재정리(제안)
- 2026-10-26~27: 비용 항 6 Job 측정·회수·분석·결과 감사 r1·promote 5차 · 결정 31–38(측정 우선) · HyMux 원고·계획 검수 · multi-GPU·변화 trace 검토 · DP 라우팅 설계(감사 2회)
- 2026-10-25: roofline·스케일링 그림 묶음 + prefill 관측 대 이상 대조(감사 r1) + T1 비교 틀
