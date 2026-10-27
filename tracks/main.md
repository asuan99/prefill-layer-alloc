# TRACK: main — hybrid 서빙 비용 모델 · 비용 항 측정 하네스 · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14: **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU + TP2 항), PD-mux 이득(G) = 검증 대상, "왜 hybrid"(H) = 같은 골격 조성 스윕(결정 24) + Llama 참조점. 수식 기반 = 정본 수식(결정 27). M3는 G0 뒤. **결정 38: 측정 우선, 데이터 없이 설계·감사를 쌓지 않음.**
## cost model 기여
- 채우는 칸: CM-1(T1)·CM-2(T2·T2′)·CM-4(T3)·CM-5(T4)·(확장) TP2·조성. 사전값 = analytic-v1(`reports/cost_model/ANALYTIC_COST_FORMULAS.md`, 계산 `scripts/costmodel/analytic_terms.py`)
- 이번에 알게 된 것(2026-10-26 비용 항 캠페인, 감사 r1 PLAUSIBLE 조건부 → promote 5차 반영): T1 F1은 비분할 P=108만 채택·sticky-D는 F2/F3(두 Job 같은 형) · T2 Nano q NEAR_LINEAR, 장문 124k LONG_SERVED · T3 ρ 1.13–1.19는 동거·400 W 클럭·관측자 비용 총효과 · T4 cap 비구속 점 2개(HELD 1·MISS 1, ratio_A 0.68 예측 → 0.81) · 조성 P1·P2·P4 · TP2 F1·F2·F5 · AJ-1 `BWD_INV_D_MISS_LE64`·`BWD_NO_KNEE_TO_108`, AJ-3 θ′ LINEAR(합성 지수) — 세부·금지 목록 `workspace/engine-port/results/cap_char/RESULT_costterms_2026-10-26.md` §17
- promote 필요: **예 (감사)** — C 미주장 감사 r1 `anchor_audit/VERDICT_claimC_online_predictive_2026-10-27.md`(+ 학회 검토 `reports/hymux_review/VENUE_C_positioning_2026-10-27.md`, venue_positioning 정정 후보 2건) · 그 외 5차 `dc7193f4` 반영 완료 · 예외: HE0 변화 trace 대표성(`reports/hymux_review/CHANGE_TRACE_VALIDITY_2026-10-26.md`)은 claims-auditor 판정 없음 → 감사 뒤 `예 (감사)`
## 현재 상태 (2026-10-27)
- **비용 항 측정 캠페인 완료**: 봉인 S1′(`af28dfe8`, PREREG rev3.11) 하네스로 6 Job(J-A1·J-A2·J-H1·J-H2·J-B2·J-C) 전부 succeeded·회수(DONE + sha256), 46 boot usable·flag 0. 요약 `cap_char/costterms_runs/`, boot 원자료는 클러스터 볼륨 `/data/costterms/*`에만(m12). 결과 `RESULT_costterms_2026-10-26.md` rev1 + 감사 `AUDIT_costterms_result_r1_2026-10-27.md` + AJ `costterms_analysis/AJ_full_2026-10-26.json`(aj_analysis v1.1)
- 실행 중 사고 2건: J-A1 1차가 `job_entry.sh`의 results symlink 아래 `resolve()` 경로 버그로 rc 13(측정 0) → 논리 경로 수리·S1′ 재봉인·`tests/test_costterms_symlink_layout.py` · AJ-2 J-H1 NA_FORM 게이트 누락 → v1.1 수리(해당 값 인용 금지)
- 그 전 경과(2026-10-25~26, 상세는 git·결정 원문 31–38 `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`): roofline 묶음 감사 r1(균일 η 해석 철회) · HyMux 원고 v4·검증 계획 문서 검수(Claude Docs 댓글, `reports/hymux_review/EXPERIMENT_CROSSWALK_2026-10-26.md`) · PREREG §1-9 AJ 판정 추가·규칙층 감사 2회 · 하네스층 감사 2회·HARNESS_DIGEST v2 · Workspace smoke(`cap_char/costterms_smoke_20261026T054338Z/`) · multi-GPU 증거 검토(`MULTIGPU_EVIDENCE_REVIEW_2026-10-26.md`) · 변화 trace 타당성 검토 · 결정 38(측정 우선)
- **동결 중 설계(결정 38, 비용 항 데이터가 생긴 지금 재개 여부 판단 대상)**: V-O·H-D `cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md`(rev4, CI 결정 32·축소 결정 33) · DP 라우팅 `dp_routing/PLAN_DP_routing_per_gpu_partition_2026-10-26.md`(감사 r2 GO-with-caveats, PREREG 작성 조건 N-1…N-6, 결정 36·37)
- 알려진 사소한 결함: `ct_harness_digest.py check --dir`에 상대 경로를 주면 spec/plans sha ERROR:2 — Job은 절대 경로라 무관, 다음 봉인 때 수리
- **C(비용 모델 구동 iteration 단위 예측형 SM 배분) 감사 r1(2026-10-27)**: "C 미주장" CONFIRMED이나 상태는 **미검증·시험 가능**("기각됨" REFUTED, HE0를 C 반대 근거로 쓴 것은 과잉). 예측형 중 offline 프로파일 최소-SM(R2 hybrid)은 서빙 측정 0건, LFF(n=3)·sat_predict는 시험됨. 컨트롤러 무관 위험: Zamba2 HE0에서 static이 decode SM에 단조 증가 → Bullet형 최소-SM 목적함수가 손해 방향(n 작음·Nano 미측정). 학회 검토: C 형태는 DuetServe(ICML'26)가 transformer에서 수행 → 신규성은 hybrid 귀속(hold-out)뿐, A+B만이면 top venue 약·ISPASS/IISWC 중~강
- 정본 반영 상태: promote 5차(2026-10-27) — COSTMODEL CM-1·2·4·5·(확장), PROJECT_STATUS 4줄, CONSENSUS §1-22·§8 정정 각주, MEMORY `measurement-first-no-design-pileup.md`, 축 지도 v11 재발행
## 다음 행동
1. **비용 모델 조립 여부 결정** → V(-O) PREREG: 측정 계수로 예측기를 세워 static 설정의 oracle 격차를 재는 것이 논문 방향 ②의 다음 고리(PLAN_V 재개)
2. G0 PREREG(tuned fused·shape C·순수 attention 대조, 전 arm 계측 OFF) — V-O·DP 라우팅의 분모·a 승계
3. (선택) 원자료 필요 항목: T2·T2′·T4 `CLOCK_SENSITIVE`, T4 MISS 원인 — 클러스터 볼륨 원자료 회수(m12)
## 사용자 결정 대기
- **다음 측정 경로** / 결정 38 동결 해제 조건 충족, C 게이트도 H-D 결과에 걸림 / V-O+H-D 한 Job PREREG 착수(권고, H-D 우선) · G0 먼저 · DP 라우팅 먼저
- **multi-GPU의 논문 내 위치** / 메인 지표 근거 0 / (A) 단일 GPU 메인 + 실측 TP 이식성·DP 라우팅 절(권고) · (B) multi-GPU 메인(G0 양의 결과 + 대형 hybrid 선행) · DP 라우팅 열린 결정 O-D1…O-D5
- **변화 trace 보강** / HE0 일반화가 실 trace 시간 척도와 다름(미감사 검토) / claims-auditor 감사 → 실 trace 재생 1종·정상 상태 격자 논증(권고) · 현행 유지
- **m12 원자료 보관** / boot 원자료가 클러스터 볼륨에만 있어 CLOCK_SENSITIVE·T4 MISS 재분석 불가 / 회수·보관 단계 코드화(권고) · 볼륨 유지만
- **roofline η 정본 반영** / 집계 η≈0.49는 층별 η 아님 / 수식 문서 §6 "실측 대조"에 한 줄만(권고) · 반영 안 함
- **presubmit reachability 차단 2건** / M4R·TC1 rev3 spec 전역 차단(측정 무관) / 범위 한정 OVERRIDE(권고) · 두 spec 폐기
- **로컬 엔진 트리 재동기화** / 구판이라 로컬 테스트 일부 실패 / amd·opchar 영향 확인 뒤 `install_runtime.sh`(권고) · 유지
- **임시 디렉터리 `/tmp/gate_t_*` 11개** / 테스트 잔여물 / 삭제(권고)
- **논문 방향** / C 감사 r1·학회 검토(2026-10-27): A+B만이면 생태계 대비 약함, C 신규성은 hybrid 귀속뿐 / **② + C 사다리(권고)**: H-D → V-O → C 게이트(H-D 여지 >3% 또는 V-O phase별 oracle 분기, 첫 마일스톤 = 순수 attention에서 예측형이 static을 이기는 재현; 최소 조건 = P_gen+recal·DuetServe판 baseline·hold-out·분할 T_dec F1·δ·CM-6 목적항·iteration 배선) · ② 단독 · ① 특성화(ISPASS/IISWC)
- LongBench v2·InfiniteBench 보류(결정 12′) · 기존 유지: D-10 · radix OFF · 장문 hold-out · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음. Workspace 둘 다 standby.
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안 · gate 감사 r1/r2 인용 금지 · telemetry 비대칭 감사 §3 · PREREG §9-2·§1-9 금지 · RESULT_costterms §17 promote 금지 목록 · "슬롯 상한이 hybrid 고유" · M3 판정서 문안 · J1 크기 문장 · AJ-2 J-H1 값
## 최근 로그
- 2026-10-26~27: 비용 항 6 Job 측정·회수·분석·결과 감사 r1·promote 5차 · 결정 31–38(측정 우선) · HyMux 원고·계획 검수 · multi-GPU·변화 trace 검토 · DP 라우팅 설계(감사 2회)
- 2026-10-25: roofline·스케일링 그림 묶음 + prefill 관측 대 이상 대조(감사 r1) + T1 비교 틀
- 2026-10-22~23: 정본 수식 고정(결정 27)·축 지도 §6·§8 · 하네스 1·2단계, rev3.4·3.5(결정 28–30) · 조성 프로브 PASS
- 2026-10-21: 결정 22–26, engsync 병합, telemetry 감사 r1, multigpu·engsync worktree 폐기, promote 4차
- 2026-10-20: 결정 14(비용 모델 주 목적) · PREREG rev0–rev3 · 엔진 묶음 gate α·β PASS · 128k 프로브
