# TRACK: main — hybrid 서빙 비용 모델 · 비용 항 측정 하네스 · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14: **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU + TP2 항), PD-mux 이득(G) = 검증 대상, "왜 hybrid"(H) = 같은 골격 조성 스윕(결정 24, dummy 가중치 a4·a13·a22·a31) + Llama 참조점. 수식 기반 = 정본 수식(결정 27). M3는 G0 뒤.
## cost model 기여
- 채우는 칸: CM-1(T1 `T_dec(D,bs,ctx)`), CM-2(T2·T2′ `T_span`), CM-4(T3 ρ·σ′), CM-5(T4), (확장) TP2·조성 항. 사전값 = `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1, 계산 `scripts/costmodel/analytic_terms.py`)
- 이번에 알게 된 것: 정본 수식 = 부팅 배너(파라미터 8,888,227,328·s 143.5 MB·k 16,384 B)와 일치, J1 floor 기울기 = 실효 약 141 TFLOP/s(피크 약 45%) [유도] · decode는 c* ≈ 17.5k 아래에서 SSM state 트래픽이 KV보다 큼 [유도] · 조성 프로브 전 단계 PASS(KV 풀 887k→298k 토큰) · 코드 사실: decode가 비면 분할 (108,0)
- 2026-10-25 roofline 분석(수식 [유도] + J1 FLOOR.json 대조, 새 측정 0) **감사 r1 완료** `reports/figures/roofline/AUDIT_r1_2026-10-25.md`: 관측 floor = 이상 하한의 2.06–2.28배는 CONFIRMED(산술)이나 **"균일 η≈0.47–0.49" 해석 REFUTED**(격리 연산자 η: GEMM≈0.8·SSD scan≈0.05·gated_norm≈0.08, opchar 인용 금지 범위·반증 용도만) · 절편 7.2 ms는 추정량 의존(7–18 ms, 격리 합에도 ≈15 ms) → 고정 비용 아님 · decode 서술은 c·층 조건부로 정정(MLP 층 b≳150 ridge 위) · 128k 프로브 점은 내부 점검 절로 이동(±10%→−8~+11%) · **§8 η 첫 관측점 승격 = 감사 권고 불가**(허용 최대: CONSENSUS 아닌 수식 §6 실측 대조에 집계 ≈0.49 한 줄)
- promote 필요: 예 (결정) — 결정 24–30 · 예 — 정본 정정(CONSENSUS §1-22 (108,0) 방향) · 예 (감사 판정) — roofline r1: 균일 η 해석 철회, §8 승격 불가(사용자 결정 대기) · 예 (결정) — 결정 31 · 예 (감사 판정) — rev3.6′ GO-with-caveats · 예 (결정) — 결정 32 · 예 (결정) — 결정 33 · 예 (감사 판정) — 하네스 r2 봉인 불가(N-1) · 예 (결정) — 결정 34 · 예 (감사 판정) — multi-GPU 인용 가능 문장·telemetry 비대칭 정정 · 예 (결정) — 결정 35 · 예 (결정) — 결정 36·37, DP 라우팅 감사 r1·r2 · 예 (결과 분석) — HE0 변화 trace 대표성 범위 · 예 (결정·작업 방식 피드백) — 결정 38
## 현재 상태 (2026-10-23)
- **비용 항 PREREG rev3.5 확정·미봉인** `workspace/engine-port/results/cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md` + **하네스 1·2단계 구현 완료** `results/cap_char/costterms_harness/`(최신 `e141a36e`: CPU 시험 50/50, 변이 44/44, ENGINE_DIGEST 검사 rc 13, smoke 모드, 원자료 클러스터 볼륨, 셀 단위 재개). Job: J-A1·J-A2·J-H1·J-H2·J-B2·J-C, 예약 약 19.4 GPU-h [유도].
  - 결정 28: T1 창 = ramp 인식(전원 합류 후 300 step·취소·실제 평균 ctx 적합), T2는 P=108만, P 축은 T2′. 결정 29: shape C 셀 손실은 smoke 후 ramp 상한 재설정. 결정 30: 장문 측정점(N=1 98k·124k, TP2 96k·124k) 별도 맨 끝 boot로 복구 + **측정점 삭제 원칙**(물리적 불가·더 강한 증거만 삭제). §11-4 삭제점 전수표.
  - launch 연결 확인(`48dd20ed`, libnuma는 측정 이미지에 이미 있음), smoke 계획 `results/cap_char/SMOKE_PLAN_costterms_2026-10-22.md`(3 boot, 상한 90분).
  - 봉인 체크리스트 §12 진행 중. 남은 핵심: 미복구 설정 제약 점 결정 → 하네스층 감사 → smoke → ramp 상한·J-B2 하드캡 재설정 → 봉인 → preflight → 제출 승인.
- **정본 수식 고정(결정 27)** + 축 지도 artifact §6 코드 흐름·§8 수식 표(v10, 표는 `render_formula_tables.py`가 생성).
- **엔진 묶음 main 병합**(`ca1f30e7`, digest `99bbef91…`), gate α·β PASS, γ는 결정 22로 관측자 효과 특성화 종결. 계측 정책: 판정 측정은 전 arm 계측 OFF. telemetry 비대칭 감사 r1: M3 J1만 비대칭(`results/engine_gate_prep/AUDIT_telemetry_asymmetry_past_results_2026-10-21.md`).
- `test_baseline_pass` 비결정성 = 가짜 엔진 결함 2건 수리(`1943b3b6`). presubmit 줄 인용 드리프트 수리(`10e33bfa`); reachability 2건 차단 남음.
- 보고용 그림 6장 `reports/figures/current_2026-10-21/`(Claim G·HE0·layer-type·CM-2·CM-5·관측자 효과).
- **roofline·스케일링 그림 묶음(2026-10-25)** `reports/figures/roofline/`(README.md = 용도·재현·한계·인용 규칙, `NEXT_SESSION_PROMPT.md` = 다음 세션 전달문) · T1 대조 틀 `t1_compare.py`(데이터 오면 `--data cells.csv`) · 감사 미호출.
- **HyMux 원고 v4·검증 계획 문서 검수(2026-10-26)**: 두 Claude Docs에 댓글(원고 14·계획 9, 본문 무수정). 닫힌 식은 analytic-v1과 일치, §3 측정 주장 4건 정본 불일치(HE0 진동 서술·42→124 cudagraph OFF·Fig1=Ha8·80 SM 포화 금지 서술), §7 원자료 0. 대조표 `reports/hymux_review/EXPERIMENT_CROSSWALK_2026-10-26.md`(사슬 ①–⑤, 구체화 A-1…D-12)
- **결정 31(2026-10-26)**: PREREG rev3.6 §1-9 분석 판정 A-1 BW(D)·A-2 시간 knee·A-3 θ′ P 선형성·A-4 조성 회귀 추가(하네스·GPU-h 불변, §12 항목 25 분석 코드 대기) · V-O(oracle 격차, D 격자 전체·귀무 N2=정본 기본 규칙)·H-D(phase oracle 대 best static) 설계 초안 · HyMux 구현은 H-D `PRESENT`일 때만
- **감사 r1·r2(2026-10-26, 상한 도달)** `cap_char/AUDIT_rev36_planV_r{1,2}_2026-10-26.md`: PREREG rev3.6′ §1-9 AJ-1…AJ-4 **봉인 가능 GO-with-caveats**(r1 死因 3 해소) · PLAN_V rev2(N-1…N-5 반영)는 **CI 추정량 결정 O-S1 선결** 후 PREREG화
- **결정 32(2026-10-26)**: O-S1 = V-O 축 N (c) 전체 seed 짝 t(3) · H-D (a) seed 8 + t(n−1) → PLAN_V rev3. H-D 비용 약 64–96 GPU-h [유도] → O-H4 축소안이 사실상 선결
- **결정 33(2026-10-26)**: O-H4(q=1.0 먼저·연속 1길이, H-D 약 24–36 GPU-h)·O-V1 (c)·O-V4 (β) → PLAN_V rev4 · **§11-4 전부 복구**(L-7·L-8·L-14·L-15·TP2 장문 108u) → 하네스 반영 진행 중 → 하네스층 감사 → smoke
- **하네스층 감사 r1·r2(2026-10-26, 상한 도달)** `cap_char/AUDIT_harness_costterms_r{1,2}_2026-10-26.md`: r1 死因 0·MAJOR 2·MINOR 13 → 수리 `ae875232`·`4430dc37`(PREREG rev3.8, 시험 67/67, 변이 78 전부 검출) → r2 **봉인 불가: N-1 HARNESS_DIGEST가 analytic_terms.py(등록 O 계산)·make_variant·engine_digest·job_entry·계산된 plan을 덮지 않음** · smoke는 진행 가능
- **결정 34(2026-10-26)**: N-1 (a)·A-49 추가·A-50 기본 제외·N-2·N-3 → **수리 완료 `4e0ae086`·`001e4b21`(PREREG rev3.9)**: HARNESS_DIGEST v2(외부 4파일 + 전 Job plan sha), 시험 71/71·변이 84/84 → 봉인 자격 충족(3회차 감사 없음). main 세션 실트리 점검: analytic_terms 1줄 추가 → check rc 13. 예약 22.4 GPU-h, ≤6 h 위반 없음
- **Workspace smoke(2026-10-26, 비측정·인용 불가)** `cap_char/costterms_smoke_20261026T054338Z/README.md`: 3 boot rc 0·ENGINE_DIGEST MATCH·실현 SM 일치 · **PD-mux bs 8 ramp 상한 0 대 실제 118 → 창 오염(모형 수리 필요)** · T4 16k `FILL_INCOMPLETE`(R 4,096이 채우기 전에 끝남, 설계 결정) · 16k 채우기 약 8.1 min(모형 6) → J-B2 캡 재계산 · Workspace standby
- **multi-GPU 증거 검토(2026-10-26)** `reports/hymux_review/MULTIGPU_EVIDENCE_REVIEW_2026-10-26.md`: N>1 성능 수치 0 · S2f 조건 ① 세 Job 모두 FAIL(fused_sym boot2 미완) · 메인 지표 후보 G_SLO(PD-mux/tuned fused, 같은 Job) · rev12 J2·J4·J8도 telemetry 비대칭(CONSENSUS:8799 정정 후보) · Bullet 게재처 미확인
- **결정 38(2026-10-26, 사용자 지적)**: 10-21 이후 측정 Job 0·PREREG 18회 개정·감사 10건 → 측정 우선 전환, 새 설계 동결, 기술 측정은 하네스 감사 1회·MINOR 공시, 새 데이터 없이 확정 결론 재개 금지. (작업 방식 피드백 — promote 때 메모리 반영)
- **비용 항 캠페인 완료(2026-10-26)**: 6 Job(J-A1·J-A2·J-H1·J-H2·J-B2·J-C) 전부 succeeded·회수(DONE + sha256), 46 boot 전부 usable·flag 0; J-A1 1차는 results symlink 경로 버그로 rc 13(측정 0) → 봉인 S1′ 재제출. AJ 전체 `cap_char/costterms_analysis/AJ_full_2026-10-26.json` · **결과 분석 `cap_char/RESULT_costterms_2026-10-26.md`(`73f293cf`, 미감사)**: T1 F1은 비분할만 채택·sticky-D는 F2/F3, round 차 ≤ 5.4% · T2 Nano q 1.088–1.090 NEAR_LINEAR, 장문 124k까지 이차형 · T3 ρ 1.13–1.19(동거 창 power cap 노출 96–100% → 총효과만) · T4 산식 B 9/10 HELD · 조성 P1–P4 · TP2 F1·F2·F5 HELD · ★AJ-2 J-H1 병기(b(n)·기울기·b(a31))는 a22·a31 FORM_INADEQUATE라 NA_FORM이어야 함(aj_analysis 게이트 누락) — 인용 금지 · claims-auditor r1 진행 중
- **결정 35(2026-10-26)**: multi-GPU = GPU별 SM 분할 + DP 라우팅 설계 초안 `dp_routing/PLAN_DP_routing_per_gpu_partition_2026-10-26.md`(독립 서버 N개, 클라이언트 측 길이 라우팅, R0–R4, G_SLO) · M3 9.5 h 정지 = 21:14:03Z ±40 ms 동시 종료, 경과 9.51 h(주기 집행 9.5 h 상한·일괄 종료와 구분 불가, 원인 미확인 — 감사 r1 정정) · **규칙층 감사 r1: rev0 NO-GO**(Llama τ 항등식·결정 14·24 충돌·τ=2c* 퇴화) → 결정 36 + 수리 3–12 → PLAN_DP rev1 → **감사 r2(최종) GO-with-caveats**(PREREG 작성 조건 N-1…N-6) · 결정 37(Llama 참조점 = fused R1·R1′) → PREREG 작성 단계
- **변화 trace 타당성 검토(2026-10-26)** `reports/hymux_review/CHANGE_TRACE_VALIDITY_2026-10-26.md`: 실 trace에 변동은 실재(Azure code 하루 6–11×, BurstGPT 분 단위 버스트, ServeGen 입력 1.63×)하나 우리 trace(17–67 s 사각파 + drain, 30↔60%)는 시간 척도가 다름 → HE0 일반화·라우팅 교대 결과가 약해짐, 정상 상태 근거 결론은 무관 · 권고: 서술 정정 + 실 trace 파라미터·재생 + 정상 상태 격자 논증
- 설계 결정 원문 1–34: `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`.
## 다음 행동
1. **봉인 S1 완료**(하네스 `b0c9d899`, 봉인 `c64de716`, 시험 73/73·변이 85/85, digest check rc 0) · ramp 하한·J-B2 캡 반영(예약 24.7 GPU-h, 최대 캡 312분) → J-A1 1차(`job-bnkxoji00zmy`) 6분 만에 `ABORT_HARNESS_DIGEST`(results symlink 아래 resolve() 경로 버그, 측정 0) → 수리·**봉인 S1′ `af28dfe8`** → **J-A1 재제출(08:32Z, `job-4euwgpvd00ao`)** → J-A2·J-H1·J-H2·J-B2·J-C · AJ 분석 코드 커밋 `ae590e71`(fetch 전, 시험 23/23, 하네스 digest 불변) · 알려진 사소한 결함: `ct_harness_digest.py check --dir`에 **상대 경로**를 주면 spec/plans sha가 ERROR:2(subprocess cwd 이중 경로) — Job은 절대 경로라 무관, 다음 봉인 때 수리 · 새 설계 동결 유지
2. (roofline 묶음) 감사 r1 반영 완료(README 정정) → §6 한 줄 추가 여부 사용자 결정 · r2는 사용자 요청 시만 · T1 데이터 오면 `t1_compare.py`로 decode 대조
3. G0 PREREG(헤드라인 셀 전 arm 계측 OFF, 조성 변형 goodput 범위 = 결정 25) → V PREREG(oracle 격차 측정 추가 검토) → M3 재개 여부
## 사용자 결정 대기
- **변화 trace 보강** / 타당성 검토 §5 / 실 trace 재생 1종 + 실 trace 유도 교대 + 정상 상태 격자 논증(권고) · DP 라우팅 30↔60% 교대 재설정(결정 36 개정 후보) · HE0 범위 문안 보강(promote)
- **multi-GPU의 논문 내 위치** / 검토 문서 §5 / (A) 단일 GPU 메인 + 실측 TP 이식성 절(J-C + TP {1,2,4} G_SLO, Nano+Llama, 약 70–112 GPU-h [유도]) · (B) multi-GPU 메인(G0 양의 결과 선행 + 대형 hybrid + TP ≤ 8, 150–240+) · 9.5 h 정지 원인 조회 선행 · DP 라우팅 설계 열린 결정 O-D1…O-D5(PLAN §4)
- **T4 16k 채우기 `FILL_INCOMPLETE` 처리**(smoke) / R 4,096 출력이 채우기 완료 전 종료 / R 상향(예약 증가 → C_mem 변화 공시) · 16k 채우기를 기술값 라벨로 유지 · L 축소 — 하네스 담당 제안 뒤 결정 · m12 원자료 보관 코드화 시점
- **roofline η 정본 반영** / 감사 r1: 집계 η≈0.49는 층별 η_X 아님(§8 불가) / 수식 문서 §6 "실측 대조"에 집계 ≈0.49 한 줄만(권고, 결정 27 규칙·promote 경로) · 반영 안 함 · 층별 η 식별 GPU 측정(조성 변형 a4–a31 prefill floor, 승인 필요)
- **presubmit reachability 차단 2건** / M4R·TC1 rev3 spec이 active로 전역 차단(측정과 무관) / 범위 한정 OVERRIDE + 승인(권고, 전례 `cp_baseline/OVERRIDE_*`) · 두 spec 폐기
- **로컬 엔진 트리 재동기화** / 구판이라 로컬 테스트 5건 실패 / amd·opchar 영향 확인 뒤 `install_runtime.sh` 절차(권고) · scratch 사본 유지
- **임시 디렉터리 `/tmp/gate_t_*` 11개 삭제** / 테스트 진단 잔여물, 에이전트 삭제가 막힘 / 삭제(권고)
- **논문 방향** / static 결론만으로는 제안이 아님 / ② 비용 모델 기반 static 설정(권고, V + oracle 격차) · ① 특성화 · ③ 메커니즘
- LongBench v2·InfiniteBench 보류(결정 12′) · 기존 유지: D-10 · radix OFF · 장문 hold-out · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음(비용 항 6 Job 회수 완료). Workspace 둘 다 standby.
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안 · gate 감사 r1/r2 인용 금지 문장 · telemetry 비대칭 감사 §3(6건) · PREREG §9-2(T1 명목 ctx·장문 경계 시험 혼동 포함) · "슬롯 상한이 hybrid 고유" · M3 판정서 문안 · J1 크기 문장
## 최근 로그
- 2026-10-25: roofline·스케일링 그림 묶음 + prefill 관측 대 이상 대조 + T1 비교 틀(GPU 0, 미감사)
- 2026-10-22~23: 정본 수식 고정(결정 27)·축 지도 §6·§8 · 하네스 1·2단계, rev3.4·3.5(결정 28–30) · 조성 프로브 PASS · test_baseline_pass 수리
- 2026-10-21: 결정 22–26, engsync 병합, telemetry 감사 r1, multigpu·engsync worktree 폐기, promote 4차, 보고 그림 6장
- 2026-10-21: 관측자 효과 분해 Job(TEL λ −1.7%, I-4 ITL +1.0%)
- 2026-10-20: 결정 14(비용 모델 주 목적) · PREREG rev0–rev3 · 엔진 묶음 gate α·β PASS · 128k 프로브(int32 원인)
