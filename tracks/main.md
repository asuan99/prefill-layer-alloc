# TRACK: main — hybrid 서빙 비용 모델 · 비용 항 측정 하네스 · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14: **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU + TP2 항), PD-mux 이득(G) = 검증 대상, "왜 hybrid"(H) = 같은 골격 조성 스윕(결정 24) + Llama 참조점. 수식 기반 = 정본 수식(결정 27). M3는 G0 뒤. **결정 38: 측정 우선, 데이터 없이 설계·감사를 쌓지 않음.**
## cost model 기여
- 채우는 칸: CM-1(T1)·CM-2(T2·T2′)·CM-4(T3)·CM-5(T4)·(확장) TP2·조성. 사전값 = analytic-v1(`reports/cost_model/ANALYTIC_COST_FORMULAS.md`, 계산 `scripts/costmodel/analytic_terms.py`)
- 이번에 알게 된 것(2026-10-26 비용 항 캠페인, 감사 r1 PLAUSIBLE 조건부 → promote 5차 반영): T1 F1은 비분할 P=108만 채택·sticky-D는 F2/F3(두 Job 같은 형) · T2 Nano q NEAR_LINEAR, 장문 124k LONG_SERVED · T3 ρ 1.13–1.19는 동거·400 W 클럭·관측자 비용 총효과 · T4 cap 비구속 점 2개(HELD 1·MISS 1, ratio_A 0.68 예측 → 0.81) · 조성 P1·P2·P4 · TP2 F1·F2·F5 · AJ-1 `BWD_INV_D_MISS_LE64`·`BWD_NO_KNEE_TO_108`, AJ-3 θ′ LINEAR(합성 지수) — 세부·금지 목록 `workspace/engine-port/results/cap_char/RESULT_costterms_2026-10-26.md` §17
- promote 필요: **예 (감사·결과)** — ① lockstep 진단 감사 r1 `anchor_audit/VERDICT_lockstep_diag_claims_r1_2026-10-27.md`: P1 진단 PLAUSIBLE(조건부, CM-4 기술값 허용 문안만) · P2 범위 재진술만(정지 귀속 NOT-YET-SUPPORTED, layer-type 死 정지 의존 REFUTED) · P3 COMPARE §2.2 정정 배너(반영함) · P4 C2 p95/p99 인용 정지(라벨 "기원 미확정 장꼬리") · P5 C 문안 출처 보강(종결) ② 변화 trace 감사 r1 `anchor_audit/VERDICT_change_trace_validity_r1_2026-10-27.md`: HE0 불변, §1-7 범위 단서 한 문장만 가, 공개 trace 통계 인용 불가(데이터 sha256 없음, 스크립트는 `reports/hymux_review/change_trace_stats_scripts_2026-10-26/` 보관) ③ C 감사 r1·학회 검토(venue_positioning 정정 후보 2건) ④ 구조 검증 보고서 `reports/hymux_review/STRUCTURE_batching_admission_2026-10-27.md`(`7a7a2c08`, 미감사): F1–F13 CONFIRMED(F6 정정: 매 반복 동기화는 decode stream뿐, prefill은 `query()`; F7 index 수는 config 의존; F12 예외 `--enable-prefill-delayer` 미사용) · 원고 미구현 서술 R4·R5·R6·R11, 부분 R7(D_floor 닫힌 식 없음, P=108−D 설정 항등식)·R9 · C6 손익분기 8.8k는 오류 → s/(k_T−k) ≈1,251 토큰 [derived]
## 현재 상태 (2026-10-27 저녁)
- **비용 항 측정 캠페인 완료·promote 5차 반영**: 6 Job(봉인 S1′ `af28dfe8`, PREREG rev3.11) succeeded·회수, 결과 `cap_char/RESULT_costterms_2026-10-26.md` rev1 + 감사 r1, boot 원자료는 클러스터 볼륨에만(m12). 사고 2건(J-A1 symlink 경로·AJ-2 NA_FORM)은 수리·인용 금지 처리
- **C 감사 r1(+r2 문안 점검, 종결) + 학회 검토**(`anchor_audit/VERDICT_claimC_online_predictive_2026-10-27.md`, `reports/hymux_review/VENUE_C_positioning_2026-10-27.md`): C(예측형 adaptive SM 배분)는 "기각"이 아니라 **미검증·시험 가능**. offline 프로파일 최소-SM은 서빙 측정 0. DuetServe(ICML'26, venue-strategist 웹 확인, 재현 0)가 transformer에서 같은 형태라는 문헌 판단상 C의 차별점은 hybrid 귀속에 달려 있고, 그 귀속은 현재 A4 재매개화 항등식 장애가 있다(감사 CONFIRMED). A+B만이면 top venue 약
- **lockstep 정지 진단 + 감사 r1**(`results/lockstep_diag/DIAG_…`, `anchor_audit/VERDICT_lockstep_diag_claims_r1_2026-10-27.md`): VESSL·Nano·flashinfer·65536·8k/32k·PHASE ON 범위에서 동거 decode 간격이 span 시간에 거의 반응 안 함(기울기 0.0015 [0.0005, 0.0027], x 157–428 ms에서 ≤0.7 ms), 별도 정지 항은 해상도(≤1 ms) 안 미관측, ρ_on 1.156–1.193은 동거·전력·관측자 합성값. N3 Job 3 첫-chunk host 차단 0.66–1.34 s 원인 미해결. KISTI·HE0는 phase 0 → 정지 존재·관여 판단 불가, 정지는 모델·backend와 교락(게이트 #83). **HE0가 정지에 좌우될 유일 경로 = 동적 arm 전환 drain(`:1344`,`:1361`) 미측정**(열린 구멍). layer-type 死는 정책 구현 자체의 layer 경계 동기화라 기판 정지와 독립
- **방향 재정리(제안, 미결정, 감사 반영)**: 부정 결론은 "KISTI·Zamba2-2.7B·triton·변화 trace 범위, VESSL·Nano 전이 미검증"으로 범위만 재진술(정지 기판 귀속 금지, layer-type 死 제외) · 승계 기본값(D=44·cap 48·budget 65536)은 Nano에서 재결정 · HyMux 원고는 구현 구조로 낮추고 R1–R3를 단일 처치 ablation 가설로(R3·R1 > R2) · 비동기 기판 arm은 후순위 · 구조 검증 T1–T5 완료(`7a7a2c08`, 원고 대조는 Revision Letter R3 기준, 본문 직접 대조 미실시)
- 동결 설계: V-O·H-D `cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md` rev4 · DP 라우팅(감사 r2). 사소한 결함: `ct_harness_digest.py check --dir` 상대 경로
## 다음 행동
1. 사용자 결정(기판 재검토·다음 측정 경로·HyMux 원고) → promote 6차(감사 2건·진단·C·구조)
2. GPU PREREG 2개로 분리(감사 권고): (가) 진단 Job = {Nano/flashinfer, Zamba2-2.7B/triton} × D{24,44,92} × 프롬프트{2k,8k} × 예산{8192,65536} 저부하 + HE0 핵심 셀 reactive bind+GATE 1·best static 1(drain 측정) + PHASE OFF 1 + 계측 §5 (i)–(v)·drain 진행 chunk 수 · (나) 정책 Job = H-D(우선)+V-O+Nano 변화 trace HE0 핵심 셀(n≥4, paired CI). 실 trace 재생 R-1은 V-O 회수 뒤 결정
3. C 관문(H-D >3% 또는 V-O phase별 oracle 분기) → 순수 attention 재현 → hybrid 4갈래 → HyMux R3→R1 ablation
## 사용자 결정 대기
- **기판 재검토(F3·F4 재논의)** / 감사 r1: 정지 기판 귀속 NOT-YET-SUPPORTED, layer-type 死는 정지와 독립, HE0 유일 열린 경로 = 전환 drain / ① 범위 재진술(모델·backend·trace, 귀속 없음) + 진단 Job(가)·정책 Job(나) 분리(권고) · ② 비동기 기판 arm 구현 · ③ 현행
- **다음 측정 경로** / 결정 38 동결 해제 조건 충족, C 관문·기본값 재결정이 이 Job에 걸림 / V-O+H-D+Nano HE0 셀 한 Job PREREG(권고, H-D 우선) · G0 먼저 · DP 라우팅 먼저
- **HyMux 원고** / 핵심 장치(span scheduler·floor admission) 미구현(F4·F10), R2 동기는 KISTI 데이터 / (a) 구현 구조로 낮추고 R3·R1을 C 관문 뒤 단일 처치로(권고) · (b) 지금 구현 · 유지
- **multi-GPU의 논문 내 위치** / 메인 지표 근거 0 / (A) 단일 GPU 메인 + 실측 TP 이식성·DP 라우팅 절(권고) · (B) multi-GPU 메인(G0 양의 결과 + 대형 hybrid 선행) · DP 라우팅 열린 결정 O-D1…O-D5
- **변화 trace 보강** / 감사 r1: HE0 불변, §1-7 범위 단서 한 문장 가, 실 trace 재생은 HE0 문안에 불필요(H-D phase-oracle이 같은 estimand) / 범위 단서만 promote + R-1(Azure 1 h 창, 배율 1, thinning seed 4, ≈13–15 GPU-h [유도])은 V-O 회수 뒤 결정(권고) · 지금 R-1 포함 · 게이트 2 개정(정상 상태 격자를 정책 비교 1차 근거로)
- **m12 원자료 보관** / boot 원자료가 클러스터 볼륨에만 있어 CLOCK_SENSITIVE·T4 MISS 재분석 불가 / 회수·보관 단계 코드화(권고) · 볼륨 유지만
- **roofline η 정본 반영** / 집계 η≈0.49는 층별 η 아님 / 수식 문서 §6 "실측 대조"에 한 줄만(권고) · 반영 안 함
- **presubmit reachability 차단 2건** / M4R·TC1 rev3 spec 전역 차단(측정 무관) / 범위 한정 OVERRIDE(권고) · 두 spec 폐기
- **로컬 엔진 트리 재동기화** / 2026-10-27 확인: `sglang_engine_dev`는 VESSL 10-20 서빙 manifest(`6015796d…`, 27항목)와 sha256 전부 일치, 차이는 기본 OFF probe 파일·훅뿐. 저장소 untracked `results/runtime_source_manifest.sha256`이 구판(5건 FAILED) / 구판 manifest 갱신·로컬 테스트 실패 원인 재확인(권고) · 유지
- **임시 디렉터리 `/tmp/gate_t_*` 11개** / 테스트 잔여물 / 삭제(권고)
- **논문 방향** / C 감사 r1·학회 검토(2026-10-27): A+B만이면 생태계 대비 약함, C 신규성은 hybrid 귀속뿐 / **② + C 사다리(권고)**: H-D → V-O → C 게이트(H-D 여지 >3% 또는 V-O phase별 oracle 분기, 첫 마일스톤 = 순수 attention에서 예측형이 static을 이기는 재현; 최소 조건 = P_gen+recal·DuetServe판 baseline·hold-out·분할 T_dec F1·δ·CM-6 목적항·iteration 배선) · ② 단독 · ① 특성화(ISPASS/IISWC)
- LongBench v2·InfiniteBench 보류(결정 12′) · 기존 유지: D-10 · radix OFF · 장문 hold-out · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음. Workspace 둘 다 standby.
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안 · gate 감사 r1/r2 인용 금지 · telemetry 비대칭 감사 §3 · PREREG §9-2·§1-9 금지 · RESULT_costterms §17 promote 금지 목록 · "슬롯 상한이 hybrid 고유" · M3 판정서 문안 · J1 크기 문장 · AJ-2 J-H1 값
## 최근 로그
- 2026-10-27 오후: catch-up · 외부 비판 문서 대조 · C 감사 r1·학회 검토 · HyMux 구조 대조 · lockstep 정지 진단(CPU) · 방향 재정리(제안)
- 2026-10-26~27: 비용 항 6 Job 측정·회수·분석·결과 감사 r1·promote 5차 · 결정 31–38(측정 우선) · HyMux 원고·계획 검수 · multi-GPU·변화 trace 검토 · DP 라우팅 설계(감사 2회)
- 2026-10-25: roofline·스케일링 그림 묶음 + prefill 관측 대 이상 대조(감사 r1) + T1 비교 틀
- 2026-10-22~23: 정본 수식 고정(결정 27)·축 지도 §6·§8 · 하네스 1·2단계, rev3.4·3.5(결정 28–30) · 조성 프로브 PASS
- 2026-10-21: 결정 22–26, engsync 병합, telemetry 감사 r1, multigpu·engsync worktree 폐기, promote 4차
