# TRACK: main — hybrid 서빙 비용 모델 · 비용 항 측정 하네스 · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14: **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU + TP2 항), PD-mux 이득(G) = 검증 대상, "왜 hybrid"(H) = 같은 골격 조성 스윕(결정 24) + Llama 참조점. 수식 기반 = 정본 수식(결정 27). M3는 G0 뒤. **결정 38: 측정 우선, 데이터 없이 설계·감사를 쌓지 않음.**
## cost model 기여
- 채우는 칸: CM-1(T1)·CM-2(T2·T2′)·CM-4(T3)·CM-5(T4)·(확장) TP2·조성. 사전값 = analytic-v1(`reports/cost_model/ANALYTIC_COST_FORMULAS.md`, 계산 `scripts/costmodel/analytic_terms.py`)
- 이번에 알게 된 것(2026-10-26 비용 항 캠페인, 감사 r1 PLAUSIBLE 조건부 → promote 5차 반영): T1 F1은 비분할 P=108만 채택·sticky-D는 F2/F3(두 Job 같은 형) · T2 Nano q NEAR_LINEAR, 장문 124k LONG_SERVED · T3 ρ 1.13–1.19는 동거·400 W 클럭·관측자 비용 총효과 · T4 cap 비구속 점 2개(HELD 1·MISS 1, ratio_A 0.68 예측 → 0.81) · 조성 P1·P2·P4 · TP2 F1·F2·F5 · AJ-1 `BWD_INV_D_MISS_LE64`·`BWD_NO_KNEE_TO_108`, AJ-3 θ′ LINEAR(합성 지수) — 세부·금지 목록 `workspace/engine-port/results/cap_char/RESULT_costterms_2026-10-26.md` §17
- promote 필요: **예 (감사·결과)** — lockstep 정지 진단 `results/lockstep_diag/DIAG_lockstep_stall_share_2026-10-27.md`(CM-4: VESSL·Nano·예산 65536에서 별도 정지 항 ≈0, 동거 간격 = T_dec·ρ; C2 꼬리 분위 인용 정지 후보) · C 미주장 감사 r1 `anchor_audit/VERDICT_claimC_online_predictive_2026-10-27.md`(+ 학회 검토 `reports/hymux_review/VENUE_C_positioning_2026-10-27.md`, venue_positioning 정정 후보 2건) · 그 외 5차 `dc7193f4` 반영 완료 · 예외: HE0 변화 trace 대표성(`reports/hymux_review/CHANGE_TRACE_VALIDITY_2026-10-26.md`)은 claims-auditor 판정 없음 → 감사 뒤 `예 (감사)`
## 현재 상태 (2026-10-27 저녁)
- **비용 항 측정 캠페인 완료·promote 5차 반영**: 6 Job(봉인 S1′ `af28dfe8`, PREREG rev3.11) succeeded·회수, 결과 `cap_char/RESULT_costterms_2026-10-26.md` rev1 + 감사 r1, boot 원자료는 클러스터 볼륨에만(m12). 사고 2건(J-A1 symlink 경로·AJ-2 NA_FORM)은 수리·인용 금지 처리
- **C 감사 r1 + 학회 검토**(`anchor_audit/VERDICT_claimC_online_predictive_2026-10-27.md`, `reports/hymux_review/VENUE_C_positioning_2026-10-27.md`): C(예측형 adaptive SM 배분)는 "기각"이 아니라 **미검증·시험 가능**. offline 프로파일 최소-SM은 서빙 측정 0. DuetServe(ICML'26)가 transformer에서 같은 형태 → 신규성은 hybrid 귀속뿐. A+B만이면 top venue 약
- **lockstep 정지 진단**(`results/lockstep_diag/DIAG_lockstep_stall_share_2026-10-27.md`, 미감사): VESSL·Nano(flashinfer)에서 정지가 ITL 꼬리를 지배하지 않음(동거 간격 20.1–21.0 ms measured, span 기울기 0.0015), 꼬리 = 동거 ρ_on 1.156–1.193. KISTI Zamba2-7B·Codestral(triton)은 0.7–1.8 s 정지 실재·기전 미상, HE0는 phase 데이터 0. TTFT 대기 77–89%. C2 꼬리 분위 정지 오염(p50 무영향). HyMux Fig.1 정지 수치 출처 = KISTI s8_frontier
- **방향 재정리(제안, 미결정)**: 부정 결론(HE0·decode-heavy 기본값·layer-type)은 KISTI·Zamba2 기판 범위로 · 승계 기본값(D=44·cap 48·budget 65536)은 Nano에서 재결정 · HyMux 원고는 구현 구조로 낮추고 R1–R3를 단일 처치 ablation 가설로(R3·R1 > R2) · 비동기 기판 arm은 후순위 · 이관 지시(구조 검증 T1–T5, `STRUCTURE_batching_admission_2026-10-27.md` 예정)는 미착수 — 로컬 엔진 트리 `../sglang_engine_dev`(v0.5.10 `1519acf`) 존재, F2·F5·F6·F10·derived 3건만 확인
- 동결 설계: V-O·H-D `cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md` rev4 · DP 라우팅(감사 r2). 사소한 결함: `ct_harness_digest.py check --dir` 상대 경로
## 다음 행동
1. CPU: claims-auditor — 부정 결론 기판 재범위화·C 문장·COMPARE §2.2 "ITL≈span"·C2 꼬리 인용 정지 · 구조 검증 보고서 T1–T5(원고–코드 대조표 포함)
2. GPU Job 1개 PREREG: H-D(우선) + V-O + **Nano HE0 핵심 셀(static D 격자)** + 계측(chunk GPU ms·drain·span rids·I-4·client 절대시각), 선택 Zamba2 arm
3. C 관문(H-D >3% 또는 V-O phase별 oracle 분기) → 순수 attention 재현 → hybrid 4갈래 → HyMux R3→R1 ablation
## 사용자 결정 대기
- **기판 재검토(F3·F4 재논의)** / 진단: 현 VESSL·Nano 기판은 lockstep 정지 없음, 그러나 HE0 등 부정 결론의 KISTI·Zamba2(triton) 기판에는 정지 실재·미귀속 / ① 부정 결론을 "KISTI·Zamba2 기판" 범위로 재서술 + Nano에서 HE0 핵심 셀 재측정(권고) · ② 비동기 기판 arm 구현 · ③ 현행
- **다음 측정 경로** / 결정 38 동결 해제 조건 충족, C 관문·기본값 재결정이 이 Job에 걸림 / V-O+H-D+Nano HE0 셀 한 Job PREREG(권고, H-D 우선) · G0 먼저 · DP 라우팅 먼저
- **HyMux 원고** / 핵심 장치(span scheduler·floor admission) 미구현(F4·F10), R2 동기는 KISTI 데이터 / (a) 구현 구조로 낮추고 R3·R1을 C 관문 뒤 단일 처치로(권고) · (b) 지금 구현 · 유지
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
- 2026-10-27 오후: catch-up · 외부 비판 문서 대조 · C 감사 r1·학회 검토 · HyMux 구조 대조 · lockstep 정지 진단(CPU) · 방향 재정리(제안)
- 2026-10-26~27: 비용 항 6 Job 측정·회수·분석·결과 감사 r1·promote 5차 · 결정 31–38(측정 우선) · HyMux 원고·계획 검수 · multi-GPU·변화 trace 검토 · DP 라우팅 설계(감사 2회)
- 2026-10-25: roofline·스케일링 그림 묶음 + prefill 관측 대 이상 대조(감사 r1) + T1 비교 틀
- 2026-10-22~23: 정본 수식 고정(결정 27)·축 지도 §6·§8 · 하네스 1·2단계, rev3.4·3.5(결정 28–30) · 조성 프로브 PASS
- 2026-10-21: 결정 22–26, engsync 병합, telemetry 감사 r1, multigpu·engsync worktree 폐기, promote 4차
