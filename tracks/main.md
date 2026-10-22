# TRACK: main — hybrid 서빙 비용 모델 · 엔진 계측 gate · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14(2026-10-20): **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU 기준 + TP2 항), PD-mux 이득(G) = 그 모델의 검증 대상, "왜 hybrid"(H) = 고유 항(s 대 KV, decode ctx 기울기, prefill 선형성, cap≡슬롯)의 Llama-3.1-8B 최소 대비. M3(multi-GPU)는 G0 뒤.
## cost model 기여
- 채우는 칸: CM-1(T1 `T_dec(D,bs,ctx)`), CM-2(T2 `T_span`), CM-4(T3 ρ), CM-5(T4 메모리·admission), (확장) TP2 항
- 이번에 알게 된 것: CM-2 floor 9 ms+0.111 ms/토큰×L(≤16k, CONFIRMED) · Nano 실측 k 16,382 B/토큰·s 143.5 MB/요청(L\*≈8.8k) · chunk -1 장문 prefill 상한은 Triton causal_conv1d int32 오프셋(≈94.8k, 반증 실험 통과·감사 전) · **관측자 효과(분해 Job, agnostic·shape A·n=4): TEL 묶음(경로+PHASE_EVENTS, 몫 미분리) λ −1.7%, I-4 ITL p95 +1.0%** — 다른 arm·부하의 보정 계수 아님
- promote 필요: 예 (결정) — 결정 24–29 · 예 — 정본 정정(§1-22 (108,0) 방향)
## 현재 상태 (2026-10-21)
- **엔진 묶음 main 병합 완료**: squash `ca1f30e7`(engsync `c875dc21`; engsync worktree·branch는 2026-10-21 폐기, git 밖 원자료 229개는 main worktree로 이동), ENGINE_DIGEST `99bbef91…` 병합 후 재계산 일치. 원자료 271개 약 217 MB는 `_vessl_fetch/_engsync_raw/`(manifest `engine_gate_prep/EXCLUDED_RAW_engsync_2026-10-21.sha256`). M3 rev12 봉인 검증 fail-closed(의도). 로컬 엔진 트리 2개는 구판(미동기화).
- **분할 gate**(설계 `workspace/engine-port/results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md`, 감사 r1 GO-with-caveats·r2 상한):
  - G-α: 1회차 FAIL(i5i6_B2 기대값) → 검사 v2 재실행 **PASS 28/28**. bs 40 미도달은 요청 0 종료와 39개 일괄 합류의 **경합**(인용 금지 3문장, `AUDIT_gate_split_design_r2_2026-10-20.md`).
  - G-β: 1회차 FAIL(하네스 결함 3) → 수정(결정 20) 재실행 **PASS** — 패치본 view 출력 sha = 원본 contig, 원본 view 94,800↑ 붕괴, 서빙 98,304 통과·되돌린 판 95,100 사망·재적용 통과.
  - gamma: **A(agnostic) OBSERVER_EFFECT**(계측 ON λ −2.6%, ITL p95 +2.0%) · **F(fused_sym) UNRESOLVED_POWER**(ITL SD 4%) → **결정 22로 "관측자 효과 특성화 완료" 종결**(PASS 아님; 봉인 조건 = α·β PASS + 결정 22).
  - **분해 Job**(5수준×4블록): λ는 **TEL 묶음(경로+PHASE_EVENTS, 몫 미분리) −1.7% FLAGGED**, ITL p95는 **I-4 증분 +1.0% FLAGGED**, I-5·I-6은 CI∋0(저검정력). 결과 `workspace/engine-port/results/engine_gate/pdmux-engine-gate-decomp-d1f2ddb-20261021022225/DECOMP_RESULT.json`. 과거 결과 영향은 아래 감사 r1.
- **계측 정책(결정 22, 2026-10-21)**: 판정 측정은 전 arm 계측 OFF + Job 내 ON/OFF 짝 공시(보정 계수 아님). 원문 `…/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 결정 22.
- **비용 항 PREREG rev3.1** `workspace/engine-port/results/cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md`: 결정 18·22 반영 — §2-2′ 계측 매트릭스(T3를 TEL ON 전용 boot로 분리, T1 boot는 I-4·I-6만) + Job 내 OFF 진단 boot, §12 봉인 체크리스트(14항목, 6 완료). 결정 23: O-14 J-C OFF 진단 포함, O-15 J-B → J-B1·J-B2. 예약 16.6 → **약 18.8 GPU-h** [유도]. V는 항 데이터 뒤 별도 PREREG.
- **CPU 회귀(병합 후 main)**: 1577건, 실패 61(실패 42 + 오류 19). 병합 전 main 대비 새 실패 8 = M3 rev12 봉인 2(의도) + 로컬 엔진 트리 구판 5(`admission_probe.py` 부재) + `test_engine_gate_driver.TestAlpha.test_baseline_pass` 1(**비결정적**: 가짜 엔진 i4_B2 containment 위반 1/1047, engsync에서도 3회 중 2회 실패 → 병합 무관, 수리 필요). 엔진 코드 회귀 0.
- **결정 25(2026-10-21)**: T3 조성 의존을 J-H에 추가(rev3.3, J-H1 약 2.9 + J-H2 약 1.9 GPU-h [유도], 전체 예약 약 19.7), goodput·G 조성 의존은 G0 PREREG 범위(dummy 변형 + 실가중치 출력 길이 재생).
- **결정 24(2026-10-21)**: H 근거 = Nano 골격 조성 스윕(attention 비율 3–4점, dummy 가중치), Llama는 참조점 1–2셀로 축소. 선행 = 변형 부팅 프로브(비측정). rev3.2 반영 완료(PREREG §4′ J-H: 변형 a4·a13·a22·a31 × boot 3 + real a4 W-INDEP gate + Llama 참조 1 boot, J-B1 폐지, 예약 약 18.5 GPU-h [유도]). **조성 프로브 전 단계 PASS**(2026-10-21, `results/cap_char/compo_probe/RESULT_compo_probe_2026-10-21.md`): a4·a13·a22·a31 부팅·배너 s·k 일치·TEL 방출(a13·a31), KV 풀이 조성에 따라 887k→298k 토큰 → J-H 공통 격자 5셀로 고정. Workspace standby 확인.
- **결정 27(2026-10-22) 해석적 비용 수식 고정**: `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1) + `scripts/costmodel/analytic_terms.py`(시험 8건). 정본 반영 완료(COSTMODEL 끝 절, PROJECT_STATUS 최상단, CLAUDE.md, MEMORY).
- **코드 흐름 분석(2026-10-22)**: 업스트림 수명주기·PD-mux 패치·하네스를 코드 단위로 정리해 축 지도 §6에 반영(v9). 정정 1건: decode가 비면 분할은 (108,0)(코드 `multiplexing_mixin.py:1151`) — CONSENSUS §1-22 해석·축 지도의 '(0,108) 복귀'는 방향 오기, promote 때 정본 정정.
- **telemetry 비대칭 감사 r1**(`workspace/engine-port/results/engine_gate_prep/AUDIT_telemetry_asymmetry_past_results_2026-10-21.md`): 비대칭은 코드 구조상 실재(writer가 PD-mux에서만 생성). P1-opint·HE0 = 전 arm OFF → **불변**, N2 Job-EA = 전 boot ON 대칭 → 불변(부하 의존 성분 공시 권고), CM-2 floor 불변, **M3 J1만 비대칭** → 부호 견고·크기 공시(A G_sym≈1·d44 −5.3%는 병기 없이 인용 금지). 정본 등급 변경 0. 인용 금지 6건은 감사서 §3.
- **128k 프로브 1–3회차** `workspace/engine-port/results/cap_char/probe_128k/`(RESULT·ANALYSIS·AUDIT): 결정 13 long-ctx 상한 64k(격자 선택), mem-fraction 0.6 통일.
- 설계 결정 원문(1–21): `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`. 비교·계획: `cap_char/COMPARE_shapeC_workloads_2026-10-20.md`, `PLAN_costmodel_terms_2026-10-20.md`, 그림 `cap_char/figures/`, 장문 길이 실측 `cap_char/longctx_stats/`.
- 운영 사고·교훈: object-volume export 절단이 gamma-A·F에서 재발(분해 Job은 요약만 export해 288/288 무결) · 1회차 128k 프로브 폴링 오판으로 A100 약 60분 유휴(완료 판정은 SUMMARY/DONE 파일로) · VESSL 인증 만료 시 사용자 `vesslctl auth login`.
## 다음 행동
1. (완료 2026-10-22 `1943b3b6`: `test_baseline_pass` 비결정성 = 가짜 엔진 결함 2건 수리, 50/50 통과; 실서버 G-α containment 최소 여유 B2 0.132 ms(허용오차 포함)) · 로컬 엔진 트리 재동기화 여부 결정
2. 신규 하네스 1단계 완료(`af8c0500`, CPU 33시험·변이 22/22) → 결정 28(A-1 ramp 창·실제 평균 ctx, A-2 T2′ 주 측정, A-6 삭제) → rev3.4·하네스 수정 완료(`e9f3903a`: ramp 창·실제 평균 ctx, T2′ 주 측정, 변이 28/28, 예약 약 18.8 GPU-h [유도], J-A 셀 133→99/라운드) → 2단계(T3·J-H2·J-C) → Job entry → 하네스 감사 → smoke(J-B2 16k 채우기 위험 확인) → 하네스층 감사 → smoke → 봉인 S1 → preflight → J-A1·J-A2·J-B·J-C 제출 승인
3. G0 PREREG 작성 시 감사 §4(a) 규칙(헤드라인 셀 전 arm 계측 OFF, 파생량은 별도 boot)을 넣는다
4. 이후: G0 튜닝(fused_tuned, cap=C_mem) → V PREREG(새 감사) → M3 재개 여부
## 사용자 결정 대기
- **presubmit reachability 차단 2건** / `reachability_spec.json`(M4R)·`reach_spec_rev3_A.json`(TC1 rev3)이 active 상태로 전역 차단(PROJECT_STATUS 1488-1495 기존 블로커, 후속 spec 없음) — rev3.3 측정과 무관 / 범위 한정 OVERRIDE 문서 + 사용자 승인으로 제출 허용(전례 `cp_baseline/OVERRIDE_*`, 권고) · 두 spec을 폐기 결정
- **로컬 엔진 트리 재동기화** / `sglang_engine_dev`·`_multigpu`가 구판이라 로컬 테스트 5건 실패·digest 재계산 불가(다른 트랙 공유) / multigpu 세션은 2026-10-21 폐기, amd·opchar 영향 확인 뒤 `install_runtime.sh` 절차로 재동기화(권고) · scratch 사본으로만 계속
- **LongBench v2·InfiniteBench(공식 가운데 절단)** / 보류(결정 12′) / 후속 재논의
- 기존 유지: D-10 확정(G0 PREREG 전) · radix(OFF+편차 공시) · 장문 hold-out 모델 · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음. 원장: `workspace/engine-port/results/engine_gate_{alpha,beta,gamma_a,gamma_f,decomp}/vessl_jobs.tsv`(gamma A·F는 수동 회수, `MANUAL_RETRIEVAL.txt`). Workspace 둘 다 standby.
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안("PD-mux는 64k를 넘을 수 없다"·"OOM 때문"·G4 "SLO 충족") · gate 감사 r1/r2 인용 금지 문장(I-4 PASS 범위·관측자 TOST 범위·gate PASS≠이후 바이트 보증·bs 40 경합) · PREREG 감사의 "hybrid 서빙 비용 모델 일반"·"Bullet/MuxWise 오예측"·"~52k 경계" · M3 판정서 문안 · J1 크기 문장 · `results/multigpu/` 안 `VERDICT_*.md` 금지
## 최근 로그
- 2026-10-21: 결정 22·23, engsync squash 병합 `ca1f30e7`(digest 일치), telemetry 비대칭 감사 r1, **multigpu·engsync worktree 폐기**(git 밖 원자료 602·229개 main으로 이동)
- 2026-10-21: 관측자 효과 분해 Job(TEL λ −1.7%, I-4 ITL +1.0%), handoff
- 2026-10-20: 프레이밍 결정 14(비용 모델 주 목적) · PREREG rev0–rev3-MEAS(감사 2회 상한) · 엔진 묶음(I-5·I-4·I-6·int64) · gate α·β PASS, γ OBSERVER_EFFECT · 128k 프로브 3회(Triton conv int32 원인)
- 2026-10-19: catch-up · Claim E 첫 화살표 감사 · headline G·shape C 결정 · J1 분석·감사 · promote 3차
- 2026-10-17–18: preflight PASS · M3 J1 완주, J2·J4·J8 외부 종료
