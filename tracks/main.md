# TRACK: main — hybrid 서빙 비용 모델 · 엔진 계측 gate · G0/M3   (worktree `prefill-layer-alloc`, branch `main`; 엔진 묶음은 worktree `prefill-layer-alloc-engsync` 브랜치 `engine-sync-i5-int64`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14(2026-10-20): **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU 기준 + TP2 항), PD-mux 이득(G) = 그 모델의 검증 대상, "왜 hybrid"(H) = 고유 항(s 대 KV, decode ctx 기울기, prefill 선형성, cap≡슬롯)의 Llama-3.1-8B 최소 대비. M3(multi-GPU)는 G0 뒤.
## cost model 기여
- 채우는 칸: CM-1(T1 `T_dec(D,bs,ctx)`), CM-2(T2 `T_span`), CM-4(T3 ρ), CM-5(T4 메모리·admission), (확장) TP2 항
- 이번에 알게 된 것: CM-2 floor 9 ms+0.111 ms/토큰×L(≤16k, CONFIRMED) · Nano 실측 k 16,382 B/토큰·s 143.5 MB/요청(L\*≈8.8k) · chunk -1 장문 prefill 상한은 Triton causal_conv1d int32 오프셋(≈94.8k, 반증 실험 통과·감사 전) · **관측자 효과(분해 Job, agnostic·shape A·n=4): TEL 묶음(경로+PHASE_EVENTS, 몫 미분리) λ −1.7%, I-4 ITL p95 +1.0%** — 다른 arm·부하의 보정 계수 아님
- promote 필요: 예 (감사) — telemetry 비대칭 감사 r1·128k 프로브 감사·PREREG rev1/rev2 규칙층 NO-GO·gate 하네스 감사 r1/r2 · 예 (결정) — 결정 10–23 · 예 — G-α/G-β PASS, gamma A/F, 관측자 효과 분해 결과 · 예 — COSTMODEL L\*≈9k 단위 정정(8.8k)
## 현재 상태 (2026-10-21)
- **엔진 묶음** engsync(최신 `c875dc21`): I-5 + I-4·I-6 + Triton conv int64 패치, ENGINE_DIGEST `99bbef91…`(30항목). main 미병합. 병합 시 M3 rev12·preflight PASS·N2/N3/E2 fail-closed(의도).
- **분할 gate**(설계 `…engsync/workspace/engine-port/results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md`, 감사 r1 GO-with-caveats·r2 상한):
  - G-α: 1회차 FAIL(i5i6_B2 기대값) → 검사 v2 재실행 **PASS 28/28**. bs 40 미도달은 요청 0 종료와 39개 일괄 합류의 **경합**(인용 금지 3문장, `AUDIT_gate_split_design_r2_2026-10-20.md`).
  - G-β: 1회차 FAIL(하네스 결함 3) → 수정(결정 20) 재실행 **PASS** — 패치본 view 출력 sha = 원본 contig, 원본 view 94,800↑ 붕괴, 서빙 98,304 통과·되돌린 판 95,100 사망·재적용 통과.
  - gamma: **A(agnostic) OBSERVER_EFFECT**(계측 ON λ −2.6%, ITL p95 +2.0%) · **F(fused_sym) UNRESOLVED_POWER**(ITL SD 4%) → **결정 22로 "관측자 효과 특성화 완료" 종결**(PASS 아님; 봉인 조건 = α·β PASS + 결정 22).
  - **분해 Job**(5수준×4블록): λ는 **TEL −1.7% FLAGGED**, ITL p95는 **I-4 증분 +1.0% FLAGGED**, I-5·I-6은 CI∋0(저검정력). 결과 `…engine_gate/pdmux-engine-gate-decomp-d1f2ddb-20261021022225/DECOMP_RESULT.json`. ★기존 telemetry는 PD-mux 경로만 방출 → 과거 PD-mux 대 fused 비교에 소폭 편향 가능(감사 전).
- **계측 정책(결정 22, 2026-10-21)**: 판정 측정은 전 arm 계측 OFF + Job 내 ON/OFF 짝 공시(보정 계수 아님). 원문 `…/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 결정 22.
- **비용 항 PREREG rev3.1** `workspace/engine-port/results/cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md`: 결정 18·22 반영 — §2-2′ 계측 매트릭스(T3를 TEL ON 전용 boot로 분리, T1 boot는 I-4·I-6만) + Job 내 OFF 진단 boot, §12 봉인 체크리스트(14항목, 3 완료). 결정 23: O-14 J-C OFF 진단 포함, O-15 J-B → J-B1·J-B2. 예약 16.6 → **약 18.8 GPU-h** [유도]. V는 항 데이터 뒤 별도 PREREG.
- **engsync 병합 검토**(git-committer, 2026-10-21): 경로 교집합 0·의미 충돌 0(앞선 "충돌"은 git 2.34에 `merge-tree --write-tree`가 없어 난 오판). 단 `results/engine_gate*` 약 904 MB(`.so` 12·Triton 캐시·`power_samples.csv` >10 MB 3개)가 21커밋 이력에 있음. engsync CPU 회귀 1570건 중 실패 66(대부분 로컬 환경: `sgl_kernel`·엔진 트리 경로 부재로 추정) — main 베이스라인 대조 중.
- **telemetry 비대칭 감사 r1**(`workspace/engine-port/results/engine_gate_prep/AUDIT_telemetry_asymmetry_past_results_2026-10-21.md`): 비대칭은 코드 구조상 실재(writer가 PD-mux에서만 생성). P1-opint·HE0 = 전 arm OFF → **불변**, N2 Job-EA = 전 boot ON 대칭 → 불변(부하 의존 성분 공시 권고), CM-2 floor 불변, **M3 J1만 비대칭** → 부호 견고·크기 공시(A G_sym≈1·d44 −5.3%는 병기 없이 인용 금지). 정본 등급 변경 0. 인용 금지 6건은 감사서 §3.
- **128k 프로브 1–3회차** `workspace/engine-port/results/cap_char/probe_128k/`(RESULT·ANALYSIS·AUDIT): 결정 13 long-ctx 상한 64k(격자 선택), mem-fraction 0.6 통일.
- 설계 결정 원문(1–21): `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`. 비교·계획: `cap_char/COMPARE_shapeC_workloads_2026-10-20.md`, `PLAN_costmodel_terms_2026-10-20.md`, 그림 `cap_char/figures/`, 장문 길이 실측 `cap_char/longctx_stats/`.
- 운영 사고·교훈: object-volume export 절단이 gamma-A·F에서 재발(분해 Job은 요약만 export해 288/288 무결) · 1회차 128k 프로브 폴링 오판으로 A100 약 60분 유휴(완료 판정은 SUMMARY/DONE 파일로) · VESSL 인증 만료 시 사용자 `vesslctl auth login`.
## 다음 행동
1. (결정 23) main 회귀 베이스라인 완료 대기 → engsync squash 병합(원자료 제외·`.gitignore` 보강, git-committer) → 병합 후 회귀 대조 → digest 재확인
   + P1-opint 엔진 trace-path 분기 CPU 확인(진행 중)
2. 신규 하네스 구현(§12 #7) → 하네스층 감사 → smoke → 봉인 S1 → preflight → J-A1·J-A2·J-B·J-C 제출 승인
3. G0 PREREG 작성 시 감사 §4(a) 규칙(헤드라인 셀 전 arm 계측 OFF, 파생량은 별도 boot)을 넣는다
4. 이후: G0 튜닝(fused_tuned, cap=C_mem) → V PREREG(새 감사) → M3 재개 여부
## 사용자 결정 대기
- **LongBench v2·InfiniteBench(공식 가운데 절단)** / 보류(결정 12′) / 후속 재논의
- 기존 유지: D-10 확정(G0 PREREG 전) · radix(OFF+편차 공시) · 장문 hold-out 모델 · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음. 원장: engsync `workspace/engine-port/results/engine_gate_{alpha,beta,gamma_a,gamma_f,decomp}/vessl_jobs.tsv`(gamma A·F는 수동 회수, `MANUAL_RETRIEVAL.txt`). Workspace 둘 다 standby.
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안("PD-mux는 64k를 넘을 수 없다"·"OOM 때문"·G4 "SLO 충족") · gate 감사 r1/r2 인용 금지 문장(I-4 PASS 범위·관측자 TOST 범위·gate PASS≠이후 바이트 보증·bs 40 경합) · PREREG 감사의 "hybrid 서빙 비용 모델 일반"·"Bullet/MuxWise 오예측"·"~52k 경계" · M3 판정서 문안 · J1 크기 문장 · `results/multigpu/` 안 `VERDICT_*.md` 금지
## 최근 로그
- 2026-10-21: 관측자 효과 분해 Job(TEL λ −1.7%, I-4 ITL +1.0%), handoff
- 2026-10-20: 프레이밍 결정 14(비용 모델 주 목적) · PREREG rev0–rev3-MEAS(감사 2회 상한) · 엔진 묶음(I-5·I-4·I-6·int64) · gate α·β PASS, γ OBSERVER_EFFECT · 128k 프로브 3회(Triton conv int32 원인)
- 2026-10-19: catch-up · Claim E 첫 화살표 감사 · headline G·shape C 결정 · J1 분석·감사 · promote 3차
- 2026-10-17–18: preflight PASS · M3 J1 완주, J2·J4·J8 외부 종료
- 2026-10-16: M3 rev12 감사·봉인
