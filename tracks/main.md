# TRACK: main — hybrid 서빙 비용 모델 · 비용 항 측정 하네스 · G0/M3   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄 이내). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
결정 14: **주 목적 = Nemotron-Nano-9B-v2 서빙 비용 모델(C)**(단일 GPU + TP2 항), PD-mux 이득(G) = 검증 대상, "왜 hybrid"(H) = 같은 골격 조성 스윕(결정 24, dummy 가중치 a4·a13·a22·a31) + Llama 참조점. 수식 기반 = 정본 수식(결정 27). M3는 G0 뒤.
## cost model 기여
- 채우는 칸: CM-1(T1 `T_dec(D,bs,ctx)`), CM-2(T2·T2′ `T_span`), CM-4(T3 ρ·σ′), CM-5(T4), (확장) TP2·조성 항. 사전값 = `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1, 계산 `scripts/costmodel/analytic_terms.py`)
- 이번에 알게 된 것: 정본 수식 = 부팅 배너(파라미터 8,888,227,328·s 143.5 MB·k 16,384 B)와 일치, J1 floor 기울기 = 실효 약 141 TFLOP/s(피크 약 45%) [유도] · decode는 c* ≈ 17.5k 아래에서 SSM state 트래픽이 KV보다 큼 [유도] · 조성 프로브 전 단계 PASS(KV 풀 887k→298k 토큰) · 코드 사실: decode가 비면 분할 (108,0)
- promote 필요: 예 (결정) — 결정 24–30 · 예 — 정본 정정(CONSENSUS §1-22 (108,0) 방향)
## 현재 상태 (2026-10-23)
- **비용 항 PREREG rev3.5 확정·미봉인** `workspace/engine-port/results/cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md` + **하네스 1·2단계 구현 완료** `results/cap_char/costterms_harness/`(최신 `e141a36e`: CPU 시험 50/50, 변이 44/44, ENGINE_DIGEST 검사 rc 13, smoke 모드, 원자료 클러스터 볼륨, 셀 단위 재개). Job: J-A1·J-A2·J-H1·J-H2·J-B2·J-C, 예약 약 19.4 GPU-h [유도].
  - 결정 28: T1 창 = ramp 인식(전원 합류 후 300 step·취소·실제 평균 ctx 적합), T2는 P=108만, P 축은 T2′. 결정 29: shape C 셀 손실은 smoke 후 ramp 상한 재설정. 결정 30: 장문 측정점(N=1 98k·124k, TP2 96k·124k) 별도 맨 끝 boot로 복구 + **측정점 삭제 원칙**(물리적 불가·더 강한 증거만 삭제). §11-4 삭제점 전수표.
  - launch 연결 확인(`48dd20ed`, libnuma는 측정 이미지에 이미 있음), smoke 계획 `results/cap_char/SMOKE_PLAN_costterms_2026-10-22.md`(3 boot, 상한 90분).
  - 봉인 체크리스트 §12 진행 중. 남은 핵심: 미복구 설정 제약 점 결정 → 하네스층 감사 → smoke → ramp 상한·J-B2 하드캡 재설정 → 봉인 → preflight → 제출 승인.
- **정본 수식 고정(결정 27)** + 축 지도 artifact §6 코드 흐름·§8 수식 표(v10, 표는 `render_formula_tables.py`가 생성).
- **엔진 묶음 main 병합**(`ca1f30e7`, digest `99bbef91…`), gate α·β PASS, γ는 결정 22로 관측자 효과 특성화 종결. 계측 정책: 판정 측정은 전 arm 계측 OFF. telemetry 비대칭 감사 r1: M3 J1만 비대칭(`results/engine_gate_prep/AUDIT_telemetry_asymmetry_past_results_2026-10-21.md`).
- `test_baseline_pass` 비결정성 = 가짜 엔진 결함 2건 수리(`1943b3b6`). presubmit 줄 인용 드리프트 수리(`10e33bfa`); reachability 2건 차단 남음.
- 보고용 그림 6장 `reports/figures/current_2026-10-21/`(Claim G·HE0·layer-type·CM-2·CM-5·관측자 효과).
- 설계 결정 원문 1–30: `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`.
## 다음 행동
1. 미복구 설정 제약 점 반영(결정 후) → **하네스층 감사**(claims-auditor, 최대 2회; A-3…A-41 해석 포함)
2. **Workspace smoke 실행 승인** → 실제 ramp로 `RAMP_SIGMA`·shape C 셀·J-B2 하드캡 재설정 → 봉인 S1 → preflight → 제출 승인
3. G0 PREREG(헤드라인 셀 전 arm 계측 OFF, 조성 변형 goodput 범위 = 결정 25) → V PREREG(oracle 격차 측정 추가 검토) → M3 재개 여부
## 사용자 결정 대기
- **§11-4 미복구 설정 제약 점** / 결정 30 원칙상 기본은 복구 / J-C bs48 (2k,8k) 사유 없는 삭제(+1.0) · J-A T1 (8,64k) context 131,072(+0.44) · Llama 장문 T2(+0.35) · J-C 앵커 cap 32(+0.78) · TP2 장문 PD-mux 108u(+0.39) — 전부 복구(권고, 합계 약 22.4 GPU-h [유도]) · shape C는 smoke 후
- **presubmit reachability 차단 2건** / M4R·TC1 rev3 spec이 active로 전역 차단(측정과 무관) / 범위 한정 OVERRIDE + 승인(권고, 전례 `cp_baseline/OVERRIDE_*`) · 두 spec 폐기
- **로컬 엔진 트리 재동기화** / 구판이라 로컬 테스트 5건 실패 / amd·opchar 영향 확인 뒤 `install_runtime.sh` 절차(권고) · scratch 사본 유지
- **임시 디렉터리 `/tmp/gate_t_*` 11개 삭제** / 테스트 진단 잔여물, 에이전트 삭제가 막힘 / 삭제(권고)
- **논문 방향** / static 결론만으로는 제안이 아님 / ② 비용 모델 기반 static 설정(권고, V + oracle 격차) · ① 특성화 · ③ 메커니즘
- LongBench v2·InfiniteBench 보류(결정 12′) · 기존 유지: D-10 · radix OFF · 장문 hold-out · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음. Workspace 둘 다 standby(조성 프로브 후 정지 확인).
## 금지 서술 (링크만)
128k 프로브 감사 금지 문안 · gate 감사 r1/r2 인용 금지 문장 · telemetry 비대칭 감사 §3(6건) · PREREG §9-2(T1 명목 ctx·장문 경계 시험 혼동 포함) · "슬롯 상한이 hybrid 고유" · M3 판정서 문안 · J1 크기 문장
## 최근 로그
- 2026-10-22~23: 정본 수식 고정(결정 27)·축 지도 §6·§8 · 하네스 1·2단계, rev3.4·3.5(결정 28–30) · 조성 프로브 PASS · test_baseline_pass 수리
- 2026-10-21: 결정 22–26, engsync 병합, telemetry 감사 r1, multigpu·engsync worktree 폐기, promote 4차, 보고 그림 6장
- 2026-10-21: 관측자 효과 분해 Job(TEL λ −1.7%, I-4 ITL +1.0%)
- 2026-10-20: 결정 14(비용 모델 주 목적) · PREREG rev0–rev3 · 엔진 묶음 gate α·β PASS · 128k 프로브(int32 원인)
- 2026-10-19: Claim E 첫 화살표 감사 · headline G 결정 · J1 분석 · promote 3차
