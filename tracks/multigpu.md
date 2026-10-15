# TRACK: multigpu — A100 ×1/2/4/8 DP·TP 확장   (worktree `prefill-layer-alloc-multigpu`, branch `multigpu-scaling`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 main의 promote.

## 목표
PD-mux 이득의 N-스케일링(1차 estimand G_sym = λ*_agnostic / λ*_fused_sym) + multi-GPU 고유 기전.
## cost model 기여
- 채우는 칸: E 축 확장(N, TP) — TP 통신 항이 CM-1에 더해지는 형태, GPU당 decode SM 요구 변화
- 이번에 알게 된 것: 없음(M2g는 정확성 게이트, 성능 판정 0)
- promote 필요: 아니오
## 현재 상태
- **사용자 결정 F3·F4(2026-10-15)**: main의 lockstep PD-mux 구조가 그대로 이 트랙의 multi-GPU PD-mux 기판. true dual worker는 AMD 전용. 단일 GPU 결과는 lockstep 기판 범위.
- M2g 정확성 게이트: J1·J2 통과, J4·J8 하네스 수리(a41e725) 후 재실행으로 TP4·TP8 4 arm 포함. PASS 표 4개 봉인(99ed704).
- TP4·TP8 G2 `TP_NUMERIC_DIVERGENT`는 결함 판정 아님 — M3 TP 결과 옆 병기 의무(MGH-3, C4).
- **main merge 완료(2026-10-15, main `dc06c870`, 조건부 가 — `results/handoff_review_2026-10-15/VERDICT_spec_v0_1_and_closure_round2_2026-10-15.md` §6)**. 이 branch는 main으로 fast-forward(`dc06c870`). 뒤처리 확인: 공유 dev tree `sglang_engine_dev_multigpu`의 ENGINE_DIGEST 재대조 match(`82a5fc5b…`, §6-2), merge 이후 main 쪽 엔진 src 변경 0, multigpu 관련 시험 11모듈 411건 OK(사본 트리에서, 실행 후 digest 불변). 회귀 0건 문장은 쓰지 않는다(C-4).
- **M3 경로가 바뀌었다**: merge로 main의 `job_entry.sh` 로그 미러(tee 파이프)·heartbeat(300 s마다 `find`)가 들어왔다 → 측정 중 상주 프로세스 추가. 이전 "M3 경로 무변경(313be0a 이후 a41e725)" 확인은 더 이상 충분하지 않다. 셸 문법·launcher 시험만 통과, VESSL 실행 0.
- main의 `sglang_engine_dev`(main dev tree)는 merge src 미동기화(`multigpu_guard.py` 없음) — main의 엔진 트리 통일 몫, 이 트랙은 손대지 않음.
## 다음 행동
1. M3는 main의 Q23(M3 PREREG 개정)·D-15(1차 분모) 결정 뒤에. 개정에 `PDMUX_LOG_MIRROR`/`PDMUX_HEARTBEAT_S` 값 등록(권고: 측정 Job은 `PDMUX_HEARTBEAT_S=0`)을 포함
1-b. 첫 GPU Job 전에 §6-1 preflight: VESSL에서 CL-3 미검증 시험을 전(`7b15214`)·후(merge) 짝으로 (id, 서명) 대조, L2 밖 신규 실패 시 중단 → 이후 dry-run → 승인 → 직렬 제출
2. 이후 `mg_label.py` 결합 → result-analyst → claims-auditor 결과 감사
## 사용자 결정 대기
- **M3 PREREG 개정 여부** (main Q23으로 이관, D-15 선행) — 연구 감사 의견(main 핸드오프 (D)절): shape C 부재·`max_prefill_tokens` 미등록·N 간 비교엔 절대 SLO 필요·TP에서 GPU당 decode SM 요구 변화 / 현행 M3 제출 · PREREG 개정 후 재감사(권고: 개정 범위만 먼저 결정)
- **M3 제출 승인** — 모형 예보 예약 약 124 GPU-h, 최장 약 9 h
## 실행 중·미회수 Job
- 없음(M2g J4·J8 재실행 회수 완료)
## 금지 서술 (링크만)
`workspace/engine-port/results/multigpu/PREREG_MULTIGPU_2026-10-12.md` §7·§15–§18, 각 `VERDICT_*.md`
## 최근 로그
- 2026-10-15: main merge(`dc06c870`) 뒤처리 — branch FF, digest 재대조, 사본 트리 시험 411 OK, M3 경로 변경(job_entry 로그 미러·heartbeat) 식별
- 2026-10-13: M2g J4·J8 재실행 회수, PASS 표 4개 봉인
- 2026-10-12: M1+V0 프로브, PREREG rev11·하네스 봉인
