# TRACK: main — anchor 사다리 / 공통 측정 명세 / G0 + multi-GPU M3(이양)   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
anchor 주장 1(generic 예측기가 hybrid에서 다른 결정을 내리는가)의 사다리. 단일 GPU(lockstep 기판)에서 측정 명세·기판 특성화를 세우고 같은 구조를 multi-GPU PD-mux로 확장한다(F4). 2026-10-15부터 multigpu 트랙(M3, N-스케일링 G_sym)도 이 트랙이 맡는다.
## cost model 기여
- 채우는 칸: CM-1(`s_attn` vs `s_ssm`), CM-4(`δ(p,d)` + lockstep span 정지), CM-5(장문 KV vs mamba 슬롯, cap), (확장) N·TP 통신 항
- 이번에 알게 된 것: 판정 0(M3 미완). 사실만 — J1 GPU0 샘플 61%가 기본 400 W 한도 throttle(0x4, 클럭 중앙 1380 MHz, 59 °C); 하네스 시간 모형이 확장 셀을 빠뜨려 rev12 예산 190 GPU-h가 과소(J1 실측 보정 시 J_N 20–21 h > 캡 16 h)
- promote 필요: 예 — merge preflight `PREFLIGHT_PASS`(caveat C-A1–C-B3), rev12 봉인·재발행 1, M3 J2·J4·J8 외부 종료(전 N G_sym 미정의), 예산 모형 과소 교훈
## 현재 상태 (2026-10-19)
- **M3 rev12**: 규칙층 감사 2 + 하네스층 감사 2 → 봉인 `e43749c`, 재발행 1 `9c3ea47`. merge preflight(CPU, 3회차) **PASS** `results/multigpu/PREFLIGHT_PASS.json`(POST `2aaf2aa`, digest `82a5fc5b`). 판정서 `results/multigpu_verdicts/VERDICT_preflight_judge_r2_and_reseal_2026-10-17.md`, 결과 `PREFLIGHT_RESULT_2026-10-18.md`.
- **M3 실행**: J1(N=1) 완주 104셀. J2·J4·J8은 2026-10-18 21:14 UTC **외부 종료**(크레딧 소진 추정) — DP 어느 config도 미완(fused_sym 또는 agnostic boot2 누락), TP 미시작 → **G_sym 전 N 미정의**. 원시 회수 `results/multigpu/manual_retrieval/<job>/`(git-ignore, 17 GB, 원격·로컬 sha 일치, Job DONE 없음 — J1도 object volume 내보내기 절단).
- **xcheck**: N1(J1)·DP2(J2) FAIL(doc 2693 첫 16토큰; fused_sym 자체 동시≠순차), DP4·DP8 PASS → X shape 방향 라벨 전부 인용 불가(P9-a). "요청마다 logprob 32개"는 실측 확인.
- **축소 범위 계산** `results/multigpu_verdicts/M3_SCOPE_OPTIONS_2026-10-19.md`(미커밋→이번 커밋): S2f(TP만)는 DP 미완이라 불가 → S1f(3-arm A·B, DP+TP 한 Job, N=2·4·8, J1 재사용) 약 128 GPU-h(94–138), 최장 ~9.5 h [유도]. 2-arm은 LO rung이 fused 기준이라 G_sym 정의가 깨짐.
- 인프라: `job_entry.sh` CPU 모드(`68f386b`, heredoc 회귀 수정 `2aaf2aa` — 주석에 `<<'PY'` 쓰지 말 것), `fetch.sh` pycache 제외, `launch.sh`는 추적 파일 미커밋이면 제출 거부(Job 원장도 매 제출 뒤 커밋), 로컬 `sglang_engine_dev`는 merge 트리로 동기화(amd·opchar 공지 `handoff-report/NOTICE_engine_tree_sync_2026-10-15.md`).
- 공통 측정 명세 v0.2+ `results/handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`(G0 입력), 결정 기록 `DECISIONS_Q1_consolidated_2026-10-15.md` §1–§8.
## 다음 행동
1. 사용자가 "power cap보다 우선 처리할 논증점이 있다"고 함 — 다음 세션 첫 질문으로 무엇인지 특정
2. M3 계속 여부·범위 결정 → (계속 시) rev13(범위 축소·외부 종료 처리 규칙·내보내기 절단 수정) → 규칙·하네스 감사 → 재봉인 → CPU preflight → 제출
3. J1(N=1) 결과 분석(result-analyst, CPU): N=1 G_sym, xcheck doc 2693 원인, throttle 공시 — 크기 문장 금지
4. G0: cap 특성화 Job(R16-3, 결정 규칙 사전등록) → G0 계측 패치 I-1–I-5 + mixed-chunk 정확성 gate → G0 PREREG
## 사용자 결정 대기
- **우선 논증점** / power cap은 핵심이 아니라는 사용자 판단 / 다음 세션에서 무엇을 먼저 할지 특정
- **M3 계속 여부·범위** / J2·J4·J8 전부 무효, 예산 모형 과소 / S1f(~128 GPU-h) · 더 축소(N=8 제외 시 추세점 3개 위험) · 보류
- **J1 결과 분석 착수** / 재실행과 무관하게 유효, CPU 작업 / 착수(권고) · 보류
- 기존 유지: D-10 확정(G0 PREREG 전) · radix(OFF+편차 공시) · 128k 프로브 · 장문 hold-out 모델 · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음. 원장 `results/multigpu/vessl_jobs.tsv`, `results/preflight_cl3/vessl_jobs.tsv`. Workspace 둘 다 standby.
## 금지 서술 (링크만)
판정서(`results/handoff_review_2026-10-15/VERDICT_*`, `results/multigpu/VERDICT_m3_rev12_*`, `results/multigpu_verdicts/VERDICT_*`)의 S·K·C·CL·MG12·MGH12·C-A·C-B 문안 · "preflight PASS가 M3 실행 바이트 보증" · M3 크기 인용(미완) · "회귀 0건" · "Sarathi식" · `results/multigpu/` 안에 `VERDICT_*.md` 추가 금지(봉인 glob)
## 최근 로그
- 2026-10-17–19: preflight CPU 3회(FAIL_NEW 2회 → 판정기 r2·봉인 재발행·heredoc 수정 → PASS) · M3 J1 완주, J2·J4·J8 외부 종료 · 수동 회수 · 축소 범위 계산 · power cap vs throttle 정리
- 2026-10-16: M3 rev12 규칙층·하네스층 감사 각 2회, 하네스 구현·수리, 봉인 `e43749c`
- 2026-10-15: Q1 N2·N3·E2 종료, 명세 v0→v0.2(감사 2), merge `dc06c870`, multigpu 이양, promote ×2
- 2026-10-14: 주 모델 Nano-9B-v2 결정; 장문 워크로드 조사
- 2026-10-13: Job-EA 회수·결과 감사·정본 rev95; 하네스 재구성
