# TRACK: main — anchor 사다리 / 공통 측정 명세 / G0 + multi-GPU M3(이양)   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
anchor 주장 1(generic 예측기가 hybrid에서 다른 결정을 내리는가)의 사다리. 단일 GPU(lockstep 기판)에서 측정 명세·기판 특성화를 세우고 같은 구조를 multi-GPU PD-mux로 확장한다(F4). 2026-10-15부터 multigpu 트랙(M3, N-스케일링 G_sym)도 이 트랙이 맡는다.
## cost model 기여
- 채우는 칸: CM-1(`s_attn` vs `s_ssm`), CM-4(`δ(p,d)` + lockstep span 정지), CM-5(장문 KV vs mamba 슬롯, cap), (확장) N·TP 통신 항
- 이번에 알게 된 것: 판정 0(M3 미완). 사실만 — J1 GPU0 샘플 61%가 기본 400 W 한도 throttle(0x4, 클럭 중앙 1380 MHz, 59 °C); 하네스 시간 모형이 확장 셀을 빠뜨려 rev12 예산 190 GPU-h가 과소(J1 실측 보정 시 J_N 20–21 h > 캡 16 h)
- promote 필요: 예 — J1(N=1) 분석 `workspace/engine-port/results/multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md`(CM-6 처리율↔ITL 역순, CM-2 무경합 prefill ~9.0k tok/s, CM-4 클럭 가정 shape 의존); 예 (결정) — headline G·shape C 편입·E 확장 필요 인정(`workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`); 예 (감사) — Claim E 첫 화살표 판정(아래 현재 상태 첫 줄); 예 — merge preflight `PREFLIGHT_PASS`(caveat C-A1–C-B3), rev12 봉인·재발행 1, M3 J2·J4·J8 외부 종료(전 N G_sym 미정의), 예산 모형 과소 교훈
## 현재 상태 (2026-10-19)
- **J1(N=1) 분석** `workspace/engine-port/results/multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md`: 12/12 KNEE_BRACKETED, 비는 기술값만(크기 문장 금지) · A는 분할 미실행(PARTITION_NOT_EXERCISED) · B·X는 처리율 fused>PD-mux인데 ITL p99는 역순 → G0는 λ\*_SLO 판정 필수 · xcheck 2693 = 수치 비결정성(규칙 (b) 설계 결함, rev13 몫) · throttle은 shape 의존 매개 성분(B·X 비 SUBSTRATE_COVARIES) · G0: arm별 prior, 확장 발화 기본 시나리오.
- **(다) 허수아비 점검**(venue-strategist): 엔진 층 Q18은 허수아비(SGLang·vLLM이 state 예약·admission 반영), Bullet/MuxWise는 오예측이 아니라 변수 부재 → 권고 문안 = 후보2(cap 결합 특성화) 본문 + 후보1(이식 시 빠진 변수가 슬롯 상한) anchor, 실험은 R16-3에 거절 사유 계측+cap×ctx 스윕.
- **사용자 결정 2026-10-19** `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`: headline = Claim G 중심 · shape C 공통 워크로드 편입 · E를 memory/admission 예측까지 넓힐 필요 인정 · 후속: G0에 순수 attention 대조 모델 편입, shape C = G0 두 번째 표준 셀, M3는 G0 뒤로 연기(J1 분석만 먼저), E = 열린 질문 + β 계측 · (다) E′ 문안 = 후보2 본문 + 후보1 anchor(결정 8).
- **Claim E 첫 화살표 감사(claims-auditor 1회차)** `workspace/engine-port/results/anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md`: 10-12 의견 "첫 화살표에서 끊긴다" = 반증 아닌 **부재**(decode per-type 유효 측정 0; Diff B·C2·opchar는 s_attn≈s_ssm 근거 아님) · a는 E의 필요조건 아님(REFUTED; 우회 α 조성 오귀속·β Q18 state 예약) · A4 K2 hold-out 필요 CONFIRMED · shape B ctx ~1% NOT-YET-SUPPORTED. 우선순위 전제는 "E 미검증, hold-out 서빙 필요·고비용"까지만. COSTMODEL:34-37 문안 정정 대상(promote).
- **M3 rev12**: 규칙층 감사 2 + 하네스층 감사 2 → 봉인 `e43749c`, 재발행 1 `9c3ea47`. merge preflight(CPU, 3회차) **PASS** `results/multigpu/PREFLIGHT_PASS.json`(POST `2aaf2aa`, digest `82a5fc5b`). 판정서 `results/multigpu_verdicts/VERDICT_preflight_judge_r2_and_reseal_2026-10-17.md`, 결과 `PREFLIGHT_RESULT_2026-10-18.md`.
- **M3 실행**: J1(N=1) 완주 104셀. J2·J4·J8은 2026-10-18 21:14 UTC **외부 종료**(크레딧 소진 추정) — DP 어느 config도 미완(fused_sym 또는 agnostic boot2 누락), TP 미시작 → **G_sym 전 N 미정의**. 원시 회수 `results/multigpu/manual_retrieval/<job>/`(git-ignore, 17 GB, 원격·로컬 sha 일치, Job DONE 없음 — J1도 object volume 내보내기 절단).
- **xcheck**: N1(J1)·DP2(J2) FAIL(doc 2693 첫 16토큰; fused_sym 자체 동시≠순차), DP4·DP8 PASS → X shape 방향 라벨 전부 인용 불가(P9-a). "요청마다 logprob 32개"는 실측 확인.
- **축소 범위 계산** `results/multigpu_verdicts/M3_SCOPE_OPTIONS_2026-10-19.md`(미커밋→이번 커밋): S2f(TP만)는 DP 미완이라 불가 → S1f(3-arm A·B, DP+TP 한 Job, N=2·4·8, J1 재사용) 약 128 GPU-h(94–138), 최장 ~9.5 h [유도]. 2-arm은 LO rung이 fused 기준이라 G_sym 정의가 깨짐.
- 인프라: `job_entry.sh` CPU 모드(`68f386b`, heredoc 회귀 수정 `2aaf2aa` — 주석에 `<<'PY'` 쓰지 말 것), `fetch.sh` pycache 제외, `launch.sh`는 추적 파일 미커밋이면 제출 거부(Job 원장도 매 제출 뒤 커밋), 로컬 `sglang_engine_dev`는 merge 트리로 동기화(amd·opchar 공지 `handoff-report/NOTICE_engine_tree_sync_2026-10-15.md`).
- 공통 측정 명세 v0.2+ `results/handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`(G0 입력), 결정 기록 `DECISIONS_Q1_consolidated_2026-10-15.md` §1–§8.
## 다음 행동
1. I-5 거절 사유 계측(engine-porter 조사·구현 중, CPU) → GPU correctness gate — cap 특성화·G0 공용 선행
2. cap 특성화 설계 초안 `workspace/engine-port/results/cap_char/DESIGN_cap_characterization_v0_2026-10-19.md` §6 열린 질문 5개 결정 → PREREG → 감사 → 봉인 → preflight → 제출
3. shape C 데이터셋·attention 대조 모델 확정(cap 특성화·G0 공용, 한 번만)
4. G0 PREREG(cap = cap 특성화 결과, λ*_SLO 판정, arm별 prior·확장 발화 기본 시나리오 — J1 분석 반영) · M3는 G0 뒤 재결정
## 사용자 결정 대기
- **cap 특성화 열린 질문 5개(설계 §6: shape C 선행 여부·128k/context_length·경계 arm·radix ON 비율 arm 제외·대조 교락)** / PREREG 전 / 설계 초안 권고안
- **shape C 데이터셋·구체값, 대조 모델 확정** / 편입 결정됨 / G0 PREREG에서 정함 · M3는 G0 뒤 재개 여부 재결정
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
