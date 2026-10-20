# TRACK: main — anchor 사다리 / 공통 측정 명세 / G0 + multi-GPU M3(이양)   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
anchor 주장 1(generic 예측기가 hybrid에서 다른 결정을 내리는가)의 사다리. 단일 GPU(lockstep 기판)에서 측정 명세·기판 특성화를 세우고 같은 구조를 multi-GPU PD-mux로 확장한다(F4). 2026-10-15부터 multigpu 트랙(M3, N-스케일링 G_sym)도 이 트랙이 맡는다.
## cost model 기여
- 채우는 칸: CM-1(`s_attn` vs `s_ssm`), CM-4(`δ(p,d)` + lockstep span 정지), CM-5(장문 KV vs mamba 슬롯, cap), (확장) N·TP 통신 항
- 이번에 알게 된 것: 판정 0(M3 미완). 사실만 — J1 GPU0 샘플 61%가 기본 400 W 한도 throttle(0x4, 클럭 중앙 1380 MHz, 59 °C); 하네스 시간 모형이 확장 셀을 빠뜨려 rev12 예산 190 GPU-h가 과소(J1 실측 보정 시 J_N 20–21 h > 캡 16 h)
- promote 필요: 예 (결정) — 결정 10–12′(대조 모델·fused_tuned·shape C·long-ctx 입력); 아니오(이전 항목 반영 완료) — 2026-10-19 반영 완료: J1(N=1) 분석·감사 허용 문안 1–5(COSTMODEL CM-2·4·6, CONSENSUS §3 항목296), Claim E 첫 화살표 판정(CM-1 정정, §6 등급 불변), 사용자 결정 1–7·8′(CM-5, PROJECT_STATUS 최상단 5줄), preflight PASS·rev12 봉인·M3 J2/J4/J8 외부 종료·예산 모형 과소 교훈(확장 행)
## 현재 상태 (2026-10-20)
- **128k 프로브 1·2회차** `workspace/engine-port/results/cap_char/probe_128k/RESULT_probe_128k_2026-10-20.md`: 0.82에서는 chunk -1 활성화 OOM(PD-mux 64k). **0.6에서는 PD-mux·fused_sym이 64k OK, 96k에서 Triton SSD `illegal memory access`**(메모리 아닌 커널 한계 추정) → 이 스택에서 PD-mux 상한 64k. chunked 8192·Llama는 124k까지 OK. G4·G7 미실행(단계 순서). 기술값, 인용 금지. 운영 사고(1회차 폴링 오판, A100 약 60분 유휴)는 SUMMARY.md 완료 판정으로 수리.
- **J1(N=1) 분석** `workspace/engine-port/results/multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md` + 감사 `AUDIT_J1_N1_analysis_2026-10-19.md`: CM-2 floor(9 ms + 0.111 ms/토큰×L) CONFIRMED · CM-6 처리율↔ITL 역순·CM-4 클럭 반대 **REFUTED**(HI 과부하·achieved 상이·미조율 fused 노브; 같은 achieved LO에선 요청 ITL p95 통과율 동일) · xcheck 원인 PLAUSIBLE·규칙 (b) 결함 CONFIRMED · throttle X HI 80–99% · G0에 튜닝(mixed-chunk) fused arm 포함 검토.
- **(다) 허수아비 점검**(venue-strategist): 엔진 층 Q18은 허수아비(SGLang·vLLM이 state 예약·admission 반영), Bullet/MuxWise는 오예측이 아니라 변수 부재 → 권고 문안 = 후보2(cap 결합 특성화) 본문 + 후보1(이식 시 빠진 변수가 슬롯 상한) anchor, 실험은 R16-3에 거절 사유 계측+cap×ctx 스윕.
- **사용자 결정 2026-10-19** `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`: headline = Claim G 중심 · shape C 공통 워크로드 편입 · E를 memory/admission 예측까지 넓힐 필요 인정 · 후속: G0에 순수 attention 대조 모델 편입, shape C = G0 두 번째 표준 셀, M3는 G0 뒤로 연기(J1 분석만 먼저), E = 열린 질문 + β 계측 · 대조 모델 Llama-3.1-8B·fused_tuned arm 추가(결정 10–11) · shape C 재정의: LongProc 제외, 출력 강제 {자연,1k,4k}(결정 12); long-ctx 통제 축 = 합성 고정 길이, 실제 분포 = LongBench v1 원본, v2·InfiniteBench 보류(결정 12′) · (다) E′ 문안 = 결정 8′(고정 state s vs KV 양 구간 특성화, cap≡슬롯은 설정 공시) · long-ctx 블록 유지.
- **Claim E 첫 화살표 감사(claims-auditor 1회차)** `workspace/engine-port/results/anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md`: 10-12 의견 "첫 화살표에서 끊긴다" = 반증 아닌 **부재**(decode per-type 유효 측정 0; Diff B·C2·opchar는 s_attn≈s_ssm 근거 아님) · a는 E의 필요조건 아님(REFUTED; 우회 α 조성 오귀속·β Q18 state 예약) · A4 K2 hold-out 필요 CONFIRMED · shape B ctx ~1% NOT-YET-SUPPORTED. 우선순위 전제는 "E 미검증, hold-out 서빙 필요·고비용"까지만. COSTMODEL:34-37 문안 정정 대상(promote).
- **M3 rev12**: 규칙층 감사 2 + 하네스층 감사 2 → 봉인 `e43749c`, 재발행 1 `9c3ea47`. merge preflight(CPU, 3회차) **PASS** `results/multigpu/PREFLIGHT_PASS.json`(POST `2aaf2aa`, digest `82a5fc5b`). 판정서 `results/multigpu_verdicts/VERDICT_preflight_judge_r2_and_reseal_2026-10-17.md`, 결과 `PREFLIGHT_RESULT_2026-10-18.md`.
- **M3 실행**: J1(N=1) 완주 104셀. J2·J4·J8은 2026-10-18 21:14 UTC **외부 종료**(크레딧 소진 추정) — DP 어느 config도 미완(fused_sym 또는 agnostic boot2 누락), TP 미시작 → **G_sym 전 N 미정의**. 원시 회수 `results/multigpu/manual_retrieval/<job>/`(git-ignore, 17 GB, 원격·로컬 sha 일치, Job DONE 없음 — J1도 object volume 내보내기 절단).
- **xcheck**: N1(J1)·DP2(J2) FAIL(doc 2693 첫 16토큰; fused_sym 자체 동시≠순차), DP4·DP8 PASS → X shape 방향 라벨 전부 인용 불가(P9-a). "요청마다 logprob 32개"는 실측 확인.
- **축소 범위 계산** `results/multigpu_verdicts/M3_SCOPE_OPTIONS_2026-10-19.md`(미커밋→이번 커밋): S2f(TP만)는 DP 미완이라 불가 → S1f(3-arm A·B, DP+TP 한 Job, N=2·4·8, J1 재사용) 약 128 GPU-h(94–138), 최장 ~9.5 h [유도]. 2-arm은 LO rung이 fused 기준이라 G_sym 정의가 깨짐.
- 인프라: `job_entry.sh` CPU 모드(`68f386b`, heredoc 회귀 수정 `2aaf2aa` — 주석에 `<<'PY'` 쓰지 말 것), `fetch.sh` pycache 제외, `launch.sh`는 추적 파일 미커밋이면 제출 거부(Job 원장도 매 제출 뒤 커밋), 로컬 `sglang_engine_dev`는 merge 트리로 동기화(amd·opchar 공지 `handoff-report/NOTICE_engine_tree_sync_2026-10-15.md`).
- 공통 측정 명세 v0.2+ `results/handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`(G0 입력), 결정 기록 `DECISIONS_Q1_consolidated_2026-10-15.md` §1–§8.
## 다음 행동
1. I-5 거절 사유 계측: CPU 구현·테스트 완료(`src/multiplex/admission_probe.py`, 사유 KV/cap≡슬롯 2분류) → 엔진 트리 동기화(ENGINE_DIGEST 변경) + GPU correctness gate — cap·long-ctx·G0 공용 선행
2. 설계 v1.1 `workspace/engine-port/results/cap_char/DESIGN_cap_longctx_v1_2026-10-19.md`(cap 블록 + long-ctx 블록, L*≈8.8k 단위 해소) §5 결정 → 128k 프로브 → PREREG → 감사 → 봉인 → preflight → 제출
3. 128k 프로브 스크립트(G1–G7) 작성·CPU 회귀 → 승인 → Llama 적재 + A100 프로브
4. G0 PREREG(cap = cap 특성화 결과, λ*_SLO 판정, arm별 prior·확장 발화 기본 시나리오 — J1 분석 반영) · M3는 G0 뒤 재결정
## 사용자 결정 대기
- **long-ctx 블록 상한·mem-fraction** / PD-mux가 96k부터 커널 실패, 0.82는 64k OOM / 상한 64k + mem-fraction 0.6 전 arm 통일(권고) · SSD 커널 원인 조사(engine-porter) 후 재시도 · 3회차 짧은 프로브로 G4·G7(≤64k) 먼저
- **LongBench v2·InfiniteBench 사용(공식 가운데 절단)** / 보류(결정 12′) / 프로브 뒤 재논의
- 기존 유지: D-10 확정(G0 PREREG 전) · radix(OFF+편차 공시) · 장문 hold-out 모델 · CONSENSUS §1 행 38
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
