# M3 사전등록 비교 — rev11(봉인) 대 rev12 DRAFT r2 (2026-10-16)

> 대상: rev11 = `PREREG_MULTIGPU_2026-10-12.md`(봉인 `313be0a`, sha256 앞 16자리 `8b617d3d59d6cb63`), rev12 = `PREREG_MULTIGPU_rev12_DRAFT_2026-10-15.md` r2(미봉인·미커밋).
> 근거 판정서: VR1 = `VERDICT_m3_rev12_draft_r1_2026-10-15.md`, VR2 = `VERDICT_m3_rev12_draft_r2_2026-10-16.md`. 사용자 결정: `../handoff_review_2026-10-15/DECISIONS_Q1_consolidated_2026-10-15.md` §4–§6.
> 지위: 사용자용 비교 요약이다. 판정 규칙의 정본은 rev12 초안 §19이고, 이 문서와 다르면 초안이 우선이다. 측정 0, GPU 사용 0, 성능 판정 0.
> 라벨: [측정]·[유도]·[결정]·[제안]·[코드]. 판정서의 반전 시험 수치는 **[모형 예보]**이며 측정값이 아니다. 모든 수치는 VESSL A100 SXM 80GB(108 SM) 기판 범위다.

## 1. 요약

M3는 multi-GPU(TP·DP, N ∈ {1, 2, 4, 8})에서 PD-mux(agnostic)가 fused 대칭 기준선(fused\_sym)보다 처리율 용량을 얼마나 더 내는지, 즉 G\_sym(N, mode, shape) = λ\*\_thr,agnostic / λ\*\_thr,fused\_sym과 그 N-추세를 같은 Job 안에서 잰다. **측정 질문과 1차 estimand는 rev11과 같다.** rev12가 바꾸는 것은 질문이 아니라 측정 범위와 셀 유효성 조건이다. 장문 요약 shape X(arXiv-Sum 16k, 실제 문서 토큰)를 더하고, 그에 필요한 서버 인자(context 20480 등)를 명시하며, 오류를 서버·클라이언트 측으로 나눠 기록한다(무효 문턱은 rev11과 같은 합산 1%). 절대 SLO용 floor 블록, X의 Job 효과 앵커, preflight 제출 gate도 새로 들어간다. 그 대가로 예산과 하드캡이 커졌다.

## 2. 비교 표

| 항목 | rev11(기존) | rev12(이번) | 바뀐 이유(근거 문서) |
|---|---|---|---|
| 측정 질문 | multi-GPU에서 PD-mux 기능 전체가 fused\_sym 대비 처리율 용량을 늘리는가, 그 N-추세 | **같음** | 질문은 바꾸지 않음(rev12 §19-1, F6 [결정]) |
| 1차 estimand | G\_sym(N, mode, shape) = λ\*\_thr,agnostic / λ\*\_thr,fused\_sym, seed 짝 bootstrap CI, log₂N 추세, 3% 크기 규칙 | **같음**. 셀 유효성·boot 사전 조건 세 곳만 바뀜(오류 분류, floor 블록, dtype 명시) | VR1 Q13·R-7, rev12 §19-1 |
| shape | A(256/512)·B(8192/64), 입력 `[1]*L` 합성 | A·B(바이트 불변) + **X = arXiv-Sum 16k**(입력 ≤ 16,384, 출력 = abstract 길이, `ignore_eos`, **실제 문서 토큰**, seed별 순열 prefix 표본) | SP §10·Q3 [결정], R-8 실제 토큰 [결정, DECISIONS §5] |
| 셀 수 | 368 [유도](본 336 + 앵커 24 + GPU 편차 8) | **560** [유도](본 504 + 앵커 48 + GPU 편차 8) + floor 블록 20 | shape X 추가, U14 X 앵커 [결정, DECISIONS §5] |
| boot 수 | 70 [유도] | 70 [유도](불변; X 앵커 셀은 기존 앵커 boot 안) | rev12 §19-2 셀 구성 |
| `context_length` | 16384 | **20480** [결정, U2 = (a)] | X 최대 입력+출력 16,744 → ctx ≥ 16,746, floor 16k → ctx ≥ 16,394 [유도, VR2 F-6]. U2(b)는 실행 불가(VR1 §4-3) |
| 기타 서버 인자 | `max_prefill_tokens` 엔진 기본(16384), span 예산은 config 파일 값, mamba cache·`mamba_ssm_dtype` 자동 | **명시 등록**: `--max-prefill-tokens` 16384, span 예산 65536(agnostic·d44, config sha 대조), `--max-mamba-cache-size` = effective cap, `--mamba-ssm-dtype` = 고정 revision config 키(없으면 float32, `torch_dtype` 폴백 금지) | SP §0·R13-1·R16-1, R-4. dtype boot 기록은 인자 기록이고 실효값은 부팅 로그로 대조, 없으면 `DTYPE_UNVERIFIED`(VR2 F-4) |
| 오류 처리(셀 무효 문턱) | 모든 오류 / 셀 요청 수 > 1% → `CLIENT_ERROR`, 짝 무효 | (서버 + 클라이언트) 오류 / 셀 요청 수 > 1% → 짝 무효(**합산 1%**, rev11과 같은 상한). 라벨은 우세 분류(`SERVER_ERROR_HI`/`CLIENT_ERROR`), 두 건수 기록. 확장·앵커·J1 기준 셀 포함 | D-1(VR1)·D-1′(VR2) → (a) [결정, DECISIONS §6]. 합산 규칙 두 arm 최악 G 편향 +2.21% [모형 예보, VR2 §2] |
| 오류 분류 기록 | 분류 없음. 서버 거절이 HTTP 200 + error 청크로 오면 1토큰 성공으로 기록될 수 있었음 [코드, VR1 §4-1] | 공용 loadgen `0eca947`: `SSE_ERROR`·`SERVER_ABORT`·`OUTPUT_SHORT`·`OUTPUT_LONG` = 서버 측, timeout = `CLIENT_EXCEPTION`(클라이언트 측). 조용한 1토큰 성공 차단, runner·`mg_boot` context 사전검사 | R-2 [코드, CPU 시험 19건; 서빙 실측 0], VR2 F-2·F-3 |
| SLO(M3) | TTFT 3 s ∧ ITL p95 50 ms 실패 분해(기술값) | **N=1 floor 기준 절대 T**: T\_i = k·floor\_{N=1}(L\_i), k ∈ {3, 5, 10}, ITL p95 ≤ 50 ms. **기술값**(판정은 λ\*\_thr뿐), 봉인 밖 분석기 | SP R6·R10, D-9 [결정], rev12 §19-4·§19-9 |
| SLO(M4) | TTFT ≤ 3 s ∧ ITL p95 ≤ 50 ms | TTFT ≤ T\_i(A′, k ∈ {3, 5, 10}, 그 M4 Job의 N=1 floor) ∧ ITL p95 ≤ 50 ms | A-13 ① (rev12 §19-4). 세부는 M4 단계 감사(MG12-8) |
| floor 블록 | 없음 | **신설 20블록** [유도]: J1 2 + 앵커 2×3 + config boot1 2×6, 두 arm(fused\_sym·agnostic) 대칭, 실제 문서 절단 prompt {1k…16k} × 16회 | SP R6-2·R10-2, R-6 채택(MG12-6 비적용) |
| Job 효과 검사 | shape A 앵커만(J\_N 앵커 boot, GPU 0) | **A + X 앵커**(X HI 4 seed, 1.6×prior\_X(1) rung, 확장 없음). X 기준 셀 포화 미달이면 전 J\_N `JOB_EFFECT_UNTESTED(J_N, X)` | U14 [결정, DECISIONS §5], VR2 F-5 |
| P10 shape | 2(A, B) | **3(A, B, X)**, 어느 shape든 BROKEN이면 MGP-13. X는 BROKEN 판정에서 비앵커 shape [제안] | rev12 §19-2. 귀무 BROKEN 상한 29–33% [모형 예보, VR1, MG12-3] |
| 로그 knob | 등록 없음 | `PDMUX_HEARTBEAT_S=0` 필수(아니면 `ABORT_LOG_KNOB`), `PDMUX_LOG_MIRROR` 네 Job 동일값(권고 1, U4 미결) | SP §0, rev12 §19-5 |
| 엔진 digest | `82a5fc5b5f587464…` 등록 | **merge 트리에서 재실측**해 등록(같으면 그대로, 다르면 M2g PASS 범위를 사용자에게 다시 물음) | merge(`dc06c870`) 뒤 재봉인, rev12 §19-6 |
| preflight gate | 없음 | **신설**: (i) `PREFLIGHT_PASS` 커밋·digest 일치 (ii) 이미지 digest 동일 (iii) **7개 경로**(`src`·`tests`·`scripts/bootstrap`·`scripts/r2_eval`·`env`·`benchmarks/pdmux_eval`·`scripts/multigpu`) diff 빈 출력. 최종 POST = 7경로를 바꾼 마지막 커밋(재봉인 뒤), preflight는 그 뒤 1회 [제안, VR2 F-1] | V2 §6 조건 1, R-5, U13 [결정: 2회차 감사 뒤] |
| 하드캡 | J1 10 h, J2·J4·J8 14 h | **J1 14 h, J2·J4·J8 16 h** [결정, U3]. 상한 예약 252 GPU-h [유도](rev11 216) | DECISIONS §4 U3, rev12 §19-10 |
| 예산(예약 GPU-h) | **124** [유도](123.9) | **189** [유도](188.9) | X·floor·U14 추가, rev12 §19-10(scratchpad 하네스 사본 모형) |
| 예산(사용률) | **89%** [유도] | **88%** [유도] | 같음 |
| 예산(최장 Job 벽시계) | **8.8 h** [유도](J2 8.76) | **13.2 h** [유도](J2 13.15). TP 완료는 X 참 용량/prior\_X ≈ 0.9–1.42 범위에서만 [모형 예보, MG12-13] | 같음 |
| 하네스 재봉인 | — (rev11 봉인 `313be0a`) | **필요**. 대상: `mg_const.py`·`mg_plan.py`·`mg_drive.py`·`mg_label.py`·`mg_job_common.sh`(`MANDATORY_REG`)·`m3_job.sh`, `scripts/multigpu/mg_loadgen.py`·`mg_runner.sh`·`mg_boot.py`, `tests/` 하네스 시험, 공용 `benchmarks/pdmux_eval/trace_loadgen.py`·`context_limit.py`(`0eca947`로 이미 등록 바이트가 깨짐) | rev12 §0·§19-12 |
| 인용 범위: 명세 v0.2+ 적용 | 명세 없음 | 적용: 같은 Job·기판 gate·엔진 digest·로그 knob·고정 노브, R4 범위 문장, R5-2 Job 기록, R5-5 재표집 수 공시, R6 A′ floor(기술값), R7 ITL·오류 분류, R8-1·R8-3·R8-4, R9-1 λ\*\_thr, R10, R11-1(부분), R13-1, R16-1·R16-2 | rev12 §19-1 표 |
| 인용 범위: 비적용 | — | R5-1·R5-3 Job ≥ 3·Holm(M3는 Job 안 판정, D-2), R8-2 튜닝·mixed-chunk·G\_mixed, R11-2–R11-5 공변 검사, §9 계측 I-1–I-7, R13-2–R13-4 span 스윕 | rev12 §19-1 표 |
| 범위 밖으로 둔 것 | — | **shape C**(긴 입력 + 긴 출력, 출력 ≥ 512): 범위 문장만(U9). **TP의 GPU당 decode SM 요구**: 미해결 공시, N별 D 스윕 없음(U10) | rev12 §19-7·§19-8 |

## 3. 바뀌지 않은 것

- arm 4종(fused, fused\_sym, agnostic, d44)과 그 정의(F6 [결정]). D44 분할, agnostic 임계표.
- cap: N=1 48, TP 48N, DP 복제본당 48(F7). SP v0.2+의 "G0 cap = cap 특성화 결과"는 G0 범위이고 M3 cap을 바꾸지 않는다.
- 판정 규칙의 골격: λ\*\_thr = HI 셀 seed별 achieved 중앙값, HI 1.6×prior·LO 브래킷, 확장 규칙(§4-2), g·`HI_SATURATED`·`KNEE_BRACKETED` 라벨, seed 짝 bootstrap(n=4), log₂N 추세, 3% 크기 규칙, P9·P10·P11·§18, 등록 순서 [DP, TP].
- 같은 Job 원칙: 한 비교의 arm은 한 Job 안, 같은 (config, shape, seed)의 네 arm은 같은 바이트 trace.
- 오류 무효 상한 1%(rev11 승계 상수, 합산 기준). 판정은 λ\*\_thr뿐이고 λ\*\_SLO는 비등록 사후 기술값.
- M2g(§3)와 PASS 표 4개(`99ed704`), shape A·B의 payload 바이트, boot 70회, `sock_read` 600 s.

## 4. 결과 해석에 주는 영향

- **새로 생기는 것**: shape X에서 장문 요약 영역(입력 ≤ 16k, 출력 p50 195)의 G\_sym(N)이 생긴다. 단 X는 shape C(긴 입력 + 긴 출력)를 대신하지 않고, 결과를 그 영역으로 외삽하지 않는다(§19-8). X 입력은 실제 토큰, A·B는 합성이라 shape 간 차이를 입력 내용 효과로 쓰지 않는다.
- **오류 처리**: 무효 문턱은 rev11과 같은 상한(합산 1%)이다. 달라진 것은 분류 기록과, 조용한 1토큰 성공이 오류로 잡힌다는 점이다. 이 차단은 CPU 시험 범위의 주장이고 서빙 실측은 0이다.
- **Job 효과**: X에도 앵커 검사가 붙지만, X 참 용량이 prior보다 크면 기준 셀이 포화하지 않아 `JOB_EFFECT_UNTESTED`가 될 수 있다(MG12-12). 미검사는 Job 효과 부재가 아니다.
- **검정력 비용**: P10이 세 shape로 넓어져 귀무에서도 BROKEN이 늘 수 있고(MG12-3), 그만큼 TP 방향 라벨 인용이 줄어든다.
- **예산과 절단 위험**: 최장 Job이 약 13 h [유도]로 늘어 캡 16 h 여유가 작다. X 참 용량이 예보 범위를 벗어나면 TP 구간이 등록 순서대로 잘린다(MG12-13). TP 결과 부재는 TP 효과 부재가 아니다.
- **서버 인자**: context 20480·명시 인자 조합은 M2g PASS가 잰 조합(16384·자동값)과 다르다. M2g PASS를 이 조합이나 16k 입력의 정확성으로 쓰지 않는다(MG12-5).
- 이 문서는 어떤 G 값이나 추세 방향도 예측하지 않는다.

## 5. 남은 단계와 사용자 결정 지점

1. **하네스 수정**(§19-12 전부: shape X·floor 블록·U14 앵커·합산 오류 규칙·F-3 `replay` 경로·F-4 부팅 로그 대조·사전 토큰화 파일 생성과 sha 등록) — CPU.
2. **하네스층 감사** — r2 diff(합산 규칙 등, 규칙층 감사를 받지 않은 바이트)를 변이 시험으로 확인. 예: "서버 0.9% + 클라이언트 0.9%인데 라벨 그대로" KILLED.
3. **재봉인**(`PREREG_REV = 12`, 엔진 digest 재실측, SP 바이트 sha 등록). digest가 다르면 M2g PASS 범위를 사용자가 다시 정한다.
4. **preflight** 1회(최종 POST = 재봉인 뒤 7경로를 바꾼 마지막 커밋). 17건이 VESSL에서도 닿지 않으면 사용자에게 다시 올리고 M3 제출을 보류한다.
5. **dry-run**(`launch.sh`, `mg_plan.py job-table`·`m3-forecast`). 캡을 넘으면 제출 전에 다시 묻는다.
6. **제출 승인**(사용자).

미결 사용자 결정(rev12 §19-13): U4 `PDMUX_LOG_MIRROR`(권고 1), U5 J1 스펙(권고 x2), U6 M3와 G0 계측 패치 순서(권고 M3 먼저), U7 16k 입력의 M2g 범위(권고 범위 공시), U9 shape C(권고 범위 문장), U10 TP decode SM(권고 미해결 공시), U11 λ\*\_SLO 위치(권고 봉인 밖), U12 P10 확장(권고 fail-closed 확장). U8은 M4 단계 감사에서 정한다.
