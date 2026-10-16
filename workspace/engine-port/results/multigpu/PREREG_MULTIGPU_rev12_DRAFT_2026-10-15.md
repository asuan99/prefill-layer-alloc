# 사전등록 개정안 초안 — PREREG_MULTIGPU rev12 (M3·M4), 2026-10-15

> **rev12 DRAFT r1 — 감사 1회차 수리판 (2026-10-16).** 1회차 판정서 `VERDICT_m3_rev12_draft_r1_2026-10-15.md`(이하 VR1, **NO-GO**, 死因 D-1)의 필수 R-1–R-5와 권고 R-6–R-9를 모두 반영했다. 수리 전 r0 = 246줄판, sha256 `29dd7d2f73b9edbe…`(미커밋).
> 근거 결정: `../handoff_review_2026-10-15/DECISIONS_Q1_consolidated_2026-10-15.md` §4(사용자 결정 2026-10-16 "진행"): 순서, R-1 대 D-7 해석, U3, venv (c), D-10, cap 특성화, context 20480.
> **지위: DRAFT r1. 미봉인. 감사 1회차 NO-GO 뒤 수리, 2회차(상한) 대기.** 대상 `PREREG_MULTIGPU_2026-10-12.md` rev11(sha256 앞 16자리 `8b617d3d59d6cb63`, 봉인 `313be0a`)은 **고치지 않는다**.
> rev12가 채택되면 이 초안의 §19를 새 판 PREREG에 붙이고 우선순위를 **§19 > §18 > §17 > §16 > §3·§4·§15**로 둔다. M2g(§3)와 PASS 표 4개(`99ed704`)는 바꾸지 않는다.
> 근거: 사용자 결정(2026-10-15) "M3 PREREG 개정 후 재감사" · IDX Q23 · 명세 v0.2+(`../handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`, 이하 SP, DRAFT·정본 아님) ·
> 판정서 V(1회차) §3 A-13 ①–⑤, V2(2회차) §6 · DESIGN §3.2–3.3 · main 핸드오프 (D)절 ①–⑥(`handoff-report/session_handoff_2026-10-13.md:219`).
> 작성 시점 상태: GPU 사용 0, 서빙 측정 0, 성능 판정 0, 커밋 0. 수치 라벨은 SP와 같다: [측정]·[유도]·[코드]·[결정]·[제안]. VR1의 반전 시험 수치는 **모형 예보**이며 측정으로 인용하지 않는다.
> 이 초안의 모든 수치는 VESSL A100 SXM 80GB(108 SM) 기판 범위이며 KISTI 수치와 섞지 않는다.

**수리 색인 (VR1 §5 → 이 판의 절)**

| 수리 | 내용 | 고친 곳 |
|---|---|---|
| R-1 (死因 D-1) | λ\*\_thr: HI 셀 서버 측 오류 > 1% → `SERVER_ERROR_HI`, 그 arm λ\* = `UNRESOLVED`, G 짝 무효. 변이 시험 등록. D-7과의 관계(사용자 승인 해석) | §19-1 R7-3 행, §19-1b, §19-12 |
| R-2 | 공용 loadgen: SSE `error` 키 = 서버 측 오류, `OUTPUT_SHORT`. 커밋 `<sha 미정>`을 등록 요구로 | §19-1b, §19-12 |
| R-3 | U2 = (a) context 20480, 근거 수치, U2(b) 실행 불가, G5(c)(ii) X 항, `mg_const.CONTEXT_LENGTH` 변경 지점 | §19-2 라벨, §19-3, §19-12, §19-13 |
| R-4 | `mamba_ssm_dtype`: 키 없으면 float32, `torch_dtype` 폴백 금지, boot마다 `server_info` 대조 | §19-3 |
| R-5 | 제출 gate = `PREFLIGHT_PASS` ∧ 이미지 digest ∧ 경로 diff 빈 출력, POST = loadgen·runner 수정 커밋, C7 "저장소에 없음" → `d138e7f` 정정, 대조 경로 판단 | §19-6, §19-13 U13 |
| R-6 (권고, 채택) | floor\_N 블록을 config boot1의 fused\_sym·agnostic 둘 다에 → MG12-6 비적용 | §19-4, §19-10, §19-11 |
| R-7 (권고, 채택) | §19-1 표에 R5-5 행, "판정 절차 그대로" 문장에 C5·floor·dtype 영향 명시 | §19-1 |
| R-8 (권고, 채택) | X 표본 = seed별 순열 prefix(N 간 중첩), 입력 토큰 = `[1]*L` 합성 등록 | §19-2, §19-4 |
| R-9 (권고, 채택) | λ\*\_SLO 분석기를 M3 fetch 전 커밋·sha 기록 | §19-9 |
| U3 | 하드캡 J1 14 h, J2·J4·J8 16 h [결정 2026-10-16] | §19-10, §19-13 |
| VR1 §6·§7 | MG12-1–MG12-10 문자 그대로, 금지 문안 §19-11 추가 | §19-11 |

## 0. rev11 대비 변경 색인

| # | 변경 | rev11 위치 | rev12 절 | 하네스 변경 |
|---|---|---|---|---|
| C1 | 명세 v0.2+ 인용 방식(적용·비적용 표), 1차 estimand G\_sym 유지 | 머리, §4-4 | §19-1 | 일부(C2–C6에 포함) |
| C2 | shape X = arXiv-Sum 16k 추가(3 shape) | §1 shape 행, §4-1, §4-2, §17-1 | §19-2 | **예** |
| C3 | `max_prefill_tokens`·`split_forward_token_budget`·`context_length`·mamba 노브 명시 등록 | §1 | §19-3 | **예**(runner 인자) |
| C4 | floor 측정 블록(N=1·N별, 두 arm 대칭), 절대 T(A′) 기술값, M4 SLO 3 s → A′ | §4-6, §5, §8 항목 9 | §19-4 | **예**(floor 블록) |
| C5 | `CLIENT_ERROR` 1%를 클라이언트 측 오류에만 + **λ\*\_thr의 `SERVER_ERROR_HI`(r1)** | §4-3 | §19-1 R7-3 행, §19-1b | **예** |
| C6 | 로그 knob `PDMUX_HEARTBEAT_S=0`, `PDMUX_LOG_MIRROR` 값 등록·기록 | 없음 | §19-5 | 예(launch env·meta 검사) |
| C7 | 엔진 digest 재등록, preflight `PREFLIGHT_PASS` + 경로 diff gate(r1) | §6-3, 머리 | §19-6 | 예(중단 코드 2개) |
| C8 | TP에서 GPU당 decode SM 요구 변화 → 미해결 공시 | 없음 | §19-7 | 아니오 |
| C9 | shape C(긴 입력+긴 출력) 부재 → 범위 문장 | 없음 | §19-8 | 아니오 |
| C10 | λ\*\_SLO = 봉인 밖 분석기의 비등록 사후 기술값(분석기는 fetch 전 커밋, r1) | 없음 | §19-9 | 아니오(분석기는 봉인 밖) |
| C11 | 예산·하드캡 재산정 | §6-1, §15-4, §16-2 항목 2, P9-d | §19-10 | 예(`HARD_CAP_S`) |
| C12 | 금지 서술 추가 | §7 | §19-11 | — |

**하네스 재봉인: 필요.** C2–C7·C11이 `mg_const.py`·`mg_plan.py`·`mg_drive.py`·`mg_label.py`·`scripts/multigpu/mg_loadgen.py`·`engine_bench_runner.sh`·`m3_job.sh`와 공용 `benchmarks/pdmux_eval/trace_loadgen.py`(R-2)를 바꾼다.
순서 [결정, DECISIONS §4]: rev12 R-1–R-9 반영(이 판) → 공용 loadgen·runner 수정 커밋(R-2·R-3, M3·G0 공용) → 그 커밋에서 preflight 1회 → rev12 2회차 감사 → 하네스 수정·하네스층 감사 → `PREREG_REV = 12`로 재봉인 → dry-run → 사용자 승인 → 제출(교훈 34). preflight는 loadgen·runner 수정 커밋 전에는 제출하지 않는다(VR1 R-5). λ\*\_SLO를 하네스 밖에 두는 결정(C10)은 재봉인을 피하려는 것이 아니다(§19-9).

## 19. rev12

### 19-1. 명세 v0.2+ 인용 방식 (C1, C5)

원칙: rev12는 SP를 **아래 표에 적은 규칙만** 인용한다. 표에 없는 SP 규칙은 M3·M4에 적용하지 않는다. SP는 DRAFT이므로 인용 대상은 **rev12 봉인 시점의 SP 바이트**(v0.2+ 2026-10-16 판 이후, 봉인 때 sha256 등록)이고, SP가 그 뒤 promote로 바뀌면 rev13 개정 없이는 따라가지 않는다.
**1차 estimand는 G\_sym(N, mode, shape) = λ\*\_thr,agnostic / λ\*\_thr,fused\_sym 그대로다**(F6 [결정], SP R8-3 표 "multi-GPU 1차"). 판정 절차(seed 짝 bootstrap CI, log₂N 추세, 3% 크기 규칙, P9·P10·P11·§18)는 rev11 그대로이나, **판정에 들어가는 셀의 유효성과 boot 사전 조건은 세 곳에서 바뀐다**(VR1 Q13, R-7):
(1) C5 + R-1: `CLIENT_ERROR`는 클라이언트 측만, 대신 λ\*\_thr HI 셀의 서버 측 오류 > 1%는 `SERVER_ERROR_HI`로 짝 무효(§19-1b) — rev11의 "모든 오류 > 1% 짝 무효"와 같은 방향(fail-closed)이나 분류가 둘로 갈린다.
(2) floor 블록: 판정 boot 시작부에 floor 블록이 들어간다. r1은 config boot1의 두 arm 모두에 두어 N=1과 N>1에서 두 arm의 boot 사전 조건이 대칭이다(§19-4, R-6).
(3) `mamba_ssm_dtype` 명시: 키 없으면 float32(§19-3, R-4) = M2g 실효값이라 처치 불변으로 유도되나 boot마다 대조한다. shape 순서 3원소화는 rev11 예보 비교만 바꾸고 판정은 바꾸지 않는다.

| SP 규칙 | M3 | M4 | 처리 |
|---|---|---|---|
| §0 같은 Job 안 arm·공통 격자 | 적용 | 적용 | §15-1과 같음. HI rung(1.6×prior)은 사전 등록 절대값, LO rung(0.6×fused HI 중앙값)은 데이터 의존 — λ\*\_thr 브래킷 설계(rev4)라 유지하고 차이를 공시 |
| §0 기판 gate | 적용 | 적용 | §16-3과 같음(SP가 MG를 따른 것) |
| §0 엔진 digest | 적용 | 적용 | §19-6 |
| §0 로그 knob | 적용 | 적용 | §19-5 |
| §0 고정 노브(`context_length` 20480 포함) | 적용 | 적용 | §19-3(`max_mamba_cache_size`·`mamba_ssm_dtype`·`context_length` 명시) |
| R4-1·R4-2 고정 rate 허용·정상성 | 적용(범위 문장) | 해당 없음(변화 trace) | M3 결과에 "정적 시스템 비교, 정상상태 고정 rate" 범위 문장 의무. 정상성은 rev11 §4-3 g 라벨이 맡는다(S1–S3는 쓰지 않음; HI는 의도된 비정상 셀이라 공시) |
| R5-1·R5-3·R5-3b Job ≥3, Holm | **비적용** | **비적용** | D-2 [결정]: M3는 Job 안 판정. SP R5-4대로 결과 문장은 "이 Job에서" 범위 |
| R5-2 Job 기록 | 적용 | 적용 | hostname·UUID·드라이버·`cpu.max`·노드 종류. 같은 UUID면 `JOB_NOT_INDEPENDENT` 기술값 |
| **R5-5 bootstrap 재표집 수 공시** | **적용(공시)** | 적용(공시) | seed 짝 bootstrap은 n=4라 서로 다른 재표집이 35개뿐임을 G 표마다 적는다(N3VR-11, VR1 R-7) |
| R5-6 duration 합산 | 적용(기술값) | 적용 | λ\*\_thr 정의(seed별 achieved 중앙값)는 그대로. 사후 attainment는 합의 비 |
| R6-1–R6-4 A′ floor | 적용(기술값) | 적용(판정) | §19-4 |
| R6-5 길이 3분위·가산형 | 적용(기술값, 분석기) | 적용(기술값) | shape X에서만 의미 있음 |
| R7-1 ITL P95 계산법 | 적용 | 적용 | `numpy.percentile` linear |
| R7-2 재채점 범위 | 적용 | 적용 | M3 4 arm은 모두 SLO를 읽지 않는 stock arm |
| R7-3 오류 분류 | **적용(C5) + λ\*\_thr 예외(R-1)** | 적용 | §4-3 `CLIENT_ERROR`는 **클라이언트 측 오류**(송신 전 연결 실패, loadgen 예외, 예정 대비 발송 지연 > 0.5 s, 분류 불가)만 센다(SP R7 multi-GPU 차이). 서버 측 오류는 건수 기록 + achieved 분자에서 빠진다(`mg_label.py:426` [코드]) — 가변 길이 X에서는 긴 요청이 선택적으로 빠져 achieved가 부푼다(VR1 死因 D-1) → **λ\*\_thr HI 셀에는 §19-1b `SERVER_ERROR_HI`를 적용**한다. 한 arm이 `CLIENT_ERROR` 또는 `SERVER_ERROR_HI`면 그 (config, shape)의 G 짝이 미정의 |
| R7-3 client timeout 하한 | 적용 | 적용 | `sock_read` 600 s ≥ 25·floor(16k) [유도: floor(16k)가 24 s 미만이면 성립, 측정 후 확인] |
| R7-4·R7-5 n\_ITL·실패 분해 | 적용(기술값) | 적용 | n\_ITL [유도]: A ≈ 511, B ≈ 63, X ≈ 194 |
| R8-1 arm 정의 | 적용 | 적용 | 바이트 동일(SP가 MG §2를 따름) |
| R8-2·R8-2b–d 튜닝·mixed-chunk·G\_mixed | **비적용** | **비적용** | SP §6 multi-GPU 차이 [제안]: 비용 때문에 multi-GPU에 넣지 않음. M3 문장은 G\_sym·G\_def 범위 |
| R8-3 문장 대응표 | 적용 | 적용 | G\_sym 행 "같은 스케줄 노브에서 PD-mux 기능 전체의 효과", 금지 "SM 분할 효과"·"PD 분리 효과". G\_sym이 multi-GPU에서는 1차, 단일 GPU에서는 2차 가족임을 결과 문장마다 적는다(V2 B-2) |
| R8-4 플래그 열 5개 | 적용 | 적용 | `enable_mixed_chunk` 열은 전 arm off |
| R9-1 λ\*\_thr | 적용(판정) | — | rev11 §4-2·4-3 + §19-1b |
| R9-1 λ\*\_SLO, R9-2, R9-6 | 적용(비등록 사후 기술값) | M4 PREREG에서 | §19-9. R9-3 사다리 조건을 M3 rung이 충족하지 않음 |
| R10-1–R10-3 N 간 SLO | 적용 | 적용 | §19-4 |
| R11-1 클럭·전력 수집 | 부분 적용 | 부분 적용 | 기존 `clk.csv`(`nvidia-smi -lms 100`, sm clk·power·throttle·index [코드, `job_entry.sh:191`]). SP의 전력 20 ms 주기는 **미충족**(공시). 0행이면 `SUBSTRATE_UNCHECKED`(그 Job 셀의 λ\*·G 크기 인용 금지) |
| R11-2–R11-5 공변 검사 | **비적용** | **비적용** | I-2(★ 엔진 패치) 없음. 범위 문장 §19-11 |
| §9 I-1–I-7 계측 | 비적용(I-1 부분) | 비적용 | §19-6 결정 대기 U6 |
| R13-1 span 예산 headline 고정 | 적용 | 적용 | §19-3. 스윕(R13-2–R13-4)은 비적용(SP §11 multi-GPU 차이) |
| R16-1 cap·mamba 명시 | 적용(**M3 cap = F7 48·48N·DP 48 유지**) | 같음 | §19-3. SP v0.2+의 "G0 cap = cap 특성화 Job 결과"는 G0 범위이고 M3 cap을 바꾸지 않는다 |
| R16-2 `CAP_BINDING`(D-10 확정 2026-10-16) | 적용 → 전 셀 `CAP_BINDING_UNRESOLVED` | 같음 | I-5 없음. 서버 로그 `#running-req`는 기술값(MGP-17) |

### 19-1b. λ\*\_thr의 서버 측 오류 규칙 (R-1, 死因 D-1 수리; R-2 선행)

- **규칙**: λ\*\_thr를 정하는 HI 셀(§4-2 확장 셀 포함, 전 shape A·B·X)에서 서버 측 오류 비율(분자 = 서버 측 오류 요청 수, 분모 = 그 셀 요청 수) **> 1%**면 그 셀에 `SERVER_ERROR_HI`를 붙이고, **그 arm의 λ\* = `UNRESOLVED`**, 그 (config, shape)의 **G 짝을 무효**로 한다(`CLIENT_ERROR`와 같은 대칭 무효, 건너뛰기·대체 seed 금지). 1%는 rev11 §4-3의 승계 상수이며 새 상수가 아니다.
  근거 [모형 예보, VR1 §2 — 측정 아님]: 최악 선택(가장 긴 문서부터 오류)에서도 1% 이하 오류의 achieved 부풀림은 3% 크기 규칙보다 작다. LO 셀의 서버 측 오류 건수는 기술값으로만 적는다.
- **서버 측 오류의 범위**: 수락 뒤 timeout, HTTP 5xx, abort, **스트리밍 응답 안의 SSE `"error"` 청크**, **`OUTPUT_SHORT`**(`ignore_eos` 요청에서 관측 출력 토큰 수 ≠ 요청 수). 뒤 둘은 R-2 loadgen 수정이 있어야 보인다(VR1 §4-1: 거절이 HTTP 200 + error 청크로 와서 현 loadgen은 1토큰 성공으로 기록 [코드, `entrypoints/http_server.py:665-676`, `trace_loadgen.py:53-89`]).
- **R-2 등록 요구(M3·G0 공용)**: 공용 loadgen `benchmarks/pdmux_eval/trace_loadgen.py` 수정 커밋 **`<sha 미정>`**(engine-porter 구현 중)을 rev12 봉인 전에 등록한다. 요구 내용: (a) SSE `"error"` 키 = 서버 측 오류("수락 뒤 오류")로 기록, 토큰으로 세지 않음 (b) `observed_output_tokens ≠ requested` → 서버 측 `OUTPUT_SHORT` (c) `sock_read=600` 문자열 유지(`mg_loadgen.py`의 H5 검사) (d) 변이 "error 청크를 토큰으로 셈" KILLED + 무변이 대조군 통과. 이 커밋이 preflight POST다(§19-6).
  공시: N2/N3/E2 봉인 캠페인은 수정 전 loadgen으로 돌았다. 이 수정을 그 결과에 소급 적용하지 않는다.
- **D-7과의 관계 [결정, 사용자 승인 해석 2026-10-16, DECISIONS §4]**: D-7(1% 문턱은 클라이언트 측 오류에만)은 **λ\*\_SLO·attainment 채점 범위**의 결정이다. 그 범위에서 서버 측 오류 = 불합격(SP R7-3)은 그대로 유지한다. R-1은 M3 **λ\*\_thr 셀 유효성** 규칙이라 D-7과 충돌하지 않는다.
- **변이 시험(하네스, §19-12)**: "한 arm X HI 셀의 가장 긴 3%가 서버 측 오류인데 라벨이 그대로" 변이본 KILLED, 무변이 대조군 통과. 추가: "`OUTPUT_SHORT`를 성공으로 셈" 변이본 KILLED.

### 19-2. shape X = arXiv-Sum 16k (C2)

- **정의**: SP §10 그대로. `ccdv/arxiv-summarization` document/test parquet sha256 `b34529df…` [측정, ARXSUM\_STATS], 필터 입력 ≤ 16,384 토큰, 출력 길이 = abstract 길이(`ignore_eos`), 출력 < 2토큰 제외·건수 공시, logprob 없음.
  유지분(Nano-9B-v2 토크나이저) 5,906/6,440, 입력 mean 7,116·p50 6,484·p99 15,722, 출력 mean 203·p50 195·p99 399, 최대 출력 616, 최대 입력+출력 16,744 [측정, ARXSUM\_STATS·VR1 원자료 재계수].
  - **토크나이저 확인 [미확인]**: ARXSUM\_STATS는 `nvidia/NVIDIA-Nemotron-Nano-9B-v2` 토크나이저로 셌다. M3 모델은 `-Base` revision `dc0661c…`다. 하네스가 M3 모델 토크나이저로 다시 세고 토크나이저 파일 sha256을 기록한다. 유지 집합이 달라지면 그 수를 공시한다(최대 입력+출력이 20,480 − 1을 넘으면 `ABORT_DATASET`).
  - 데이터 sha256 불일치 → `ABORT_DATASET`(GPU 작업 전, 새 중단 코드) [하네스].
- **입력 토큰 내용 (R-8) [제안]**: shape X는 arXiv-Sum의 **(입력 길이, 출력 길이) 쌍만** 쓴다. 보내는 입력은 A·B와 같은 `[1]*L` 합성 토큰이다(현 loadgen `"input_ids": [1] * input_tokens` [코드, `trace_loadgen.py`], 형식 변경 없음). 근거: arm 비교는 같은 바이트 trace의 짝이고, 같은 생성기를 세 shape에 쓰면 내용 차이가 shape 간에 섞이지 않는다. 범위 문장(의무): "shape X는 arXiv-Sum 길이 분포의 합성 토큰 요청이며 실제 문서 내용으로 잰 것이 아니다."
- **trace (R-8)**: seed s마다 유지분 문서 ID(오름차순)의 등록 순열 π\_s(seed 20261012 + s에서 생성, 봉인 때 sha 기록)를 한 번 만들고, 요청 수 n인 셀은 **π\_s의 앞 n개**를 쓴다. 따라서 같은 seed에서 N이 작은 config의 표본은 N이 큰 config 표본의 부분집합이다(N 간 중첩). 확장 셀(§4-2)은 같은 π\_s의 앞 n\_ext개, LO 셀도 같은 π\_s의 앞 n\_LO개. Poisson 도착은 seed별. 같은 (config, shape, seed)의 네 arm은 같은 바이트 trace를 쓴다. 중첩이어도 N이 다르면 표본 집합이 다르다(MG12-10).
- **prior(rung 위치 전용)**: prior\_X = 0.72 req/s/GPU [유도, 가산 근사: 1/λ ≈ 7,116/(0.690×8,192) + 203/(2.99×512) ≈ 1.39 s]. 측정이 아니며 λ\*·G로 인용하지 않는다(MG12-1).
  HI n = clamp(round10(1.6×0.72N×170), 40, 4000) [유도]: N=1 200, N=2 390, N=4 780, N=8 1,570(유지분 5,906 이하; 확장 n은 N=8에서 3,140 ≤ 5,906이라 비복원 가능 [유도, VR1 Q3]).
- **셀 구성(§4-1 대체)**: arm 4 × shape 3(A, B, X) × (HI 4 seed + LO 2 seed). boot 배치는 rev11 그대로 — boot 1 = 세 shape HI, boot 2 = 세 shape LO. shape 순서는 boot마다 seed 20261012 무작위(`shape_order`가 3원소로).
  확장 규칙(§4-2)은 (config, shape) 단위 그대로, X에도 적용. 셀 수 [유도]: 본 셀 7 config × 4 arm × 3 shape × 6 = **504**(rev11 336), 앵커 24·GPU 편차 8 → **536**(rev11 368). boot 70회 불변. floor 블록 20개 별도(§19-4).
- **라벨**: g·`HI_SATURATED`·`LO_UNSATURATED`·`KNEE_BRACKETED`·E-time·`FRONTEND_SATURATED` 정의는 shape 무관 그대로. G5(c)(ii) KV 구속의 (ISL+OSL)는 X에서 **16,384 + 616(유지분 최대 출력) [측정]**으로 둔다(R-3)
  [유도: N=1에서 48 × 17,000 ≈ 0.82 M 토큰 < 부팅 로그 2.72 M [측정, M2g J1 `cell_meta.json`, ctx 16384] → 구속 아님 예상. ctx 20480에서 KV 풀은 context 무관으로 유도 [코드, VR1 Q8]].
- **P-DP·P10·§18 확장**: P10의 shape 집합을 {A, B, X}로 넓힌다. 어느 shape든 BROKEN이면 MGP-13 발화(fail-closed 유지), 세 shape 모두 `NO_TREND_DETECTED`일 때만 HELD. §18 교차 shape 보정 규칙에서 X는 B와 같은 "비앵커 shape"로 다룬다. P9-a는 shape마다(G\_sym(1, X) 미정의면 X 방향 라벨 인용 불가).
  공시: shape가 셋이면 우연한 DP 방향 라벨로 BROKEN이 날 확률이 커지고 그만큼 TP 인용이 줄어든다(검정력 비용, MG12-3).
- **앵커**: Job 효과 검사(§15-3, §16-1)는 shape A 그대로다. X에 대한 Job 효과는 검출하지 않는다 → X의 추세 라벨에 "Job 효과 검사는 shape A 앵커 기준" 공시 의무(MG12-4). VR1 Q5 권고(J\_N 앵커 boot에 X HI 4 seed, Job당 ≈0.5 h)는 **채택하지 않는다**(필수 아님, U14로 남김).
- **M2g 범위**: M2g G1·G2는 등록 prompt 64개·max\_new\_tokens 128로 쟀다. 16k 입력은 M2g PASS 범위 밖이다(C-8, MG12-5). rev12는 M2g를 다시 돌리지 않고 범위 공시로 처리한다(U7). TP4·TP8 `TP_NUMERIC_DIVERGENT` 병기 의무(MGH-3)는 X에도 붙는다.
- **하네스 변경 지점** [코드 확인분]: `mg_const.SHAPE_IO`·`PRIOR_PER_GPU`·`EST_DRAIN_S`·`EST_WARMUP_S`에 X, `mg_plan.shape_order`·`boots_boot1`·`m3_static_boots`·`m3_timeline`의 `("A","B")` 하드코딩, `mg_drive.py:239`의 `("A","B")`, loadgen의 길이 표본 경로(고정 길이 → 순열 prefix 길이 쌍), G5(c)(ii) 식, P10 shape 루프(`mg_label.py`), 변이 시험 "X를 빠뜨린 P10" KILLED.

### 19-3. `max_prefill_tokens`·span 예산·context 길이 명시 등록 (C3)

| 노브 | 등록값 | 대상 | 근거 |
|---|---|---|---|
| `--max-prefill-tokens` | **16384**(명시 인자) | 전 arm·전 config·전 shape | SP §10. 엔진 기본값과 같음 [코드, `server_args.py:349`]. 첫 요청은 상한을 넘어도 들어가고 둘째부터 상한을 본다 [코드, `schedule_policy.py:767,790`] → B·X에서 prefill 1회당 1–2요청 [유도] |
| `split_forward_token_budget` | **65536** | agnostic·d44 | SP R13-1 headline 고정. 두 config 파일에 이미 있음 [코드, `pdmux_agnostic_manual.yml`·`pdmux_r2.yml`]. 하네스가 boot마다 config 바이트 sha256과 값을 기록, 다르면 `ABORT_CONFIG_DRIFT` |
| `--context-length` | **20480** [결정, U2 = (a), 2026-10-16] | 전 boot | 필요 하한: X 최대 입력+출력 16,744 → ctx ≥ 16,745, floor 16k 점 16,384 + `max_new_tokens` 8 = 16,392 → ctx ≥ 16,393 [측정·유도, VR1 §4-3]. 20480은 둘을 덮고 여유 3,736. 거절 경로: 입력 + max\_new ≥ ctx면 `ValueError`(`validate_total_tokens` 기본 True, `allow_auto_truncate` 기본 off [코드, `tokenizer_manager.py:252, 785-834`]) |
| `--max-mamba-cache-size` | effective cap(N=1 48, TP 48N, DP 복제본당 48) | 전 boot | SP §0·R16-1(M3는 F7). rev11은 자동값이었고 G5(c)(i)에서 slot = cap 확인 [측정, M2g PASS] |
| `--mamba-ssm-dtype` | **고정 revision(`dc0661c…`)의 `config.json`에 `mamba_ssm_dtype` 키가 없으면 `float32`(엔진 기본 = M2g 실효값), 있으면 그 값. `torch_dtype` 폴백 금지** (R-4) | 전 boot | Nano-9B-v2 HF config에 키 없음, `torch_dtype` bf16 [코드, VR1 Q8, `configs/mamba_utils.py:45-105`]. `-Base` revision의 키 유무는 미확인 → 하네스가 그 revision 파일에서 읽어 결정·기록. **boot마다 `server_info.json`의 실효 dtype과 대조**, 다르면 `ABORT_CONFIG_DRIFT` |

- **U2(b)는 쓰인 대로 실행 불가**(VR1 §4-3): "16384 유지 + 입력+출력 < 16384 필터"는 floor 16k 점(16,392)을 깨고, 필터 경계 16,383은 스케줄러가 `max_new_tokens`를 `max_req_len − 입력 − 1`로 1토큰 깎아 ≤ 16,382여야 한다 [코드, `scheduler.py:1636-1644`]. 그래서 (a)만 등록한다.
- **변경 지점**: runner는 `--context-length`를 이미 `PDMUX_CONTEXT_LENGTH`/record 필드로 받는다 [코드, `engine_bench_runner.sh:96-97, 140`] → runner 기본값 16384는 그대로 두고(N2/N3 재현), M3 하네스가 20480을 넘긴다. **`mg_const.CONTEXT_LENGTH = 16384`(`mg_const.py:127`) → 20480**. §19-12 runner "전달 추가"는 나머지 노브 3개(`--max-prefill-tokens`·`--max-mamba-cache-size`·`--mamba-ssm-dtype`)만 해당한다.
- 범위 문장(의무): "M3의 TP=N은 prefill 1회 입력 토큰 상한 16,384와 span 예산 65,536에 고정된 결과다. TP가 prefill 배치를 키울 여지(상한 확대)와 span 예산 효과는 측정 범위 밖이다."
- `context_length` 20480의 영향 [코드, VR1 Q8]: KV 풀 `max_total_num_tokens`·mamba 슬롯·`max_num_reqs`는 context 무관, 커지는 것은 req\_to\_token 풀과 flashinfer cudagraph `kv_indices`·mask(MB 단위)뿐으로 유도된다. 하네스는 boot 로그의 `max_total_num_tokens`·mamba slot을 M2g 같은 (config, arm) 값과 나란히 적고, 차 > 0.1%면 `KV_POOL_DRIFT` 플래그(판정 불변, 공시) [제안, VR1 §4-3 권고]. M2g PASS 승계 범위는 MG12-5.
- 이 다섯 노브는 G0 PREREG(Q24)와 같은 값·같은 runner 코드여야 한다(SP v0.2+ §0에 `context_length` 20480 등록). runner·loadgen은 공용 파일이므로 변경은 한 번만 하고 두 PREREG가 같은 커밋을 인용한다.

### 19-4. 절대 SLO와 floor (C4)

- **floor 블록(R6-2, R10-2)**: 각 Job의 **N=1(GPU 0) fused\_sym boot 시작부**(J1은 fused\_sym boot1, J2/J4/J8은 fused\_sym 앵커 boot)에서 무경합 단일 요청 TTFT를 입력 길이 {1k, 2k, 4k, 8k, 12k, 16k} × 16회 순차로 잰다.
  prompt = `[1]*L` 합성 토큰(R-8, §19-2와 같은 생성기), `max_new_tokens` 8 [제안]. 16k 점은 ctx ≥ 16,393이 필요하며 20480이 덮는다(§19-3). 중앙값을 초 단위로 동결해 `FLOOR.json`(sha256)으로 남긴다. 같은 블록을 agnostic N=1 boot에도 둔다(R6-4 (a) 비, 기술값).
  **R10-3 기술값용 floor\_N은 각 N>1 config의 boot1 시작부에서 fused\_sym과 agnostic 둘 다** 같은 절차로 잰다(R-6 채택). 블록 수 [유도] = J1 2 + 앵커 2×3 + config 2×6 = **20**(r0 14, +6 블록 ≈ 0.27 h [유도, VR1 R-6]).
  이 배치에서 G\_sym 두 arm의 boot 사전 조건은 N=1과 N>1에서 모두 대칭이다 → **MG12-6은 적용되지 않는다**(§19-11에 원문은 싣되 비적용 표시).
- **T 정의**: T\_i = k·floor\_{N=1}(L\_i), floor = **그 Job의** N=1 fused\_sym 블록 값(R10-2), log–log 보간, 범위 밖 외삽 금지, L < 1k는 floor(1k)(R6-3).
  shape A(L=256)는 floor(1k), shape B는 floor(8k), X는 요청별 보간. ITL 술어 = 요청 내부 token-ITL P95 ≤ 50 ms(N 무관 절대값).
- **M3에서의 지위**: 판정은 λ\*\_thr뿐이다(rev11 §8 항목 9 유지, D-9 [결정]). §4-6 "TTFT 3 s, ITL p95 50 ms 실패 분해"를 **A′ 절대 T(k ∈ {3, 5, 10})** 실패 분해로 바꾸고 봉인 밖 분석기(§19-9)가 `requests.jsonl`에서 계산한다.
  loadgen 요약의 `ttft_slo_ms` 기반 값(`mg_loadgen.py:76`)은 인용 금지(구 기술값). R10-3 N별 상대 T 채점을 병기한다.
- **Job 효과와 floor**: J\_N 앵커 floor가 J1 floor와 어느 길이점에서든 3% 넘게 다르면 `FLOOR_JOB_EFFECT(J_N)`를 붙인다. 이 플래그는 λ\*\_thr G 판정(§15-3, P9-b)을 바꾸지 않고, 그 Job의 SLO 기반 문장(사후 λ\*\_SLO, M4)에 공시 의무로 붙는다. "같은 사용자 SLO"는 Job 효과 범위 안의 근사라고 적는다(R10-2).
- **M4(§5 개정, A-13 ①)**: §5 SLO 행 "TTFT ≤ 3 s ∧ token-ITL p95 ≤ 50 ms"를 **"TTFT ≤ T\_i(A′, k ∈ {3, 5, 10}, 그 M4 Job의 N=1 floor) ∧ token-ITL p95 ≤ 50 ms"**로 바꾼다. M4 Job은 시작부에 N=1 floor 블록을 둔다.
  k 세 개는 한 가족으로 보고 seed 짝 bootstrap CI를 Bonferroni 98.33%로 낸다 [제안]. M4 rate의 λ\*\_fused(config, A)는 M3 값을 **rate 위치에만** 쓴다(다른 Job 값으로 arm을 비교하지 않음). M4의 나머지(shape, ρ, 라운드)는 rev11 그대로이며, M4 Job 승인 전 M4 단계 감사에서 다시 본다(MG12-8).
  참고 [유도]: shape A의 T = k·floor(1k)라 rev11의 3 s보다 훨씬 엄격할 수 있다. 값은 floor 측정 전에는 모른다.
- **하네스 변경**: floor 블록 step 유형(두 arm), `FLOOR.json` 기록·sha, 앵커·config boot1에 블록 삽입, 변이 시험 "J1 floor로 J\_N 채점" KILLED(같은 Job 원칙)·"config boot1 agnostic floor 블록 누락" KILLED.

### 19-5. 측정 Job 로그 knob (C6)

- M3 J1·J2·J4·J8(그리고 이후 M4) 측정 Job은 `launch.sh --env PDMUX_HEARTBEAT_S=0`으로 제출한다. `job_entry.sh`는 값을 읽은 뒤 env에서 지운다 [코드, `job_entry.sh:299-300`] → 하네스가 `meta/`의 기록값을 읽어 0이 아니면 `ABORT_LOG_KNOB`(셀 시작 전).
- `PDMUX_LOG_MIRROR`: 등록값 **1(기본, tee 미러 유지)** [제안 — SP §0 문장이 미러는 "값 기록·공시"로 남긴 취지], 대안 0(U4). **네 Job이 같은 값**이어야 하며 다르면 Job 효과와 섞이므로 결합 단계에서 `LOG_KNOB_MISMATCH`(추세 라벨 인용 불가). 값은 `meta/log_mirror.txt`에서 읽어 결과표에 적는다.
- bring-up·프로브 Job과 preflight Job은 기본값 허용(SP §0).

### 19-6. 엔진 digest와 preflight (C7, R-5)

- **엔진 digest**: rev11의 `ENGINE_DIGEST.txt` = `82a5fc5b5f587464…`. merge(main `dc06c870`) 뒤 공유 dev tree 재대조 일치, merge 이후 main 쪽 엔진 src 변경 0 [측정, 트랙 기록 2026-10-15].
  rev12 봉인 때 **merge 트리에서 다시 실측**해 등록한다. 같으면 `82a5fc5b…`를 그대로 적고, 다르면 새 값을 적고 M2g PASS 표의 유효 범위를 사용자에게 다시 묻는다(PASS 표는 `82a5fc5b…` 트리에서 잰 것). 불일치 시 `ABORT_ENGINE_DIGEST`(rev11 §6-3).
  G0 계측 패치(I-2–I-5, ★)가 M3 전에 들어가면 digest가 바뀐다 → 순서는 U6.
- **preflight(V2 §6 조건 1, G0와 공용 1회)**: 설계 = `../handoff_review_2026-10-15/PREFLIGHT_merge_cl3_design_2026-10-15.md`(**저장소 커밋 `d138e7f`**; r0의 "저장소에 없다"는 거짓이었다 — VR1 Q9 정정). VESSL 컨테이너에서 CL-3 미검증 시험((1) 13 id, (2) `test_zamba2_instrumentation`, (3) venv로 멈춘 17건)을 PRE(`7b15214`)·POST 짝으로 (id, 서명) 대조하고, L2 밖 새 실패가 0이고 구멍이 0일 때 `PREFLIGHT_PASS`.
  **POST = 공용 loadgen·runner 수정 커밋**(R-2·R-3, `<sha 미정>`). `dc06c870` 고정이 아니다. 그 커밋 전에는 preflight를 제출하지 않는다. venv 후보 (c)(`--system-site-packages` venv 생성)는 사용자 허용 [결정 2026-10-16], 모드는 `meta/venv_mode.txt`.
- **M3 제출 gate (R-5)** — 셋 다 참일 때만 GPU 작업을 시작한다. `m3_job.sh`가 셀 전에 대조하고, 하나라도 거짓이면 `ABORT_PREFLIGHT_NOT_PASSED`(사유 기록):
  (i) `PREFLIGHT_PASS` 산출물과 그 sha256이 커밋되어 있고 그 산출물의 merge 트리 digest가 rev12 등록 digest와 같음
  (ii) preflight Job과 M3 Job의 **이미지 digest 동일**
  (iii) `git diff --stat <preflight POST> <M3 제출 커밋> -- workspace/engine-port/{src,tests,scripts/bootstrap,scripts/r2_eval,env,benchmarks/pdmux_eval,scripts/multigpu}` **빈 출력**.
- **대조 경로 판단(VR1 R-5의 5경로 + 2경로)**: 원칙 = "preflight가 실행한 바이트 중 M3 Job이 실행하는 것은 전부 gate 안에 둔다". PASS는 그 바이트에 대해서만 말한다.
  - `benchmarks/pdmux_eval` **추가**: runner가 `python -m pdmux_eval.trace_loadgen`으로 부르고(`engine_bench_runner.sh:116, 226` [코드]) R-2 수정이 여기 들어간다. preflight 스위트가 이 패키지를 import하는 시험(`test_benchmark_tools`·`test_campaign_provenance`·`test_lambda_star_per_shape`·`test_n3_capacity` 등 [코드, grep])을 돌린다. 빼면 POST 뒤 loadgen 변경이 PASS를 그대로 받는다 — VR1 Q9가 runner에 대해 찾은 구멍과 같은 구조이며 감사가 놓친 경로다. G0도 같은 loadgen을 쓰므로 공용 경로다.
  - `scripts/multigpu` **추가(M3 전용)**: `mg_loadgen.py`가 `trace_loadgen.replay`를 감싸고 `mg_runner.sh`가 boot를 띄우며, `tests/test_mg_harness.py`가 둘을 import·실행한다 [코드, `test_mg_harness.py:19-21, 144`]. G0는 쓰지 않으므로 G0 gate에는 넣지 않아도 된다(preflight 설계 §6).
  - `results/multigpu/`(`mg_const`·`mg_plan`·`mg_drive`·`mg_label`·`m3_job.sh`) **넣지 않음**: 이 하네스는 preflight **뒤** 재봉인 단계에서 바뀐다(순서 [결정]). 넣으면 gate가 항상 거짓이 된다. 이 바이트의 검증은 하네스층 감사 + 재봉인 sha + 재봉인 커밋의 CPU 회귀가 맡고, preflight PASS는 이 하네스에 대해 아무것도 말하지 않는다(§19-11 금지).
  - **결과적 순서 제약**: rev12 하네스 변경 중 gate 경로에 닿는 것 — `scripts/multigpu/mg_loadgen.py`(길이 표본·요청별 기록)와 `tests/` 아래 하네스 시험(`test_mg_harness.py` 변이 시험) — 은 **POST 커밋에 함께 들어가야** 한다. 2회차 감사나 재봉인이 그 경로를 다시 고치면 (iii)이 거짓 → preflight를 새 POST에서 다시 돈다(fail-closed, ≈1 GPU-h [유도, 설계 §7]). 어느 쪽으로 갈지는 U13.
- 17건이 VESSL에서도 닿지 않으면(venv 후보 (a)–(c) 모두 실패) "검증 불가"로 남기고 사용자에게 다시 올린다(V2 §6). 이 경우 M3 제출은 보류한다.
- 문장 제약: preflight 통과를 merge 전후 회귀가 없다는 근거로 쓰지 않는다(판정서 C-4 금지 문안 승계, V2 §6-3).

### 19-7. TP에서 GPU당 decode SM 요구 변화 (C8) — **미해결로 공시, 등록하지 않음**

- 사실관계: agnostic 임계표(divisor 36)와 d44의 SM 분할은 **GPU마다 같은 절대 SM 수**로 적용된다. TP=N에서는 GPU당 decode 연산이 같은 배치 대비 약 1/N이고 층마다 all-reduce가 붙는다(C-6, §2). 그래서 같은 ITL에 필요한 GPU당 decode SM 수가 N=1과 다를 수 있다((D)⑤). TP all-reduce 커널이 decode 분할의 CTA를 나눠 쓰는지는 측정 0이다(nsys 이미지 부재, MGH-24′).
- **판단: 등록하지 않고 미해결로 공시한다.** 근거: (1) N=1에서도 decode SM 요구(d\_min)는 측정되지 않았다(N2 d 표 인용 금지 N2C-15). (2) N별 D 스윕을 넣으면 PD-mux 쪽만 N마다 튜닝한 arm이 생긴다(SP S-2와 같은 비대칭). arm 4종은 F6 [결정]이다. (3) D 3점만 더해도 PD-mux arm 수가 2배 이상이 되어 §19-10 예산의 Job 하드캡을 넘는다 [유도].
- **대신 등록하는 것(기술값)**: (i) PD-mux arm의 rank별 E-time·실현 (p, d) 분포(기존 telemetry, rank 접미사), (ii) rank 0 decode step 시간 분포를 N별로 병기, (iii) 범위 문장(의무): "M3의 PD-mux arm은 1-GPU에서 정한 SM 분할(agnostic 표·D44)을 TP=N의 GPU마다 그대로 쓴 값이다. TP에서 GPU당 decode SM 요구가 달라지는지는 측정하지 않았으며, G\_sym(N)의 N-추세에는 분할 크기 부적합 성분이 섞여 있을 수 있다."
- 후속 경로(이 등록 밖): N2형 decode 프로파일을 TP=N에서 재는 별도 사전등록.

### 19-8. shape C(긴 입력+긴 출력) 부재 (C9) — **범위 문장으로 처리, 추가하지 않음**

- 판단 근거: (1) Q3 [결정]은 표준 장문 셀을 arXiv-Sum 16k로 정했다. 출력 ≥ 512인 장문 shape는 값부터 미결이다((D) 열린 항목 (b)). (2) (D)의 권고 순서는 1-GPU에서 shape C를 먼저(5-a), 그다음 M3 개정이다. (3) shape 하나를 더 넣으면 본 셀이 4/3배가 되어 J2·J4 S1이 약 16 h를 넘는다 [유도].
- shape X는 shape C를 대신하지 않는다. Little 법칙 근사로 X의 GPU당 decode 동시 수는 λ≈0.72 × 출력 203 × ITL(20–30 ms 가정) ≈ 2.9–4.4다 [유도]. B(≈1)와 shape C((D) 추정 7–10)의 사이에 있다.
- 범위 문장(의무): "M3 G\_sym(N)은 shape A(입력 256/출력 512), B(8192/64), X(arXiv-Sum, 입력 ≤ 16k, 출력 p50 195) 범위다. 긴 입력과 긴 출력이 함께 있는 영역(출력 ≥ 512)은 측정하지 않았으며 G(N)·추세를 그 영역으로 외삽하지 않는다."

### 19-9. λ\*\_SLO 산출 위치 (C10, A-13 ②, R-9) — **봉인 밖 분석기, 비등록 사후 기술값**

- 결정: D-9 [결정]을 따른다. λ\*\_SLO(k)는 봉인 밖 분석기(`workspace/engine-port/benchmarks/pdmux_eval/` 쪽 새 스크립트)가 SP R9-1 보간식으로 낸다. 이름은 "비등록 사후 기술값"이며 G 판정·추세·제목 문장에 쓰지 않는다(MG12-7).
- **분석기 고정 (R-9)**: 분석기(창 정의·셀 포함 규칙·보간 선택·A′ 실패 분해 포함)를 **M3 첫 Job fetch 전에 커밋**하고 그 커밋 sha와 파일 sha256을 M3 결과 문서에 적는다. fetch 뒤 분석기를 고치면 새 판을 "fetch 뒤 변경"으로 표시하고 fetch 전 판의 출력을 1순위 기술값으로 남긴다. 분석기는 `benchmarks/pdmux_eval`에 들어가므로 그 커밋이 preflight POST 뒤면 §19-6 gate (iii)이 거짓이 된다 → 분석기 파일 위치는 gate 경로 밖(예: `results/multigpu/analysis/`)으로 두거나 POST 전에 커밋한다 [제안, U13과 함께 결정].
- 하네스에 넣지 않는 이유(digest 회피가 아님 — rev12는 어차피 재봉인한다): (1) D-9 사용자 결정. (2) M3 rung은 (config, shape)당 HI·LO 두 점뿐이고 비가 약 2.7배다 [유도]. SP R9-3을 충족하지 않아 등록해도 대부분 `BELOW/ABOVE_LADDER`·`UNRESOLVED`이거나 R9-6 해상도 > 3%다. (3) 재봉인 범위를 C2–C7로 한정한다.
- 하네스를 바꿔 λ\*\_SLO를 등록하려면(대안): R9-3 사다리를 새로 등록해야 하고(셀 증가) 하네스 재봉인과 하네스층 감사가 필요하다. 이 초안은 그 대안을 택하지 않는다.
- 분석기 의무: 봉인 전 N3 rev7 아카이브 dry-run은 **하지 않는다**(사후 기술값이라 대상 아님). 결과에는 rung 수, 인접 비, `CLIFF_ADJACENT`, 해상도(R9-6, n = §19-2 HI n), 셀 창 길이를 함께 싣는다(HI attainment는 창 길이 의존, MG12-7).

### 19-10. 예산 재산정 (C11)

모형 = 하네스 시간 모형(`mg_plan.py` 문서 머리, "boot 4분, n/λ, drain")에 shape X(prior 0.72 [유도], drain 15 s·warm-up 20 s [제안])와 floor 블록 14개(블록당 160 s [유도])를 더한 scratchpad 사본 계산이다(r0 값, r1의 +6 블록 미반영).
**전부 [유도], 측정 아님.** rev11 값은 같은 모형을 봉인 하네스로 다시 돌린 것이다(`mg_plan.py job-table`, 2026-10-15). 예약 = 스펙 GPU 수 × Job 벽시계, 사용 = boot별 사용 GPU 수 × boot 시간.

| Job | 스펙(예약 GPU / 측정 GPU) | rev11 벽시계 h | rev11 예약/사용 GPU-h | rev12 벽시계 h | rev12 예약/사용 GPU-h | rev12 사용률 |
|---|---|---|---|---|---|---|
| J1 | a100x2 (2 / 1) | 4.02 | 8.0 / 4.0 | 5.85 | 11.7 / 5.8 | 49% |
| J2 | a100x2 (2 / 2, 앵커 구간 1) | 8.76 | 17.5 / 16.6 | 12.41 | 24.8 / 23.8 | 96% |
| J4 | a100x4 (4 / 4, 앵커 구간 1) | 8.72 | 34.9 / 32.4 | 12.34 | 49.4 / 46.6 | 94% |
| J8 | a100x8 (8 / 8, 앵커 구간 1) | 7.94 | 63.5 / 57.7 | 11.59 | 92.7 / 86.3 | 93% |
| **합** | 동시 실행 시 GPU 15장 | 최장 **8.76** | **123.9 / 110.7**(89%) | 최장 **12.41** | **178.6 / 162.5**(91%) | |

- 증가분 +54.7 예약 GPU-h 중 floor 블록 ≈ 2.6, shape X ≈ 52 [유도]. r1의 floor +6 블록(J2·J4·J8 각 +2)은 합계 ≈ +0.27 h [유도, VR1 R-6]이며 위 표에는 넣지 않았다. prior\_X를 0.5로 바꿔도 합계 변화 < 0.1 h다(셀 창 170 s 고정) [유도].
- 시나리오(같은 모형, 벽시계 h): S1 λ=prior 11.6–13.3(VESSL 보정 포함) · S2 TP λ=0.7·prior 최장 14.80(J2, `vessl_boot_cell`) · S3 공동 확장(agnostic 1.48·prior) 무절단 J1 12.7–13.3, J2 26.0–27.3, J4 24.2–25.5, J8 22.2–23.5.
- **하드캡 [결정, U3 2026-10-16]**: J1 **14 h**(rev11 10 h), J2·J4·J8 **16 h**(rev11 14 h).
  - J1 근거: J1 10 h에서는 S3가 N1 확장·LO를 자른다 → G\_sym(1) 미정의 → P9-a로 **모든** 방향 라벨 인용 불가. 14 h는 S3(최대 13.3 h)를 덮는다.
  - J2·J4·J8 근거: 16 h는 S1·S2 전 보정을 덮는다(rev9 "예상치 기준"). S3에서는 등록 순서 [DP, TP]대로 TP가 잘린다 → `TRUNCATED`, `TP_UNCONTROLLED_TRUNCATION` 규칙(§17-2 (b)) 그대로(MG12-9).
  - 상한(전 Job 캡 도달) 예약 = 2×14 + 2×16 + 4×16 + 8×16 = **252 GPU-h** [유도](rev11 216).
  - 봉인 dry-run(`mg_plan.py job-table`·`m3-forecast`)이 "S3에서 DP 구간 종료 < 캡"을 출력해야 한다(VR1 Q12). 위 수치는 scratchpad 계산이라 재현 대상이 아니다.
- **스펙·사용률**: J2·J4·J8은 측정 구간에서 스펙 GPU 전부를 쓰고, 앵커 구간(약 0.4 h)만 1/N이다 → 93–96%. J1은 x2에서 GPU 0만 쓰므로 49%다(rev9 사용자 결정: CPU quota·노드 종류를 J2와 맞춤). 대안 J1 = `resourcespec-a100x1`(예약 11.7 → 5.9 GPU-h, 사용률 약 98%, 노드 종류가 달라져 Job 효과 성분이 늘 수 있음) → U5.
- 대안 배치(채택 안 함): shape X를 N마다 별도 Job으로 떼면 최장 벽시계는 약 9 h로 돌아가지만, X Job마다 앵커·preflight가 중복되고 X의 Job 효과가 A/B와 다른 Job에서 생긴다. 같은 Job 원칙상 권고하지 않는다.
- dry-run 값이 위 캡을 넘으면 제출 전에 다시 묻는다(rev11 §6-1 규칙 유지).

### 19-11. 금지 서술 추가 (C12)

§7·§15-5·§16·§17·§18 전부와 SP의 S-1–S-8·K1–K9, 판정서의 C-1–C-6·CL-1–CL-3을 승계한다. 추가:
- shape X 결과를 "실서비스 장문 요약 일반", "긴 입력+긴 출력 영역", "shape C", "실제 문서 내용에서의 결과"로 쓰는 것(§19-2 `[1]*L`).
- M2g PASS를 16k 입력에서의 TP 정확성으로 쓰는 것.
- G\_sym(N) 추세를 "TP에서 PD-mux 분할이 더/덜 필요해진다"로 쓰는 것(§19-7: N별 분할 요구 미측정).
- 사후 λ\*\_SLO와 A′ 실패 분해를 G 판정·추세·제목 문장에 쓰는 것. `FLOOR_JOB_EFFECT`가 붙은 Job의 SLO 문장을 공시 없이 쓰는 것.
- "M3가 SP R11-2 공변 검사를 통과했다" 또는 그 반대. 범위 문장(의무): "rev12 M3는 I-2(시계 정렬)가 없는 엔진 digest로 돌아 SP R11-2 공변 검사의 범위 밖이다. 클럭·전력은 기술값으로만 기록했다." (SP R-16 문장을 rev12에 맞게 바꾼 것. rev11 결과에 소급 적용하지 않는다는 원문은 그대로 유효하다.)
- loadgen 요약의 TTFT 3 s 기준 값을 인용하는 것.
- preflight 통과를 merge 회귀 부재의 근거로 쓰는 것(판정서 C-4 승계). "preflight PASS가 rev12 하네스(`results/multigpu/`)를 검증했다"(§19-6 gate 밖).
- `max_prefill_tokens` 16384·span 65536 아래의 TP 결과를 "TP의 prefill 용량 이점" 일반으로 쓰는 것.
- `SERVER_ERROR_HI`로 무효가 된 짝을 "효과 없음"이나 "그 arm이 졌다"로 쓰는 것.
- **VR1 §7 금지 문안(문자 그대로)**: "TP에서 PD-mux 분할(agnostic 표·D44)이 더/덜 적합하다", "G\_sym(N) 추세는 SM 분할 크기 때문이다"(양방향) · "λ\*\_thr(X)는 토큰 처리율 우위를 뜻한다" · "M3가 context 16384에서도 같다" · "preflight PASS가 rev12 runner를 검증했다"(R-5 이전) · 이번 반전 시험의 예보 수치(편향 ≤ 1.1%, BROKEN 29–33%, 부풀림 +3.45%)를 측정값으로 인용.

VR1 §7 허용 문안: "rev12 M3는 이 Job에서 shape A·B·X(arXiv-Sum, 입력 ≤ 16k, 출력 p50 195)의 λ\*\_thr 기반 G\_sym(N)을 잰다. 이는 같은 스케줄 노브에서 PD-mux 기능 전체의 효과이고 multi-GPU 1차·단일 GPU 2차 가족이다."

**승계 caveat (VR1 §6, 문자 그대로)**:

- **MG12-1** "prior\_X = 0.72는 rung 위치 전용 유도값이다. 일 보존 FIFO 모형 예보에서는 참 λ\*/prior ∈ [0.35, 2.0]이면 G 편향이 1.1% 이하이고, 2.8 이상이면 `LADDER_TOO_LOW`(fail-closed)다. 이 예보는 배치 엔진의 비-일보존성을 반영하지 않는다."
- **MG12-2** "shape X의 가변 길이는 LO g 산포를 키운다(모형 예보: 두 arm 기준 `KNEE_NOT_BRACKETED` 5–35%). X의 G 미정의는 효과 없음이 아니다."
- **MG12-3** "P10을 세 shape로 넓히면 귀무에서도 BROKEN 확률이 커진다(모형 예보: shape당 11–12%, 세 shape 독립 상한 29–33%). TP 라벨 인용 불가는 TP 효과 부재가 아니다."
- **MG12-4** "shape X의 Job 효과는 검출되지 않았다. X의 N-추세 라벨은 shape A 앵커 기준 Job 효과 검사 아래의 값이다."
- **MG12-5** "M3는 context\_length 20480과 mamba 노브 명시 인자로 돌았고, M2g PASS는 16384와 자동값에서 잰 것이다. 코드상 KV 풀·mamba 슬롯·커널 수치는 바뀌지 않는 것으로 유도되나, M2g PASS를 이 인자 조합이나 16k 입력의 정확성으로 쓰지 않는다."
- **MG12-6** (R-6 미채택 시) "N>1 config에서 floor 블록은 fused\_sym boot1에만 있어 G\_sym 두 arm의 boot 사전 조건이 N=1과 다르다. G\_sym(N) 추세에 이 비대칭 성분이 섞여 있을 수 있다."
- **MG12-7** "M3 λ\*\_SLO는 2 rung(비 약 2.7, HI는 과포화 비정상 셀) 위의 비등록 사후 기술값이다. R9-3을 충족하지 않으며 크기·부호 문장에 쓰지 않는다."
- **MG12-8** "M4의 A′ 등록은 shape A에서 T = k·floor(1k)이고 cliff 위치가 측정되지 않았다. M4 단계 감사 전에 M4 SLO 결과를 예측하거나 인용하지 않는다."
- **MG12-9** "하드캡 상향은 예산 결정이지 판정 규칙이 아니다. S3에서 TP가 먼저 잘리는 것은 등록 순서 [DP, TP]의 결과다."
- **MG12-10** "X의 λ\*\_thr는 요청 수 기준(req/s)이며 표본 길이 구성에 의존한다. N이 다르면 X 표본 문서 집합이 다르다."

비고: MG12-6은 R-6 채택(§19-4, config boot1의 두 arm 모두에 floor\_N 블록)으로 **이 판에서는 적용되지 않는다**. 원문은 VR1과의 바이트 대조를 위해 그대로 싣는다.

### 19-12. 하네스·공용 코드 변경 목록 (재봉인 대상)

| 파일 | 변경 | 회귀·변이 시험 | 시점 |
|---|---|---|---|
| `benchmarks/pdmux_eval/trace_loadgen.py`(공용) | R-2: SSE `error` 키 = 서버 측 오류, `OUTPUT_SHORT`, `sock_read=600` 유지 | "error 청크를 토큰으로 셈" KILLED + 무변이 대조군 | **POST 커밋 `<sha 미정>`** |
| `scripts/r2_eval/engine_bench_runner.sh`(공용) | `--max-prefill-tokens`·`--max-mamba-cache-size`·`--mamba-ssm-dtype` 전달 추가(`--context-length`는 이미 받음) | 기존 runner 시험 + 인자 출력 dry-run. runner 변경은 M2g 등록 digest 밖임을 공시 | **POST 커밋** |
| `scripts/multigpu/mg_loadgen.py`·`mg_runner.sh` | 순열 prefix 길이 쌍 trace, 요청별 (수락 여부, 오류 유형, 예정·실제 발송 시각) 기록 | 분류 불가 → 클라이언트 측 | **POST 커밋**(§19-6, U13) |
| `tests/` 아래 하네스 시험 | 아래 행들의 변이 시험 | — | **POST 커밋**(U13) |
| `mg_const.py` | `PREREG_REV = 12`, shape X 상수, 데이터 sha, 노브 5개, **`CONTEXT_LENGTH` 16384 → 20480(`:127`)**, `HARD_CAP_S["m3"]` 14/16 h, 새 중단 코드 | selftest 갱신 | 재봉인 |
| `mg_plan.py` | 3 shape, floor 블록 step(20개), 셀 수 단언(536), 예산 | "shape 2개로 계획" 변이 KILLED | 재봉인 |
| `mg_drive.py` | `:239` shape 튜플, floor 블록 실행·`FLOOR.json`, `PREFLIGHT_PASS`·로그 knob·dtype 대조 | 중단 경로 시험 | 재봉인 |
| `mg_label.py` | P10 3 shape, X를 비앵커 shape로, `CLIENT_ERROR` 클라이언트 측 분류, **`SERVER_ERROR_HI`**, `FLOOR_JOB_EFFECT`, `LOG_KNOB_MISMATCH`, `KV_POOL_DRIFT` | "서버 오류를 CLIENT\_ERROR로 셈"·"X HI 가장 긴 3% 서버 오류인데 라벨 그대로"·"`OUTPUT_SHORT`를 성공으로 셈"·"X 누락 P10"·"J1 floor로 채점" 변이 KILLED + 무변이 대조군 | 재봉인 |
| `m3_job.sh` | gate (i)–(iii) 대조(§19-6), `PDMUX_HEARTBEAT_S=0` 확인 | 셸 시험(gate 경로 diff 비지 않음 → ABORT) | 재봉인 |

### 19-13. 사용자 결정 항목

| # | 항목 | 선택지 | 상태 |
|---|---|---|---|
| U1 | rev12 개정 범위 | 이 초안 전체 / 일부(예: X 제외) | 전체로 진행(사용자 "진행" 2026-10-16) |
| U2 | 16k 입력의 context 처리 | (a) `--context-length 20480` / (b) 16384 유지 + 필터 | **결정 2026-10-16: (a)**. (b)는 실행 불가(§19-3) |
| U3 | 하드캡 | J1 14 h·J\_N 16 h / rev11 유지 / X 별도 Job | **결정 2026-10-16: J1 14 h, J2·J4·J8 16 h** |
| U4 | `PDMUX_LOG_MIRROR` 값 | 1 / 0 | 미결, 초안 권고 1(네 Job 동일) |
| U5 | J1 스펙 | x2·GPU 0 / x1 | 미결, 초안 권고 x2 유지 |
| U6 | M3와 G0 계측 패치(I-2–I-5) 순서 | M3 먼저 / 패치 뒤 | 미결, 초안 권고 M3 먼저 |
| U7 | 16k 입력의 M2g 범위 | 범위 공시만 / X prompt G1형 소형 정확성 단계 추가 | 미결, 초안 권고 범위 공시 |
| U8 | M4 A′ 가족 처리 | Bonferroni 98.33% / k=5만 판정 | M4 단계 감사에서 확정 |
| U9 | shape C | 범위 문장 / M3에 추가 | 미결, 초안 권고 범위 문장 |
| U10 | TP GPU당 decode SM | 미해결 공시 / N별 D 스윕 | 미결, 초안 권고 미해결 공시 |
| U11 | λ\*\_SLO 위치 | 봉인 밖 사후 기술값 / 하네스 등록 | 미결, 초안 권고 봉인 밖(분석기 fetch 전 커밋) |
| U12 | P10 3 shape 확장의 검정력 비용 | fail-closed 확장 / X를 P-DP 판정에서 빼고 기술값 | 미결, 초안 권고 fail-closed 확장 |
| **U13** | gate 경로에 닿는 rev12 하네스 변경의 시점(§19-6) | (a) `scripts/multigpu`·`tests/` 하네스 변경을 POST 커밋에 함께 / (b) 재봉인 커밋에서 preflight 재실행(≈1 GPU-h [유도]) | **새 항목**, 초안 권고 (a) — 2회차 감사가 그 경로를 다시 고치면 (b)로 떨어짐(fail-closed) |
| **U14** | X Job 효과 앵커(VR1 Q5 권고) | 넣지 않음 / J\_N 앵커 boot에 X HI 4 seed(Job당 ≈0.5 h [유도]) | **새 항목**, 초안 권고 넣지 않음(MG12-4 공시) |

### 19-14. claims-auditor 재감사 질문 (규칙층, 2회차 — r0 대비 diff만)

- Q1 (R-1): §19-1b `SERVER_ERROR_HI`가 D-1 반전 경로(X HI 셀의 선택적 서버 측 오류로 achieved 부풀림)를 닫는가. 확장 셀 포함·전 shape 적용·LO 기술값 처리가 새 fail-open을 만들지 않는가. D-7과의 관계 서술이 사용자 승인 해석과 맞는가.
- Q2 (R-2): 공용 loadgen 커밋 `<sha 미정>` 등록 요구(a)–(d)가 VR1 §4-1의 조용한 1토큰 성공 경로를 덮는가. 봉인 전 sha 확정 의무가 충분한가.
- Q3 (R-3·R-4): context 20480 근거 수치, U2(b) 실행 불가 서술, G5(c)(ii) X 항, `mamba_ssm_dtype` 규칙(키 부재 → float32, 폴백 금지, boot 대조)이 VR1 §4·Q8과 맞는가.
- Q4 (R-5): gate (i)–(iii)과 대조 경로 7개(`benchmarks/pdmux_eval`·`scripts/multigpu` 추가, `results/multigpu` 제외)가 "바뀐 파일을 덮지 않는 gate" 결함을 닫는가. U13의 순서 제약이 fail-closed인가. §19-9 분석기 위치가 gate와 충돌하지 않는가.
- Q5 (R-6·R-8): floor\_N 두 arm 배치로 MG12-6 비대칭이 실제로 사라지는가. `[1]*L` 합성 입력과 순열 prefix 중첩이 새 교란(shape 간 내용 차이, N 간 표본 상관)을 만드는가.
- Q6 (범위): r1이 수리 색인 밖에서 rev11·r0 판정 의미를 바꾼 곳이 있는가(r0 sha256 `29dd7d2f…` 대비 diff).

### 19-15. 이 초안이 하지 않은 것

- rev11 PREREG·하네스·PASS 표·명세·정본·트랙 파일 수정, 커밋, GPU 사용, 제출. 예산 표는 scratchpad의 하네스 사본 계산이다(봉인 하네스는 읽기와 `job-table` 출력만).
- 공용 loadgen·runner 수정(engine-porter 진행 중, sha 미정), preflight 도구의 POST 기본값(`preflight_cl3.sh:30` `dc06c870`)·`post_vs_job_head` 경로 목록(`:136-137`, 5경로) 갱신 — 도구 수정은 이 문서 밖.
- M4의 나머지 세부(shape 집합, ρ 재검토), G0 PREREG와의 공용 floor 코드 확정.
