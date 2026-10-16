# 판정서: M3 PREREG rev12 초안 — 규칙층 감사 1회차 (2026-10-15)

> claims-auditor(읽기 전용) 최종 보고를 main 세션이 저장(2026-10-15). 감사 상한 2회 중 1회차. 저장소 변경 0, 계산은 scratchpad(`…/scratchpad/s/`)에서만.
> 대상: `PREREG_MULTIGPU_rev12_DRAFT_2026-10-15.md`(미추적, 246줄). 기준 rev11 `PREREG_MULTIGPU_2026-10-12.md` sha256 `8b617d3d59d6cb63…` 직접 대조 일치.
> 원자료: arXiv-Sum test parquet(sha `b34529df…` 일치), Nano-9B-v2 토크나이저(sha `3277c00f…` 일치) — 다른 세션 scratchpad 사본, 읽기만. 엔진 트리 `~/Experiments/KISTI/sglang_engine_dev/python/sglang/srt/`.

## 1. 총평

| 대상 | 등급 | 死因 |
|---|---|---|
| **rev12 초안** | **NO-GO** | **1건, D-1(N2)**: C5와 C2가 겹쳐 shape X의 λ\*_thr가 서버 측 오류로 부풀 수 있다(fail-open) |

D-1만 고치면 다음 판정은 `GO-with-caveats` 후보. 봉인 전 R-2–R-5도 필수(사실 오류·gate 범위 결함, 死因 아님). 2회차는 이 판정 이후 diff만, 死因이 남으면 처방 없이 "사용자 결정 필요".

## 2. 반전 시험 표

길이: arXiv-Sum 유지분(입력 ≤ 16,384) 5,906건의 실제 (입력, 출력). 작업량: 초안 §19-2 가산식 그대로 w_i = I_i/(0.690·8192) + O_i/(2.99·512), 평균 1.391 s → prior 0.719. 대기열: 일 보존 단일 서버 FIFO(배치 엔진은 비-일보존이라 이 모형 편향은 하한). g·achieved·확장·LO 규칙: rev11 §4-2·4-3과 `mg_label.analyze_cell` 정의, seed 4, 반복 60.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| prior_X 오류 → rung → λ\*_thr | 참 λ\*/prior m ∈ {0.35…3.2}, G_true ∈ {0.97, 1.00, 1.03}, N=1 | 반전 없음 | m ≤ 2.0: `KNEE_BRACKETED` G 편향 평균 ≤ 0.08%, 최대 1.09%(m=1.5). m=2.8: 73–92% `LADDER_TOO_LOW`, m=3.2: 100%(fail-closed). 대가: LO `KNEE_NOT_BRACKETED` 두 arm 기준 m=1에서 3–5%, m ≤ 0.7에서 15–35% → MG12-1·2 |
| **C5: λ\*_thr에서 서버 측 오류(셀 유효 유지)** | 한 arm X HI 셀 서버 측 오류 f ∈ [0,5]%, 가장 긴 문서부터. 등록 상한 없음 | **반전** | achieved 부풀림(n_ok/span): f=1% +1.18%, 2% +2.33%, **3% +3.45%**, 5% +5.66%, 10% +11.1%. 참 G_sym(N,X)=0.97 → 측정 1.0035(3% 띠 안, 크기 라벨 반전). 참 G(8)=0.96, G(1)=1.00 → 측정 1.014로 `G_DECREASES` → `NO_TREND`/`G_INCREASES`. rev11은 오류 >1%면 `CLIENT_ERROR`로 짝 무효(fail-closed). **死因 D-1** |
| `CLIENT_ERROR` 짝 무효화 | 클라이언트 측 오류를 한 arm에 | 반전 없음 | 짝을 함께 지움. `lambda_star`는 4 seed 모두 OK여야 정의(`mg_label.py:573-579`). V2 B-3 재확인 |
| 하드캡 상향 절단 순서 | 10/14 h → 14/16 h, S1–S3 | 반전 없음 | 순서 상수 [DP, TP] 불변(§17-2(a)). J1에서 `G1_UNDEFINED`가 정의됨으로. 초안 S3 수치(J2 26.0–27.3 h)로 DP 구간 ≈13.4 h + 앵커 < 16 h, TP만 잘림 → MG12-9 |
| P10 shape 2 → 3 | 귀무, σ ∈ {1,2,3}%, seed 공동 bootstrap | 반전 없음, fail-closed 비용↑ | 거짓 방향 라벨 shape당 10.8–12.5%(명목 5% 초과, n=4 백분위 bootstrap). BROKEN: 셋 29–33%(독립 상한), 둘 20–23%. 크기 3%까지 걸면 shape당 0–9.5% → MG12-3 |
| `context_length` {16384, 20480} | 두 값 | G 판정 반전 없음(같은 trace라 대칭) | 16384: 유지분 21/5,906(0.36%)이 입력+출력 ≥ 16384 → 스트리밍 경로에서 **조용한 1토큰 "성공"**(§4). 20480: 0건(최대 16,744). 16384에서 floor(16k) 측정 불가 → R-2·R-3 |

## 3. Q1–Q13

- **Q1 PLAUSIBLE.** DRAFT 명세 바이트를 sha로 고정 인용은 정당(SP v0.2 커밋됨, promote 비추종은 보수적). 누락: **R5-5**(n=4 → 재표집 35개뿐 공시)가 표에 없음 → "적용(공시)" 행 추가(R-7). R4-2 비적용은 SP 문언과 다르나 HI는 의도된 비정상 셀이라 공시로 충분. R7-3 "클라이언트 측만"은 SP §4 multi-GPU 차이 문장(λ\*_SLO rung 무효화 맥락)을 λ\*_thr에 옮긴 것이라 D-1을 만든다(Q2). J1 floor가 fused_sym boot1 시작부라 arm 순서 무작위상 첫 boot가 아닐 수 있음 — M3는 SLO 판정 아니므로 공시로 충분.
- **Q2 REFUTED, 死因 D-1(N2).** `achieved = len(ok)/span`(`mg_label.py:424-429`)은 고정 길이 A·B에서 서버 측 오류에 중립(빠른 오류: span↓)이거나 벌(느린 오류). 가변 길이 X에서는 긴 요청이 선택적으로 빠지면 req/s↑. rev11은 `CLIENT_ERROR`(모든 오류 >1%)로 지웠고 C5가 이를 X에 대해 연다. 서버 과부하가 클라이언트 측(연결 거부·발송 지연)으로 보이는 경우는 짝 무효라 fail-closed — 반증 못 함. 처방 R-1.
- **Q3 PLAUSIBLE.** g = (Q_c 기울기)/r은 요청 단위 (x−μ)/x라 기댓값은 길이 무관, 가변 길이는 산포만 키움. LO `KNEE_NOT_BRACKETED` 5–35% fail-closed. prior 오류 범위에서 fail-open 0. 확장 n은 N=8에서 3,140 ≤ 5,906 → 비복원 가능.
- **Q4 PLAUSIBLE.** BROKEN 둘 20–23%, 셋 29–33%(모형 상한). fail-closed라 수용 가능, "TP 인용 불가 ≠ TP 효과 없음"(MG12-3) 의무. 1차 원인은 n=4 백분위 bootstrap 반보수성(rev11 기존 성질, diff 아님).
- **Q5 PLAUSIBLE.** X의 Job×arm 상호작용 미검출(B도 rev11에서 같은 지위). 원자료 없어 반전 불가 → MG12-4. 권고(필수 아님): J_N 앵커 boot(fused_sym, agnostic)에 X HI 4 seed 셀(추가 boot 0, Job당 8셀, ≈0.5 h).
- **Q6 PLAUSIBLE.** `FLOOR_JOB_EFFECT`를 λ\*_thr 밖 공시로 두는 것은 R10-2와 맞음. floor prompt(절단 문서, `max_new_tokens` 8)는 R6-2 취지와 맞음. 결함: (1) 16k 점은 ctx ≥ 16,393 필요(16384+8 ≥ 16384 → `ValueError`), U2(b)면 깨짐. (2) config boot의 floor_N 블록이 fused_sym boot1에만(14블록 = J1 2 + 앵커 2×3 + config 6) → N=1에서는 두 arm 대칭, N>1에서는 비대칭 → **N과 공변**(교란 #10 계열). 반전 수치 없어 死因 아님 → MG12-6, 권고 R-6.
- **Q7 PLAUSIBLE(조건부).** shape A T = k·floor(1k)는 k=3에서 수백 ms로 cliff 위험 — 측정 전이라 판정 불가 → M4 단계 감사(MG12-8). Bonferroni 98.33%는 k 3개 FWER를 닫음(보수적). M3 λ\*_fused를 rate 위치에만 쓰는 것은 rev11 §5와 같고 같은 Job 원칙 위반 아님. Job 효과가 ρ 실제 위치를 흔듦(공시).
- **Q8 PLAUSIBLE, R-4 필수.** 20480이 A·B에 주는 영향(코드): KV 풀 `max_total_num_tokens` = rest_memory/cell_size, context 무관(`model_runner_kv_cache_mixin.py:170-194`). mamba 슬롯 = max_running(radix off, `:221-228`), `_calculate_mamba_ratio`는 radix off에서 1. max_num_reqs는 명시 48·48N(estimated 하한 2048, `:861-868`). 커지는 것은 req_to_token 풀(행 ctx+4)과 flashinfer cudagraph `kv_indices`·mask(`max_num_tokens × ctx`)뿐, 각 25%↑ MB 단위(N=1 ≈4–26 MB, TP8 ≈31–52 MB), KV 산정 뒤 비-KV 예비분에서. A(768)·B(8,256)는 `max_req_len` 클리핑 무관. M2g PASS는 ctx 16384, `mamba_ssm_dtype` None에서 잼(M2g `server_info.json` 전수: KV 2.72 M(N=1), cuda_graph_max_bs 256/512) → 서버 인자 조합이 PASS 범위 밖(MG12-5). **R-4**: Nano-9B-v2 HF config에 `mamba_ssm_dtype` 키 없음(`max_position_embeddings` 131072, `torch_dtype` bf16) → 엔진 기본 float32(`configs/mamba_utils.py:45-105`). 초안 "HF config 값을 읽어 고정"은 정의되지 않음; `torch_dtype` 폴백이면 bf16으로 M2g와 처치가 달라짐. `-Base` revision config 키 유무 미확인.
- **Q9 REFUTED(gate 범위), 死因 아님, R-5 필수.** (i) "설계 문서가 저장소에 없다"는 거짓(`d138e7f`). (ii) preflight 설계 §6 공용 조건 = 이미지 digest 동일 + 제출 커밋의 `{src, tests, scripts/bootstrap, scripts/r2_eval, env}`가 POST(`dc06c870`)와 바이트 동일. rev12는 `scripts/r2_eval/engine_bench_runner.sh`(CL-3 17건의 대상)를 바꾸는데, 초안 `ABORT_PREFLIGHT_NOT_PASSED`는 엔진 digest만 대조 → runner 변경 전 PASS를 그대로 받음. **gate가 바뀐 파일을 덮지 않는다.** (iii) "17건 검증 불가 → 제출 보류"는 fail-closed로 충분.
- **Q10 PLAUSIBLE.** 우회 경로: "TP에서 G_sym↑ → 분할(agnostic 표·D44)이 TP에 더/덜 맞는다", "TP가 decode 보호를 돕는다". 추세 라벨 자체는 §19-7 범위 문장을 붙일 때만(§7 금지 문안).
- **Q11 PLAUSIBLE, 권고 R-9.** 회수 뒤 분석기 작성은 창·셀 포함·보간 선택 자유를 엶. 제목 금지만으로는 본문 인용 경로가 남음 → 분석기를 M3 fetch 전 커밋(sha). HI는 과포화 비정상 셀이라 attainment가 창 길이 의존(MG12-7).
- **Q12 PLAUSIBLE.** J_N 16 h는 S1·S2를 덮어 rev9 "예상치 기준"과 맞음. J1 14 h는 S3 기준(P9-a 근거 있음)이나 rev9 기준 이탈 → 사용자 결정 U3. 절단은 등록 순서대로만. 수치는 초안 scratchpad 계산이라 재현 못 함 — 봉인 dry-run(`mg_plan.py job-table`·`m3-forecast`)이 "S3에서 DP 구간 종료 < 캡"을 출력해야 함.
- **Q13 REFUTED(부분).** §19-1 "판정 절차 rev11 그대로"는 사실상 거짓: C5가 λ\*_thr 셀 유효성을 바꿈(D-1), floor 블록이 판정 boot 사전 조건을 비대칭으로 바꿈(MG12-6), `mamba_ssm_dtype` 고정 규칙이 처치를 바꿀 수 있음(R-4). shape 순서 3원소화는 rev11 예보 비교만 바꾸고 판정은 불변.

## 4. context-length 문제 검증

1. **사실: CONFIRMED(코드), 초안보다 나쁨.** `_validate_one_request`(`tokenizer_manager.py:785-834`)는 입력 ≥ ctx, 또는 `validate_total_tokens`(`:252` True)에서 입력 + max_new ≥ ctx일 때 `ValueError`(`allow_auto_truncate` 기본 off). **스트리밍 요청에서는 HTTP 200 + `data: {"error": …}` 청크 + `[DONE]`**(`entrypoints/http_server.py:665-676`). `trace_loadgen.py:53-89`는 `stream: True`이고 `meta_info` 없는 청크를 `completion_tokens = last+1`로 셈 → 거절 요청이 **TTFT = 거절 지연(ms), 출력 1토큰, error = ""인 성공**으로 기록. `mg_label.is_error`는 못 잡고, 하네스에 `observed_output_tokens` 대 `requested` 검사 없음(grep 0). 스케줄러는 `max_new_tokens`를 `max_req_len − 입력 − 1`로 조용히 깎음(`scheduler.py:1636-1644`, `tp_worker.py:301-305`).
2. **G0에도 같은 문제: CONFIRMED.** SP §10은 `context_length` 미등록, runner 기본 16384. 유지분 21/5,906(0.36%)이 조용한 1토큰 성공(셀 n=200이면 셀당 0.7건). **R6-2 floor 16k 점이 거절 지연(ms)으로 오염 → 12k–16k 요청의 T가 거의 0.** G0 PREREG(Q24)는 ctx ≥ 16,745(최대 입력+출력 16,744)와 "floor 최대 길이 + max_new < ctx"를 등록해야 함.
3. **20480의 영향.** A·B: 코드상 KV·슬롯·cap·커널 불변(Q8). M2g: 서버 인자가 달라 PASS를 그대로 승계 불가(MG12-5). 권고: M2g 같은 (config, arm)과 `max_total_num_tokens` 기술 대조, 차 > 0.1%면 플래그(M2g 회차 간 산포는 수 토큰). 20480은 X 최대 16,744와 floor 16,392를 덮고 여유 3,736. **U2(b)는 쓰인 대로 실행 불가**(floor 16k 깨짐; 필터 "입력+출력 < 16384"는 16,383을 받는데 스케줄러가 1토큰 깎음 → ≤ 16,382여야). 정정: runner는 `--context-length`를 이미 `PDMUX_CONTEXT_LENGTH`/record 필드로 받음(`engine_bench_runner.sh:96-97, 140`), §19-12 "전달 추가"는 다른 노브 3개만 해당. `mg_const.CONTEXT_LENGTH = 16384`(`:127`)도 바꿔야.

## 5. 필수 수리

- **R-1 (D-1, 필수).** λ\*_thr 판정에 한해: "HI 셀 서버 측 오류 비율(분모 = 셀 요청 수) > 1%면 `SERVER_ERROR_HI`, 그 arm λ\* = `UNRESOLVED` → G 짝 무효(`CLIENT_ERROR`와 같은 대칭 무효)." 1%는 rev11 §4-3 승계 상수(새 상수 아님), 최악 선택에서도 부풀림 상한 +1.18% < 3%. λ\*_SLO·attainment에서 서버 측 오류 = 불합격(SP R7-3, D-7)은 유지. ⚠️ 사용자가 D-7을 "M3 λ\*_thr에도 문자 그대로"로 확정하면 충돌 → 2회차에서 "사용자 결정 필요". 변이: "한 arm X HI 셀 가장 긴 3%가 서버 측 오류인데 라벨 그대로" KILLED, 무변이 대조군 통과.
- **R-2 (필수, M3·G0 공용).** loadgen이 SSE `"error"` 키를 서버 측 오류("수락 뒤 오류")로 기록, `observed_output_tokens ≠ requested`(ignore_eos)를 서버 측 오류 `OUTPUT_SHORT`로 분류. 변이 "error 청크를 토큰으로 셈" KILLED. 이 수리 없이는 C5·R-1의 서버 측 분류가 이 경로를 못 봄.
- **R-3 (필수).** U2 = (a), 근거 수치 등록(최대 입력+출력 16,744, floor 16,392 → ctx ≥ 16,745). U2(b) 실행 불가 명기. G5(c)(ii) X 항 = "16,384 + 616(유지분 최대 출력) [측정]". G0 PREREG에도 같은 값 연결.
- **R-4 (필수).** `mamba_ssm_dtype` = "고정 revision `config.json`에 키가 없으면 float32(엔진 기본 = M2g 실효값), 있으면 그 값. `torch_dtype` 폴백 금지." 실효값은 boot마다 `server_info.json`과 대조.
- **R-5 (필수).** M3 제출 gate = `PREFLIGHT_PASS` ∧ 이미지 digest 동일 ∧ `git diff --stat <preflight POST> <M3 제출 커밋> -- {src,tests,scripts/bootstrap,scripts/r2_eval,env}` 빈 출력. runner 변경 커밋을 preflight POST로 삼도록 순서 지정(preflight 설계 §2·§6 개정 필요). C7 "저장소에 없음" → `d138e7f` 정정.
- R-6 (권고; 미채택 시 MG12-6 의무): floor_N 블록을 config boot1에서 fused_sym과 agnostic 둘 다에(+6 블록, ≈0.27 h), 또는 판정 boot 밖 전용 짧은 boot로.
- R-7 (권고): §19-1 표에 R5-5 행(적용, 공시). "판정 절차 그대로" 문장에 C5·floor·dtype 영향 명시.
- R-8 (권고): X 표본은 seed별 순열 prefix로 뽑아 N 간 중첩 부분집합이 되게 등록. 입력이 `[1]*L` 합성 토큰인지 실제 문서 토큰인지 등록.
- R-9 (권고): λ\*_SLO 분석기를 M3 fetch 전 커밋·sha 기록.

## 6. 승계 caveat (인용 금지 문장, 문자 그대로)

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
- 승계 유지: rev11 §7·§15–§18, MGP-\*, MGH-\*, SP S-1–S-8·K1–K9, 판정서 C-1–C-6·CL-1–CL-3.

## 7. 허용·금지 문안

허용: "rev12 M3는 이 Job에서 shape A·B·X(arXiv-Sum, 입력 ≤ 16k, 출력 p50 195)의 λ\*\_thr 기반 G\_sym(N)을 잰다. 이는 같은 스케줄 노브에서 PD-mux 기능 전체의 효과이고 multi-GPU 1차·단일 GPU 2차 가족이다."

금지(§19-11에 추가): "TP에서 PD-mux 분할(agnostic 표·D44)이 더/덜 적합하다", "G\_sym(N) 추세는 SM 분할 크기 때문이다"(양방향) · "λ\*\_thr(X)는 토큰 처리율 우위를 뜻한다" · "M3가 context 16384에서도 같다" · "preflight PASS가 rev12 runner를 검증했다"(R-5 이전) · 이번 반전 시험의 예보 수치(편향 ≤ 1.1%, BROKEN 29–33%, 부풀림 +3.45%)를 측정값으로 인용.

## 8. 요약

rev12 초안 = **NO-GO**, 死因 D-1(C5가 X의 λ\*_thr를 서버 측 오류로 부풀림; 3%에서 +3.45%). 처방 R-1, 봉인 전 필수 R-2–R-5, 권고 R-6–R-9. 반증 못 한 것: prior_X, `CLIENT_ERROR` 짝 무효, 하드캡·절단 순서(단 caveat 없이는 CONFIRMED 아님). 2회차는 R-1–R-5 diff만. R-1이 D-7 범위와 충돌한다고 사용자가 판단하면 2회차에서 바로 "사용자 결정 필요".
