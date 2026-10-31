# 기본 서빙 구조(stock SGLang · stock vLLM) 기준선 지표 — 인벤토리와 관측 계획 (2026-11-01)

> **지위**: 계획 문서. GPU 지출 0 · 새 측정 0 · 새 판정 0. 정본(PROJECT_STATUS·CONSENSUS)을 바꾸지 않는다.
> 결정이 필요한 항목은 §6에 모았고, 반영은 main 세션 promote 몫이다.
> 목적: PD-mux 결과(CAP1-R·CAP2·TUNE·policy_implfix)와 나란히 놓을 **기본 서빙 구조의
> 시간·처리량·TTFT·TPOT·r**을 먼저 확보한다.

## 0. 지표 정의 (모든 행에 같은 정의 적용)

| 지표 | 정의 | 원자료 요구 |
|---|---|---|
| 시간 | (a) run duration — 라운드 합산(게이트 7, `max()` 금지) (b) 요청 E2E latency p50/p95/p99 | 요청별 send·finish 시각(없으면 TTFT+ΣITL로 재구성, 그 사실 공시) |
| 처리량 | req/s · output tok/s · total tok/s (완료 요청 기준, duration 합산 분모) | 요청별 입출력 길이 |
| TTFT | 요청별 첫 토큰 시각 − send, p50/p95/p99 | 요청별 `ttft` |
| TPOT | 요청별 (E2E − TTFT)/(out − 1), p50/p95/p99. ITL p95(요청 내부)는 goodput 술어용으로 병기(게이트 4) | 요청별 `itls` |
| goodput | TTFT ≤ SLO AND 요청 내부 ITL p95 ≤ SLO(게이트 4), 용량 먼저 재고 cliff 회피(게이트 6) | 위 전부 |
| **r** | 아래 §0-1 | forward 단위 GPU 시간 |

### 0-1. r은 기본 구조에서 그대로 정의되지 않는다 — 대응량 두 개를 따로 이름 붙인다

PD-mux의 `r_span` = (prefill 배치 안 최대 `chunk_gpu_ms`) ÷ (짝지은 `decode_fwd_gpu_ms` 중앙값)
(`policy_implfix/PREREG_POLICY_IMPLFIX_rev1_2026-10-31.md` §5). 이때 span은 **다른 SM에서 decode와
동시에** 돈다. 기본 구조에는 span도 SM 분할도 없으므로 같은 이름을 쓰면 안 된다.

- **`r_ser`(stock SGLang, mixed-chunk OFF)**: prefill(EXTEND) forward와 decode forward가 **같은 stream에서
  번갈아** 돈다. r_ser = (두 decode 사이에 낀 EXTEND forward GPU 시간 합) ÷ (인접 decode forward GPU 시간
  중앙값) — **decode가 그만큼 직렬로 멈춘다**는 뜻. Gate 2 HOLB 계측의 `same_stream_fw_ms / decode_dur_ms`와
  같은 양(`p1_gates/gate2/DIRECT_BLOCKING_DESIGN.md` §1, CUDA event).
- **`r_mix`(SGLang `--enable-mixed-chunk`, vLLM V1 chunked prefill)**: prefill 토큰과 decode 토큰이 한 forward에
  섞인다. r_mix = (T_mixed − T̂_dec(bs)) ÷ T̂_dec(bs), T̂_dec(bs)는 같은 bs의 decode-only step 중앙값.
- 세 값(r_span·r_ser·r_mix)은 **한 표에 같은 열로 합치지 않는다.** 비교는 "decode 1 step 대비 prefill이
  얼마나 끼어드는가"라는 기술 수준까지만.

## 1. 이미 있는 것 (탐색 결과, 2026-11-01)

| # | 원천 | 기판·모델·cudagraph | 있는 지표 | 원자료(로컬) | 쓸 수 있는 범위 |
|---|---|---|---|---|---|
| V1 | `vllm_bench/results/real_vllm.csv` · `results/serving-eval/` | KISTI · vLLM 0.22.1 · Zamba2 1.2B/2.7B/7B·Falcon-H1-3B · random 2048/128 · rate 2/8/inf · n=1 | req/s·out tok/s·TTFT/TPOT/ITL med·p99. ShareGPT json은 E2E만(TTFT 없음) | csv·json·log 있음 | **이력 전용** — 근거 문서가 `deprecated/`, Zamba 배제(결정 46), Nano 없음, n=1 |
| S1 | P1-opint `p1_opint/P1_OPINT_RESULT_2026-08-05.md` §3.1 | KISTI · Zamba2(triton)·Granite-micro(flashinfer) · ON | goodput·req/s·out tok/s·TTFT p50/95/99·TPOT p50/95, raw에 duration·E2E·ITL·요청별 ttft/itl | jsonl 24개 로컬(untracked) | 정상상태 셀만 인용(CONSENSUS 행 1). fused **미조율**. Zamba는 인용만 |
| S2 | Gate 2 HOLB/E-A/2-S `p1_gates/gate2/` | KISTI · Zamba2·Granite · ON · n=10 | X_60·ITL p95·TTFT p95 + **forward GPU 시간** | `*.holb.jsonl` 853개(Granite 180) 로컬 | **r_ser 재계산 가능, 아무도 안 했다** |
| S3 | cp_baseline G1·V-probe·W1·AF1 | KISTI · Zamba2 / Nano(W1·AF1) · ON | attainment·SD·knee·포화 req/s | jsonl 로컬 | "판정 0" 문서들, arm×node 교락 |
| S4 | M3 J1 `multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md` | **VESSL · Nano** · ON · fused(chunk 8192, overlap)/fused_sym | λ*_thr·ITL p95 + 요청별 send/completion/itl | `multigpu/manual_retrieval/` 17 GB 로컬 | N=1·저검정력, HI TTFT 인용 금지. **현 기판·모델의 유일한 fused 서빙 원자료** |
| S5 | engine gate β `BETA1_fused/decode_probe.jsonl` · 비용 항 `108v` | VESSL · Nano · ON | decode forward GPU 시간(bs×ctx), solo prefill 시간 | 로컬 | 서빙 아님 — r 분모(T_dec) 사전값으로만 |
| S6 | probe_128k | VESSL · Nano/Llama-8B | 단일 요청 TTFT·OOM | tracked | probe |

**공백**: ① VESSL·Nano·W1/W3·변화 trace에서 stock 구조 서빙 측정 0 ② 조율한(tuned) fused 0 ③ 현 기판의 vLLM 0
④ r_ser·r_mix 측정 0(r_span도 implfix Job 회수 전) ⑤ total tok/s는 요약표에 없음.

## 2. Phase A — CPU에서 먼저 뽑는다 (GPU 0, 이 머신)

| 단계 | 할 일 | 산출 |
|---|---|---|
| A1 | 공통 추출기 `stock_baseline/extract_stock_metrics.py`: bench_serving jsonl·M3 per-request·policy telemetry를 받아 §0 지표를 같은 코드로 계산(duration 합산, 요청 내부 ITL p95). 열에 `substrate·engine·arm·model·workload·rate·n·인용상태` | `STOCK_METRICS_TABLE.csv` + 회귀 테스트(합산 vs max 변이 검출) |
| A2 | S1(Granite·Zamba2 fused/agnostic n=5) 재집계 → 원문 §3.1 수치와 대조(재현 실패 시 중단·보고) | 표 KISTI 블록 |
| A3 | S2 HOLB에서 **r_ser 분포**(Granite 우선, Zamba2는 이력 열) — 배치별 r_ser, rate별 p50/p95, r_ser>1 비율 | `R_SER_HOLB.md`(기술값) |
| A4 | S4 M3 J1 fused·fused_sym을 같은 추출기로 → VESSL·Nano의 기본 구조 첫 행(N=1 공시) | 표 VESSL 블록 |
| A5 | S5 decode_probe로 T_dec(bs, ctx) 표 → r_ser·r_mix 예보(analytic-v1 `analytic_terms.py`의 prefill 시간 ÷ T_dec) — Phase B에서 맞는지 채점할 사전 예측 | 예보표(봉인 후 Phase B) |
| A6 | 결과 분석 result-analyst → 인용 문안 claims-auditor 1회 | — |

KISTI 행과 VESSL 행은 **같은 표에 두되 섞어 비교하지 않는다**(기판 주의).

## 3. Phase B — 없는 것은 이렇게 관측한다 (GPU, 사전등록 뒤)

### 3-1. 설계 골격 (PREREG `STOCK-BASE` rev0, 하네스 감사 1회 → 봉인 → dry-run → 승인 → 제출)

- **기판**: VESSL A100 SXM 80GB ×1 (implfix·CAP 계열과 같은 기판). **arm은 한 Job 안**.
- **모델**: Nemotron-Nano-9B-v2(주 모델). Llama-8B 참조점은 선택(§6).
- **arm**
  - `SG-stock`: SGLang v0.5.10 **패치 경로 미사용**(PD-mux OFF), 기본 chunked prefill·스케줄러, flashinfer, cudagraph ON.
  - `SG-mix`: 위 + `--enable-mixed-chunk`(r_mix 관측용; 기본값 아님을 명시).
  - `VL-stock`: vLLM stock(무패치), 격리 venv/이미지, 실제 선택된 attention 백엔드·CUDAGraphMode를 로그에서 채취·병기.
  - `PM-ref`: PD-mux static D44(기존 CAP·implfix와 연결하는 다리 arm, 같은 Job).
- **워크로드**: CAP 계열과 같은 W1·W3 정의·seed. 순서 = ① arm별 용량 사다리(λ* 먼저, 게이트 6) ② 각 arm λ*의
  {0.5, 0.8}× 고정 rate 셀 ③ 변화 trace LO↔HI 3라운드 — seed n≥4 paired(게이트 2·3). 엔진 비교라 정상상태 셀도
  허용하되 어느 셀이 무엇의 근거인지 PREREG에 못박는다.
- **클라이언트 하나로 통일**: 두 엔진 모두 OpenAI 호환 `/v1/completions`에 같은 하네스 클라이언트(bench_serving,
  vLLM에는 `--backend vllm`)를 붙여 TTFT·ITL 측정 지점을 같게 한다. 요청별 send·first-token·finish 시각 기록.
  측정 전 **정의 일치 smoke**: 같은 고정 요청을 두 엔진에 보내 클라이언트 TTFT와 서버 로그 시각 차가 같은 범위인지 확인.
- **r 계측은 별도 계측 boot**(측정 arm과 분리, PC boot 선례 C5 — 계측 boot 수치는 goodput 근거 아님):
  - SGLang: `decode_probe.py`는 DECODE·MIXED forward만 재고 EXTEND는 개수만 센다(`forward_mode_kind`).
    → **EXTEND 시간 기록을 기본 OFF 플래그로 추가**(engine-porter, CPU 테스트·변이 검출, digest 갱신). 그러면 r_ser·r_mix를
    같은 코드로 얻는다.
  - vLLM: 무패치 원칙이므로 소스를 고치지 않는다. 계측 boot에서 torch profiler 창(`/start_profile`~`/stop_profile`) 또는
    nsys로 step별 GPU 시간을 얻고, step의 prefill·decode 토큰 수와 짝지어 r_mix 계산. 관측자 비용은 측정 arm과 분리해 공시.
  - PD-mux `PM-ref`의 r_span은 implfix 계측과 같은 정의 재사용.

### 3-2. 선행 작업 (GPU 전)

1. vLLM 버전 고정 → 의존성 채취(`pip freeze`), VESSL 이미지 구성 확인(Workspace bring-up 1회, 측정 아님).
2. Nano 체크포인트가 그 vLLM 버전에서 부팅·정상 출력하는지(고정 프롬프트 출력 비교 가드).
3. decode_probe EXTEND 플래그 구현·CPU 회귀(engine-porter).
4. A5 예보 봉인.

### 3-3. Job 크기 (사용률 계산은 dry-run 때 확정)

GPU 1장 순차: arm 4 × (사다리 ≈10단 + 고정 셀 2 × seed 4 + 변화 trace 1 × seed 4) + 계측 boot 3. 부팅 수가 커지므로
**Job 2개로 분할 권고**: (가) 용량 사다리 + 정의 일치 smoke + 계측 boot (나) 고정 셀·변화 trace(λ* 회수 뒤 rate 확정).
각 Job 안에서 arm 교차 순서(ABBA)로 시간 drift 상쇄.

## 4. 지켜야 할 것

- 게이트 1: vLLM·SGLang stock 수치도 서빙 실증만. 5060 Ti 로컬 수치는 배관 확인용.
- KISTI 기판·Zamba 결과를 VESSL·Nano 결과와 섞어 비교하지 않는다. vLLM 0.22.1(V1)은 이력 열.
- fused 미조율을 "기본 구조 baseline"이라고만 부르고 "최강 baseline"이라고 쓰지 않는다(조율 arm 없으면 공시).
- 3% 미만 차이는 headline 아님, n≥4 없이 엔진 우열 서술 금지.
- 계측 boot의 r·관측자 비용 수치를 goodput 근거로 인용하지 않는다.

## 5. 순서

A1 → A2(재현 대조) → A3·A4·A5 병행 → result-analyst → (사용자 결정 §6) → PREREG → 하네스 감사 1회 → 봉인·dry-run →
승인 → Job (가) → 회수 → Job (나) → 분석 → claims-auditor 1회 → `handoff 결정` → promote.
Phase A는 implfix 4 Job 회수·분석과 독립이라 지금 시작할 수 있다.

## 6. 사용자 결정 대기

1. **vLLM 버전**: 제출 시점 최신 stable(권고 — 약한 대조군 반박 방어) / 0.22.1(과거와 연속, 단 V1은 이력 전용이라 이득 작음)
2. **조율 fused arm 포함 여부**: 미포함(권고: 1차는 기본 구조 그대로, 조율은 G0·M3 감사 요구와 묶어 2차) / 포함
3. **Llama-8B 참조점**: 2차로 미룸(권고) / 1차 포함
4. **decode_probe EXTEND 플래그 추가**(엔진 digest 변경): 진행(권고) / r_ser를 HOLB 재계산(KISTI)만으로 둠
