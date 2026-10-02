# TTFT 측정 정의 검토 — 정상성 검사·대기/처리 분해 규약 제안 (2026-09-28)

> **지위**: 검토 메모(정본 아님). GPU 0 · 새 성능 판정 0건 · Claim 등급 변경 0건.
> `handoff-report/session_handoff_2026-09-28.md` §4-5(이월 항목 "TTFT 측정 정의 검토")의 처리 문서.
> **사용자 지시(2026-09-28)**: 보고용·검토용 artifact에는 반영하지 않는다 — §3.3·§3.4는 문안만 둔다.
> **회부**: 규약 채택 = claims-auditor 사전 감사(`handoff-report/vessl_execution_plan_2026-09-24.md` §3-4 경로) ·
> 정본 등재 여부 = doc-steward · 엔진 사실 확인 3건(§3.2) = engine-porter.
> 등급 표기는 `reports/synthesis_2026-09-22/EXPERIMENT_DESIGN_2026-09-22.md` §0을 따른다([READ]=코드·정본에서 읽음, [제안]=감사 전).

## 0. 한 줄

현 하네스의 TTFT는 두 loadgen 모두 **클라이언트 송신 시각 → 첫 토큰 수신** 이며, open-loop(Poisson 도착,
동시성 상한 없음) 채점 런에서는 송신 = 예정 도착이므로 서빙 벤치 표준(도착 기준)과 같다 [READ]. 문제는
정의가 아니라 세 가지 운용 조건이다: (i) `bench_serving` 경로는 예정 도착 시각을 기록하지 않아 송신 표류를
사후 검증할 수 없다, (ii) `--max-concurrency`를 건 런은 클라이언트 대기가 TTFT에서 빠져 도착 기준이 아니다,
(iii) 과부하 셀의 TTFT 크기는 런 길이의 함수라 시스템 상수가 아니다(정본이 이미 §3 항목32 · 게이트 #12 ·
F-E로 보유). 제안은 새 규칙이 아니라 **기존 세 장치를 결과 규약 한 곳에 묶고, 서버측 stage stamp로
대기/처리 분해를 병기**하는 것이다 [제안].

## 1. 확인한 사실 [READ]

### 1.1 두 loadgen의 TTFT 시계

| 경로 | 도착 생성 | TTFT 시계 시작 | 요청별 기록 필드 | 송신 표류 검증 |
|---|---|---|---|---|
| `sglang.bench_serving`(dev tree v0.5.10, `get_request` → `limited_request_func` → backend request func) | 요청 사이 지수 간격 `asyncio.sleep` — **상대** 스케줄, 누적 표류 보정 없음 | request func 안의 `st = perf_counter()` — **세마포어 획득 뒤** | `--output-details`: `input_lens/output_lens/ttfts/itls/generated_texts/errors`뿐. 도착·송신 타임스탬프 **없음** | **불가** (Gate 2 사전등록이 F-D를 삭제한 이유와 동일: `workspace/engine-port/results/p1_gates/gate2/PREREG_GATE2_2026-08-06.md:182-186`) |
| `benchmarks/pdmux_eval/trace_loadgen.py` | `benchmark_started + target`까지 sleep — **절대** 스케줄(`workspace/engine-port/benchmarks/pdmux_eval/trace_loadgen.py:49-51`) | `sent` 직후(`:93-95`) | `scheduled_arrival_s`, `sent_s`, `completion_s`, `ttft_ms`, `token_itl_ms`(`:106-109`) | **가능**: `sent_s − scheduled_arrival_s` |

`benchmarks/pdmux_eval/analyze.py`는 두 경로 모두 요청별 `ttft_ms`를 그대로 SLO 술어(`RequestResult.passes`)에
넣는다(`workspace/engine-port/benchmarks/pdmux_eval/analyze.py:212-221`). 즉 **분석기는 TTFT의 출처를 구분하지 않는다.**

### 1.2 `--max-concurrency` 런은 도착 기준 TTFT가 아니다

세마포어가 request func 바깥을 감싸므로, 상한에 걸린 요청은 클라이언트 큐에서 기다린 시간이 TTFT에서
빠진다(closed-loop 지표). 현행 스크립트에서 이 플래그가 걸린 곳은 워밍업(`--max-concurrency 1 --num-prompts 8`),
λ0 I3 포화 프로브(`--request-rate inf --max-concurrency 64`), longctx 프로브 일부다. **채점 런은 `--request-rate R`만
쓴다**(E2: `workspace/engine-port/results/r2_eval/e2_sticky_prereg/e2_sticky.sbatch:338-342`, λ0:
`workspace/engine-port/results/r2_eval/lambda0_prereg/lambda0.sbatch:372-376`) — open-loop. 이 구분은 저장소가 이미
알고 있다: `workspace/engine-port/results/r2_correctness/r2_correctness.sbatch:572-579`가 "closed-loop에서 클라이언트
`Concurrency:` 지표는 구성상 정보가 없다"고 등재했다.

### 1.3 분해 재료 — 서버측 stage stamp

- upstream `SchedulerReqTimeStats`(dev tree `sglang/srt/observability/req_time_stats.py`): `wait_queue_entry_time`
  (scheduler `handle_generate_request`에서 set — PD-mux 경로도 이 값을 **읽는다**: `workspace/engine-port/src/multiplex/
  multiplexing_mixin.py:932-952`의 `_slo_prefill_age_ms`, `workspace/engine-port/src/multiplex/dual_worker.py:58-59`),
  `forward_entry_time`(upstream `get_new_batch_prefill`에서 배치 단위 set), `prefill_run_batch_start/end_time`,
  `prefill_finished_time`(output processor, chunked 포함), `completion_time`. 편의 함수 `get_queueing_time() =
  forward_entry − wait_queue_entry`. 로깅은 server arg `--enable-request-time-stats-logging`(기본 off).
- PD-mux telemetry `runtime_snapshot`: `prefill_queue_depth`(= 대기열 길이, `multiplexing_mixin.py:498`)와
  `oldest_prefill_age_ms`(대기열 + 진행 중 split-prefill 배치 중 최고령, `wait_queue_entry_time` 기준). **집계
  수준**의 대기 신호이며 요청별이 아니다.
- 클라이언트 측에는 첫 토큰 수신 시각만 있다.

### 1.4 이미 정본에 있는 것 (재도출 아님)

- `reports/CONSENSUS.md:4020-4028` §3 항목32: 과부하 arm과 정상 arm을 같은 rate에서 비교한 지연차의 **크기**는 런 길이
  의존이라 시스템 상수가 아니다 — F-E 발화 셀은 **부호만** 인용.
- `PROJECT_STATUS.md:10900-10906` 방법론 게이트 #12: 임계 지시함수 판정은 **임계 사다리 + 큐-성장 검정**(launch 순서
  전반부/후반부 위반율)으로 견고성을 보여라.
- F-E 정의(`PREREG_GATE2_2026-08-06.md:176-181`): 런을 도착 순서로 3등분한 `TTFT_p50(3분위)/TTFT_p50(1분위)`의
  rep 평균, 임계 **1.7**; flag 시 그 arm이 참여하는 비교는 크기 인용 금지·부호만, 셀·rep은 버리지 않는다.
- `reports/bench_noise_root_cause.md:41`·`:104`: 과부하 큐의 TTFT 평탄역 ≈ backlog/service_rate, 고정 프롬프트 수 런의
  goodput은 과부하에서 런 길이 의존이라 ill-posed.
- `CLAUDE.md` 방법론 게이트 6: SLO 임계 지시함수는 metric cliff를 피해 측정, 용량 먼저.

## 2. 문헌 표준 대조 (스코프 한정)

vLLM·SGLang `bench_serving` 계열의 open-loop 벤치는 TTFT를 **클라이언트가 요청을 보낸 시점부터 첫 토큰 수신까지**로
잰다. 클라이언트가 예정대로 송신하는 open-loop에서는 도착 = 송신이므로 우리 하네스와 같은 정의다. 단 두 가지를
분명히 한다: (a) "도착 기준"이 뜻하는 것은 **엔진 대기열 진입이 아니라 클라이언트 송신**이며, 그 사이의 HTTP·
토크나이즈·디스패치 지연도 TTFT 안에 있다(서버 stamp로만 분리 가능) — 문헌 수치와 우리 수치를 나란히 둘 때
이 구간이 같은 쪽에 있는지 확인한다. (b) 문헌 전수 조사는 하지 않았다 — 특정 논문의 정의를 인용하려면
venue-strategist가 원문에서 확인한다. 이 절은 정의의 **일치 여부**만 다루고 수치 비교 가능성은 주장하지 않는다.

## 3. 규약 제안 [제안 — 감사 전]

### 3.1 정상성(stationarity) 검사 — 결과 규약에 명시

채점 셀 × arm × rep마다 아래 세 값을 계산해 표에 병기한다.

| 기호 | 정의 | 출처 | 상태 |
|---|---|---|---|
| S1 | F-E 비율(도착 순서 3분위/1분위 `TTFT_p50`, rep 평균, 임계 1.7) | 기존 정의 그대로 | 임계 기존 |
| S2 | 게이트 #12 큐-성장: launch 순서 전반부 vs 후반부 SLO 위반율 차 | 기존 정의 그대로 | 임계 **사전등록에서 고정** |
| S3 | telemetry `prefill_queue_depth` 시계열의 단조 추세(benchmark phase, 후반/전반 평균비 또는 순위 상관) | 신규 — 엔진 측 독립 계측 | 임계 **사전등록에서 고정**, 이 메모는 정하지 않음 |

표기 규칙: 하나라도 발화 → 셀 라벨 `UNSTABLE`. 그 셀의 TTFT p50/p95/p99와 goodput은 **크기를 제출하지 않고**
부호·라벨만 표에 적는다(원시값은 아티팩트에 남긴다 — 페어링·n 보존). 발화 없음 → `STATIONARY`, 크기 인용 가능.
이는 §3 항목32의 "부호만"을 셀 라벨로 기계화한 것이지 새 판정 기준이 아니다.

**전제(명시, 2026-09-28 사용자 질의로 추가)**: 이 검사는 **요청 거부 없음** 조건의 것이다. 우리 엔진의 "admission 차단"은
보류(running batch 승격 지연)이지 거부가 아니며, upstream `--max-queued-requests`(대기열 상한 초과 시 abort)는 기본값
None이고 우리 스크립트·config 어디에도 설정돼 있지 않다 — 모든 기존 측정은 무한 대기열 조건이다. 거부를 켜는 캠페인은
별도 설계 결정이며, 그 경우 (i) 거부된 요청을 goodput 분모에 포함 (ii) 거부율을 TTFT 옆에 병기 (iii) arm 간 대기열
상한 동일성 검증이 규약에 추가돼야 한다. 거부는 과부하를 없애지 않고 나타나는 자리를 TTFT에서 거부율로 옮길 뿐이다.

왜 셋인가: S1·S2는 판정 대상 통계량(TTFT·위반율) 자신으로 검정하는 자기참조 검사라, 그 통계량이 포화하면
검정력도 같이 죽는다(교훈 (9): 게이트 자신이 항등식일 수 있다). S3는 엔진 대기열 길이라는 **다른 계측기**다.
S3 주의: telemetry 표본 시각은 `timestamp_monotonic_s`를 그대로 쓰고 구간 부과를 하지 않는다(§3 항목120의
구간 부과 아티팩트 계열). 표본 격자가 arm마다 다를 수 있으므로 S3는 시간가중이 아니라 표본 순위 기반으로 둔다.

### 3.2 TTFT 분해 규약

```
TTFT_client = L_client + L_ingress + W_queue + S_prefill + L_egress
  L_client  = sent_s − scheduled_arrival_s          (trace_loadgen 경로에서만 관측 가능)
  L_ingress = wait_queue_entry_time − created_time   (API 서버 → 스케줄러 대기열)
  W_queue   = forward_entry_time − wait_queue_entry_time   (= get_queueing_time())
  S_prefill = prefill_finished_time − forward_entry_time    (chunk 사이 decode 끼어듦 포함)
  L_egress  = 클라이언트 첫 토큰 수신 − prefill_finished_time
```

보고 규칙: `W_queue`와 `S_prefill`을 요청별로 구해 p50/p95를 TTFT 옆에 병기한다. 셀마다 잔차
`TTFT_client − (W_queue + S_prefill)`를 확인하고, 잔차가 사전등록 문턱을 넘으면 클라이언트/입출력 병목
의심으로 표기한다(`r2_correctness.sbatch:572-579`의 client-bottleneck 검사와 같은 계열). **"TTFT가 늘었다"는 서술은
분해 결과가 있을 때 `W_queue`가 늘었는지 `S_prefill`이 늘었는지를 반드시 붙인다** — 둘은 다른 기전(admission/
대기열 vs prefill SM 부족)이고 정책 처방이 다르다.

전제 3건(**engine-porter 확인 전에는 [제안] 유지**):

1. ~~`forward_entry_time`이 PD-mux 경로에서 set되는가~~ → ★**해소(2026-09-30, engine-porter 정적 독해 + 메인 세션 재확인,
   `workspace/engine-port/results/hybrid_sched_s0/engine_porter_review_2026-09-30.md` §5)**: admission 시 1회 set되고 split-prefill
   재진입에서는 재호출되지 않으며(`update_split_prefill_batch`가 span 중 조기 반환, 함수 자체도 첫 값 유지) legacy·true-dual
   양쪽 동일. chunked 재진입은 PD-mux에서 발생하지 않는다(`chunked_prefill_size == -1` 강제). ⇒ `W_queue`·`S_prefill` 정의는
   그대로 성립. **단 추가 발견 2건이 규약을 바꾼다**: (i) `prefill_run_batch_start/end_time`은 `EXTEND`에서만 기록되어
   **PD-mux(`SPLIT_PREFILL`)에서 미기록** — 분해에 쓰지 않는다. (ii) decode OOM **retract** 시 `wait_queue_entry_time`이 무조건
   덮어써져(`req_time_stats.py` `set_wait_queue_entry_time` 말미 대입, 기존 값 있으면 `retract_time`도 기록) `W_queue`가 음수
   가능 — 규약: `retract_time > 0`인 요청은 분해 표에서 **제외하고 건수를 병기**한다. R2 admission 보류 시간은 `W_queue`에,
   admission 직후 첫 전환 드레인은 `S_prefill`에 들어간다(해석 시 명기).
2. `--enable-request-time-stats-logging` 출력이 요청 `rid`와 조인 가능한 형식인가(로그 파싱 vs `meta_info`).
   조인이 안 되면 요청별 분해가 아니라 분포끼리의 대조만 가능하다.
3. 이 플래그의 hot-path 비용. 모든 arm에 동일하게 켜면 비교 교락은 아니지만 절대값이 움직이므로,
   켠 상태를 실행 기록에 남기고 끈 런과 수치를 섞지 않는다.

`bench_serving` 경로 한계: 클라이언트 도착 시각이 없어 `L_client`를 검증할 수 없다. 분해가 필요한 캠페인은
`trace_loadgen` 경로를 쓰거나, `bench_serving` 덤프에 요청별 송신 시각을 추가하는 하네스 후속 항목(Gate 2
사전등록이 F-D 삭제 때 이미 등재한 것과 같은 항목)을 먼저 처리한다.

### 3.3 문구 교정 (c) — artifact 미적용

- 대체 문안: "TTFT 폭발" → **"대기열이 쌓여 요청이 용량 밖으로 밀려남(TTFT는 런 길이에 따라 커진다)"**.
- 적용 범위: 이 메모 이후 새로 쓰는 문서. 정본·과거 보고서의 기전 사슬 라벨("decode 굶김 → ITL↑ → batch 정체 →
  admission 차단 → TTFT 폭발", `CONSENSUS.md` §1-4 등 20곳 이상)은 **일괄 치환을 권고하지 않는다** — 라벨로
  굳어 있고 치환 시 기존 인용이 깨진다. 최초 등장 1곳에 "여기서 폭발은 과부하 큐의 런 길이 의존 상승을 뜻하며
  크기는 시스템 상수가 아니다(§3 항목32)"라는 각주를 다는 것이 doc-steward 판단 사항.
- 보고용/검토용 artifact(v17–v19)의 그림 3·시뮬레이션 설명은 **사용자 지시로 이번에 편집하지 않았다.**

### 3.4 검토용 6절 두 줄 (d) — 문안만, 미적용

1. "TTFT·goodput의 크기는 셀이 정상성 검사(F-E · 큐-성장 · telemetry 대기열 추세) 세 개를 모두 통과할 때만
   제출한다. 발화 셀은 부호와 `UNSTABLE` 라벨만 적는다."
2. "TTFT는 서버측 stamp로 대기(`W_queue`)와 prefill 처리(`S_prefill`)로 분해해 병기한다. 분해 전제 3건이
   확인되기 전에는 클라이언트 TTFT만 보고한다."

## 4. 하지 않은 것 / 다음

- **감사 안 받음.** §3 전체는 claims-auditor 사전 감사 대상(`vessl_execution_plan_2026-09-24.md` §3-4에 항목 추가).
- **engine-porter 확인 3건**(§3.2) 전에는 분해 규약을 사전등록에 넣지 않는다.
- **정본 반영은 doc-steward 판단.** 새 게이트 번호를 만들지 않는 쪽을 권고한다 — §3.1은 기존 #12·항목32·F-E의
  통합이고 새 번호를 만들면 번호 축이 하나 더 는다(`MEMORY.md` 번호 축 안내 참조).
- **artifact 미반영**(사용자 지시). 문안은 §3.3·§3.4에 있다.
- 측정하지 않았다. 이 메모의 어떤 수치도 새 결과가 아니다.

## 5. 追記 (2026-10-01) — p95 술어의 PAUSE 사각지대

PAUSE(k step 정지)는 정지된 요청마다 `k·T_d` 크기의 드문 큰 간격을 만들고, 출력 64 토큰 요청의 간격 63개 중 3개 이하는 요청내 p95에
잡히지 않는다. PAUSE arm이 포함된 비교는 요청내 **max-ITL(또는 p99)**을 primary에 병기한다(선례: Gate 2 `X_τ`). 상세
`reports/definition_blind_spots_2026-10-01.md` D-7.
