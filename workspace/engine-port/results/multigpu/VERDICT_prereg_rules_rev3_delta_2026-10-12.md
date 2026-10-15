# 규칙층 델타 감사 판정서 — PREREG_MULTIGPU_2026-10-12 rev3 (M2.5, 커밋 8b3b074)

> 감사자: claims-auditor(읽기 전용, GPU 0), 2026-10-12. 판정문 요지를 메인 세션이 저장했다(수치·처방·caveat 문안은 원문 그대로).
> 직전 판정서 표·처방은 등록 상수로 쓰지 않고 코드와 원자료로 재계산했고, 그 처방 자체도 검사했다(게이트 #110·#113).

## 0. 판정 — `MIXED`
| 범위 | 등급 | 이유 |
|---|---|---|
| **M2g(§3)** | **`GO-with-caveats` 유지** | 실행 규칙 반전 없음. 단 G5(c) NVML 폴백(D8)이 M3 G 인용 가능성을 정하므로 rev4 수리 전까지 G5(c) `MEMORY_MISMATCH` 출력은 인용하지 않는다 |
| **M3(§4)** | **`NO-GO`** | D3·D4·D6 해소, D5는 (i)–(iii) 해소. 새 死因 **D7**(N3, 예산 절단 예보 동반), **D8**(N3, 폴백 분기 조건부). 둘 다 GPU 없이 수리 가능 |
| M4 | 판정 안 함 | — |

"M3 거의 GO"·"D3–D6 해소 = M2.5 통과"로 요약하지 않는다.

## 1. 직전 지적 처리
| 항목 | 상태 | 근거 |
|---|---|---|
| D3 | shape A에서 해소 | 1/3 구간 평균 차이가 PD-mux 입장 구멍의 정상 오프셋을 상쇄. 단 같은 표본원이 fused shape B에서 다른 기전으로 죽음 → D7(직전 감사가 놓침) |
| D4 | 해소 | VESSL agnostic B 실측, ρ ≤ 0.9에서 거짓 실패 0건 |
| D5 | (i)–(iii) 해소, 폴백이 D8 | radix OFF → `_calculate_mamba_ratio` = 1(`model_runner_kv_cache_mixin.py:390-392`), mamba 슬롯 = max_running_requests(`:222-229`) |
| D6 | 해소 | PID 식별: 주 프로세스 runner `$!`(`engine_bench_runner.sh:202-204`), detokenizer `sglang::detokenizer`(`detokenizer_manager.py:395`), DP controller `sglang::data_parallel_controller`(`data_parallel_controller.py:599`). MGP-30 |

## 2. 코드 사실 (dev tree `sglang_engine_dev_multigpu/python/sglang/srt`)
- `SGLANG_LOG_MS`: `environ.py:170` `EnvBool(False)`, "1" 수용(`:122`), 효과는 `utils/common.py:1136` `.%(msecs)03d` 추가뿐(벽시계·로컬 시간대), `SGLANG_LOGGING_CONFIG_PATH` 설정 시 무시(`:1125`).
- `decode_log_interval` 기본 40(`server_args.py:418`), runner는 이것도 `SGLANG_LOG_MS`도 넘기지 않음.
- `max_queued_requests` 기본 None(`server_args.py:345`) → 대기열 상한 없음. 클라이언트 `sock_read=600` > 예보 최대 대기 ≈380 s.
- **fused prefill 우선**: `scheduler.py:2284-2287`, decode는 `get_new_batch_prefill`이 None일 때만(`batch_is_full` 또는 대기열 빔, `:2366-2369`). mixed-chunk off.
- `#queue-req` = `len(self.waiting_queue)`(`scheduler_metrics_mixin.py:657`, `:395`).
- **arm 간 엔진 노브 비대칭**: PD-mux arm만 `--chunked-prefill-size -1 --disable-overlap-schedule`(`engine_bench_runner.sh:145-151`). fused는 엔진 기본(A100 80GB chunked 8192, `server_args.py:1200`)·overlap ON.
- 클라이언트 `sent_s` = perf_counter − `benchmark_started`(`trace_loadgen.py:40,104`), 서버 벽시계와의 다리 없음.
- `pdmux_agnostic_manual.yml` 아직 없음.

## 3. 반전 시험
원자료: (a) VESSL N3 4회차 `n3b_pdmux-n3-capacity-93c3cca-20261007064124`(Nemotron-9B, `pdmux_n3_A_agn.yml`, shape B, K=40, 서버 로그 46셀), (b) KISTI fused 로그 `prefill_knee/knee_srv_837931.log`, `r0a/srv_fused_rep*_8349*.log`, `cp_baseline/w1_probe_900785/srv_cp8192.log`, (c) Nemotron 캡처 로그 fused `cp_baseline/cgate_902397`, PD-mux `longctx_conflict/probes/{s1_905958,p1_905707,qa_r3_906506}`, N3 `srv_g_*`.

| 자유 표면 | 판정 변화 | 근거 |
|---|---|---|
| **Decode 줄만 + 최소 20줄(fused, B, HI)** | **반전·도달 불가 → D7** | fused queue>0일 때 Prefill:Decode 줄 knee_837931 **114:0**, r0a **138:4 / 133:4**, cp 900785 **335:22**. 주기 모형(cap 48, OSL 64, 1주기 = 48·t\_p + 63τ, t\_p 0.5–1.45 s): 170 s에 decode step 150–400, K=40 줄 **4–11개**. 20줄 넘으려면 fused λ\* ≥ 3.6 req/s(VESSL B λ\* 약 5배). 확장으로 주기 수 불변 → fused B 전 N `SATURATION_UNRESOLVED` → `LADDER_TOO_LOW` → **G\_B 정의역 ∅**. K=1이면 ≈170줄이나 2 s burst 3개로 뭉쳐 3등분 빈 구간 발생, "K=1 vs K=40 라벨 불변" 검사는 실제 fused B에서 성립 불가(합성 로그에서만 통과) |
| 확장 트리거 "포화 증거 없음" | **결정적 발화 + 예산 반전(예보) → D7** | B 21 boot 추가, 확장 HI seed당 ≈550 s, 4 seed+boot ≈42 min × 21 ≈ **14.7 h** → 30–40 h로 하드캡 초과, 꼬리(TP4·DP4·NUMA·nsys·N=1 2회차) `TRUNCATED`. UNRESOLVED를 "포화 아님"과 묶은 것은 교훈 21 계열 |
| HI g ≥ 0.10, [0.05, 0.20] | 반전 없음 → MGP-26 | N=1 agnostic B x/λ\* 1.66: g 0.334/0.372(명목)·0.417/0.441(서버 도착). **"g ≥ 0.10 ⇔ x ≥ 1.11λ\*"는 실측 불일치**: x/λ\* 1.114에서 g 0.044–0.081(명목)·0.086–0.114(서버 도착) |
| LO g ≤ 0.05, B 3등분 잡음 | 거짓 실패 없음, 거짓 통과 있음 → MGP-27 | ρ 0.55–0.65 최대 0.011, ρ 0.88 최대 0.025, **ρ ≈ 1.0: 0.010–0.060**(4셀 중 3–4셀 ≤ 0.05) |
| 송신 창 정의·시계 다리 | g 크기만 변함 | 0.044 ↔ 0.086, 0.081 ↔ 0.114 |
| DP 도착률 전체/N | 반전 없음 | round_robin → Erlang-N |
| `SGLANG_LOG_MS` | 반전 없음 | 3등분 한 칸 ≈57 s. 파서는 `.mmm`·` DP{k}`/` TP{k}` 둘 다 수용 필요 |
| G5(c) (i)–(iii) 1% | 반전 없음 | — |
| **G5(c) NVML memory.used 폴백 1%** | **반전 → D8** | cudagraph 캡처 메모리 ∝ SM group 수: fused cap48 **0.25 GB**, PD-mux sm\_group 3 **0.65 GB**, sm\_group 6 **1.27 GB**(KISTI·VESSL 동일). 사용량 ≈66.7 GB 대비 d44 +1.02 GB **1.5% > 1%**, agnostic ≈+0.6 GB 0.9%. unchunked prefill(B 배치당 최대 16384 토큰) 추가 → 전 config 인용 금지, 정의역 ∅. 측정 시점 미등록 |
| FRONTEND `/proc` 델타 | 데이터 없음 → MGP-30 | 전 스레드 합산 |
| `HARNESS_LIMITED` throttle 비율(rev1부터) | x1에서 전부 반전, x8 미실증 → MGP-31 | VESSL x1 N3 46셀: throttled\_usec/벽시계 **9.5–28%**(전 셀 LIMITED), nr\_throttled/nr\_periods **0.22–0.81%**(0셀). x8 quota 88(M1: nr\_throttled 3, 0.19 s) |

## 4. 死因과 처방
- **D7 (N3, 예산 절단 예보)**: fused는 대기열이 있고 running < cap이면 decode를 안 돈다 → B에서 170 s에 Decode 줄 4–11개 → fused B HI 라벨 미정의 → G\_B 정의역 ∅, B의 P-DP·추세 출력공간 빔. 확장 트리거가 21 boot를 결정적으로 추가, 하드캡 절단 예보. 직전 감사 처방 P-D3/D4도 같이 죽음(게이트 #113).
  **P-D7**: (1) g의 대기열 원천을 클라이언트 측 Q\_c(t) = #{sent\_s ≤ t} − #{sent\_s + ttft ≤ t}로, 3등분 구간마다 시간 적분 평균. arm 대칭, 서버 로그·시계 다리 불필요, 송신 창 안 사건만(교훈 9 항등식 아님). TTFT 없는 오류 요청은 별도 플래그·Q\_c 제외 등록. DP는 엔진 전체 Q\_c·r = n/T, 복제본 균형은 기술치. (2) 대안: 서버 Prefill+Decode 줄 합집합 + 시계 다리 + 줄 구성비 공시(1이 더 깨끗). (3) 확장 트리거는 g < 0.10만, `SATURATION_UNRESOLVED`는 확장 안 함(교훈 21). (4) 변이 검사 갱신: "fused prefill 우선 합성 로그에서 Decode 줄만 쓰는 추정량" KILLED, "K=1 vs K=40" 삭제(1 채택 시). (5) 예산 재예보.
- **D8 (N3, PID 비가시 분기 조건부)**: PD-mux는 SM group마다 캡처해 fused보다 0.4–1.0 GB 더 씀, d44 1.5% → 전 config `MEMORY_MISMATCH`. 폴백은 직전 P-D5 처방, 1%는 rev3 — 공동 결함(게이트 #113).
  **P-D8**: arm 간 `memory.used` 비교는 기술치(boot 직후 + 셀 종료). 오염 검사 대체: (a) boot 전 부분집합 GPU `memory.used` < 1 GiB, (b) NVML `nvmlDeviceGetComputeRunningProcesses` 항목 수 = 기대 프로세스 수. 항목 수도 0으로 숨겨지면 (a)만.

## 5. caveat (문자 그대로 승계)
- **MGP-26**: "g ≥ 0.10을 'x ≥ 1.11 λ\*'로 번역하지 않는다. VESSL agnostic B 실측(x/λ\* = 1.114)에서 g는 송신 창 추정에 따라 0.044–0.114였다. `HI_SATURATED`는 x/λ\*의 수치 하한을 주지 않는다."
- **MGP-27**: "`LO_UNSATURATED`(g ≤ 0.05)는 x_lo < λ\*를 보증하지 않는다. VESSL agnostic B에서 x ≈ λ\*(0.69 vs 0.691)일 때 g는 0.010–0.060이었다. `KNEE_BRACKETED`를 '무릎이 x_lo와 x_hi 사이에 있다'로 쓰지 않는다."
- **MGP-28**(출처 정정, rev3 §10 마지막 문단 대체): "g의 정의(Decode 줄, 3등분, (2T/3)·r, DP 복제본 도착률 = 전체/N)와 문턱 0.10/0.05는 rev2 델타 감사 처방 P-D3/D4를 문자 그대로 채택한 것이다. 유체 근사 g ≈ 1 − λ\*/x는 사후 정당화이며 0.05를 도출하지 않는다(LO의 유체값은 0이다)." 저자 문장 "감사자 예보를 승격한 것이 아니라 유체 근사로 정했다"는 사실이 아니다(교훈 80). 판정 불변이라 死因 아님.
- **MGP-29**: "fused arm은 overlap schedule ON·chunked prefill 8192(엔진 기본값)이고 PD-mux arm은 overlap OFF·chunked −1(엔진 강제)이다. G(N)을 'SM 분할 효과'나 'PD 분리 효과'로 귀속하지 않는다(정본 P1 'SM 분할 자체' 귀속 NOT-YET-SUPPORTED 승계)."
- **MGP-30**: "`/proc/<pid>/stat` utime+stime은 프로세스 전 스레드의 합이다. `FRONTEND_SATURATED` 미발화를 '이벤트 루프 비포화'로 쓰지 않는다. 스레드별(`/proc/<pid>/task/*/stat`) 최댓값을 함께 기록한 경우만 예외다."
- **MGP-31**: "`HARNESS_LIMITED`의 'throttled 시간 비율'은 분모를 등록하기 전에는 인용하지 않는다. VESSL x1 N3 46셀에서 throttled_usec/벽시계(9.5–28%)와 nr_throttled/nr_periods(0.22–0.81%)는 라벨을 전부 반대로 낸다. x8(quota 88)에서의 크기는 미실측이다." rev4에서 하나 등록, 둘 다 기록.
- **MGP-32**: "D7·D8의 반전 수치는 KISTI 로그와 VESSL x1 로그에서 나왔고 x8 실측이 아니다. 구조(prefill 우선 스케줄, group별 캡처)는 엔진 코드가 정한다."
- MGP-0…MGP-25 유지. MGP-0의 "M3 `NO-GO`"는 rev3에도 적용.

## 6. 다음 gate
rev4에 P-D7·P-D8·MGP-31 분모·MGP-28 정정 반영(CPU) → 델타 감사(D7, D8, Q\_c 반전 시험: 오류 요청·DP 집계). 하네스 항목: `pdmux_agnostic_manual.yml`, (2안 시) `--decode-log-interval`/`SGLANG_LOG_MS` 전달, 시계 다리. M2g는 MGP-1…5·MGP-23 승계 + G5(c) 폴백 결과 인용 금지 조건으로 하네스 작성·하네스층 감사 진입 가능. 제출은 사용자 승인.
