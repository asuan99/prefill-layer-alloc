# Gate 1 — telemetry 재현런 사전등록

작성: 2026-08-06 (experiment-runner), 제출 **전** 고정. 결과를 본 뒤 이 문서의 결정 규칙을
바꾸지 않는다(CLAUDE.md 방법론 게이트, `PROJECT_STATUS.md` 방법론 게이트 #8).

## 동기 (요청 원문 요약)

`reports/CONSENSUS.md` §1-1(rev12)는 job 873944/873945(P1 운영점 대조, 2026-08-05)의
"인용 금지" 목록에 **"파티션·동시성 기전 문장·split 수치"**를 올려 두었다 — 이유는
`PDMUX_TELEMETRY_PATH` 미설정으로 realized `(prefill_sms, decode_sms)` 재집계가
불가능했기 때문이다(Stage 0 D108 전례, §1-21). 이 gate의 목적은 **오직 그 금지를 풀
수 있는지 판정**하는 것이다 — 새 성능 주장이나 goodput/latency 수치는 만들지 않는다
(이 job은 §1-1의 성능 판정에 **아무것도 추가하지 않는다**, m3r2_pinforce.sbatch가
확립한 선례와 동일한 스코프: "DIAGNOSTIC ONLY... NO ITL NUMBER FROM THIS JOB MAY EVER
BE QUOTED").

## 재현 대상과 변경점 (job 873944 대비 최소 diff)

- **모델/설정**: `Zyphra/Zamba2-2.7B`, `--context-length 4096`,
  `--attention-backend triton` — 873944의 실제 sbatch 인자
  (`sacct -j 873944 --format=SubmitLine`: `p1op_run.sbatch Zyphra/Zamba2-2.7B 4096 triton
  zamba2-27b`)와 완전히 동일.
- **정책**: `agnostic`(pdmux v1) 단독. `plain`(fused) arm은 재현하지 않는다 — fused는
  `--enable-pdmux`를 쓰지 않으므로 realized-partition 텔레메트리가 애초에 존재하지 않아
  이 gate와 무관하다.
- **서버 플래그**: `p1op_run.sbatch`의 `boot_server()` 함수를 policy=agnostic으로 호출한
  것과 바이트 단위로 동일 — `--trust-remote-code --dtype bfloat16 --attention-backend
  triton --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48
  --context-length 4096 --enable-pdmux --pdmux-config-path pdmux_a100_smoke.yml
  --chunked-prefill-size -1 --disable-overlap-schedule`. **cudagraph는 ON**(`--disable-
  cuda-graph`/`--disable-piecewise-cuda-graph` 둘 다 미사용, 873944와 동일 — 이것이
  §1-1이 검증하려는 "운영점"의 정의 그 자체이므로 이 gate에서도 반드시 고정한다).
- **워크로드**: `random-ids`, `--random-input-len 2000 --random-output-len 96
  --random-range-ratio 1.0 --num-prompts 120`, `--warmup-requests 8`(bench_serving
  내장, phase 분리) — 873944 Phase 1과 동일.
- **rate/rep**: rate ∈ {2, 3}, n = **1 rep**(성능 판정이 아니라 계측 파이프라인 자체의
  가부만 묻는 것이므로 CLAUDE.md의 n≥5 요건 대상이 아님 — 요청 브리핑이 명시한 예산
  0.2 GPU-hr에 맞춘 축소). seed는 873944 Phase 1의 공식 `9000 + 100*REP + R`을 REP=1로
  그대로 재사용(`R=2`→9102, `R=3`→9103) — 실제로 그 트레이스를 재생하는 것은 아니지만
  (agnostic만 돌리므로 pair는 없다), 프로젝트의 seed 관례를 그대로 따른다.
- **유일한 실질 변경**: 이하 텔레메트리 계측 3종.
  1. `PDMUX_TELEMETRY_PATH=<gate1 dir>/gate1_telemetry_<jobid>.jsonl` — 요청 브리핑이
     지시한 바로 그 추가.
  2. `PDMUX_TRACE_FORCE_PREFILL=1` — 아래 "케이던스" 절 참조. 코드 주석
     (`multiplexing_mixin.py:497-517`)이 명시하듯 이 플래그는 **시간가중 통계에
     영향을 주지 않는 순수 additive 샘플**(count 기반 통계만 오염, 우리는 count 기반을
     보조지표로만 쓴다) — 서버가 실제로 하는 일(스케줄링, 파티션 선택, 배치 구성)은
     바이트 단위로 불변, telemetry emission만 늘어난다. 선례:
     `results/s8_frontier/m3r2_pinforce.sbatch`("pin measurement gets the dense
     sampling, and it is quarantined in a job that emits no latency claim").
  3. `PDMUX_DUAL_WORKER_TRACE_EVERY=8`(기본 32) — scheduled(비강제) 서브샘플링 격자를
     4배 촘촘히 한다. force-prefill은 prefill-in-flight 구간만 강화하므로, 두 여집합
     음성대조(decode-idle / decode-busy-but-prefill-idle)의 표본 수를 확보하려면 이
     grid 자체도 촘촘해야 한다.
  이 세 가지 모두 **런타임 동작(스케줄링/배치/SM 분할 결정)을 바꾸지 않는다** — 관측
  주기만 바꾼다. `PDMUX_DUAL_WORKER`/`PDMUX_TRUE_DUAL_WORKER`는 **미설정**(둘 다 R1
  observer/true-dual-worker 아키텍처 전용, agnostic v1과 무관 — CLAUDE.md 경고 그대로
  준수, `_write_dual_worker_trace`의 `architecture` 필드가 "legacy"로 남는지 결과에서
  확인한다).

## ⚠️ 바이너리·플래그 드리프트 사전 확인 (제출 전 완료)

`p1op_873944.out`에서 `sync_engine_tree.sh`가 쓴 manifest 경로를 확인하고
(`workspace/engine-port/results/p1_opint/runtime_source_manifest_zamba2-27b_873944.sha256`),
**현재(2026-08-06) 워킹트리로 동일 스크립트를 재실행해 만든 manifest**와 정렬(sort) 후
`diff`했다 — **바이트 단위로 완전히 동일**(전 항목 SHA-256 일치, `MANIFESTS IDENTICAL`).
이 사전등록 문서와 같은 디렉터리에 참조 사본을 보존한다
(`reference_manifest_873944.sha256`). 결론: 워킹트리에 미커밋 변경(P5 계측
`src/models/zamba2.py` 등)이 많지만, **873944가 실제로 sync한 파일 집합(zamba2 모델/설정
포함, 2026-08-04에 sync 대상으로 편입된 버전)과 지금 sync될 파일이 정확히 같다** — 드리프트
없음. sbatch 자체도 방어적으로 이 비교를 다시 실행해 불일치 시 즉시 중단한다(아래 참조).

## 텔레메트리 필드의 의미 (코드 근거 — 인용을 위한 사전 확인)

`workspace/engine-port/src/multiplex/dual_worker.py:604-623`
(`DualWorkerState.metrics()`): `prefill_sms`, `decode_sms`, `stream_index`는 전부
`self.arbiter.sm_counts[self.arbiter.stream_index]`의 **재진술**이지 하드웨어 직접
프로브가 아니다(메모리 `scale-8b-sm-sensitivity` "S0 종결" 항목과 동일한 scope 제약 —
이 gate도 **selector-level, behavioural**이지 hardware 층은 아니다). `sm_counts`는
`pdmux_context.py:divide_sm(108, cc(8,0), groups=2)`가 계산한 값을 그대로 쓴다 — 이번
설정(`sm_group_num: 4`, 즉 groups=2)에서 직접 계산하면:

```
possible_values = [x for x in range(4,105,2) if x>=108-x and 108-x>=16]  # 54..92 step 2, 20 values
step = max(1, 20//2) = 10
selected = possible_values[::10][:2] = [54, 74]
divisions = [(54,54),(74,34)] -> reversed -> [(74,34),(54,54)]
SM_COUNTS = [(108,0), (74,34), (54,54), (0,108)]   # idx 0,1,2,3
```

즉 요청 브리핑의 `(74,34)`/`(54,54)`는 임의로 고른 상수가 아니라 **이 config로 실제
계산되는 유일한 두 분할값**이다(끝점 선택 아님, 방법론 게이트 #8 위반 아님).

## ⚠️⚠️ 게이트가 항등식인가 — 코드 경로 직접 추적 (필수 자문, 방법론 게이트 #9)

`multiplexing_mixin.py:adjust_stream_groups()`(agnostic v1: `manual_divisions` 없음,
`sticky_partition_enabled` 없음, `PDMUX_SLO_SCHED` 없음)의 분기는:

```
if not running_batch.is_empty() and split_prefill_batch:      # 곧 "decode busy AND prefill in-flight"
    idx = max(1, min(2, decode_bs*2//decode_bs_divisor))       # ⇒ idx ∈ {1,2} 결정적, 108/0 도달 불가
elif not running_batch.is_empty():                             # decode busy, prefill NOT in-flight
    idx = real_sm_group_num - 1 = 3                             # ⇒ (0,108)
else:                                                            # decode empty
    idx = 0                                                      # ⇒ (108,0), §1-22 auto-revert
```

`self.split_prefill_batch`(prefill in-flight의 정의 그 자체)가 truthy인지가 정확히
telemetry의 `prefill_active_batch_size>0`을 만드는 상태이므로, **"decode-busy ∧
prefill in-flight" 조건은 코드상 idx∈{1,2}만 도달 가능한 바로 그 분기와 사실상 동치다.**
⇒ **이 gate가 묻는 문자 그대로의 질문("그 조건에서 split이 두 값 중 하나인가")은 코드
구조상 결정적으로 참에 가깝게 설계되어 있다 — 사후 관측이 아니라 사전 코드 읽기로
그렇다는 것을 알 수 있다.** 순수 판별력만 보면 이 조건은 §1-31이 발견한 `E1_COND_PIN`
항등식(`prefill_sms != target ⟺ decode_running_batch_size == 0`, 77,688 스냅샷 무위반)과
같은 계열이다.

**그렇다면 왜 그래도 돌리는가 — 이 gate가 실제로 판별할 수 있는 것은 다음 세 가지다:**

1. **샘플링 케이던스로 인한 라벨-실행 오정합(§0 종결의 기전).** `_dual_worker_sync`가
   이번 iteration의 `stream_idx` 갱신 *이전에* 호출되므로(`multiplexing_mixin.py:993-997`
   vs `:1080` `adjust_stream_groups()` 호출 지점), 매 prefill 에피소드의 **첫
   iteration**은 `prefill_active_batch_size`가 아직 0인 상태에서 관측된다(어드미션이
   그 iteration 뒷부분에 일어남) — 이는 과소계상이지 오분류가 아니다. 그러나 32(기본값)
   또는 8(이 job에서 채택) sync마다 1개만 기록되는 서브샘플링 격자가 **에피소드 경계
   근처의 전이 iteration을 우연히 포착**하면, "prefill 방금 시작 vs stream_idx 아직
   안 바뀜" 같은 진짜 과도상태가 라벨에 찍힐 수 있다 — S2가 발견한 "선택기 케이던스"
   기전(`results/s2_sticky/S2_REPLICATION_2026-08-05.md`)과 **같은 종류의 위험**이다.
   `PDMUX_TRACE_FORCE_PREFILL=1`은 이 위험을 정확히 겨냥한 완화책이다 — prefill이
   in-flight인 **모든** sync에서 강제 기록하므로(코드 주석: "additionally emit ...
   bypassing the count subsampling"), 경계 근처 전이가 서브샘플링으로 누락될 확률이
   극히 낮아진다.
2. **버그로 인한 실제 위반**(예: `decode_bs_divisor`/`real_sm_group_num` 산술 오류,
   `sm_counts` 인덱싱 오프바이원 등) — 코드는 결정적이라고 "말하고" 있지만 실행이
   그대로 하는지는 별개 질문이다.
3. **여집합 음성대조의 실제 값**(아래 "여집합 음성대조" 절) — decode-idle과
   prefill-not-in-flight 상태에서 idx가 진짜 {0}/{3}으로 떨어지는지는 코드로는
   "설계상 그렇다"이지만 **§1-22 자체가 "설계대로"라고 기록한 그 문장을 직접 재확인하는
   것**이므로 무의미하지 않다 — 오히려 이 gate 전체에서 가장 판별력 있는 부분은
   **주 조건이 아니라 두 음성대조**일 가능성이 높다.

**따라서 이 gate의 해석은 다음과 같이 미리 못 박는다**: 주 조건(≥90%)이 통과해도
그것은 "PD-mux가 물리적으로 그 SM을 갈랐다"는 강한 주장의 근거가 **아니며**,
"selector-level에서 라벨이 코드가 말하는 대로 나타났고, 알려진 두 실패 모드(케이던스
오염, §0/S2와 동일 기전 / 버그)가 이번 실행에서는 발현하지 않았다"는 **약한** 주장의
근거일 뿐이다. hardware 층 프로브(S3)는 여전히 미실행이며 이 gate가 대신하지 않는다.
이 caveat은 §1-1 기전 문장을 해금하더라도 **그대로 함께 인용되어야 한다**.

## 집계 단위 (방법론 게이트 — 이 프로젝트에서 6번 재발)

**주(primary) = 시간가중(time-weighted).** 각 `runtime_snapshot`을 다음 스냅샷까지의
간격(wall-clock, `timestamp_monotonic_s`)으로 가중해 구간 총 시간 중 조건을 만족하는
시간의 비율을 잰다 — `results/s8_frontier/e1_pin_check.py`의
`compute_time_weighted_pin_gate`/`compute_conditional_pin_gate`와 동일 컨벤션(2026-07-30
bug #6 수정 이후 이 프로젝트의 표준). **왜 시간가중이 추정 대상과 맞는가**: 우리가 묻는
질문은 "prefill이 실제로 도는 동안, 그 **일(work)**이 어느 파티션에서 수행됐는가"이고,
이벤트 루프 iteration의 길이는 prefill 배치 크기·context 길이에 따라 수십 배 차이 난다
(한 prefill iteration 235ms vs decode iteration 11ms, `e1_pin_check.py` bug #2 주석) —
개수 가중은 짧은 iteration을 체계적으로 과대표집한다. **부(secondary) = 개수가중.**
`PDMUX_TRACE_FORCE_PREFILL=1`이 개수가중 통계를 prefill 쪽으로 기계적으로 편향시키므로
(코드 주석 명시), 개수가중 결과는 **참고용으로만** 병기하고 판정에는 쓰지 않는다 — 판정은
전적으로 시간가중.

에피소드(연속 조건-충족 구간) 단위의 진단(에피소드 수, 스냅샷 간 `sample_index` 델타
중앙값 = 몇 event-loop iteration마다 기록되는가)도 함께 보고해 표본 크기와 케이던스를
투명하게 남긴다(`e1_pin_check.py`가 이미 확립한 "n_episodes/n_pa_snapshots를 항상
같이 보고하라"는 관행을 따른다).

## 여집합 음성대조 (방법론 게이트 #10)

주 조건(A = decode-busy ∧ prefill in-flight)의 여집합 두 갈래를 **같은 파이프라인으로**
계산해 병기한다:

- **B = decode-idle**(`decode_running_batch_size==0`, prefill 상태 무관). 코드상 기대값 =
  거의 전부 `(108,0)`(§1-22 auto-revert, 이미 확정된 사실의 재확인).
- **C = decode-busy ∧ prefill NOT in-flight**(`decode_running_batch_size>0 ∧
  prefill_active_batch_size==0`). 코드상 기대값 = 거의 전부 `(0,108)`(elif 분기,
  "normal full-SM decode stream").

**판별 기준**: A가 ≥90% 통과했는데 B 또는 C가 예상과 다르게 상당한 질량을
`{(74,34),(54,54)}`에 두고 있다면(예: >10%), 이는 라벨-실행 시차가 실재한다는 직접
증거이므로 **A의 90% 통과를 무효화**하고 §1-22 auto-revert가 지배하는 것으로 재해석한다
(사전등록, 결과 확인 후 변경 금지). B/C가 코드 기대값과 부합하면 그것 자체가 이 gate의
가장 실질적인 판별 정보다(위 "항등식" 절 참조).

## 사전등록 판정 규칙 (고정)

기본 population = A(decode-busy ∧ prefill in-flight)의 **시간가중** 분포,
rate 2와 rate 3의 **시간 합산 pooled**(개별 rate도 참고용으로 병기).

1. **해금**: pooled A의 시간가중 `frac({(74,34),(54,54)})` ≥ **0.90**, **그리고** B/C
   여집합 음성대조가 코드 기대값과 부합(각각 `frac({(108,0)})`/`frac({(0,108)})` ≥ 0.90
   — 위반 시 위 "판별 기준"에 따라 A의 통과를 기각).
2. **불해금**: 위 조건 미충족 — §1-22 auto-revert가 지배한다고 보고 §1-1의 기전 문장은
   계속 금지.
3. **UNMEASURABLE**(위 1/2 어느 쪽도 아님): A 구간의 시간가중 총 시간이 사실상 0이거나
   (`t_total_s < 0.5s`) 에피소드 수가 2 미만이면, 표본이 판정을 지지하지 못한다고
   명시하고 판정을 내리지 않는다(`m3r2_pinforce.sbatch`의 `MIN_PA_SNAPSHOTS` 관행과
   동일 정신 — 작은 n을 "통과"로 위장하지 않는다).
4. 해금되더라도, 위 "게이트가 항등식인가" 절의 caveat(**selector-level, hardware 층
   미확인**)은 §1-1에 함께 인용되어야 하며 생략 불가.
5. 결과가 rate 2와 rate 3에서 갈리면(하나는 해금 조건 충족, 하나는 아님) **더 엄격한
   쪽(불해금 또는 UNMEASURABLE)을 채택**한다 — 두 rate 중 하나라도 실패하면 "일반적으로
   해금된다"고 주장할 수 없다.

## 성능 수치에 대한 명시적 금지

이 job의 산출물(bench_serving JSONL, 서버 로그)에서 TTFT/TPOT/ITL/goodput/throughput을
**어떤 형태로도 인용하지 않는다** — n=1, cudagraph-ON 첫 boot 검증이 필요한 조합인지도
재확인 안 됨(873944에서 이미 boot_ok=1로 검증됐으므로 재검증 위험은 낮으나, 성능
수치로서의 자격은 어차피 n=1이라 없다). 이 job이 답하는 것은 **오직** 위 해금/불해금
판정 하나다.

## 아티팩트 레이아웃

```
workspace/engine-port/results/p1_gates/gate1/
  PREREG_GATE1_2026-08-06.md         (본 문서)
  pdmux_a100_smoke.yml               (p1_opint/에서 그대로 복사, 미수정)
  reference_manifest_873944.sha256   (873944 manifest 사본, 드리프트 대조용)
  gate1_run.sbatch                   (제출 스크립트)
  gate1_analyze.py                   (사후/인라인 분석 — A/B/C 시간가중+개수가중,
                                       에피소드/케이던스 진단, 해금 판정 출력)
  gate1_<jobid>.out / .err           (SLURM 표준 출력 — 결과 dir에 보존, 루트 아님)
  gate1_telemetry_<jobid>.jsonl      (PDMUX_TELEMETRY_PATH)
  gate1_srv_<jobid>.log              (서버 로그, capture/error evidence)
  gate1_r{2,3}_<jobid>.jsonl         (bench_serving raw, 인용 금지 — 존재만)
  runtime_source_manifest_gate1_<jobid>.sha256  (이번 실행의 sync manifest)
  gate1_result_<jobid>.txt           (판정 출력 전문)
```

## 부기(addendum) — job 874465(1차 시도) 계측 실패, 결정 규칙 불변

**2026-08-06, 결과 확인 후 작성**(위 판정 규칙 자체는 바꾸지 않음 — 계측 파이프라인
버그를 고쳐 재실행하는 것이며 방법론 게이트 #8 위반이 아니다: 판정 규칙·population
정의·집계 단위·임계값 전부 원문 그대로 유지, 바뀌는 것은 **텔레메트리 수집 성공
여부**뿐이다).

job 874465(`PDMUX_DUAL_WORKER_TRACE_EVERY=8` + `PDMUX_TRACE_FORCE_PREFILL=1`)를 1차로
제출·완료했다. rate=2 window는 정상 수집됐고(population A 100%, B 100%, C 97.22% —
아래 "1차 결과" 참조), **rate=3 window는 텔레메트리 행이 0개**였다 —
`gate1_analyze.py`가 사전등록 규칙 3(UNMEASURABLE)·규칙 5(per-rate stricter-wins)를
그대로 적용해 전체 판정을 `UNMEASURABLE`로 정직하게 반환했다(억지로 UNLOCK을 만들지
않음).

**근본 원인(직접 확인)**: `gate1_telemetry_874465.jsonl`의 마지막 줄
timestamp_monotonic_s=12695532.12가 서버 자신의 SIGTERM(10:32:41, srvlog)보다 **72초
먼저** 끊겨 있다 — 그 사이 서버는 정상적으로 요청을 계속 처리했다(`Successful
requests: 120` 양쪽 rate 모두 정상 완료, `Median TTFT`/`Median TPOT` 모두 정상 범위).
즉 **스케줄러가 멈춘 게 아니라 텔레메트리 파이프라인만 끊겼다.**
`AsyncJsonlTelemetry`(`multiplex/telemetry.py`)는 bounded queue(`max_events=65536`)에
`put_nowait`로 넣고 가득 차면 **조용히 드롭**(`dropped_events` 카운터만 증가, 파일에는
아무것도 안 쓰임)하는 설계다. rate=2 window 실측 enqueue율이 초당 ~900대(idle-spin
구간이 지배적 — decode/prefill 둘 다 비어 있을 때 이벤트 루프가 매우 빠르게 도는데,
`TRACE_EVERY=8`이 그 스핀까지 촘촘히 잡은 것으로 보임)였고, Lustre `/scratch`에 줄 단위
로 쓰는 단일 writer 스레드가 이를 못 따라가 큐가 가득 찼다 — 이후 emit() 전부가 무음
드롭. 이 job은 성능 주장을 하지 않으므로(사전등록에 이미 명시) 이 실패가 §1-1에 아무
영향을 주지 않지만, **계측 자체의 실패 모드**로 향후 gate에 재사용할 가치가 있어
기록한다.

**조치(2026-08-06, 재실행 전)**: `gate1_run.sbatch`에서 `PDMUX_DUAL_WORKER_TRACE_EVERY`를
**기본값(32, `results/s8_frontier/m3r2_pinforce.sbatch`의 검증된 선례와 동일)으로 되돌림**
— `PDMUX_TRACE_FORCE_PREFILL=1`은 유지(population A는 이 플래그만으로 이미 충분히
조밀했다 — n_episodes=57, n_snapshots=1656, 케이던스 중앙값 1 iteration). SIGTERM과
SIGKILL 사이에 5초 drain 유예도 추가(방어적, 근본 수정은 아님). **판정 규칙·population
정의·집계 단위·임계값은 전부 원문 그대로**, 이 addendum이 유일한 변경 사유임을 명시.

**1차 결과(참고용, rate=2만, 완전한 pooled 판정 아님 — 새 job의 완전한 결과로 대체됨)**:
- `GATE1_POOLED_A(rate2 only)`: t_total_s=17.542, n_episodes=57, frac_target=**1.0000**
  (100% `(74,34)`, `(54,54)`는 0회 방문 — decode batch가 36 미만이었던 것과 정합,
  `decode_bs_divisor=36`).
- `GATE1_POOLED_B(rate2 only)`: t_total_s=66.095, n_episodes=21, frac_target=**1.0000**
  (100% `(108,0)`, §1-22 auto-revert 재확인).
- `GATE1_POOLED_C(rate2 only)`: t_total_s=31.790, n_episodes=73, frac_target=**0.9722**
  (97.22% `(0,108)`, 2.35%가 `(74,34)`로 새어나감(leak), 0.42%가 `(108,0)`) — **이
  2.78%가 바로 위 "게이트가 항등식인가" 절이 예견한 라벨-실행 시차의 실측 증거다**:
  코드 분기는 결정적으로 idx=3을 요구하지만, 에피소드 경계 근처 전이 iteration이 드물게
  포착되면서 순수 100% 정체성에서 벗어난다 — 이 gate가 "코드를 다시 읽는 것"과 다른,
  실측 정보를 실제로 담고 있다는 직접 증거.

## 부기 2 — 메인 세션의 정정: rate=2도 절단됐고, coverage guard 추가(874478 시작 전)

**2026-08-06, 메인 세션이 원자료를 직접 재계산해 확인, 이 문서 개정 시점 = 874478
여전히 PENDING(제출 후 실행 전).** 위 부기 1이 놓친 것 두 가지.

**정정 1 — rate=2 window도 "완전 포착"이 아니었다.** `gate1_telemetry_874465.jsonl`은
100,224행, `timestamp_monotonic_s` 범위 12695378.32 → 12695532.12(span 153.8s)로
끝난다. rate=2 window는 t0=12695439.65 → t1=12695555.08(115.4s)인데, 텔레메트리는 창의
**92.5초 지점(80.11%)에서 끊긴다** — **마지막 22.96초(19.9%)가 결측**이고, 이건
무작위 결측이 아니라 **창의 꼬리 — 부하와 decode batch가 가장 큰 구간**이다. 위 부기 1의
"`rate2_A frac=1.0000`(100% `(74,34)`)"는 **`(54,54)`가 가장 나오기 쉬운 구간이 정확히
잘려나간 절단된 창 위의 값**이었다 — `UNLOCK-eligible`으로 채점한 것은 과대 주장이었다
(rate=3 자체가 전체 판정을 `UNMEASURABLE`로 끌어내려 최종 출력은 결과적으로 안전했을
뿐, 그건 설계가 아니라 우연이었다).

**정정 2 — `dropped_events` 카운터는 이 실패를 원리적으로 관측할 수 없다.**
100,224행 **전부 `dropped_events==0`**이고 `telemetry_close` 행은 **0건**이다.
`telemetry.py`의 `emit()`이 카운터를 **그 이벤트 자신에 실어 보내므로**(`fields.setdefault
("dropped_events", self.dropped_events)`), 큐가 영구 포화되면 카운터를 나르는 이벤트 자체가
함께 드롭된다 ⇒ **카운터가 자기가 잡아야 할 실패에 의해 검열된다.** 즉 이전 addendum의
"큐 포화로 드롭됐다"는 근본원인 진단은 **이 아티팩트만으로는 확증도 반증도 불가**하다
(진단이 틀렸다는 뜻이 아니라 — enqueue율 추정(rate=2 window 전체 평균 ~750/s, 실은
데이터가 있는 92.5s 구간에서 계산한 것이라 실제 순간 enqueue율은 더 높았을 수 있음)과
서버 로그가 계속 정상 응답한 정황은 여전히 원인으로서 유력하지만, 카운터 자체로는 뒷받침
안 된다는 뜻이다).

**부수 확인**: rate=2 창 86,404행 중 83,927행(97%)이 population B(decode-idle)이고
population A는 1,656행뿐이다 — 대역폭을 지배하는 건 gate가 가장 관심 없는 모집단이다.

### (a) coverage guard — `gate1_analyze.py`에 추가 완료, 874478 시작 전(PENDING 상태에서 작업)

`telemetry_coverage = (min(last_row_ts,t1) − max(first_row_ts,t0)) / (t1−t0)`를 매 window
마다 계산해 **항상 출력**(`GATE1_COVERAGE` 줄, pass/fail 무관), **coverage < 0.98이면 그
rate 창은 `UNMEASURABLE(partial coverage: X%)`**로 판정 — 기존 `UNMEASURABLE`과 동일
엄격도(severity 1)로 취급하도록 `severity()` 헬퍼와 stricter-wins 로직을 일반화했다(모든
`UNMEASURABLE*` 접두 라벨이 severity 1, `NO_UNLOCK`이 0(가장 엄격), `UNLOCK`/
`UNLOCK-eligible`이 2). **이건 판정 규칙 변경이 아니라 타당성 선행조건 추가다**:
target set·집계 단위(시간가중)·≥0.90 임계값·population 정의 전부 불변이고, coverage
guard는 **오직 UNMEASURABLE 쪽으로만 판정을 이동시킬 수 있다** — coverage가 좋은 창은
이전과 동일하게 채점되고, coverage가 나쁜 창은 이전엔 (행이 하나라도 있으면) 채점됐던 것이
이제 "측정 안 됨"으로 정직하게 표시될 뿐이다. 이 방향성이 방법론 게이트 #8(결과를 본 뒤
결정 규칙 변경 금지) 위반이 아니라고 판단한 근거다 — **단방향으로만 더 엄격해지며 이번
판정을 UNLOCK 쪽으로 미는 경우는 수학적으로 존재하지 않는다**(coverage guard가 하는 일은
기존 pass 조건 위에 추가 AND 조건을 얹는 것뿐).

**874465 재채점 결과**(`gate1_result_874465_RESCORED.txt`, 동일 스크립트로 재실행):
`GATE1_COVERAGE rate=2: telemetry_coverage=80.11%` → `GATE1_PER_RATE:
[('2','UNMEASURABLE(partial coverage: 80.11%)'), ('3','UNMEASURABLE(empty window)')]` →
`GATE1_VERDICT: UNMEASURABLE(partial coverage: 80.11%)`. 부기 1의 `UNLOCK-eligible`
라벨은 **이 재채점으로 대체**됐다 — 정본은 이 재채점 결과다.

### (b) 874478 위험 평가 — cadence 8→32의 해상도 손실

`PDMUX_DUAL_WORKER_TRACE_EVERY` 8→32(4배 성김)는 이번 조치가 겨냥한 인프라 병목(writer
스레드가 idle-spin 지배 enqueue율을 못 따라감)에는 맞는 방향이지만, **population B/C의
표본 밀도도 함께 4배 성겨진다**(A는 `PDMUX_TRACE_FORCE_PREFILL=1` 덕에 영향 없음 — 874465
에서 이미 median gap=1, n_episodes=57로 조밀했다). 874465의 population C는
n_episodes=73, n_snapshots=821(31.79s 동안, episode당 평균 ~11.2 snapshot)이었고 여기서
2.35%(0.748s)의 `(74,34)` 누출을 검출했다 — cadence가 4배 성겨지면 episode당 평균
snapshot이 ~2.8개로 줄어, **누출이 발생한 그 특정 episode(들) 안에 표본이 하나도 안
찍힐 확률이 유의미하게 올라간다** ⇒ 874478의 population C/B가 874465보다 "더 깨끗하게"
(예: frac_target closer to 1.0000) 나오더라도, 그것이 **실제로 더 순수해서인지 아니면
단지 성긴 표본이 짧은 누출 구간을 놓쳐서인지 이 실험 하나로는 구분할 수 없다** — 이는
과대추정이 아니라 **과소검출(false-clean) 방향의 편향**이므로 §1-1 해금 판정(A/B/C
전부 ≥0.90 필요)을 UNLOCK 쪽으로 유리하게 왜곡할 위험이 있다. **완화책**: 874478 결과
해석 시 이 caveat을 반드시 병기하고, coverage guard로 걸러지지 않는 이 잔여 위험은
"UNLOCK이 나오면 그 결론은 cadence-32 해상도 한도 내에서만 성립"으로 제한한다. 근본
해결(population B만 선별 서브샘플링 등 엔진 코드 변경)은 **이번 gate 범위 밖 후속
항목으로만 기록**한다(엔진 코드 수정 없음, engine-porter로 이관 가능한 후보).

### (c) 케이던스 민감도 대조 — 874478 완료 후 보고 예정

874465 rate=2(cadence 8, **80.11% 절단 창**)와 874478 rate=2(cadence 32, 완전 포착
목표)는 같은 arm·같은 워크로드(모델/ctx/backend/정책/워크로드/seed 컨벤션 전부 동일,
유일한 차이가 `PDMUX_DUAL_WORKER_TRACE_EVERY`)이므로 **케이던스가 시간가중 residency
추정을 바꾸는지**의 직접 대조가 부산물로 생긴다(§0 종결이 정확히 이 축이었으므로 값어치
있음). 874478 완료 후 별도로 보고하되, **874465 쪽이 절단된 창(80.11%)이라는 점을 반드시
동시 병기**하고 — coverage가 다른 두 창을 coverage 캐비트 없이 나란히 놓고 "cadence
효과"라고 결론 내리지 않는다(coverage 차이와 cadence 차이가 교락돼 있다).

### (d) 지시 준수

874478은 취소하지 않았다(대기 시간 낭비 방지, 위 coverage guard만으로 결과 해석이
안전해짐). 이 addendum 이후로도 성능 헤드라인은 만들지 않는다 — 산출은 여전히 "§1-1
기전 문장 해금: 예/아니오/UNMEASURABLE"뿐이다.
