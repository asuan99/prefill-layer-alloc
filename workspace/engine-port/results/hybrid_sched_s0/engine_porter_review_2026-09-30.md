# engine-porter 읽기 전용 검토 — S0 rev1 §7 (훅 실현 가능성) + TTFT 분해 전제 1

2026-09-30 · engine-porter(정적 독해, 파일 수정 0, 테스트 실행 0) · 메인 세션이 반환문을 기록 · GPU 0
대상: `DESIGN_S0_REV1_2026-09-30.md` §1·§5·§7, `reports/ttft_measurement_definition_review_2026-09-28.md` §3.2 전제 1.
미러 `src/multiplex/*.py`와 dev tree 사본은 `cmp` 바이트 동일 확인(줄번호 공통). 성능 주장 없음.

## 1. §7-1 decode PAUSE 훅

- **cudagraph와 양립.** PD-mux decode 그래프는 모든 stream group 인덱스에 `f"{stream_idx}_{bs}"` 키로 캡처(dev
  `cuda_graph_runner.py:808-815`, 조회 `cuda_graph_runner.py:680-681`). step을 통째로 건너뛰면 replay만 안 할 뿐 캡처 상태 불변. split prefill은
  원래 eager(`model_executor/model_runner.py:2709-2728`). ⇒ PAUSE는 step 단위 on/off이며 green-ctx를 step 중간에 바꾸지 않는다.
- **green-ctx는 하드 분할 — 108 SM을 prefill에 주려면 idx0 `(108,0)` 전환 필수.** `initialize_stream_groups`(dev
  `pdmux_context.py:124-138`): idx0=(108,0) 평범한 스트림, idx1..k=green-ctx 분할, 마지막=(0,108). 분할 안에서 decode를 멈춰도
  prefill 스트림은 `prefill_sm`에 묶인다. 전환마다 드레인(`multiplexing_mixin.py:1341-1357`) 1회.
- **`adjust_stream_groups` 세 갈래 어느 것도 PAUSE 상태를 idx0으로 보내지 않는다**(`multiplexing_mixin.py:1175-1211`). 최소 경로 = R2 게이트에서
  `target_decode_sms==0` → `matches[0]`(`multiplexing_mixin.py:672-682`)이 idx0 반환 → 기존 전환 코드 재사용. 단 "idx0이면 정지"를 암묵 신호로
  쓰면 안 됨(decode 빈 뒤 merge로 idx0에서 running 비어 있지 않은 정상 상태 존재, `multiplexing_mixin.py:1305`, `multiplexing_mixin.py:1531-1541`) → 명시 플래그 필요.
- ★**정확성 함정: `update_running_batch`도 쌍으로 건너뛰어야 한다.** 이 함수는 끝에서 `prepare_for_decode()`(dev
  `managers/scheduler.py:2648`)로 KV 슬롯 할당·`kv_allocated_len` 전진. mixin은 `multiplexing_mixin.py:1297`에서 decode 실행(`multiplexing_mixin.py:1376-1404`)보다 먼저 부른다.
  decode만 건너뛰면 PAUSE step마다 KV 할당 누적·seq 상태 어긋남.
- 부수 효과: ITL 표본(`_r2_itl_samples`/`_slo_tpot_ema`, `multiplexing_mixin.py:1265-1280`)은 정지 iteration 제외 필요 — 그러면 정지된 요청의
  실제 토큰 간격이 컨트롤러에 안 보여 G-PAUSE slack이 피해를 과소 관측(요청 단위 gap 계측 별도 필요). 금지 조합 fail-fast
  (`PDMUX_STICKY_PARTITION`·`PDMUX_SLO_SCHED`·`PDMUX_LA_COORD`, `_init_sticky_partition` `multiplexing_mixin.py:388-455` 방식). TP>1 [미확인]:
  rank별 판정 불일치 시 collective hang → broadcast 필요(현 캠페인 TP=1).
- **true-dual: coordinator 수준 정지 지점 없음.** `PhaseCoordinator`(`dual_worker.py:233-275`)는 장부, runtime(`dual_worker.py:465-491`)은
  제출된 작업만 실행, 메인 루프가 매 iteration 락스텝 join(`multiplexing_mixin.py:1476-1481`, `multiplexing_mixin.py:1497-1502`). PAUSE = `multiplexing_mixin.py:1390` `submit(DECODE, …)`
  건너뛰기. `select_partition(0)`은 드레인 후 `safe_to_switch()`(`dual_worker.py:197-207`) 통과 [미확인: GPU 실측].
- correctness gate 테스트(CPU `tests/pdmux_loop_fakes.py`): 정지 step에 `prepare_for_decode` 0회 · 재개 후 결과열 OFF와 동일 ·
  PAUSE 중 `stream_idx==0` · 해제 후 분할 복귀 · legacy/true-dual 양쪽 · 기본값 OFF 바이트 동일 · 금지 조합 fail-fast ·
  n_max 강제 · 변이본(decode만 건너뜀) 실패(교훈 53). GPU: PAUSE OFF/ON 디코딩 출력 동치(gate 2), observer effect(gate 4).

## 2. §7-2 F1 재정렬과 `--schedule-policy`

- upstream 정책은 PD-mux 경로에 자동 적용: `update_split_prefill_batch`(`multiplexing_mixin.py:1218-1237`) → `get_new_batch_prefill`(dev
  `managers/scheduler.py:2315`) → `policy.calc_priority`(`managers/scheduler.py:2386`) → `PrefillAdder`(`managers/scheduler.py:2403-2489`, head-of-line `break`).
- **radix OFF 제약**: `_validate_and_adjust_policy`(`schedule_policy.py:167-181`)가 tree_cache disable이면 `lpm`/`dfs-weight`를
  **조용히 FCFS로** 바꾼다. λ0 계열은 `--disable-radix-cache`(`lambda0.sbatch:342`) ⇒ 실효 옵션은 `fcfs`/`lof`/`random`/
  `routing-key`뿐. 어느 것도 입력 길이·추정 지연으로 정렬하지 않음 ⇒ G-SLF는 기존 옵션으로 표현 불가. CLI choices의
  `"priority"`는 enum(`schedule_policy.py:87-93`)에 없어 init ValueError[정적 독해]. priority scheduling은 fcfs/lof에서만(`server_args.py:6158-6162`).
- PD-mux는 `chunked_prefill_size == -1` 강제(`server_args.py:6130-6132`) ⇒ chunked_req 경로 없음.
- 새 정렬 키 삽입 지점: (A) mixin에서 `get_new_batch_prefill()` 직전 `waiting_queue` 정렬(기본 OFF env; fcfs면 `calc_priority`
  no-op `schedule_policy.py:120-125`, 필터가 순서 보존 `managers/scheduler.py:2497`; `--enable-priority-scheduling` ON이면 덮어쓰므로 조합 거부) · (B) upstream
  `CacheAgnosticPolicy` enum+`calc_priority` 분기+choices(2파일 패치·manifest·`dev_tree_edits.md`; ★`profile.ENGINE_SOURCE_MODULES`
  `profile.py:403-412`에 `schedule_policy.py`가 없어 `engine_source_hash`가 안 움직임 → provenance 별도).
- mamba 슬롯 키: `PrefillAdder`에 mamba 슬롯 검사 없음(`schedule_policy.py:465-468`, `schedule_policy.py:486-489` 토큰 잔량만). running 상한은
  `get_num_allocatable_reqs`(`managers/scheduler.py:2309-2313`). M0 보강: `max_mamba_cache_size = max_running_requests`는
  `disable_radix_cache ∧ max_running_requests is not None`일 때만(`model_runner_kv_cache_mixin.py:222-228`), λ0는 명시
  (`lambda0.sbatch:343`). 풀 소진 시 할당기 동작 [미확인].

## 3. §7-3 `SplitDecision`에 PAUSE 추가 시 컨트롤러 상호작용

- `target_decode_sms=0`으로 표현하면 `stabilize`의 downshift 분기(`controller.py:330-338`: `enough_slack` 3 epoch + dwell
  `controller.py:236-239`)에 걸려 PAUSE가 수 epoch 지연·차단 ⇒ hysteresis 밖 별도 필드(`pause_decode_steps: int = 0`) 권장. `RuntimeSnapshot`
  필드 집합은 `tests/test_host_worker_metrics.py:295,311`에 고정.
- **케이던스 동결**: `_cadence_due`/`_overload_epoch_elapsed`(`controller.py:168-194`)는 `decode_iterations` 증분 ≥4 AND 시간이고,
  `decode_iterations`는 decode 실행 시만 증가(`multiplexing_mixin.py:1380`) ⇒ PAUSE 중 정규 평가 절대 due 안 됨, HOLD 반복. **PAUSE 종료를
  컨트롤러에 맡기면 종료되지 않는다** → 런타임 카운터(n step 또는 prefill final 제출) 필요. dwell도 동결(upshift 복귀는
  dwell 미검사 `controller.py:339-340`).
- 긴급 경로 `_live_underprediction`(`controller.py:162-166`)은 PAUSE 취소 안전판으로 쓸 수 있으나, ITL 표본을 빼면 p95 조건이 stale →
  실제 발화 가능한 것은 occupancy 조건뿐.
- **데드락 경로**: 컨트롤러 밖 런타임 규칙(Bullet식 G-PAUSE)으로 두면 래치된 `r2_admission_limited`가 HOLD로 계속 전달되고
  `_r2_admission_holds`(`multiplexing_mixin.py:2045-2067`)는 running이 빌 때만 해제 — decode 정지면 영원히 안 빔, 평가는 케이던스 동결 ⇒
  **prefill 차단 + decode 정지 = 진행 0**. 필수 불변식: (i) PAUSE는 `split_prefill_batch is not None ∧ not
  wait_prefill_kernel_done`에서만 (ii) 하드 상한 n_max (iii) `r2_admission_limited`와 PAUSE 상호 배타 assert (iv) prefill
  final 제출 시 자동 해제 — CPU fake 루프 테스트 + 변이본(i/ii 제거) 실패.
- R2 target은 prefill span 안에서만 적용, 그 뒤는 `adjust_stream_groups`(`multiplexing_mixin.py:1358-1368`) ⇒ PAUSE 해제 후 복귀 인덱스 명시 필요.

## 4. §7-4 S0 offline 호출 경계

- `profile.py`는 stdlib만(`profile.py:11-18`), `controller.py`는 stdlib+profile(`controller.py:5-21`, 상대 import 실패 시 `from profile import` 폴백).
  로드는 `tests/test_profile_controller.py:12-28` 방식(`spec_from_file_location` → `sys.modules["profile"]`) — ★stdlib `profile`을
  가리므로 같은 프로세스에서 `import sglang` 금지. `workloads.py` 의존 [미확인].
- 호환성 검사 우회: `estimate(..., runtime=None)`이면 `is_compatible` 건너뜀(`profile.py:314`); `HybridInformedPolicy`의
  `runtime_environment` 기본값 None(`controller.py:380`, `controller.py:399-404`) — 서빙과 다른 경로("호환 프로파일" 조건 모사)임을 명기.
- ★**순수 vs 상태**: `estimate()`/`predict()` 순수. 두 `decide()`는 `CoarseGrainedController` 상태(케이던스·dwell·streak·
  `admission_limited`) 의존 — 비평가 iteration은 케이던스가 만든 HOLD. S0는 (a) 비교 수준(estimator vs decide) 사전 등록
  (b) decide 수준이면 G/H 별도 인스턴스·시간순 입력 (c) 단조 `timestamp_s`/`decode_iterations` (d) M4 셔플은 컨트롤러 동역학도
  깨뜨림. `GenericDynamicPolicy.decide`에 `bucket_changed` 없음(`controller.py:425-427`); G의 SQUEEZE는 한 칸씩·하한 `states[0]`(`controller.py:436-443`).
- `HybridModelProfileV1.validate()`(`profile.py:99-111`): schema `pdmux.hybrid-model-profile/v1`, `model_id`/`model_config_hash`
  비어 있지 않음, `attention_layers+ssm_layers>0`, `attention_kind ∈ {GQA,MQA,MHA,NONE,MIXED}`, `points` 비어 있지 않음. 생성자
  필수: `model_id, model_revision, model_config_hash, environment(RuntimeEnvironment), attention_layers, ssm_layers,
  attention_kind, num_attention_heads, num_kv_heads, parameter_count, points`. `DecodeLatencyPoint`(`profile.py:48-65`) 필수 8필드, 검증
  `repeats≥1, measured_steps≥1, 0≤p50≤p95≤p99`.
- 합성 시 함정: `heldout_residual_p95_ms` 미검증(음수 통과, margin=`max(0.1·SLO, residual)` `profile.py:310-313`) · 격자 완전성 미검사
  (허용 D 상태에 점 없으면 `KeyError`가 조용히 `continue` `profile.py:329-334`) · 108 점 없으면 긴급 추정 `upper=inf, confidence=0`
  (`profile.py:347-363`) → `stabilize` D108-risk overload 참(`controller.py:311-315`) ⇒ 16/24/34/44/108 전부에 batch×context 격자 채울 것.
- ★**M1 관련 사실**: estimator는 `attention_ratio`·`attention_layers`·`attention_decode_curve`·`ssm_decode_curve`를 **읽지
  않는다**(`profile.py:239-301`, `points`와 `heldout_residual_p95_ms`만). 저장소 전체 소비처 `profile_cli.py:106`뿐. ⇒ M1을
  프로파일 메타필드 편집으로 하면 **항등식**; 비용 모형에서 `points`를 재생성해야 한다. (메인 세션 독립 재확인: grep 동일.)

## 5. TTFT 분해 전제 1 (해소) + 추가 발견 2건

- chunked 재진입은 PD-mux에서 발생하지 않는다(`chunked_prefill_size == -1` 강제).
- **split-prefill 재진입에서 `set_forward_entry_time`은 다시 호출되지 않는다**: span 진행 중 `update_split_prefill_batch`가
  `multiplexing_mixin.py:1219-1220`에서 조기 반환 → `get_new_batch_prefill` 미진입; 스탬프는 admission 시 1회(`managers/scheduler.py:2523`); 함수 자체가
  `forward_entry_time == 0.0`일 때만 기록(`req_time_stats.py:612-617`). **true-dual도 동일**: admission과
  `process_batch_result`는 메인 스케줄러 스레드(`multiplexing_mixin.py:1227`, `multiplexing_mixin.py:1529-1533`), worker는 `run_batch`만. `prefill_finished_time`도
  동일 경로(SPLIT_PREFILL ∈ `is_extend()` `forward_batch_info.py:119`, `scheduler_output_processor_mixin.py:180-181`,
  `==0.0` 가드 `req_time_stats.py:670-673`; 호스트가 이벤트 `query()` 성공을 관측한 iteration에 찍혀 폴링 지연 포함).
- **추가 발견 1**: `prefill_run_batch_start/end_time`은 `forward_mode == EXTEND`에서만 기록(`managers/scheduler.py:2676-2677`, `managers/scheduler.py:2814-2815`)
  — split은 `SPLIT_PREFILL`이라 **PD-mux에서 미기록**. 분해에 이 필드를 쓰지 말 것.
- **추가 발견 2**: decode OOM retract 시 `_add_request_to_queue`(`managers/scheduler.py:1961-1969`)가 `set_wait_queue_entry_time`을 재호출하고
  이 함수는 마지막에 `self.wait_queue_entry_time = ts`를 **무조건 덮어쓴다**(`req_time_stats.py:590-610`; 메인 세션 독립
  재확인 — 기존 값이 있으면 `set_retract_time(ts)`도 기록). `forward_entry_time`은 첫 값 유지 ⇒ retract된 요청의
  `W_queue = forward_entry − wait_queue_entry`가 **음수 가능**. 분해 규약에 retract 요청 제외 또는 `retract_time` 플래그 규칙
  필요. 부수: `oldest_prefill_age_ms`도 retract 시 리셋. R2 admission 보류 시간은 `W_queue`에, admission 직후 첫 전환
  드레인(`multiplexing_mixin.py:1341-1357`)은 `S_prefill`에 들어간다.

## 6. S1 최소 구현 범위 (참고 — S0 rev1 `NO-GO`라 착수하지 않음)

| 대상 | 파일·함수 | 변경 | 테스트 |
|---|---|---|---|
| PAUSE 표현 | `controller.py` `SplitDecision` +`pause_decode_steps=0`, hysteresis 밖 별도 메서드 | 기본값 0이면 기존 동일 | 상호 배타·긴급 취소·케이던스 동결 |
| PAUSE 실행 | `multiplexing_mixin.py` `event_loop_pdmux`: `update_running_batch`(`multiplexing_mixin.py:1297`)+decode 제출(`multiplexing_mixin.py:1376-1404`) 쌍 건너뜀, idx0 전환/해제(`multiplexing_mixin.py:1341-1357`), ITL 표본 제외(`multiplexing_mixin.py:1278`), n_max·final 자동 해제 | 기본 OFF env, 헬퍼는 클래스 끝(줄번호 보존), `line_citations.json` 갱신 | fake 루프: legacy/true-dual, prepare_for_decode 0회, idx0 체류, 복귀, OFF 바이트 동일, 금지 조합, 변이본 실패 |
| 조합 거부 | `init_pdmux`/`_init_sticky_partition` 계열 | sticky·SLO_SCHED·LA_COORD·TP>1 거부 | fail-fast |
| telemetry | `RuntimeSnapshot`/trace | `decode_paused`·정지 카운트 전 arm 대칭 | `test_host_worker_metrics.py:295,311` 갱신, 대칭 테스트 |
| true-dual | `dual_worker.py` | 로직 변경 없음 | fake 루프 true-dual |
| F1 정렬(선택) | mixin `get_new_batch_prefill` 직전 정렬(방식 A) | 기본 OFF, priority scheduling 거부 | 정렬 키·OFF=FCFS 동일 |
| GPU gate | 대여 GPU | OFF/ON 출력 동치, observer effect | gate 2·4 통과 전 기본 ON 금지 |

**실현 불가·고비용**: green-ctx 분할 안 드레인 없는 PAUSE는 prefill SM 이득 0 · PAUSE 종료를 케이던스에 맡길 수 없음 ·
admission latch 아래 상한 없는 PAUSE = 데드락 · radix OFF에서 `lpm`/`dfs-weight` 조용히 FCFS · 정지 요청의 실제 토큰 간격이
현 컨트롤러 입력에 없음 · TP>1 broadcast 미구현 · M1을 메타필드로 하면 항등식.
