# ENGINE CHECK — N2(3-a) co-load(8192 입력 / 출력 1) 합류 여부 · 동거 창 실현 조건 (2026-10-07)

- 작성: engine-porter (코드 독해 전용, CPU, 엔진 코드 수정 0, GPU 실행 0)
- 대상: `PREREG_N2_3A_2026-10-07.md` rev1 §8-2 위험 (i)(차단 항목), §7-2, §8-2 (ii), §6-2
- 근거 트리:
  - 미러 `workspace/engine-port/src/multiplex/multiplexing_mixin.py` (로컬 dev tree
    `~/Experiments/KISTI/sglang_engine_dev/python/sglang/srt/multiplex/multiplexing_mixin.py`와 `diff -q` 동일)
  - 로컬 dev tree(= SGLang v0.5.10 git checkout, HEAD `1519acf`)의 `managers/scheduler.py`·
    `managers/schedule_batch.py`·`managers/scheduler_output_processor_mixin.py`.
    `schedule_batch.py`·`scheduler_output_processor_mixin.py`는 `git status`상 미수정(pristine upstream).
    `scheduler.py`의 로컬 수정(`devtree_manual_edits.patch` 루프 디스패치 1곳 + probe hook 2개 패치)은
    아래 인용 함수(`update_running_batch`, `get_new_batch_prefill`)를 건드리지 않는다.
    `devtree_manual_edits.patch`는 로컬 트리에 적용된 상태(`patch -R --dry-run` 통과)다.
  - upstream 대조: `git show HEAD:python/sglang/srt/multiplex/multiplexing_mixin.py`에도 같은
    "process_batch_result → filter 없이 merge_batch" 순서가 있다(upstream 126행 `update_running_batch`,
    211–215행 merge). 즉 아래 (a)의 거동은 이 프로젝트 패치가 아니라 upstream PD-mux 루프의 성질이다.
- 한계: VESSL Job이 설치할 봉인 트리와 로컬 트리의 바이트 동일성은 여기서 재확인하지 않았다(프리플라이트
  digest 몫). CPU fake 실행은 하지 않았다 — 로컬에서 `sglang.srt.managers.schedule_batch` import가
  `sgl_kernel` 부재로 실패하고, 해당 로직을 재구현한 fake는 검증 대상 코드를 복사한 항등식이 된다
  (교훈 9 / `CONSENSUS §3` 항목18). 따라서 아래 결론은 **코드 독해**이며 GPU 실증은 §(d)의 텔레메트리 서명이 맡는다.

## (a) 출력 1 co-load 요청은 decode running batch에 "합류"하는가

**결론: 객체 수준에서는 `running_batch.reqs`에 병합되지만, 병합 시점에 이미 finished이며 다음 루프의
`update_running_batch` → `filter_batch`가 decode 발사 전에 제거한다. decode iteration에는 한 번도
참여하지 않는다 ⇒ `decode_iteration.bs`는 b에서 바뀌지 않는다(co-load 기인 분만).**
확신도: 높음(코드 독해, 경로 단일·분기 없음). GPU 실증 전이므로 "작동한다"는 주장은 아니다.

경로(legacy `event_loop_pdmux`, `PDMUX_TRUE_DUAL_WORKER` unset, 줄 번호 = 미러/로컬 트리):

1. prefill 마지막 span 완료 관측 → `multiplexing_mixin.py:1516-1527`
   (`_dual_worker_prefill_ready` → `_phase_on_prefill_end` 기록, 병합 **전**).
2. `process_batch_result(split_prefill_batch, …)` `multiplexing_mixin.py:1528-1530`
   → `scheduler.py:2864-2870`: `ForwardMode.SPLIT_PREFILL`은 `is_extend()`(`forward_batch_info.py:119`)라
   `process_batch_result_prefill`로 간다.
3. `scheduler_output_processor_mixin.py:175-192`: chunked prefill이 아니므로(`--chunked-prefill-size -1`은
   pdmux 필수, `server_args.py:6129-6131`) `req.output_ids.append(next_token_id)` → `req.check_finished()`(188)
   → `schedule_batch.py:1164-1169` `len(output_ids) >= max_new_tokens(=1)` ⇒ `FINISH_LENGTH` →
   `release_kv_cache`(192)로 KV·mamba 슬롯 반환, `stream_output`(325)로 응답 종료.
4. **filter 없이 병합**: `multiplexing_mixin.py:1531-1532` `self.running_batch.merge_batch(self.split_prefill_batch)`
   (`schedule_batch.py:2262-2306`, `reqs.extend` — finished 여부를 보지 않음). 이후 `adjust_stream_group = True`(1541).
   (참고: 정규 upstream 루프는 병합 전에 `last_batch.filter_batch()`를 한다 — `scheduler.py:2246-2252`. PD-mux 루프는 이 단계를 건너뛴다.)
5. 다음 루프: `recv` → `_dual_worker_sync`(1283-1285) → 입장 `update_split_prefill_batch`(1287-1294) →
   **`self.running_batch = self.update_running_batch(self.running_batch)` (1297)** → `scheduler.py:2576`
   `batch.filter_batch(...)` → `schedule_batch.py:2197-2202` `keep_indices = [i … if not reqs[i].finished()]`
   ⇒ co-load 요청 제거, bs 감소 시 `batch_is_full = False`(`scheduler.py:2642-2643`).
6. decode 발사는 그 뒤 `multiplexing_mixin.py:1379-1405`. 4와 6 사이에 GPU decode 실행은 없다.

부수 효과(4–5 사이의 "finished지만 아직 reqs 안에 있는" 1루프 구간) — bs 필터와 무관하지만 초안에 영향:

- **입장 cap 계산이 이들을 센다**: 5의 입장(1292)이 filter(1297)보다 먼저라
  `get_new_batch_prefill`의 `running_bs = len(self.running_batch.reqs)`(`scheduler.py:2370`, 2445-2447)가
  b + n_done(직전 span)이다 → §(b)-F3.
- **runtime_snapshot이 이들을 센다**: `_dual_worker_sync`는 병합 직후(1543)와 다음 루프 recv 직후(1285)
  둘 다 filter 전이다. 그 sync가 32-grid 표본이면 `_r2_runtime_snapshot`(457-516)의
  `active_decode_sequences = b + n_done`, `context_p50/p95/max`에 8193 길이 요청이 섞인다 → §(b)-F5.
- **`prefill_span_start.running_bs`도 filter 전 값**(`_phase_on_prefill_start` 2182-2205, 호출 1292→1235→870) → §(d).

## (b) `bs == b` 필터가 버리는 동거 창 — 구조 추정과 우회

**구조 추정(코드 독해 기준)**: co-load 기인으로 버려지는 W_co iteration = **0**. W_co 안에서 `bs ≠ b`가
생길 수 있는 경로는 co-load와 무관한 셋뿐이다: (i) b개 자신의 종료(초안 §8-3의 O 크기 규칙이 담당)
(ii) KV 부족 retraction(`scheduler.py:2587-2592`, 초안 `RETRACT` 셀 제외가 담당) (iii) 클라이언트 취소(U-1, 창 뒤).

**반사실(합류했다면)**: 예컨대 co-load에 `max_new_tokens`를 안 실으면 SGLang 기본값 128
(`sampling/sampling_params.py:42`)이 적용되어 실제로 decode에 합류한다. 그러면 span k 종료 뒤 bs = b + Σn_done
(2→4→6→8)로 계단 상승, `bs == b`는 첫 co span 종료 전 구간(≈ W_co의 1/4, span 길이 균등 가정 — 추측)만 남고,
A2도 약 128 iteration 동안 bs = b+8로 오염된다. bs가 캡처 bs 밖으로 가면 decode cudagraph 패딩도 바뀔 수 있다(미확인).
⇒ 실패 모드는 엔진이 아니라 **요청 페이로드 등록 누락**이다.

**우회·등록 권고**

- F1 (필수) co-load 요청 본문에 `sampling_params.max_new_tokens = 1`을 **명시** 등록. `trace_loadgen.py:53-59`는
  `output_tokens`를 그대로 `max_new_tokens`로 싣으므로 `output_tokens: 1`이면 충족. `ignore_eos`는 무관(길이 판정이 먼저, `schedule_batch.py:1164`).
- F2 (필수) `max_new_tokens = 0` 금지 — `is_prefill_only` 경로(`schedule_batch.py:884-889`)로 바뀌며 split-prefill
  하 거동 미확인.
- F3 (권고) 클라이언트 단언: co-load 응답 전부 `completion_tokens == 1` ∧ `finish_reason` = length. 위반이면 셀 무효(fail-closed).
- "co-load req id로 텔레메트리에서 제외"는 **불가**: `PDMUX_PHASE_EVENTS` 세 이벤트 어디에도 rid가 없다(§d).
  제외가 필요 없다는 것(위 a)을 서명으로 확인하는 쪽이 맞다.
- `bs == b` 필터 자체는 바꿀 필요 없음.

## (c) sticky ON/OFF에서 동거 창의 분할 실현 조건 (교훈 88 관점)

선행 사실: 등록 env `PDMUX_R2_POLICY=fixed` ⇒ `self.r2_policy`가 `FixedPolicy`(`multiplexing_mixin.py:265-268`)이고
`_slo_on = True`(1256). 루프의 분할 결정은 두 분기다.

- **R2 분기**(1314-1357): 조건 `running 비어있지 않음 ∧ split_prefill_batch ∧ not wait_prefill_kernel_done`.
  `_r2_decide_idx`(626-682) → `FixedPolicy.decide`(`controller.py:88-102`)는 legacy에서 항상 `safe_boundary=True`
  (true-dual runtime None, 628-631)라 target = D. 인덱스 = `sm_counts`에서 decode_sms == D인 첫 index.
  `_tgt != stream_idx`일 때만 드레인+전환(1341-1356). 긴급 108 경로 없음.
- **elif 분기**(1358-1372): R2 분기가 아닐 때 `adjust_stream_group`이면 **index 변화와 무관하게** 양 stream
  synchronize(1360-1361) 후 `adjust_stream_groups`(1151-1216): sticky ON ∧ decode busy → `_sticky_fixed_idx`(1169-1179);
  sticky OFF ∧ prefill 없음 ∧ decode busy → `real_sm_group_num − 1` = `(0,108)`(1199-1200); decode empty → 0(1201-1211).
- A-agn에서 `sm_counts` = [(108,0), (92,16), (84,24), (74,34), (64,44), (0,108)](`pdmux_context.py:124-127`)라
  `_r2_decide_idx`의 `matches[0]`과 sticky의 `divisions[0]`(`multiplexing_mixin.py:438-448`)은 D마다 같은 index(1–4)다.

**sticky ON, 동거(C_on)**: backlog가 있는 동안 매 span 입장 루프는 R2 분기 → target = 현 index → 드레인 없음,
D 유지. 마지막 span 종료 후(backlog 소진)는 elif → sticky 분기로 같은 index 유지(단 synchronize 2회는 발생).
⇒ D 유지 조건 = decode busy(코드상 무조건). 확신도 높음(독해).

**sticky OFF, 동거(C_off)**: 첫 co span 입장 루프에서 R2 분기가 `(0,108)`→D 전환(드레인 1회), 이후 backlog가
남는 경계마다 R2 분기 → 같은 index → 드레인 없음. 마지막 span 종료 뒤 elif → `(0,108)` 복귀(드레인 1회).
**누수 조건**: decode busy인데 span 경계 다음 루프의 입장이 None을 돌려주면(대기열 순간 공백, `batch_is_full`,
KV 예산 부족 — `scheduler.py:2365-2368`, 2446-2459) 그 루프는 elif로 `(0,108)`에 가고 다음 입장에서 R2 분기가
D로 되돌린다(누수 ≥1 iteration + 드레인 2회). 이 iteration은 `prefill_in_flight=false`로 찍혀 §7-2(b)에서 빠진다.
  - F3과 연결: b ≥ 45면 span 경계 직후 입장이 1요청 또는 cap 판정으로 줄 수 있으나 None은 아니다(첫 루프에서 최소 1).
    b ∈ {8, 32}에서는 cap 기인 누수 없음.
  - `wait_prefill_kernel_done` 루프(최종 span 발사 후 완료 대기)는 R2 분기 조건이 거짓이고 `adjust_stream_group`도
    거짓이라 전환이 없다 — 그 decode는 `prefill_in_flight=true`·`prefill_kernel_done_wait=true`로 찍힌다(2226-2243).
    초안 (b) 술어는 이 iteration을 "동거"로 센다 — 실제로는 prefill 커널이 이미 끝났을 수 있다(기술 열로 분리 권고).

**교훈 88(엔진 강제 상호배타 지원영역)**: fixed 정책에서 실현 가능한 (분할, 상태) 쌍은
sticky OFF = {D∧동거, 108∧단독}, sticky ON = {D∧동거, D∧단독}(+ decode-empty idx0)이다.
OFF의 D∧단독, 양쪽의 108∧동거는 **구조적으로 도달 불가**. 따라서
(i) ρ_off(C_off / A_on)는 필연적으로 **boot 간(sticky 설정 간) 비교**이고 boot 분산을 포함한다;
(ii) W_co 내부의 코드 경로(R2 분기, 같은 index, 드레인 없음)는 ON/OFF가 동일하다 — ON/OFF 차이는 boot 효과와
co 구간 진입·이탈 경계(OFF만 108↔D 드레인)에서만 생긴다. 이는 초안 F-N2-2 예보와 일치하는 독해 근거이나,
"sticky가 동거 step 시간을 바꾸지 않는다"는 **GPU 실증 전 주장 불가**.

## (d) `PDMUX_PHASE_EVENTS`로 (a)를 실측 확인할 수 있는가

필드(2182-2270): `prefill_span_start{span_seq, n_new, extend_tokens, running_bs, queue_len, cap, stream_idx, prefill_sms, decode_sms}`,
`prefill_span_end{span_seq, n_done, running_bs, queue_len, stream_idx}`,
`decode_iteration{iter_seq, t_launch, t_sync, bs, stream_idx, prefill_sms, decode_sms, prefill_in_flight, prefill_kernel_done_wait, queue_len}`.
rid·finished 플래그는 없다. `decode_iteration.bs`는 filter **후** 값(mark는 1405, filter는 1297),
`prefill_span_start.running_bs`·`prefill_span_end.running_bs`는 각각 filter 전·병합 전 값이다.

**확인 가능 — 판별 서명(등록 권고, 라벨 예: `COLOAD_NOT_JOINED` / `COLOAD_JOINED`)**:

- S1: 각 co `prefill_span_end` 직후 첫 `decode_iteration.bs` == 그 span_end의 `running_bs`(병합 전 = b, 자기 종료 없을 때).
  합류했다면 b + n_done.
- S2: W_co 및 A2 앞 128 iteration에서 `bs > b`인 `decode_iteration` 수 == 0.
- S3(양성 대조 — 병합 후 filter를 직접 본다): 경계 바로 다음 루프에서 입장한 co span k ≥ 2의
  `prefill_span_start.running_bs` == b + n_done(k−1). 입장이 한 루프 늦으면 b가 된다 ⇒ 등록은
  `∈ {b, b + n_done(k−1)}`, 기대 다수 = 후자. 이 값이 b + n_done인데 S1·S2가 성립하면 "병합→filter"가 실측된다.
  S3 실패(항상 b)는 독해 오류 신호로 보고(판정 무효 아님).
- 이 세 서명은 rid 없이 (a)를 판별하며, 초안 §7-2(e)의 "`bs ≠ b` 분포 기록"을 판정 가능한 형태로 만든다.

## 초안에 필요한 수정 (rev2 후보 — 사후 조정이 아니라 실행 전 등록)

1. §8-2 (i) "미확인 — 차단": → "코드 독해로 합류 없음(위 (a) 경로), GPU 실증은 S1–S3"으로 갱신. `_dual_worker_prefill_ready`는
   관측 훅일 뿐 병합 경로가 아님(병합은 `multiplexing_mixin.py:1531-1534`).
2. co-load 페이로드 등록: `max_new_tokens = 1` 명시, `0` 금지, 클라이언트 `completion_tokens == 1` 단언(F1–F3).
3. §8-2 (ii) "b ≤ 46에서만 동거 가능" → "첫 co span은 b ≤ 46, 경계 직후 입장하는 이후 span이 2요청이려면 b ≤ 44
   (입장 cap이 filter 전 running_bs = b + n_done를 센다, `scheduler.py:2370`·2445). b = 45–46은 span 구성이 1요청으로
   바뀜". E-α(b ∈ {8,32})는 무영향.
4. §7-2(c) "co span K/2 = 4개": span 구성은 도착 시점(`recv` 1283)에 의존 — 첫 입장 때 8개가 다 도착하지 않았으면
   첫 span이 1요청일 수 있다(추측, 미측정). "Σ n_new over co spans == K ∧ 각 n_new ≤ 2 ∧ span_seq 짝"으로 완화하거나
   W_co 예보를 span 수 가변으로 다시 쓸 것.
5. §6-2 ctx 산술: `runtime_snapshot` 중 `active_decode_sequences ≠` 인접 `decode_iteration.bs`인 표본(병합 직후 filter 전 sync,
   1543·1285)은 ctx 기준점에서 제외. 그렇지 않으면 W_co/A2 정렬에 8193 길이가 섞인다.
6. §7-2(b)에 `prefill_kernel_done_wait=true` iteration 비율을 기술 열로 분리(동거로 세지만 prefill 커널이 이미 끝났을 수 있음).
7. §7 입력 규율(게이트 #260) 보강: `prefill_span_start.running_bs`는 filter 전 값이므로 decode bs로 쓰지 말 것(S3 전용).

## 판정

- 차단 항목 §8-2 (i): **코드 독해 수준에서 해소**(co-load는 decode iteration에 참여하지 않음). 조건: 수정 2(F1) 등록.
- correctness gate 관점: 이 메모는 어떤 성능·정확성 gate도 통과시키지 않는다. 엔진 변경 0, cudagraph 영향 0.
