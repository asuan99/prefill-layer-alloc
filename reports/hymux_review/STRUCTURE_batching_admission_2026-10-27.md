# PD-mux 배치·admission 구조 재검증과 원고–코드 대조 (2026-10-27, main 트랙)

> **지위**: 정본 아님. 정적 코드 읽기와 CPU 재계산만 했다(GPU 0, 측정 0). **새 성능 판정 0건.** Claim 등급·게이트 #6·HE0 불변.
> 작성: engine-porter, 2026-10-27. 지시서 = 사용자 제공 이관 지시(T1–T5, 기준 HEAD `bfc82560`). 이 문서의 기준 HEAD는 `bb1d936d`이며
> `git diff bfc82560 bb1d936d -- workspace/engine-port/src`는 비어 있다. 그래서 지시서의 줄 번호 F1–F11은 그대로 다시 확인했다.
> 라벨: `measured` = 부팅 배너·원자료 값, `derived` = 그 값을 식으로 조합한 값, `cited` = 다른 문서 값을 재계산 없이 옮긴 값.
> **원고 v4 본문은 직접 대조하지 않았다(응답서 기준).** §2의 원고 위치는 Revision Letter(Round 3) 발췌와
> `reports/hymux_review/EXPERIMENT_CROSSWALK_2026-10-26.md`에서 가져왔다.
> 지시서 §2의 채팅 결론 C1–C6은 미감사 상태다. 이 문서는 그 결론들을 코드에 대조했을 뿐 채택하지 않는다(§1.3).

## 1. 재검증

### 1.1 어느 트리를 읽었는가

로컬 엔진 트리 `~/Experiments/KISTI/sglang_engine_dev`는 SGLang git 태그 `v0.5.10`(`1519acf`) 체크아웃이다. 수정 파일 14개와
추적 안 된 추가 파일이 있다. 이 트리를 VESSL에서 실제로 서빙한 트리의 manifest와 대조했다. 대상은 128k 프로브 Job의
`results/cap_char/probe_128k/longctx_probe_20261020T034615Z/meta/runtime_source_manifest.sha256`(install commit `63de3582`,
sync manifest `6015796d…`)이고, 27개 항목의 sha256이 **전부 일치**했다. 미러와도 비교했다. `src/multiplex/`의
chunk_probe·controller·dual_worker·green_readout·holb_probe·multigpu_guard·multiplexing_mixin·profile·telemetry와
`src/models/nemotron_h.py`는 HEAD 미러와 바이트 동일하다.

로컬 트리가 HEAD보다 오래된 지점은 두 군데다. 첫째, `admission_probe.py`와 `decode_probe.py`가 없다. 둘째, `scheduler.py`에 그 두 probe의
훅(`admission_probe_scheduler_hook.patch`, `decode_probe_scheduler_hook.patch`, 커밋 `85283a06`·`ca1f30e7`)이 들어 있지 않다.
두 훅 모두 기본값 OFF이고, env를 켰을 때만 클래스 수준 래퍼를 씌운다. admission 조건 자체는 바꾸지 않는다(패치 본문 확인).
저장소에 git 밖으로 남아 있는 로컬 manifest `workspace/engine-port/results/runtime_source_manifest.sha256`은 트리보다 더 오래된 것이다.
여기서 5건이 FAILED로 나온다(mixin·telemetry·holb·chunk·scheduler). 그래서 이 판정에는 이 파일을 쓰지 않았다.

T1이 읽은 파일은 두 부류로 나뉜다. `managers/schedule_policy.py`, `mem_cache/common.py`, `managers/schedule_batch.py`는
`git diff`가 비어 있으므로 v0.5.10 원본 그대로이고, 원본 기준 판정으로 유효하다. 나머지 셋은 패치된 파일이지만 앞의 manifest와 일치한다.
- `managers/scheduler.py`: 패치 hunk가 `:186–:451`, `:2680` 이후, `:3490`에 있다. admission 함수(`:2309–:2560`)는 건드리지 않는다.
- `model_executor/model_runner_kv_cache_mixin.py`: 패치는 `cell_size == 0` 가드(`:177–:192`) 하나뿐이다.
- `mem_cache/memory_pool.py`: 패치는 `get_v_head_dim` 하나뿐이다.

**결론: T1 대상 구역에서 "로컬 트리가 구판이라 UNVERIFIED"로 남는 판정은 없다.** 단, 이것은 2026-10-20 서빙 트리 기준이다.
HEAD로 다시 sync하면 바뀌는 것은 기본값 OFF probe 훅뿐이라는 점까지만 정적으로 확인했다. tracks/main.md의 "로컬 엔진 트리 재동기화"
결정 대기는 이 작업의 차단 요인이 아니다.

PD-mux 루프의 뼈대(F2·F4·F5·F6)는 이 프로젝트가 만든 것이 아니다. **SGLang v0.5.10 원본 `multiplexing_mixin.py`(221줄)에 이미 있다.**
원본 기준 위치는 좌석 검사 `:82`, `get_new_batch_prefill` 호출 `:86`, `split_forward_token_budget` 사용 `:166`, 매 반복 `decode_stream.synchronize()` `:194`,
`merge_batch` `:215`다. 이 사실은 아래 C2 검토에 쓴다.

### 1.2 F1–F13 판정

줄 번호는 HEAD 미러 `workspace/engine-port/src/multiplex/multiplexing_mixin.py`(로컬 트리와 동일) 기준이고, 엔진 원본 파일은 로컬 트리 경로 기준이다.

| # | 판정 | 근거(재확인 줄)와 단서 |
|---|---|---|
| F1 | CONFIRMED | 스트림 쌍 `:1248–1249`, decode `run_batch` `:1404`, prefill `run_batch` `:1468`. 스트림 그룹은 `pdmux_context.py:initialize_stream_groups`가 부팅 때 전부 만든다: 0 = 일반 스트림 (전 SM, 0), 1…k = green-context 분할, 마지막 = (0, 전 SM). |
| F2 | CONFIRMED | `:1219–1220`. R2 latch 검사 `:1221`이 그 다음에 온다. SGLang 원본 `:82`에도 같은 줄이 있다. |
| F3 | CONFIRMED | `:1227` `get_new_batch_prefill()`. `longctx_probe.sh:106,108,110`, `substrate_probe.sh:381`, `compo_probe.sh:101`의 `--chunked-prefill-size -1`. `-1`이면 `scheduler.py:902–904`에서 `chunked_prefill_size = None`이 되고, PrefillAdder의 chunk 분기는 사라진다. `max_prefill_tokens`(`rem_input_tokens`) 검사는 남는다. 이 검사는 두 번째 요청부터만 막는다. 또 `longctx_probe.sh`의 fused arm(`:110`)도 chunk -1이다. |
| F4 | CONFIRMED | `:1416–1424`. 기본값 65536은 `pdmux_context.py:20`, 설정은 `benchmarks/configs/pdmux_r2.yml:9`. `extend_num_tokens`는 배치 전체의 extend 토큰 합이다. `update_split_prefill_batch(self, sm_count)`는 인자 `sm_count`를 받기만 하고 쓰지 않는다(`:1218–1237`). |
| F5 | CONFIRMED | `:1531–1534`. `update_running_batch`는 배치를 줄이기만 한다(filter·retract, `scheduler.py:2572`, retract `:2592`). |
| F6 | CONFIRMED, 서술 정정 필요 | 매 반복 `decode_stream.synchronize()`는 `:1494`에서 무조건 실행된다. 하지만 **prefill stream은 매 반복 동기화되지 않는다.** 한 반복은 span 하나를 *발행*할 뿐이다. 마지막 span의 완료는 `prefill_exe_done.query()`로 확인만 한다(`:1518`). 따라서 "lockstep"은 host 루프가 decode 박자로 돈다는 뜻이다. decode가 prefill GPU 작업을 기다린다는 뜻은 아니다. 같은 정정이 `results/lockstep_diag/DIAG_lockstep_stall_share_2026-10-27.md` §1에도 있다. 두 스트림을 함께 drain하는 지점은 둘이다: R2/SLO 전환 `:1341–1345`, 그리고 `adjust_stream_group` 경로 `:1358–1362`. 뒤의 것은 입장과 merge 뒤에 켜진다. |
| F7 | CONFIRMED, 범위 단서 | `:1326–1356`. 6개 index는 `pdmux_r2.yml`(`sm_group_num: 6`, 분할 4개 + (108,0) + (0,108))일 때만 맞다. 128k 프로브가 쓴 `pdmux_agnostic_manual.yml`은 4개다. R2 정책이 고를 수 있는 것은 decode SM {16,24,34,44,108}에 해당하는 index뿐이다(`_r2_decide_idx` `:626–682`). index 0은 decode가 비었을 때 `adjust_stream_groups`의 대체 경로로만 간다(`:1199–1211`). |
| F8 | CONFIRMED | 정의 `:388–391`(`PDMUX_STICKY_PARTITION`, 기본값 "0"), 사용 `:1163–1168`. |
| F9 | CONFIRMED | `PROJECT_STATUS.md:27`(결정 F3). `decode_future.result()`가 `:1490`, `prefill_future.result()`가 `:1504`. |
| F10 | CONFIRMED | `span_scheduler`·`HyMux`·`theta_stall`·`D_floor`를 `src/`와 설치 트리에서 찾으면 0건이다. 저장소 전역 대소문자 무시 grep에서 나온 5건(`spread_floor_mid_v1`, `NO_VALID_FLOOR`, `itl_floor`)은 부분 문자열이 우연히 겹친 오탐이다. 가장 가까운 기전은 R2 admission latch다(§2 R6). |
| F11 | CONFIRMED | `src/models/nemotron_h.py:859–891`. 배치 구성은 `tp_worker.py:537–541`에서 고정된다. 여기서 `split_index == 0`일 때 `ForwardBatch`를 한 번만 만들고 이후 재사용한다. `split_index`는 배치 스칼라다(`schedule_batch.py:1395–1397`). dense 원본 `models/llama.py:544–574`의 `forward_split_prefill`도 같은 구조다. hidden_states와 residual을 span 경계 너머로 넘긴다. |
| F12 | CONFIRMED, 예외 1개 | PrefillAdder(`schedule_policy.py:375–844`)에서 mamba 관련 코드는 `is_hybrid_ssm_cache`(`:429`) 하나다. 이 값은 evictable 토큰을 셀 때 쓰는 함수만 바꾼다(`:457–476`). 고갈 동작은 다음과 같다. `common.py:297–325`가 `alloc_req_slots`를 거쳐 `HybridReqToTokenPool.alloc`(`memory_pool.py:530–565`)을 부른다. 여기서 `mamba_pool.alloc(1)`이 None이면 `AssertionError`(`:543–545`)가 나고, req 슬롯이 부족하면 `RuntimeError`(`common.py:318–324`)가 난다. `scheduler.py:2490`의 mamba idx 반환은 radix 매칭 누수 방지용이고 입장 검사가 아니다. **예외**: opt-in `--enable-prefill-delayer`(기본값 False, `server_args.py:363`)를 켜면 `get_new_batch_prefill`(`:2316–2336`)이 mamba 토큰 사용률을 지연 판단에 넣는다. 프로젝트 스크립트에서 이 옵션을 쓰는 곳은 0건이다(grep). R2 스냅샷은 mamba 점유율을 계산하고 버린다(`_mamba_occupancy`, `:484`). |
| F13 | CONFIRMED | 3분기는 로컬 트리 `model_runner_kv_cache_mixin.py:218–243`에 있다. 08-02 handoff가 인용한 `:223–229`는 이 구간 안이다. clamp는 `_resolve_max_num_reqs` `:857–876`이 건다. radix OFF이면 `_calculate_mamba_ratio`가 1이다(`:390–392`). `estimated` 하한이 2048이므로 cap 48에서 clamp에 걸리는 값은 `max_mamba_cache_size`다. 스크립트 근거: `mg_boot.py:96–97`, `engine_gate_driver.py:373`, `longctx_probe.sh:106,108`. 정리하면, 요청 수 상한의 술어는 `scheduler.py:2446`(`get_num_allocatable_reqs`, `:2309–2313`, `pp_max_micro_batch_size := max_running_requests`, `:681–684`)이고, dense 모델에도 똑같이 있는 술어다. |

T1이 요구한 admission 조건 전수는 다음과 같다(코드 순서). PD-mux만의 앞단 관문이 둘 있다. 하나는 좌석(F2)이고, 다른 하나는 R2 latch다.
latch는 `PDMUX_R2_POLICY`가 generic이나 hybrid일 때만 켜질 수 있다. `FixedPolicy`는 `admission_limited`를 세우지 않는다(`controller.py:88–103`).

그 뒤의 SGLang 공통 조건은 다섯 가지다.
1. 대기열이 비었거나 `batch_is_full`이면 반환한다(`scheduler.py:2366`).
2. 이번 배치 후보 수가 `max_running_requests − running_bs` 이상이면 막는다(`:2446`).
3. KV 예약을 검사한다(`schedule_policy.py:764–773`). 새 요청의 `extend + min(max_new_tokens, CLIP 4096)`이 `rem_total_tokens` 이상이면 거절한다. `rem_total_tokens`는 가용 토큰과 evictable 토큰의 합에서 두 가지를 뺀 값이다. 하나는 실행 중 요청 몫인 `남은 max_new × new_token_ratio`이고, 다른 하나는 이번 배치에 이미 넣은 몫인 `extend + max_new + page`다.
4. `max_prefill_tokens`를 검사한다. 두 번째 요청부터 적용된다(`:767–768`, `:791–795`).
5. 선택 조건으로 prefill_max_requests, prefill delayer, CP, LoRA가 있다.

mamba 슬롯 검사는 이 경로 어디에도 없다.

### 1.3 채팅 결론 C1–C6의 코드 대조 (미감사, 채택 아님)

- **C1**: 구조 부분은 F1·F2·F6과 맞는다. 배치는 둘이고, host 스케줄러 루프는 하나다. 다만 F6 정정을 반영해야 한다. GPU 수준 lockstep은 코드에서 필연이 아니다. Semi-PD·chunked-prefill과의 "독립성 높고 낮음" 비교는 이 저장소 코드로 확인할 수 없다. 보고서에는 **올렸지만 수정했다**. 결합 지점에 drain 두 곳(F6 단서)을 더했다.
- **C2**: F11에서 배치 구성 고정을 확인했다. 이 고정은 dense `llama.py`의 같은 구조에도 있다. 그런데 F2·F4·F6은 이 프로젝트의 구현 선택이 아니다. **SGLang 원본 PD-mux의 선택**이다(§1.1). "PD-mux 계열 공통 비용"이라는 일반화는 SGLang 하나만 읽고 확인한 것이다. MuxWise 원 구현과 Bullet은 미확인이다. **수정해서 올렸다.**
- **C3**: 앞 절반은 F11과 llama.py 대조로 구조상 지지된다. dense layer-wise prefill도 같은 고정 비용을 낸다. 뒤 절반의 "hybrid 고유" 서술은 tracks/main.md 금지 서술 목록에 해당하므로 **올리지 않았다**. 코드 사실만 적는다. 슬롯이 admission에 닿는 길은 부팅 때 cap clamp 하나다(F12). 그 술어는 dense에도 있는 `max_running_requests`다. 현재 설정에서는 슬롯 상한과 cap이 같다(F13).
- **C4**: "슬롯은 decode 중에 늘지 않는다"는 코드와 맞는다. 슬롯은 `alloc_for_extend`(`common.py:351`)에서 요청당 1개(`MAMBA_STATE_PER_REQ_NO_CACHE = 1`, `:22`)를 잡고 decode에서는 더 잡지 않는다. 하지만 "결합 조건이 둘"은 admission 술어로 보면 성립하지 않는다. 술어는 KV 예약과 cap이고, 이는 dense와 같다. 슬롯은 별도 술어가 아니며, 고갈되면 프로세스가 죽는다. **수정해서 올렸다**(이 단락).
- **C5**: 메모리 쪽 방향은 이 부팅 설정(ctx 131072, mem 0.6, cap 48)에서 derived로 성립한다(§3). arXiv-Sum 평균 L+O 7,319 [derived: 입력 평균 7,116 + 출력 평균 203, measured `ARXSUM_STATS_2026-10-15.txt`]에서 c_mem 125, KV 전용 상한 216이므로 cap 48이 먼저 구속한다. arXiv-Sum 표준 셀의 부팅 풀은 이 프로브와 다르므로 그대로 옮길 수 없다. "실제 포화는 prefill 연산 + F2"는 검증되지 않은 가설이다. Q1로 옮겼다. **부분만 올렸다.**
- **C6**: **수치가 틀렸다.** 채팅의 8.8k는 s/k다. 슬롯 하나가 Nano 자신의 KV 몇 토큰과 같은지를 뜻하며, `analytic_terms.L_star`와 같은 양이다. hybrid와 Transformer의 손익분기가 아니다. 같은 Job의 Llama-3.1-8B 부팅 값으로 손익분기 s/(k_T − k)를 다시 내면 **1,251 토큰 [derived]**이다. 이는 원고 M3의 ≈1.25k와 맞는다. **정정해서 올렸다**(§3 D4). 반대쪽 끝을 G0로 검증하자는 제안은 Q3로 옮겼다.

## 2. 원고(응답서 기준)–코드 1:1 대조표

원고 위치는 Revision Letter(R3) 발췌의 문단 번호(.NNNN)와 응답서가 가리키는 절이다. 원고 v4 본문 대조는 하지 않았다.
판정은 넷 중 하나다: **구현됨**(코드가 서술과 일치), **부분**, **미구현**, **해당 없음**(코드 서술이 아님).

| # | 원고 서술 (요지) | 원고 위치 | 코드 대응 | 판정 |
|---|---|---|---|---|
| R1 | SGLang PD-mux는 span 크기를 token×층 예산 65,536으로 정한다 | M2 .3076, §4·§5.3·§7.1·§7.2(M2 개정 .4139) | `:1416–1424`, `pdmux_context.py:20`. 단서: extend 토큰이 65,536을 넘으면 `max(1, …)` 때문에 span이 1층이 되어 예산을 넘는다 | 구현됨(baseline 서술) |
| R2 | 기존 방식은 prefill SM 수를 보지 않는다 | M2 .3391 | `sm_count` 인자를 쓰지 않음(`:1218–1237`), 예산은 상수 | 구현됨(baseline 서술) |
| R3 | 기존 방식은 모든 층의 비용이 같다고 본다 | M2 .3566 | 층 수만 센다. 층 종류를 보는 `PDMUX_SLO_SPAN_TYPE`(`:1430–1439`)는 실패한 실험의 잔재이고 살아 있는 기능이 아니다 | 구현됨(baseline 서술) |
| R4 | **HyMux span scheduler가 층 종류별 비용과 현재 prefill SM 수로 계산한 시간 예산 θ_stall로 span을 자른다** | M2 .3805, §4 span scheduler(M2 개정 .4139) | 없음. span 길이는 상수 예산 ÷ extend 토큰이다(F4). 시간·SM·층 종류 입력이 모두 없다 | **미구현** |
| R5 | **split_forward_count가 span scheduler를 호출한다** | 지시서 §3이 인용한 HyMux 설계 절. 응답서 대응은 §4(M2 개정 .4139) | `:1443`은 인라인으로 계산한 값을 대입할 뿐, 호출이 없다 | **미구현** |
| R6 | **admission에 floor 실현 가능성 검사가 붙는다** | 지시서 §3. 대조표 X7·D-11 "HyMux 구현 후" | 없음. 가장 가까운 것은 R2 latch(`controller.py:311–328`, `mixin :1221`, `:2045–2071`)다. 이 latch는 (i) 2 epoch 연속 과부하(D108 위험 ∧ 상한 > SLO, 또는 KV·배치 점유 ≥ 0.90) 뒤에 사후로 걸리고, (ii) 후보 요청이 아니라 현재 실행 상태를 보고, (iii) 걸리면 실행 배치가 빌 때까지 모든 입장을 막으며, (iv) generic·hybrid 정책에서만 켜진다 | **미구현**(다른 기전의 latch만 있음) |
| R7 | **partitioner가 실제로 계산하는 것도 P = N_SM − D_floor**, D_floor(b,c) ≈ (W + 2bs + kbc)/(ω(1−m)θ) | M1-4 .8298·.8661, §3.2·Figure 2(c)(M1-4 개정 .9164) | hybrid 정책만 D_floor 비슷한 값을 낸다. 방식은 측정 프로파일 조회다(`profile.py:303–360`): {16,24,34,44} 가운데 보간한 p95 ITL 상한이 SLO − margin 이하인 최소 D를 고르고, 없으면 108. 닫힌 식은 코드에 없다. P는 계산하지 않고 `manual_divisions` 행에서 읽는다. `pdmux_r2.yml`의 행 합이 108이라 P = 108 − D가 *설정상* 성립할 뿐이다. generic(`controller.py:410–457`)에는 D_floor가 없고, fixed는 상수, R2 없는 기본 PD-mux는 decode bs 문턱 선택기(`mixin :1179–1196`)다 | **부분**(P = N − D는 표 항등식, D_floor 식은 미구현) |
| R8 | SLO 위반에 반응하는 보정은 위반이 난 뒤에야 경계를 따라간다. 사후 보정은 오차를 메우지 못한다 | M1-1 개정 .1294/.1388, Abstract·§3.2 "반응형 보정의 한계" | 반응형 코드는 세 곳이다: `GenericDynamicPolicy`의 slack<0 상향(`controller.py:428–435`), 공용 `_live_underprediction` 즉시 상향(`:151–166`, `:281–292`), 구형 `PDMUX_SLO_SCHED` `_slo_decide_idx`(`mixin :1151–1162`, `:1325–1330`). **hybrid 정책도 같은 공용 즉시 상향을 공유하므로 순수 예측형이 아니다.** 이 행은 코드 대응까지만 적는다. 성능 서술은 HE0 정본 범위를 따른다 | 대응 코드 있음(비교 대상) |
| R9 | 짧은 요청 구간에서는 토큰이 아니라 요청 수가 먼저 상한에 닿는다. 그래서 admission이 필요하다 | M3 .4742, §1 "동시성"·§3.3(M3 개정 .5004/.5082) | 요청 수 상한은 `max_running_requests` 술어(`scheduler.py:2446`)이고 dense와 같다. 슬롯은 부팅 clamp로만 닿고(F12), 프로젝트 설정에서는 cap과 같다(F13). 요청당 state 하한을 보는 admission 로직은 없다. 슬롯이 고갈되면 프로세스가 죽는다. 측정한 부팅(cap 48)에서는 L+O ≲ 33k 구간에서 cap이 KV보다 먼저 구속한다(§3 D3, derived). 다만 그 cap은 손으로 정한 값이다 | **부분**(현상 서술은 현 설정과 양립, 서술이 함의하는 admission은 미구현, 이 엔진에서는 슬롯과 cap을 분리 측정할 수 없음) |
| R10 | 요청당 메모리가 같아지는 길이 s/(k_T − k) ≈ 1.25k | M3 .4742 | 코드 서술이 아님. 부팅 값으로 1,251, analytic으로 1,252(Llama 형태). a31 조성이면 1,298 [전부 derived, §3] | 해당 없음(수식 재현됨) |
| R11 | Figure 5(c): HyMux는 prefill 중에도 60 ms 안에 토큰을 낸다 | M5 .6705/.6983(응답서가 "측정 아닌 개념도"라고 명시) | HyMux 런타임 없음 | **미구현**(개념도로 공시됨) |

미구현 서술은 R4·R5·R6·R11, 부분은 R7·R9다. 이들은 지시서 §3의 선택지 (a)/(b) 대상이다(§5).

## 3. derived 재계산

스크립트: `reports/hymux_review/structure_derived_2026-10-27.py`(`python3 -I …`, `--json` 지원). 입력은 nano `boot_facts.json`
(sha256 `fb7bc298…`, Job `SHA256SUMS`와 일치)과 같은 Job의 llama `boot_facts.json`(`2ed597d7…`)이다. analytic 대조는 `analytic_terms.py`(analytic-v1)로 했다.

입력의 성격부터 정정한다. `pool_bytes`·`slot_bytes`·`kv_bytes_per_token`은 프로브가 배너의 소수 둘째 자리 GiB 값으로 계산한 값이다.
그리고 **`c_mem` 열은 식의 출력이다. 실현 동시성을 측정한 값이 아니다.** 지시서 §1은 이 열을 measured로 표시했지만 derived가 맞다.
입력의 측정 근거는 배너다: KV 12.11+12.11 GiB, conv 0.09 GiB, ssm 6.46 GiB, kv_cache_tokens 1,587,762 [measured].
같은 식으로 다시 계산하면 c_mem 다섯 점(16k→80, 32k→48, 64k→27, 96k→18, 124k→14)이 저장값과 모두 일치한다 [derived].

**D1. 슬롯 1개 ≈ Nano KV 몇 토큰.** s/k = 8,763.1 토큰 [derived]. 채팅 값 8.8k는 맞다. analytic L\* = 8,761.5이고,
부팅 s와 analytic s의 차는 −0.012%, k의 차는 −0.030%다 [derived].

**D2. 문맥 0의 메모리 동시성 천장.** pool/slot = 230.19, floor를 취하면 230 [derived]. 채팅 값은 산술로는 맞다. 다만 이 값은 KV 풀 전체를
슬롯으로 바꿀 수 있다고 가정한 상한이다. SGLang은 부팅 때 두 풀을 고정 분할하므로 이 부팅에서 실제로 만든 슬롯은 49개(48+1)다.
따라서 230은 "이 부팅의 천장"이 아니다. 풀을 다시 나누면 도달할 수 있는 가상의 상한이다.

**D3. cap 48일 때 슬롯 구속과 메모리 구속의 경계.** 풀을 자유롭게 나눈다고 보면 c_mem(L) ≥ 48인 최대 L이 33,260 토큰이다 [derived].
이 부팅의 실제 분할에서 KV 풀만 보면 kv_cache_tokens/48 = 33,078 토큰이다 [derived]. 두 정의 모두 약 33k라 채팅 값은 맞다.
두 값이 가까운 이유는 풀이 cap 48 기준으로 잡혔기 때문이다. 실제 분할에서의 동시성 상한은 min(48, KV 전용 상한)이다.
16k→48, 32k→48, 64k→24, 96k→16, 124k→12 [derived]. c_mem 열보다 이쪽이 이 부팅에 맞는 양이다.

**D4. C6 손익분기.** 채팅 값 8.8k는 D1과 같은 s/k다. hybrid와 Transformer의 손익분기가 아니다(**채팅 값 오류**).
s/(k_T − k)를 같은 Job의 Llama-3.1-8B 부팅 k_T(131,095 B/토큰 [measured 배너 유도])로 계산하면 1,251.2 토큰이다 [derived].
analytic으로 Llama 형태를 넣으면 1,251.6, a31 조성(n_A 31)을 넣으면 1,298.0이다 [derived]. 원고 M3의 ≈1.25k는 Llama-3.1-8B 기준과 맞는다.
Llama-3.1-8B는 Nano와 짝을 맞춘 Transformer가 아니므로 이 값은 기준 모델을 하나 고른 결과다.

**가정 공시.** 위 계산은 모든 요청이 L+O 토큰의 KV를 동시에 최대로 점유한다는 보수적 가정을 쓴다. 실제 admission의 KV 예약은 이와 다르다.
새 요청에는 `extend + min(max_new, 4096)`을, 실행 중 요청에는 `남은 max_new × new_token_ratio`를 잡는다(§1.2). C5 점검값(arXiv-Sum 최대 L+O 16,744
[cited `dp_routing/PLAN_DP_routing_per_gpu_partition_2026-10-26.md:37`])에서는 c_mem 79, KV 전용 상한 94, 실제 분할 상한 48이다 [derived].

## 4. 열린 질문 (질문형, 답하지 않음)

- **Q1.** arXiv-Sum 부하에서 TTFT 꼬리 가운데 F2(좌석 하나) 직렬화가 차지하는 몫은 얼마인가? SM 분할 효과와 이것을 가르는 최소 arm 쌍은 무엇인가?
  코드 사실은 다음과 같다. `PDMUX_PHASE_EVENTS`(`mixin :2129–2302`)의 `prefill_span_start`는 배치 입장 시각(host)과 `queue_len`·`n_new`·`running_bs`·`cap`을 기록하고, `prefill_span_end`는 완료 관측 시각을 기록한다. 요청 ID와 요청별 대기열 진입 시각은 기록하지 않는다. 그래서 **요청별 대기 시간은 이 이벤트로 직접 잡히지 않는다.** 잡히는 것은 좌석이 점유된 구간(start→end)과 그때의 대기열 길이다. `PDMUX_ADMISSION_PATH` probe(HEAD 미러에만 있음)는 `prefill_seat` 거절을 창 단위 *횟수*로 센다(`admission_probe.py:717–729`). 시간은 세지 않는다.
- **Q2.** cap과 슬롯을 분리한 설정(`--max-mamba-cache-size` < `max_running_requests`, 또는 메모리 기반 분기 ③)에서 서버가 부팅되고 서빙되는가? 그때 F12의 `AssertionError`·`RuntimeError` 경로에 빠지지 않는가? 정적으로는 `_resolve_max_num_reqs`가 cap을 슬롯 수로 깎는 것까지만 보인다. 실행 correctness는 `experiment_dag_and_execution_2026-10-04.md:157`의 미확인 상태 그대로다.
- **Q3.** C6의 손익분기(Llama-3.1-8B 기준 derived 1,251, a31 기준 1,298)는 G0의 순수 attention 대조에서 실제 동시성·goodput 쪽으로 어느 방향으로 움직이는가? 메모리 손익분기와 시간(decode 대역폭) 손익분기는 같은 길이에서 나타나는가?
- **Q4.** `mamba_full_memory_ratio` 축(`session_handoff_2026-08-02.md` 열린 항목 5, `:197–202`)은 V-O(`cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md`)와 DP 라우팅(`dp_routing/PLAN_DP_routing_per_gpu_partition_2026-10-26.md`의 τ = s/k)의 어디에서 만나는가? 분기 ③으로 부팅하면 τ의 의미가 바뀌는가?

## 5. 사용자 결정 대기

- **원고 HyMux 설계 절의 처리**(지시서 §3, blocking). 대상은 미구현 R4·R5·R6·R11과 부분 R7·R9다.
  - (a) 원고를 실제 구조로 낮춘다: 고정 token×층 예산 span, 분할 index 컨트롤러(fixed / generic / hybrid-profile 조회), R2 admission latch.
  - (b) 구현하고 측정한 뒤 원래 서술을 유지한다(대조표 D-11 경로, C-4 통과 후).
  - 둘 중 하나를 사용자가 고른다. 이 문서는 원고를 고치지 않았다.
- 참고(차단 아님): 로컬 엔진 트리는 2026-10-20 서빙 트리와 바이트 동일하다. HEAD와의 차이는 기본값 OFF probe 2개(파일과 훅)뿐이다. tracks/main.md의 재동기화 결정 대기와 연결되지만, T1–T3에는 영향이 없었다.
