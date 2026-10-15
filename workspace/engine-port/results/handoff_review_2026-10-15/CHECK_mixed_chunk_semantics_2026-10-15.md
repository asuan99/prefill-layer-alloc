# CHECK — SGLang mixed-chunk 의미 대조 + hybrid 경로 + GPU gate 설계 (2026-10-15)

- 대상: SPEC_common_measurement_v0 §6 R8-2b 선행 확인 (ii) 코드 대조, (i) GPU 정확성 gate 설계(실행 안 함).
- 작성: engine-porter, 코드 읽기만(CPU). 엔진 수정·GPU·커밋 없음.
- 트리: `~/Experiments/KISTI/sglang_engine_dev/python/sglang/srt`(이하 `srt/`). multigpu 트리
  (`sglang_engine_dev_multigpu/python/sglang/srt`)의 관련 9개 파일(scheduler, schedule_policy, schedule_batch,
  mamba.py, mamba2_metadata, hybrid_linear_attn_backend, nemotron_h, server_args, model_runner)은 `diff -q` 결과 바이트 동일이다 [코드].
- 표기: [코드] = 파일:행에서 직접 확인. [해석] = 추론. VESSL 설치 트리가 이 로컬 트리와 같은지는 이 문서에서 확인하지 않았다.

## 1. SGLang mixed-chunk 경로 (사실)

1. 켜짐 조건: `is_mixed_chunk = chunked_prefill_size is not None and enable_mixed_chunk` (`scheduler.py:906-909`) [코드].
   인자 단계에서 끄는 곳은 dual-chunk attn(`server_args.py:2475`), eagle spec(`:3032`), ngram spec(`:3116`), dLLM(`:3600`)뿐이다. hybrid·mamba·radix 때문에 끄는 분기는 없다 [코드].
2. 예산: `PrefillAdder(..., max_prefill_tokens, chunked_prefill_size, running_bs if is_mixed_chunk else 0, ...)` (`scheduler.py:2402-2416`).
   생성자에서 `rem_input_tokens = max_prefill_tokens - running_bs`, `rem_chunk_tokens -= running_bs` (`schedule_policy.py:398, 405-406`) [코드].
   그래서 `chunked_prefill_size`는 **decode 토큰(요청당 1)을 포함한 반복당 총 토큰 예산**이다. prefill이 받는 몫은 C − running_bs이다 [코드].
3. 순서: 진행 중인 chunked_req를 먼저 넣고(`scheduler.py:2420-2422`, `add_chunked_req` `schedule_policy.py:599-627`), 그다음 waiting queue를 FCFS/policy 순으로 채운다 [코드].
   잘리는(truncate) 요청은 최대 1개다. 잘리면 `rem_chunk_tokens`가 0 이하가 되어 `budget_state()`가 OTHER를 돌려주고 루프가 끝난다(`schedule_policy.py:816-838, 502-516`) [코드].
   그 앞에는 짧은 prefill 여러 개가 통째로 들어갈 수 있다(`:803-815`; 비절단 분기) [코드].
4. 혼합: prefill 배치를 만든 뒤 아래 조건을 **모두** 만족할 때만 running batch 전체를 `mix_with_running`으로 붙인다(`scheduler.py:2550-2568`) [코드].
   - `is_mixed_chunk`
   - running batch가 비어 있지 않음
   - `not (new_batch.return_logprob or running_batch.return_logprob)`
   - `new_batch.input_embeds is None`
   `mix_with_running`은 `ForwardMode.MIXED`를 붙이고, decode 요청마다 `extend_len=1`과 `prefix_len=len(input)+len(output)+delta`를 준다. overlap이면 delta=0, 아니면 −1이다(`schedule_batch.py:1873-1904`) [코드].
   혼합 뒤에는 `running_batch`를 빈 배치로 바꾸고(`scheduler.py:2564`), 다음 반복에서 last_batch(extend)를 filter·merge해 되돌린다(`scheduler.py:2235-2262`) [코드].
5. 섞지 않는 경우 [코드]
   - (a) 새 prefill도 chunked_req도 없음(`scheduler.py:2365-2368`) → 순수 DECODE 반복(`:2284-2296`).
   - (b) running batch가 비어 있음 → 순수 EXTEND.
   - (c) 진행 중인 요청 중 **하나라도** logprob을 요청함 → 혼합이 꺼진다. 이때는 prefill-only 반복이 decode를 멈추는 비혼합 동작으로 돌아간다(`:2285` "Run prefill first").
   - (d) input_embeds 사용.
6. 혼합 반복에서는 `update_running_batch`(decode KV 부족 검사·retract, `scheduler.py:2570-2590`)를 건너뛰고 `prepare_for_decode`만 호출한다(`:2561`) [코드].
   decode 증가분은 PrefillAdder의 `rem_total_token_offset`(남은 max_new_tokens × new_token_ratio)로만 예약된다(`schedule_policy.py:407, 418-424`) [코드].
7. cudagraph
   - decode cudagraph는 `is_cuda_graph()` = DECODE/TARGET_VERIFY/IDLE/DLLM_EXTEND에서만 탄다(`forward_batch_info.py:167-173`, `model_runner.py:2817-2833`) [코드]. MIXED는 `forward_extend`로 간다(`model_runner.py:2872`).
   - piecewise cudagraph는 MIXED→EXTEND로 정규화해서 탈 수 있다(`piecewise_cuda_graph_runner.py:685-694`). NemotronH는 piecewise 비활성 목록에 없다(`model_config.py:1352-1358`) [코드].
   - 명세 fused arm은 `--disable-piecewise-cuda-graph`를 쓴다(SPEC §6 l.119). 그래서 **혼합 반복은 전부 eager**이고, decode가 혼합된 반복의 decode 토큰도 eager로 돈다 [코드+해석].
8. overlap: 혼합 경로는 overlap을 지원한다(delta 처리 `schedule_batch.py:1888-1889`). 연속 extend의 overlap 해제는 env `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 기본 False일 때만 끈다(`scheduler.py:1424-1428`, `environ.py:253`) [코드].
   [해석] 혼합 반복이 연속으로 이어져도 기본 설정에서는 overlap이 유지된다.

## 2. Sarathi-Serve stall-free batching과 대조

Sarathi 쪽 서술은 논문(Agrawal et al., OSDI'24, Alg.1) 기억에 기반한다. 원문 재확인 전 [해석]으로 둔다.

| 항목 | Sarathi-Serve | SGLang mixed-chunk (이 트리) | 같음? |
|---|---|---|---|
| 예산 정의 | 반복당 token budget τ, decode 토큰 포함 | `chunked_prefill_size`가 decode 토큰(running_bs)을 포함(§1-2) | 같음 |
| decode 우선 | running decode 전부를 먼저 넣고 남은 예산을 prefill에 줌 | running_bs를 먼저 빼고 decode 전원을 붙임(§1-2, §1-4) | 같음 |
| 진행 중 prefill 우선 | partial prefill을 새 요청보다 먼저 이어 붙임 | chunked_req 먼저(§1-3) | 같음 |
| 반복당 chunk 수 | 마지막 요청만 부분 chunk, 앞 요청은 통째로 | 같음. 잘리는 요청 ≤ 1(§1-3) | 같음 |
| 혼합을 안 하는 조건 | 없음(항상 stall-free) | logprob 요청이 하나라도 있거나 input_embeds면 혼합 해제 → decode stall(§1-5c,d) | **다름** |
| 2차 상한 | 없음 | `max_prefill_tokens`(기본 16384, `server_args.py:349`)에서 running_bs를 뺌. C ≤ 8192이면 묶이지 않음 | 실질 같음 [해석] |
| decode 메모리 검사 | 스케줄러가 preempt | 혼합 반복은 retract 검사를 건너뜀(§1-6) | **다름** |
| 예산 크기 | TBT SLO에서 정함(소수의 작은 값) [해석] | 명세 격자 {2048, 4096, 8192} | **운영 영역 다름** |
| 실행 경로 | 구현 의존(vLLM V0는 혼합 eager) | piecewise off → 혼합 반복 eager, 순수 decode만 cudagraph(§1-7) | 구현 차이 |
| hybrid decode 커널 | 해당 없음(Transformer) | 혼합 반복의 decode 토큰이 mamba **prefill(chunk-scan) 커널**을 탐(§3-2) | **다름** |
| overlap | 없음 | 지원(§1-8) | 구현 차이 |

**판정**
- 토큰 예산 의미(decode 포함, decode 우선, partial 먼저, 잘리는 요청 1개)는 Sarathi와 같다 [코드+해석].
- 하지만 다음 셋 때문에 "Sarathi식"을 **arm 이름이나 결과 문장의 주어로 쓰면 안 된다**.
  - (1) 조건부 혼합(logprob → stall 복귀)
  - (2) 예산 격자가 Sarathi의 SLO-유도 영역과 다름
  - (3) hybrid에서 decode 토큰이 다른 커널을 탐
- 허용 문안 [제안]: "SGLang mixed-chunk(`--enable-mixed-chunk`; decode 토큰을 포함한 반복당 토큰 예산에서 decode를 먼저 채우는, Sarathi-Serve와 같은 예산 의미의 SGLang 구현)".
- 비교 문장은 계속 "SGLang mixed-chunk 대비"로 쓴다. "Sarathi-Serve 대비", "stall-free batching 일반 대비"는 금지한다.

## 3. hybrid(mamba, Nano-9B-v2 = NemotronH) 상호작용

1. 백엔드: Mamba2AttnBackend는 extend와 decode 모두 `Mamba2Metadata.prepare_mixed`를 쓰고, mixer가 직접 prefill/decode를 나눈다(`hybrid_linear_attn_backend.py:641-647, 710-718`) [코드].
   MIXED는 `is_extend()`에 포함되므로(`forward_batch_info.py:111-120`) extend 분기를 탄다(`hybrid_linear_attn_backend.py:173-213`). assert·NotImplemented·fallback 분기는 **없다** [코드].
2. **decode 토큰이 "prefill"로 분류된다**: `num_prefills = len(extend_seq_lens)`, `num_decodes = len(seq_lens) - num_prefills` (`mamba2_metadata.py:194-196`) [코드].
   `mix_with_running`이 decode 요청에도 `extend_lens`를 1씩 덧붙이므로(`schedule_batch.py:1898`) MIXED에서 num_decodes = 0이다 [코드].
   결과적으로 decode 토큰은 길이 1짜리 varlen 시퀀스가 되어 아래 경로를 탄다 [코드].
   - `causal_conv1d_fn(has_initial_state=prefix>0, cache_indices=mamba slot)` (`mamba.py:500-519`)
   - `initial_states = ssm_state[state_indices]`를 쓰는 `mamba_chunk_scan_combined` (`mamba.py:522-552`)
   순수 DECODE 반복에서는 `selective_state_update` 계열 decode 경로(`mamba.py:565-690`)를 쓴다.
   [해석] 수학적으로는 같은 재귀(1-토큰 scan + 초기 상태)다. 하지만 커널과 누적 순서가 달라 **비트 동일은 기대할 수 없다** → greedy 토큰 완전 일치를 gate로 쓰면 오탐이 생길 수 있다(§4).
   [해석] 성능 쪽: 반복마다 decode 요청 수만큼 1-토큰 시퀀스가 chunk-scan에 들어간다. 그래서 `prep_initial_states` `.item()` 동기화(`mamba2_metadata.py:201`)와 chunk index 계산 비용이 혼합 반복마다 붙는다. 이것은 비혼합 fused와 다른 비용 구조이고, 결과 해석에서 "mixed가 느린 이유" 후보다(측정 전 주장 금지).
3. state 인덱싱: `mamba_cache_indices = req_to_token_pool.get_mamba_indices(req_pool_indices)` (`hybrid_linear_attn_backend.py:165-167`) [코드].
   `merge_batch`가 `req_pool_indices`를 [prefill | decode] 순서로 이어 붙이고(`schedule_batch.py:2281-2283`), 토큰 순서도 같은 순서(`schedule_batch.py:1880-1881`)이므로 slot과 토큰이 정렬된다 [코드].
   decode 요청은 기존 `mamba_pool_idx`를 그대로 쓴다. 혼합 때문에 MambaPool slot을 새로 할당·해제하지 않는다 [해석, 할당 경로가 prepare_for_extend의 신규 요청에만 있음].
4. `merge_batch`는 `mamba_track_indices/mask/seqlens`를 None으로 지운다(`schedule_batch.py:2291-2293`) [코드]. 이 필드는 `enable_mamba_extra_buffer()`(= `mamba_scheduler_strategy == extra_buffer`, radix 필요)일 때만 쓰인다(`schedule_batch.py:1762`, `server_args.py:6066`).
   [해석] 명세는 radix off이고, 기본 `auto→no_buffer`(`server_args.py:964-967`)라서 영향이 없다. **radix ON + extra_buffer + mixed 조합은 혼합 반복에서 prefix-state 추적이 빠지는 위험**이 있다 → 이 조합은 쓰지 않는다.
5. `--max-mamba-cache-size` / radix off / `max-running-requests 48`:
   radix off이고 mrr가 명시되면 `max_mamba_cache_size = max_running_requests`(48)이다(`model_runner_kv_cache_mixin.py:218-228`) [코드].
   혼합은 동시 요청 수를 바꾸지 않으므로 slot 수요가 늘지 않는다 [해석].
   `--max-mamba-cache-size`를 명시하면 그 값이 우선한다(`:216-220`). 이 값을 48보다 작게 주면 admission이 slot에서 막힌다. 이것은 mixed와 무관하고 arm 공통이다 [해석].
   radix off + no_buffer에서는 overlap이 유지된다(`server_args.py:2200-2215`는 radix ON일 때만 overlap을 끈다) [코드].
   C ≥ 2048 > 48이므로 `rem_chunk_tokens`가 음수가 되는 경로(`add_chunked_req`의 ≤0 대체, `schedule_policy.py:603-607`)는 도달하지 않는다 [코드+해석].
6. 모델 코드: `nemotron_h.py:645`의 dev 편집(`PDMUX_FIXED_DECODE_SM*`)은 `is_decode()`에서만 발화한다. MIXED에서는 발화하지 않는다 [코드]. 해당 env는 fused_mixed arm에서 설정하지 않아야 한다.
7. upstream 테스트: `test/` 안에서 `--enable-mixed-chunk`를 쓰는 테스트는 scheduler/abort 계열 4개뿐이고, mamba/hybrid 모델과 결합한 테스트는 0개다 [코드, grep]. → hybrid+mixed는 **upstream에서도 미검증 조합**이다 [해석].
8. **PD-mux arm 오염 위험**: PD-mux loop도 `get_new_batch_prefill()`을 호출한다(`multiplex/multiplexing_mixin.py:1227`). PD-mux arm에서 mixed가 켜지면 decode 배치가 prefill 배치에 흡수되어 PD 분리가 깨진다 [코드+해석].
   현재 `enable_pdmux`와 `enable_mixed_chunk`를 상호 배제하는 검사가 없다(`server_args.py:6125` 주변 grep에 mixed 없음) [코드].

## 4. (i) GPU 정확성 gate 설계 (G0 계측 패치 gate Job에 동승, 실행 안 함)

전제: 명세 fused 플래그(overlap ON, radix off, `--disable-piecewise-cuda-graph`, mrr 48), Nano-9B-v2, 단일 A100. 모든 요청은 **logprob 없이** 보낸다(§1-5c: logprob이 있으면 혼합이 꺼져 gate가 공회전한다).

- G-M1 혼합 실현 확인(공회전 방지)
  - 기존 계측: `PDMUX_HOLB_PATH`를 설정하면 `holb_probe`가 forward마다 `ForwardMode.name`을 `mode_counts`에 센다(`holb_probe.py:114, 147, 280`). overlap/비overlap 양쪽 run_batch에 걸려 있다(`scheduler.py:2709, 2748, 2763`) [코드]. **새 카운터는 필요 없다.**
  - 판정: mixed-on boot에서 `mode_counts["MIXED"] > 0`이고 mixed-off boot에서 `== 0`이어야 한다. 하나라도 어기면 FAIL(배선 오류).
  - 보조: chunk_probe channel 2는 mixed일 때 decode 토큰을 포함한다(`chunk_probe.py:1016-1021` 경고). 그래서 혼합 판정에 쓰지 않는다.
- G-M2 출력 동치(완화형)
  - 시나리오: 긴 decode 요청 8개(max_new_tokens 256, temperature 0)를 먼저 보낸다. decode가 진행 중일 때 C를 넘는 긴 prompt 4개(예: 6k–12k 토큰, 다중 chunk)를 주입해 혼합을 강제한다.
  - 같은 요청 집합과 같은 순서로 4개 boot를 돌린다: off-A, off-B(같은 설정 재실행 또는 주입 시점을 바꾼 run), on@2048, on@8192.
  - 지표: 요청별 first-divergence 위치. 노이즈 바닥 D0 = off-A vs off-B, 처치 D1 = off-A vs on@C.
  - [제안] PASS 조건
    - (a) D1의 완전일치 비율 ≥ D0 − 1 요청
    - (b) 첫 32 토큰 일치가 100%
    - (c) 크래시·assert·NaN 0
  - [해석] §3-2의 커널 차이 때문에 엄격한 비트 동일은 gate로 부적절하다. 대신 "혼합이 noise 바닥보다 더 갈라놓지 않는다"로 정의한다.
- G-M3 의미 sanity(선택): GSM8K 50문항 greedy 정답률이 off vs on@2048에서 ±2문항 안이어야 한다. mamba state 오인덱싱처럼 큰 오류를 잡는 보완책이다.
- G-M4 PD-mux 배제 확인: PD-mux arm boot 로그와 holb `arm_enable_mixed_chunk`(`holb_probe.py:575`)가 False여야 한다.
- 예상 소요 [해석]: boot 4회 × 약 4분(cudagraph 캡처 포함) + G-M2 각 약 1–2분 + G-M3 약 3분 ≈ **25–35분**. G0 gate Job에 같은 GPU로 직렬 동승한다.

## 5. 명세 R8-2b에 고칠 점 [제안]

1. 인용 보강: `scheduler.py:906` 외에 다음을 함께 적는다.
   - 예산 차감 `scheduler.py:2411` + `schedule_policy.py:398, 405-408`
   - 혼합 조건 `scheduler.py:2550-2568`
   - `schedule_batch.py:1873`
   "`2562`"만으로는 조건(logprob/embeds)이 드러나지 않는다.
2. 호칭: 위 판정대로 "SGLang mixed-chunk"를 주어로 쓰고, "Sarathi와 같은 예산 의미"는 괄호 설명으로만 둔다. R8-3 G_mixed의 금지 문장("vLLM·Sarathi 대비")은 (ii) 이후에도 **유지**한다.
3. 플래그 공시 R8-4에 두 가지를 더한다.
   - "요청에 logprob 없음"(혼합 유효 조건)
   - 실현 지표 `MIXED forward 비율`(holb `mode_counts`). 혼합이 실제로 거의 안 일어난 셀은 G_mixed 근거에서 뺀다.
4. 격자 해석: {2048, 4096, 8192}에서 prefill 몫은 C − running_bs(≤48)이다.
   Sarathi식 엄격-TBT 영역(작은 예산)은 이 격자에 없으므로 "chunked prefill 튜닝 공간"을 이 3점으로 한정해 서술한다.
   작은 예산(예: 512/1024)을 넣을지는 사용자 결정이다. 그 경우 ITL 50 ms SLO와의 관계를 먼저 본다.
5. hybrid 단서: "혼합 반복에서 decode 토큰은 mamba chunk-scan(prefill) 커널로 처리되고, piecewise off라 eager로 돈다"를 G_mixed 결과 꼬리표로 의무 공시한다. 원인 귀속(SM 분할 대 커널 경로)에 쓰지 않는다.
6. 구성 가드: PD-mux arm에서 `enable_mixed_chunk=False`를 launch 단계 assert로 고정한다(§3-8). radix ON·extra_buffer와 mixed는 결합하지 않는다(§3-4).
7. (i)의 판정 기준을 "greedy 토큰 완전 일치"에서 §4 G-M2 완화형(노이즈 바닥 대비)으로 바꾼다. 비트 동일을 요구하면 정상 동작도 FAIL하게 된다.

## 6. 단언 범위

이 문서가 단언하는 것은 코드상 의미와 경로뿐이다. hybrid+mixed의 **출력 정확성은 미검증**이다(GPU gate 0회). 그러므로 fused_mixed는 §4 gate 통과 전에는 튜닝·판정 Job에 넣지 않는다.
