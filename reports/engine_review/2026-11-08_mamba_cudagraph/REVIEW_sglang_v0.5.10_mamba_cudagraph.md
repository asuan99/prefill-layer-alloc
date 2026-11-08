## SGLang v0.5.10 Mamba2(NemotronH) CUDA graph 경로 코드 검토 (읽기 전용)

검토 범위와 한계: 코드를 읽기만 했다. 실행, GPU 사용, 수정은 하지 않았다. 아래 파일:줄은 모두 `~/Experiments/KISTI/sglang_engine_dev/python/sglang/srt/` 기준이다. CUDA 커널 소스는 `~/Experiments/KISTI/sglang_engine_dev/sgl-kernel/csrc/`에서 봤다. correctness gate는 이 검토로 하나도 닫히지 않는다. 결론은 전부 코드 수준이며, GPU에서 동치성을 확인한 것은 없다.

### 0. 트리 동일성
- [코드 확인] `git status`로 수정 14파일과 미추적 파일(`configs/mamba2.py`, `configs/zamba2.py`, `models/mamba2.py`, `models/zamba2.py`, `multiplex/{chunk_probe,controller,dual_worker,green_readout,holb_probe,multigpu_guard,profile,telemetry}.py`)을 확인했다.
- [코드 확인] Mamba 경로의 핵심 11파일을 `prefill-layer-alloc/workspace/engine-port/env/sglang_v0.5.10_python_tree.sha256`(v0.5.10 `1519acf`, git archive)과 sha256으로 대조했고 전부 일치(SAME)했다. 대상은 `mamba/mamba.py`, `mamba/ops/mamba_ssm.py`, `mamba/ops/ssu_dispatch.py`, `mamba/causal_conv1d.py`, `mamba/causal_conv1d_triton.py`, `mamba/mamba2_metadata.py`, `hybrid_linear_attn_backend.py`, `cuda_graph_runner.py`, `piecewise_cuda_graph_runner.py`, `forward_batch_info.py`, `attention_registry.py`다. 경로 위에서 pristine과 다른 파일은 `models/nemotron_h.py`와 `mem_cache/memory_pool.py` 두 개다(6절).
- [코드 확인] `results/stock_baseline/vessl/*/meta/engine_tree_pristine_tree_manifest.sha256`는 로컬에 없다. find 결과 0건이라 git 밖 아티팩트로 보인다. 그래서 SG-stock이 pristine 트리에서 돌았는지는 로컬에서 확인하지 못했다.
- [코드 확인] sgl-kernel 소스 버전은 `sgl-kernel/pyproject.toml`의 0.4.1이고, venv 고정값 `sglang-kernel==0.4.1+cu130`과 같다. 다만 바이너리 wheel이 이 소스로 빌드됐는지는 확인할 수 없었다[추론].

### 1. decode graph에서 padding 칸의 state 보호
**실제로 호출되는 커널**
- [코드 확인] NemotronH는 `use_triton_causal_conv=True`를 고정으로 넘긴다(`models/nemotron_h.py:415`). 그래서 decode conv는 sgl_kernel CUDA 커널이 아니라 `causal_conv1d_update_triton`을 탄다(`mamba/mamba.py:603-615`). 같은 이유로 prefill conv도 `causal_conv1d_fn_triton`을 직접 호출한다(`mamba.py:504-518`). 이미 짚은 host sync `causal_conv1d.py:27`(`.cpu().tolist()`)은 NemotronH 경로에서는 실행되지 않는다. 호출자가 `seq_lens_cpu=mixed_metadata.extend_seq_lens_cpu`를 넘기기 때문이다(`mamba.py:518`).
- [코드 확인] SSU는 `ops/__init__.py`, `ssu_dispatch.py:selective_state_update`를 거쳐 `--mamba-backend` 기본값 `"triton"`(`server_args.py:490`)인 `TritonSSUBackend`로 가고, 최종적으로 `ops/mamba_ssm.py:300 selective_state_update`가 불린다. 하네스가 `--mamba-backend flashinfer`를 줬는지는 확인하지 못했다. FlashInfer 커널 본문은 트리 밖이라 검토 범위가 아니다.

**pad=−1이 들어올 때의 동작**
- [코드 확인] Triton conv update는 `_causal_conv1d_update_kernel`에서 `conv_state_batch_coord == pad_slot_id`이면 곧바로 `return`한다(`causal_conv1d_triton.py:651-654`). conv_state load/store가 모두 일어나지 않는다. `USE_PAD_SLOT=pad_slot_id is not None`은 항상 True다(`:1180`).
- [코드 확인] Triton SSU는 `mask &= state_batch_idx != pad_slot_id` 조건으로 state load를 막고(`mamba_ssm.py:176-179`, other=0.0), 최종 store도 같은 mask를 쓴다(`:296-297`). intermediate store도 `if state_batch_idx != pad_slot_id`로 막혀 있다(`:265`). 주소 계산은 `state_ptr += (-1)*stride`라서 버퍼 바깥을 가리키지만, 접근이 전부 mask에 걸리므로 실제 메모리 접근은 없다(`:154-155`).
- [코드 확인] 참고로 NemotronH는 쓰지 않는 sgl_kernel CUDA `causal_conv1d_update_kernel`도 `conv_state_batch_coord == params.pad_slot_id`이면 early return한다(`sgl-kernel/csrc/mamba/causal_conv1d.cu:577-580`).

**padding 칸의 출력이 가는 곳**
- [코드 확인] Triton conv update는 출력을 `out = torch.empty_like(x)`로 잡는다(`causal_conv1d_triton.py:1060`). padding 행은 early return으로 쓰이지 않으므로 초기화되지 않은 값이 남는다. 이 값이 SSU 입력이 되고, SSU는 padding 행에도 `out`을 쓴다(`mamba_ssm.py:284`, state=0이므로 사실상 x·D). 쓰는 곳은 `preallocated_ssm_out_d`의 해당 행이다(`mamba.py:683`).
- [코드 확인] graph 출력은 `next_token_logits[: self.raw_num_token]`로 잘려 나간다(`cuda_graph_runner.py:1167`).
- [추론] 이후 gated norm, out_proj GEMM, MLP는 모두 행 단위 연산이라, padding 행의 쓰레기 값(NaN일 수도 있음)이 실제 행으로 섞여 들어가는 경로는 찾지 못했다. 다만 Mixer2RMSNormGated와 MoE 경로 본문은 행 독립성까지 정밀하게 읽지는 않았다.

**state index와 query_start_loc 버퍼**
- [코드 확인] 두 버퍼는 고정 주소다. `init_cuda_graph_state`에서 bs별로 `state_indices_list[i]`(pad로 채움)와 `query_start_loc_list[i]`를 한 번 할당한다(`hybrid_linear_attn_backend.py:401-439`). capture 때 이 텐서를 metadata로 넘기고(`:475-494`), replay 전에는 `_replay_metadata`가 in-place로 갱신한다. `state_indices_list[bs-1][:len].copy_(mamba_indices)`(`:507`)와 `query_start_loc_list[bs-1]`의 `copy_`/`fill_`(`:508-519`)이다. 같은 replay 경로가 `buffers.req_pool_indices`의 padding 행도 0으로 덮어쓴다(`:503`).
- [코드 확인] padding 개수는 `count_nonzero(seq_lens_cpu == 1)`로 센다(`:500-502`, fill value 1은 `:561-562`). 이 연산은 CPU 텐서에서 일어나므로 GPU sync가 없다.
- [코드 확인] decode 커널(conv update와 SSU)은 query_start_loc를 쓰지 않는다. `query_start_loc_p`는 prefill에서만 사용된다(`mamba.py:472`).
- [추론] padding 판정은 seq_len==1인 행을 padding으로 보는 휴리스틱이다. decode에서는 seq_len이 2 이상이므로 실제 요청이 padding으로 잘못 판정될 일은 없다고 본다.
- [코드 확인] 슬롯 0은 예약된 dummy다. `free_slots = arange(1, size+1)`이고 주석에 "padded slot 0 is used for writing dummy outputs"라고 적혀 있다(`memory_pool.py:325-328`). 매핑의 초기값은 0이다(`:511`). 하지만 decode padding은 슬롯 0이 아니라 −1을 쓴다.

### 2. graph 사용 조건
- [코드 확인] `_forward_raw`는 `forward_mode.is_cuda_graph()`이고 `graph_runner.can_run()`이 참일 때만 graph를 탄다(`model_runner.py:2817-2833`). `is_cuda_graph`는 DECODE, TARGET_VERIFY, IDLE, DLLM_EXTEND일 때 참이다(`forward_batch_info.py:167-173`). EXTEND, MIXED, SPLIT_PREFILL은 여기서 제외된다.
- [코드 확인] `can_run`의 조건(`cuda_graph_runner.py:668-731`)은 다음과 같다.
  - bs 조건: padding이 꺼져 있지 않으면 `bs ≤ max_bs`. PD-mux에서는 graph_key가 `"{stream_idx}_{bs}"`다.
  - 그 밖의 조건: mlp_sync일 때 `can_run_dp_cuda_graph`, encoder_lens, capture_hidden_mode 일치, TBO, ngram.
  - 이 함수에는 forward mode나 mixed 여부 검사가 없다. 그 판단은 위의 `is_cuda_graph`가 맡는다.
  - capture_bs는 `bs ≤ req_to_token_pool.size`로 걸러진다(`:479`). SG-stock 설정(max_bs 48, max_running 48)에서는 모든 decode step이 graph를 탄다[추론, bs가 48을 넘을 수 없음].
- [코드 확인] extend 배치가 graph를 타지 않는 경로는 `forward_extend`다. `piecewise_cuda_graph_runner`가 None이거나 `can_run`이 False이면 eager로 돈다(`model_runner.py:2650-2688`).
- [코드 확인] SG-stock(piecewise OFF)에서는 `init_piecewise_cuda_graphs`가 `disable_piecewise_cuda_graph`일 때 바로 return하므로(`model_runner.py:2469-2477`) prefill과 extend가 전부 eager다. mixed chunk는 기본값 off이고(`server_args.py:630`; `scheduler.py:906-909`는 chunked가 켜져 있고 enable_mixed_chunk일 때만 켬), 기본 스케줄러는 prefill 배치를 먼저 돌린다(`scheduler.py:2284-2295`). 그래서 decode 배치는 순수 DECODE이고 graph를 탄다. 하네스가 `--enable-mixed-chunk`를 줬다면 MIXED 배치는 eager가 된다(미확인).
- [코드 확인] PM-ref의 prefill은 SPLIT_PREFILL 모드로 `forward_split_prefill`을 타고, graph 판단 없이 eager로 돈다(`model_runner.py:2709-2726`, `2867-2872`). 추가로 chunked가 −1이면 `piecewise_cuda_graph_max_tokens = chunked_prefill_size = -1`이 되고(`server_args.py:1259`), 생성된 capture 목록이 비어(`:1411-1413`) piecewise 자체가 꺼진다(`model_runner.py:2485-2489`). dev 주석 `scheduler.py`(chunk_probe 설명)에도 같은 내용이 있다.

### 3. piecewise ON이었다면(upstream 기본)
- [코드 확인] 캡처 토큰 크기는 다음과 같다(`server_args.py:1397-1413`).
  - max는 `piecewise_cuda_graph_max_tokens = chunked_prefill_size`이고, 이 설정에서는 8192다(`:1254-1261`).
  - 크기 목록: 4–32(간격 4), 48–256(16), 288–512(32), 576–1024(64), 1280–4096(256), 4608–8192(512).
- [코드 확인] 적용 조건(`piecewise_cuda_graph_runner.py:415-428`): input_embeds가 없고, 요청한 logprob이 extend 구간을 덮지 않으며, `num_tokens ≤ max_num_tokens`여야 한다. 자동 비활성 조건 목록은 `server_args.py:1078-1139`에 있는데, NemotronH는 비활성 아키텍처 목록에 없다(`configs/model_config.py:1352-1358`). `enable_pdmux`도 이 목록에 없다.
- [코드 확인] Mamba는 split op이고 eager로 돈다. `NemotronHMambaDecoderLayer.forward`는 `is_in_piecewise_cuda_graph()`일 때 `nemotron_mamba2_with_output`을 부르고(`nemotron_h.py:432-435`), 이 함수는 `@register_split_op`이다(`:1022-1024`). split op 안에서는 `_forward_mamba` 전체가 eager로 돈다. 여기에는 in_proj와 out_proj GEMM도 들어 있다(mixer.forward가 둘 다 수행, `mamba.py:411, 695`).
- [코드 확인] attention 커널도 split op이라 eager다. RadixAttention이 extend에서 forward context가 있으면 `unified_attention_with_output`(split op)을 부른다(`layers/radix_attention.py:117-125, 138-140`). 결과적으로 graph 안에 남는 것은 norm, attention의 qkv/o proj, MLP/MoE 층이다.
- [코드 확인] padding 처리: split op은 `num_actual_tokens`까지 잘라서 실행하고, 그 행만 결과를 복사한다(`nemotron_h.py:1036-1052`). 나머지 행은 `torch.empty_like` 상태로 남는다(`:433`). graph 버퍼의 padding 토큰은 `out_cache_loc.zero_()`로 처리된다(`piecewise_cuda_graph_runner.py:605-607`). metadata는 padding하지 않은 원래 forward_batch로 `init_forward_metadata`를 호출해 만든다(`:774`).
- [코드 확인] 혼합(prefill+decode) 처리: MIXED는 EXTEND로 정규화된다(`piecewise_cuda_graph_runner.py:685-694`). `mix_with_running`은 decode 요청을 extend_len=1, prefix>0으로 넣는다(`managers/schedule_batch.py:1874-1899`). 그래서 `prepare_mixed`에서 num_decodes는 0이 되고(`mamba2_metadata.py:196`), decode 요청도 SSU가 아니라 chunk-scan varlen 경로에서 initial state를 받아 처리된다[추론, `has_initial_states=prefix>0`, `:200`].

### 4. chunked prefill과 state 이어받기
- [코드 확인] chunked 요청은 mamba 슬롯을 계속 유지한다. `HybridReqToTokenPool.alloc`에서 `req.mamba_pool_idx`가 이미 있으면 그 값을 재사용한다(`memory_pool.py:539-541`). 슬롯 해제는 finish 시점의 `release_kv_cache` → `free_mamba_cache`에서만 일어난다(`mem_cache/common.py:509-514`).
- [코드 확인] 새로 할당하는 슬롯은 alloc 시점에 conv와 ssm을 0으로 채운다(`memory_pool.py:347-361`).
- [코드 확인] 초기 state 플래그는 `has_initial_states = extend_prefix_lens > 0`이고, `prep_initial_states = any(...).item()`이다(`mamba2_metadata.py:200-201`, 이미 짚은 host sync).
  - conv: `has_initial_state[i]`가 참일 때만 conv_states에서 이전 토큰을 읽고, 거짓이면 0을 쓴다(`causal_conv1d_triton.py:116-158`). 최종 conv state는 chunk_offset 0 프로그램이 기록한다(`:160-261`). chunk_offset>0 프로그램은 conv_state가 아니라 x에서 이전 토큰을 읽는다(`:263-`).
  - SSM: `initial_states = where(has_initial, ssm_state[idx], 0)`이고(`mamba.py:524-530`), 끝나면 `ssm_state[idx] = varlen_state`로 쓴다(`:562`).
- [코드 확인] chunk metadata(`seq_idx`, `chunk_indices`, `chunk_offsets`)는 `prep_initial_states`일 때만 계산된다(`mamba2_metadata.py:203-221`). 이 계산 함수(`:129-150`)는 GPU 스칼라로 Python 루프를 돌기 때문에 시퀀스마다 암묵적 host sync가 생긴다[추론, 텐서 인덱스로 슬라이스하고 `torch.arange(N)`의 N이 텐서]. 이미 짚은 `:201`과는 별개다.
- [추론] SG-stock(8192)은 긴 prompt의 두 번째 chunk부터 initial-state 경로를 탄다. PM-ref(chunked −1, radix OFF)는 prefix가 항상 0이라 이 경로를 전혀 타지 않는다. 두 arm의 prefill 수치 경로가 구조적으로 다르다는 뜻이다(7절).

### 5. PD-mux(stream group) 특유 위험
**같은 pool 텐서를 다른 슬롯으로 동시에 쓰는가**
- [코드 확인] 그렇다. stream group마다 `_get_attention_backend()`로 HybridLinearAttnBackend와 Mamba2AttnBackend를 새로 만든다(`model_runner.py:1963-1968`, `attention_registry.py:222`). 하지만 모두 같은 `model_runner.req_to_token_pool`을 참조한다(`hybrid_linear_attn_backend.py:142`). 따라서 conv/temporal state 텐서는 공유되고, metadata 버퍼(`state_indices_list`, `query_start_loc_list`)만 group별로 따로 있다. `init_cuda_graph_state`가 group마다 호출되고(`cuda_graph_runner.py:659-663`), prefill은 별도의 `self.attn_backend`를 쓴다(`model_runner.py:2715-2716`).
- [코드 확인] graph 키와 metadata 버퍼는 같은 `get_current_stream_idx()`로 고른다(`cuda_graph_runner.py:1117-1119, 1157`). 캡처는 group별로 `graph_capture(stream=sg[1])`에서 한다(`:811-817`). 모든 graph는 하나의 전역 graph memory pool을 공유한다(`:1023-1028`).
- [추론] decode graph(running 요청의 슬롯)와 prefill(새 요청의 슬롯)은 동시에 같은 텐서의 서로 다른 행을 읽고 쓴다. 쓰는 방식은 index_put과 gather, 커널의 행 단위 포인터라서, 슬롯 집합이 서로소인 한 메모리 수준의 경합은 없다.

**순서를 보장하는 곳**
- [코드 확인] 순서 보장 지점은 다음과 같다(`multiplexing_mixin.py`, `event_loop_pdmux`). sync 구조 자체는 pristine에도 있다(HEAD 판 `:137-138, 184, 194, 215`).
  - 새 슬롯 할당과 0 채움: `update_split_prefill_batch`가 prefill_stream 컨텍스트에서 실행된다(`:1286-1292`, `:1218-1232`).
  - decode 끝: 매 iteration `decode_stream.synchronize()`(`:1494`) 뒤에 `process_batch_result`가 오므로 finish된 요청의 슬롯 해제는 drain 이후에 일어난다.
  - prefill에서 decode로 넘기기: 마지막 span 뒤에 `prefill_stream.record_event()`를 하고(`:1479`), `prefill_exe_done.query()`가 참인지 TP allreduce로 확인한 뒤에야 `merge_batch`를 한다(`:1513-1532`). decode graph가 그 슬롯을 처음 읽는 시점은 prefill 커널이 끝난 뒤다.
  - partition 전환: 양쪽 stream을 `synchronize()`한 다음 `set_current_stream_idx`와 `update_decode_attn_backend`를 바꾼다(`:1344-1352`, `:1361-1365`).
  - graph pool을 공유해도 한 번에 하나의 decode replay만 돌기 때문에, 직렬화는 위 sync들이 보장한다[추론].
- [추론] padding 행은 replay 중에 `req_pool_indices=0`으로 `mapping[0]`을 읽는데, 이 값은 곧바로 −1로 덮어써진다(`hybrid_linear_attn_backend.py:503-506`). 그래서 prefill 쪽이 매핑을 동시에 써도 영향이 없다.
- [추론] 미검증 항목이 하나 있다. `merge_batch` 등으로 tensor가 prefill stream과 decode stream 사이를 옮겨갈 때 caching allocator 수명이 제대로 지켜지는지(`record_stream`이 없음)다. 이것은 Mamba에 국한된 문제가 아니고, 오염 근거를 찾지도 못했다(근거 없음).
- [코드 확인] true dual(`PDMUX_TRUE_DUAL_WORKER=1`)에서는 decode와 prefill을 서로 다른 host thread에서 낸다(`:1381-1400`, `:1450-1466`). stream idx는 module-global이고(`pdmux_context.py:10, 144-164`), 바뀌는 것은 drain과 `safe_to_switch` 이후뿐이다(`multiplexing_mixin.py:1346-1350`). Triton JIT를 두 thread에서 처음 동시에 호출할 때의 안전성은 검토하지 않았다(근거 없음). gate 5에 따라 기본값은 OFF다.

### 6. dev tree 수정분의 영향
- [코드 확인] `models/nemotron_h.py`에서 경로에 걸리는 곳은 두 군데다.
  - (a) `NemotronHModel.forward`에 decode 층별 green-ctx stream 전환과 타이밍 코드가 추가됐다(`:608-621, 642-737`). 환경변수 `PDMUX_FIXED_DECODE_SM[_FILE]`과 `SGLANG_NH_LAYER_TIMING`이 없으면 `_tgt`가 `_base`를 돌려주므로(`:691-697`) 실행 경로는 사실상 no-op이다. Python 오버헤드와 `torch.cuda.current_stream()` 호출만 늘어난다.
  - 이 환경변수가 graph 캡처 중에 켜져 있으면 문제가 생긴다[추론]. 전환 결정이 캡처 시점 값으로 굳어서, replay 중에 파일로 sweep해도 반영되지 않는다. 또 timing 블록의 `.item()`과 `torch.cuda.synchronize()`(`:724, 733`)는 캡처 중에 실행할 수 없는 연산이다. 이 코드는 이미 죽은 트랙인 층 경계 전환용이므로 운영점에서는 꺼져 있어야 한다.
  - piecewise ON이면 이 추가 코드가 dynamo trace에 들어가므로 pristine과 graph break 양상이 다를 수 있다[추론]. SG-stock과 PM-ref는 둘 다 piecewise OFF라 해당이 없다.
  - (b) `forward_split_prefill`이 추가됐다(`:859-891`). pristine NemotronH에는 이 함수가 없어서 PM-ref의 prefill이 이 함수에 의존한다. 이 함수는 `layer.forward`를 거쳐 `_forward_mamba`로 가는 동일한 Mamba 경로를 쓰고, metadata init은 `split_index==0`일 때 한 번만 한다(`model_runner.py:2715`). 그 결과 Mamba state에 대해 새로 생기는 위험은 찾지 못했다.
- [코드 확인] `mem_cache/memory_pool.py`의 수정은 `get_v_head_dim`에 `full_layer_nums==0` 분기를 넣은 것뿐이다(`:1412-1421`). NemotronH(attention 층 있음)는 이 분기를 타지 않는다.
- [코드 확인] `layers/attention/triton_backend.py:97-103`: `mambaish_config`이면 v_head_dim을 `get_v_head_dim()`에서 가져온다. NemotronH에서 triton attention backend를 쓸 때 scratch 크기 산정에만 영향이 있고, Mamba state와는 무관하다.
- [코드 확인] 나머지 파일:
  - `model_runner.py`: Zamba2Config 추가, ngram getattr, coord 전용 `init_decode_metadata_coord`/`forward_split_decode`(`:2730-2747`). coord 함수는 `PDMUX_LA_COORD`일 때만 쓰인다(`scheduler.py` dispatch).
  - `scheduler.py`: holb/chunk probe(기본 OFF).
  - `multiplexing_mixin.py`: SLO/R2/dual 분기를 추가했지만 기존 sync 지점은 그대로 유지했다.
  - `pdmux_context.py`: divide_sm 분기의 assert.
  - `parallel_state.py`: thread-local role.
  - `server_args.py` / `model_runner_kv_cache_mixin.py`: 순수 Mamba2 전용.
  - `falcon_h1.py`, `granitemoehybrid.py`, `configs/__init__.py`, `hf_transformers_utils.py`: NemotronH 경로 밖.
- [추론] 로컬 dev tree와 VESSL에서 실제로 돈 트리(bundle sync와 `devtree_manual_edits.patch` 적용본)가 바이트 단위로 같은지는 이번 검토 범위 밖이다.

### 7. 위험 평가
- **padding 칸 state 오염**: 오염 근거가 없다. 위험은 낮다. conv와 SSU 모두 커널 본문에서 −1을 만나면 load/store를 건너뛴다(1절). 남는 것은 padding 행의 쓰레기 값이 실제 행으로 새는지인데, 행 독립 연산이라 근거를 찾지 못했다. norm과 MoE 본문까지는 정밀하게 읽지 않았다.
- **PD-mux 동시 접근**: 오염 근거가 없다. 슬롯이 서로소이고, 넘겨받기(event query 후 merge)·해제(decode sync 후)·전환(양쪽 drain)이 모두 막혀 있다(5절).
- **남은 경로 1 (낮음)**: 실행 중 재캡처다. `recapture_if_needed`는 capture_hidden_mode가 바뀌면 `capture()`를 다시 부른다(`cuda_graph_runner.py:1036-1066`). 이때 warmup `run_once`가 2회 eager로 돌고(`:1020-1023`), capture metadata에는 −1 처리가 없으며(`hybrid_linear_attn_backend.py:474-475`), `req_pool_indices` 버퍼에는 직전 replay의 실제 인덱스가 남아 있다. 결과적으로 살아 있는 요청의 슬롯 state가 warmup으로 갱신될 수 있다[추론]. 이 경로가 열리려면 hidden state 반환이나 spec 같은 조건이 있어야 하는데, 이 하네스에서 그런 조건이 생기는 경로는 찾지 못했다. 부팅 시점 캡처는 매핑이 0이라 dummy 슬롯 0에만 쓴다(`memory_pool.py:325-328, 511`).
- **남은 경로 2**: 환경변수로 켜는 dev 층별 stream 전환과 timing이 캡처 중에 켜지는 경우다(6절 a). 운영점에서는 꺼져 있어야 한다.
- **비결정성과 출력 비동치 (오염은 아님)** [추론]:
  - SG-stock(chunked 8192, initial-state 이어받기)과 PM-ref(한 번에 전체 prompt)는 prefill 수치 경로가 다르다.
  - decode는 padding된 graph bs에 따라 GEMM의 M 크기가 바뀐다. cuBLAS 알고리즘은 batch-invariant가 아니므로 greedy 출력이 arm 사이에서 bitwise로 같다는 보장이 없다. SSU와 conv update는 행마다 독립인 grid라 batch-invariant로 본다.
  - 그래서 gate 2(legacy와 새 경로의 출력 동치)는 "같은 입력, 같은 배치 구성" 조건에서 판정해야 한다. 이 검토는 gate 2를 닫지 않는다.
- **host sync (correctness 아님)**: `mamba2_metadata.py:201`과 `:129-150`(chunked일 때만). `causal_conv1d.py:27`은 NemotronH에서 실행되지 않는다(1절).

### vLLM 0.31.0과 비교할 때 볼 지점
1. Mamba2 층이 decode에서 FULL graph 안에 들어가는지, 아니면 splitting op으로 piecewise 밖에 있는지(SGLang은 decode는 전체 graph, extend는 split op eager).
2. graph용 metadata builder가 padding state index를 PAD_SLOT_ID로 채우는지, 그리고 conv/SSU 커널(Triton인지 CUDA인지)이 그 값으로 load/store를 막는지.
3. 혼합 배치의 순서(decode를 먼저 두는지 prefill을 먼저 두는지)와, decode 요청을 SSU로 처리하는지 chunk-scan으로 처리하는지.
4. chunked prefill의 `has_initial_states`/`chunk_indices` 계산 위치(CPU/GPU)와 host sync 개수, 슬롯을 할당할 때 0으로 채우는지 아니면 플래그만 쓰는지.
5. 실행 중 재캡처나 warmup이 살아 있는 슬롯을 건드릴 수 있는지, 슬롯 해제와 재사용이 비동기 실행과 어떻게 순서가 잡히는지.

> 출처: 로컬 dev tree ~/Experiments/KISTI/sglang_engine_dev (v0.5.10 1519acf + 수정 14파일), 읽기 전용 코드 검토(engine-porter 에이전트), 2026-11-08.
