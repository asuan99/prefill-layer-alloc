## vLLM 0.31.0에서 Mamba2(NemotronH) 층에 CUDA graph가 적용되는 방식 (코드 검토, 읽기 전용)

grep, sed, Read만 썼고 실행이나 import는 하지 않았습니다. 경로는 sdist 루트 `vllm-0.31.0/` 기준입니다. 결론부터 말하면 질문 1의 전제가 틀렸습니다. padding 칸은 PAD_SLOT_ID(-1)가 아니라 **NULL_BLOCK_ID(0)**로 채워집니다. 그리고 남은 state 오염 경로를 하나 찾았는데([추론], 미재현) 조건이 드뭅니다. 혼합 배치를 다룬 2절 끝과 6절에 정리했습니다.

### 1. decode FULL graph에서 padding 칸의 state 보호
- **값은 NULL_BLOCK_ID=0입니다.** 상수는 `PAD_SLOT_ID = -1`, `NULL_BLOCK_ID = 0`입니다(`vllm/v1/attention/backends/utils.py:51-52`). [코드 확인]
  - runner가 padding 칸의 block table을 0으로 채웁니다. 주석은 "Block 0 is reserved for padding"입니다(`vllm/v1/worker/gpu_model_runner.py:2332-2334`). [코드 확인]
  - Mamba builder도 FULL 경로에서 같은 칸에 `NULL_BLOCK_ID`를 한 번 더 씁니다(`vllm/v1/attention/backends/mamba_attn.py:801`). [코드 확인]
  - block 0은 BlockPool이 처음에 꺼내 null block으로 예약하므로 실제 요청에 배정되지 않습니다(`vllm/v1/core/block_pool.py:187-188`). [코드 확인]
  - PAD_SLOT_ID(-1)의 쓰임은 두 곳뿐입니다. 하나는 prefill용 `causal_conv1d_fn`의 program grid padding(`batch_ptr`)입니다(`utils.py:1089-1099`, `ops/causal_conv1d.py:86-89`). 다른 하나는 attention `slot_mapping`의 padding입니다(`gpu_model_runner.py:4111-4113`). [코드 확인]
- **padding 칸이 decode로 분류되는 근거:**
  - padding 칸의 `query_start_loc`은 마지막 누적값으로 채워집니다. 그래서 query_len이 0입니다(`gpu_model_runner.py:2063`). [코드 확인]
  - padding 칸의 `is_prefilling`은 강제로 False입니다(`gpu_model_runner.py:2355`). [코드 확인]
  - 따라서 split 단계에서 prefill로 잡히지 않습니다(`utils.py:856-863`). 결과적으로 `num_decodes`가 padded bs와 같아집니다. [코드 확인 + 추론]
- **causal_conv1d_update 커널은 건너뜁니다.**
  - index가 `null_block_id`이면 state를 읽기 전에 `return`합니다(`ops/causal_conv1d.py:830-839`). [코드 확인]
  - 기본값은 `null_block_id=NULL_BLOCK_ID`입니다(`ops/causal_conv1d.py:1106`). `HAS_NULL_BLOCK`도 켜져 있습니다(`:1272`). [코드 확인]
  - 그 칸의 출력은 쓰지 않습니다. padding 출력은 버려지므로 무해합니다. [추론]
- **SSU(triton) 커널은 load와 store를 모두 막습니다.**
  - `mask &= state_batch_idx != null_block_id` 조건으로 읽기를 막고, 읽힌 값은 0입니다(`ops/mamba_ssm.py:381-384`). [코드 확인]
  - 마지막 store도 같은 mask를 씁니다(`ops/mamba_ssm.py:493`). [코드 확인]
  - spec decode가 아니면 dst는 src와 같습니다(`ops/mamba_ssm.py:601`, `mamba_mixer2.py:1029-1030`). [코드 확인]
  - 출력 `out`은 padding 칸에도 0 state 기준으로 기록됩니다(`ops/mamba_ssm.py:454`). 버려지는 값이라 무해합니다. [추론]
  - triton SSU dispatch 경로는 `ops/ssu_dispatch.py:154-201`입니다. [코드 확인]
- **index 버퍼는 고정 주소이고 in-place 복사됩니다.**
  - 버퍼는 init에서 한 번 할당합니다: `self.state_indices_tensor_d = torch.empty((decode_cudagraph_max_bs, 1+num_spec))`(`mamba_attn.py:167-171`). 크기는 `min(max_num_seqs, max_cudagraph_capture_size)`입니다(`mamba_attn.py:124-129`). [코드 확인]
  - 복사 조건은 `num_prefills==0`, `num_decodes<=max_bs`, FULL 모드 존재입니다. 이때 `self.state_indices_tensor_d[:num_decodes].copy_(..., non_blocking=True)`로 복사하고, 그 slice를 metadata로 넘깁니다(`mamba_attn.py:791-801`). [코드 확인]
  - metadata build는 forward 전에 같은 stream에서 일어납니다. 그래서 replay 전에 복사가 끝난다고 봅니다. [추론]
  - capture 때도 같은 함수를 거칩니다(`mamba_attn.py:231-269`, `:770`). [코드 확인]

### 2. cudagraph dispatch 규칙
- **uniform decode 판정:** `max_num_scheduled_tokens == 1+num_spec`이고 `num_tokens == max_q*num_reqs`이면 uniform입니다(`gpu_model_runner.py:3853-3870`). 형태만 봅니다. [코드 확인]
- **NONE이 되는 경우:**
  - `num_tokens > max_cudagraph_capture_size(96)`이면 즉시 NONE입니다(`vllm/v1/cudagraph_dispatcher.py:272-279`). [코드 확인]
  - cascade attention이나 encoder output이 있으면 FULL이 금지됩니다(`gpu_model_runner.py:3972-3974`). [코드 확인]
- **FULL key 구성:**
  - 대상은 decode용 capture size 가운데 `<= 1*max_num_seqs(48)`인 것만입니다(`cudagraph_dispatcher.py:205-229`). 따라서 1–48 구간의 9개이고, 로그의 "FULL 9개"와 일치합니다. [코드 확인]
  - key의 `num_reqs`는 `min(padded//1, max_num_seqs)`, `uniform=True`입니다(`cudagraph_dispatcher.py:141-143`). [코드 확인]
- **PIECEWISE key 구성:** 15개 크기 전부이고 `num_reqs=None`, `uniform=False`입니다(`cudagraph_dispatcher.py:187-201`). [코드 확인]
- **조회 순서:** FULL key를 먼저 찾고, 없으면 relaxed PIECEWISE key, 그래도 없으면 NONE입니다(`cudagraph_dispatcher.py:305-322`). [코드 확인] 경우별로 정리하면 다음과 같습니다. [추론]
  - uniform decode ≤48 → FULL
  - 혼합 배치(prefill+decode) ≤96 토큰 → PIECEWISE
  - 96 토큰 초과 → NONE
- **attention metadata padding:** FULL일 때만 패딩합니다(`pad_attn = cudagraph_mode == FULL`, `gpu_model_runner.py:4300`, `:4363-4365`). PIECEWISE에서는 토큰 수만 capture 크기로 패딩하고 Mamba/attention metadata는 패딩하지 않습니다. [코드 확인]
- **cudagraph 지원 수준:**
  - Mamba builder 기본값은 `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`입니다(`mamba_attn.py:96`). 정의는 `vllm/v1/attention/backend.py:567-576`입니다. [코드 확인]
  - FlashAttention은 FA3이면 ALWAYS, 아니면 UNIFORM_BATCH입니다(`flash_attn.py:620-624`). A100이면 FA2라서 UNIFORM_BATCH라고 봅니다. [추론]
  - 전체 지원 수준은 그룹별 최솟값입니다(`gpu_model_runner.py:7109-7133`). [코드 확인]
- **자동 조정 경로:** `resolve_cudagraph_mode_and_sizes`(`vllm/config/compilation.py:1371-`)가 담당합니다. [코드 확인]
  - mixed_mode가 FULL인데 지원이 ALWAYS가 아니면 FULL_AND_PIECEWISE나 FULL_DECODE_ONLY로 강등합니다(`:1391-1415`). [코드 확인]
  - 지원이 NEVER이면 PIECEWISE나 NONE으로 갑니다(`:1417-1442`). [코드 확인]
  - spec decode인데 지원이 UNIFORM_BATCH 미만이면 강등합니다(`:1446-1462`). [코드 확인]
  - FULL_AND_PIECEWISE는 mixed=PIECEWISE라 이번 설정에서는 **강등이 발화하지 않습니다**. [추론]
  - hybrid 전용 검사가 하나 있습니다. FULL 모드에서 `max_num_seqs > Mamba block 수`이면 ValueError입니다(`:1516-1530`). [코드 확인]
- **1-token prefill 처리:** 앞선 state가 있는 1-token prefill 행은 decode 행으로 재분류됩니다. FULL graph가 형태만 보고 replay되기 때문에 생긴 처리입니다(`mamba_attn.py:480-520`). [코드 확인] 반대로 **앞선 state가 없는 1-token prefill(prompt 길이 1)은 prefill로 남습니다**(`mamba_attn.py:492-497`). 이 경우는 6절 위험 1로 이어집니다.

### 3. piecewise에서 Mamba
- **split op 등록:** `vllm::mamba_mixer2`가 `_attention_ops`에 들어 있습니다(`compilation.py:767`). `splitting_ops=None`이면 이 목록이 그대로 split op이 됩니다(`:1160`). [코드 확인]
- **eager 실행:** split graph는 compile 대상에서 빠집니다(`vllm/compilation/backends.py:1198-1202`). cudagraph wrap은 compile 대상 submodule에만 적용됩니다(`backends.py:738-768`). 따라서 piecewise에서 mamba_mixer2는 eager로 돕니다. [코드 확인]
- **custom op 정의:** `direct_register_custom_op("mamba_mixer2", mutates_args=["output"])`이고(`mamba_mixer2.py:1265-1269`), forward에서 호출합니다(`:588`). [코드 확인]
- **host 동기화 위치:** prefill/혼합 metadata의 `.item()`과 `.tolist()`는 전부 **CPU 텐서에 대한 호출**입니다. GPU→CPU 동기화는 아닙니다. [코드 확인] 다만 prefill 개수만큼 Python 루프 비용이 붙습니다. [추론]
  - `mamba_attn.py:318-321`: chunk metadata 루프 안 `.item()`, CPU 텐서. 그 입력은 `_prefill_cpu_metadata`가 `seq_lens_cpu_upper_bound`로 D2H 없이 만듭니다(`:353-378`).
  - `mamba_attn.py:513`: `torch.any(prefill_to_decode).item()`, CPU.
  - `mamba2_attn.py:164`: `bool((num_computed_tokens_p_cpu>0).any())`, CPU.
  - `utils.py:842`, `:865`, `:868`: split 단계 `.item()`, CPU의 `query_start_loc_cpu`.
  - `utils.py:1075`: conv1d metadata `nums.sum().item()`, CPU(`:1066` assert).
  - H2D 업로드는 `async_tensor_h2d`로 non-blocking입니다(`mamba_attn.py:408-414`).
  - `mamba_attn.py:713`, `:718`의 `.tolist()`는 ReplaySSM 전용이고 이번 설정에서는 비활성입니다.
  - 동기화하는 `.item()` 경로가 하나 있습니다. `causal_conv1d_fn` 안 `metadata is None`일 때의 경로입니다(`ops/causal_conv1d.py:651`). mixer는 `metadata=attn_metadata`를 넘기므로 타지 않습니다(`mamba_mixer2.py:867`, `ops/causal_conv1d.py:678-700`).
  - `mamba2_attn.py:43-49`의 `compute_varlen_chunk_metadata`(`.item()`·`.tolist()`)는 패키지 안에 호출처가 없습니다(테스트용).
  - `num_computed_tokens`는 GPU에서 계산되고 동기화가 없습니다(`vllm/v1/attention/backend.py:488-493`). mixer의 prefill state 적재와 저장도 `torch.where`와 index_put이라 동기화가 없습니다(`mamba_mixer2.py:884-888`, `:991`).

### 4. chunked prefill과 state
- **state 이어받기:** prefix caching이 꺼져 있으면 `mamba_cache_mode="none"`이고 `mamba_block_size=max_model_len`입니다(`vllm/model_executor/models/config.py:668-675`). 요청마다 state slot 1개를 계속 씁니다(`mamba_attn.py:574-593`, `[:,0]`). [코드 확인]
- **초기 state 플래그:**
  - `has_initial_states_p = num_computed_tokens > 0`입니다(`mamba_attn.py:611-613`). [코드 확인]
  - `prep_initial_states`는 배치 안에 앞선 state가 있는 prefill이 하나라도 있으면 True입니다(`mamba2_attn.py:152-164`). [코드 확인]
  - 이 플래그로 conv는 `has_initial_state`와 `cache_indices`에서 state를 읽고(`mamba_mixer2.py:854-869`), SSD는 `ssm_state[idx]`에 where-mask를 씌운 initial_states를 씁니다(`:876-888`). [코드 확인]
  - 끝 state는 `ssm_state[state_indices_tensor_p] = varlen_states`로 덮어씁니다(`:991`). [코드 확인]
- **chunk 경계 metadata:** 이미 계산된 길이가 chunk_size로 나누어떨어지지 않으면 첫 chunk로 남은 부분을 채웁니다(`mamba_attn.py:324-336`). chunk는 한 시퀀스만 담습니다(`:299-310`). 결과물 `cu_chunk_seqlen`, `seq_idx`, `last_chunk_indices`는 SSD에 넘어갑니다(`mamba_mixer2.py:904-907`). [코드 확인]
- **혼합 배치는 커널 호출이 나뉩니다.**
  - 배치는 decode가 앞에 오도록 정렬됩니다(`reorder_batch_threshold=1`, `mamba_attn.py:95`, `:227`). [코드 확인]
  - 토큰을 `torch.split([num_decode_tokens, num_prefill_tokens])`로 나눕니다(`mamba_mixer2.py:780-789`). [코드 확인]
  - prefill은 `causal_conv1d_fn`과 `mamba_chunk_scan_combined_varlen`으로 처리합니다(`:837-999`). decode는 `causal_conv1d_update`와 `selective_state_update`로 처리합니다(`:1001-1152`). 한 layer 안에서 별도 호출 2쌍입니다. [코드 확인]

### 5. hybrid 모델 전용 제약·우회
- **page 정렬:**
  - `_align_hybrid_block_size`가 담당합니다(`vllm/platforms/interface.py:835-`). [코드 확인]
  - mode가 none이면 `attn_block_size = kernel_align * cdiv(mamba_page, kernel_align*attn_page_1tok)`입니다(`:977-982`). [코드 확인]
  - 이 값이 기존 block_size보다 크면 "Setting attention block size to %d tokens"를 남기고 올립니다(`:987-993`). [코드 확인]
  - 나머지 차이는 `mamba_page_size_padded`로 맞춥니다(`:998-1010`). [코드 확인]
  - 1312는 이 식의 산출로 보입니다(16의 배수). 모델 차원으로 직접 재계산하지는 않았습니다. [추론]
- **cudagraph 강제 변경:**
  - hybrid면 `HybridAttentionMambaModelConfig` → `MambaModelConfig`가 호출됩니다(`vllm/config/vllm.py:2797-2798`, `models/config.py:462-476`, `:627-675`). docstring은 "FULL_AND_PIECEWISE by default"라고 하지만, 본문은 cache mode와 block size만 조정하고 cudagraph 모드는 건드리지 않습니다. [코드 확인]
  - hybrid에서 cudagraph를 끄는 조건은 찾지 못했습니다. 보이는 것은 Mamba block 부족 시 ValueError(`compilation.py:1516-1530`)와 DBO 비호환 목록(`vllm.py:3267-3268`)뿐입니다. [코드 확인] 기본 FULL_AND_PIECEWISE는 `vllm.py:316`, `:339`에 있습니다.
- **fp32 state 강제:** NemotronH에서 `mamba_ssm_cache_dtype`이 "auto"면 HF config 값을 쓰고, 그 값도 없으면 float32입니다(`models/config.py:678-705`, 등록은 `:1051`). 사용자가 명시하면 덮어쓰지 않으니 "강제"가 아니라 "기본값"입니다. [코드 확인] stochastic rounding은 기본으로 꺼져 있습니다(`vllm/config/mamba.py:43`).

### 6. 위험 평가
1. **앞선 state가 없는 1-token prefill이 FULL replay에 섞이면 state가 오염될 수 있습니다.** [추론, 미재현]
   - 그런 행은 prefill로 남습니다(`mamba_attn.py:492-497`). 그런데 dispatch는 형태만 보고 uniform으로 판정합니다(`gpu_model_runner.py:3863-3866`). 따라서 FULL graph가 선택될 수 있습니다.
   - 이때 `num_prefills>0`이라 고정 버퍼 복사를 건너뜁니다(`mamba_attn.py:791-795`). graph는 직전 step의 `state_indices_tensor_d`로 replay되므로, decode 행이 엉뚱한 slot을 읽고 쓸 수 있습니다.
   - runner나 scheduler에서 이 경우를 막는 장치는 찾지 못했습니다(`gpu_model_runner.py` 안 `num_prefills` 검사 0건).
   - 발생 조건은 prompt 길이 1, 또는 decode만 있는 배치에서 남은 token budget이 정확히 1인 경우입니다. 일반 벤치(긴 prompt, 큰 budget)에서는 사실상 발화하지 않는다고 봅니다.
2. **padding 칸은 보호됩니다.** padding을 NULL block으로 채우고 두 커널이 skip/mask하는 구조는 코드로 확인했습니다(1절). block 0 예약도 확인했습니다(`block_pool.py:187-188`). 남은 위험의 근거는 없습니다.
3. **비결정성 1: prefill 결과가 스케줄에 따라 달라집니다.** chunk 경계가 scheduler의 chunk 크기, 즉 동시 부하에 따라 달라집니다(`mamba_attn.py:324-344`). 그러면 같은 요청도 SSD 누산 순서가 바뀌어 fp32 수준에서 run마다 차이가 날 수 있습니다. [추론] atomic 연산은 Mamba ops에서 쓰지 않습니다(grep 0건, `scatter_states.py:79`는 non-atomic 명시).
4. **비결정성 2: decode SSU는 행 사이에 독립입니다.** program이 (m, b, h) 단위로 독립이고 dstate 축 reduction은 program 안에서 끝납니다(`ops/mamba_ssm.py:312-314`, `:449`). batch에 따라 바뀌는 것은 BLOCK_SIZE_M(dim 타일)뿐입니다(`ops/mamba_ssm.py:621`). 그래서 padding 크기나 FULL과 PIECEWISE 차이로 Mamba decode 수치가 바뀔 근거는 없습니다. [추론] attention 쪽은 범위 밖입니다.
5. **PIECEWISE의 padding 토큰:** `ssm_output`이 `torch.empty`라 padding 토큰 위치에 쓰레기 값이 남습니다(`mamba_mixer2.py:576-583`, `:781`). state에는 쓰지 않으므로 오염 근거는 없습니다.

### SGLang v0.5.10과 비교할 때 볼 지점
- padding 칸 표지값: vLLM은 NULL block(0), 커널 skip/mask입니다. SGLang이 -1이나 0 중 무엇을 쓰고 커널이 skip하는지 봅니다.
- decode-only FULL graph 진입 조건: vLLM은 형태로만 판정해 1-token fresh prefill 구멍이 있습니다(6절 1). SGLang이 forward_mode로 판정하는지 봅니다.
- 혼합 배치 처리: vLLM은 PIECEWISE에서 mamba_mixer2를 eager로 돌리고 decode/prefill 커널을 나눕니다. SGLang은 혼합 배치를 graph 밖에서 돌리는지 봅니다.
- prefill metadata: vLLM은 CPU 사본만 써서 D2H를 피합니다. SGLang Mamba metadata에 `.item()`/`.cpu()` 동기화가 있는지 봅니다.
- 상태 설정: fp32 SSM state 기본값, page 정렬로 attention block이 커지는 현상(1312), chunk 경계 정렬 규칙이 SGLang과 같은지 확인합니다.

> 출처: PyPI sdist vllm-0.31.0.tar.gz (sha256 dde46f1efbe846e5bd0c5c4ec6ec71842c42254c7ff3d9bebb2a57028715d380), 읽기 전용 코드 검토(general-purpose 에이전트), 2026-11-08. 줄 번호는 sdist 루트 기준.
