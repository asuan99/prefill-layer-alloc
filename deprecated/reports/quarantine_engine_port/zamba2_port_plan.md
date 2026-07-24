# Zamba2 → SGLang v0.5.10 Port Plan (P1.2)

작성: 2026-07-03 · 상태: 설계 확정, 구현 진행 중. 참조 완독(vLLM `zamba2.py` 992줄 + sglang NemotronH 템플릿 + Zamba2-2.7B config).
기반: editable dev tree `/scratch/ehmoon/whlee/sglang_engine_dev` (sglang v0.5.10, kernel 0.4.1+cu130) — P1.0서 pdmux 부팅 검증됨.

## 1. 아키텍처 (측정: Zamba2-2.7B config.json)
- 54 layers, `layers_block_type` = 대부분 `mamba` + 9개 `hybrid`(`hybrid_layer_ids=[6,12,18,24,30,36,42,47,51]`). temporal 하이브리드.
- `num_mem_blocks` 개의 **공유 transformer 블록**을 hybrid 위치마다 재사용, 위치별 **LoRA 어댑터**(`adapter_rank=128`)로 특화.
- hidden 2560, attention_hidden 5120, ffn 10240, attn_head_dim 160, n heads = 5120/160=32. mamba: d_state/d_conv/expand/ngroups/n_mamba_heads/headdim.

## 2. vLLM → SGLang 클래스 매핑
| vLLM (참조) | SGLang 대응 | 비고 |
|---|---|---|
| `Attention`(per-position, dpa_list) | **`RadixAttention`** ×N(hybrid 위치별 distinct `layer_id`) | KV는 위치별 유일 → 각 RadixAttention에 고유 layer_id(공유 attn이지만 KV 분리). ★가장 까다로움 |
| `MambaMixer2`(raw dims) | sglang `MambaMixer2(cache_params: Mamba2CacheParams, hidden_size, n_groups, rms_norm_eps, activation, use_rms_norm, quant_config, prefix)` | **시그니처 다름**: ssm_state/conv_kernel/intermediate/num_heads/head_dim가 `Mamba2CacheParams`에 번들. Zamba2용 CacheParams를 mamba_* 필드로 구성해야 함 |
| `ColumnParallelLinear`/`MergedColumnParallelLinear`/`QKVParallelLinear`/`RowParallelLinear`/`ReplicatedLinear` | 동명 `sglang.srt.layers.linear.*` | 대부분 그대로 |
| `get_rope` | `sglang.srt.layers.rotary_embedding.get_rope` | use_mem_rope일 때만 |
| `RMSNorm`/`VocabParallelEmbedding`/`ParallelLMHead`/`LogitsProcessor` | 동명 sglang | forward가 forward_batch 필요 |
| `Attention(q,k,v)` forward | `RadixAttention.forward(q,k,v,forward_batch)` | **forward_batch 스레딩**: 모든 layer.forward에 `forward_batch: ForwardBatch` 인자 추가 |
| `HasInnerState/IsHybrid/SupportsMambaPrefixCaching` | sglang의 hybrid mixin(NemotronHForCausalLM 참조) | `get_mamba_state_shape_from_config` 등 sglang 시그니처로 |
| `AutoWeightsLoader`+`WeightsMapper` | sglang weight loader idiom | 아래 §4 |

## 3. 레이어 구조 (그대로 이식할 로직)
- **Zamba2LoRA**: A(ColumnParallel, gather_output=True) → B(Column/MergedColumn); forward=B(A(x)).
- **Zamba2Attention**: 공유 qkv_proj/o_proj + 위치별 `linear_{q,k,v}_adapter_list[block_idx]` LoRA(있으면 Q/K/V에 가산) + 위치별 `dpa_list[block_idx]`(=RadixAttention). use_mem_rope시 rotary. `scale=(head_dim/2)**-0.5`.
- **Zamba2MLP**: gate_up_proj(Merged) + 위치별 `gate_up_proj_adapter_list[block_idx]` LoRA 가산 → GeluAndMul → down_proj.
- **Zamba2AttentionDecoderLayer**(공유 블록): `input_layernorm`는 **2*hidden**(concat[hidden, original]), self_attn, `pre_ff_layernorm`, feed_forward. forward가 `original_hidden_states` concat.
- **Zamba2MambaDecoderLayer**: MambaMixer2 + input_layernorm. forward: `hidden + transformer_hidden_states`(있으면) → norm → mamba → `residual + output`.
- **Zamba2HybridLayer**: shared_transformer(block_idx) → `linear`(ReplicatedLinear) → mamba_decoder(transformer_hidden_states 주입).
- **Zamba2Model**: `cycle(num_mem_blocks개 공유 블록)`; `layer2block_map={layer_idx:block_idx}`; layers_block_type 순회로 Hybrid/Mamba 레이어 배치. forward: `original_hidden_states=clone(embed)` 유지, 전 레이어에 전달; final_layernorm.
- **Zamba2ForCausalLM**: model + lm_head(tie_weights) + logits_processor. mamba state shape/dtype 클래스메서드.

## 4. 가중치 매핑 (HF Zamba2 → sglang param)
- qkv 스택: `q_proj/k_proj/v_proj` → `qkv_proj`(shard q/k/v). (단 Zamba2 HF는 이미 qkv? 확인 필요 — vLLM은 stacked_params_mapping 사용.)
- LoRA 어댑터 HF 저장형: `A_log`→`A`, `...0.weight`→`A.weight`, `...1.weight`→`B.weight`(HF는 Sequential[0,1]로 저장). vLLM `hf_to_vllm_mapper` 동일 적용.
- Mamba: `A_log`→`A`(그리고 sglang mamba 내부 파라미터명 정합 확인).

## 5. Config
- HF `transformers.Zamba2Config` 재사용(AutoConfig, trust_remote_code). NemotronH처럼 sglang `configs/zamba2.py`가 필요하면 mamba2_cache_params 헬퍼만 추가(§2 MambaMixer2용). **결정: 우선 HF config + Zamba2용 Mamba2CacheParams 빌더 헬퍼**로 시작.

## 6. 등록
- `sglang/srt/models/zamba2.py`에 `EntryClass = [Zamba2ForCausalLM]` (sglang은 models/ 자동 스캔). architectures="Zamba2ForCausalLM" 매칭.

## 7. 검증 계획 (checkpoint = 사용자 약속)
1. **로드/인스턴스화**: `--load-format dummy`로 Zamba2-2.7B config 부팅(레이어 조립·state pool 배선 확인).
2. **실가중치 로드**: hf_cache Zamba2-2.7B 가중치 전부 매핑(loaded_params 누락 0).
3. **logit-parity vs vLLM**: 동일 프롬프트 prefill 첫 토큰 logits를 vLLM(repo vllm_venv)과 비교(±tol). ← **여기서 사용자 체크포인트.**
4. 이후 pdmux + layer-aware(P1.3).

## 8. 리스크/미해결
- ★ **per-position RadixAttention**: 공유 attn 가중치인데 KV는 위치별. sglang RadixAttention가 layer_id로 KV pool 슬롯을 잡으므로, hybrid 위치 9개 각각에 distinct layer_id 부여 필요. sglang의 hybrid attn/mamba layer_id 할당 규약(NemotronH가 mamba/attn 레이어에 layer_id를 어떻게 나눠주는지) 정독 필요.
- ★ **Mamba2CacheParams(Zamba2)**: sglang이 config에서 cache_params를 어떻게 만드는지(NemotronH는 `config.mamba2_cache_params` 사전계산) — Zamba2용 빌더 작성.
- **hybrid mem pool**: HybridLinearKVPool + mamba state 크기 산정(`get_mamba_state_shape_from_config`)을 sglang 시그니처로.
- **input_layernorm 2*hidden**: concat 경로가 sglang forward_batch 흐름과 충돌 없는지.
- torch>2.6 green-ctx perf 경고(공통, P1.4서 측정).

## 9. 산출물
- 대상 파일: `sglang_engine_dev/python/sglang/srt/models/zamba2.py`(+필요시 `configs/zamba2.py`). dev tree라 editable 즉시 반영. 완성분은 repo `workspace/engine-port/`에 사본/patch로 트래킹.
