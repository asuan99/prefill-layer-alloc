# P1.3 — Layer-type-aware decode SM reservation on NemotronH (findings + design)

> ⚠️ **SUPERSEDED (2026-07)**: 본 문서의 layer-aware 설계·"이상적 케이스" 프레이밍은 이후 실측·serving으로 **기각**되었다(§02 민감도 반전 + §07 4-pass 은퇴). 정본: [sm_policy_report.html](sm_policy_report.html). 본 문서는 구현·이력 계층으로만 유효.

작성: 2026-07-03 · 상태: **기반 확보(pdmux가 hybrid서 최초 동작)**, layer-aware 구현은 설계.
대상: NemotronH-8B (temporal: 24 mamba `M` + 24 mlp `-` + 4 attn `*`, 52층, head_dim 128). 이유: 이미 sglang 지원 + flashinfer OK(Zamba2 head_dim 160 문제 없음) → layer-aware를 여기서 먼저 구축.

## 1. 기반 확보 (측정)
- `[measured]` **NemotronH-8B base 동작**: 정상 출력("The capital of France is"→" Paris…"). 단 **py3.14서 `--disable-piecewise-cuda-graph` 필수** — piecewise CUDA graph가 `torch._inductor`→`torch.ao.quantization`을 import하는데 `EdgeOrNode.__module__=`(typing.Union) 대입이 py3.14서 불가. (regular cuda graph도 현재 disable; 별도 open item.)
- `[measured]` **pdmux + hybrid 갭 발견·수정**: pdmux는 `forward_split_prefill`(레이어 윈도우 prefill)을 요구하나 **dense 14모델만 구현, hybrid(NemotronH/Zamba2)엔 없음** → `AttributeError: 'NemotronHForCausalLM' has no attribute 'forward_split_prefill'`. **NemotronHForCausalLM에 `forward_split_prefill` 추가**(patch: `src/patches/nemotron_h_forward_split_prefill.patch`; qwen2 패턴 미러, NemotronH의 (hidden, residual) 쌍을 forward_batch에 stash해 윈도우 간 전달).
- `[measured]` ⇒ **PD-multiplexing이 hybrid 모델(NemotronH)서 최초 동작**(job 827062): green-ctx 4 stream groups `[(108,0),(74,34),(54,54),(0,108)]`, split-prefill이 mamba/attn/mlp 레이어 통과, serve OK, 출력 "Paris" 정확.
- `[derived]` Zamba2도 동일 `forward_split_prefill` 추가 필요(현재 없음) — original_hidden_states 스레딩 때문에 NemotronH보다 약간 복잡. P1.3d 후 추가.

## 2. layer-aware 메커니즘 설계 (P1.3d — 핵심 기여)
### 현행 pdmux (MuxWise)
- prefill: `forward_split_prefill`로 레이어 윈도우 분할(prefill_stream). decode: `run_batch` 전체 모델 1회(decode_stream). SM 파티션: `adjust_stream_groups`가 **스케줄 iteration당 1회** decode_bs로 결정(모델 전체 고정).
### layer-aware 확장
- 목표: 비싼 **attn-decode 레이어**엔 decode SM floor 예약, 싼 **ssm/mlp-decode 레이어 윈도우**엔 SM을 prefill로 환원. NemotronH는 attn이 4/52라 환원 여지 큼(이상적 케이스).
  - ⚠️ **정정(2026-07)**: "이상적 케이스" 프레이밍은 *attn-decode가 비싼 SM-민감 층*이라는 전제에 기댔으나, §02 민감도 실측이 이를 **반증**했다 — NemotronH는 **mamba가 SM-민감**(2.6×, 62% of step)이고 attn은 둔감(GQA memory-bound). 따라서 예약 대상은 mamba+mlp(48/52)이고 환원가능은 attn 4/52뿐 → layer-aware 순이득≈0(≈agnostic). 나아가 진짜 "환원 여지 큼"인 Zamba2(45/54)조차 serving에선 **최악**으로 판명(§07). layer-aware는 어떤 하이브리드에서도 agnostic을 못 이겨 은퇴.
- 필요: decode 포워드도 **layer-type 윈도우로 분할**(`forward_split_decode` 신설), 윈도우 경계서 green-ctx 파티션 전환, prefill 진행과 상보적 조율.
- **CUDA graph 이점(우연)**: py3.14서 decode도 eager(graph disable) → 그래프 제약 없이 **per-layer-type 스트림 전환 자유**. 첫 구현에 유리(정확성 우선, 후에 graph 재도입).
- 훅 지점: `multiplex/multiplexing_mixin.py::event_loop_pdmux` — 현재 prefill만 split. decode도 layer-type run으로 바꾸고, `pdmux_context.set_current_stream_idx`로 attn-window엔 낮은 stream_idx(큰 decode SM), ssm-window엔 높은 stream_idx(작은 decode SM=prefill 환원) 설정. attn backend는 `update_decode_attn_backend(stream_idx)`로 파티션별 유지.
- 정책 3종(sim §14와 1:1): **fused**(baseline), **agnostic_protect**(전 레이어 고정 floor), **layer_aware_protect**(attn만 floor). NemotronH `hybrid_override_pattern`으로 attn 레이어 id={`*` 위치} 식별.

## 3. 측정 계획 (P1.4)
- NemotronH서 decode ITL / prefill throughput / goodput@SLO를 fused vs agnostic vs layer_aware로 비교(λ·SLO 스윕). sim §14 예측(temporal서 layer_aware가 agnostic 대비 goodput↑) 실측 재현/반증.
- Zamba2(torch_native)로 헤드라인 재현(fast attn backend 확보 후).

## 4. Open items
- `[measured]` py3.14 + piecewise/inductor: `--disable-piecewise-cuda-graph` 우회 중. regular CUDA graph 재도입(perf) = torch.ao Union.__module__ shim 또는 py3.13 env.
- `[measured]` triton attn backend hybrid 버그: `layer_id=0 not in full attention layers`(layer0가 attn 아닐 때 init 가정) — Zamba2 fast backend 후보라 별도 수정 필요.
- Zamba2 `forward_split_prefill` 추가(pdmux on Zamba2).

## 5. 산출물
- patch: `src/patches/nemotron_h_forward_split_prefill.patch`. boot: `triage/p1_3_nemotronh_boot.sbatch [plain|pdmux]`.
