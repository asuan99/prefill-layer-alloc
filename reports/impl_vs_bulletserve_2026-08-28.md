# 구현체 대조: 본 프로젝트의 layer-wise(split) prefill vs. BulletServe

- 작성일: **2026-08-28**
- 증거 수준: **코드 독해(static read)만**. 본 문서에는 새 측정·성능 판정이 **없다**.
  성능/정책 주장은 전부 정본(`PROJECT_STATUS.md`, `reports/CONSENSUS.md`)을 따른다.
- 대조 대상 스냅샷
  - 본 프로젝트: `workspace/engine-port/src/` + editable tree
    `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/` (SGLang v0.5.10 기반)
  - BulletServe: `github.com/zejia-lin/BulletServe` **main @ 445afae2**
    (커밋일 2025-12-02, 2026-08-28 tarball로 취득). ASPLOS 2026, arXiv 2504.19516.

---

## 0. 한 문장 요약

둘 다 "prefill을 layer 단위로 쪼개서, 그 경계마다 SM 배분을 갱신한다"는 **같은 골격**을 갖지만,
**쪼개는 구현 방식**(전용 SPLIT_PREFILL forward mode vs. pipeline-parallel 배관 재사용),
**쪼개는 이유**(단일 event loop의 유일한 인터리브 지점 vs. 별도 프로세스의 재-provisioning 지점),
**span 크기 결정**(토큰 예산 기반 동적 vs. 고정 layers/step),
**경계에서의 비용**(green-ctx 전환 = 양 stream drain vs. libsmctrl bitmask write)이 전부 다르다.
가장 큰 구조적 차이는 layer-wise 자체가 아니라 **프로세스 모델**이다 —
우리는 1 프로세스 1 event loop, Bullet은 MPS 위 2 프로세스(prefill/decode)다.

---

## 1. 요약 대조표

| 축 | 본 프로젝트 (engine-port / SGLang PD-mux) | BulletServe @445afae2 |
|---|---|---|
| layer-wise prefill 진입점 | `ForwardMode.SPLIT_PREFILL` + 모델별 `forward_split_prefill(ids, pos, fb, (start,end))` | `model.forward` **monkey-patch**로 `model.start_layer/end_layer`를 step마다 재설정 |
| 중간 활성값 전달 | `forward_batch.hidden_states` / `forward_batch.residual` (배치 객체에 stash) | `PPProxyTensors` (pipeline-parallel proxy tensor 재사용) |
| 첫/마지막 span 특수처리 | `start==0`이면 embed, `end==L`이면 `norm_f`+`logits_processor` | `pp_group.is_first_rank / is_last_rank`를 step마다 토글 |
| 모델 이식 비용 | **모델마다 메서드 작성 필요** (본 프로젝트가 hybrid 4종에 이식) | **모델 코드 무수정** (PP를 지원하면 자동) |
| 기본값 | PD-mux 사용 시 **필수 경로** (event loop가 이걸 전제) | `--num-prefill-layers` **기본 -1 = 비활성**(옵션 노브) |
| span 크기 | `split_forward_token_budget // extend_num_tokens` (기본 65536) → **토큰 수에 반비례하는 동적 layer 수** | `num_prefill_layers` **상수**, `num_steps = ceil(L/n)` |
| span 경계에서 하는 일 | 컨트롤러 평가 → 필요 시 green-ctx stream group 교체(**양 stream synchronize**) + decode attn backend 교체 | `update_bullet_before_forward` → 공유메모리 상태 갱신 + 예측기 호출 → 필요 시 `libsmctrl_set_stream_mask` (**drain 없음**) |
| SM 분할 기전 | CUDA **green context**(부팅 시 (P,D) 조합을 미리 생성, 인덱스로 선택) | **libsmctrl** TPC 비트마스크(스트림 단위, 런타임 임의 변경). A100=54 TPC |
| MPS | 불필요 | **필수** (프로세스 간 공간 공유) |
| 프로세스 모델 | **1 프로세스**, 1 scheduler event loop가 prefill span ↔ decode step 교대 (옵션: `PDMUX_TRUE_DUAL_WORKER`로 host thread 2개) | **2 프로세스** (`is_bullet_prefill` / `is_bullet_decode`), 각자 scheduler 루프 |
| KV / req pool 공유 | 동일 프로세스 → 핸드오프 0 | IPC/shared-mem RPC (`memory_pool_rpc_v2`, `req_rpc`, `radix_cache_rpc`) + batch 전달 |
| decode CUDA graph | 운영점 = **ON** (decode step 전체를 한 번에) | decode 프로세스만 ON, **prefill 프로세스는 `cuda_graph_runner=None`** |
| decode도 layer로 쪼개나 | 기본 loop는 **아니오**(step 전체). `event_loop_pdmux_coord`(`PDMUX_LA_COORD=1`)에서만 `forward_split_decode`로 쪼갬 — **이 계열은 정본에서 반증됨** | 아니오 (layerwise는 prefill 전용) |
| 분할 정책 | 정적 threshold 표(`manual_divisions`) 또는 `PDMUX_SLO_SCHED`/`PDMUX_R2_POLICY` 컨트롤러 | `enable_sm_partition` OFF → 고정 TPC / ON → **외부 `.so` 예측기**(`predictor_param_file`)의 `set_adaptive_num_tpcs` |
| 대상 모델 | hybrid attention+SSM (NemotronH / Zamba2 / Falcon-H1 / Granite-4) | dense transformer (Llama3.1-70B, Qwen3-235B 등) |

---

## 2. layer-wise prefill 실행 경로 — 코드 대조

### 2.1 본 프로젝트: 전용 SPLIT_PREFILL forward mode

호출 사슬:

```
event_loop_pdmux                       src/multiplex/multiplexing_mixin.py:993
  └ run_batch(split_prefill_batch)     (forward_mode = SPLIT_PREFILL)
      └ ModelRunner.forward_split_prefill   sglang_engine_dev/.../model_runner.py:2709
          └ model.forward_split_prefill(ids, pos, fb, (start, end))
                                             src/models/nemotron_h.py:859
```

모델 측 본체(NemotronH 기준, `src/models/nemotron_h.py:859-891`):

```python
start, end = split_interval
if start == 0:
    forward_batch.hidden_states = model.embed_tokens(input_ids)
    forward_batch.residual = None
for i in range(start, end):
    forward_batch.hidden_states, forward_batch.residual = model.layers[i].forward(...)
if end == self.config.num_hidden_layers:
    hidden_states, _ = model.norm_f(...)
    return self.logits_processor(...)
```

특징:

- 중간 상태를 `forward_batch`에 **stash**한다. hybrid 모델이라 `(hidden, residual)` **쌍**을
  span 경계 너머로 넘겨야 했고, 그게 본 프로젝트가 모델 4종(`nemotron_h`, `zamba2`,
  `falcon_h1`, `granitemoehybrid`)에 각각 `forward_split_prefill`을 쓴 이유다
  (`src/models/*.py`, 각 1개씩). mamba conv/ssm state는 layer가 span 안에서 **정확히 한 번**
  실행되므로 추가 조치 없이 보존된다.
- attn metadata는 `split_index == 0`일 때만 `init_forward_metadata`
  (`model_runner.py:2709-2716`) — span마다 재계산하지 않는다.
- prefill 배치는 `split_prefill_batch`로 **끝까지 상주**하다가 마지막 span에서만
  `process_batch_result` → `running_batch`에 merge (`multiplexing_mixin.py:1265-1275`).

### 2.2 BulletServe: pipeline-parallel 배관 재사용

호출 사슬:

```
forward_thread_loop_bullet                tp_worker_overlap_thread.py:140
  └ if is_prefill and num_prefill_layers > 0:            :182
      for step in layerwise_prefill_step_generator():    :185  (tp_worker.py:380)
          update_bullet_before_forward(...)              :186  (tp_worker.py:221)
          forward_batch_generation(..., pp_proxy_tensors=prev)
          torch.cuda.current_stream().synchronize()      :193
          pp_proxy_tensors = PPProxyTensors(logits_output)
```

핵심은 `bullet/model_monkey_patch.py:34`의 `model_monkey_patch_layerwise_forward`:

```python
self.model.start_layer = pp_start + cur_step * layers_per_step
self.model.end_layer   = min(pp_start + (cur_step+1) * layers_per_step, pp_end)
self.model.pp_group.is_first_rank = (cur_step == 0) and pp_first_rank
self.model.pp_group.is_last_rank  = (cur_step == num_steps-1) and pp_last_rank
return self.origin_forward(*args, **kwargs)
```

즉 **"단일 GPU 위에서 시간축으로 흉내낸 pipeline parallel"**이다. 모델 코드는 이미
`start_layer/end_layer`와 `is_first/last_rank`를 존중하도록 짜여 있으므로 **모델 파일을
전혀 건드리지 않는다**. 설치는 `model_runner.py:308-313`에서 prefill 프로세스에만 적용된다.

### 2.3 이 차이가 실제로 의미하는 것

| | 본 프로젝트 | Bullet |
|---|---|---|
| 새 모델 지원 | 모델당 메서드 1개 추가 (hybrid는 residual/state 배선이 실제 작업량) | 무료 — 단, **PP 경로가 있는 모델에 한정** |
| 정확성 리스크 | 모델별 수작업 → 모델마다 검증 필요 | 전역 monkey-patch → `is_first/last_rank`를 참조하는 다른 코드와 상호작용 리스크 |
| hybrid(SSM) 적용 가능성 | 이미 됨 | **미확인**. mamba state를 `PPProxyTensors`로 넘기지 않으므로, hybrid 모델에 그대로 적용되는지는 코드만으로 단정 불가 |

---

## 3. span 크기 결정 정책 — 여기가 개념적으로 가장 다름

**본 프로젝트** (`multiplexing_mixin.py:1169-1195`):

```python
forward_count = max(1, split_forward_token_budget // extend_num_tokens)   # 기본 65536
```

전 캠페인 config가 `split_forward_token_budget: 65536`을 쓴다
(`results/*/pdmux_*.yml`). 결과적으로:

| prefill 배치 토큰 수 | span당 layer 수 | 52~54층 모델의 span 개수 |
|---|---|---|
| 512 | 128 → clip | **1 (= 쪼개지지 않음)** |
| 2048 | 32 | 2 |
| 4096 | 16 | ~4 |
| 8192 | 8 | ~7 |

즉 **"span당 GPU 작업량을 일정하게 유지"**하는 설계이고, 짧은 prefill에서는 layer-wise가
사실상 무효화된다(span 1개 = 통짜 forward). 옵션 `PDMUX_SLO_SPAN_TYPE`은 여기에 더해
span 경계를 attn/ssm **타입 경계**에 맞추는데, 이 계열은 정본에서 반증된 축이다.

**Bullet**: `num_prefill_layers` **상수**. 토큰 수와 무관하게 항상 `ceil(L/n)` 스텝.
"span당 반응 지연을 일정하게 유지"하는 설계에 가깝다. 단 기본값 `-1`(비활성)이고
저장소의 benchmark/scripts/docs 어디에서도 이 플래그를 쓰지 않는다 —
**릴리스에서는 실험용 노브**로 보는 게 맞다.

부수적으로 Bullet은 prefill 배치 자체도 토큰으로 캡한다:
`bullet_max_prefill_tokens = 1664`, `bullet_max_concurrent_tokens = 2048`
(`server_args.py:290-291`, 적용은 `scheduler.py:1696`). 즉 Bullet의 실질 granularity는
**(배치 토큰 캡) × (layers/step)** 두 축의 곱이다. 우리는 배치 캡이 아니라
토큰 예산 하나로 layer 수를 유도한다.

---

## 4. layer 경계에서 무슨 일이 일어나는가 (= 두 시스템의 진짜 분기점)

### 본 프로젝트 — 경계는 "인터리브 지점 + 전환 지점"

단일 event loop이므로, prefill span이 끝나야 다음 decode step이 issue된다.
span 경계에서:

1. 컨트롤러 평가 (`_slo_decide_idx` / `_r2_decide_idx`, `multiplexing_mixin.py:1072-1105`)
2. 인덱스가 **바뀔 때만** `prefill_stream.synchronize(); decode_stream.synchronize()`
   → `set_current_stream_idx(tgt)` → `update_decode_attn_backend(tgt)`
3. green context stream group은 **부팅 시 고정 생성**
   (`pdmux_context.py:104-141`, `create_greenctx_stream_by_value`) → 전환 = 미리 만든 조합 선택

즉 전환은 **동기적이고 비싸다**(양 stream drain). 그래서 컨트롤러가
"평가는 매 span, drain은 변할 때만"으로 설계돼 있다.

### Bullet — 경계는 "재-provisioning 지점"일 뿐

`tp_worker.py:221-268`에서 매 step:

```python
self.shared_mng.rem_layers = model.end_layer - model.start_layer
num_tpcs, policy = shared_mng.set_adaptive_prefill_num_tpcs(longest_queue_ms)
if num_tpcs != self.last_num_tpc:
    self.smctrl.set_stream_mask(self.forward_stream, 0, num_tpcs,
                                reversed=is_bullet_decode and not disable_decode_tpc_reverse)
```

- `set_stream_mask`은 **비트마스크 write 1회**(`sm_controller.py:122`), drain 없음.
  이후 launch되는 커널에만 적용된다.
- prefill은 `[0, n)` TPC, decode는 `reversed=True`로 **반대쪽 끝에서** 잡아
  두 마스크가 서로 겹치지 않게 만든다 (green context처럼 구조적으로 disjoint가
  보장되는 게 아니라, **양 끝에서 잡는 관례**로 보장).
- 상대 phase가 idle이면 `TOTAL_TPCS`로 즉시 복귀(`shared_mng.py:191-214`) — 우리 쪽
  `adjust_stream_groups`의 "decode 비었으면 idx 0" 폴백과 같은 역할.

**따라서 Bullet에서 layer-wise prefill의 목적은 "decode에게 GPU를 양보"가 아니다**
(별도 프로세스라 애초에 양보가 필요 없음). 목적은 **긴 prefill 도중에도 SM 배분을
갱신할 수 있는 결정 지점을 만드는 것**이다. 반대로 우리 쪽에서 layer-wise는
**decode step이 끼어들 수 있는 유일한 틈**이다. 같은 메커니즘, 다른 존재 이유.

---

## 5. 프로세스/동시성 모델

| | 본 프로젝트 | Bullet |
|---|---|---|
| 구조 | 1 scheduler 프로세스, prefill/decode가 같은 Python 루프 | prefill 엔진 / decode 엔진 **2 프로세스** (MPS) |
| 동시성 | CUDA stream 2개 + (옵션) host thread 2개(`TrueDualWorkerRuntime`) | OS 프로세스 2개, 서로 blocking 없음 |
| 매 iteration 동기화 | `decode_stream.synchronize()` 매 루프 (`multiplexing_mixin.py:1240`) | 각자 자기 루프. layerwise 스텝마다 `current_stream().synchronize()`는 **prefill 프로세스 내부에만** 영향 |
| 조율 채널 | 프로세스 내 파이썬 상태 | `SharedManager`(공유 numpy 배열: prefill_size, decode_size, decode_total_context, prefill/decode_num_tpcs, rem_layers 등) + ZMQ RPC |
| KV 핸드오프 | 없음 | prefill→decode batch 전송(`pd_client.send_batch_v2`, `scheduler.py:1894-`), KV/req pool은 IPC 공유 |
| 시간축 제어 | 없음 | decode 측 `sleep_factor` / `max_sleep_ms`, `dont_decode_when_queueing` (= "spatial-**temporal**"의 temporal 절반) |

이 축이 우리 negative 결과(`CONSENSUS` HE0: 단일-GPU 동적 제어가 best static을 못 넘음,
死因 = positioning + entanglement)와 직접 맞닿는다. 우리 얽힘의 상당 부분은
**같은 루프에서 두 phase를 교대**시키는 데서 온다. Bullet은 그 얽힘을 프로세스 분리로
아예 제거하고, 대신 KV/req 공유와 MPS 의존을 값으로 치른다.
**단, 이건 코드 구조 관찰이지 "Bullet이면 HE0이 뒤집힌다"는 측정 근거가 아니다.**

---

## 6. 우리에게만 있는 것 / Bullet에만 있는 것

**본 프로젝트에만**
- hybrid(attention+SSM) 4모델의 split-prefill 이식 + mamba 상태/`update_decode_attn_backend`
  파티션별 backend group
- decode도 layer-type 창으로 쪼개는 `event_loop_pdmux_coord` / `forward_split_decode`
  (**정본에서 반증된 축** — coordinated per-type TPOT 42→124ms)
- green-context 실측 readout(`green_readout.py`), telemetry/campaign 배관
  (`telemetry.py`, `PDMUX_RUN_ID`/`PDMUX_WORKLOAD_ID`), R2 정책(`fixed|generic|hybrid`),
  sticky partition, HOL-blocking 프로브
- 컨트롤러 자체 비용 계측(`SLO-CTLCOST`), switch 시 안전성 검사(`safe_to_switch`)

**Bullet에만**
- 프로세스 분리 + IPC KV/radix/req pool 공유, MPS 기반 공간 공유
- libsmctrl TPC 마스킹(런타임 임의 변경, drain 없음) + `reversed` 상보 마스킹
- 학습/해석 기반 duration 예측기 `predic_duration` / `set_adaptive_num_tpcs`
  → **⚠️ 이 `.so`는 저장소에 없다**(`predictor_param_file`로 외부 주입,
  `shared_mng.py:118-125`). 즉 **Bullet의 적응형 SM 정책은 공개 코드로 재현 불가**이고,
  `enable_sm_partition` 없이는 `fixed_prefill_tpcs/fixed_decode_tpcs` 고정 분할로 동작한다.
- decode 측 temporal 제어(sleep/양보), prefill 대기 시 decode 억제
- GPU timing 계측(`timing.py`, `*_timing.py` 모델 변형), predictor용 로그 덤프

---

## 7. 코드 독해 중 관찰된 세부 (검증 안 함, 주장 아님)

1. **`rem_layers`가 한 스텝 stale해 보인다.** `tp_worker.py:233`은
   `model.end_layer - model.start_layer`를 읽는데, 이 값은 **직전 forward**의 monkey-patch가
   설정한 값이다(현재 스텝의 창은 아직 설정 전). `layers_per_step`이 균일하면 마지막
   partial step을 빼고 값이 같아 실질 영향은 작을 수 있으나, 예측기 입력으로 쓰이므로
   기록해 둔다. **실행 검증은 하지 않았다.**
2. **`num_prefill_layers`는 릴리스 기본 비활성**이고 저장소 benchmark/docs에서 쓰이지 않는다.
   따라서 "Bullet의 layer-wise prefill"을 논문 주장으로 인용할 때는, 공개 코드에서
   기본 경로가 아님을 함께 적어야 한다.
3. **CUDA 버전 제약**: libsmctrl은 `CUDA <= 12.6`(README). 본 환경은 **CUDA 13**이므로
   Bullet 기전을 이 노드에서 그대로 재현하는 건 별도 검토가 필요하다.
4. Bullet quickstart는 `--disable-radix-cache`를 쓴다 — prefix cache와 프로세스 분리의
   상호작용을 피하는 것으로 보인다(코드상 `radix_cache_rpc.py`는 존재).

---

## 8. 후속으로 할 수 있는 것 (제안, 미실행)

- **비용 0**: 본 문서의 대조를 `reports/system_vs_engine_vs_sim.md`(코드 아닌 논문 기반으로
  작성된 기존 대조)와 병합/정정. 특히 그 문서의 "Bullet: bitmask write, drain 없음",
  "별도 프로세스 + cudagraph decode" 서술은 **코드로 확인됨**(§4, §5) — 근거를 논문에서
  코드로 승격 가능.
- **저비용**: `split_forward_token_budget`를 낮춰 span 개수를 늘렸을 때의 거동
  (= Bullet의 고정 layers/step에 가까운 설정)을 우리 엔진에서 스윕. 단, 정본 방법론 게이트
  (변화 trace, n≥4, metric cliff)를 지켜야 하며 **현재 정책 순위를 바꾸는 실험이 아님**.
- **주의**: 이 대조는 "Bullet이 더 낫다/우리가 더 낫다"의 근거가 **아니다**. 두 시스템은
  기판·프로세스 모델·모델 종류가 모두 달라 직접 비교 불가.
