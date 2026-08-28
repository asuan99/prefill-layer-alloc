# 구현체 대조: 본 프로젝트의 layer-wise(split) prefill vs. MuxWise · BulletServe

- 작성일: **2026-08-28**
- 증거 수준: **코드 독해(static read)만**. 새 측정·성능 판정 **없음**. 성능/정책 주장은
  전부 정본(`PROJECT_STATUS.md`, `reports/CONSENSUS.md`)을 따른다.
- 대조 스냅샷
  - 본 프로젝트: `workspace/engine-port/src/` + editable tree
    `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/` (SGLang **v0.5.10** 기반)
  - **MuxWise**: Zenodo `10.5281/zenodo.18062118` (`muxwise.zip`, md5 `89b9fb84…`)
    = `github.com/ykcombat/sglang` **@ eeac5148** (SGLang **v0.5.3rc0**, 브랜치 `slo_config`).
    **이미 저장소에 있음**: `workspace/engine-port/external/muxwise-zenodo.zip`(md5 동일) ·
    `external/muxwise/`(같은 커밋). `diff -rq` 결과 **차이 0**.
  - **BulletServe**: `github.com/zejia-lin/BulletServe`. 저장소 클론
    `external/bullet/` **@ 371cb2f**("add AE") = GitHub main `445afae2`의 자손.
    layer-wise 관련 파일은 두 스냅샷이 동일(diff 0). ASPLOS 2026 / arXiv 2504.19516.

---

## 0. 한 문단 요약

**MuxWise는 대안이 아니라 우리 코드의 직계 조상이다.** Zenodo 아티팩트의
`srt/multiplex/pdmux_context.py`는 우리 dev tree 파일과 **2줄**(import 위치 1, 에러 문구 1)만
다르고, `srt/multiplex/multiplexing.py`(195줄)는 우리 `multiplexing_mixin.py`(1716줄)의
골격 그 자체다. 즉 "layer-wise prefill을 어떻게 하는가"는 **MuxWise ≡ 본 프로젝트**이고,
차이는 전부 우리가 얹은 델타(hybrid 모델 이식 · 컨트롤러 · 계측)다.
**BulletServe만이 진짜 다른 설계**다 — 같은 골격(prefill을 layer로 쪼개고 경계마다 SM 재배분)을
갖되, 쪼개는 구현(PP 배관 재사용), 존재 이유(양보가 아니라 재-provisioning),
경계 비용(bitmask write vs. green-ctx drain), 프로세스 모델(2 프로세스 vs 1)이 모두 다르다.

---

## 1. 3자 요약표

| 축 | **MuxWise** (eeac5148, v0.5.3rc0) | **본 프로젝트** (v0.5.10 base) | **BulletServe** (371cb2f) |
|---|---|---|---|
| 관계 | **우리 base의 upstream**(SGLang PR #10692) | MuxWise + 델타 | 독립 fork (SGLang 기반) |
| SM 분할 기전 | CUDA **green context**, 부팅 시 (P,D) 조합 생성 | **동일** | **libsmctrl** TPC 비트마스크 (+MPS 필수) |
| layer-wise prefill 진입점 | `ForwardMode.SPLIT_PREFILL` + 모델별 `forward_split_prefill` | **동일** | `model.forward` **monkey-patch**로 `start_layer/end_layer` 이동 |
| 중간 활성값 | `forward_batch.hidden_states/residual` | **동일**(hybrid는 쌍 전달) | `PPProxyTensors` (pipeline-parallel proxy 재사용) |
| span 크기 | `split_forward_token_budget // extend_num_tokens` (65536) | **동일 공식·동일 값** | `num_prefill_layers` **상수** (기본 -1 = 비활성) |
| 분할 선택 정책 | `adjust_stream_groups` = decode_bs threshold 표<br>**코드 주석: `# TODO(jason-fxz): This is a temporary demo`** | 같은 함수 + SLO/R2 컨트롤러·sticky·fixed idx | 외부 `.so` 예측기(`set_adaptive_num_tpcs`) 또는 고정 TPC |
| 경계에서의 전환 비용 | 양 stream `synchronize()` (drain) | **동일** | `set_stream_mask` **비트마스크 write 1회, drain 없음** |
| 프로세스 모델 | 1 프로세스 1 event loop | 동일 (+옵션 host thread 2개) | **2 프로세스**(prefill/decode) + IPC KV/req pool |
| decode CUDA graph | 파티션별 그래프 세트(`graphs[f"{idx}_{bs}"]`) | **동일**(운영점 ON) | decode 프로세스만 ON, prefill은 `cuda_graph_runner=None` |
| 지원 모델 | dense 10종 (llama/qwen/gemma/qwen2_moe/apertus…) | + **hybrid 4종**(nemotron_h·zamba2·falcon_h1·granitemoehybrid) ← 본 프로젝트 기여 | dense (Llama3.1-70B, Qwen3-235B) |
| decode도 layer 분할 | 아니오 | `event_loop_pdmux_coord`만 (**정본에서 반증**) | 아니오 |
| 비교 baseline | **chunked prefill**(`--chunked-prefill-size N --enable-mixed-chunk`) | fused(P1) | chunked prefill |
| 평가 하드웨어/모델 | H200 132SM / CodeLlama-34B | A100 108SM / hybrid 1.2B–8B | A100·A800·H100·H20 / 70B~235B |

---

## 2. MuxWise 대조 — "우리 델타 목록"

### 2.1 동일성 근거 (재도출 불필요하도록 기록)

- zip 최상위에 커밋 sha 파일명 `eeac5148f456428a05823c330e1d40afb0ef2438`가 들어 있다.
- `pdmux_readme.md`는 **SGLang PR 본문**(Motivation: "Support PD-Multiplexing",
  Modifications에 `srt/multiplex/multiplexing.py`·`pdmux_context.py`·
  `forward_batch_split_prefill()`·"CudaGraphRunner: Record and replay a set of CUDA graphs
  for each sm partition" 명시).
- `diff external/muxwise-zenodo/sglang-slo_config external/muxwise` → **0 차이**
  (Zenodo 아티팩트 = 저장소 클론).
- `diff muxwise/…/pdmux_context.py  dev_tree/…/pdmux_context.py` → **2 hunk**
  (`from sgl_kernel import spatial`의 위치, `"must greater than 3"` → `"must be >= 3"`).
- upstream 최신(`external/sglang-latest`)의 `multiplex/multiplexing_mixin.py`도 **218줄**로
  여전히 MuxWise 데모 루프 + `init_pdmux` 이동 수준. 즉 **upstream PD-mux ≈ MuxWise ≈ 우리 base**.

### 2.2 layer-wise prefill 자체 — 차이 없음

MuxWise `multiplex/multiplexing.py:130-152`와 우리
`src/multiplex/multiplexing_mixin.py:1169-1195`의 span 계산은 **문자 그대로 같은 코드**다:

```python
forward_count = max(1, split_forward_token_budget // extend_num_tokens)  # 65536
next_split_index = min(split_index + forward_count, num_hidden_layers)
```

우리 전 캠페인 config가 `split_forward_token_budget: 65536`을 그대로 쓴다 —
**MuxWise의 H200 예시 config와 같은 값**. 결과적인 granularity:

| prefill 배치 토큰 | span당 layer | 52~54층 모델 span 수 |
|---|---|---|
| 512 | 128 → clip | **1 (쪼개지지 않음)** |
| 2048 | 32 | 2 |
| 4096 | 16 | ~4 |
| 8192 | 8 | ~7 |

즉 **"짧은 prefill에서는 layer-wise가 사실상 무효"**라는 성질은 우리가 만든 게 아니라
**upstream에서 상속한 것**이다.

### 2.3 우리가 얹은 것 (= 실제 델타)

| 항목 | MuxWise | 본 프로젝트 |
|---|---|---|
| `multiplex/` 코드량 | 358줄 (2파일) | **4039줄** (8파일) |
| 이벤트 루프 | `event_loop_pdmux` 195줄 | + `event_loop_pdmux_coord`, R2/SLO 분기, dual-worker 제출, HOLB 훅 |
| 신규 모듈 | — | `dual_worker.py`(625) `holb_probe.py`(600) `controller.py`(306) `profile.py`(317) `green_readout.py`(167) `telemetry.py`(144) |
| 모델 | dense만 | hybrid 4종 `forward_split_prefill` 이식 (`src/models/*.py`) |
| 정책 | decode_bs threshold **데모** | `PDMUX_SLO_SCHED`(TPOT-EMA 피드백) · `PDMUX_R2_POLICY=fixed\|generic\|hybrid` · sticky partition |
| 계측 | 없음 | telemetry jsonl(run/workload id), green-ctx readout, HOL-blocking 프로브, 컨트롤러 자기비용(`SLO-CTLCOST`) |
| 안전장치 | 없음 | thread-local role 패치, `safe_to_switch()`, 미정의 플래그 조합 **부팅 거부** |
| 벤치 규율 | rate 1..25 단조 스윕 1회 | 변화 trace(rate 3↔12, 3라운드), n≥4, goodput = TTFT ∧ ITL-p95 |

### 2.4 연구적으로 의미 있는 한 줄

MuxWise의 분할 선택 함수에는 저자가 직접 **`# TODO(jason-fxz): This is a temporary demo`**라고
적어 두었다. 우리 dynamic 컨트롤러 라인(SLO-aware / binding-first / gate / R2)은 **바로 그
placeholder를 대체하려는 시도**였고, 정본 결론(HE0, n≥4, 5.4σ)은 *그 시도가 best
decode-heavy static을 넘지 못한다*는 것이다. 즉 **upstream이 임시라고 표시한 정적 정책이
우리 기판·워크로드에서는 여전히 이기는 쪽**이다. (범위 한정: A100 green-ctx · hybrid 모델 ·
우리 trace. MuxWise의 H200/CodeLlama-34B 조건으로 이식 금지.)

### 2.5 참고 — MuxWise의 레퍼런스 운영 곡선 (H200 132 SM)

```
manual_divisions:  [112,20,1] [104,28,5] [96,36,10] [80,52,15] [64,68,20] [56,76,25]
                    decode share 15% → 58%,  decode_bs threshold 1 → 25
```
loogle(long-context)용은 3행 `[80,52,1] [64,68,5] [56,76,10]`으로 **decode 쪽을 더 크게** 잡는다.
우리 A100 격자(decode_states {16,24,34,44} = 15%~41%)와 범위가 겹치되, MuxWise는
**decode 58%까지** 올라가는 행을 둔다.

### 2.6 ⚠️ 대조 중 발견 — config 3번째 필드 의미 불일치 (검증 필요, 결과 영향 미확인)

`manual_divisions`의 3번째 값의 **엔진 의미는 `decode_bs_threshold`**이고, 선택 루프는
break 없이 마지막 만족 행을 고른다(MuxWise 원본 코드 그대로 상속):

```python
for i in range(len(manual_divisions)):
    _, _, threshold = manual_divisions[i]
    if decode_bs >= threshold:
        stream_idx = i + 1        # break 없음 → 조건 만족하는 최대 i
```

그런데 우리 config 53개 중 **10개**에서 이 필드가 **idle SM 수**(= 108 − P − D)로 채워져 있다:

| config | 행 | 3번째 값 | 108−P−D |
|---|---|---|---|
| `s0_deconfound/pdmux_p16_d16.yml`, `s8_scaleup/pdmux_p16_d16.yml`, `s8p_prefill/pdmux_pf16_d16.yml` | [16,16,**76**] | 76 | 76 |
| `…/pdmux_p16_d24.yml` (×2) | [16,24,**68**] | 68 | 68 |
| `s8p_prefill/pdmux_pf24_d16.yml` | [24,16,**68**] | 68 | 68 |
| `…/pdmux_p16_d44.yml` (×2), `s8p_prefill/pdmux_pf44_d16.yml` | [16,44,**48**] / [44,16,**48**] | 48 | 48 |
| `s8_scaleup/pdmux_p16_d54.yml` | [16,54,**38**], [16,44,**48**] | 38 / 48 | 38 / 48 |

10개 전부 세 값의 합이 정확히 108이다. 해당 yml의 주석도 "…, **68 idle**"처럼 idle SM으로
서술한다. 두 가지 귀결이 따라온다 — **manual-division 선택 경로가 실제로 실행될 때만**:

1. 단일 행 config에서 `decode_bs < threshold`면 `stream_idx`가 **바인딩되지 않아
   `UnboundLocalError`**가 난다(else 분기 없음).
2. `pdmux_p16_d54.yml`은 decode_bs ≥ 48에서 guard-satisfier 행 `[16,44]`로 넘어간다 —
   decode SM이 batch 증가에 따라 **줄어드는** 역방향 선택.

**도달성 추적(코드 독해)**: 해당 3개 캠페인은 전부 `PDMUX_R2_POLICY=fixed`이고
`PDMUX_SLO_SCHED`·`PDMUX_STICKY_PARTITION`은 설정하지 않는다. 그러면 파티션은 루프의
R2 분기(`_r2_decide_idx`)가 잡고, `adjust_stream_groups`의 manual-division 루프에 도달하려면
`running_batch` 非공 ∧ `split_prefill_batch` 非None ∧ `adjust_stream_group=True` ∧
R2 분기 조건 거짓(= `wait_prefill_kernel_done`)이 동시에 성립해야 한다. tp_size=1에서는
allreduce가 항상 만장일치라 이 조합이 성립하기 어렵다. `pdmux_p16_d54.yml` 주석도 guard 행이
"never selected"임을 전제한다.

**따라서 현재 판단: 결과 오염 가능성은 낮아 보이나 확인된 것은 아니다.** 이 절은
"발견 기록"이지 판정이 아니다. 필요한 후속(제안): (a) 해당 캠페인 telemetry에서 realized
partition이 의도한 (P,D) 하나로만 관측되는지 재확인 → **result-analyst**, (b) 그 결과로
기존 결론에 파급이 있는지 → **claims-auditor**. 지금 정본을 고칠 근거는 아니다.

---

## 3. BulletServe 대조 — 유일하게 다른 설계

### 3.1 layer-wise prefill 실행 경로

**본 프로젝트/MuxWise** — 전용 forward mode:

```
event_loop_pdmux                      src/multiplex/multiplexing_mixin.py:993
 └ run_batch(split_prefill_batch)     (forward_mode = SPLIT_PREFILL)
    └ ModelRunner.forward_split_prefill      model_runner.py:2709
       └ model.forward_split_prefill(ids, pos, fb, (start, end))
                                             src/models/nemotron_h.py:859
```
```python
start, end = split_interval
if start == 0:
    forward_batch.hidden_states = model.embed_tokens(input_ids); forward_batch.residual = None
for i in range(start, end):
    forward_batch.hidden_states, forward_batch.residual = model.layers[i].forward(...)
if end == num_hidden_layers:
    return self.logits_processor(...)          # norm_f 후 logits
```
attn metadata는 `split_index == 0`일 때만 `init_forward_metadata`(span마다 재계산 없음).
hybrid라 `(hidden, residual)` **쌍**을 span 너머로 넘겨야 했고, 그래서 모델 4종에 각각 이식했다.
mamba conv/ssm state는 각 layer가 span 안에서 **정확히 한 번** 실행되므로 추가 조치 불필요.

**BulletServe** — pipeline-parallel 배관 재사용 (`bullet/model_monkey_patch.py:34`):

```python
self.model.start_layer = pp_start + cur_step * layers_per_step
self.model.end_layer   = min(pp_start + (cur_step+1) * layers_per_step, pp_end)
self.model.pp_group.is_first_rank = (cur_step == 0) and pp_first_rank
self.model.pp_group.is_last_rank  = (cur_step == num_steps-1) and pp_last_rank
return self.origin_forward(*args, **kwargs)
```
호출부는 `tp_worker_overlap_thread.py:182-195` → `tp_worker.py:380`
(`layerwise_prefill_step_generator`), 중간 활성값은 `PPProxyTensors`.
**모델 파일 무수정**(PP 경로가 있는 모델 한정). 설치는 prefill 프로세스에만
(`model_runner.py:308-313`).

| | 본 프로젝트 | Bullet |
|---|---|---|
| 새 모델 지원 | 모델당 메서드 1개 (hybrid는 residual/state 배선이 실작업) | 무료 — 단 **PP 지원 모델 한정** |
| 정확성 리스크 | 모델별 수작업 → 모델마다 검증 | 전역 monkey-patch → `is_first/last_rank` 참조 코드와의 상호작용 |
| hybrid(SSM) | 이미 됨 | **미확인**(mamba state를 proxy로 넘기지 않음) |

### 3.2 span 크기 정책

- 우리/MuxWise: **토큰 예산 기반 동적** → "span당 GPU 작업량 일정".
- Bullet: `num_prefill_layers` **상수**, `ceil(L/n)` → "span당 반응 지연 일정".
  **기본 -1(비활성)**, 저장소 benchmark/scripts/docs 어디서도 쓰지 않음 → 릴리스에선
  실험용 노브. 대신 배치 자체를 `bullet_max_prefill_tokens=1664` /
  `bullet_max_concurrent_tokens=2048`로 캡(`server_args.py:290-291`,
  적용 `scheduler.py:1696`). 즉 Bullet의 실질 granularity는 **(배치 토큰 캡)×(layers/step)**.

### 3.3 layer 경계에서 무슨 일이 일어나는가 (진짜 분기점)

- **우리/MuxWise**: 단일 event loop → prefill span이 끝나야 다음 decode step이 issue된다.
  경계에서 (i) 컨트롤러 평가, (ii) **인덱스가 바뀔 때만** 양 stream `synchronize()` 후
  green-ctx stream group 교체 + `update_decode_attn_backend`. 전환은 **동기적이고 비싸다**.
- **Bullet**(`tp_worker.py:221-268`): 매 step 공유메모리 상태 갱신 → 예측기 호출 →
  `libsmctrl_set_stream_mask` **비트마스크 write 1회, drain 없음**. prefill은 `[0,n)` TPC,
  decode는 `reversed=True`로 **반대쪽 끝**을 잡아 겹치지 않게 한다(구조적 disjoint 보장이
  아니라 관례). 상대 phase가 idle이면 `TOTAL_TPCS`로 즉시 복귀.

**결론**: Bullet에서 layer-wise prefill의 목적은 "decode에게 양보"가 **아니다**(별도
프로세스라 양보가 불필요). 목적은 **긴 prefill 도중에도 SM 배분을 갱신할 결정 지점**을
만드는 것. 우리 쪽에서 layer-wise는 **decode step이 끼어들 수 있는 유일한 틈**이다.
같은 메커니즘, 다른 존재 이유.

### 3.4 프로세스/동시성 모델

| | 본 프로젝트/MuxWise | Bullet |
|---|---|---|
| 구조 | 1 scheduler 프로세스 | prefill/decode **2 프로세스**(MPS) |
| 매 iteration 동기화 | `decode_stream.synchronize()` 매 루프(`multiplexing_mixin.py:1240`) | 각자 자기 루프 |
| 조율 채널 | 프로세스 내 파이썬 상태 | `SharedManager` 공유 numpy 배열(prefill_size, decode_size, decode_total_context, prefill/decode_num_tpcs, rem_layers) + ZMQ RPC |
| KV 핸드오프 | 없음 | prefill→decode batch 전송 + KV/req/radix pool IPC 공유 |
| 시간축 제어 | 없음 | decode `sleep_factor`/`max_sleep_ms`, `dont_decode_when_queueing` |

이 축이 정본 HE0(死因 = positioning + **entanglement**)와 직접 맞닿는다. 우리 얽힘의 상당 부분은
**같은 루프에서 두 phase를 교대**시키는 데서 온다. Bullet은 그 얽힘을 프로세스 분리로 제거하고
KV 공유·MPS 의존을 값으로 치른다. **단, 이건 구조 관찰이지 "Bullet이면 HE0이 뒤집힌다"는
측정 근거가 아니다.**

### 3.5 Bullet에만 / 우리에만

**Bullet에만** — 프로세스 분리 + IPC pool 공유, libsmctrl 런타임 마스킹(drain 없음),
`reversed` 상보 마스킹, decode 시간축 양보, GPU timing 계측(`timing.py`, `*_timing.py`),
그리고 **duration 예측기**. ⚠️ 이 예측기 `.so`는 **저장소에 없다**(`predictor_param_file`로
외부 주입, `shared_mng.py:118-125`) → **Bullet의 적응형 SM 정책은 공개 코드로 재현 불가**이고,
`enable_sm_partition` 없이는 `fixed_prefill_tpcs/fixed_decode_tpcs` 고정 분할로 동작한다.

**우리에만** — hybrid 4모델 split-prefill + 파티션별 decode attn backend, decode도 layer-type
창으로 쪼개는 `event_loop_pdmux_coord`(**반증된 축**), telemetry/green readout/HOLB 프로브/
R2 정책/sticky partition, 컨트롤러 자기비용 계측, 전환 안전성 검사.

---

## 4. 코드 독해 중 관찰 (검증 안 함, 주장 아님)

1. **Bullet `rem_layers`가 한 스텝 stale해 보인다.** `tp_worker.py:233`이 읽는
   `model.end_layer - model.start_layer`는 **직전 forward**의 monkey-patch가 세팅한 값이다.
   균일 `layers_per_step`이면 마지막 partial step 외엔 값이 같아 실질 영향은 작을 수 있으나,
   예측기 입력이므로 기록해 둔다.
2. **Bullet `num_prefill_layers` 기본 비활성** — 논문 주장으로 인용할 때 공개 코드에서
   기본 경로가 아님을 함께 적어야 한다.
3. **CUDA 버전 제약**: libsmctrl은 `CUDA <= 12.6`(README). 본 환경은 **CUDA 13** →
   Bullet 기전을 이 노드에서 재현하려면 별도 검토 필요.
4. Bullet quickstart는 `--disable-radix-cache` 사용.
5. MuxWise 아티팩트 디렉터리명이 `sglang-slo_config`지만 **SLO 로직은 없다**
   (`grep -w SLO python/sglang/srt/` → stock 메트릭 `max_running_requests_under_SLO` 1건뿐).
   브랜치명일 뿐이므로 "MuxWise에 SLO 정책이 있다"고 인용하면 안 된다.
6. MuxWise/우리 공통: `--disable-overlap-schedule` 전제(그들의 `start_pdmux.sh`).

---

## 5. 후속 제안 (미실행)

- **비용 0**: 기존 `reports/system_vs_engine_vs_sim.md`(논문 기반 대조)의 Bullet 서술
  ("bitmask write·drain 없음", "별도 프로세스 + cudagraph decode")는 **코드로 확인됨** →
  근거를 논문에서 코드로 승격 가능. 반대로 "MuxWise"를 *외부 비교 대상*으로 서술한 대목이
  있다면 **"우리 base의 upstream"**으로 정정해야 한다(§2.1).
- **낮은 비용, 우선**: §2.6의 config 3번째 필드 건을 telemetry realized-partition으로 확인
  (result-analyst). 오염이 없다면 그 사실 자체를 기록해 재발 방지 규율로 남긴다.
- **선택**: `split_forward_token_budget`를 낮춰 span 수를 늘린 스윕(= Bullet의 고정
  layers/step에 근접). 정본 방법론 게이트(변화 trace·n≥4·metric cliff) 준수 필수이며,
  **현재 정책 순위를 바꾸는 실험이 아니다**.
- **주의**: 이 대조는 우열 근거가 아니다. MuxWise는 조상이라 비교 대상이 아니고,
  Bullet은 기판·프로세스 모델·모델 종류가 모두 달라 직접 비교 불가.
