# P0 Triage — Layer-type-aware PD-Multiplexing 실엔진 이양 기반 판정

작성일: 2026-07-02 · 작성: P0 트리아지(Claude Code) · 상태: **사람 판정 대기(게이트)**
분석 대상: MuxWise(`ykcombat/sglang@slo_config`) · Bullet(`zejia-lin/BulletServe@asplos26ae`) · 최신 SGLang(`sgl-project/sglang@926140d789`)
증거 로그: [triage/notes.md](../triage/notes.md) (파일·심볼·라인 인용 원본) · T5 산출: [triage/t5_greenctx/](../triage/t5_greenctx/)

라벨 규약: `[measured]` = 코드/실행에서 직접 확인 · `[derived]` = 그로부터의 추론. **라벨 없는 정량치 없음.**

---

## 0. 한 줄 결론 (판정 권고 — 최종 결정은 사람)

**전략 B(최신 SGLang 기반)를 권고한다. 단, P0가 밝힌 사실이 A/B 프레이밍 자체를 바꾼다: 최신 SGLang은 MuxWise의 PD-multiplexing 3모듈을 이미 upstream으로 병합해 유지 중이고(PR #11592/#12275), Zamba2가 필요로 하는 Mamba2 SSD 인프라도 완비돼 있다. 즉 B의 "3모듈 forward-port" 비용은 사실상 0이며(수작업 이식이 아니라 upstream 채택), 남는 일은 (a) Zamba2 모델 포트와 (b) 우리의 layer-type-aware 확장뿐이다.** 전략 A(구 MuxWise 0.5.3rc0에 백포트)는 없는 Mamba2 인프라를 백포트해야 하므로 B에 대해 열위(dominated)다.

> ⚠️ 판정을 뒤집을 수 있는 유일한 조건: **MuxWise 논문 수치의 정확 재현**이 목표라면 동결 아티팩트(`slo_config`@torch2.6 docker)가 그 비교에는 필요하다(§6). 확장을 *구축*하는 목적이면 B가 우월.

---

## 1. 요약 표 — 전략 A vs B 비용 비교

| 비용 항목 | 전략 A: 구 MuxWise(0.5.3rc0)에 백포트 | 전략 B: 최신 SGLang 채택 | 근거 |
|---|---|---|---|
| **PD-mux 3모듈 확보** | 이미 base에 포함(MuxWise 그 자체) — 0 | **이미 upstream에 포함·유지 중 — 0** (forward-port 불요) | `[measured]` §3.1, §4.1 |
| **mamba2/SSD 인프라 백포트** | **필요(大)**: `MambaMixer2`+`ops/`(ssd_*)+`Mamba2AttnBackend`+NemotronH-scaffold가 0.5.3rc0에 **부재** → 신 SGLang에서 백포트 | 불요 — 최신에 완비 | `[measured]` §3.2, §4.2 |
| **Zamba2 모델 포트 표면** | 위 백포트 완료 *후* 동일 | NemotronH(temporal) 템플릿 + `MambaMixer2` 재사용; 신규=shared-attn+LoRA+ABAB | `[measured]` §4.2 |
| **하드코딩 수정 지점** | H200 하드코딩 **없음**(device-query); manual_divisions YAML만 A100 재튜닝 | 동일(같은 코드가 upstream) | `[measured]` §3.3 |
| **upstream 유지보수 승계** | 없음(0.5.3rc0 동결, 9209 커밋 뒤처짐) | **있음**(최신, 지속 유지) | `[measured]` §3.1 |
| **주요 리스크** | 백포트 충돌(9개월/9209커밋 격차의 인프라 이식) | Zamba2 shared-block/LoRA 정확 이식; green-ctx perf on torch>2.6(공통) | `[derived]` §5 |
| **총평** | B에 대해 **열위**(추가 백포트 부담, 유지보수 승계 없음) | **권고** | `[derived]` §6 |

---

## 2. T1 — 코드베이스 확보 (로그인 노드 glogin01, 네트워크 정상)

- `[measured]` **MuxWise**: `ykcombat/sglang` branch `slo_config`, HEAD `eeac5148f`. 클론 `external/muxwise`. 리포에 다수 `pdmux_*` 브랜치(`pdmux_scheduler/_cuda_graph/_attention/_comm`, `slo_scheduler`, `greenctx_stream`, `test_green_ctx` 등) — 이후 upstream화된 작업의 흔적.
- `[measured]` **Zenodo 18062118** = 단일 `muxwise.zip`(10 MB, **CC-BY-4.0**, "MuxWise: Artifact"). 압축해제 `sglang-slo_config/`. `python/sglang` 트리가 github `slo_config` HEAD와 **바이트 동일**(`diff -rq`=0) → github 클론이 정본(git 히스토리 보유). `external/muxwise-zenodo/`.
- `[measured]` **Bullet**: `zejia-lin/BulletServe`, `external/bullet`, branch `asplos26ae`(ASPLOS'26 AE) 체크아웃.
- `[measured]` **최신 SGLang**: `sgl-project/sglang`, `external/sglang-latest`, HEAD `926140d789`(`git describe`=`gateway-v0.3.1-6011-g926140d789`).
- `[measured]` **LICENSE**: 셋 다 **Apache-2.0**(Bullet·MuxWise는 SGLang Apache-2.0 승계, 별도 라이선스 추가 없음).
- `[measured]` **Docker** `combathhhhhh/pdmux:sglpr_torch2.6_bench` — pull 안 함. 태그의 torch2.6은 MuxWise 커밋 `3bf0ca434`("perf issues on torch > 2.6.x")와 정합. 단 `pyproject.toml`은 **torch==2.8.0** 핀 → torch2.6=권장 perf base, 2.8=빌드 핀.

---

## 3. T2 — MuxWise 정밀 분석

### 3.1 Base 버전 & 변경 규모
- `[measured]` MuxWise = SGLang **0.5.3rc0**(`version.py`, `pyproject.toml`; `torch==2.8.0`, `flashinfer_python==0.4.0rc1`).
- `[measured]` `git merge-base(slo_config, upstream)` = `608854821` (**2025-09-25**). slo_config = 그 위 **47 커밋**. 최신은 merge-base 대비 **9209 커밋(~9개월)** 앞섬. 최신 핀 torch==2.11.0 / flashinfer 0.6.12.
- `[measured]` `git diff --stat 608854821..slo_config` = **31 files, +2396/−91**. 코어 엔진 발자국 ≈ 11 파일; 나머지는 벤치 스크립트(`benchmark/pdmux/bench_serving.py` +1055)·문서·yml·PNG. → **변경은 국소적·bounded**.

### 3.2 Hybrid-mamba 인프라 (0.5.3rc0)
- `[measured]` **부분 스캐폴딩만**: `HybridReqToTokenPool`/`HybridLinearKVPool`/`MambaPool`/`conv_state`/`ssm_state` 존재(memory_pool.py, schedule_batch.py, model_runner.py, hybrid_linear_attn_backend.py).
- `[measured]` mamba 레이어 디렉토리 = `causal_conv1d.py`,`causal_conv1d_triton.py`,`mamba.py`뿐. `mamba.py`는 `mamba_v2_sharded_weight_loader` 헬퍼만(**Mamba2 SSD 믹서 아님**). `ops/` SSD 디렉토리·`MambaMixer2` **부재**.
- `[measured]` 유일 하이브리드 모델 = `Qwen3Next`(gated-deltanet linear attn, **Mamba2 SSD 아님**). `NemotronH`=0, `FalconH1`=0, `Zamba2`=0.
- `[derived]` Zamba2는 **Mamba2 SSD** 사용 → A는 SSD 인프라 백포트가 선결(§1 표).

### 3.3 GreenContext 층 ★
- `[measured]` SM 제어 = 컴파일 확장 **`sgl_kernel.spatial`**(리포의 ctypes 계획 아님): `get_sm_available(gpu_id)`, `create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)`. 소스 `sgl-kernel/csrc/spatial/greenctx_stream.{cu,h}`, `python/sgl_kernel/spatial.py`.
- `[measured]` **`sgl_kernel.spatial`은 MuxWise·최신 SGLang 양쪽에 존재하고 `greenctx_stream.cu`+`spatial.py`가 바이트 동일.** upstream 병합(PR #7649/#8701/#9231); #8701이 cuda<12.4 비호환을 수정 → **CUDA 12.4+ 지원(클러스터 CUDA 13.0 충족)**. ⇒ B는 green-ctx 이식 불요. 리포의 ctypes `green_ctx_controller.py` 계획을 대체.
- `[measured]` **H200(132 SM) 하드코딩 없음**. `pdmux_context.py`: `total_sm_count = spatial.get_sm_available(gpu_id)`(device query). `get_arch_constraints(cc)`: major==8(A100)→(min_per_part=4, multiple=2), major==9(Hopper)→(8,8) — **A100 명시 지원**. `divide_sm()`가 질의된 SM 수로 파티션 산정.
- `[measured]` **A100 적응점(코드 아닌 config)**: 동봉 `sharegpt.yml`/`loogle.yml`의 `manual_divisions`가 **H200 튜닝**(각 항목 합=132; 예 `[112,20]`,`[104,28]`…). A100(108 SM)은 divisions 재프로파일 또는 `manual_divisions` 생략(auto `divide_sm()`) 필요.

### 3.4 3-모듈 경계
- `[measured]` **엔진(bubble-less multiplex)**: `srt/multiplex/multiplexing.py::SchedulerMultiplexMixin.event_loop_pdmux()`(신규 +195). prefill_stream∥decode_stream(green-ctx 쌍). prefill은 **레이어 단위 split**(`ForwardMode.SPLIT_PREFILL`, forward_batch_info.py:90): `split_index`를 `split_forward_count = split_forward_token_budget // extend_num_tokens` 레이어씩 진행, `num_hidden_layers`까지. `Scheduler`(scheduler.py:223)가 mixin 상속(230), scheduler.py:2868서 dispatch.
- `[measured]` **디스패처(SLO-aware)**: `adjust_stream_groups()` — `decode_bs`→`stream_idx`(SM 파티션): manual `decode_bs_threshold` 표 또는 `stream_idx=decode_bs*(N-2)//decode_bs_divisor`. `TODO: temporary demo` 주석. `model_runner.update_decode_attn_backend(stream_idx)`(model_runner.py:1871) — 파티션별 attn 백엔드.
- `[measured]` **에스티메이터(contention-tolerant)**: slo_config에 **온라인 predictor 코드 없음**(contention|estimator|predictor|roofline|solo_run grep=0). **오프라인 `manual_divisions`(YAML)로 실현** — 논문의 "estimator"=오프라인 프로파일. (온라인 SLO 스케줄러는 별도 `slo_scheduler` 브랜치일 가능성; slo_config엔 미포함.)
- `[measured]` 코어 훅(소편집): scheduler.py(+39), model_runner.py(+34), tp_worker.py(+26), parallel_state.py(+32, `set_pdmux_status`), pynccl.py(+38), schedule_batch.py(+13), flashinfer_backend.py(+10), server_args.py(+37).

### 3.5 CUDA Graph × 파티션
- `[measured]` decode 그래프를 **(stream_idx, batch_size)별** 캡처: 키 `f"{current_stream_idx}_{cuda_graph_bs}"`(cuda_graph_runner.py +161). 파티션별 decode 스트림 루프. → 그래프 메모리 × N_partitions.

### 3.6 forward-port(B) 충돌 플래그 (T2 관점, §4.1이 대부분 해소)
- `[measured]` `--enable-pdmux`는 **overlap schedule / chunked prefill / disaggregation / pp>1 비호환** 단언(server_args.py:2746+). 최신 SGLang은 overlap이 기본 → *수작업* forward-port였다면 충돌. **그러나 §4.1: upstream이 이미 최신 스케줄러 위에서 pdmux를 유지** → 이 충돌은 upstream이 해소함(pdmux는 자체 event loop로 공존, overlap은 pdmux 모드에서 비활성).
- `[measured]` torch>2.6.x perf 경고: A(2.8)·B(2.11) **모두** 해당(§5 공통 리스크).

---

## 4. T4 — 최신 SGLang (전략 B base 검증) ★★

### 4.1 PD-multiplexing이 이미 upstream (MuxWise 메커니즘이 SGLang 본류에 병합)
- `[measured]` 최신에 MuxWise 엔진이 그대로 존재: `srt/multiplex/multiplexing_mixin.py::SchedulerMultiplexMixin`(`event_loop_pdmux` L96, `adjust_stream_groups` L49) + `srt/multiplex/pdmux_context.py`(`PDMuxConfig`,`manual_divisions`,`divide_sm`,`get_arch_constraints`,`spatial.*`). `Scheduler` 상속(scheduler.py:301), dispatch(scheduler.py:4173), `SPLIT_PREFILL`(forward_batch_info.py:99, split_index L477), run_batch 분기(scheduler.py:3294), server_args `enable_pdmux`(L2416).
- `[measured]` 출처: PR **#11592** "[Feature] PD-Multiplexing Context and Scheduler" + **#12275**(lazy import spatial). 이후 스케줄러 리팩토링(#25609/#25610 request-ingress, mixin 타입힌트 #15916)과 eagle speculative cuda-graph 통합까지 유지.
- `[measured]` 최신 `pdmux_context.py`도 `PDMuxConfig`/`manual_divisions`/`decode_bs_divisor` 보유 → slo_config 디스패처+오프라인-config **기능 동등**.
- `[derived]` ⇒ **B의 3모듈 forward-port 비용 ≈ 0**(수작업 이식이 아니라 upstream 채택). 프롬프트의 B 프레이밍(수작업 forward-port)은 "upstream pdmux 채택"으로 대체됨.

### 4.2 Zamba2 포트 표면 (재사용 vs 신규)
- `[measured]` **재사용(최신에 존재)**: `MambaMixer2`(layers/attention/mamba/mamba.py:191)=완전 Mamba2 SSD 믹서; `Mamba2AttnBackend`; 완전 SSD `ops/`(ssd_chunk_scan/combined/state_passing/bmm/chunk_state); `RadixAttention`; hybrid pool+`MambaPool` 배선; **`hybrid_override_pattern` per-index 레이어선택**(NemotronHModel:760, `ALL_DECODER_LAYER_TYPES[pattern[idx]]`) = Zamba2 ABAB 인터리브 스캐폴드.
- `[measured]` 최근접 템플릿 = **NemotronH**(temporal: `NemotronHMambaDecoderLayer`=`MambaMixer2`, `NemotronHAttentionDecoderLayer`=`NemotronHAttention`+`RadixAttention`). FalconH1=spatial(attn+mamba 동일 레이어) — layer-aware "미적용" 대조군.
- `[derived]` **신규 작성(NemotronH/FalconH1에 없음)**: (i) **shared attention+MLP 블록**(위치 간 가중치 공유; NemotronH는 레이어별 독립), (ii) 공유 블록의 **per-invocation LoRA**, (iii) Zamba2 config/`hybrid_override_pattern`, (iv) ABAB 배선. **참조 구현: vLLM `zamba2.py`**(리포가 이미 vLLM Zamba2 구동 — `vllm_bench/`). 포트 = vLLM Zamba2 로직을 SGLang model API로 옮기며 `MambaMixer2`+NemotronH 스캐폴드 재사용.
- `[derived]` A는 위 재사용분(MambaMixer2/ops/Mamba2AttnBackend/scaffold)이 0.5.3rc0에 **전부 부재**(§3.2)라 신규작성 전에 백포트 선결 → **B보다 순증.**

### 4.3 research-delta (P0 게이트 항목 아님 — 참고)
- `[derived]` layer-type-aware 기여는 **decode 경로**에 훅: 현 pdmux는 스케줄 iteration당 decode SM 파티션 1개(모델 전체)를 고정. layer-aware = layer-type 윈도우별 green-ctx 파티션 전환(attn-decode 레이어에 floor 예약, ssm-decode 레이어서 prefill로 환원). green-ctx 스트림 인프라·(stream_idx,bs) 그래프 구조는 존재; 확장=decode의 per-layer-type 스트림 전환 — 이는 최신에만 있는 mamba2 레이어와 상호작용 → **B 추가 이점**.

---

## 5. T3 — Bullet (설계 차용만; base로는 기각)

- `[measured]` Bullet=SGLang fork, `--enable-bullet-engine`, **libsmctrl(CUDA ≤ 12.6)+MPS** 필요 → CUDA-13 클러스터서 사용 불가(기각 확인).
- `[measured]` **(1) libsmctrl 표면**: 20 파일 중 python swap면=`srt/bullet/sm_controller.py`(`_LibSMCtrl` ctypes, **TPC 단위** 마스크, `ScheduleBudget(prefill_ratio,decode_ratio)`)+`csrc/src/libsmctrl*.c`+`tp_worker.py` 1참조. 우리 세계선(green-ctx)에서 이 층 전체가 `sgl_kernel.spatial`로 **대체**(차용 아님).
- `[measured]` **(2) 2-프로세스 MPS disagg**: `srt/bullet/` — `launchers.py`/`loop_forever.py`(prefill·decode 독립 프로세스), IPC=`rpc_server/req_rpc/radix_cache_rpc/memory_pool_rpc_v2` + 공유메모리 `shared_nparray/shared_mng/ring_array/smem_mutex`.
- `[measured]` **(3) 차용 대상 설계 위치**: SRM roofline estimator=`srt/bullet/observability.py::PredictorInfoEntry`(phase, prefill_len, decode_bs, decode_tokens, **prefill_tpc, decode_tpc**)+`PredictorInfos`; 타이밍 계측=`models/llama_timing.py`,`qwen2_timing.py`,`bullet/timing.py`,`model_monkey_patch.py`; (ES,PS,RS)류 상태=`observability.py::ReqTimingState`(ttft/tpot/decode_duration)+`ReqTimingStateDict`.
- `[measured]` **(4) ablation**: `artifact_evaluation/run_all.sh`(bullet full vs vanilla `--chunked-prefill-size 1024`, rate 10..25, sharegpt, Llama-3.1-8B)+`plot.py`.
- `[derived]` **차용 리스트(설계, 코드 아님)**: SRM predictor 형태(observability.PredictorInfoEntry)→우리 layer-type SM floor 도출; P∥D 비동기 개념(단, MuxWise가 단일 프로세스 green-ctx로 더 간단히 실현→그걸 채택); reordering/delayed-decode는 디스패처 옵션. libsmctrl/MPS 층은 CUDA-13서 이식 불가.

---

## 6. T5 — A100 green-ctx smoke (유일 GPU 잡)

> **스코핑 결정(정직)**: 프롬프트의 "MuxWise 격리 venv + 소형 dense 서버 기동"을 그대로 하려면 torch2.8/flashinfer0.4(MuxWise 핀)를 CUDA-13에 세우거나 최신 torch2.11 전체 sglang+sgl-kernel 빌드가 필요 — 빌드 마찰 확률이 높고 2h 예산을 소진할 개연. **가장 novel한 리스크는 green-ctx SM 제어가 A100/CUDA13에서 실제로 동작하는가**이므로, 전체 서버 대신 **실제 upstream `greenctx_stream.cu`(sha256 de20703f, 무수정)를 JIT 컴파일**해 `get_sm_available`(순수 python 경로)+`create_greenctx_stream_by_value`(컴파일 커널)를 직접 검증. 설치 0(격리: TORCH_EXTENSIONS_DIR=scratch). 전체 서버 기동은 P1로 이연.

- `[measured]` **사전 증거(리포 자산)**: `results/stage2/ctx_switch_overhead_a100-sxm4-80gb.json` — `device=NVIDIA A100-SXM4-80GB`, `total_sm=108`, `backend=green_ctx`, 파티션 14/28/40/54/68/82/94/108 생성, cpu_swap ~0.45µs. 즉 `cuGreenCtxCreate`(sgl_kernel.spatial과 동일 driver API)가 **본 A100/CUDA13에서 이미 동작 실증**.
- `[measured]` 클러스터 conda `pytorch_2.9.1_cuda13`: torch **2.9.1+cu130**, nvcc **13.0.88**, python 3.14.
- `[measured]` **T5 잡(824014, EXIT 0) 결과 — green-ctx가 A100/CUDA13에서 실동작 확인**:
  - 환경: A100-SXM4-80GB, cc **8.0**, driver **580.105.08**, CUDA **13.0**, torch **2.9.1+cu130**, nvcc 13.0.88.
  - **check1** `get_sm_available` (순수 python 경로, spatial.py와 동일) = **108** ✓.
  - **check2** 실제 upstream `greenctx_stream.cu`(sha256 `de20703f`, 무수정) JIT 컴파일 **75.9s** 성공 → `create_greenctx_stream_by_value(64, 44, 0)` → **actual smA=64, smB=44**(108 정확 분할), 두 스트림 포인터 non-null. **fallback 경고 없음** = `cuGreenCtxStreamCreate`(cuda≥12.5) **직접 경로** 사용.
  - **check3** 두 파티션 스트림 각각에서 matmul 실행 성공 ✓.
  - `[derived]` ⇒ pdmux가 쓰는 정확한 green-ctx SM 제어 커널이 **A100(108 SM)/CUDA13/torch2.9에서 빌드·실행·파티션 생성 모두 성공**. SM-제어 층은 A/B 공통으로 **실증 완료**. (arch-constraint 예약은 §3.3의 device-query와 정합: cc8→multiple=2, 64/44 유효.) 프롬프트의 "108 SM 인식·파티션 생성 성공·preset" 요구 충족.
  - 미이행(의도적 이연): 전체 서버 `--enable-pdmux` 기동은 §6 스코핑대로 P1 이연(전체 sglang+sgl-kernel+flashinfer 빌드는 별건).

---

## 7. 권고와 근거 (최종 판정은 사람)

**권고: 전략 B — 최신 SGLang을 base로 채택.** 근거:

1. `[measured]` **판정 기준 "B 유리 조건" 전부 충족**: (a) MuxWise 변경이 국소적(31파일/코어 11)일 뿐 아니라 **이미 upstream 병합·유지**(PR#11592/#12275) → forward-port bounded를 넘어 *불요*. (b) 최신의 mamba2/NemotronH가 Zamba2 포트 표면을 크게 축소. (c) MuxWise base(0.5.3rc0)는 mamba2 SSD 부재라 A의 백포트 비용 > B의 이식 비용.
2. `[measured]` **"A 유리 조건" 미충족**: forward-port 비용 폭발 없음(upstream이 흡수); 스케줄러 리팩토링으로 훅 소멸 없음(upstream이 훅 유지); mamba 인프라 백포트가 가볍지 않음(A100 base엔 SSD 全부재).
3. `[measured]` green-ctx SM 제어는 upstream·device-adaptive·A100 명시지원·CUDA13 호환이라 **A/B 공통으로 이식 리스크 낮음**(§3.3, §6). H200 하드코딩 없음 → A100 수정은 YAML manual_divisions 재튜닝(config)뿐.

**B 실행 시 실제 작업(≈ 우리 연구의 본체)**: (i) 최신 SGLang에 Zamba2 모델 포트(NemotronH 템플릿+`MambaMixer2` 재사용, vLLM zamba2.py 참조; 신규=shared-attn+LoRA+ABAB), (ii) 이미 있는 pdmux 엔진/디스패처에 **layer-type-aware decode SM 예약** 추가(§4.3), (iii) A100용 manual_divisions 재프로파일.

---

## 8. 미확인 / 차단 / 공통 리스크

- `[measured, 공통 리스크]` **torch>2.6.x green-ctx perf 저하 경고**(MuxWise 커밋 3bf0ca434). A(2.8)·B(2.11) **모두** 해당 — 전략 미차별. docker의 torch2.6만 회피. **미확인**: 우리 A100/CUDA13 스택서 실제 perf 저하폭. → P1서 측정(green-ctx 하 multiplex step 오버헤드). 최악의 경우 torch2.6 고정이 필요할 수 있으나 이는 A/B와 독립.
- `[derived, 미확인]` upstream pdmux가 slo_config 대비 *논문 수치 재현*에 충분한가는 미검(기능 동등은 확인, 성능 동등은 미측). MuxWise 논문 baseline과의 정확 비교가 필요하면 동결 아티팩트(`slo_config`@torch2.6 docker) 별도 유지.
- `[derived, 미확인]` Zamba2 shared-attention 블록의 **per-invocation LoRA**를 SGLang weight-loader/attn-backend에 얹는 정확 배선(vLLM은 되지만 SGLang API 차이) — 포트 착수 시 조기 노출 필요(P1 이후).
- `[measured, 미확인]` 최신 `event_loop_pdmux`의 decode 경로가 **단일 파티션/iteration** 전제 — layer-type별 파티션 전환을 넣을 때 (stream_idx,bs) CUDA graph 재캡처/무효화 비용 미측(§4.3).
- `[측정 완료, 비차단]` 네트워크·클론·라이선스·버전·인프라 grep은 모두 확인(§2–§5).

---

## 9. 산출물 위치

- 증거 원본: [triage/notes.md](../triage/notes.md) (T1–T4 파일·심볼·라인)
- T5 하네스/결과: [triage/t5_greenctx/](../triage/t5_greenctx/) (`test_greenctx.py`, `run_t5.sbatch`, `t5_result.json`, 잡 로그)
- 클론: `external/{muxwise, muxwise-zenodo, bullet, sglang-latest}`
