# local_smoke_5060ti_2026-09-18

- **기판**: NVIDIA GeForce RTX 5060 Ti · driver 580.178.04 · CUDA 13.0(드라이버) / torch cu128 ·
  compute capability 12.0 (Blackwell sm_120) · VRAM 16,311 MiB (idle desktop 사용분 ~450–540 MiB 포함).
- **엔진**: vLLM 0.12.0 (stock, 패치 없음), torch 2.9.0+cu128. SGLang이 아님 — 이 환경의
  `sgl_kernel`은 sm100 빌드만 있고 import가 `undefined symbol:
  _ZNK3c106SymInt22maybe_as_int_slow_pathEv`로 실패해 서빙 불가(재확인, 우회 시도 안 함).
- **모델**: `tiiuae/Falcon-H1-3B-Base` rev `c096902c69be0eed2e5369d3420a8bb960d293e1` (1순위,
  아래 §1) · `Zyphra/Zamba2-2.7B` rev `31afeeac4c66b4851a54290ba57c995a68c87861` (§2).
- **날짜**: 2026-09-18 (KST 15:24–15:44 구간).

> **게이트 1: 이 디렉터리의 어떤 수치도 A100 기판 결론과 비교·혼용할 수 없다.
> PD-mux 아님. 성능 판정 아님.** 여기 적힌 TTFT/ITL/throughput은 순전히 "이 로컬
> RTX 5060 Ti + vLLM stock 환경에서 3B급 hybrid 모델이 뜨고 응답하는가"를 확인하기
> 위한 참고 수치이며, `prefill-layer-alloc`의 A100(108 SM) 기판 결과·HE0·layer-type
> 死 판정·λ\*·격자와 절대 섞지 말 것. SM 분할/green-context/PD-mux는 시도하지 않았다
> (사용자 지시로 범위 밖).

## 왜 vLLM인가 (SGLang이 아니라)

로컬 GPU(`RTX 5060 Ti`, sm_120, compute cap 12.0)에서 SGLang은 애초에 두 겹으로 막힌다:
1. `sglang/srt/multiplex/pdmux_context.py:get_arch_constraints`가 major 6–9만 지원 —
   PD-mux 경로는 `ValueError`로 거부(이번 작업은 PD-mux를 안 쓰므로 직접 걸리진 않음).
2. 설치된 `sgl_kernel`(`~/.local/lib/python3.10/site-packages/sgl_kernel`)이 **sm100
   빌드**뿐이라 `import sgl_kernel` 자체가 `undefined symbol:
   _ZNK3c106SymInt22maybe_as_int_slow_pathEv`로 실패 — SGLang 엔진을 아예 못 띄운다.
   `sgl_kernel`을 빌드·수정하려 하지 않았다(사용자 지시로 범위 밖).

miniconda base(python3.13)의 **vLLM 0.12.0**은 `torch.cuda.get_arch_list()`에 `sm_120`이
있고, `Zamba2ForCausalLM`·`NemotronHForCausalLM`·`FalconH1ForCausalLM`·
`GraniteMoeHybridForCausalLM`가 전부 모델 레지스트리에 등록돼 있어 이 GPU에서 쓸 수 있는
유일한 hybrid-capable 엔진이었다. ⇒ **이 캠페인은 vLLM stock 기준**이며 SGLang v0.5.10
기준 A100 결과와는 엔진 자체가 다르다(비교 불가의 추가 이유).

---

## §1. Falcon-H1-3B-Base

### 모델 확보

```
export HF_HOME=/home/wonho/Experiments/KISTI/prefill-layer-alloc/hf_cache
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(
    repo_id='tiiuae/Falcon-H1-3B-Base',
    revision='c096902c69be0eed2e5369d3420a8bb960d293e1',
    cache_dir='/home/wonho/Experiments/KISTI/prefill-layer-alloc/hf_cache/hub')"
```
- 저장 위치: `/home/wonho/Experiments/KISTI/prefill-layer-alloc/hf_cache/hub/models--tiiuae--Falcon-H1-3B-Base/snapshots/c096902c69be0eed2e5369d3420a8bb960d293e1`
- 크기: 5.9 GB (bf16 safetensors 2-shard), 소요 159초.
- architecture: `FalconH1ForCausalLM` — 32 layers, hidden 2560, **parallel hybrid**
  (`attn_layer_indices: null` — attention과 mamba가 alternating이 아니라 매 레이어에서
  병렬로 결합되는 구조; Zamba2/NemotronH의 alternating hybrid와 다름). `torch_dtype: bfloat16`.

### 부팅 커맨드 (실제 사용값 — 과거 A100 설정 `max_model_len 4096 / gpu_mem 0.85 /
max_num_seqs 64`는 참고만 하고 그대로 쓰지 않음, VRAM 1/5이므로 보수적으로 낮춤)

```
python3 -m vllm.entrypoints.openai.api_server \
  --model tiiuae/Falcon-H1-3B-Base \
  --revision c096902c69be0eed2e5369d3420a8bb960d293e1 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 8 \
  --port 8000
```

### 부팅 배관 사실 (전문 로그 `falconh1_serve_stdout.log`)

- attention 백엔드: `FLASH_ATTN` (후보 `['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN',
  'FLEX_ATTENTION']` 중 자동 선택).
- cudagraph_mode: `CUDAGraphMode.FULL_AND_PIECEWISE` (capture sizes `[1,2,4,8,16]`) —
  캡처 성공, eager fallback 없음. capture 소요 71초.
- chunked prefill 활성 (`max_num_batched_tokens=2048`); cascade attention은 hybrid라서
  비활성; "Setting attention block size to 2080 tokens to ensure attention page size >=
  mamba page size" / "Padding mamba page size by 0.24%" (hybrid KV/mamba 캐시 정렬 배너).
- 모델 로드: 5.8981 GiB, 0.92초(safetensors) / 전체 가중치 로드 2.54초.
- KV 캐시: `Available KV cache memory: 5.28 GiB` → `GPU KV cache size: 83,968 tokens` →
  `Maximum concurrency for 2,048 tokens per request: 41.50x`.
- 엔진 초기화(profile+kv cache+warmup) 총 89.36초. 서버 기동까지 총 ~103초
  (15:28:35 시작 → 15:30:18 라우트 등록).
- VRAM 사용량(부팅 후, `nvidia-smi`): **12,503–12,764 MiB / 16,311 MiB**
  (데스크톱 compositor 기저사용 ~450–540 MiB 포함).

### correctness 확인

`POST /v1/completions`, prompt "The capital of France is", `temperature=0`,
`max_tokens=20` → `" Paris.\nThe capital of France is Paris.\nThe capital of France is\nThe"`
— 일관된 응답, 에러 없음.

### 참고 bench (이 기판 한정, dataset=vLLM `random`, input_len=256/output_len=64, 8 prompts,
request_rate=inf) — 원본 `bench_random_smoke.json`

| metric | 값 |
|---|---|
| completed/failed | 8 / 0 |
| duration | 1.73 s |
| request throughput | 4.63 req/s |
| output token throughput | 296.3 tok/s |
| TTFT mean/median/p99 | 366.6 / 506.9 / 507.7 ms |
| TPOT(=ITL) mean/median/p99 | 21.5 / 19.4 / 25.0 ms |

(dataset은 vLLM 자체 `random` 샘플러 — 프로젝트의 ShareGPT `hf_cache/raw/` 사본은 이번
스모크에서 쓰지 않았다. n=8 단발이므로 분산 추정 불가 — 반복측정 아님, 그냥 배관 확인.)

---

## §2. Zamba2-2.7B (스트레치, 2순위)

### 모델 확보

동일 절차, `repo_id='Zyphra/Zamba2-2.7B'`, `revision='31afeeac4c66b4851a54290ba57c995a68c87861'`.
- 저장 위치: `.../hf_cache/hub/models--Zyphra--Zamba2-2.7B/snapshots/31afeeac4c66b4851a54290ba57c995a68c87861`
- 다운로드 12 files, 20 GB, 400초. **주의**: repo에 vLLM이 안 쓰는 raw PyTorch 체크포인트
  (`model.pt`, `Zamba2_2p7b_direct_from_pytorch.pt`, 합 ~15 GB)가 같이 딸려와 safetensors
  (~5.3 GB, 2-shard, bf16 추정 — config에 명시적 `torch_dtype` 키 없음)보다 훨씬 큼; vLLM은
  safetensors만 로드.
- architecture: `Zamba2ForCausalLM` — 54 layers, hidden 2560, **alternating hybrid**
  (`hybrid_layer_ids: [6,12,18,24,30,36,42,47,51]`, 9개 레이어만 attention 포함, 나머지
  mamba). `max_position_embeddings: 4096` — ★long-context 실험 시 이 모델로는 불가
  (CLAUDE.md 기재 사실 확인됨).

### 부팅 커맨드 (동일 보수 설정, 포트만 분리)

```
python3 -m vllm.entrypoints.openai.api_server \
  --model Zyphra/Zamba2-2.7B \
  --revision 31afeeac4c66b4851a54290ba57c995a68c87861 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 8 \
  --port 8001
```

### 부팅 배관 사실 (전문 로그 `zamba2_serve_stdout.log`)

- attention 백엔드: `FLASH_ATTN` (후보 `['FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION']`
  — Falcon-H1과 달리 `FLASHINFER`가 후보에 없음).
  - cudagraph_mode: `CUDAGraphMode.FULL_AND_PIECEWISE`, capture 27초, eager fallback 없음.
  - "Setting attention block size to 48 tokens..." / "Padding mamba page size by 43.12%..."
  (Falcon-H1보다 훨씬 큰 패딩 — 9-layer만 attention이라 구조가 다름).
- 모델 로드: 5.0655 GiB, 1.46초(safetensors) / 전체 로드 2.54초.
- KV 캐시: `Available KV cache memory: 6.16 GiB` → `GPU KV cache size: 5,088 tokens` →
  `Maximum concurrency for 2,048 tokens per request: 15.27x`.
  ★Falcon-H1(83,968 tokens, 41.5x)보다 KV 캐시 토큰 수가 **1/16**로 훨씬 작다 — 동일
  `--max-model-len 2048`·동일 `--gpu-memory-utilization 0.75`인데도 아키텍처 차이(alternating
  hybrid, attention_hidden_size 5120, kv_channels 80)로 페이지 정렬 오버헤드가 커진 것으로
  보인다. 순위 판정 아님, 배관 사실만 기록.
- 엔진 초기화 총 36.17초. 서버 기동까지 총 ~51초(15:41:44 → 15:42:35).
- VRAM 사용량(부팅 후): **12,472 MiB / 16,311 MiB**.

### correctness 확인

동일 프롬프트 → `" Paris.\nThe capital of the United States is Washington, D.C.\nThe capital of"`
— 일관된 응답, 에러 없음.

### 참고 bench (동일 조건: `random`, input_len=256/output_len=64, 8 prompts, request_rate=inf)
— 원본 `bench_random_smoke_zamba2.json`

| metric | 값 |
|---|---|
| completed/failed | 8 / 0 |
| duration | 2.12 s |
| request throughput | 3.77 req/s |
| output token throughput | 241.3 tok/s |
| TTFT mean/median/p99 | 598.8 / 672.9 / 673.5 ms |
| TPOT(=ITL) mean/median/p99 | 24.1 / 23.0 / 31.4 ms |

---

## 파일 목록

```
README.md                       (본 문서)
falconh1_download.log           huggingface_hub snapshot_download stdout
falconh1_serve_stdout.log       vLLM 서버 전체 부팅+서빙 로그(§1)
bench_random_smoke.json         vLLM bench serve 결과(§1, 8 prompts)
zamba2_download.log             huggingface_hub snapshot_download stdout
zamba2_serve_stdout.log         vLLM 서버 전체 부팅+서빙 로그(§2)
bench_random_smoke_zamba2.json  vLLM bench serve 결과(§2, 8 prompts)
```

## 이 기판에서 확인할 수 없어 대여 GPU로 넘어가는 것

- **PD-mux/green-context/SM 분할**: 이 GPU(compute cap 12.0)는 SGLang
  `get_arch_constraints`가 major 6–9만 지원하므로 애초에 대상 밖(사용자 지시로도 범위 밖).
  대여 GPU가 A100(sm80)이나 H100(sm90)으로 정해지면 그쪽에서 SGLang v0.5.10 + green-context
  경로를 재현해야 한다.
- **cudagraph 운영점에서의 정책 비교·처리량 순위**: 여기서 찍은 TTFT/ITL/throughput은
  n=8 단발 참고치일 뿐이며, 실험 규약(n≥5, paired, 변화 trace, warm-up/correctness/
  benchmark phase 분리)을 전혀 지키지 않았다 — 애초에 "성능 판정"을 하려는 목적이 아니었다.
- **9B/8B급 hybrid 모델(NemotronH-8B/9B, Falcon-H1-7B, Zamba2-7B, Mamba-Codestral-7B)**:
  16GB VRAM에 안 들어가거나(9B 계열) 여유가 빠듯해(7B 계열) 사용자가 3B급으로 범위를
  명시적으로 한정했다 — 이번 작업에서 시도하지 않음.
- **SGLang 엔진 자체의 로컬 서빙**: `sgl_kernel` sm120 빌드 부재로 이 GPU에서는 근본적으로
  안 된다(빌드·수정 시도 안 함, 범위 밖). 대여 GPU가 SGLang이 지원하는 기판이면 그쪽에서
  기존 canonical 파이프라인(`workspace/engine-port/`)을 그대로 쓸 수 있다.

## 참고 (이 세션에서 발견, 손대지 않음)

- `results/serving-eval/`(최상위, `workspace/engine-port/results/`가 아님) 아래에 이미
  2026-09-18 11:32 시각의 `nemotron_h`/`falcon_h1`/`zamba2` ShareGPT 서빙 결과·로그가 존재한다
  (`vllm_nemotron_h_sharegpt_753583.log`는 `nvidia/Nemotron-H-8B-Base`, 경로가 옛 KISTI
  Neuron `/scratch/ehmoon/whlee/...` 를 가리킴, vLLM 0.22.1). 이는 **이번 작업이 만든 것이
  아니고 마이그레이션 번들에서 이미 풀려 있던 이전(A100) 캠페인 아카이브로 보인다** — 확인만
  하고 건드리지 않았다. 이 디렉터리 이름이 이번에 새로 만든
  `workspace/engine-port/results/local_smoke_5060ti_2026-09-18/`와 혼동되지 않도록 주의.
