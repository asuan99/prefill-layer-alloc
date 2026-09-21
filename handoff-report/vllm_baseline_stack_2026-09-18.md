# vLLM 비교 대조군 — 프레임워크·패키지·라이브러리 정리 (2026-09-18)

사용자 요청(2026-09-18): "vLLM도 비교대조군으로 추가할 필요가 있으니까 이와 관련된
패키지 또는 프레임워크 및 라이브러리들에 대해서 정리해".

> **지위**: 환경·의존성 정리 문서. **GPU 지출 0 · 새 측정 0 · 새 성능 판정 0건.**
> 연구 결론·Claim 등급·게이트는 이 문서로 바뀌지 않는다. vLLM을 **무엇의 대조군으로
> 쓸지**(§5)는 아직 결정되지 않았고 이 문서가 결정하지도 않는다.
>
> 짝 문서: `gpu_rental_checklist_2026-09-18.md`(기판 판정) ·
> `migration_plan_local_dev_remote_gpu_2026-09-18.md`(이양 후 운영).

## 0. 먼저 — vLLM은 "추가"가 아니라 **이미 있었고 지금 부활 대상**이다

| 시점 | 무엇 | 상태 |
|---|---|---|
| 2026-06-21 | **vLLM 0.22.1** 로 sim의 fused baseline 검증(격리 venv `vllm_venv`, torch 2.11+cu130, py3.14, A100-80GB gpu40) | `deprecated/reports/historical_root/vllm_validation.md`. Zamba2-2.7B·Falcon-H1-3B. **"GQA가 PD-mux 이득 구간 폭을 결정(16×)"을 반증**한 것이 이 측정의 최대 산출 |
| 2026-06-09 (로그 mtime) | **vLLM 0.22.1** ShareGPT 서빙 로그 **13건** (Zamba2-7B-Instruct·Nemotron-H·Falcon-H1) | `results/serving-eval/vllm_*.log` + `serving_*.json`(`stats`+`per_request` 구조 = 우리 분석 파이프라인과 붙는 형태) + `nvml_*.csv` |
| — | 관련연구 축에서의 vLLM | `reports/paper/venue_positioning.md:112` — **vLLM v0.20.0 blog(2026-04) = hybrid(SSM) disaggregation**, "mux 아님 — disagg 축"으로 분류 |
| 2026-09-17 이양 | `vllm_venv`(8.8 GB) | **저장 안 함**(재설치 대상). 로그·json·csv는 저장소에 있음 |

★ 그러므로 해야 할 일은 "새 도입"이 아니라 **(a) 어느 vLLM 버전으로 재구성할지 정하고
(b) 과거 0.22.1 측정과의 연속성을 어떻게 취급할지 정하는 것**이다.

★★ 주의: `PROJECT_STATUS.md:10225`의 *"`vllm`은 다른 엔진이라 쓰지 않는다"* 는 **SLURM
`--comment` 의 `appl=` 필드 신고값** 이야기이고 대조군 논의와 무관하다. 인용하지 말 것.

## 1. ★두 스택은 같은 venv에 들어갈 수 없다 (실측)

로컬 설치본으로 확인한 **선언 의존성 충돌**:

| 패키지 | SGLang 스택(정본 `env/venv_packages_2026-09-17.txt`) | vLLM 0.12.0 선언 | 충돌 |
|---|---|---|---|
| `torch` | **2.9.1+cu130** | **`==2.9.0`** (정확 고정) | ★충돌 |
| `transformers` | **5.8.0** | **`<5,>=4.56.0`** | ★★**메이저 충돌**(가장 심각) |
| `flashinfer-python` | **0.6.10** | **`==0.5.3`** | ★충돌 |

⇒ **격리 venv(또는 컨테이너)가 필수다.** 과거에 `vllm_venv`를 8.8 GB 따로 둔 이유가
정확히 이것이며, 앞으로도 단일 venv 통합을 시도하지 말 것. 두 엔진은 **같은 GPU에
순차로** 돌리고, venv만 갈아탄다.

※ 위 vLLM 열은 **로컬에 설치된 0.12.0** 기준이다. 버전마다 다르다 — 과거 실험의
0.22.1은 **torch 2.11+cu130**을 요구했다(`vllm_validation.md`). 즉 **핀은 vLLM 버전을
고르고 나서 확정**된다(§3).

## 2. 라이브러리 계층 — 공유되는 것 / 엔진마다 다른 것

| 층 | SGLang PD-mux 경로 | vLLM 경로 | 비고 |
|---|---|---|---|
| 서빙 엔진 | **SGLang v0.5.10** + 우리 패치(`src/multiplex/` 8파일 + `devtree_manual_edits.patch` 5파일) | **vLLM** (버전 미정, §3) | vLLM은 **패치 없이 stock** 사용이 원칙 — 대조군을 손대면 대조군이 아니다 |
| SM 분할 | **`sgl_kernel.spatial`** (green context: `cuDevSmResourceSplitByCount`/`cuGreenCtxCreate`), 드라이버 **≥ 12.4** | **없음** | ★vLLM에는 green-context SM 분할 대응물이 **없다**. 이것이 §5의 대조군 성격을 결정한다 |
| PD 분리 | co-located multiplexing(단일 GPU, SM 시공간 분할) | **disaggregation**(별도 엔진/GPU + mamba state 전이, v0.20.0 blog 2026-04) | **축이 다르다** — `venue_positioning.md`가 이미 그렇게 분류 |
| SSM 커널 | 외부 **`mamba_ssm==2.3.1`** + **`causal_conv1d==1.5.3.post1`** | ★**자체 내장**(`vllm.model_executor.layers.mamba.ops.{mamba_ssm,causal_conv1d}`, `mamba_mixer2`) — 로컬에서 모듈 존재 확인. 선언 의존성에 mamba/causal **없음** ⇒ **외부 패키지 선택적** | ★중요한 비대칭: 같은 이름의 "mamba 커널"이 **서로 다른 구현**이다. 두 엔진의 decode 비용 차이를 "엔진 차이"로 귀속하기 전에 이 층을 분리해야 한다 |
| Attention 커널 | `flashinfer-python 0.6.10` · `flash_attn 2.7.4.post1` · `triton 3.5.1`. ★**백엔드 강제 교락 기록됨**: Nemotron-H+`triton` 부팅 거부 · Zamba2+`flashinfer` 스케줄러 사망(`CONSENSUS §1-3`, §3 항목103) | `flashinfer-python==0.5.3`; 로그상 **FLASH_ATTN(v2)** 자동 선택, 후보 `['FLASH_ATTN','TRITON_ATTN','FLEX_ATTENTION']` | 두 엔진의 백엔드 선택이 다르면 **비교가 백엔드 비교로 오염**된다 — arm마다 무엇이 선택됐는지 로그에서 채취해 병기해야 한다 |
| CUDA graph | 우리 운영점 = **cudagraph-ON** | 0.22.1 로그: `CUDAGraphMode.FULL_AND_PIECEWISE`, capture sizes 1…128 | 같은 "ON"이 아니다. piecewise/full 혼합 여부를 명시할 것 |
| 모델 정의 | 우리 트리가 `models/` 5파일을 install(manifest 25엔트리) | stock vLLM 레지스트리 | 같은 HF 체크포인트를 써도 **구현이 다르다** |
| 벤치 클라이언트 | `bench_serving`(우리 하네스가 감싼다) | `vllm bench serve` (0.22.1 기준 `--dataset-name random` 또는 ShareGPT) | ★**클라이언트가 다르면 TTFT/ITL 정의·측정 지점이 다를 수 있다.** 과거 vLLM 측정이 GIL-client 아티팩트(TPOT 170–205 ms floor) 계열 위험을 공유한다 |

## 3. 선결 결정 — 어느 vLLM 버전인가

| 후보 | 근거 | 값과 비용 |
|---|---|---|
| **0.22.1** (과거 실험과 동일) | `vllm_validation.md` + 서빙 로그 13건이 이 버전 | 과거 측정과 **직접 연속**. 단 torch **2.11+cu130** 요구 = SGLang 스택(2.9.1)과 더 멀어짐(격리 venv라 무해). 2026-06 버전이라 hybrid 지원이 현재보다 낮다 |
| **0.28.0** (2026-08-26 stable) | hybrid 모델이 "first-class citizen"(Qwen3-Next·Nemotron Nano 2·Granite 4.0 완전 지원), Mamba hybrid **prefix caching**·ReplaySSM·FlashInfer Mamba SSU 등 신규 | 대조군으로서 **가장 강한 baseline**(= 우리 이득 주장에 가장 불리한 = 가장 정직한 대조군). 과거 0.22.1 측정과는 **연속성 없음**(재측정 필요) |
| **0.12.0** (로컬에 이미 설치됨) | 즉시 사용 가능, 우리 4모델 arch 전부 등록(확인함) | 구버전. **로컬 배관 확인용으로만** 쓰고 성능 대조군으로 쓰지 말 것 |

**권고**: 성능 대조군은 **0.28.0**(또는 대여 시점 최신 stable). 이유 = 시스템 학회가
반드시 묻는 "약한 baseline을 이긴 것 아닌가"를 정면으로 막는 쪽이 0.28.0이고,
**약한 대조군으로 이기는 것은 이 프로젝트의 게이트 규율과 상충**한다. 과거 0.22.1
측정은 **sim 검증 이력**으로만 남기고 성능 비교의 근거로 재사용하지 않는다.

## 4. 모델 지원 현황 (로컬 vLLM 0.12.0 레지스트리에서 직접 확인)

| 우리 모델 | 필요한 arch | vLLM 0.12.0 | 비고 |
|---|---|---|---|
| NemotronH / Nemotron-Nano-9B-v2 | `NemotronHForCausalLM` | **OK** | 현재 주 모델(ctx 131072) |
| Zamba2 (1.2B/2.7B/7B) | `Zamba2ForCausalLM` | **OK** | 과거 서빙 로그 있음 |
| Falcon-H1 (3B/7B) | `FalconH1ForCausalLM` | **OK** | 과거 서빙 로그 있음 |
| Granite-4 (h-micro) | `GraniteMoeHybridForCausalLM` | **OK** | ★우리 체크포인트의 실제 `architectures` 값과 **대조 필요**(미확인) |

※ 총 245 arch 등록. 최신 0.28.0은 이보다 넓다. **체크포인트별 `config.json`의
`architectures` 필드와 대조하는 것이 실제 확인**이며, 레지스트리에 이름이 있다는 것은
그 전 단계다.

## 5. 대조군의 성격 — 세 갈래, 서로 다른 것을 산다 (미결정)

vLLM에는 green-context SM 분할이 **없으므로**, "vLLM PD-mux vs 우리 PD-mux"는 성립하지
않는다. 가능한 대조는 셋이고 **요구되는 실험과 사는 주장이 다르다**:

| 갈래 | 비교 대상 | 사는 것 | 위험 |
|---|---|---|---|
| **(a) fused baseline 강도 검증** ★가장 값있음 | vLLM fused(stock) ↔ SGLang fused ↔ 우리 PD-mux | "PD 분리 이득이 **SGLang fused가 약해서** 생긴 것이 아니다"를 방어. 현재 정본의 "PD 분리는 4모델 전부서 fused를 이김"은 **SGLang fused 기준**이며 이 방어가 없다 | vLLM fused가 우리 PD-mux보다 빠를 수 있다 — 그 결과도 **그대로 보고해야 한다**. 이 갈래를 여는 것은 **반증 가능성을 스스로 들이는 일**이다(그래서 값있다) |
| **(b) 엔진-무관 특성화 확인** | vLLM에서도 mamba decode가 SM/배치에 둔감한가 | 우리 기전 주장(decode floor·SM 비민감성)이 **한 엔진의 구현 아티팩트가 아님**을 지지 | vLLM은 SM을 못 나누므로 **SM 축은 못 본다**. 배치·컨텍스트 축만 가능 |
| **(c) disaggregation 축 대조** | vLLM hybrid SSM disagg ↔ 우리 co-located mux | venue positioning의 "축이 다르다"를 **실측으로** 뒷받침 | **GPU ≥2장 전제**. 1장 기판에서는 성립 불가 ⇒ 대여 사양이 바뀐다 |

**결정 필요**: (a)만 할지, (a)+(b)를 할지, (c)까지 갈지. (c)는 대여 GPU 수를 바꾼다.

## 6. 기판·게이트 주의 (예외 없음)

- **게이트 1**: vLLM 수치도 서빙 실증이어야 하고, **기판이 다르면 A100 결론과 섞을 수 없다.** 로컬 5060 Ti에서 얻은 vLLM 수치는 배관 확인용이며 논문 근거가 아니다.
- **게이트 2**(방법론): 정책 비교 벤치는 stationary ShareGPT r8이 아니라 **변화 trace**. 단 (a)(b)는 정책 비교가 아니라 **엔진/특성화 비교**라 stationary도 가능 — 어느 쪽인지 사전에 못박을 것.
- **n≥4 · paired CI · duration 합산 · metric cliff 회피**가 그대로 적용된다.
- **동일 trace·동일 seed·동일 SLO 정의**로 비교해야 한다. 두 엔진의 벤치 클라이언트가 다르므로(§2 마지막 행) **TTFT/ITL 정의 일치를 먼저 실증**해야 한다 — 이게 이 트랙의 가장 큰 함정이다.
- 새 트랙이므로 **사전등록 + 규칙층 감사**가 선행돼야 한다(E2·λ0 선례).

## 7. 설치 절차 (대여 서버, 격리 venv 기준)

```bash
# --- (1) SGLang 스택: 기존 정본 그대로 ---
python -m venv sglang_engine_venv --system-site-packages      # 또는 컨테이너
#   패키지 핀: workspace/engine-port/env/venv_packages_2026-09-17.txt (156개)
#   torch 2.9.1+cu130 · triton 3.5.1 · flashinfer 0.6.10 · sglang-kernel 0.4.1+cu130
#   mamba_ssm 2.3.1 · causal_conv1d 1.5.3.post1 · flash_attn 2.7.4.post1 · transformers 5.8.0
#   py3.14 shim: workspace/engine-port/env/sitecustomize.py
SGLANG_ENGINE_DEV=<sglang>/python workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh <manifest.sha256>
patch -p1 -d <sglang>/python < workspace/engine-port/env/devtree_manual_edits.patch

# --- (2) vLLM 스택: 완전히 별개 venv ---
python -m venv vllm_venv                                      # --system-site-packages 쓰지 말 것
vllm_venv/bin/pip install "vllm==<선택한 버전>"                # torch/flashinfer를 스스로 끌어온다
#   외부 mamba_ssm / causal_conv1d 는 설치하지 않는다(vLLM 내장 사용, §2)
#   설치 후 반드시 기록: pip freeze > env/vllm_packages_<날짜>.txt
```

★ **두 venv를 동시에 activate하지 말 것.** transformers 메이저가 충돌해 조용히 잘못된
쪽이 import될 수 있다. 실행 스크립트에서 venv를 명시적으로 지정한다
(`PDMUX_VENV` / 새 `PDMUX_VLLM_VENV`가 그 자리).

## 8. 로컬(RTX 5060 Ti)에서 할 수 있는 것 / 없는 것 — 실측 기반

로컬 실측(2026-09-18): `NVIDIA GeForce RTX 5060 Ti · 16,311 MiB · **compute_cap 12.0** ·
driver **580.178.04**` / `torch 2.9.0+cu128`, `get_arch_list()`에 **`sm_120` 포함** /
**`sgl_kernel` 미설치** / `vllm 0.12.0` 설치됨, 4모델 arch 전부 등록.

| 항목 | 로컬에서 | 근거·한계 |
|---|---|---|
| vLLM **배관** 확인(부팅·arch 해석·bench CLI 형태) | **가능**(작은 모델: Zamba2-1.2B/2.7B·Falcon-H1-3B·Granite-4-h-micro) | 16 GB이므로 **9B는 불가**(가중치만 ≈17 GB) |
| vLLM **성능 수치** | **근거로 쓸 수 없음** | 게이트 1. 5060 Ti는 A100과 다른 기판 |
| SGLang PD-mux 서빙 | **불가** | ①`sgl_kernel` 미설치 ②16 GB ③기판 상이. ★단 §9 참조 — "왜 불가"의 일부는 **미확인**이다 |
| CPU 회귀·분석·문서 | **가능** | 현 기준선 **519 tests / failures=7 / errors=36 / skipped=11**, 대부분 엔진 dev tree 부재 아티팩트 |

## 9. ★미확인 사항 (단언하지 말 것)

1. **`sgl_kernel`의 sm_120 빌드 유무는 확인되지 않았다.** `CLAUDE.md` "환경 / 실행"은
   *"설치된 `sgl_kernel`에도 sm120 빌드가 없다"* 고 적지만, 로컬에는 **`sgl_kernel`이
   설치조차 되어 있지 않다**(실측). 설치를 시도한 기록도 없다 ⇒ 이 서술은 **미검증**이며
   doc-steward 회부 대상(`gpu_rental_checklist_2026-09-18.md` §5의 arch 서술 정정과 같은 묶음).
2. **로컬 GPU의 compute capability는 `12.0`이다**(major 12), `CLAUDE.md`가 함의하는
   "major 10"이 아니다. 자동 격자 경로에서 `else` 분기에 걸리는 결론은 같지만 숫자가 다르다.
   그리고 우리 경로는 `manual_divisions`라 그 분기를 **타지 않는다**(같은 §5 회부 건).
3. **`create_greenctx_stream_by_value`가 sm_120에서 동작하는지 미측정.** 드라이버
   580.178.04는 green context 요구(≥12.4)를 충족하므로, 실패한다면 사유는 드라이버가
   아니라 커널 빌드·아키텍처 쪽이다 — **어느 쪽인지 확인된 바 없다**.
4. **Granite-4 체크포인트의 실제 `architectures` 값**이 `GraniteMoeHybridForCausalLM`인지 미대조.
5. **최신 vLLM(0.28.0)의 실제 의존 핀**(torch/transformers/flashinfer)은 미확인 — §1 표는
   로컬 0.12.0 기준이다. 버전을 정한 뒤 그 버전에서 다시 채취해야 한다.
6. 두 엔진의 **TTFT/ITL 측정 지점 일치 여부** 미확인(§6).

## 10. 출처

**1차(이 세션 실측)**: 로컬 `nvidia-smi` · `torch.cuda.get_arch_list()` ·
`importlib.util.find_spec` (sgl_kernel/vllm/mamba_ssm/causal_conv1d/flashinfer/flash_attn) ·
`vllm.__version__` = 0.12.0 · `ModelRegistry.get_supported_archs()`(245 arch, 4모델 확인) ·
`importlib.metadata.requires('vllm')`(torch==2.9.0 · transformers<5,>=4.56.0 ·
flashinfer-python==0.5.3) · `vllm.model_executor.layers.mamba.ops.*` 모듈 존재 ·
저장소 `env/venv_packages_2026-09-17.txt` · `results/serving-eval/vllm_zamba2_sharegpt_753640.log`
(v0.22.1 배너, Zamba2-7B-Instruct, max_model_len 4096, gpu_mem 0.85, max_num_seqs 64).

**2차(문서 인용)**: `deprecated/reports/historical_root/vllm_validation.md`(0.22.1,
torch 2.11+cu130, 2026-06-21) · `reports/paper/venue_positioning.md:112`(vLLM v0.20.0
disagg 분류) · `CONSENSUS.md §1-3`·§3 항목103(백엔드 강제 교락).

**3차(웹, 2026-09-18 검색)**: vLLM stable **0.28.0**(2026-08-26) · hybrid 모델
first-class(Qwen3-Next·Nemotron Nano 2·Granite 4.0) · Mamba hybrid prefix caching·
ReplaySSM·FlashInfer Mamba SSU · hybrid SSM disaggregated serving blog(2026-04-21) ·
`mamba-ssm`/`causal-conv1d`는 vLLM에서 선택적(없으면 느린 reference 폴백).
⇒ **웹 출처는 버전을 정할 때 1차로 재확인할 것**(릴리스는 계속 움직인다).
