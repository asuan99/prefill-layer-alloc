# C-2 — "mamba 층과 attention 층은 SM에 다르게 반응하는가": 계측 타당성 평가와 설계 (2026-08-11, engine-porter)

> **문서 지위**: 설계 문서. **구현 아님, 결과 아님, 성능 주장 없음.**
> `%smid` 설계 문서(`SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md`)와 같은 형식.
> 사전등록이 아니다(§8의 결정 규칙은 **후보**이며 확정이 아니다).
>
> **공유 dev 트리 무변경.** 이 조사 중 `/scratch/ehmoon/whlee/sglang_engine_dev/`에
> 쓰기 작업은 **0건**이다(읽기 grep·diff만). 프로토타입은 세션 스크래치패드에서
> CPU 산술만 했고, GPU 커널 실행 0건, git 조작 0건.

---

## 0. 읽는 순서 (요약)

1. **§2를 먼저 읽어라.** 부모 에이전트의 첫 질문("계측이 이미 있는가")의 답이고,
   이 문서에서 가장 중요한 발견이다. **계측은 있다. 그리고 이미 한 번 돌았고,
   자기 게이트에 실패했다.**
2. **§3 검증표** — *확인한 것*과 *추정한 것*이 기계적으로 분리돼 있다.
3. **§9("닫히지 않는 것")을 §5(권고)보다 먼저 읽어라.** 방법론 게이트 #28의 요구다.
4. §6이 (L,B,SM) 격자의 산술 도출이다. **arm별 도달 상한이 다르고, 그 차이가
   교차-arm 비교를 구조적으로 막는다** — 이게 §6의 결론이다.

**한 줄 결론**: per-layer-type 계측은 **새로 만들 필요가 없다** — 세 hybrid 모델
(`zamba2.py`/`nemotron_h.py`/`granitemoehybrid.py`)에 CUDA-event 기반 층-타입 타이밍이
**이미 있고 dev 트리에 설치돼 있다**. 문제는 계측의 **부재**가 아니라 **이미 판명된
오염**이다: 2026-08-05 job 873783이 이 계측을 SM 스윕에 돌렸고, **GATE N(mamba decode의
ctx-불변성)이 FAIL**했다 — 기전은 host-paced forward에서 idle이 열려 있는 span에
청구되는 것이고, 하필 **고-SM·단-ctx 코너**(이 질문이 필요로 하는 바로 그 코너)에서
per_mamba가 최대 **2.78× 부풀고 SM에 대해 비단조**가 됐다. 따라서 이 설계의 본체는
"어떻게 재나"가 아니라 **"어떻게 device-bound 영역으로 이동하나"**이고, 그 이동은
부모가 트래픽 논거로 요구한 **B·L 상향과 정확히 같은 방향**이다(§6.4 — 우연 아님).
권고는 **P0(무료 CPU 감사) → P1(단일 셀 host-floor + GATE N, ≈0.5 GPU-hr)** 2단이며,
**P1이 실패하면 eager 사다리는 죽고 nsys 경로로 넘어간다.** 운영점(cudagraph-ON)
per-layer 계측은 **구조적으로 불가능**하다(§3-V12) — 이 캠페인은 정의상 off-operating-point다.

---

## 1. 무엇을 사려는가 — 그리고 그게 코드 사실인지 추정인지 (방법론 게이트 #28)

게이트 #28: *"실험이 무엇을 풀어주는지가 코드 사실인지 추정인지, 실행 전에 코드로
확인한 뒤 정당화하라."* 이 게이트는 이번 세션에 신설됐고 이미 세 번 payoff 서술이
틀린 것으로 드러났다. 아래는 **낙관 없이** 적은 확인 결과다.

### 1.1 ★ 먼저: 이것은 layer-type **정책**을 되살리려는 게 아니다

**두 명제는 다르다.**

| | 명제 | 지위 |
|---|---|---|
| (P) | layer-type **정책**(attn/mamba에 SM을 다르게 배분) | **全형태 死** — 서빙 실증으로 확정. `reports/per_layer_type_postmortem.md` |
| (I) | layer-type **계측**(두 층 타입의 시간을 분리 측정) | **필요하고, 현재 7-8B 엔진 격자에 0건** |

정책이 죽은 자리는 **(C2) 착취 비용**이다 — coordinated per-type TPOT 42→124ms,
`PDMUX_LA_COORD_OPT`로도 85ms, 잔차는 구조적 오버랩 손실 + cudagraph 비양립.
postmortem §2 표가 보이는 구조는 **"(C1) 성립 ⟺ (C2) 실패"가 모든 행에서 성립**이다.
⇒ **이 캠페인이 (C1)에 어떤 답을 내놓아도 (P)의 판정은 한 눈금도 움직이지 않는다.**
Diff B가 1.0으로 나오든 10으로 나오든, 죽은 링크는 C2다. 이 문장에 동의할 수 없다면
이 실험을 사지 마라.

그럼 무엇을 사는가 → §1.2.

### 1.2 이 측정이 바꾸는 정본 문장 (코드·문서에서 직접 지목)

| # | 대상 문장 | 현재 지위 | 이 측정이 바꾸는 것 |
|---|---|---|---|
| **A** | `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md` §9 "말할 수 없다" 1번: **"per-layer-type 귀속 0건"** | 원문 그대로 | ★ **직접 해소된다** — 측정된 셀에 한해 0건이 아니게 된다. **이것이 유일하게 확실한 payoff다** |
| **B** | `PROJECT_STATUS.md` "8B decode-SM 민감도 측정 노트" 2026-08-11 스코프 주석: "'모델-무관'의 원인은 … 트래픽의 68–94%가 **weight sweep**이기 때문" | **[C] 트래픽 회계** — 진단 자신이 §9-1에서 "**시간 회계가 아니다**"라고 자백 | 같은 기전 주장을 **[M] 시간 측정**으로 승격/반증. 단 **모델 내부 대비에 한정**(§9-4) |
| **C** | 같은 진단 §6.6 / §9-4: 고-SM 평탄화 원인 후보 (i)wave quant (ii)**층 직렬 지연** (iii)점유율 (iv)cudagraph 직렬화 — "커널 단위 측정 0건이라 배제·선택 불가" | 4후보 미분리 | **부분 분리**: 평탄화가 세 층 타입에 **균일**하면 (ii)에 유리, **한 타입에 집중**되면 (i)/(iii)에 유리. ⚠️ **(iv)는 검정 불가** — eager 계측이라 cudagraph가 애초에 없다(§9-3) |
| **D** | `per_layer_type_postmortem.md` §0/§1-A: **decode Diff B ≈ 4.0×** (`decode_vs_prefill_sensitivity.png`) | **characterization 마이크로 러그**의 수치. 방법론 게이트 #1이 엔진 결론으로의 사용을 금지 | **엔진 러그의 첫 Diff B 관측치**를 준다. 단 (P) 판정 불변(§1.1) |

**즉시 나오는 정직한 결론**: 이 실험이 여는 **새 성능 주장은 0개**다. 사는 것은
(a) 진단 §9-1의 "0건"이 없어지는 것, (b) 기전 주장 하나가 회계에서 측정으로 옮겨가는 것,
(c) 고-SM 평탄화 4후보 중 **최대 2개**의 상대적 지지도. **§1-1의 PD-분리 귀속·goodput·
프론티어·SLO에는 한 글자도 닿지 않는다.**

---

## 2. ★★ 기존 계측 인벤토리 — "이미 있는가"에 대한 답

**답: 있다. 세 모델에. dev 트리에 설치돼 있다. 그리고 이미 SM 스윕에 한 번 돌았다.**

### 2.1 코드 인벤토리 (전부 직접 읽어 확인)

| 모델 | env 게이트 | 버킷 | 누산기 | 이벤트 | closure 게이트 | shape guard | emit 태그 | **sync/manifest** |
|---|---|---|---|---|---|---|---|---|
| **Zamba2** `src/models/zamba2.py` | `SGLANG_ZAMBA_TIMING`<br>`_TIMING_EVERY`·`_CLOSURE_MIN`·`_PREFILL_KNEE` | `attn`/`mlp`/`other`/`mamba` + 진단 `attn_core` | **매 emit마다 블록 리셋** (+legacy 러닝평균 병기) | **풀링·재사용** (`_zt_event`, :90-98) | **있음** (독립 이벤트 쌍 분모, :594-624) | **있음** (`nseq,ntok`, `dirty`) | `ZBLT2`/`ZBPT2` | ★ **YES** — `sync_engine_tree.sh`가 설치, SHA-256 manifest 포함 |
| **NemotronH** `src/models/nemotron_h.py` | `SGLANG_NH_LAYER_TIMING` (:646) | `M`/`-`/`*` (층 전체 span, 루프를 타일링) | ★ **러닝평균, `_mode` 변경시에만 리셋** (:672-674, :729-737) | ★ **매 층 매 forward마다 신규 Event 2개** (:710) | 없음 | 없음 | `NHLT` | ✗ **NO** — 수동 복사본(`dev_tree_edits.md` 항목 6/8/9), manifest 밖 |
| **Granite** `src/models/granitemoehybrid.py` | `SGLANG_GRANITE_TIMING` (:474) | `M`/`*` **2개뿐** | ★ 동일 러닝평균 (:512-521) | ★ 동일 신규 Event (:494-495) | 없음 | 없음 | `GMHLT` | ✗ **NO** — 수동 복사본 |
| **Falcon-H1** `src/models/falcon_h1.py` | — | **없음** | — | — | — | — | — | — |

**⇒ 2026-08-04 P5 계측 리워크(버킷 대칭화·블록 누산기·shape guard·closure 게이트·
이벤트 풀링)는 Zamba2에만 적용됐다.** NemotronH·Granite는 **리워크 이전 형태**를
그대로 들고 있다 — 특히 **defect (2) 러닝평균**이 살아 있어, n=30·60·90 emit이
각각 "1..30 / 1..60 / 1..90의 누적 평균"이라 **정상상태를 분리할 수 없다**.
(mirror ↔ dev 트리는 세 파일 모두 `diff` **IDENTICAL**이므로 드리프트는 없다.)

### 2.2 SM 축을 어떻게 거는가 (세 모델 공통)

`_get_gctx_decode_stream(n_sm)` (zamba2.py:492 / nemotron_h.py:608 / granitemoehybrid.py:415)이
`sgl_kernel.spatial.create_greenctx_stream_by_value(n_sm, max(4, total-n_sm), dev)`로
**자기 자신의 green ctx 쌍을 lazy 생성**한다. arm은 `PDMUX_FIXED_DECODE_SM_FILE`이
가리키는 파일을 **decode forward마다 새로 읽어** 고른다(zamba2.py:523-528).

★ 두 가지 결과:
- **pdmux 불필요.** r0c 하네스는 `--enable-pdmux` 없이 돌았다 ⇒ 이 캠페인은
  **standalone 단일 서버**로 살 수 있다(§11 비용의 근거).
- **arm 파일 읽기가 forward마다 발생하는 syscall**이다 ⇒ host 경로 비용에 기여한다(§7).

### 2.3 ★ 이미 돌았다 — job 873783 (2026-08-05)과 그 판정

`results/r0c/decode_knee_vs_ctx_v2.sbatch` + `analyze_decode_knee_vs_ctx_v2.py`,
판정문 `results/r0c/P5_GATES_BATCH1_2026-08-05.md` (**UNAUDITED·정본 아님**).

- 설계: **Zamba2-2.7B**, ctx {256,1024,4096,16384} × SM {8,16,24,44,full} × n=1,
  conc32, outtok32, **eager**(`--disable-cuda-graph --disable-piecewise-cuda-graph`),
  triton attn, `--disable-radix-cache`, mem-frac 0.80, max-running-requests 48,
  랜덤화된 arm 순서 + 폐기 warm-up 라운드 + 블록 전량 보존.
- 판정: **GATE C(closure ≥0.85) 20/20 PASS**, ★ **GATE N(mamba decode의 ctx-불변성) FAIL**
  (bs=32 부분집합에서도 full 2.78× / sm44 1.97× / sm24 1.28× 위반), **G6 = NO-GO for
  literal repeats**.
- 기전(그 문서의 §2.4, 서빙 직접 측정): 위반 셀은 정확히 **host-paced** 셀이다 —
  `fwd_ms`가 device work가 2–8× 다른 arm들에서 **같은 ~44–46 ms 바닥에 고정**된다
  (ctx256: full 44.0 / sm44 45.4 / sm24 45.5). CUDA-event span은 GPU 타임라인 경과를
  재므로, device가 굶으면 그 idle이 **열려 있는 span에 청구**된다. 런치가 많은 span
  (mamba mixer, `other`)은 흡수하고 2-GEMM `mlp` span은 흡수하지 않는다 —
  그래서 `per_mlp`는 0.4%로 깨끗하고 `per_mamba`·`per_other`가 **같은 순위로** 깨진다.
- ★ **이 질문에 치명적인 대목**: `per_mamba`가 **SM에 대해 비단조**였다 —
  ctx256에서 8/16/24/44/full = 0.9598/0.5305/0.4976/0.5186/**0.5546**,
  ctx1024에서 0.9625/0.5296/0.3900/0.3413/**0.4623**. 즉 **"SM을 더 주면 층이 느려지는"
  물리적 위반**이, 하필 우리가 필요로 하는 **고-SM 코너**에서 발생했다.
- 배제된 대안(그 문서가 데이터로 배제): green-ctx 아티팩트(최대 위반이 green ctx를
  안 쓰는 `full` arm) · JIT/warm-up(라운드 간 ±0.5%) · DVFS/열(부호 반대) ·
  L2/KV 압박(부호 반대) · 노드 이질성(전부 gpu41).
  **배제되지 않음**: co-tenancy, **계측 자신의 host 비용**(명시적으로 "untested").

**⇒ 이 설계가 상속하는 것은 "계측을 만들어라"가 아니라 "G6의 미해결 조건을 만족시켜라"다.**
그 문서가 나열한 후보 처방 (a)–(e) 중 이 설계가 채택하는 것은
**(b) device work를 host floor 위로 올린다** + **(e) 직렬화·계측 자기비용 측정**이다.
(c)(cudagraph-ON으로 이동)는 §3-V12로 **원리적으로 불가**임이 이번에 확인됐다.

### 2.4 마이크로 러그에는 이미 (B,L) 스윕이 있다 — 그리고 이상 징후가 있다

`workspace/characterization/`는 **엔진 밖 단일 층 실행기**를 갖고 있다:
`src/models/layer_runner.py`(`run_ssm_layer`/`run_attn_layer`/`run_mlp_layer`/
`verify_sm_control`), `experiments/e3_decode_floor/`(축: batch{8,32,128,256} ×
ctx{1k,4k,16k} × SM 격자).

`results_v2/e3/decode_floor_zamba2_7b_a100_sxm4_80gb.csv`는 **이미 Zamba2-7B를
batch 1→512, ctx4096에서 층 타입별로 쟀다**:

```
attn : b=1..8 floor_sm 68–81 → b≥16 floor_sm 108,  flat_bw% 42–64
ssm  : b=1 floor_sm 27 → b=4 40 → b=8 68 → b=16 81 → b≥32 108,  flat_bw% 0.003–0.115
```

★ **두 가지를 동시에 말해야 한다**:
1. **"마이크로 벤치로 (B,L)을 올려 보라"는 이미 지불됐다.** 새 마이크로 캠페인은
   새 정보를 거의 안 산다. **비어 있는 것은 엔진 러그다.**
2. ★ **그 데이터는 그대로 쓰면 안 된다** — `ssm`의 `flat_region_bw_util_pct`가
   **0.003–0.115%**다. state read+write가 대역폭의 0.1%일 수는 없으므로
   **바이트 회계 또는 측정 자체에 결함이 있다**(미조사). 게다가 파일명이 주장하는
   하드웨어는 **A100 SXM4**인데 엔진 캠페인의 실측은 **A100 80GB PCIe**다
   (진단 §6.2, `nvidia-smi -q`) — **러그 간 하드웨어 불일치 미해소**.
   ⇒ 이 CSV를 **엔진 결론의 근거로도, 예측의 앵커로도 인용 금지**. 감사 전까지는
   "마이크로 러그에 (B,L) 스윕이 존재한다"는 **사실**만 인용 가능.

---

## 3. ★ 검증표 — 이 문서에서 **확인한 것 / 추정한 것**

미검증 가정이 이 프로젝트의 결론을 두 번 뒤집었으므로 분리해 둔다.

### 3.A 이 세션에서 **직접 확인**한 것 (재현 명령은 부록 A)

| # | 사실 | 확인 방법 | 결과 |
|---|---|---|---|
| V1 | 세 hybrid 모델에 per-layer-type CUDA-event 계측이 **존재** | `grep -n "SGLANG_.*TIMING\|_lt_acc\|_ZT\["` on `src/models/*.py` | zamba2 / nemotron_h / granitemoehybrid = 존재, falcon_h1 = 0건 |
| V2 | mirror ↔ dev 트리 **드리프트 0** | `diff -q src/models/{f}.py $DEV/sglang/srt/models/{f}.py` | 3/3 **IDENTICAL** |
| V3 | zamba2.py만 sync·manifest에 들어 있다 | `sync_engine_tree.sh` 본문 (install 목록 + `sha256sum` 목록) | nemotron_h/granitemoehybrid는 **없음** ⇒ 수동 복사본 |
| V4 | NemotronH/Granite 누산기는 **러닝평균**(defect 2 형태) | nemotron_h.py:672-674·729-737, granitemoehybrid.py:512-521 | `_lt_acc`는 `_mode` 변경시에만 0; emit은 `acc/n` 누적 평균 |
| V5 | 두 모델은 층마다 **신규 Event 2개**를 만든다(풀링 없음) | nemotron_h.py:710, granitemoehybrid.py:494-495 | Zamba2(`_zt_event` 풀)와 host 비용이 **비대칭** |
| V6 | Zamba2 계측은 2026-08-04 리워크본 | `_ZT_BUCKETS`·`_ZT_DIAG`·closure 분모·`_zt_blk`·`ZBLT2` 태그 | `env/dev_tree_edits.md` 항목 17과 일치 |
| V7 | SM 핀은 **pdmux와 독립**이며 green ctx를 모델이 직접 만든다 | `_get_gctx_decode_stream` (3파일) → `spatial.create_greenctx_stream_by_value(n, max(4,total-n))` | r0c 하네스는 `--enable-pdmux` 없이 실행 |
| V8 | arm 선택은 **decode forward마다 파일 읽기** | zamba2.py:523-528, nemotron_h.py:652-657 | `open(_f).read()` — host 경로 syscall |
| V9 | 이 계측은 **이미 SM 스윕에 돌았다** | `results/r0c/{decode_knee_vs_ctx_v2.sbatch, P5_GATES_BATCH1_2026-08-05.md}`, job 873783 | Zamba2-2.7B, 4 ctx × 5 SM × n=1, eager |
| V10 | 그 실행은 **GATE N FAIL**, G6 **NO-GO** | P5_GATES §2·§7 | per_mamba 최대 2.78× 부풀림; **SM에 비단조**(§2.3 인용) |
| V11 | 그 실패 기전은 **host-pacing**이고, `mlp` span은 오염되지 않는다 | P5_GATES §2.3-2.4 | `per_mlp` ctx-불변 0.4%; `per_mamba`·`per_other`가 함께 깨짐 |
| V12 | ★ **cudagraph-ON에서 per-layer span은 실행되지 않는다** | `cuda_graph_runner.py:547` `capture_forward_mode=ForwardMode.DECODE`; `:1161` `self.graphs[graph_key].replay()` | replay는 Python forward를 **호출하지 않는다** ⇒ 계측은 **구조적으로 eager 전용** |
| V13 | 두 캠페인이 cudagraph 경계 **반대편**에 있다 | s8 srv 로그 `disable_cuda_graph=False`(capture bs [1..48]) vs `decode_knee_vs_ctx_v2.sbatch:114` `cudagraph=DISABLED … not the operating point` | s8=운영점·per-layer 0건 / r0c=eager·per-layer 有 |
| V14 | sglang에 **stock layerwise NVTX 훅**이 있다 | `server_args.py:616` `enable_layerwise_nvtx_marker`; `model_runner.py:1196-1198` → `PytHooks.register_hooks`; `utils/nvtx_pytorch_hooks.py:286-287` `register_forward_pre_hook/forward_hook` | Python 훅이므로 **V12와 같은 이유로 eager 전용** |
| V15 | nsys가 설치돼 있고 `--cuda-graph-trace=<granularity>`를 지원 | `/apps/cuda/13.0.2/bin/nsys --version` → 2025.3.2.474; `nsys profile --help` | granularity 옵션 **존재**(동작은 미검증 → E5) |
| V16 | ctx 상한 (config 직독) | `hf_cache/hub/models--*/snapshots/*/config.json` | **Zamba2-7B-Instruct 4096** · **NemotronH-8B-Base-8K 8192** · Qwen2.5-7B 131072 · Granite-4.0-h-micro 131072 · Falcon-H1-7B 262144 · Mamba-Codestral-7B 키 없음 |
| V17 | 메모리 앵커 (서버 로그 직독, mem-frac 0.80 / max-running-requests 48) | `s8_deconf_{Hs8,Ha8}_C1024_d16_865533_srv.log` | Hs8 weights 15.11 GiB · mamba(48) 4.66 · KV 43.34 ⇒ **풀 예산 ≈48.0 GiB**<br>Ha8 weights 13.97 · mamba(48) 6.94 · KV 42.20 ⇒ **≈49.2 GiB** |
| V18 | `max_mamba_cache_size`는 `max_running_requests`에서 파생 | `model_runner_kv_cache_mixin.py:228` | B를 올리면 mamba 슬롯 메모리도 비례 증가 |
| V19 | ★ **mamba state 주기 체크포인트가 존재** | `schedule_batch.py:2170` `seq_lens_cpu % mamba_track_interval == 0`; 기본값 256 (`server_args.py:552`), CLI `--mamba-track-interval` (:5091) | mamba 경로 안의 **조건부 추가 작업** ⇒ O(1)-in-ctx 음성대조의 **신규 교란원**(§8-A1) |
| V20 | 층-타입별 weight/트래픽 분해가 진단의 총계를 재현 | 스크래치 `ltsm/grid.py` (config 산술) | Hs8 층 weights **14.05 GB**(진단 "층 14.05") · Ha8 shared block **668 MB/호출** ⇒ 13호출 **8.68 GB**(진단 8.684) |
| V21 | backend가 **arm마다 다르다** | `FINDINGS_8B_2026-07-28.md` §1 | NemotronH=flashinfer(triton은 어서션 금지) / Zamba2=triton(flashinfer는 cudagraph capture 사망) ⇒ **attn 버킷이 서로 다른 커널 구현** |
| V22 | 마이크로 러그에 (B,L) 스윕이 이미 있다 + 이상 징후 | `results_v2/e3/decode_floor_zamba2_7b_*.csv` | batch 1→512 ctx4096; **ssm `flat_bw%` 0.003–0.115%** (설명 안 됨), 파일명 하드웨어 = **SXM4**(엔진은 PCIe) |
| V23 | Zamba2 계측 CPU 회귀는 관측 범위에서 통과 | `python -m unittest … test_zamba2_instrumentation -v` | 관측된 **9건 전부 ok, 실패 0**. ⚠️ 최종 요약 라인 미확보(로그인 노드 실행이 잘림) — **"전체 통과"로 인용 금지** |
| V24 | 전체 `unittest discover`는 이 세션에서 **채점 못 했다** | 동일 환경, 7분 초과 stall | ★ **측정 실패이지 게이트 실패가 아니다**(방법론 게이트 #21). 컴퓨트 노드에서 재실행 필요 |

### 3.B 이 문서가 **추정**한 것 (검증 안 됨 — 프로브 필요)

| # | 추정 | 왜 지금 확인 못 했나 | 어떻게 확인하나 |
|---|---|---|---|
| E1 | 7-8B eager의 **host floor** 높이 (2.7B·54층의 ~44 ms에서 층 수로 외삽 ⇒ Zamba2-7B(81층) ~66 ms, NemotronH-8B(52층) ~42 ms) | GPU 필요. 층 수 외삽은 층 종류별 런치 수를 무시한다 | **§11-P1** (직접 측정) |
| E2 | **계측 자신의 host 비용** | P5_GATES §2.4가 "untested"라고 명시. 이 세션에서도 미측정 | §11-P1 (`TIMING=0/1` 대칭 대조) |
| E3 | `mem_fraction_static`을 0.80 초과로 올렸을 때의 풀 예산 | V17은 0.80 실측 1점뿐 | §11-P1 부팅 로그 |
| E4 | 목표 셀의 실제 step 시간 | s8의 achieved_BW를 arm 내부에서 외삽했으나, **진단 §9-5가 다른 (B,L)로의 이식을 금지**한다 | §11-P1 |
| E5 | nsys `--cuda-graph-trace=node`가 replay된 sglang decode 그래프에서 **커널 행을 낸다** + 컴퓨트 노드 권한 | 실행 안 함. `--version`·`--help`만 확인(V15) | §11-P4 |
| E6 | 커널명→층 타입 귀속의 **완전성** (cuBLAS GEMM이 mamba in/out_proj와 attn qkv/o_proj에 공유됨) | 트레이스가 없어 확인 불가 | §11-P4 (그래프 노드 **순서**로 분절 가능한지) |
| E7 | `Event(enable_timing=True, external=True)`가 캡처된 그래프 안에서 **쓸 수 있는 시간을 준다** | torch 2.9.1 docstring이 `external`이 "event record/wait **노드**를 만든다"고 문서화하지만, **timing 의미론은 침묵** | 별건. 이 설계는 여기에 의존하지 않는다 |
| E8 | `--mamba-track-interval`을 크게 걸면 체크포인트가 발화하지 않는다 | 어서션 2개(page_size 배수, spec 토큰 수 이상)만 확인 | §11-P1 (마스크 발화 계수) |
| E9 | 각 비용 수치(§11) | r0c 태스크 소요에서 외삽 | 스모크(방법론 게이트 #26) |

---

## 4. 계측 경로 비교표

"licence"는 *성공했을 때 쓸 수 있게 되는 문장*이다.

| # | 경로 | 실제로 무엇을 재는가 | 부착 지점 (file:line) | 오버헤드 | licence **하는** 것 | licence **못 하는** 것 | 주 실패 모드 |
|---|---|---|---|---|---|---|---|
| **1** | ★ **기존 CUDA-event 층 span 재사용** (권고) | 한 decode forward 안에서 층 타입별 **GPU 타임라인 경과 합** | 이미 있음: zamba2.py:594-624 / nemotron_h.py:708-737 / granitemoehybrid.py:492-521. **부착 작업 0줄** | forward당 `torch.cuda.synchronize()` 1회 + span당 Event 2 + emit당 `elapsed_time` 다수 (**E2 = 미측정**) | 같은 forward 안 **모델 내부** 층-타입 대비(파라미터·backend·노드·서버 confound가 **구성상 소거**) | 운영점(V12) · 커널 내부 · 교차-arm 비교(V21) · **host-bound 셀**(V10) | ★ **V10의 재발** — idle이 span에 청구돼 per_mamba 부풀림·비단조 |
| **2** | **nsys 커널 타임라인 + 커널명 분류** | 실제 커널의 시작/종료. `--cuda-graph-trace=node`면 **그래프 노드까지** | 엔진 무패치. 외부 프로세스 | 수 %(추정). **직렬화 없음**(ncu와 다름) | ★ **운영점(cudagraph-ON)에서의 층-타입 시간** — 경로 1이 원리적으로 못 하는 것 | GEMM 귀속 모호(E6) · Granite/NemotronH attn = flashinfer(precompiled) · **SM 정체성 0** | E5(그래프 노드 트레이스 미검증) · 권한(E5) · 커널명→타입 매핑이 새 자유모수 |
| **2b** | **stock `--enable-layerwise-nvtx-marker`** (V14) | 모듈 단위 NVTX 구간 | `model_runner.py:1196` (엔진 무패치, 플래그만) | 훅 2개/모듈 | eager에서 모듈 경계를 nsys 타임라인에 **라벨링** | ★ V12와 **같은 이유로 replay에서 실행 안 됨** ⇒ 경로 2의 운영점 장점을 못 살림 | 경로 1과 같은 러그. 경로 2의 보조로만 가치 |
| **3** | **ncu / CUPTI PM** | 커널·SM 카운터 | 외부 | ★ **metric replay가 직렬화** | (권한 있으면) 커널 점유율·wave | 층 타입 시간은 **직렬화 후의 것**이라 우리 질문의 대상이 아님 | `RmProfilingAdminOnly: 1`(smid V10) + **측정 대상 파괴**. §9-C의 (i)(iii) 후보 조사용으로만 별건 |
| **4** | ★ **층 구성 차분 설계** (부모 제안, instrument-free) | `num_hidden_layers`/`hybrid_override_pattern`을 바꾼 **truncated config** 2종의 **whole-step ITL(SM)** 차이 | 모델 코드 무패치. config + weight 로더 shim(`make_mamba2_hf_sglang.py` 전례) | **프로브 0** | 층 타입의 SM 응답을, **이미 신뢰되는 측정량(step ITL)**만으로 | ★ **config 간 confound**(weight·풀 크기·런치 수 동시 변경). 층을 줄이면 device work도 줄어 **더 host-bound**가 된다 | 로더 shim · truncated 모델의 수치 무의미(타이밍 전용임을 선언해야) |
| **5** | **마이크로 러그 재실행** (`layer_runner.py`, e3) | 엔진 밖 단일 층 | `workspace/characterization/` | 낮음 | 엔진 측정이 **검정할 예측**의 생성 | ★ **엔진 결론 금지**(방법론 게이트 #1). V22의 미해명 이상(ssm flat_bw 0.1%)·하드웨어 불일치 | 이미 지불됨(§2.4) ⇒ **새로 사지 않는다** |

**표에서 즉시 나오는 결론 4개**

1. **경로 1은 부착 비용이 0이다** — 이 설계의 공수는 계측이 아니라 **동작점 이동과
   admissibility 게이트**에 들어간다.
2. **경로 1은 운영점을 원리적으로 못 본다**(V12). 운영점을 원하면 **경로 2뿐**이고,
   경로 2는 E5/E6이 열려 있다. **두 경로는 서로의 대체재가 아니라 서로 다른 러그다.**
3. **경로 3은 채택하지 않는다** — 직렬화가 측정 대상을 파괴한다(smid 문서 §3 결론 1과 같은 논리).
4. **경로 4는 실재하는 대안이지만 "프로브 아티팩트"를 "config 간 confound"로 바꿀 뿐**이고,
   층을 줄이면 host-bound가 **악화**되므로 V10 문제를 해결하지 못한다. 차선으로만 등재.

---

## 5. 권고 설계 — 사다리 L0 → L1 (→ L2 보류)

### L0 — **무료 CPU 감사** (권고 1순위, GPU 0)

1. **NemotronH·Granite 계측을 2026-08-04 리워크 체크리스트로 감사**한다:
   블록 누산기(defect 2) · shape guard · 이벤트 풀링 · closure 게이트 가능 여부.
   결과가 "미이식"이면 §11의 공수 항목에 그대로 들어간다(예상: 미이식).
2. **manifest 격리 계획**을 먼저 고정한다. 두 파일은 지금 manifest **밖**이다(V3).
   sync에 넣는 순간 **기존 캠페인의 manifest가 바뀐다** ⇒ G1-a 대기 상태를 확인하고
   **별도 캠페인 manifest로 격리**한다(smid 문서 §10-P5와 같은 규율).
3. `unittest discover`를 **컴퓨트 노드에서** 완주시킨다(V24는 채점 아님).

### L1 — **단일 셀 admissibility 프로브** (권고 2순위, ≈0.5 GPU-hr) — **이 설계의 생사**

**Zamba2-7B 하나, 목표 셀 하나**에서 다음을 동시에 얻는다:

```
boot: Zamba2-7B-Instruct, eager, triton, --disable-radix-cache,
      --max-running-requests 96, --mem-fraction-static 0.80,
      --mamba-track-interval <크게>, --context-length 2560
      SGLANG_ZAMBA_TIMING=1, SGLANG_ZAMBA_TIMING_EVERY=8
arms: SM ∈ {44, 92}  (PDMUX_FIXED_DECODE_SM_FILE)
cells: (L=1024, B≈88) × (L=2048, B≈88)   ← 같은 realized bs, 다른 ctx
plus:  같은 (L=1024,B≈88)에서 SGLANG_ZAMBA_TIMING=0 의 step time
```

여기서 **네 가지가 한 번에 판정된다**:
- **A1 GATE N(재정의판)**: `per_mamba`가 ctx 1024↔2048에서 **같은 realized bs**로
  불변인가. (P5_GATES §2.5가 요구한 재정의 — 옛 게이트는 ctx와 bs를 혼동했다.)
- **A2 host floor**: `fwd_ms`가 SM44와 SM92에서 **같은 값에 고정되는가**. 고정되면
  그 셀은 device-bound가 아니다 ⇒ 캠페인이 그 셀에서 죽는다.
- **E2 계측 자기비용**: `TIMING=0/1`의 step time 차 (대칭 대조, §7).
- **E8 mamba-track**: 체크포인트 마스크 발화 계수 0 확인.

**A1·A2가 통과해야만** §6의 전체 격자를 산다. 통과 못 하면 §5-차선으로 간다.

### L2 — 전체 격자 (L1 통과 후에만)

§6의 (L,B,SM) 격자를 두 hybrid에서 돌린다. **NemotronH를 포함하려면 L0-1의
리워크 이식이 선행**돼야 한다(안 하면 러닝평균 때문에 정상상태를 못 뽑는다).

### 차선책 (권고가 막혔을 때)

1. **A1/A2 실패(= 7-8B eager도 host-bound)** → 경로 1은 **이 하드웨어에서 죽는다**.
   차선은 **경로 2(nsys)**이며, 그때는 **§11-P4를 먼저** 산다. 경로 4로 내려가지 않는다
   (층을 줄이면 host-boundness가 악화되므로 같은 벽에 더 세게 부딪힌다).
2. **NemotronH 리워크 이식이 correctness gate를 못 넘음** → **Zamba2 단독**으로 간다.
   단일 모델 결과를 "hybrid 일반"으로 승격하지 않는다(§9-4).
3. **A1은 통과하는데 SM 비단조가 남음** → 그 자체를 **보고**하고 Δε 계산을 **중단**한다.
   비단조는 추정량 결함의 신호이지 아키텍처 발견이 아니다(V10의 교훈).

---

## 6. 동작점 — (L, B, SM) 격자의 산술 도출

### 6.1 왜 B와 L을 올려야 하는가 — **층 타입 안쪽**의 회계

진단 §3.4는 **스텝 전체** 트래픽의 68–94%가 weight sweep이라고 했다. per-layer-type
계측에서 중요한 것은 그 총계가 아니라 **각 층 타입 안쪽**의 조성이다. 층 1회 호출당:

- mamba 층: `w_mamba` + `B · κ · state_per_layer`  (κ=2, **L에 불변**)
- attn 층 : `w_attn`  + `B · L · kv_per_token_per_layer`
- mlp 층  : `w_mlp`   (요청 고유 항 **없음**)

config 산술(V20, 진단 총계 재현):

| | `w_mamba` | `w_attn` | `w_mlp` | state/층/seq | KV/tok/층 |
|---|---|---|---|---|---|
| **Hs8** NemotronH-8B | 219.3 MB | 83.9 MB | 352.3 MB | 4.059 MiB | 4,096 B |
| **Ha8** Zamba2-7B | 156.9 MB | 359.7 MB | 308.3 MB | 1.792 MiB | 28,672 B |

**교차점 `B*`(그 층 타입의 요청-고유 트래픽 == 그 층의 weight sweep)**:

| | mamba 층 (L 불변) | attn 층 @L=1282 | @2048 | @4096 | @8192 |
|---|---|---|---|---|---|
| **Hs8** | **B\* = 25.8** | 16.0 | 10.0 | 5.0 | 2.5 |
| **Ha8** | **B\* = 41.7** | 9.8 | 6.1 | 3.1 | — (ctx 상한) |

★ **s8 측정점(Hs8 B=12 / Ha8 B=9, L≈1282)에서의 조성**:
mamba 층은 state 비중 **31.7% / 17.8%**(= 여전히 weight 지배),
attn 층은 KV 비중 **42.9% / 47.9%**. ⇒ **두 층 타입이 서로 다른 물리를 하고 있는지를
묻기에는, mamba 쪽이 아직 "weight를 훑는 GEMV"에 가깝다.**
**attn 층은 이미 절반이 KV이므로, 벌려야 하는 쪽은 주로 mamba 축(B)이다.**

### 6.2 arm별 도달 가능한 (L,B) — 메모리 예산 (V17 로그 앵커 기반)

`per_seq = mamba_slot + L × kv_per_token`, 예산 = 풀 예산(Hs8 48.0 / Ha8 49.2 GiB).

| arm | L | **B_max(mem)** | B=48 | B=64 | B=96 | B=128 |
|---|---|---|---|---|---|---|
| **Hs8** | 1024 | **425** | 65.1 / 70.6 | 71.3 / 76.2 | 78.8 / 82.8 | 83.2 / 86.5 |
| | 2048 | **374** | 65.1 / 82.8 | 71.3 / 86.5 | 78.8 / 90.6 | 83.2 / 92.8 |
| | 4096 | **300** | 65.1 / 90.6 | 71.3 / 92.8 | 78.8 / 95.0 | 83.2 / 96.2 |
| | 8192 | **216** | 65.1 / 95.0 | 71.3 / 96.2 | 78.8 / 97.5 | 83.2 / 98.1 |
| **Ha8** | 1024 | **98** | 53.5 / 79.7 | 60.5 / 83.9 | **69.7 / 88.7** | ✗ |
| | 2048 | **57** | 53.5 / 88.7 | ✗ | ✗ | ✗ |
| | 4096 | **31** | ✗ | ✗ | ✗ | ✗ |

(셀 값 = **mamba 층 state 비중 % / attn 층 KV 비중 %**. `✗` = 메모리 초과.)

### 6.3 ★ 결론 — arm 간 envelope이 다르고, 그 차이가 비교를 구조적으로 막는다

"두 층 타입 **모두** 아키텍처-고유 항이 ≥2/3"을 기준으로 하면:

- **Hs8**: `B ≥ 64`이면 L∈{1024,2048,4096,8192} **전부** 만족. 메모리 여유 3–6×.
- **Ha8**: **(L=1024, B≈96) 단 한 점**에서만 만족하고, 그 점은 `B_max=98`의
  **98%**다. L≥2048에서는 **어떤 B로도 불가능**(B_max 57 → mamba 57.7%).

⇒ **설계 제약(반드시 명시)**: 두 arm이 **동시에** 조건을 만족하는 matched 셀은
사실상 `(L=1024, B≈96)` 하나뿐이고 그마저 Ha8의 메모리 절벽 위다. 게다가 그 셀조차
**backend 비대칭(V21)** 때문에 attn 버킷이 서로 다른 커널이다.
⇒ **이 캠페인은 within-model 설계로 간다.** 각 hybrid가 "이 모델 안에서 mamba 층과
attn 층이 SM에 다르게 반응하는가"에 답하고, **arm 간 종합은 하지 않는다**
(기존 "arm 간 절대비교 금지"와 동일 규율).

**권고 격자** (L1 통과 조건부):

| arm | L | B | SM | n |
|---|---|---|---|---|
| Ha8 Zamba2-7B | 1024 | 88 | {44, 54, 74, 92} | 3 |
| Ha8 Zamba2-7B | 2048 | 48 | 동일 | 3 |
| Hs8 NemotronH-8B | 2048 | 96 | 동일 | 3 |
| Hs8 NemotronH-8B | 8192 | 96 | 동일 | 3 |

- **SM 축 44–92**: 저-SM은 굶주림이 아키텍처 차이를 덮는다(진단 §6.4 국소 탄력도
  ε(16→24)=0.83–0.90 vs ε(44→92)=0.16–0.41). 4점이면 구간 탄력도 + 곡률을 본다.
- ★ **`full`(green ctx 없음) arm은 라벨 참조로만 기록하고 SM 값으로 쓰지 않는다** —
  "무분할 = 108 SM"은 미검증 물리 주장이다(Gate 2-S rev4에서 철회, smid 문서 §1.1).
- **ctx 상한 준수**: Ha8은 `max_position_embeddings=4096`이라 L=2048+out이 안전한 상한
  근처다. Hs8은 8192.
- **`--max-running-requests`**를 B에 맞춰 올린다 ⇒ `max_mamba_cache_size`도 따라 오른다(V18).

### 6.4 ★ 두 요구가 같은 방향이라는 것 — 그리고 그게 우연이 아닌 이유

부모의 트래픽 논거는 "아키텍처 고유 항이 지배하도록 B·L을 올려라"이고,
§2.3의 host-pacing 처방은 "forward당 device work를 host floor 위로 올려라"다.
**둘 다 `B × (요청 고유 트래픽)`을 키우라고 말한다** — 전자는 *조성*을 위해,
후자는 *절대량*을 위해. 같은 노브가 두 문제를 동시에 민다.

⚠️ **그러나 "그러므로 해결된다"고 쓰면 안 된다.** floor 높이는 **미측정(E1)**이고,
계측 자기비용(E2)이 floor를 **올린다**. 두 곡선이 실제로 교차하는지는 §11-P1이
직접 재기 전까지 열린 질문이다.

### 6.5 ★ 이 격자는 **배포 동작점이 아니다** (전방 자백)

s8/서빙에서 관측된 decode batch는 **9–12**이고 `--max-running-requests`는 48이다.
이 설계는 B를 **64–96**으로 올린다. ⇒ **여기서 나오는 어떤 결과도 s8 격자·서빙
동작점으로 이식할 수 없다**(진단 §9-5의 이식 금지가 그대로, 방향만 반대로 적용된다).
사는 것은 "이 아키텍처의 층 타입들이 아키텍처-고유 트래픽 지배 영역에서 어떻게
반응하는가"이고, **"따라서 배포 시에도"는 금지 문장이다.**

---

## 7. 관측자 부하 — 어떻게 bound하는가

전제: 이 프로젝트의 관측자 효과 게이트 **G5는 UNDETERMINED**(TOST 재채점: 72셀 중
초과 지지 0 · 등가 29 · 검정력 부족 43)다. ⇒ **"프로브 무해"를 전제할 수 없고,
무해를 *증명*하는 통계 게이트를 새로 만들어서도 안 된다**(그게 G5와 같은 형태다).

**이 계측은 smid의 census와 달리 프로브가 무해하지 않다** — 서빙 루프 **안쪽**에서
돌고, forward마다 `torch.cuda.synchronize()`를 부르며, 그것이 **정확히 V10의 오염
기전을 키우는 방향**이다. 따라서:

1. **통계가 아니라 기전으로 bound한다.**
   - **직접 측정**: `TIMING=0` vs `TIMING=1`의 `fwd/step` 시간(§11-P1). 이것은
     "무해 입증"이 아니라 **부하의 크기를 숫자로 적어 아티팩트에 남기는 것**이다.
     결과가 크면 그것은 셀 admissibility(§8-A2)를 **좁히는 방향으로만** 쓴다.
   - **대칭**: 계측은 **모든 SM arm에 동일하게** 들어간다. 공통항이므로 arm 간
     대비에서 소거된다 — 단 **소거된다는 주장 자체를 통계로 증명하지 않는다.**
     소거는 설계상 보장이지 측정 결과가 아니다.
2. **비대칭 하나를 미리 자백한다**: Zamba2는 이벤트를 풀링하고(V6) NemotronH·Granite는
   층마다 새로 만든다(V5). ⇒ **모델 간 관측자 부하가 다르다.** 이것은 within-model
   설계에서는 무해하지만, **모델 간 수치 비교를 추가로 금지하는 사유**다.
3. **`open(_f).read()` per forward**(V8)도 부하의 일부다. 필요하면 상수 env로
   대체할 수 있으나(그러면 arm 전환이 재부팅), **그 변경 자체가 measured 조건을
   바꾸므로 arm 대칭으로만** 적용한다.
4. **금지**: on/off paired TOST로 무해 입증 — G5 형태. 하지 않는다.
5. ⚠️ **잔여 자백**: 위 1–3은 *호스트 발행 비용*을 재지, **동기화가 파이프라인을 끊어
   생기는 오버랩 손실**을 bound하지 않는다. 이것이 V10 기전 자체이며,
   §8-A1/A2가 **그 결과를 탐지**할 뿐 **제거하지는 못한다.**

**cudagraph에 대해서**: 운영점은 cudagraph-ON이고, **그 경로에는 층 단위 계측이
들어가지 않는다**(V12). 따라서 "그래프 캡처된 경로에 층 단위 계측이 들어가는가?"의
답은 **아니오, 구조적으로 들어갈 수 없다**이다. `--disable-cuda-graph`는 **명시적
플래그로 선언**하고, 그것이 이 캠페인을 off-operating-point로 만든다는 사실을
모든 보고 블록에 병기한다(방법론 게이트 "게이트는 모든 보고 블록에").

---

## 8. 결정량과 결정 규칙 **후보** (사전등록 초안 — 확정 아님)

**표기**: `t_τ(s)` = SM arm `s`에서 층 타입 `τ ∈ {mamba, attn, mlp}`의 **호출 1회당**
시간의 셀 중앙값. `ε_τ = ln(t_τ(44)/t_τ(92)) / ln(92/44)`.

### 8.A Admissibility — Δε를 **계산하기 전에** 통과해야 하는 것

| ID | 질문 | 결정량 | 규칙 후보 | 실패 시 |
|---|---|---|---|---|
| **A1** | 추정량이 오염됐는가 (**물리 불변량 음성대조**) | 같은 realized bs에서 `t_mamba(ctx=L₁)/t_mamba(ctx=L₂)` | mamba decode는 state 형상이 L에 등장하지 않으므로 **정확히 O(1) in ctx**다. 비 ≤ **1.05**면 admissible. ⚠️ 실현 bs를 셀 라벨의 `mode`가 아니라 **혼합 분포**로 기록(P5_GATES §6.3-3) | ★ 그 셀 **폐기**. 임계를 재조정하지 않는다(게이트 #8) |
| **A2** | 셀이 device-bound인가 | `fwd_ms(SM44)` vs `fwd_ms(SM92)` | 두 값이 **구별되지 않으면** host-paced ⇒ 폐기. 마진은 §11-P1이 **측정한** floor에서 정하고, 이 캠페인 데이터에서 고르지 않는다 | 동일 |
| **A3** | 단조성 (물리 대조) | `t_τ(s)`가 SM에 대해 비증가인가 | 어떤 타입에서든 비단조가 나오면 **그 셀에서 Δε 계산 중단**하고 관측을 그대로 보고 | V10의 재발 신호 |
| **A4** | 보완/음성대조 | `t_mlp`의 ctx-불변성 | mlp는 2-GEMM span이라 idle을 흡수하지 않는다(V11). A1은 깨지는데 A4가 깨끗하면 = **오염 확진** | 진단으로만 |
| **A5** | mamba-track 발화 | 체크포인트 마스크 발화 수(V19) | 측정 창에서 **0**이어야 함. 아니면 발화율을 아티팩트에 기록 | A1의 교란원 |
| **A6** | closure (Zamba2만) | `Σbuckets / 독립 분모 쌍` | 기존 `SGLANG_ZAMBA_CLOSURE_MIN`(0.95) 유지 | ⚠️ **GATE C는 실제로 발화한 결함에 눈멀었다**(P5_GATES §4) — 통과를 "측정 성립"으로 읽지 마라 |

### 8.B Primary — "다르게 반응하는가"

| ID | 결정량 | 규칙 후보 |
|---|---|---|
| **R1** | **Δε = ε_attn − ε_mamba** (모델·셀 내부, n 부팅에 대해 paired) | 셀별 CI가 0을 배제하고 Holm 보정 후에도 유지되면 **"이 셀에서 두 타입의 SM 응답이 다르다"**. CI가 0을 포함하면 ★ **UNDETERMINED** — **"같다"고 쓰지 않는다**(귀무 채택형 = G5 서명 오류) |
| **R2** | **Δε′ = ε_mlp − ε_mamba** (내부 기준선) | mlp는 순수 GEMM·L 불변·B 선형이므로 **같은 forward 안의 공통 기판 참조**다. `ε_mamba ≈ ε_mlp ≠ ε_attn`이면 "mamba는 dense-GEMM 층처럼 반응한다"가 지지된다 |
| **R3** | 절대 수준 `t_τ(s)` | 탄력도만 보고하지 말고 **레벨을 병기**한다(비만 보면 분모 붕괴가 비를 폭발시킨다 — P5_GATES §6.1의 교훈) |
| **D1** | 곡률 | 4 SM 점의 국소 탄력도 3개. **구간 평균으로 축약 금지**(정본 C2 caveat: 16→24와 44→92가 4× 다르다) |

**의도적으로 넣지 않은 것**
- **goodput·TTFT·ITL 우열 없음.** 결정량은 층 시간과 그 탄력도뿐이다.
- **등가 검정(TOST) 없음.** 이 저장소의 서명 오류다. R1의 실패는 "같다"가 아니라 UNDETERMINED다.
- **교차-arm 통계 없음** (§6.3).
- **정책 결정량 없음.** `PDMUX_LA_SM_MAP` / `layer_aware` 경로는 **건드리지 않는다**(§1.1).

**새 자유모수 회계**: 새로 도입되는 자유모수는 **A1의 1.05**와 **A2의 floor 마진** 둘뿐이다.
1.05는 물리(정확한 O(1))에서 오고 이벤트 오버헤드 여유로만 정당화된다;
floor 마진은 §11-P1이 **직접 측정**한다. SM 끝점 {44,92}는 기존 인용 규율에서 왔고,
(L,B)는 §6 산술에서 왔다 — **둘 다 조정 가능한 손잡이가 아니다.**

---

## 9. ★★ 이 실험으로도 **닫히지 않는 것** (필수 절)

> 이 프로젝트는 게이트가 여러 번 죽었다. 비싼 것을 사기 전에 payoff를 코드·원리로
> 확인하는 것이 방법론 게이트 #28이다. 아래는 그 확인 결과이며 **낙관 없이** 적는다.

1. **layer-type 정책은 되살아나지 않는다.** 死因은 (C1) lever 부재가 아니라
   (C2) 착취 비용이었다(§1.1). Δε가 얼마로 나오든 postmortem §2 표는 그대로다.
2. **운영점(cudagraph-ON)은 보이지 않는다.** V12는 코드 사실이다. 이 캠페인은
   **정의상 eager**이고, 따라서 고-SM 평탄화 후보 중 **(iv) cudagraph 직렬화는
   검정되지 않는다** — 애초에 존재하지 않는 조건에서 재기 때문이다.
3. **배포 동작점이 아니다.** B=64–96은 관측된 서빙 batch(9–12)의 5–10배다(§6.5).
4. **모델 간 종합 불가.** backend 비대칭(V21) + 파라미터·형상 동시 상이 +
   관측자 부하 비대칭(V5/V6). 결과는 **모델별로만** 서술된다. "hybrid는 …"는 금지.
5. **SM 축은 target이지 realized가 아니다.** `_get_gctx_decode_stream`이 요청한 값이
   물리적으로 전달됐는지는 이 캠페인이 보지 않는다 — 그건 **C-1(`%smid`) 트랙의 질문**이고
   현재 미해결이다. Stage 0 D108 오류(pin target ≠ realized)의 거울상 위험이 **그대로 남는다.**
   ⇒ 두 설계는 경쟁이 아니라 **보완**이다: C-1이 이 캠페인의 x축을 검증한다.
6. **층 시간 ≠ 커널 내부.** wave quantization·점유율은 층 시간으로 분리되지 않는다.
   §1.2-C의 "부분 분리"는 (ii) vs (i)/(iii)의 **상대적 지지도**까지이고, 어느 것도
   확정하지 않는다.
7. **`other`/`mlp` 버킷이 무엇을 흡수하는지는 여전히 부분적으로만 안다.** P5_GATES §2.3이
   보인 대로 `other`는 `mamba`와 **같은 순위로** 깨진다 — 두 버킷은 오염에 대해
   독립 증거가 아니다.
8. **NemotronH의 attn 층은 4개뿐**이다(52층 중). 층당 시간은 잴 수 있으나 **표본 수가
   구조적으로 작고**, 스텝 트래픽 기여는 3% 수준이다 ⇒ 그 모델의 attn 추정치는
   Zamba2(13호출)보다 잡음이 크다. 이것을 "NemotronH는 attn이 안 중요하다"로 읽으면 안 된다.
9. **`goodput`·SLO·fused 대비 우열·PD 분리 귀속에 대해 아무것도 말하지 않는다.**
10. **A1/A2가 통과해도 "오염이 없다"는 증명이 아니다.** 두 게이트는 **탐지기**이고,
    탐지 실패는 부재가 아니다(방법론 게이트 #21). 통과는 "이 셀에서 알려진 오염
    시그니처가 관측되지 않았다"로만 쓴다.

**⇒ 정직한 payoff 요약**: 이 실험은 **새 성능 주장을 0개 열고**, 진단 §9-1의
"per-layer-type 귀속 0건"을 **측정된 셀에 한해** 없애며, 기전 주장 하나를
[C]트래픽 회계에서 [M]시간 측정으로 옮긴다. 그 대가로 **배포 동작점도, 운영점도,
정책 함의도 사지 못한다.** 이 문장에 동의할 수 없다면 이 실험을 사지 마라.

---

## 10. long-ctx 트랙과의 연결 — **한 캠페인으로 살 수 있는가**

**답: 아니오. 하네스와 부팅은 공유할 수 있지만 주장은 공유할 수 없다.**

| 항목 | 이 설계(C-2) | long-ctx 트랙 |
|---|---|---|
| estimand | 층 타입별 시간과 그 SM 탄력도 | 프론티어 ITL(D) vs TTFT(108−D), goodput |
| cudagraph | **OFF 강제**(V12) | **ON**(운영점) |
| 러그 | 특성화 | 정책 |
| 상태 | 미측정 | Stage 0 철회로 **판정 이전**, ctx≤16k 전 구간 미측정 |

**구조적 비양립**: long-ctx 프론티어 질문은 cudagraph-ON에서만 의미가 있고,
per-layer 계측은 cudagraph-ON에서 **실행되지 않는다**. ⇒ **같은 arm에서 둘 다 살 수 없다.**

**그럼에도 공유 가능한 것 2가지**:
1. **프롬프트/부하 생성 machinery와 L 축**(r0c 하네스의 `REPS=CTX/8` 파일 기반 프롬프트).
2. **ctx 상한 지식**: long-ctx를 계측된 hybrid에서 사려면 **모델 선택이 강제**된다 —
   Zamba2-7B는 **4096**에서 끝나고 NemotronH-8B는 **8192**에서 끝난다(V16).
   ctx 16k+에서 층-타입 계측이 가능한 모델은 **Granite-4.0-h-micro(131072)** 뿐인데
   그 계측은 리워크 이전 형태이고(§2.1) 3B급이다. Falcon-H1-7B(262144)는 **계측이 아예 없다**.
   ⇒ **"7-8B hybrid에서 long-ctx per-layer-type"은 현재 모델 자산으로 불가능하다.**
   이 사실 자체가 long-ctx 트랙 계획에 입력돼야 한다.

---

## 11. 비용과 **가장 싼 결정적 프로브**

### 11.1 비용 (전부 §3.B-E9 = 추정)

| 항목 | 구현 공수 | GPU-hr | 배관 리스크 |
|---|---|---|---|
| **L0 CPU 감사 + manifest 격리 계획** | 반나절 | **0** | 낮음 |
| **P1 단일 셀 admissibility 프로브** | 반나절(하네스 파라미터화) | **≈0.4–0.6** | 낮음 — 기존 r0c 하네스 재사용 |
| **P4 nsys 실현성**(선택, 경로 2용) | 반나절 | **≈0.2** | 중 — 권한·그래프 노드 트레이스 미검증 |
| **L2 Zamba2 단독 전체 격자**(2 셀 × 4 SM × n=3) | 1일(하네스+분석기 재정의) | **≈2–3** | 중 — A1 재정의·bs 혼합 기록 |
| **NemotronH 리워크 이식**(계측 defect 2 + closure + shape guard + 풀링) | **2–3일 + correctness gate 재통과** | +**≈2–3**(격자 동일) | ★ **높음** — manifest 밖 파일을 sync에 넣으면 **기존 캠페인 재현 경로가 바뀐다**(V3) |
| **합계 (권고: L0+P1+L2 Zamba2 단독)** | ≈2일 | **≈2.5–3.6** | — |
| (참고) NemotronH까지 포함 | ≈5일 | **≈5–7** | 높음 |

비교 기준: Gate 2-S 본 캠페인 6.35 GPU-hr, r0c job 873783 ≈2 GPU-hr(4 태스크),
G1-c 0.10 GPU-hr. ⇒ **Zamba2 단독은 Gate 2-S의 절반 이하**, **두 모델은 그와 동급**이다.

⚠️ **방법론 게이트 #26**: 분석기(A1 재정의·bs 혼합 기록)가 신규이고 예상 비용이
≥1 GPU-hr이므로 **본 제출 전 n=1 배관 스모크 필수**. P1이 그 역할을 겸한다.

### 11.2 선행 프로브 (순서대로)

| ID | 확인 대상 | 비용 | 프로브 | 실패 시 |
|---|---|---|---|---|
| **P0** | 계측 존재·드리프트·sync 상태 | **0 (완료)** | V1–V3, V6. 부록 A | — |
| **P0b** | NemotronH·Granite 리워크 미이식 여부 | **0 (CPU)** | 2026-08-04 체크리스트 대조 (§5-L0-1) | 공수 항목에 반영 |
| **P0c** | 전체 CPU 회귀 완주 | **0 (컴퓨트 노드 CPU)** | `unittest discover` — V24는 채점 아님 | — |
| ★ **P1** | ★ **A1(GATE N 재정의) + A2(host floor) + E2(계측 자기비용) + E8(mamba-track)** — 4개 동시 | **≈0.4–0.6 GPU-hr** | §5-L1의 단일 부팅 | ★ **eager 사다리 사망.** 경로 2(nsys)로 이동하거나 캠페인 취소 |
| **P2** | Ha8 (L=1024, B≈88) 메모리 실현 | **P1에 포함** | 부팅 로그 `KV Cache is allocated. #tokens:` 직독 | B 하향 또는 mem-frac 상향(E3) |
| **P3** | SM 44/54/74/92 green ctx 생성 성공 | **P1에 포함** | `create_greenctx_stream_by_value` 예외 없음 | 격자 축소 |
| **P4** | E5/E6 — nsys 그래프 노드 트레이스 + 권한 | **≈0.2 GPU-hr** | `nsys profile --cuda-graph-trace=node` 로 s8 구성 1셀 | 경로 2 사망 ⇒ 운영점 층-타입은 **어떤 경로로도 불가**로 등재 |
| **P5** | manifest 격리 | **0** | NemotronH를 sync에 넣기 전 G1-a 대기 상태 확인 | 드리프트 시 캠페인 비교가능성 훼손 |

★ **가장 싼 결정적 프로브 = P1 (단일 부팅, GPU 30분 내외).**
`%smid` 설계에서 P1+P2가 한 것과 같은 역할이다 — **이 하나가 §5 전체의 생사를 가른다.**
P0b/P0c는 무료이므로 P1보다 먼저 하되, P1을 지연시키지 않는다.

---

## 12. 실패 모드 (사전 열거)

| # | 실패 | 징후 | 대응 (사전 지정) |
|---|---|---|---|
| F1 | ★ **V10 재발** — 7-8B에서도 host-bound | `fwd_ms`가 SM44/92에서 동일; A1 비 >1.05 | **즉시 중단.** 그 셀 폐기. B를 더 올릴 여지가 없으면 §5-차선 1 |
| F2 | SM 비단조 | `t_τ`가 SM 증가에 대해 증가 | A3 발화 ⇒ **Δε 계산 중단**, 관측 그대로 보고. 아키텍처 발견으로 서술 금지 |
| F3 | realized bs 혼합 | 셀 내 `nseq` 분포가 다봉 | ★ P5_GATES §2.1의 정확한 함정. CELL 라벨에 `statistics.mode` 쓰지 말고 **혼합 분포**를 기록 |
| F4 | mamba-track 발화 | A5 계수 >0 | 발화율 기록 + `--mamba-track-interval` 상향 후 재측정 |
| F5 | OOM (Ha8 절벽) | 부팅 실패 또는 KV 토큰 부족 | B 하향(88→64). **mem-frac 상향은 변수 추가**이므로 별도 셀로 기록 |
| F6 | manifest 드리프트 | `sync_engine_tree.sh` manifest 변경 | P5. NemotronH 이식은 **별도 캠페인 manifest로 격리** |
| F7 | 계측 자기비용이 floor를 지배 | `TIMING=1/0` 차가 셀 시간의 큰 비율 | 그 숫자를 **아티팩트에 기록**하고 A2 마진을 그만큼 **좁힌다**. "무해"로 서술 금지 |
| F8 | 아티팩트 부재를 결과로 라벨링 | emit 라인 0 → "차이 없음" | ★ 방법론 게이트 #21(7회 재발). **아티팩트 부재 = UNDETERMINED**. 분석 코드에 `n_total == 0` 분기를 **먼저** 넣고 단위 테스트로 고정 |
| F9 | closure PASS를 "측정 성립"으로 오독 | GATE C 통과 보고 | ★ P5_GATES §4의 존재 증명(20/20 PASS ∧ 3–5/5 GATE N VIOLATION). **두 게이트는 비중복** |
| F10 | 옛 파서 사용 | `ZBLT mode=` grep이 0건 | 태그는 `ZBLT2`다(V6). 기존 하네스 6종이 빈 결과를 낸다(dev_tree_edits 항목 17) |

---

## 13. 하지 말 것 (이 트랙에 고정)

- **공유 dev 트리 무단 수정 금지.** NemotronH 리워크는 승인·미러(`src/patches/`)·
  `env/dev_tree_edits.md` 기록·manifest 격리가 전부 끝난 뒤에만.
- **layer-aware 정책 경로를 건드리지 마라** — `PDMUX_LA_SM_MAP`·`layer_aware`·
  `PDMUX_LA_COORD*`는 이 캠페인에서 **미사용**. 이 설계는 `_fixed`(uniform pin) 경로만 쓴다.
- **성능 주장 금지.** 산출물은 층 시간과 탄력도이지 goodput·TTFT·우열이 아니다.
- **교차-arm 비교 금지**(§6.3, §9-4).
- **배포 동작점으로의 이식 금지**(§6.5).
- **`full` arm을 "108 SM"으로 인용 금지**(§6.3).
- **s8 결론(2.36–2.91×, ε 44→92)을 이 격자로 이식 금지 / 이 격자 결과를 s8로 역이식 금지.**
- **prefill 축으로 이식 금지** (AI≈1035, compute-bound — 진단 §6.5).
- **achieved_BW를 arm 순위로 읽기 금지.**
- **마이크로 러그(e3/`layer_runner.py`) 수치를 엔진 결론의 근거로 인용 금지**(게이트 #1),
  그리고 감사 전까지는 **예측 앵커로도 금지**(V22의 미해명 이상).
- **A1/A2 임계를 이 캠페인 데이터로 재조정 금지**(게이트 #8).
- **"두 타입이 같다"고 쓰지 마라** — CI가 0을 포함하면 UNDETERMINED다(G5 서명 오류).
- **본 문서를 사전등록으로 인용 금지.** 사전등록은 claims-auditor 설계 감사를 거친
  별도 문서여야 한다.

---

## 14. 다음 액션 (제안 — 승인 필요)

1. **P0b + P0c**(무료): NemotronH·Granite 계측 리워크 감사 + 컴퓨트 노드 CPU 회귀 완주.
2. ★ **P1 제출**(≈0.5 GPU-hr, experiment-runner 소관): §5-L1 단일 부팅.
   **A1/A2 결과가 나오기 전에는 §6 격자를 사지 않는다.**
3. A1/A2 통과 시에만: §6 격자를 **Zamba2 단독**으로 사전등록 초안 작성 →
   claims-auditor **설계 감사** 회부 → 제출.
4. NemotronH 포함은 **3이 끝나고 별도 승인** 후에만(manifest 격리 필수).
5. (선택) **P4** — 운영점 층-타입이 필요해지면. 지금은 1–3을 지연시키지 않는다.

---

### 부록 A — 이 문서를 만들며 실제로 실행한 명령 (전부 CPU / 읽기)

```bash
# 계측 존재·드리프트 (V1, V2)
grep -rn "SGLANG_NH_LAYER_TIMING\|SGLANG_GRANITE_TIMING\|SGLANG_ZAMBA_TIMING" \
  workspace/engine-port/src/models/
for f in nemotron_h granitemoehybrid zamba2; do
  diff -q workspace/engine-port/src/models/$f.py \
    /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/models/$f.py; done

# sync/manifest 포함 여부 (V3)
cat workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh

# cudagraph 구조 사실 (V12, V13)
grep -n "capture_forward_mode\|graphs\[graph_key\].replay" \
  /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/model_executor/cuda_graph_runner.py
grep -m1 "disable_cuda_graph=" \
  workspace/engine-port/results/s8_scaleup/s8_deconf_Hs8_C1024_d16_865533_srv.log
sed -n '114p' workspace/engine-port/results/r0c/decode_knee_vs_ctx_v2.sbatch

# stock nvtx 훅 (V14) / nsys (V15)
grep -rn "enable_layerwise_nvtx_marker" \
  /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/
/apps/cuda/13.0.2/bin/nsys --version
/apps/cuda/13.0.2/bin/nsys profile --help | grep -A3 cuda-graph-trace

# ctx 상한 (V16) / 메모리 앵커 (V17)
python3 -c "import glob,json,os;[print(d.split('models--')[1].split('/')[0],
  json.load(open(os.path.realpath(d))).get('max_position_embeddings'))
  for d in sorted(glob.glob('hf_cache/hub/models--*/snapshots/*/config.json'))]"
grep -E "Load weight end|Cache is allocated|Memory pool end" \
  workspace/engine-port/results/s8_scaleup/s8_deconf_{Hs8,Ha8}_C1024_d16_865533_srv.log

# mamba state 체크포인트 (V19)
grep -rn "mamba_track_interval" \
  /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/managers/schedule_batch.py

# 층-타입 트래픽 산술 (V20) / (L,B) envelope (§6.2)
#   세션 스크래치패드: .../scratchpad/ltsm/{grid.py,envelope.py}
#   (스크래치이므로 영속 아님 — §6.1 형상값과 V17 앵커만으로 재작성 가능)

# 마이크로 러그 (V22)
awk -F, 'NR>2 {print $1,$3,$4,$5,$8,$11}' \
  workspace/characterization/results_v2/e3/decode_floor_zamba2_7b_a100_sxm4_80gb.csv
```

**GPU 커널 실행 0건, dev-tree 쓰기 0건, git 조작 0건.**
