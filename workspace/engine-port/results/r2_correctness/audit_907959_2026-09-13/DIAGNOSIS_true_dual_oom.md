# 진단 보고서 — job 907959의 true-dual OOM (engine-porter, GPU 0, 코드 무수정)

**작성** 2026-09-13 · **대상** job 907959(새 (Nano-9B-v2, flashinfer) 쌍 correctness 게이트, `VERDICT FAIL`)
**규율** 코드·테스트 무수정 · git 무커밋 · GPU 사용 0 · 결과 감사 판정서(`VERDICT.md`)와 독립 수행

---

## 0. 한 줄 결론 (가설 — GPU 측정으로 아직 미검증)

**주 가설 H1**: true-dual의 role worker thread는 `event_loop_pdmux`의 `@torch.inference_mode()` **밖**에서 돈다(inference_mode는 **thread-local**). 그래서 TD arm의 모델 forward는 **autograd가 켜진 채** 실행된다. NemotronH는 `mamba_num_groups=8` 때문에 gated RMSNorm이 **네이티브(순수 torch) 경로**를 타고, 그 마지막 줄 `self.weight * x.to(input_dtype)`에서 **`requires_grad=True`인 bare `nn.Parameter`가 평범한 torch 연산에 들어간다** → `MulBackward0`가 활성화 텐서를 **저장(retain)** 한다. split-prefill은 `forward_batch.hidden_states`를 56층 내내 살려 두므로 이 retention이 **층마다 누적**된다. legacy는 inference_mode 안이라 같은 텐서를 즉시 해제한다.

**이는 산정이지 측정이 아니다.** 아래에서 관측 / 산정 / 코드사실을 분리한다.

---

## 1. 크래시 텐서 동정

### 관측
- OOM 지점 `mixer2_rms_norm_gated.py:97` `return self.weight * x.to(input_dtype)`, 요청 **198.00 MiB**, free 55.19 MiB, **allocated 77.25 GiB**, reserved-but-unallocated 1014.25 MiB, private pool(cudagraph) 156 MiB.
- 부팅 배너(5 boot 동일): weight 16.58 GB · KV 20.76+20.76 GiB · mamba ssm 6.46 + conv 0.09 GiB · cudagraph 1.27 GB → **`available_gpu_mem = 11.59 GB`**.
- TD1 마지막 로그: `Prefill batch #new-seq: 8, #new-token: 6245` → `Decode batch #running-req: 14, #full token: 21285` → 예외. **14-seq prefill 로그는 없다**(`report_prefill_stats`는 `scheduler_output_processor_mixin.py:328`, 즉 prefill **완료 시** 호출 — 죽은 batch는 로그를 못 남긴다).
- L1 로그 2592행: 같은 지점에서 **`#new-seq: 14, #new-token: 10125`** prefill을 **성공**. TD1/TD2 마지막 telemetry: `prefill_queue_depth=14`, `owned_prefill_waiting=14`, `decode_running_batch_size=14`.
- warm-up boot(legacy)는 같은 설정에서 **`#new-token: 16384` prefill을 `#running-req: 48`, `#full token: 36372`에서 완주**.

### 산정
`preallocated_ssm_out` = bf16 `[T, num_heads×head_dim] = [T, 128×80=10240]`(`mamba.py:474-481`) ⇒ 97행 alloc = `T × 20480` B. 캐싱 할당자는 큰 세그먼트를 **2 MiB 올림**:

```
T = 10125 → 10125 × 10240 × 2 = 207,360,000 B = 197.75 MiB → 198.00 MiB  ✅ 정확히 일치
```
(196, 198] MiB 역산 구간은 `T ∈ [10036, 10137]` — 워크로드에서 자연히 나오는 값은 **10125뿐**이고 (a) L1이 같은 지점에서 돌린 batch, (b) TD telemetry의 14-seq 대기와 일치. **독립 두 채널이 같은 답.**

⇒ **TD가 죽은 batch = legacy가 성공시킨 14-seq / 10,125-token split-prefill batch.** 입력은 arm 간 동일.

---

## 2. 왜 true-dual에서만 터지는가

### H1의 코드 사실

| 사실 | 위치 |
|---|---|
| `event_loop_pdmux`가 `@torch.inference_mode()` — **thread-local** | `multiplexing_mixin.py:992` |
| legacy는 `run_batch(split_prefill_batch)`를 **메인 스레드**에서 호출 | `multiplexing_mixin.py:1216` else-branch |
| TD는 `_run_prefill`/decode lambda를 `RoleWorkerThread._loop`에 넘김 | `dual_worker.py:336`, `multiplexing_mixin.py:1205-1216` |
| 활성화 훅은 stream/role만 세팅, **grad guard 없음** | `multiplexing_mixin.py:501-509` `_activate_role_context` |
| `run_batch`·`tp_worker`·`model_runner`·`forward_split_prefill` 경로에 `no_grad`/`inference_mode` **0건**(grep 전수) | — |
| `NemotronHForCausalLM.forward`엔 `@torch.no_grad()`가 있으나 **`forward_split_prefill`엔 없다** | `nemotron_h.py:840` vs `:858` |

### CPU 측정 (실제 `RoleWorkerThread` 사용, GPU 0)
```
main_grad_enabled   = False    main_inference_mode   = True
worker_grad_enabled = True     worker_inference_mode = False
param_requires_grad = True     graph_built_on_worker_tensor = True
```
추가: `w * y`(w=requires_grad Parameter) → `MulBackward0._saved_other is y == True`, `del y` 후에도 텐서 **생존**(inference_mode에서는 즉시 해제). sgl_kernel의 in-place custom op(Autograd 커널 없음)를 흉내 내 호출 → 예외 없이 `requires_grad`·`grad_fn` **유지** ⇒ 체인이 층 경계 fused RMSNorm을 **통과**한다. `torch.ops.sgl_kernel.fused_add_rmsnorm`/`rmsnorm` 디스패치 덤프로 **CUDA 커널만 등록**됨을 확인.

### 왜 NemotronH만 노출되는가 (핵심)
SGLang의 거의 모든 가중치는 autograd에서 차단돼 있다 — linear/embedding은 `Parameter(..., requires_grad=False)`(`quantization/unquant.py:118-124`, `parameter.py:86`), `RMSNorm.forward_cuda`는 **`self.weight.data`**(`layernorm.py:206,207`), `Mixer2RMSNormGated.forward_cuda`의 **커널 분기**도 `weight=self.weight.data`(`mixer2_rms_norm_gated.py:114`).

**유일한 누출구가 `mixer2_rms_norm_gated.py:97`** — 여기만 `.data` 없이 bare Parameter를 쓴다. 그 줄은 `n_groups != 1`일 때만 실행된다(§3).

| 모델 | `mamba n_groups` | 97행 도달? |
|---|---|---|
| Zamba2-2.7B (907032/907100/907456/X1) | **1** | ❌ 커널 분기 → 면역 |
| Falcon-H1-3B/7B · Granite-4.0-h-micro | 1 | ❌ |
| **Nemotron-Nano-9B-v2 / Nemotron-H-8B** | **8** | ✅ **노출** |
| Zamba2-7B | 2 | ✅ (미사용) |

⇒ **"왜 지금, 왜 이 모델에서 처음"이 코드로 설명된다.** 이전 TD job 3건의 통과는 **구조적 면역**이지 TD가 안전했다는 증거가 아니다.

### 산정 (측정 아님)
층당 retain 바이트: mamba 27층 `MulBackward0` 저장 bf16 `[T,10240]` = `T×20480` B · MLP 25층 `ReLU2`(`activation.py:162-164`) `ReluBackward0` 저장 bf16 `[T,15680]` = `T×31360` B · attention 4층 ≈ 0.

```
R(T) = T × (27×20480 + 25×31360) = T × 1,336,960 B = T × 1.2750 MiB

T= 2310 →  2.88 GiB  (O층)        ✅ 통과(관측)
T= 3241 →  4.04 GiB               ✅ 통과(관측)
T= 6245 →  7.78 GiB               ✅ 통과(관측, 빠듯)
T=10125 → 12.61 GiB > 11.59 GiB   ❌ 고갈  ← 관측된 크래시
T=16384 → 20.40 GiB               (legacy는 완주 — 관측)
```
- TD 예측 prefill 토큰 천장 ≈ **6.5k–7.7k**(working set 2.0–3.5 GiB 가정). 관측 6245 통과 / 10125 실패 — **구간 안**.
- 누적이 여유를 소진하는 층 index = **38~43, 전부 `'M'`** ⇒ `_forward_mamba`에서 죽는 traceback과 일치.
- 관측 live delta `77.25 − (79.25 − 11.59 − ~0.5) ≈ 10.09 GiB` — 산정 누적(8.7–9.7 GiB) + working set과 정합.

**주의**: R(T)는 "그래프가 prefill 전체에 걸쳐 유지된다"는 가정 위의 **산정**이다. 층별 실제 retention은 GPU 측정으로만 확정된다.

### 반증된/약한 대안

| 가설 | 반증 증거 |
|---|---|
| **H2** prefill·decode worker 중간 텐서 동시 생존 | decode는 전 스텝 cudagraph replay(`T/F=2347/0`), private pool **156 MiB**뿐. legacy도 GPU 상에서는 두 스트림에 겹쳐 발행. 14-seq decode transient는 GiB 단위가 아님 |
| **H3** role별 스트림/그래프 워크스페이스 중복 | `_activate_role_context`가 `stream_groups[idx]`의 **같은 스트림 객체**를 쓴다(새 스트림 생성 없음). 5 boot 전부 `available_gpu_mem=11.59 GB`로 **정적 풋프린트 동일** |
| **H4** 단편화 | reserved-but-unallocated 1014 MiB뿐, **77.25 GiB가 live** |
| **H5** 원래 빠듯했고 TD가 살짝 넘음 | legacy warm-up이 **16384-token prefill @ bs=48 @ 36k full token** 완주, TD는 10125 @ bs=14 @ 21k 사망. 선형 가정 시 T=10125에서 legacy peak ≤ 7.16 GiB ⇒ TD/legacy ≳ **1.6×**(가정 명시한 하한) |
| **H6** `unsafe_decisions`/컨트롤러 | §4 — 구조적으로 무해 |

---

## 3. `forward_native`를 타는 이유

```python
# mixer2_rms_norm_gated.py:109-110 (forward_cuda 안)
if ((self.n_groups % self.tp_size) != 0) or self.n_groups != 1:
    return self.forward_native(x, gate)
```
`group_size = 10240//8 = 1280`, `n_groups = 10240//1280 = 8`, `tp_size=1` ⇒ `(8%1!=0)=False or (8!=1)=True` → **항상 native**.

- **백엔드(flashinfer/triton)와 무관** — 순전히 `config.mamba_num_groups=8`의 귀결.
- **arm 무관**(legacy도 native를 탄다) ⇒ native 자체가 TD-only 원인은 아니다.
- 다만 **필수 공범**: 커널 분기는 `self.weight.data`, native는 `self.weight` ⇒ **native만 autograd에 Parameter를 노출**한다.
- 부수: native는 `[T,10240]` fp32 임시텐서를 여러 개 만든다(각 395.5 MiB @ T=10125). legacy에선 즉시 해제.

---

## 4. `unsafe_decisions=16` / `request_errors=31`

**`unsafe_decisions` = arm 정의상 산물, 인과 없음.** `_r2_decide_idx`가 `target == current` **이면서** `decision.safe == False`일 때 방출(`multiplexing_mixin.py:463-471`). `safe`의 소스는 `true_dual_worker_runtime.arbiter.safe_to_switch()`이고 **legacy는 runtime이 `None`이라 `safe=True`가 상수** ⇒ L arm 카운터는 **구조적으로 항상 0**. `FixedPolicy`(`controller.py:88-102`)는 D44 고정이라 어느 쪽이든 `target=44=current` ⇒ **전이 자체가 없다**. 관측 16건 전부 `current=44, requested=44`, 로그 기반 `unsafe_transition=0`. ⇒ **prefill∧decode in-flight의 표지**일 뿐 크래시와 인과 없음. (단 "TD만 16, L은 0"이 arm 정의상 비대칭이라는 점은 기록 필요.)

**`request_errors=31` = 결과.** 서버 SIGQUIT 후 진행/대기 31건 실패, 1건은 크래시 직전 완료. 시간 순서(OOM 22:08:56 → SIGQUIT → 클라이언트 에러)가 이를 지지. C층 cross-arm 31은 같은 사건의 재표현.

---

## 5. 결정성이 함의하는 것

두 TD boot이 **완전히 같은 서버 상태**(`#running-req: 14`, `#full token: 21285`, mamba usage 0.29→0.58, 직전 prefill 시퀀스 `55→368→1468→3241→6245` 동일)에서 **같은 198.00 MiB 요청**에 **같은 스택**으로 죽었다.

- ⇒ **경합(race)이 아니다.** 907032의 split-prefill ownership race(수정 `874873b`)와 다른 종류(race라면 죽는 지점·batch·할당 크기가 흔들린다).
- ⇒ **결정적 용량 초과**. 원인은 "배치 구성의 결정적 함수"여야 한다. H1(토큰 수 × 층 수에 선형인 retention)은 만족, H2(worker 타이밍)는 만족하기 어렵다 — **H2에 대한 두 번째 독립 반증**.

---

## 6. piecewise 크래시(2026-09-02) 및 잠복 위험과의 관계 — **셋 다 다른 사건**

| 축 | 900731 piecewise 크래시 | arch 목록 잠복 위험 | 907959 OOM |
|---|---|---|---|
| 실패 종류 | `'typing.Union' object has no attribute '__module__'`(파이썬 capture 오류) | (동일 계열, 미발현) | `torch.OutOfMemoryError` |
| 시점 | **부팅 중 그래프 캡처** | 부팅 중 캡처 | **정상 서빙 중, 스텝 내부** |
| arm 비대칭 | 없음 | 없음 | **TD만** |

관측 확인: **4 boot + warm-up 전부** 로그에 `Disable piecewise CUDA graph because the capture size is not set`(L1:65행, TD1:66행 …) ⇒ `cps=-1`이 **실제로** piecewise를 껐다(추론이 아니라 로그 실증). ★단 `server_cmd()`에 `--disable-piecewise-cuda-graph`가 **없다**(`r2_correctness.sbatch:272-280`) — `DECISION_PIECEWISE_OFF_2026-09-02.md` §2("모든 실험의 모든 arm에서 명시")와 **하네스 사이에 드리프트**가 있다(별건 기록). 이 job에서는 로그로 상태가 확인됐고 arm 대칭이라 판정에 교락되지 않는다.

---

## 7. 옵션 표 (구현하지 않음)

★ = **조건 튜플을 바꿔 재측정 → 새 사전등록 필요**

| # | 무엇을 바꾸나 | 오염되는 측정량 / 스코프 | 비용 | H1 하 예상 |
|---|---|---|---|---|
| **A** | **worker task를 `torch.inference_mode()`(또는 `no_grad`)로 감싼다** — `_activate_role_context` 한 곳 | 스코프 튜플 축은 **안 바뀜**(model/backend/ctx/mem/cap/split/cudagraph 불변). 그러나 **엔진 소스 해시가 바뀐다** → `engine_source_hash` 축 이동, 907100/907456/X1의 TD 데이터와 "같은 엔진" 아님. ★부분적. 성격은 **오염 제거**(TD를 legacy와 같은 grad 상태로 정렬) | 코드 수 줄 + 패치 미러 + manifest + 전체 CPU 테스트 + GPU 재측정 1회 | 10125 batch 완주, C errors → legacy 수준 |
| **B** | `mixer2_rms_norm_gated.py:97`을 `self.weight.data * …`로(커널 분기 관례와 일치) | 상동. **국소적**이지만 "worker가 grad 안에서 돈다"는 근본 상태는 남음 | A보다 작음 | 상동, 잔여 autograd 부기 오버헤드 존속 |
| **C** | `--mem-fraction-static 0.82 → 0.80/0.75` | ★**튜플 축**. KV/mamba 풀·`max_total_num_tokens`가 바뀌어 용량·배치 형성 자체가 이동 | GPU 1회 | 여유 +2~5 GiB → 천장 +2~4k 토큰. **근본 미해결** |
| **D** | `--max-running-requests 48 ↓` | ★튜플 축 | GPU 1회 | 큰 도움 없음 — 크래시는 **한 prefill batch 토큰 수**가 좌우(bs=14였다) |
| **E** | C층 동시성 32 ↓ | ★튜플 축(클라이언트 무수정이 등록 축). B2는 C층에도 적용되므로 판정 의미가 바뀜 | 작음 | 10125 batch 미형성 ⇒ **증상 은폐** |
| **F** | `--chunked-prefill-size -1 → 2048/4096` | ★튜플 축 + **piecewise capture 경로 재개방**(NemotronH가 disabled 목록에 없음 → 900731 계열 재발, `--disable-piecewise-cuda-graph` 동반 필수) | GPU + DECISION 재적용 | 증상 소멸하나 비교 대상이 달라짐 |
| **G** | `pdmux_r2.yml`의 `split_forward_token_budget` ↓ | ★튜플 축 | 작음 | **효과 없음(예측)** — 그래프는 split 창을 넘어 `forward_batch.hidden_states`에 매달림 |
| **H** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 사실상 튜플 이동 | 0 | **효과 없음(예측)** — 미할당 예약 1014 MiB뿐, 문제는 live 77.25 GiB |
| **I** | TD 기본 OFF 유지 + 현 상태 등재만 | 없음 | 0 | correctness gate는 계속 FAIL |

**비용순 권고**: §8의 1회 측정으로 H1을 가른 뒤 **A**(또는 A+B) → correctness gate 재실행. C/D/E/F/G는 전부 "조건을 바꿔 증상을 피하는" 쪽이며 새 사전등록을 요구한다.

---

## 8. 가설을 가르는 1회 측정

### M1 (권장, **코드 무수정**, H1 vs H2 분리) — 약 **0.10 GPU-h**
TD boot **1개**만, **동시성 1**로 **단일 10,125-token(≈14 seq 상당) prefill**(기존 I2/I3 프로브 레시피 재사용, C층 없음).
- **H1 예측**: 동시 decode가 없어도 **똑같이 198 MiB에서 OOM**(retention은 prefill 자체의 함수).
- **H2 예측**: **완주**.
- 부수: 같은 boot을 legacy로 한 번 더 → 대조.

### M2 (가장 직접적, telemetry 1필드, **양 arm 대칭**) — 약 **0.35 GPU-h**
`runtime_snapshot`에 `torch.cuda.memory_allocated()` / `max_memory_allocated()`를 **두 arm 모두**에 추가하고 907959를 그대로 재실행.
- **H1 예측**: TD는 한 prefill 안에서 **층 진행에 비례해 단조 증가**(끝나면 급락), legacy는 평탄. 6245-batch에서 TD 피크 ≈ **7.8 GiB**.
- **H3/H4 예측**: 계단형·prefill 무관 증가 또는 평탄.
- ⚠ 필드 추가 = 하네스 sha 변경 ⇒ 이 job과 "같은 하네스" 아님.

### M3 (A안 검증, A 구현 시에만) — 약 **0.35 GPU-h**
907959 **무수정 재제출**(엔진만 A 적용). 성공 기준 사전 고정: `srv_TD*.log`에 **`#new-seq: 14, #new-token: 10125`** 출현 + `request_errors=0` + S/O 동치 유지 + cudagraph ON 유지.

> 비용 기준: job 907959 = 21분 ≈ 0.35 GPU-h(5 boot). 단일 boot 변형 ~6–8분.

---

## 9. 부수 관찰 (판정 아님, 이관 항목)

1. ★**선행 TD 데이터의 스코프 단서**: 907032/907100/907456/X1은 Zamba2-2.7B(`n_groups=1`)라 이 retention 기전에 **구조적으로 면역**이다. 그러나 **"worker thread가 grad 켜진 채 돈다"는 성질 자체는 그 job들에도 있었다**(같은 코드). 토큰 값은 안 바뀌지만 autograd 부기(version counter·디스패치)는 **arm 비대칭 오버헤드**로 남는다 — 크기 **미측정**. claims-auditor 이관.
2. **하네스 드리프트**: `server_cmd()`에 `--disable-piecewise-cuda-graph` 부재(§6).
3. **판정 규칙 상의 긴장**: C층은 "진단 전용(mismatch 판정 불참)"인데 `B2 NO_CRASH`는 boot 전체(C층 포함)에 걸린다 ⇒ S·O 동치 게이트를 전부 통과했는데 C층 크래시로 FAIL. 규칙은 사전등록돼 있으므로 **재채점 금지**이나 다음 사전등록에서 명시적으로 다뤄야 한다.
4. **모델 census 교차검증**: `hybrid_override_pattern`에서 mamba 27 / MLP 25 / attention 4(합 56) — 로그의 `ssm_state 6.46GB`(27층×49슬롯×5 MiB)·`conv 0.09GB`·`K 20.76 GiB`(4층)와 **세 경로로 일치**, prereg §1 census와 부합.

---

## 10. 규율 준수 확인

- 코드·테스트·설정 **무수정**, git 무커밋, **GPU 사용 0**. `results/r2_eval/lambda0_prereg/**` 미접근.
- **"true-dual이 메모리를 X배 더 쓴다"를 단정하지 않았다.** 산정치(R(T), 1.6× 하한)는 가정을 명시했고 관측치(198 MiB, 77.25 GiB, 11.59 GiB, 10125/6245/16384)와 구분했다.
- 단언하는 것은 하나: **이 job은 사전등록된 verdict rule v2 하에서 FAIL이며(`TD1:B2_NO_CRASH`, `TD2:B2_NO_CRASH`), 새 (Nano-9B-v2, flashinfer) 쌍의 GPU correctness gate는 열려 있지 않다.** true-dual 기본값 ON 금지 유지.

## 주요 파일

- 아티팩트: `.../results/r2_correctness/job_907959/`(`srv_TD1.log:2495-2594` 스택, `srv_L1.log:2592` 10125 batch, `verdict.txt`, `tel_TD*.jsonl`)
- 크래시 지점: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py`(`:39` weight 생성, `:97` 크래시, `:109-110` native 분기, `:114` 커널 분기의 `.data`)
- 역할 스레드: `.../srt/multiplex/dual_worker.py:320-340` · `.../srt/multiplex/multiplexing_mixin.py:501-509`(activate) · `:992`(inference_mode) · `:1205-1216`(submit)
- 모델: `.../srt/models/nemotron_h.py:402-440`, `:858-890` · `.../srt/layers/attention/mamba/mamba.py:474-481,690` · `.../srt/layers/activation.py:156-164`
- 문서: `.../results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` §1·§4 · `.../results/cp_baseline/DECISION_PIECEWISE_OFF_2026-09-02.md`
- 하네스: `.../results/r2_correctness/r2_correctness.sbatch:272-280`
