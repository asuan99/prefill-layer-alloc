# 사전등록 — 907959 재실행 (role-worker grad guard 수리 + 대칭 메모리 계측) — **rev2**

**작성** 2026-09-13 (rev1) · **rev2** 2026-09-14 · **작성자** engine-porter ·
**GPU 신규 지출 0** (이 문서까지는 CPU만)
**대상 job** 없음(미제출) · **직전 회차** job 907959 = `VERDICT FAIL`
**선행 문서(전부 sha 고정, 아래 §0-c)** 사전등록 rev3 · 규칙층 판정서 2건 · 결과 감사 판정서 · 엔진 진단서
**rev1 규칙층 판정서** `rerun_prereg/VERDICT_rerun_rules_2026-09-14.md`
sha256 `71f3cfb7d8db6ce11ea3b24c5b4b3e894af0be1add0f693c40a7bc92f0226e65` — 등급 **`NO-GO`**(死因 `N2` ×2)

> ★**이 회차는 성능 실험이 아니다.** 처치는 **정확성/대칭성 수리** 하나이며, 어떤 수치도
> "true-dual이 빨라졌다 / 메모리를 덜 쓴다"로 인용될 수 없다(§11 인용 금지).

---

## 0-a. rev1 → rev2 변경표 (재감사용 색인)

rev1은 규칙층 감사에서 **`NO-GO`(死因 `N2` ×2)** 를 받았다. 두 死因 모두 **문서 텍스트**
문제이며, 감사는 **코드·하네스·테스트·manifest·라인 인용·예산·승계 목록은 고칠 것이
없다**고 독립 재검증 후 공시했다(판정서 §1 전 항목 일치, §4 반증 실패 8건). **rev2에서
코드/하네스/테스트/manifest는 단 한 바이트도 바뀌지 않았다** — §0의 sha 표는 rev1과 동일하다.

| # | 死因 / caveat | rev1 | rev2 | 절 |
|---|---|---|---|---|
| **死因 2** | F-a의 반증 분지에 판정 채널이 없어, H1을 반증하는 관측(907959 TD)이 등록 문안 직독으로 **`UNREALIZED`(측정 실패)** 로 강등된다 | "같은 배치에서 OOM하면 REFUTED / 그 배치가 안 생기면 UNREALIZED" | **F-a1/F-a2/F-a3로 분해.** 1차 판정은 **배치 크기와 무관**하게 "형성한 모든 배치를 OOM 없이 완주했는가"이며, 거짓이면 `UNREALIZED`가 아니라 **H1 미지지**. 검정력은 F-a2로 분리, 배치 동정은 F-a3(기록 전용)로 강등 | **§6-1** |
| **死因 1** | F-b의 `peak(boot)` 정의역이 ∅일 수 있고(907959 L2에서 실제로 ∅), ∅ 처분이 미등록 | `prefill_active_batch_size == 14` 위에만 정의, `UNREALIZED` 분지 없음 | **F-b1(1차, 기울기 축) + F-b2(2차, 10,125 축)** 로 분해. 둘 다 **∅ 처분을 명시 등록**. F-b1은 정의역 보장형(공통 서두 ordinal) | **§6-2** |
| ★**rev2 신규 발견** | 감사 R-2(b)가 제안한 "29개 토큰 수" 정의역에 **두 개의 잔여 자유 표면**이 남아 있었다 — (i) 29개 중 **3개가 boot마다 중복 출현**(`7`×2, `17`×8, `1468`×2)해 `peak(boot,T)`가 한 값이 아니고, (ii) **in-flight 표본화가 형성된 배치의 61–68 %만 관측**한다(§6-2-3 실측). 부수적으로 판정서가 든 `T=3241` 회귀 끝점은 **4 boot 어디서도 in-flight 관측되지 않았다** | — | ordinal 기반으로 재정의해 (i)을 **소거**, 정의역을 **in-flight가 아닌 epoch**로 잡아 배치 종료 후 표본까지 유효 표본으로 쓰고(그 편이 촘촘하나 907959로는 검증 불가), (ii)는 **측정된 기저율과 함께 희소/∅ 처분을 등록**. 감사의 29개 열거는 등록 상수로 **그대로 보존**(§6-2-2), 검정력은 **하한 2.88 GiB / 상한 4.04 GiB**로 양쪽 등록(RR-15) | **§6-2** |
| C1 | 계측 비용 "0.14 µs"가 계측기 비용인 것처럼 읽힌다 | "0.14 µs" | 조건을 명시하고 **"0.14 µs"를 인용 금지로 등재**(RR-8). ±3 % 결론은 유지 | **§2.3 · RR-8** |
| C2 | 리셋이 스케줄러 스레드 발행이라 TD에서만 in-flight decode를 자를 수 있다 | 미등재 | **잔존 비대칭 + 편향 방향(TD 과소 보고 = 대역 A 쪽) + 상한 156 MiB**를 등록, Δ 인용 시 병기 의무 | **§2.2 · RR-9** |
| C3 | `INSTR=0`이 warm-up의 JIT 예열을 줄여 배치 형성 타이밍을 흔든다 | 미등재 | 등재 + **"907959와 같은 조건에서 배치가 형성될 것"이라고 쓸 수 없음** | **§7 · RR-10** |
| C4 | §11.2 재수록 3건이 **절단**됐다(특히 N-8 말미 "상류 병목 단정 금지") | 절단본 | **sha 핀 원문에서 전사**(R-3 반영) + §11.2를 전사 원본으로 쓰지 말라는 지시 | **§11.2** |
| C5 | 감사자 변이 M-J(실현값→설정값 에코)가 13개 테스트를 전부 통과 | "게이트 #176을 테스트로 고정" 뉘앙스 | **그렇게 쓸 수 없음**을 등재(RR-11). 코드는 실현값을 읽고 있어 결함 아님 | **§2.1 · RR-11** |
| C6 | F-d의 두 절이 항등식(거짓이 될 수 없음) | 4절 모두 예보처럼 서술 | 항등식 2절을 **기록 항목**으로 강등, 거짓 가능한 2절만 예보로 남김 + 거짓의 두 원인 병기 | **§6-4 · RR-12** |
| C7 | 대역 B가 도달 가능 Δ의 약 62 %를 흡수 · 보조 "곡선"은 곡선이 아니고 표본율이 arm마다 1.45× 다르다 | 미등재 | 등재(RR-13), 보조 산출물의 arm 비교 금지 | **§6-2 · RR-13** |
| C8 | `PDMUX_WORKER_GRAD_GUARD=none` arm을 D조건으로 달지 않는다 | §10이 "별도 사전등록 필요"라고만 | 감사 판정(scored boot으로 넣으면 **설계상 무조건 FAIL**, 비채점 형태도 하네스 변경 필요, 실비 ≈0.025 GPU-h)을 등재 | **§10** |
| C9 | §10 지지 문장은 10,125 배치가 형성될 때만 쓸 수 있는데 대체 문장이 없다 | 단일 문장 | **형성/미형성 두 경우의 문장을 각각 등록** | **§10** |
| C10 | RR-1…RR-7의 승계 사슬이 다음 회차까지 강제되지 않는다 | 미등재 | 다음 회차 사전등록의 **의무 조항으로 등록**(RR-14) | **§11.3** |

**변경하지 않은 것**: §0(sha 표) · §1(처치) · §2.1/§2.2의 설계 · §3(판정 규칙 승계) ·
§4(manifest) · §5(CPU 회귀 수치) · §8(예산) · §9(제출 게이트) · §11.1(승계 33건 목록).

---

## 0. 무엇이 바뀌고 무엇이 안 바뀌는가 (한 눈에)

| 축 | 907959 | 이 회차 | 성격 |
|---|---|---|---|
| model / backend / ctx / mem-fraction / max-running / split D44 / cudagraph ON / seed | 동일 | **동일** | 불변 |
| verdict rule v2 · 채점기 sha | `ec355e17…` | **`ec355e17…` (무수정 승계)** | 불변 |
| 클라이언트 sha | `95e10b49…` | **`95e10b49…`** | 불변 |
| **`engine_source_hash`** | `5b20b11d17a17e1b…` | **`eba74cbd2cebfbd0…`** | ★이동 |
| **`multiplexing_mixin.py` sha256** | `0fe9d570f151698c…` | **`e2a97b423f93ff6d…`** | ★이동 |
| **하네스 `r2_correctness.sbatch` sha256** | `bd40ad91b46a5f32…` | **`655382632e4804b3…`** (전체: `655382632e4804b360c40e399573f023e5e2b6a9c49a46587ebfddd2081aaede`) | ★이동 |
| `R2C_INSTRUMENT` | 1 | **0** (§7) | ★이동 |
| `PDMUX_MEM_TELEMETRY` | (존재하지 않음) | **1, 양 arm 동일** | ★신설 |
| runtime manifest 항목 수 | 25 | **25**(24개 해시 불변, 1개 이동) | 구조 불변 |

★**따라서 이 job은 907100 / 907456 / X1과 "같은 엔진"이 아니다.** 그 job들의 결론은
**(Zamba2-2.7B, triton, `engine_source_hash` 구판)** 한정으로 **그대로 동결**되며, 이 회차는
그것들을 되살리지도 확장하지도 무효화하지도 않는다. 역으로 이 회차의 결과도 그 쌍에
이식되지 않는다(NP-4 승계).

### 0-b. 스코프 튜플 (이 문서의 모든 문장에 붙는다)

> **(model `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` = `NemotronHForCausalLM`, 56층 ·
> backend `flashinfer` 휠 **0.6.10 이라고 보고한 설치본** · ctx **16384** ·
> `mem-fraction-static 0.82` · `max-running-requests 48` ·
> `--disable-radix-cache --chunked-prefill-size -1 --disable-overlap-schedule --random-seed 1` ·
> split **fixed D44** = green stream index **4** · **cudagraph ON** ·
> **verdict rule v2 (2026-09-11), 채점기 sha `ec355e17…`** ·
> **`engine_source_hash` `eba74cbd2cebfbd0ac112fc4693cac516dacf95ce4250d899e96572e7fafe41a`** ·
> **harness sha `655382632e4804b3…`** · `R2C_INSTRUMENT=0` · `PDMUX_MEM_TELEMETRY=1` ·
> `PDMUX_TRACE_FORCE_PREFILL=1` · `PDMUX_WORKER_GRAD_GUARD` **미설정(엔진 기본값
> `inference_mode`)** · A100-SXM4-80GB 108 SM 1장)**

### 0-c. 승계 문서 고정 (sha256)

| 문서 | sha256 |
|---|---|
| `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` (rev3) | `e832fc49a867a1764422171a36d2984308eb4f34bbe1321cfb6d685cdbed54ec` |
| `newpair_prereg/VERDICT_newpair_rules_2026-09-13.md` (1차 규칙층) | `45ee250d2fa0cebfdc32592772213d1384a3177875e362f4bb5856848ed2b8ec` |
| `newpair_prereg/VERDICT_newpair_rev2_2026-09-13.md` (재감사) | `cd1ffe8b6befdf5376aab66981603299e4a6b8df78c6c3edb1547beb17d86b91` |
| `audit_907959_2026-09-13/VERDICT.md` (결과 감사) | `f544544fc973e0abc2d1aa97905e4e005d47e657c30e64d9ea7717d3a48ddc71` |
| `audit_907959_2026-09-13/DIAGNOSIS_true_dual_oom.md` (엔진 진단) | `8c857bede1576d9e8042d759889aec3a9f034ed6167fea4093dd947bb9565bee` |
| `job_907959/verdict.txt` (직전 라벨) | `640fd3adaa531b2dffcdac2ba76d9f1dfa98bed0cb30ffbd772da5c9df846529` |
| **`rerun_prereg/VERDICT_rerun_rules_2026-09-14.md`** (rev1 규칙층 판정서, **`NO-GO`**) | `71f3cfb7d8db6ce11ea3b24c5b4b3e894af0be1add0f693c40a7bc92f0226e65` |

---

## 1. 처치 (단 하나)

### 1.1 무엇을 고쳤는가

`event_loop_pdmux`는 `@torch.inference_mode()`로 데코레이트돼 있고 **그 가드는
thread-local**이다. legacy는 `run_batch`를 그 데코레이트된 스레드에서 부르지만,
true-dual은 `RoleWorkerThread`에 넘긴다 — 데코레이터가 닿은 적 없는 스레드다. 결과:
**TD arm의 모델 forward만 autograd가 켜진 채 돌았다.**

수리: `multiplexing_mixin._activate_role_context`가 role worker task를 **가드 안에서**
실행한다. 한 줄짜리 변경이며 **CUDA device/stream 가드와 같은 `with` 문**에 들어간다.

```
with guard(), torch.cuda.device(self.gpu_id), torch.cuda.stream(context.stream):
```

★**옵션 B(`mixer2_rms_norm_gated.py:97`의 `.data` 수리)는 이번에 하지 않는다.** 두 처치를
같은 회차에 넣으면 어느 쪽이 들었는지 가를 수 없다. B는 **상류 관찰로만 등재**한다:
그 줄은 이 저장소의 sync 대상도 manifest 대상도 아니므로(§4), B를 언젠가 하려면 manifest
항목 추가가 선행돼야 한다.

### 1.2 `inference_mode` vs `no_grad` — 선택과 그 근거 (실행 전 고정)

**선택: `torch.inference_mode()`.**

**(a) 대칭성이 이 수리의 목적이다.** `inference_mode`를 쓰면 true-dual에서 inference 의미론
아래 도는 연산 집합이 **legacy가 이미 그렇게 도는 집합과 정확히 같다**(메인 루프는
데코레이터를 유지하고, 두 worker가 같은 가드에 들어간다). 즉 legacy에서 inference tensor가
아니던 텐서가 true-dual에서 inference tensor가 되는 일이 없다. `no_grad`를 쓰면 worker가
만든 텐서는 **version counter를 가진 보통 텐서**가 되어, legacy는 inference tensor를,
true-dual은 보통 텐서를 다루게 된다 — **correctness 게이트의 두 arm이 서로 다른 텐서
의미론 위에서 모델을 돌리게 된다.** arm 비대칭을 다른 arm 비대칭으로 바꾸는 것은 수리가
아니다.

**(b) `no_grad`가 피하는 위험이 무엇이고, 왜 여기서는 발화하지 않는가 (CPU 측정, torch
2.9.1).** inference tensor가 no-grad 텐서보다 엄한 지점은 정확히 둘이며 **둘 다 소비자가
inference mode 밖에 있을 때만** 발화한다:

```
(c) inference tensor를 inference mode 밖에서 in-place 갱신
    -> RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
(d) inference tensor를 밖에서 autograd 추적 연산에 투입
    -> RuntimeError: Inference tensors cannot be saved for backward.
```

이 프로세스에서 worker 결과의 **모든 소비자가 inference mode 안**이다 — 코드로 확인한 것:

| 경로 | 확인 |
|---|---|
| `prefill_future.result()` / `decode_future.result()` | `event_loop_pdmux` 본문 안에서 읽힌다 ⇒ 데코레이터 안 |
| KV / mamba 풀, cudagraph static buffer | 부팅 시 할당된 **보통 텐서**. inference mode 안에서 보통 텐서를 in-place 갱신하는 것은 **허용**되고 텐서는 보통 텐서로 남는다(측정 확인). legacy가 이미 같은 일을 한다 |
| 보통 텐서의 view를 inference mode 안에서 만든 경우 | `is_inference() == False` — 밖에서 in-place 갱신 가능(측정 확인) |
| 텔레메트리 writer 스레드 | 파이썬 스칼라만 직렬화, 텐서 미접촉 |
| 두 worker가 동시에 가드 보유 | `InferenceMode`는 thread-local, 독립 — 측정 확인(배리어로 동시 보유 강제) |

**(c) 잔여 위험과 탈출구.** 위 분석이 어떤 미래 모델에서 틀릴 경우를 위해
`PDMUX_WORKER_GRAD_GUARD`(`inference_mode`|`no_grad`|`none`)를 둔다. **기본값이자 이 회차가
측정하는 값은 `inference_mode`이고, 하네스는 이 변수를 설정하지 않는다.** 다른 값으로
돌리는 것은 **새 스코프 튜플**이며 이 사전등록의 대상이 아니다. `none`은 수리 이전 거동을
재현하며 **회귀 테스트의 음성 대조 전용**이다.

### 1.3 수리가 소급 적용되지 않는 것 · 다만 소급되는 사실 하나

- **907100 / 907456 / X1의 결론은 이 수리로 바뀌지 않는다.** 그 job들은 Zamba2-2.7B
  (`mamba n_groups = 1`)라 `Mixer2RMSNormGated.forward_native`의 bare-Parameter 줄에
  **구조적으로 도달하지 않았다** ⇒ 이 retention 기전에 면역이었다.
- ★**그러나 "worker thread가 grad 켜진 채 돈다"는 성질 자체는 그 job들에도 있었다**
  (같은 코드). 토큰 값은 바뀌지 않지만 **autograd 부기(version counter·디스패치)의 arm
  비대칭 오버헤드**는 그 job들에 **존재했다** — 크기는 **미측정**이며, 이 회차도 그것을
  측정하지 않는다. (진단서 §9-1, claims-auditor 이관 항목.)

---

## 2. 계측 — `PDMUX_MEM_TELEMETRY=1` (양 arm 대칭)

### 2.1 무엇이 어디에 실리는가

`runtime_snapshot`의 **base payload**에 아래 9개 키가 추가된다. `runtime.metrics()`(true-dual
에만 존재)가 **아니라** base payload인 것이 핵심이다 — 그렇지 않으면 이 회차가 없애려는
바로 그 비대칭을, 그것을 재는 계측기가 다시 만든다.

| 키 | 내용 |
|---|---|
| `gpu_mem_allocated_b` | `allocated_bytes.all.current` — 순간 live 바이트 |
| `gpu_mem_peak_allocated_b` | `allocated_bytes.all.peak` — **마지막 리셋 이후** 피크 |
| `gpu_mem_reserved_b` | `reserved_bytes.all.current` |
| `gpu_mem_peak_epoch` | 리셋 창 번호(리셋마다 +1) |
| `worker_grad_guard` | 설정된 가드 이름(양 arm) |
| `{prefill,decode}_worker_grad_enabled` | ★**실현값** — worker 스레드가 가드 안에서 읽은 `torch.is_grad_enabled()` |
| `{prefill,decode}_worker_inference_mode` | ★**실현값** — 같은 위치의 `torch.is_inference_mode_enabled()` |

**대칭 보장**: 9개 키 전부 legacy 기록에도 **같은 주기로** 실린다. legacy는 worker 스레드가
없으므로 4개 realized 키가 `null`로 남는데, **그 `null`이 곧 기록해야 할 사실**이다(값을
0/False로 날조하지 않는다). CUDA 미초기화 환경에서도 3개 메모리 키는 **존재하되 `null`**
이다 — "측정된 0"으로 오독될 값을 만들지 않는다(테스트로 고정).

### 2.2 ★리셋 지점 — 등록 (이 선택이 측정 대상을 바꾼다)

> **`torch.cuda.reset_peak_memory_stats()`는 split-prefill 배치 시작(`_dual_worker_start_prefill`)
> 에서 정확히 한 번 호출한다. 즉 `gpu_mem_peak_allocated_b`는 "이 prefill 배치가 시작된
> 뒤의 device 할당 피크"다.**

- **왜 이 지점인가**: 리셋하지 않으면 `max_memory_allocated`는 프로세스 시작부터 단조라
  첫 큰 배치 이후 모든 스냅샷이 같은 수를 보고하고 **층별 곡선이 사라진다**. 배치 시작에서
  리셋하면 한 배치 안에서 값이 **단조 비감소**이므로 **그 배치의 마지막 표본이 곧 그 배치의
  피크**이고, 표본을 하나 놓쳐도 피크가 숨지 않는다.
- **왜 "스냅샷마다 리셋"이 아닌가**: 더 잘게 보이지만, 두 표본 사이에 뜬 피크가 **엉뚱한
  창에 귀속**되고 배치 내 단조성이 깨져 곡선이 읽히지 않는다.
- **왜 안전한가(코드 확인)**: `grep -rn 'max_memory_allocated|reset_peak_memory_stats|
  max_memory_reserved' sglang/srt/` **결과 0건**. 저장소의 히트는 전부 `multimodal_gen/`,
  `_mps_stub.py`, 벤치마크, test_utils이며 이 서버는 그중 무엇도 로드하지 않는다.
- **대칭**: 호출 지점은 `update_split_prefill_batch`의 무조건 경로에 있어 **양 arm이 같은
  사건에서** 리셋한다. 읽기와 **같은 플래그**로 게이트되므로 플래그 없는 실행은 무접촉.
- ★**(C2) 그럼에도 남는 arm 비대칭 — 등록한다.** 리셋은 **스케줄러 스레드에서 발행**되므로
  true-dual에서는 **다른 스레드의 decode forward 도중에 떨어질 수 있고**, legacy에서는
  (직렬화돼 있어) 그럴 수 없다. 즉 TD 쪽 피크 창만 in-flight decode의 일부를 잃는다.
  **편향 방향 = TD를 과소 보고하는 쪽 = 대역 A("F-b 충족") 쪽**이며, 크기 상한은 decode
  transient(cudagraph private pool **156 MiB** = ε의 15.6 %) 이하다. ★`Δ`(또는 `S`)를
  인용할 때 **이 편향 방향을 반드시 병기**한다(RR-9). "대칭 계측"은 **필드 위치와 발행
  스레드 양쪽**을 봐야 하며, 이 계측기는 전자만 만족한다.
- **리셋 안 하는 것**: `gpu_mem_allocated_b`(순간값)는 리셋이 필요 없고 그것만으로도
  retention 질문에 답한다. 그래서 피크 **대신**이 아니라 **함께** 싣는다.

### 2.3 ★관측자 효과 — 호출 빈도·비용·처분 (실행 전 고정)

- **빈도 = 기존 텔레메트리 주기와 동일.** 메모리 읽기는 **emit되는 `runtime_snapshot`마다
  1회**이며 새 emit을 만들지 않는다. 실측 빈도(job 907959): L1 = 12,458 snapshot / 56.0 s =
  **222.5/s**, TD1 = 7,554 / 49.3 s = **153.2/s** (이 하네스는
  `PDMUX_TRACE_FORCE_PREFILL=1`이라 prefill-in-flight sync마다 강제 emit한다).
- **★기각한 구현과 그 이유(측정치)**: `torch.cuda.memory_allocated()` /
  `max_memory_allocated()` / `memory_reserved()`는 각각 `memory_stats()`를 부르고, 그것이
  allocator의 중첩 stat dict를 **122개 항목으로 평탄화 + 정렬**한다. 이 venv에서 측정한
  순수 파이썬 비용은 **호출당 54.3 µs** ⇒ 3회면 스냅샷당 **163 µs** ⇒ 222.5/s에서
  **벽시계의 약 3.6%** 가 스케줄러 스레드에 추가된다. **그 철자만으로 ±3% 관측자 효과
  예산을 넘긴다.**
- **채택한 구현**: `torch.cuda.memory_stats_as_nested_dict()` **1회** + 중첩 dict 3키 읽기.
  ★**(C1 정정) 조건을 정확히 적는다 — 이 두 수는 서로 다른 입력에서 측정됐다.**
  `54.3 µs`는 **populated dict에 대한 flatten+sort**(감사자 재현 52.19 µs, leaf 122개)이고,
  `0.146 µs`는 **이미 만들어진 nested dict에서 3키를 읽는 비용**이다. 따라서
  **"계측기 비용이 0.14 µs"라고 쓸 수 없다**(RR-8). 남는
  `torch._C._cuda_memoryStats` C++ 호출 1회는 **여전히 미측정**이며, 동등한 nested dict를
  파이썬으로 구성하는 프록시(**4.18 µs** = 222.5 snap/s에서 벽시계 **0.09 %**)로만 상한이
  잡힌다. **±3 % 예산 결론은 그 상한 하에서 살아남는다**(0.09 % ≪ 3 %).
- **동기화 없음**: `memory_stats_as_nested_dict`는 allocator 락 아래 부기를 복사할 뿐
  device를 건드리지 않고, CUDA 초기화 전에는 예외 대신 `{}`를 돌려준다.
- ★**처분(등록)**: 이 계측기는 **paired on/off 측정이 아직 없다** ⇒ **P1 등 타이밍 수치를
  발표하는 어떤 실행에서도 켜서는 안 된다.** 이 회차는 correctness 게이트이고 타이밍 수치를
  발표하지 않으므로 켠다. P1 사용 전 paired on/off 측정이 **선행 조건**이다.
- **기본값 OFF**: 엔진 기본은 OFF이며, 플래그가 꺼진 실행의 기록은 수리 이전과
  **키 집합이 동일**하다(검증: pre-repair 모듈 대 post-repair 모듈로 같은 루프를 돌려
  45개 키 중 **43개가 값까지 동일**, 다른 2개는 `timestamp_monotonic_s`/`timestamp_s`뿐).

### 2.4 판정 규칙 무영향 (검증)

`r2_correctness_check.py`의 `SNAP_KEYS` 12개와 **충돌하는 새 키가 없고**, 플래그 ON에서도
`SNAP_KEYS` 전부가 생산된다 — 둘 다 테스트로 고정(§5). 채점기는 **한 글자도 고치지
않는다**(sha `ec355e17…`).

---

## 3. 판정 규칙 — verdict rule v2 무수정 승계

- 채점기 `r2_correctness_check.py` sha256 = **`ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30`**,
  변경 없음. 규칙 원문은 job 디렉터리의 `verdict_rule.txt`로 다시 핀된다.
- **라벨 해석·출력공간 전수·처분표**: `PREREG_NEWPAIR_2026-09-13.md` §8을 **무수정 승계**
  (FAIL / `NO_VERDICT_INFRA` / `INCONCLUSIVE` / `NO_VERDICT_UNREALIZED` / PASS의 뜻과
  각 라벨에서 쓸 수 있는 문장). **(D3) TD-TD S 불일치 = FAIL** 및 **(D2′) 귀무대조가 공허한
  S FAIL → `NO_VERDICT_INFRA` 보고** 조항도 문자 그대로 승계하며, D2′가 발화하면 **NPC-A의
  전사 의무가 함께 발화**한다.
- **§9 실패 모드 표 (a)–(g) 무수정 승계.** 단 (b)의 D12 분기는 이 회차에서 의미가 바뀐다:
  `R2C_INSTRUMENT`가 이미 **0**이므로 "INSTR=0으로 1회 재제출해 warm-up 잔류를 가른다"는
  진단 수단은 **이미 소진돼 있다**. 첫 scored boot이 부팅 중 OOM이면 그것은 mem 0.80
  재등록 사유가 아니라 **새 사전등록 사유**로 기록한다.
- **(D5) 재제출 정책 무수정 승계**: 재제출은 **`NO_VERDICT_INFRA`에 한해 최대 1회**.
  **`FAIL` · `INCONCLUSIVE` · `NO_VERDICT_UNREALIZED`는 재제출로 덮어쓰지 않는다.**
  재제출하면 두 job의 라벨을 **둘 다** 보고한다.

---

## 4. 수리의 provenance — manifest에 잡히는가

`sync_engine_tree.sh`의 25항목 manifest는 `sglang/srt/multiplex/multiplexing_mixin.py`를
**3번째 항목**으로 해시한다. 실제 확인:

```
907959 : 0fe9d570f151698cb4120320bea8bea1111dff052e179085349b97bcbf1a1697  .../multiplexing_mixin.py
이 회차: e2a97b423f93ff6d09f08b1e649596dad010285775845261ed27233c0346f243  .../multiplexing_mixin.py
나머지 24항목: 전부 불변
```

⇒ 엔진 소스를 고쳤고 **해시가 움직였다**. provenance 구멍 없음. 파생으로
`engine_source_hash`(8개 모듈 내용 해시)도 `5b20b11d17a17e1b…` → `eba74cbd2cebfbd0…`로
이동한다 — 이 회차 전에 측정된 어떤 hybrid profile도 **fail-closed로 비호환** 처리된다
(설계된 거동이며, 이 게이트는 `PDMUX_R2_POLICY=fixed`라 profile을 쓰지 않는다).

★**이번 회차가 고치지 않은 파일**: `sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py`
(옵션 B). 이 파일은 **manifest에 없다** — 언젠가 B를 하려면 manifest 항목 추가가
선행돼야 하며, 그것은 별도 회차다.

---

## 5. CPU 회귀 — 실행 전에 이미 통과한 것

★**전체 discovery 수치는 기준선이 움직인다** — 병행 워크스트림 B가 `test_lambda0_prereg`
계열을 rev4로 편집 중이라 같은 세션 안에서 **590 → 615 → 630**으로 바뀌었고 실패 항목도
바뀌었다(590회차: `TestMutationHarness.{test_harness_covers_all_three_modules, test_no_escapes}`
2건 + 간헐 `errors=1` / 630회차: `TestEveryVerdictIsReachable.test_rev3_rule_still_fails_here`
1건). **모든 실패가 `test_lambda0_prereg` 단일 모듈**이며 이 변경과 무관하다. 따라서
합격 기준은 **NPC-H 승계대로 본 트랙으로 한정**한다.

| 항목 | 결과 (2026-09-13→14 KST 측정) |
|---|---|
| ★**본 트랙** `test_r2_correctness_*` | **51 / 51 OK** |
| 신규 `test_worker_grad_guard.py` + `test_mem_telemetry_symmetry.py` | **25 / 25 OK** (13 + 12) |
| 엔진 인접 12개 모듈(`test_host_worker_metrics`, `test_dual_worker`, `test_trace_force_prefill`, `test_true_dual_prefill_ownership`, `test_r2_admission_{latch,persistence}`, `test_sticky_partition`, `test_profile_controller`, `test_controller_defaults`, `test_green_readout`, `test_probe_flush_durability`, `test_line_citations`) | **179 / 179 OK** |
| 전체 discovery | **630 tests, 실패 1** — `test_lambda0_prereg`(워크스트림 B). **이 변경이 만든 실패 0건** |
| line-citation 드리프트(`--check --all`) | **88 compared, 0 violation** (수리가 밀어낸 인용 17건을 도구의 내용-앵커 제안대로 이동시킨 뒤 재스냅샷; 지문 다중집합이 이동 전후 **동일** = 내용 무변경) |

**변이 시험(교훈 53)** — 같은 관측 함수를 git HEAD의 **수리 이전** mixin과 설치된 수리본에
각각 적용:

```
PRE-REPAIR  (git HEAD): grad_enabled=True  inference_mode=False graph_built=True  -> GATE FAILS
POST-REPAIR (installed): grad_enabled=False inference_mode=True  graph_built=False -> GATE PASSES
```

즉 §5의 게이트는 **수리에 의존한다**. 추가로 `DEFAULT_WORKER_GRAD_GUARD`를 `"none"`으로
바꾼 변이본도 게이트를 실패시킨다(테스트로 고정).

---

## 6. ★반증 가능한 사전 예보 (실행 전 고정, 사후 예보가 아님을 스스로 증명한다)

### 6-0. 이 예보가 사전 예보인 이유 — **수리 전 값이 이미 관측돼 있다**

job 907959에서 **이미 관측된 것**(재해석 불가, 원자료에 그대로 있음):

- TD1/TD2 **둘 다** split-prefill 시퀀스 `55 → 368 → 1468 → 3241 → 6245`를 **완주**한 뒤,
  다음 배치에서 `mixer2_rms_norm_gated.py:97`의 **198.00 MiB** 요청에 `torch.OutOfMemoryError`.
- L1은 같은 지점에서 **`#new-seq: 14, #new-token: 10125`** 배치를 **완주**(`srv_L1.log:2592`).
- `tel_L1.jsonl`에는 `prefill_active_batch_size == 14`인 스냅샷이 **146개**,
  `prefill_chunk_progress` ∈ {6,12,18,24,30,36,42,48,54,56}. `tel_TD1.jsonl`에는 **0개**.
- 진단서가 **실행 전에** 계산한 TD 예측 천장 ≈ **6.5k–7.7k 토큰**, 관측은 6245 통과 /
  10125 실패 ⇒ 구간 안. 같은 모형이 T=10125에서 예측한 초과 retention = **12.61 GiB**.

⇒ 아래 F-a/F-b/F-c는 **아직 존재하지 않는 데이터**에 대한 예보이고, 그 반대 분지가
**이미 관측된 상태**라서 사후적으로 고를 여지가 없다.

### 6-1. F-a — TD가 자기가 형성한 배치를 완주한다 (★rev2에서 재정의)

#### 왜 rev1을 버리는가 (死因 2)

rev1은 반증 분지를 **"같은 배치에서 죽었는가"** 위에 세웠는데, **그 배치를 동정하는 채널이
등록돼 있지 않았다.** `report_prefill_stats`는 prefill **완료 시** 호출되므로(rev1 §6-1이
스스로 적었다) **죽은 배치는 로그 줄을 남기지 않는다** ⇒ "형성됐지만 죽었다"와 "형성되지
않았다"를 등록된 채널로 구별할 수 없다. 결과 감사가 907959에서 그 구별을 해낸 방법
(C층 프롬프트 합 24,993 · TD 완주 11,377 · 잔여 13,616 = 10,125+3,491 · `#queue-req` 14→0 ·
`198.00 MiB ↔ T ∈ [10036,10137]` 역산)은 **다채널 법의학 재구성**이고 rev1에 한 줄도
채널로 등록돼 있지 않았다.

그 결과 rev1의 문안을 **907959 자신의 TD 원자료**에 먹이면: 10125 줄 **0건** ·
`==14` 스냅샷 **0개** · 완주 최대 **6,245 < 10,036** ⇒ 직독으로 **`UNREALIZED`(측정 실패,
H1의 실패가 아님)** 이 나온다. 즉 **이 프로젝트가 H1의 정본 증거로 삼는 관측**
(결과 감사 §3-4 = `CONFIRMED(scoped)`)을 측정 실패로 강등한다 — 결과 감사 §11이 명시적으로
막아 둔 방향("게이트 #21의 **역**도 규율이다 — 진짜 게이트 실패를 측정 실패로 강등하지
마라")의 재개방이다. rev2는 1차 판정을 **배치 동정에서 완전히 분리**한다.

#### F-a1 — 1차 판정 (배치 크기·배치 동정과 무관)

> **TD 2/2 boot이 자기가 형성한 split-prefill 배치를 하나도 빠짐없이
> `torch.OutOfMemoryError` 없이 완주했다.**
> **채널(넷 모두 AND, 전부 항상 정의된다)**:
> 1. `srv_TD1.log`·`srv_TD2.log`에 `torch.OutOfMemoryError` **0건**,
> 2. 같은 두 로그에 `Scheduler hit an exception` **0건**,
> 3. `BOOT_FAILURES.txt` **부재**(또는 TD 라벨 줄 0개),
> 4. `r2_correctness_report.json`의 `per_boot.TD1.checks.request_errors == 0` **AND**
>    `per_boot.TD2.checks.request_errors == 0`.
>
> ★**이 술어가 거짓이면 F-a는 충족되지 않으며, 그것은 `UNREALIZED`가 아니라
> "H1이 이 회차에서 지지되지 않았다"는 뜻이다.** 측정 실패로 강등하는 경로는 F-a1에
> **존재하지 않는다.**

- **F-a1 거짓일 때의 처분 (등록)**: (i) 라벨은 채점기가 내는 대로 보고한다(F-c는 라벨을
  예보하지 않는다, §6-3), (ii) `FAIL`은 **D5에 따라 재제출 금지** — 원인 분해는 새
  사전등록의 대상, (iii) **수리 자체는 되돌리지 않는다**(대칭성 수리로서 독립적으로 옳다),
  (iv) 이 회차가 수집한 메모리 축(F-b1)이 다음 사전등록의 설계 입력이 된다.
- **검증(실행 전 확인 완료)**: 907959의 TD 원자료를 이 술어에 먹이면
  `torch.OutOfMemoryError` 각 **1건** · `Scheduler hit an exception` 각 **1건** ·
  `BOOT_FAILURES.txt` = `SERVER_DIED_DURING_CLIENT boot=TD1 / boot=TD2` ·
  `request_errors` 각 **31** ⇒ **F-a1 거짓**. 즉 rev2의 1차 판정은 정본과 **같은 방향**을
  낸다(rev1은 반대 방향을 낼 수 있었다). legacy 2 boot은 네 채널 모두 0/부재 —
  대조도 성립한다.

#### F-a2 — 2차 판정 (검정력만; 1차를 덮지 않는다)

> **TD가 완주한 최대 `#new-token`이 907959의 TD 천장 `6,245`를 넘었다.**
> 넘지 못했는데 **F-a1이 참이면 F-a2만 `UNREALIZED`** 로 기록한다(부하가 약해 이 회차에
> 검정력이 없었다는 뜻이며, H1의 실패가 아니다). F-a1이 거짓이면 F-a2는 채점하지 않는다.

- 문턱 `6,245`는 **측정된 상수**다(907959의 TD1·TD2 완주 최대, 두 boot 동일; 감사자와
  작성자가 각각 독립 재계산). 재량 0.

#### F-a3 — 기록 전용 (판정에 들어가지 않는다)

> (i) `srv_TD*.log`에 `#new-seq: 14, #new-token: 10125` 줄이 출현했는가,
> (ii) `tel_TD*.jsonl`의 `prefill_active_batch_size == 14` 스냅샷 수와 그중
> `prefill_chunk_progress == 56`의 수,
> (iii) 크래시가 있었다면 `Tried to allocate N MiB`에서 역산한
> `T = N_bytes / (10240 × 2)` 와 그 구간,
> (iv) 각 boot의 완주 split-prefill 배치 열 `(#new-seq, #new-token)` 전체.
>
> **전부 보고 항목이며 어떤 판정에도 들어가지 않는다.** (iv)는 F-b1의 정의역을 구성하는
> 원자료이므로 결과 문서에 **반드시 전사**한다.

### 6-2. F-b — 메모리 축 (★rev2에서 재정의: 1차 = 기울기, 2차 = 10,125 축)

#### 6-2-1. 왜 rev1을 버리는가 (死因 1) — ∅ 정의역이 **이미 관측돼 있다**

rev1의 `peak(boot)`는 `prefill_active_batch_size == 14` 위에만 정의돼 있었고, **F-a에는 있던
`UNREALIZED` 분지가 F-b에는 없었다.** 그런데 907959 원자료에서 그 정의역은 **실제로
비어 있었다**(작성자 독립 재집계, 감사와 일치):

```
boot   ==14 스냅샷   완주 배치 열의 꼬리                     완주 최대 #new-token
L1        146        … 8/6245, 14/10125, 3/3491                    10,125
L2          0        … 9/7345, 16/12516          <- 14-seq 배치 없음  12,516
TD1         0        … 8/6245                                       6,245
TD2         0        … 8/6245                                       6,245
```

⇒ **같은 arm·같은 코드·같은 seed·같은 클라이언트의 두 legacy boot이 C층 배치를 다르게
형성했다.** `max{}` over ∅ 상황에서 채점자에게 남는 선택 3가지(빈 boot 제외 / 그 boot의
최대 배치로 대체 / `UNREALIZED`)를 rev1 문안이 **하나도 금지하지 않았고**, 앞의 둘은 Δ를
**0.236–0.413 GiB**(ε의 23.6–41.3 %) 움직여 **대역 B→A, D→A**를 뒤집는다.

#### 6-2-2. F-b1 — 1차 추정량: **공통 서두 ordinal 위의 peak–T 기울기** (정의역 보장형)

**정의역 구성 (재량 0, 순서대로 기계적으로 적용).**

1. **공통 서두 `P`** — 각 booted boot의 `srv_<label>.log`에서 정규식
   `Prefill batch, #new-seq: (\d+), #new-token: (\d+)`으로 완주 배치 열을 순서대로 뽑는다.
   `P` = 4 boot **전부**에서 `(#new-seq, #new-token)`이 **ordinal별로 동일**한 최대 서두 길이.
   **907959에서 `P = 38`**(작성자 전수 재계산; 39번째에서 L2가 `9/7345`로 갈라진다).
2. **epoch↔ordinal 대응** — `gpu_mem_peak_epoch`는 split-prefill 배치가 **형성될 때마다**
   1씩 증가하므로(§2.2), 어떤 boot에서도 죽은 배치가 없는 구간에서는 **epoch n ↔ n번째
   형성 배치**다. F-b1은 `n ≤ P` 구간만 쓰며, 그 구간은 모든 boot에서 완주가 확인된
   구간이다(구성 1이 로그 줄의 존재를 요구한다).
3. **2채널 일치 검사(필수)** — epoch `n`은 배치 `n`이 **형성된 순간부터 배치 `n+1`이
   형성될 때까지**의 구간이므로, 그 구간의 스냅샷은 (a) 배치 `n`이 in-flight인 것과
   (b) 배치 `n`이 끝난 뒤 idle인 것 둘 다를 포함한다. ★**(b)도 유효한 표본이다** — 피크는
   다음 리셋까지 유지되므로 idle 스냅샷이 오히려 그 배치의 **최종 피크**를 담는다.
   따라서 검사는 다음과 같다:
   > `V(b, n)` = { `prefill_active_batch_size` : `gpu_mem_peak_epoch == n` } 에서 **0을 제외**한 값 집합.
   > `V(b, n)` 이 **공집합이 아니면서** `{ #new-seq(ordinal n) }` 과 **다르면** ⇒ 그 `n`을
   > `D_b`에서 **제외**한다(epoch↔ordinal 대응이 깨진 증거).
   > `V(b, n)` 이 공집합이어도(= idle 표본만 있어도) **제외하지 않는다.**
   > `gpu_mem_peak_epoch == n` 스냅샷이 **아예 0개**면 제외한다(값이 없다).

   (죽은 배치 동정 채널이 없다는 死因 2의 교훈을 이 축에도 적용한다 — 대응을 **가정하지
   않고 검사**한다. 단 검사가 유효 표본을 버리지 않도록 정의역은 in-flight가 아니라
   **epoch**로 잡는다.)
4. **추정량** — boot `b`의 정의역 `D_b` 위에서
   `peak(b, n) = max{ gpu_mem_peak_allocated_b : gpu_mem_peak_epoch == n }`
   (in-flight·idle 스냅샷을 **모두** 포함),
   회귀변수 `T(n)` = ordinal `n`의 `#new-token`.
   `slope_b` = `peak(b, ·)` 대 `T(·)`의 **OLS 기울기**(MiB/token).
   **판정량** `S = | mean(slope_TD1, slope_TD2) − mean(slope_L1, slope_L2) |`.

**★감사가 제안한 "29개 토큰 수"와의 관계(등록 상수 보존 + 잔여 자유 표면 제거).**
판정서 §6-R-2(b)가 등록 상수로 열거한 29개 값

```
{1, 6, 7, 9, 17, 24, 34, 54, 55, 75, 183, 368, 371, 453, 732, 1113, 1255, 1280, 1409,
 1468, 1499, 1672, 1839, 1845, 2019, 2187, 2197, 2310, 3241}
```

은 **위 `P = 38` 서두의 서로 다른 토큰 수 집합과 정확히 일치한다**(작성자 독립 재계산).
그대로 보존한다. 다만 **토큰 값**으로 정의역을 잡으면 자유 표면이 남는다 — 그 29개 중
**`7`은 boot마다 2회, `17`은 8회, `1468`은 2회 출현**하므로 `peak(boot, T)`가 한 값이 아니고,
"첫 번째 / 마지막 / 최대 / 평균" 중 무엇을 쓸지가 미등록 재량이 된다. **ordinal로 잡으면
그 재량이 소거된다**(38개 ordinal이 29개 값의 38회 출현을 1:1로 지정한다). rev2는 그래서
ordinal을 쓴다. 907959의 `P = 38` 서두는 다음과 같다(결과 문서가 전사할 기준):

```
n :  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
T :  7   1   6  24   9   7  34  75 183 371 732 1113 1468 1845 2187 54 453 1255 17
n : 20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
T :1280 17 1409 17 1499 17 1672 17 1839 17 2019 17 2197 17 2310 55 368 1468 3241
(#new-seq = 1 for n = 1..37, = 4 for n = 38)
```

#### 6-2-3. ★rev2 신규 발견 — 표본화가 배치를 놓친다 (그리고 그것이 정의역 선택을 바꾼다)

`PDMUX_TRACE_FORCE_PREFILL=1`은 "prefill이 in-flight인 **sync**마다" 방출하는 것이지
"배치마다"가 아니다. 두 sync 사이에 시작하고 끝난 짧은 prefill은 **in-flight 스냅샷을 하나도
남기지 않는다.** 907959 실측(작성자, 텔레메트리의 prefill-in-flight 에피소드를 경계로 세어
로그의 완주 배치 수와 대조):

| boot | 완주 배치 수 | in-flight 에피소드 | 공통 서두(38) 중 **in-flight로 관측된** ordinal 수 |
|---|---|---|---|
| L1 | 41 | 27 | **26 / 38 (68 %)** |
| L2 | 40 | 27 | **26 / 38 (68 %)** |
| TD1 | 39 | 26 | **25 / 38 (66 %)** |
| TD2 | 39 | 24 | **23 / 38 (61 %)** |

- ★**이 61–68 %는 "in-flight 술어"의 커버리지다** — 즉 rev1이 쓴 `prefill_active_batch_size == 14`
  같은 술어가 볼 수 있는 범위이며, **F-b1의 epoch 기반 정의역과 같지 않다.** epoch `n`은
  배치 `n` 형성부터 배치 `n+1` 형성까지이므로 **배치가 끝난 뒤의 idle 스냅샷도 epoch `n`에
  속하고 그 배치의 최종 피크를 담는다**(§6-2-2 단계 3). 스냅샷은 초당 154–223개 방출되므로
  epoch 기반 정의역은 in-flight 정의역보다 **훨씬 촘촘할 것으로 예상**된다.
- ★★**그러나 그 예상은 검증되지 않았다** — 907959에는 `gpu_mem_peak_epoch` 필드가 아예
  없으므로 epoch별 표본 수를 **원자료로 확인할 방법이 없다.** 그래서 rev2는 낙관적 예상을
  등록하지 않고 **§6-2-4(i)(ii)의 희소/∅ 처분을 실행 전에 등록**한다. 정의역이 얇게 나오면
  그것은 `UNREALIZED`이지 H1의 실패가 아니다.
- ★**ordinal 38(`4 seq / 3241 tok`)은 4 boot 어디서도 in-flight로 관측되지 않았다.** 따라서
  판정서 §6-R-2(b)의 *"T=1→3241 구간 … 예상 신호 4.04 GiB = ε의 4배"* 는 **실현이 보장되지
  않는다.** 등록하는 두 수치:
  - **보수(검정력 하한)** — ordinal 34, `T = 2310` 까지만 들어올 경우:
    `2310 × 1.2750 MiB = 2.88 GiB` = **ε의 2.9배**(진단서 §2의 O층 값과 일치).
  - **낙관** — ordinal 38이 epoch 정의역에 들어올 경우: `3241 × 1.2750 MiB = 4.04 GiB`
    = ε의 4배(판정서의 값).
  어느 쪽이든 **10,125 축보다 검정력이 크다**(정의역에 ∅ 처분이 붙어 있고 점이 20여 개).
  결과 문서는 **실제로 들어온 최대 `T`를 반드시 병기**한다(RR-15).
- in-flight 관측 누락은 **짧은 배치에 편향**돼 있다(긴 prefill일수록 sync를 많이 거친다)
  ⇒ 회귀의 레버리지를 주는 큰 `T`는 보존되는 쪽이다. 다만 이 편향은 **측정되지 않았고
  arm마다 다를 수 있다**(스냅샷 발행률 L1 223.4/s vs TD1 154.4/s = 1.45×, caveat C7).

#### 6-2-4. F-b1 판정 규칙 (실행 전 고정)

> **(i) 희소/∅ 처분** — 어떤 booted boot에서든 `|D_b| < 10` 이거나 `max{T(n) : n ∈ D_b} < 2000`
> 이면, 또는 booted boot이 4개 미만이면 ⇒ **F-b1은 `UNREALIZED`이며 라벨을 내지 않는다.**
> (907959 기저율: **in-flight** 기준 서두 관측 23–26 / 38 — epoch 기준은 더 촘촘할 것으로
> 예상되나 그 필드가 없어 **확인 불가**이다(§6-2-3). 두 조건 충족은 **보장이 아니다.**)
> **(ii) 도메인 중첩 요건** — `|D_TD1 ∩ D_TD2 ∩ D_L1 ∩ D_L2| < 8`이면 F-b1은 `UNREALIZED`
> (arm 간 레버리지가 달라 기울기 비교가 교락된다).
> **(iii) 판정** — 문턱 `θ = 0.3 × 1.2750 = 0.3825 MiB/token`.
>
> | 대역 | 조건 | 해석 (등록) |
> |---|---|---|
> | **A′** | `S ≤ θ` | **F-b1 충족.** 수리 후 두 arm의 peak–T 기울기가 구별되지 않는다 |
> | **B′** | `θ < S ≤ 0.6375` (= 예측 기울기의 50 %) | ★**H1 부분 반증** — 미확정으로 기록, 다음 회차 설계 입력 |
> | **C′** | `S > 0.6375` | ★**H1을 지배 원인으로 보는 해석은 `REFUTED`** |
> | **D′** | `mean(slope_TD) < mean(slope_L) − θ` | 예상 밖. **해석하지 않고 보고만 한다** |
>
> **(iv) 필수 병기** — `D_b` 전체(ordinal·T·`peak`·`stream_index`)와 `slope_b`·절편·R²,
> 그리고 각 점의 `stream_index`. ★**다수 점이 prefill-only(비분할 idx 0) 구간이므로
> 이 추정량은 D44 운영점의 값이 아니다**(결과 감사 N-7 계열). "D44에서 쟀다"고 쓸 수 없다.

**θ가 판단이라는 공시**: `0.3825 MiB/token`은 **ε와 정확히 같은 성격의 사전 판단**이다.
rev2의 R-2는 **판단을 없애지 않고 ∅ 정의역을 없앤다** — 死因은 후자였다.

#### 6-2-5. F-b2 — 2차: 10,125 축 (rev1의 축을 ∅ 처분과 함께 유지)

> **(a) ∅ 처분 (신규 등록)** — 4 booted boot 중 **하나라도** `prefill_active_batch_size == 14`
> 스냅샷이 **0개**면 ⇒ **F-b2는 `UNREALIZED`이며 대역 라벨을 내지 않는다.** 빈 boot을
> 평균에서 제외하는 것도, 그 boot의 다른 배치로 대체하는 것도 **금지한다.**
> (907959에서 L2가 실제로 0개였다 — 이 분지는 가설이 아니라 관측된 사건이다.)
> **(b)** 4 boot 전부 ≥1개이면 rev1의 정의를 그대로 쓴다:
> `e*` = 그 boot에서 `==14` 스냅샷이 가장 많은 `gpu_mem_peak_epoch`,
> `peak(boot) = max{ gpu_mem_peak_allocated_b : gpu_mem_peak_epoch == e*, prefill_active_batch_size == 14 }`,
> `Δ = mean(peak(TD1), peak(TD2)) − mean(peak(L1), peak(L2))`, **ε = 1.00 GiB**.

| 대역 | 조건 | 해석 (등록) |
|---|---|---|
| **A** | `Δ ≤ +ε` | **F-b2 충족.** 수리 후 두 arm의 배치 피크가 구별되지 않는다 |
| **B** | `+ε < Δ ≤ +6.30 GiB` | ★**H1 부분 반증.** 미확정으로 기록, 다음 회차 설계 입력 |
| **C** | `Δ > +6.30 GiB` (= 예측 초과 12.61 GiB의 절반 이상) | ★**H1을 지배 원인으로 보는 해석은 `REFUTED`** |
| **D** | `Δ < −ε` | 예상 밖. **해석하지 않고 보고만 한다** |

**ε의 근거(판단임을 명시)**: 관측된 TD 전용 양성 성분 중 가장 큰 것이 cudagraph private
pool **156 MiB**이고, 14-seq / 21k full-token decode의 순간 성분은 GiB 단위가 아니다.
1.00 GiB는 그 최대 식별 성분의 약 6.4배이자 예측 초과분(12.61 GiB)의 약 8 %다.
★**수리 후의 양성 arm 간 피크 차이는 측정된 적이 없다** — ε는 측정이 아니라 사전 판단이며,
대역 B는 그래서 "실패"가 아니라 "미확정"이다.

**★대역 B의 흡수율(C7 등재)**: F-a1이 충족된 세계에서 도달 가능한 Δ의 상한은
`11.59 GiB − peak_L` ≈ **8.1–9.6 GiB**(진단서의 legacy working-set 2.0–3.5 GiB 기준)이며,
"미확정"으로 등록된 대역 B(1.00–6.30 GiB)가 그 구간의 **약 62 %**를 흡수한다. 즉
**F-a1 충족 조건부로 F-b2가 H1에 불리한 판정을 낼 여지는 좁다.** RR-5·RR-13과 함께 인용하라.

#### 6-2-6. 보조 산출물 (예보 아님, 기록 전용)

`gpu_mem_allocated_b` 대 `prefill_chunk_progress` 를 **양 arm 각각** 기록한다.
★**이것을 "곡선"이라 부르지 않는다**: 907959 L1의 `==14` 스냅샷 146개 중
progress 6·12·…·54는 **각각 정확히 2개**이고 나머지 **128개가 종단 progress 56**에 몰려 있다.
게다가 스냅샷 발행률이 arm마다 1.45× 다르다(L1 223.4/s vs TD1 154.4/s). 피크는 창 내
단조성 덕에 이 교락에 면역이지만 **순간값의 arm 비교는 표본율 교락을 안는다** ⇒
**arm 간 비교에 쓰지 않는다**(C7 / RR-13).

### 6-3. F-c — 게이트 위생

> `r2_correctness_report.json`의 `per_boot.<label>.checks.request_errors == 0` **4 boot 전부**
> (907959: L1/L2 = 0, **TD1/TD2 = 31**) · `verdict.txt`의 **S 티어 6쌍 전부 0 · O 티어
> within-arm 2쌍 + cross-arm 4쌍 전부 0** · **cudagraph ON 유지**(모든 boot의 모든
> `Decode batch` 줄이 `cuda graph: True`, 즉 `decode T/F=n/0`) · `BOOT_FAILURES.txt` 없음 ·
> 4 boot 전부 `booted=True crash_free=True`.

- 어긋나면: 채점기 라벨이 그대로 처분이다(§3). F-c는 **라벨을 예보하지 않는다** —
  §3.3(flashinfer 적응 split-KV) 때문에 PASS·INCONCLUSIVE 어느 쪽도 사전 확률을 주장하지
  않는다는 rev3의 입장을 그대로 승계한다.

### 6-4. F-d — 가드가 **실현**됐음을 텔레메트리가 보인다 (게이트 #176) — ★rev2에서 범위 축소

> **예보(거짓이 될 수 있는 부분만)** — TD boot의 스냅샷에
> `prefill_worker_grad_enabled == false` **그리고** `prefill_worker_inference_mode == true`가
> 나타나고, decode 쪽 두 키도 같다.

- 어긋나면: 두 원인이 가능하며 **구별하지 않고 측정 실패로 처리한다** — (i) 설치 트리가
  수리본이 아니다, (ii) `PDMUX_MEM_TELEMETRY` 플래그가 전파되지 않았다. 어느 쪽이든
  **`NO_VERDICT_INFRA`(측정 실패)** 로 보고하고 §3의 재제출 1회를 쓴다. 게이트 실패가 아니다.
- ★**기록 항목(예보 아님 — 항등식이라 거짓이 될 수 없다, C6)**:
  `worker_grad_guard == "inference_mode"`는 하네스가 `PDMUX_*`를 전부 unset하므로 **거짓이 될
  수 없고**, legacy의 realised 4키 `null`은 **worker 스레드가 없다는 것의 연역**이다.
  둘 다 처분이 붙지 않는 **기록**이며 **"F-d가 충족됐다"를 정보량 있는 확인으로 인용할 수
  없다**(RR-12).
- ★**게이트 #176 충족을 "테스트로 고정했다"고 쓸 수 없다**(C5): 감사자 변이 **M-J**
  (`_r2_record_worker_guard`의 `torch.is_grad_enabled()` / `torch.is_inference_mode_enabled()`를
  설정값 에코로 교체)가 **등록된 13개 테스트를 전부 통과**했다. 제출된 코드는 실현값을 읽고
  있으므로 결함은 아니지만, **실현값 성격은 파일 단위 manifest 해시로만 보호되고 변이 대조로는
  보호되지 않는다**(RR-11). (같은 주입에서 M-C[가드를 텔레메트리 플래그로 게이팅]·
  M-D[PREFILL만 가드]·M-K[가드 진입 후 yield 전 탈출]은 전부 잡혔다.)

---

## 7. `R2C_INSTRUMENT` — 끈다 (0), 근거

**결정: `R2C_INSTRUMENT=0`.**

1. **새 정보가 없다.** λ_inf(A)=3.0939는 907959에서 이미 얻었고, λ_inf(B)=0.6956은 결과
   감사가 **F5 셀 B 반증**으로 **인용 불가** 처리했다(N-8). 같은 레시피를 다시 돌리면
   **같은 결함을 재생산**할 뿐이다 — 그 결함의 수리(prefill 지배 shape의 포화 계기를
   `#running-req` 대신 `#queue-req`로 바꾸는 것)는 감사 §13-D2가 **별도의 새 사전등록**을
   요구한다.
2. **처치 축을 하나로 유지한다.** 이 회차의 처치는 grad guard 하나다. 계측 레시피까지 같이
   바꾸면 어느 쪽이 무엇을 움직였는지 가를 수 없다.
3. **예산·최악 경계가 줄어든다**(§8): read-out 3000 s timeout 분지가 사라져 구조적 최악이
   161분 → 약 111분이 되고 `--time=02:30:00` 안에 **들어온다**(907959 구성에서는 들어오지
   않았다).

★**따라서 감사 §13-D1(i)의 수리 — 하네스가 `I3_max_running_req.txt`를 셀별 2줄로 쓰게
하는 것 — 은 이 회차에 포함하지 않으며, 그 결함은 다음 회차로 남는다.** 이 문장을
등록해 두는 이유는, 나중에 "왜 D1(i)이 안 닫혔는가"가 회고적 재량으로 보이지 않게 하기
위해서다. (감사가 권장한 D1(ii) — 술어가 `I_log_offsets.txt` + `srv_warmup.log`에서 스스로
셀별 재계산 — 은 **GPU 0·소급 적용 가능**이고 λ0 워크스트림 소관이라 이 회차와 독립이다.)

### ★7-b. `INSTR=0`의 **미등재 부작용** — rev2에서 등록 (C3)

위 세 이유 외에 `R2C_INSTRUMENT` 1→0은 **warm-up boot의 클라이언트 부하(I2/I3)를
제거**한다. `r2_correctness.sbatch:175`의 `TRITON_CACHE_DIR="$OUT/.triton_cache"`는
**job마다 새로 생성**되므로, 907959에서는 warm-up이 예열해 둔 JIT 캐시가 이 회차에서는
**덜 채워진 채** 첫 scored boot이 돈다.

★**이것은 C층 배치 형성 타이밍을 흔들며, 그 배치 형성이 바로 F-a3(기록)과 F-b1의 정의역,
F-b2의 존재 조건이 의존하는 양이다.** 따라서:

> **"907959와 같은 조건에서 (같은) 배치가 형성될 것"이라고 쓸 수 없다**(RR-10).
> 배치 형성의 비결정성은 같은 arm 안에서도 관측됐다(§6-2-1: L1과 L2가 C층 배치를 다르게
> 형성). rev2의 1차 판정(F-a1·F-b1)이 **배치 동정에 의존하지 않도록** 설계된 이유가 이것이다.

---

## 8. 예산과 `--time` (실측 재산정)

**실측 기준선**: job 907959 = **21분 21초 / 5 boot**, `R2C_INSTRUMENT=1`. 아티팩트 타임스탬프
분해 — 21:51:36 시작 → 약 22:04 (env sync + provenance + warm-up boot + I1/I2/I3) ≈ **12.5분**,
22:05 → 22:12 (scored 4 boot) ≈ **7.5분**. scored 구간이 짧았던 것은 **TD 2 boot이 조기
사망**했기 때문이다(L boot ≈ 2.2분, TD boot ≈ 1.3분).

| 시나리오 | 계산 | 합 |
|---|---|---|
| 전형(이 회차, INSTR=0, TD가 완주) | env/prov 1 + warm-up boot 1.5 + 4 × 2.2 + 채점/teardown 0.5 | **≈ 11.8분 ⇒ 0.20 GPU-h** |
| 현실적 최악(한 boot의 클라이언트가 길어짐) | 위 + 8분 | **≈ 20분 ⇒ 0.33 GPU-h** |
| 구조적 최악(모든 timeout 발화) | warm-up health 900 s + 4 × (health 480 + client 900 + teardown 20) + 부대 180 s | **≈ 111분 ⇒ 1.85 GPU-h** |

**`--time`**: `02:30:00` **유지**(지시자 수정 없음). 월타임 상한은 어떤 측정량도 바꾸지
않고 큐 대기만 바꾼다. 이 구성에서는 **구조적 최악까지 덮는다**.

**트랙 장부**: R2 correctness 누적 `0.786` → 전형 시 **≈ 0.99 GPU-h**(현실적 최악 1.12,
하드캡 2.64). longctx_conflict 트랙 장부(15.42 GPU-h)와는 별개.

---

## 9. 실행 전 필수 조건 (제출 게이트)

1. `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh`가 돌아
   manifest의 `multiplexing_mixin.py` 해시가 **`e2a97b42…`** 임을 확인.
2. 전체 CPU 회귀를 돌리고 **`test_r2_correctness_*` 51/51**,
   `test_worker_grad_guard` 13/13, `test_mem_telemetry_symmetry` 12/12를 확인
   (전체 discovery의 실패는 병행 워크스트림 상태에 좌우된다 — **NPC-H** 승계).
   ★추가로 `scripts/discipline/check_line_citations.py --check --all`이 **0 violation**임을
   확인한다(수리가 `multiplexing_mixin.py`의 줄 번호를 밀기 때문에, 이 검사가 이 회차에서는
   회귀가 아니라 **처치의 부작용 점검**이다 — 교훈 #80의 3번 사례와 같은 형태).
3. **(D9 승계) 제출 전 커밋**: `provenance.txt`의 `commit=`이 실제로 돈 하네스를 가리키도록
   아래가 커밋돼 있어야 한다.
   ```
   workspace/engine-port/src/multiplex/multiplexing_mixin.py                   (M)
   workspace/engine-port/results/r2_correctness/r2_correctness.sbatch          (M)
   workspace/engine-port/tests/test_worker_grad_guard.py                       (??)
   workspace/engine-port/tests/test_mem_telemetry_symmetry.py                  (??)
   workspace/engine-port/results/r2_correctness/rerun_prereg/                  (??)
   workspace/engine-port/results/kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md   (M, 인용 이동)
   workspace/engine-port/results/kernel_mech/a1/DECISION_A1_Q3_CHANNEL_2026-08-25.md (M, 인용 이동)
   workspace/engine-port/results/kernel_mech/a1/a1_q3k1_rule.py                (M, 인용 이동)
   workspace/engine-port/scripts/discipline/line_citations.json                (M, 지문 재등록)
   ```
   ★이 문서를 쓴 세션은 커밋하지 않았다(병행 워크스트림이 같은 저장소를 미커밋 상태로
   편집 중). 통합·커밋은 메인 세션이 한다.
4. **claims-auditor의 규칙층 감사 통과 전 제출 금지.**
   ★**rev1은 `NO-GO`(死因 `N2` ×2)를 받았다**(판정서 sha `71f3cfb7…`). rev2는 그 두 死因을
   §6-1·§6-2에서 제거하고 caveat C1–C10을 반영했으며, ★**감사가 지적하지 않은 잔여 자유
   표면 2건**(29개 토큰 값의 중복 출현, 텔레메트리의 배치 관측률 61–68 %)을 추가로 닫았다
   (§6-2-2·§6-2-3). **rev2는 재감사를 새로 받아야 제출할 수 있다** — rev1의 판정으로
   갈음하지 않는다.

**제출 명령 (한 줄)**

```bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc && sbatch workspace/engine-port/results/r2_correctness/r2_correctness.sbatch
```

(`R2C_INSTRUMENT`는 기본값 0, `R2C_MEM_TELEMETRY`는 기본값 1이므로 **환경변수를 붙이지
않는다**. CLI로 `--comment`를 덮어쓰지 말 것 — 지시자에 이미 있다.)

---

## 10. 이 회차가 **하지 못하는** 것 (귀속 한계, 실행 전 고정)

★**이 회차는 처치를 분리하지 못한다.** 907959 대비 **세 축이 동시에 움직인다**:
`engine_source_hash`(grad guard), 하네스 sha(메모리 계측 + INSTR), `R2C_INSTRUMENT` 1→0.
따라서 F-a1이 충족되더라도 **"grad guard가 원인이다"는 이 job 단독으로 성립하지 않는다.**

### ★(C9) 지지 가능한 문장 — 두 경우를 **각각** 등록한다

rev1은 문장을 하나만 등록했는데 그 문장은 **10,125 토큰 배치가 형성될 때에만** 쓸 수 있고,
907959의 legacy 2 boot 중 1 boot이 그 배치를 형성하지 않았다(§6-2-1) ⇒ **형성되지 않았을 때
쓸 문장이 없었다.** rev2는 둘 다 등록한다.

> **(경우 1) `#new-token ∈ [10036, 10137]` 배치가 TD 2/2에서 형성·완주된 경우**
> — "위 스코프 튜플에서, 수리된 엔진의 true-dual boot 2/2가 907959의 true-dual boot 2/2를
> 죽였던 것과 **같은 크기의** split-prefill 배치를 완주했다."
>
> **(경우 2) 그 배치가 형성되지 않은 경우 (등록된 유일한 대체 문장)**
> — "위 스코프 튜플에서, 수리된 엔진의 true-dual boot 2/2가 **자기가 형성한 split-prefill
> 배치를 하나도 빠짐없이 OOM 없이 완주했고**, 완주한 최대 토큰 수는 `<값>`이었다
> (907959의 true-dual 천장은 6,245였다)." ★이 문장은 **"같은 크기"를 주장하지 않는다.**
> 완주 최대가 6,245 이하이면 이 문장에 **"이 회차는 907959의 TD 천장을 넘는 부하를
> 만들지 못했다(F-a2 `UNREALIZED`)"** 를 반드시 병기한다.
>
> 위 둘 **말고 다른 서술 문장은 쓸 수 없다.**

기전 귀속을 강화하는 **독립 증거**(이 job이 만드는 것이 아니라 이미 있는 것): (i) 코드 사실
— worker 스레드가 데코레이터 밖이라는 것과 `n_groups != 1`이 native 분기를 강제한다는 것,
(ii) CPU 측정 — bare Parameter 곱이 grad 상태에 따라 그래프를 만들거나 만들지 않는다는 것,
(iii) 907959의 **결정성** — 두 TD boot이 같은 배치·같은 할당 크기·같은 스택에서 죽었다.

### ★(C8) `PDMUX_WORKER_GRAD_GUARD=none` arm — **이 회차의 조건으로 달지 않는다**(감사 판정 등재)

진짜 분리는 같은 엔진 안의 `none` arm이지만, 규칙층 감사가 **이번 회차의 D조건으로 달지
않기로 판정**했고 rev2는 그 판정을 등재한다. 근거(실현가능성 검사, 게이트 #113):

- `r2_correctness_check.py:366`이 `boots.txt`에서 라벨을 열거하고 `:378`·`:446-447`이
  `startswith("TD")` / `startswith("L")`로 arm을 가른다 ⇒ `none` boot을 **scored boot으로
  넣으면 TD arm에 합류해 설계상 반드시 크래시 → 게이트가 무조건 `FAIL`** 이 된다.
- 유일하게 성립하는 형태는 **비채점 진단 boot**(warm-up 패턴)이며, 비용은 907959의 조기
  사망 TD boot 실측 1.3분 기준 **≈1.5분 ≈ 0.025 GPU-h**(rev1이 추정한 0.1 GPU-h가 아니다).
- 그 형태조차 **하네스 변경 + 테스트 + argv 픽스처 재고정**을 요구해 §7-2("처치 축을 하나로
  유지한다")를 스스로 깬다.

⇒ **별도 회차로 남긴다.** 그 회차는 **F-a1이 충족된 경우에만** 의미가 있다.

---

## 11. 인용 금지 — 승계 + 신규

### 11.1 문자 그대로 승계 (전부, 하나도 빼지 않는다)

아래 항목은 §0-c에 sha로 고정된 원문에서 **한 글자도 고치지 않고** 승계되며, 이 회차의
결과 문서는 그 원문을 **문자 그대로 전사**해야 한다.

- `PREREG_NEWPAIR_2026-09-13.md` §12: **NP-1′, NP-2, NP-3′, NP-4, NP-5, NP-6, NP-7′,
  NP-8, NP-9, NP-10** (10건)
- 같은 문서 §12의 재감사 캐비앳: **NPC-A, NPC-B, NPC-C, NPC-D, NPC-E, NPC-F, NPC-G,
  NPC-H, NPC-I, NPC-J** (10건) + **(D13-i) 스코프 배선 필수 전사 항목 2개**
- `audit_907959_2026-09-13/VERDICT.md` §12: **N-1 … N-13** (13건)

**합 33건 + 전사 의무 2건.** 결과 문서가 이 중 하나라도 누락하면 그 문서는 이 사전등록을
위반한 것이다.

### 11.2 ★특히 주의해서 승계할 3건 — **sha 핀 원문에서 전사** (rev2에서 절단 복원, C4)

> ★★**이 절은 전사 원본이 아니다.** rev1의 재수록 3건은 모두 **말미가 절단**돼 있었고
> (N-7: 정정의 출처, **N-8: "상류 병목 단정 금지" 조항**, NP-8: `(신규, D13)` 태그),
> 특히 N-8의 누락은 **상류 병목 귀속을 막는 바로 그 조항**이었다. 아래는 rev2에서
> sha 핀 원문(`audit_907959_2026-09-13/VERDICT.md` = `f544544f…`,
> `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` = `e832fc49…`)에서 **다시 전사**한 것이다.
> ★**결과 문서는 §11.1이 지시한 대로 sha 핀 원문에서 전사해야 하며, 이 절을 전사 원본으로
> 쓰지 마라.**

> **N-7 (I2의 분할 — ★NP-9/D6-ii 정정)** — **I2(동시성 1)의 TTFT 43.03 ms·ITL median 12.96 ms는 D44 값이 아니라 비분할(108 SM) 값이다.** 엔진은 prefill∧decode가 **동시에** 활성일 때만 D44(green idx 4)를 쓰고, 그 외에는 green context가 `null`인 idx 0/5에서 돈다(드라이버 read-out + 같은 job scored boot 텔레메트리 12,458 스냅샷 전수: idx0 11,725 / **idx4 592** / idx5 141). 동시성 1에서는 중첩이 구성상 불가능하다. **사전등록 §5-I2 (D6-ii)와 NP-9의 "D44 한정" 태그는 I2에 대해 거짓이며, I3에 대해서는 혼합비가 미측정이라 검증되지 않았다.** (λ0 rev1 판정서 `:151` "B=1은 pdmux 분할 미적용 구간"이 옳았다.)

> **N-8 (★F5 / λ_inf(B) 인용 금지)** — 사전등록 §5-I3 (D4)의 셀별 재계산 레시피로 계산한 결과 **I3b(8192,64)의 셀별 `#running-req` 최댓값은 2**(I3a는 48)다. **F5는 셀 B에서 반증됐고, 등록된 처분에 따라 `λ_inf(B)=0.6956 req/s`를 포화 처리율/상한 앵커로 인용할 수 없다.** 하네스가 기록한 `I3_max_running_req.txt = 48`은 **전역 최댓값**이며 셀 A에서만 온 값이다. 미달의 원인은 **미확정으로 기록**한다(사전등록 D4가 상류 병목 단정을 금지).

> **NP-8 (신규, D13)** — **PASS는 P2 캠페인 착수를 승인하지 않는다.** 이 게이트는 P2의
> **필요조건 하나**를 닫을 뿐이다. 남은 블로커: λ0(0단계) 사전등록이 `NO-GO`(死因 N2+N3),
> W4를 단일 λ\*로 파라미터화할 수 없다는 열린 항목(사용자 결정 대기), 그리고 방법론
> 게이트 #6(지표 절벽 대비 용량) 미충족(λ0 Q1).

### 11.3 신규 인용 금지 (이 회차 고유)

> **RR-1 (성능 금지)** — 이 회차의 어떤 수치도 성능 결과가 아니다. 특히
> **"true-dual이 빨라졌다 / 메모리를 덜 쓴다 / 오버헤드가 줄었다"는 쓸 수 없다.** 처치는
> 정확성·대칭성 수리이고, 이 job은 arm 간 타이밍을 비교하지 않으며, 계측기 자신의
> 관측자 효과가 아직 미측정이다(§2.3).

> **RR-2 (귀속 금지)** — §10 그대로. "grad guard가 OOM의 원인이었다"는 **이 job 단독으로
> 쓸 수 없다**(세 축 동시 이동). 쓸 수 있는 서술 문장은 **§10이 등록한 (경우 1)·(경우 2)
> 두 개뿐**이며, 어느 쪽을 쓰는지는 10,125 배치의 형성 여부가 결정한다(C9).

> **RR-3 (소급 금지)** — 이 수리는 907100 / 907456 / X1의 결론을 **바꾸지도 되살리지도
> 않는다.** 그 결과들은 (Zamba2-2.7B, triton, 구 `engine_source_hash`) 한정 동결이다. 단
> **"그 job들의 worker도 grad 켜진 채 돌았다"는 사실은 참**이며, 그로 인한 autograd 부기의
> arm 비대칭 오버헤드는 **크기 미측정**이다 — 그 job들을 "오염됐다"고도 "깨끗했다"고도
> 쓸 수 없다.

> **RR-4 (계측기 인용 한계)** — `gpu_mem_*` 필드는 **PyTorch 캐싱 할당자의 부기**이지
> 장치 전체 사용량이 아니다(드라이버·컨텍스트·다른 프로세스 미포함). `available_gpu_mem`
> 배너와 직접 비교하지 말 것. `gpu_mem_peak_allocated_b`는 **§2.2의 리셋 규약에 상대적**
> 이며 `gpu_mem_peak_epoch` 없이 인용할 수 없다.

> **RR-5 (ε와 θ는 측정이 아니다)** — F-b2의 `ε = 1.00 GiB`와 F-b1의
> `θ = 0.3825 MiB/token`은 **둘 다 사전 판단**이다. 대역 B/B′(부분 반증)는 "H1이 틀렸다"도
> "맞았다"도 아니며, **수리 후 양성 arm 간 피크 차이가 측정된 적이 없다**는 사실과 함께
> 인용해야 한다. rev2의 재정의는 **판단을 없애지 않고 ∅ 정의역을 없앤 것**이다.

> **RR-6 (`worker_grad_guard` 필드)** — 이 필드는 **설정값**이고,
> `*_worker_grad_enabled` / `*_worker_inference_mode`가 **실현값**이다. 둘을 같은 것으로
> 인용하지 말 것(게이트 #176). legacy의 `null`은 "grad가 꺼져 있었다"가 아니라
> **"worker 스레드가 없었다"**는 뜻이다.

> **RR-7 (D1(i) 미해결)** — 이 회차는 `R2C_INSTRUMENT=0`이므로 감사 §13-D1(i)
> (`I3_max_running_req.txt` 셀별 2줄)의 하네스 수리를 **하지 않았다.** 그 결함이 닫혔다고
> 쓸 수 없다.

#### rev2 추가분 (RR-8 … RR-15) — 규칙층 판정서 caveat C1·C2·C3·C5·C6·C7·C10의 반영

> **RR-8 (계측 비용 "0.14 µs" 인용 금지, C1)** — `0.14 µs`는 **계측기의 비용이 아니다.**
> 그것은 **이미 만들어진 nested dict에서 3키를 읽는 비용**(재측정 0.146 µs)이고,
> 기각된 철자의 `54.3 µs`는 **populated dict에 대한 flatten+sort**(재현 52.19 µs, leaf 122개)
> 로 **서로 다른 입력에서 측정**됐다. 남는 `torch._C._cuda_memoryStats` C++ 호출 1회는
> **여전히 미측정**이며 파이썬 프록시(4.18 µs = 벽시계 **0.09 %**)로만 상한이 잡힌다.
> **±3 % 예산 결론은 살아남지만 "0.14 µs"라는 표현은 인용할 수 없다.**

> **RR-9 (계측기 자신의 잔존 arm 비대칭, C2)** — `reset_peak_memory_stats()`는 **스케줄러
> 스레드에서 발행**되므로 true-dual에서는 다른 스레드의 decode forward 도중에 떨어질 수 있고
> legacy에서는 그럴 수 없다. **편향 방향 = TD 과소 보고 = 대역 A/A′ 쪽**, 크기 상한
> **156 MiB**(ε의 15.6 %). ★`Δ`(F-b2) 또는 `S`(F-b1)를 인용할 때 **이 편향 방향을 반드시
> 병기**하라. "대칭 계측"은 **필드 위치와 발행 스레드 양쪽**을 봐야 하며 이 계측기는
> 전자만 만족한다.

> **RR-10 (`INSTR=0`의 배치 형성 부작용, C3)** — `R2C_INSTRUMENT` 1→0은 warm-up의 I2/I3
> 부하를 없애 **JIT 캐시 예열을 줄인다**(`TRITON_CACHE_DIR`는 job마다 새로 생성). 이는 C층
> 배치 형성 타이밍을 흔든다 ⇒ **"907959와 같은 조건에서 배치가 형성될 것"이라고 쓸 수 없다.**

> **RR-11 (게이트 #176을 "테스트로 고정했다"고 쓸 수 없다, C5)** — 감사자 변이 **M-J**
> (실현값 읽기를 설정값 에코로 교체)가 **등록된 13개 테스트를 전부 통과**했다. 제출된 코드는
> 실현값을 읽고 있어 결함이 아니지만, **F-d의 실현값 성격은 파일 단위 manifest 해시로만
> 보호되고 변이 대조로는 보호되지 않는다.**

> **RR-12 (F-d의 두 절은 항등식, C6)** — `worker_grad_guard == "inference_mode"`는 하네스가
> `PDMUX_*`를 전부 unset하므로 **거짓이 될 수 없고**, legacy의 realised 4키 `null`은
> **worker 스레드가 없다는 것의 연역**이다. **"F-d가 충족됐다"를 정보량 있는 확인으로
> 인용할 수 없다.** 거짓이 될 수 있는 것은 TD의 realised 키뿐이고, 그 거짓은 "설치 트리가
> 수리본이 아니다"뿐 아니라 **"계측 플래그가 전파되지 않았다"** 로도 발생한다.

> **RR-13 (대역 B의 흡수율과 보조 산출물, C7)** — F-a1 충족 조건부로 도달 가능한 Δ의
> 상한은 ≈ **8.1–9.6 GiB**이고 "미확정"으로 등록된 대역 B(1.00–6.30 GiB)가 그 구간의
> **약 62 %** 를 흡수한다 ⇒ **F-b2가 H1에 불리한 판정을 낼 여지는 좁다.** 또한
> `gpu_mem_allocated_b` 대 `prefill_chunk_progress`는 **"곡선"이 아니며**(907959 L1의
> `==14` 146 스냅샷 중 128개가 종단 progress 56에 몰림), 스냅샷 발행률이 arm마다 **1.45×**
> 다르므로 **순간값의 arm 비교에 쓸 수 없다.**

> **RR-14 (승계 사슬의 강제, C10)** — RR-1 … RR-15는 §11.1의 구조상 **이 회차의 결과
> 문서까지만** 의무화돼 있다. ★**다음 회차 사전등록은 이 문서를 sha로 핀하고
> RR-1 … RR-15를 자신의 §11.1 목록에 명시적으로 편입해야 한다**(특히 RR-7: 감사 §13-D1(i)
> 미해결). 편입하지 않으면 그 회차는 이 조항을 위반한 것이다.

> **RR-15 (rev2 정정 — 기울기 축의 실현 가능 범위)** — 규칙층 판정서 §6-R-2(b)의
> *"T=1→3241 구간 … 예상 신호 4.04 GiB = ε의 4배"* 는 **실현이 보장된 값으로 인용할 수
> 없다.** ordinal 38(`4 seq / 3241 tok`)은 907959의 **4 boot 어디서도 in-flight로 관측되지
> 않았다**(§6-2-3). 검정력의 **하한**은 ordinal 34(`T = 2310`) 기준 **2.88 GiB = ε의 2.9배**,
> **상한**은 ordinal 38이 epoch 정의역에 들어올 때 **4.04 GiB = ε의 4배**다.
> ★결과 문서는 **실제로 정의역에 들어온 최대 `T`와 `|D_b|`를 boot별로 병기**해야 하며,
> 그것 없이 검정력을 주장할 수 없다. "10,125 축보다 검정력이 크다"는 결론은 두 경우 모두
> 유지된다.

---

## 12. 자기 적용 (게이트 #113) · 실행 전 공시

- **처방의 실현가능성**을 전부 적었다: F-a1은 로그 정규식 2개 + 파일 존재 + JSON 키 2개
  (**전부 항상 정의된다**), F-a2는 로그 1개 최댓값, F-b1은 로그 정규식 1개 + 이미 존재하는
  텔레메트리 필드 3개(`gpu_mem_peak_allocated_b`·`gpu_mem_peak_epoch`·
  `prefill_active_batch_size`)의 OLS, F-b2는 같은 필드, F-d는 텔레메트리 4키.
  **새 계측 0 · 새 엔진 코드 0 · 분석 스크립트 1개(결과 회차에서 작성).**
- ★**rev2의 정직한 공시 — 남은 판단과 남은 ∅ 위험**:
  (i) `ε = 1.00 GiB`와 `θ = 0.3825 MiB/token`은 **사전 판단**이다(RR-5). rev2는 판단을
  없애지 않고 **∅ 정의역과 미등록 재량**을 없앴다.
  (ii) F-b1의 정의역조차 **완전 보장은 아니다** — 텔레메트리가 형성 배치의 61–68 %만
  관측한다(§6-2-3, 907959 실측). 그래서 rev2는 **희소/∅ 처분(`|D_b| < 10`, `max T < 2000`,
  중첩 < 8)을 실행 전에 등록**했고, 그 처분은 `UNREALIZED`이지 H1의 실패가 아니다.
  (iii) 1차 판정 **F-a1은 이 위험에서 완전히 자유롭다** — 네 채널 모두 항상 정의된다.
- **반증 실패 공시(현시점)**: 수리가 **GPU에서** OOM을 없앤다는 것은 **아직 아무 증거도
  없다.** 지금 있는 것은 코드 사실 + CPU 측정 + 907959의 결정성뿐이다. 이 문서는 그 셋을
  근거로 **예보**를 등록할 뿐이며, 예보가 어긋나는 분지(§6-1 F-a1 거짓, §6-2 대역 C/C′)를
  **처분까지 포함해** 먼저 적었다.
- **rev1 → rev2에서 코드·하네스·테스트·manifest·라인 인용·예산은 한 바이트도 바뀌지
  않았다** — 규칙층 감사가 §1에서 전 항목을 독립 재검증해 일치를 공시했고, §4에서 반증
  실패 8건을 공시했다. rev2는 **문서 텍스트만** 고쳤다.
- **새 성능 판정 0건.** HE0 · 정책 순위 · stake #1 · 게이트 #13/#16 · Claim D 등급(미검증) ·
  Zamba2/triton 동결 — 전부 불변.
- **GPU 신규 지출 0** (이 문서 시점). `results/r2_eval/lambda0_prereg/**` 미접촉.
  R2 correctness 트랙 장부 **0.786 GPU-h 불변**.

---

### 관련 파일 (절대경로)

- 수리: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/multiplexing_mixin.py`
- 하네스: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness.sbatch`
- 채점기(무수정): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness_check.py`
- 테스트: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_worker_grad_guard.py` ·
  `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_mem_telemetry_symmetry.py`
- rev1 규칙층 판정서(`NO-GO`): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/rerun_prereg/VERDICT_rerun_rules_2026-09-14.md`
- 직전 회차 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907959/`
- 승계 문서: 같은 트랙 `newpair_prereg/` · `audit_907959_2026-09-13/`
- 크래시 지점(이번 회차 무수정):
  `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py:97`
