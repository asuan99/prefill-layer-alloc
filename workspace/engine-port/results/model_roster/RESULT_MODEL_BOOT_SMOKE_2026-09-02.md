# RESULT — 모델 × 백엔드 × 모드 **지원 표** (jobs 900731 → 900752)

2026-09-02 · 2 job **0.36 GPU-hr**(900731 10:07 + 900752 11:44) ·
**성능 판정 0건 · arm 순위 0건 · 정본 변경 0건 · 등급 변경 0건**

사전등록: `model_boot_smoke.sbatch` 헤더(스모크, `PROJECT_STATUS` "방법론 게이트" #26) ·
결정: [`../cp_baseline/DECISION_PIECEWISE_OFF_2026-09-02.md`](../cp_baseline/DECISION_PIECEWISE_OFF_2026-09-02.md)

---

## 0. 한 줄

**piecewise CUDA graph를 끄면 세 모델 모두 장문(8,000 토큰)을 서빙한다.** 그리고
`granite-4.0-h-micro-base`·`Falcon-H1-3B-Base`는 **triton·flashinfer 양쪽에서** 돈다 —
게이트 #83의 백엔드 강제는 **`nemotron_h` 계열에만** 걸린다.

## 1. 지원 표 (job 900752, ctx 16384, 입력 8,000 토큰 × 8건)

| 모델 | flashinfer / `cp2048` | flashinfer / `d44` | **triton / `cp2048`** | **triton / `d44`** |
|---|---|---|---|---|
| `ibm-granite/granite-4.0-h-micro-base` | OK 8/8 | OK 8/8 | **OK 8/8** | **OK 8/8** |
| `tiiuae/Falcon-H1-3B-Base` | OK 8/8 | OK 8/8 | **OK 8/8** | **OK 8/8** |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` | OK 8/8 | OK 8/8 | **BOOT_FAIL** | **BOOT_FAIL** |

**유일한 부팅 거부**(자구 확인):
```
AssertionError: NemotronHForCausalLM does not support triton attention backend,
                as the first layer might not be an attention layer
```
⇒ 게이트 #83이 등재한 *"Nemotron-H + triton = 부팅 거부"* 가 **이 트리에서 자구까지 확인**됐다.
★그러나 그 강제는 **`nemotron_h` 계열에 국한**되며, `granitemoehybrid`·`falcon_h1`에는
**적용되지 않는다** — 로스터가 미확인으로 남겨 둔 칸이 이제 채워졌다.

참고 수치(★**성능 아님** — n=1, 8건, 워밍업 통제 없음, 반복 없음):
`granite` 1.03–1.58 req/s · `Falcon` 0.90–1.15 · `Nemotron-Nano` 0.50–0.73.

## 2. ★1차 표(job 900731)는 **무효**였다 — 계측 실패였지 모델 성질이 아니었다

1차는 12셀 중 **10셀을 "실패"로 표시**했는데, 그중 유효한 것은 **Nemotron+triton 2셀뿐**이었다.
나머지는 전부 **내 하네스 결함**이다. 게이트 #21(*"측정 실패를 게이트 실패로 라벨링 마라"*)을
표로 만들 뻔했으므로 원인을 남긴다.

**결함 A — 전 셀 서빙 0건.** `--dataset-name random`에 `--dataset-path`를 넘기지 않으면
샘플러가 ShareGPT를 **Hub에서 내려받으려 하고**(`sglang/benchmark/datasets/random.py:89`
`download_and_cache_hf_file`), 오프라인이라 `LocalEntryNotFoundError`로 죽는다.
모델과 **무관하게** 12셀 전부 같은 이유였다.

**결함 B — `cp2048`만 부팅 실패(Falcon·Nemotron).** chunked prefill 비호환이 **아니라**
piecewise 캡처 크래시:
```
Piecewise CUDA Graph failed with error:
  'typing.Union' object has no attribute '__module__' ...
To work around this error, add --disable-piecewise-cuda-graph to your launch command.
```
`cps > 0`이 piecewise 캡처를 켜므로 cp arm에서만 발화했고 `d44`(cps −1)는 통과했다 —
**F3 기전의 또 한 번의 관측**이다. ★그리고 이것은 **모델 × 환경 성질**이다:
같은 `cps 2048`이 **Zamba2에서는 정상 부팅**했다(job 900680,
`piecewise_cuda_graph_max_tokens=2048`). ⇒ 사용자 지시로
[piecewise를 전 arm에서 끄는 결정](../cp_baseline/DECISION_PIECEWISE_OFF_2026-09-02.md)을 등재했고,
그 결과가 §1의 표다.

★**900752의 비대칭 1건(기록)**: 이 job의 `d44` arm은 플래그를 **안 받았다**
(`disable_piecewise_cuda_graph=False`, cps −1이라 캡처는 어차피 비어 있음). 결정 문서 §2의
*"pdmux arm에도 명시적으로"* 는 제출 **이후** 반영됐으므로, 다음 실행부터 적용된다.

## 3. 후보 모델 실측 속성 (config 직독)

| | `granite-4.0-h-micro-base` | `Falcon-H1-3B-Base` | (참조) `Zamba2-2.7B` |
|---|---|---|---|
| 클래스 | `granitemoehybrid` | `falcon_h1` | `zamba2` |
| **MoE 여부** | **dense**(`num_local_experts=0`) | dense | dense |
| 층 수 / hidden | 40 / 2048 | 32 / 2560 | 54 / 2560 |
| **ctx** | **131,072** | **131,072** | **4,096** |
| 구조 분류(로스터 §3) | **(A) interleaved** | (B) 블록 내 병렬 | (A) interleaved |
| 층 패턴 | ★`layer_types`에 **명시**(mamba×5 + attention 반복) | 균일 | interleaved |
| 백엔드 | triton ✅ / flashinfer ✅ | triton ✅ / flashinfer ✅ | **triton만** |

★`granitemoehybrid`라는 **클래스명에도 불구하고 micro-base는 dense**다
(`num_experts_per_tok=0`, `num_local_experts=0`). 로스터 §4-2가 경고한 MoE 크기 정합 함정은
`granite-4.0-h-tiny` 쪽 이야기이고 **micro-base에는 해당하지 않는다.**

## 4. 권고 — `granite-4.0-h-micro-base` + `triton`

| 기준 | granite-micro | Falcon-H1-3B | Nemotron-Nano-9B |
|---|---|---|---|
| 크기(정본 Zamba2-2.7B 대비) | **3.19B (1.19×)** | 3.15B (1.17×) | 9B (3.3×) |
| ctx | 131K | 131K | 131K |
| **정본 백엔드(triton) 사용** | **✅** | ✅ | **❌ 불가** |
| 층 타입 축 정의됨 | **✅ (A)** | ❌ (B) | ✅ (A) |
| cp·pdmux 양 모드 서빙 | ✅ | ✅ | ✅(flashinfer만) |

⇒ **granite-micro가 유일하게 네 기준을 다 만족**한다. 특히 **triton 유지**는 정본 Zamba2 결과와
백엔드 축을 공유한다는 뜻이라, 로스터 §4-3이 경고한 *"정본 정책 결과와의 직접 연결 단절"*
비용을 **모델 크기 축 하나로 줄인다**(백엔드까지 함께 바뀌는 Nemotron 경로와 대비).

★Falcon-H1-3B는 CP-2에는 쓸 수 있으나(층 타입 축이 필요 없는 질문) 프로젝트의 다른 축과
어긋나므로 **2순위**로 둔다. Nemotron-Nano는 크기 3.3× + 백엔드 교체로 **3순위**.

## 5. ★그러나 모델 전환만으로는 원래 질문이 안 열린다

`cps` 기본값 8192를 요청 단위로 발화시키려면 **8192 토큰을 넘는 프롬프트**가 필요한데,
[ShareGPT 전수 센서스](../cp_baseline/RESULT_SHAREGPT_CENSUS_2026-09-02.md)가 이미 세었다:
**전 코퍼스에 56건**(0.1%). 4096 초과도 290건뿐이다.

⇒ **구속 조건은 모델이 아니라 데이터**다. ctx 131K를 얻어도 ShareGPT로는 `cps ≥ 4096`을
채울 수 없다. 남은 데이터 선택지는 셋:

| 안 | 상태 | 성격 |
|---|---|---|
| ShareGPT 장문 꼬리(floor ≤ 2048) | ✅ 준비됨(1,633건) | **실제 trace**, 단 `cps ≤ 2048`까지만 |
| `random` 합성(`--random-input-len`) | ✅ **이 스모크가 8,000 토큰으로 실증** | 길이 완전 통제, 실제 trace 아님 |
| LongBench-v2 | ❌ **오프라인 불가** — 로컬 캐시가 64K(메타데이터만) | 실제 장문, 다운로드 필요 |

## 6. 한계

1. **스모크다.** n=1, 8건, 반복·워밍업 통제 없음. §1의 처리량은 **어떤 판정의 근거도 아니다**.
2. ctx **16384**에서만 확인했다. 131K 부팅은 이 job이 재지 않았다(Nemotron-Nano는 job 896767에서
   131072 부팅 선례 있음, 다른 두 모델은 **미확인**).
3. `d44` 이외의 pdmux 분할(d16/d24/d54)은 **미확인**.
4. `granite`·`Falcon`의 **correctness는 재지 않았다** — 서빙이 됐다는 것과 출력이 옳다는 것은
   다르다. 정책 실험 전에 engine-porter의 correctness gate가 필요한지 판단이 선행돼야 한다.
