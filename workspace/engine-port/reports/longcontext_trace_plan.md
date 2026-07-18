# 실 trace를 **long-context로 전환**하는 문제 — 논의와 계획

발의: 사용자, 2026-07-17. 작성: 같은 날.
계기: *"motivation으로 attn과 mamba가 차이가 난다는 지점은 **long-sequence에서 심화**되는데, 실 trace에도 그런 데이터셋들이 존재할 것이고, 그에 따라 **실행의 범위가 달라질 수 있는** 가능성이 있다."*

관련: [CONSENSUS.md](CONSENSUS.md)(정본) · [research_arc.md §S-M](research_arc.md)(반증 지점·측정 환경·유효 경계) · [realtrace_findings_and_open_branches.md](realtrace_findings_and_open_branches.md)(얽힘 기전).
**이 문서는 계획이다. 여기의 어떤 수치도 아직 측정된 것이 아니다** — 확정 결론은 CONSENSUS에만 쓴다.

---

## 0. 한 줄

**동기는 정당하다**: 이 아크의 창립 동기(attn↔mamba 격차)는 **L 의존**인데, **서빙 반증은 전부 L<2k에서 났다**(ShareGPT mean 352·98%가 L<2000).
**그러나 이것은 layer-aware의 부활 경로가 아니다** — 재배분 lever(**Diff B**)는 **long-L에서 오히려 ≈1.0으로 닫힌다**(실측).
long-context가 실제로 흔들 수 있는 것은 **(1) 최적 static split의 위치, (2) 얽힘 기전의 병목(running-batch → KV), (3) HE0의 구조적 근거("두 regime의 최적이 충돌하지 않는다")** 이며,
**(3)이 동적 제어에 남은 유일한 upside**다.

---

## 1. 왜 이 질문이 정당한가 (동기의 정확한 형태)

| | |
|---|---|
| **창립 동기** | hybrid의 attn 층과 mamba 층은 연산 특성이 다르다 ⇒ 다르게 다뤄야 한다 |
| **그 차이가 켜지는 곳** | **Diff A(비용비) = attn/mamba**: `attn ~ L^1.68` vs `mamba ~ L^0.61`, **교차점 ≈3k tok**. L=2k서 **0.47×**(mamba가 오히려 비쌈) → L=8k **2.4×** → L=32k **10.1×** (`results/prefill_knee/diffA_vs_diffB_table.md`) |
| **그런데 서빙 판정이 난 곳** | ShareGPT **mean 352 · p50 204 · p95 1042** (98%가 L<2000) / synthetic **in2000·in3600** ⇒ **전부 교차점 아래 = Diff A가 닫혀 있거나 역전된 구간** |

⇒ ★**"attn과 mamba가 다르다"는 전제가 실제로 성립하는 구간을 이 아크는 *서빙으로 한 번도 밟지 않았다*.**
S0/S2/S9/S10의 서빙 반증은 각자의 환경에서 유효하지만(→ [research_arc.md §S-M.1](research_arc.md)), 그 환경은 **short-context**다.

---

## 2. ★먼저 적는 정직한 반론 (기대치를 미리 못박는다)

이 아크는 **결과를 본 뒤 서사를 맞추다가 4번 뒤집혔다.** 그래서 **측정 전에 예측을 등록**한다.

**(a) lever는 Diff A가 아니라 Diff B다 — 그리고 Diff B는 long-L에서 닫힌다.**
실측 Diff B(=SM 민감도비): **L≥8000서 0.96–1.04**, L=32000서도 **1.01–1.04**. 반면 **L=2000서 ≈1.35**.
⇒ **재배분할 여지는 L이 *짧을수록* 열린다.** long-context는 layer-aware 가설에 **불리한 방향**이다. **long-context 실험은 layer-aware의 재판이 아니다.**

**(b) (D) granularity 비용은 L과 무관하게 남는다.** S2에서 **TPOT 42→124ms**로 정량화된 sub-step 재분할 비용은 구조적이며, L이 길어져도 사라지지 않는다.

**(c) ★long-context 데이터셋은 대개 *출력이 짧다*.** LongBench-v2 로더 기본 `output_len = 10`(`sglang/benchmark/datasets/longbench_v2.py`).
우리 기전상 **최적 D_sm = max(모델 floor, 부하항[∝ λ×output_len])** 이므로, **출력이 짧으면 부하항이 죽고 최적은 prefill-heavy 극단에 고정**된다.
⇒ **long-doc QA 단독 trace는 HE0를 뒤집기는커녕 *강화*할 공산이 크다**(움직일 최적이 없음 = LO phase의 극단판).

### 사전 등록 예측 (측정 전)

| ID | 예측 | 근거 | 반증되면 |
|---|---|---|---|
| **H_L1** | long-context서 **최적 static이 prefill-heavy(d16 쪽)로 이동** | 부하항 ↓(출력 짧음) → 모델 floor 지배 | "최적 = 부하 함수" 모델(CONSENSUS §1-5)이 틀렸거나 KV 병목이 새 항을 추가 |
| **H_L2** | **PD-mux(agnostic) 이득 자체는 유지 또는 확대** | prefill이 길어져 overlap 기회↑ | PD 분리(§1-1)마저 L 의존 |
| **H_L3** | **layer-type 정책은 여전히 死**(더 확실히) | Diff B가 long-L서 ≈1.0 | (거의 불가) Diff B≈1.0인데 이득이 나면 = 내 lever 모델 자체가 틀림 |
| **H_L4** | ★**혼합(long-doc ↔ chat) trace에서만 두 regime의 최적이 *충돌*** | HE0의 구조적 근거가 "충돌 없음"이므로 | 충돌시켜도 동적이 지면 ⇒ **동적 트랙 완전 종결** |

★**H_L4가 이 계획의 진짜 목표다.** 나머지는 그 전제 조건을 까는 작업이다.

---

## 3. 무엇이 걸려 있나 (payoff)

| 질문 | long-context가 바꾸나 | 왜 |
|---|---|---|
| layer-type 정책 부활 | ❌ 거의 없음 | Diff B가 닫히는 방향(§2-a) |
| **최적 static split의 위치** | ✅ 예 | 부하항이 바뀜 ⇒ 실전 권고("peak decode 기준 decode-heavy")의 **범위 한정**이 필요해질 수 있음 |
| ★**얽힘 기전의 병목 재편** | ✅ **가능성 높음** | 현 기전은 `max_running_requests`(48) 포화 → admission 차단. **long-context서는 KV pool이 먼저 cap을 때린다** ⇒ admission이 **KV-bound**로 바뀌고 "decode 굶김 → TTFT 폭발" 경로의 상수가 달라짐. **기전 일반성의 시험** |
| ★**HE0 반전(동적의 upside)** | ⚠️ **조건부** | **혼합 trace에서만**. 단일 long-doc trace로는 오히려 강화(§2-c) |
| PD-mux 이득 크기 | ✅ 아마 확대 | prefill이 길수록 분할의 여지 |

---

## 4. 하드 블로커 (덮고 갈 수 없는 것)

### 4.1 ★정본 벤치 모델이 long-context를 못 한다

**정책 캠페인(E3)은 전부 `Zyphra/Zamba2-2.7B` 단독**인데 **`max_position_embeddings = 4096`**(로컬 config 확인). Zamba2-7B도 **4096**.
⇒ **모델 교체가 필수**이고, 이는 **정본 벤치의 모델 변경** = 기존 수치(d44 3.220 등)와 **직접 비교 불가** ⇒ **전 baseline 재측정** 필요.

| 후보(로컬 캐시 보유) | max ctx | 층 | 적합성 |
|---|---|---|---|
| ★**`ibm-granite/granite-4.0-h-micro-base`** | **131072** | 40 | **권장**. 진짜 hybrid(attn/mamba 층 분리) + long ctx. E1서 서빙 이력 있음 |
| `tiiuae/Falcon-H1-3B-Base` | **131072** | 32 | long ctx는 되나 **층마다 attn+mamba 병렬 = 단일 타입** ⇒ layer-type 논의엔 무의미(≡agnostic). **PD-mux 전용 대조군**으로는 유효 |
| `tiiuae/Falcon-H1-7B-Instruct` | 262144 | 44 | 위와 동일 성격, 더 큼 |
| `nvidia/Nemotron-H-8B-Base-8K` | **8192** | 52 | **8k까지만** — 중간 지점(교차점 3k는 넘음) 실험엔 사용 가능 |
| `Zyphra/Zamba2-2.7B` (현 정본) | **4096** | 54 | ❌ long-context 불가 |

### 4.2 메모리 · 얽힘 상수가 **동시에** 변한다 (confound 경보)

A100 80GB · `mem-fraction-static 0.82`에서 **ctx 32k × running 48**은 KV가 안 들어간다.
⇒ `--max-running-requests`를 낮추거나 ctx를 낮춰야 하는데, **`max_running_requests`는 얽힘 기전의 축**이다(CONSENSUS §1-4).
★**"변수는 하나씩"(§3-8) 규율상, running 값을 바꾼 채 정책을 비교하면 해석 불가.** ⇒ **running을 명시적 sweep 축으로 승격**하고, 정책 비교는 running 고정 하에서만.

### 4.3 ★goodput 정의가 long-context서 무너진다

현 SLO는 **TTFT ≤ 3s ∧ TPOT ≤ 60ms**. 그런데 **32k prefill 단독으로 3s를 넘길 수 있다**(E4 micro: L=32k attn 108ms/층 × 9 + mamba 15ms × 54 ≈ 1.8s **@full 108 SM**, 분할하면 그 이상).
⇒ **모든 정책이 goodput 0** = 신호 소멸. 게다가 우리는 **"임계 지시함수는 절벽을 피해 측정하라"**(§3-6)는 교훈을 이미 비싸게 샀다.
**해결 없이는 실험 자체가 무의미하므로 L0 전에 정한다**:

| 안 | 내용 | 평가 |
|---|---|---|
| **(i) 길이 정규화 SLO** ★권장 | `TTFT_SLO(L) = a + b·L` (예: 3s + 0.5ms/tok) | 길이 혼합 trace에 유일하게 공정. a·b를 **먼저 용량 측정으로 교정** |
| (ii) TTFT SLO 상향 | 일괄 30s 등 | 짧은 요청이 전부 통과 → 변별력 상실 |
| (iii) 분포 지표로 전환 | TTFT p95 / 용량(req/s) 직접 비교 | 절벽 회피 확실, 단 기존 goodput 계보와 단절 |

---

## 5. 데이터셋 옵션 (엔진 내장 확인 완료)

`sglang/benchmark/datasets/__init__.py`의 `DATASET_MAPPING`: `sharegpt` · `random` · `random-ids` · `generated-shared-prefix` · `longbench_v2` · `mooncake` · `mmmu` · `image` · `custom` · `openai` · `autobench` · `speed-bench`.

| 옵션 | 실 trace? | 장점 | 단점 / 주의 |
|---|---|---|---|
| **`random` (in 8k/16k/32k)** | ❌ | **즉시 가능**·완전 통제·E1 계보와 연속 | 실 trace 아님. **L1의 도구**(최적 이동 확인용) |
| ★**`longbench_v2`** (THUDM/LongBench-v2) | ✅ | 진짜 long-doc, **로더 내장**, 길이 분포 넓음 | **출력 기본 10 tok**(`--sharegpt-output-len`으로 강제 가능하나 그러면 "실 trace"가 반쪽) · **데이터 미보유**(`hf_cache/raw/longbench_cache/`는 **비어 있음**, 8K) · **`HF_HUB_OFFLINE=1`이라 사전 다운로드 필수** |
| **`mooncake`** | ✅✅ | 실 서빙 **arrival + 길이 trace** = 가장 현실적 | prefix 재사용 전제 — 우리는 **`--disable-radix-cache`** 중이라 해석 주의(또는 이 실험만 radix 허용 = 또 다른 변수) |
| **`generated-shared-prefix`** | ❌ | long prefix 통제 | 위와 동일한 radix 문제 |
| ★**혼합 (ShareGPT ↔ long-doc 교대)** | ✅ | ★**H_L4 전용** — regime 충돌을 *만드는* 유일한 벤치 | 하네스 신규 필요(E3-vary의 rate 교대를 **dataset 교대**로 확장) |

**권고**: `random`(통제) → `longbench_v2`(실 trace) → **혼합**(판정). `mooncake`는 radix 결정을 별도로 내린 뒤.

---

## 6. 단계 계획 (게이트 포함)

> 각 단계는 **다음 단계의 전제를 검증**한다. 게이트를 통과 못 하면 **거기서 멈춘다** — 이 아크가 비싸게 배운 것은 *전제를 안 밟고 나아간 대가*다.

| 단계 | 내용 | 게이트 (통과 못 하면 중단) |
|---|---|---|
| **L−1** | **SLO 정의 확정**(§4.3) + **용량 먼저 측정**(각 config의 최대 처리율) | 절벽 밖 rate 대역이 존재하는가 |
| **L0** | ★**모델 교체 baseline**: `granite-4.0-h-micro`로 **E3-vary(ShareGPT rate 3↔12)를 그대로 복제**, static sweep d16–d54, **n≥4** | **HE0가 모델을 넘어 재현되나**(decode-heavy static이 최선?). 재현 안 되면 → 그 자체가 큰 발견이고, 이후 long-context 해석의 기준이 통째로 바뀜 |
| **L1** | **`random` long**: in ∈ {2k, 8k, 32k} × out 96, **static sweep** (동적 없음) | **H_L1 검증**: 최적 static이 prefill-heavy로 이동하나? **Diff A가 열리는데 최적이 안 움직이면** = 비용비는 split 결정에 무관하다는 강한 증거 |
| **L2** | **`longbench_v2` 실 trace**: static sweep + **PD-mux vs fused** | **H_L2**: PD 분리 이득이 long-context서도 사나 |
| **L3** | ★**혼합 trace**(long-doc phase ↔ chat phase 교대): **best-static vs 동적(bind+GATE)**, **n≥4** | ★**H_L4 = HE0 반전 시나리오**. 여기서도 static이 이기면 ⇒ **동적 트랙 최종 종결**(짧은·긴·혼합 전부 소진) |
| ~~L4~~ | ~~layer-aware 재판~~ | **하지 않는다** — §2-a(Diff B가 long-L서 닫힘)가 사전에 배제. **하려면 짧은 L(≤2k)에서** 해야 하며 그건 이 문서가 아니라 **S3 정정의 열린 질문** |

**L3가 유일하게 "이길 수도 있는" 실험이다.** L0–L2는 전부 *전제 검증*이며, 결과가 어느 쪽이든 **CONSENSUS의 권고 범위를 한정**하는 값어치가 있다.

---

## 7. 측정 규율 (CONSENSUS §3 상속 + long-context 특유)

1. **n≥4 없이 정책 결론 금지.** long-context는 런당 시간이 길어 n을 깎고 싶어지는데, **그게 정확히 §2-4(underpowered)를 재발**시킨다. ⇒ n을 지키고 **prompts 수를 줄인다.**
2. **용량을 먼저 재고 절벽을 피한다**(§3-6). long-context는 rate 대역이 훨씬 좁다.
3. **`switch_count` + split 체류분포 + TTFT/ITL p50/95/99 병기.**
4. **라운드 duration은 합산**(`f921ae8`의 3× 버그 재발 방지).
5. **변수는 하나씩** — 특히 §4.2의 `max_running_requests`. **모델·ctx·running을 동시에 바꾸지 말 것**(L0가 존재하는 이유).
6. **길이 fingerprint를 매 런 기록**(워크로드 변동을 *주장*하지 말고 *검증*한다 — 이미 한 번 당했다).

---

## 8. 현재 상태 · 인접 작업

- **`knee2d_wide` (job 857371, array 0–7, 진행 중)**: L ∈ {256…32768} × B ∈ {1…16}, **L 256/512/1024만 완료**.
  ⚠️ 이건 **반대 방향(짧은 L) 조사**로, [research_arc.md §S3 정정](research_arc.md)의 열린 질문(*"L≈200–2000서 Diff B가 열리나"*)용이다. **이 문서의 계획과 혼동하지 말 것.**
  (그 sweep의 부산물로 드러난 것: 구 `knee2d` 격자는 **B축이 가짜**였다 — `chunked-prefill-size`가 토큰을 잘라 요청한 B=48이 실제로는 bs=1–10으로 관측됨.)
- **데이터**: `hf_cache/raw/longbench_cache/`는 **비어 있음**. `HF_HUB_OFFLINE=1`이므로 **L2 전에 다운로드 필요**(로그인 노드에서 별도 수행).
- **미결정**: radix-cache 정책(`mooncake`/shared-prefix를 쓸 것인가), SLO 정의(§4.3 (i) vs (iii)).
