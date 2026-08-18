# 워크로드 축 정의와 실측 앵커 — "길다/짧다"를 어느 축으로 말하는가

2026-08-18 · 메인 세션 · GPU 지출 **0**(기존 아티팩트 재집계) · 새 성능 판정 **0건** ·
등급 변경 **0건** · **정본 아님**(claims-auditor 미실행)

> **왜 이 문서가 필요한가**: 이 프로젝트와 문헌 양쪽에서 prefill/decode의 "길다·짧다"가
> **서로 다른 축**으로 쓰이고 있고, 그 축이 명시되지 않아 모순처럼 보이는 진술들이 생긴다
> (예: *"long-context는 prefill-heavy"* 와 *"요즘 워크로드는 decode가 길다"*).
> 이 문서는 축을 못 박고 **우리 자신의 실측 앵커**를 기록한다.

---

## 0. 한 줄

**우리 워크로드는 토큰 수로는 prefill이 1.44배 길고, 순차 forward 수로는 decode가 약 7배
길다. 둘 다 참이며 서로 다른 축이다. PD-mux가 작동하는 창(동거 7.7–20.3%)을 정하는 것은
토큰 수가 아니라 후자다.**

---

## 1. 네 개의 축

| 축 | 정의 | 왜 다른가 |
|---|---|---|
| ① **토큰 수** | `input_len` vs `output_len` 절댓값 | 가장 흔히 인용되나 시간과 직결되지 않음 |
| ② **순차 forward step 수** | ★**이 엔진**: prefill = `⌈n_layers / max(1, budget//L)⌉` (`n_layers`에서 포화) · decode = `output_len` | decode는 **autoregressive라 병렬화 불가** — 배치는 step의 *폭*만 넓히고 *개수*는 못 줄인다 |
| ③ **연산량(FLOPs)** | prefill `O(L²)`attn + `O(L·d²)` · decode `O(out·d²)` | `L`이 커지면 `L²` 항이 지배 |
| ④ **병목 성격** | prefill compute-bound · decode memory-BW-bound | 축 자체가 다름 (이 저장소의 roofline 측정과 대응, **8B 격자 한정 · 이식 금지**) |

★**①과 ②는 `L`에 따라 부호가 뒤집힌다.** 그래서 "long-context = prefill-heavy"와
"chat 워크로드 = decode가 길다"가 **동시에 참일 수 있다.**

### ★★축 ② 정정 (2026-08-18, venue-strategist 지적 → 메인 세션 코드 확인)

**초판이 축 ②를 `⌈L/chunk⌉`(토큰 청크 기반)로 적은 것은 이 엔진에서 틀렸다.**
`multiplexing_mixin.py:1128-1140`의 실제 회계는 **레이어 기반**이다:

```python
forward_count  = max(1, split_forward_token_budget // extend_num_tokens)   # budget=65536
next_split_index = min(split_index + forward_count, num_hidden_layers)
```

⇒ `forward_count`는 **한 forward가 전진하는 레이어 수**이고, prefill은
`num_hidden_layers`를 다 지날 때까지 쪼개진다:

| L | 층/forward | prefill forward (54층) | (40층) |
|---|---|---|---|
| **341**(우리 앵커) | 192 | **1** ✔(200요청→200step 관측과 일치) | 1 |
| 1,155 | 56 | 1 | 1 |
| 2,048 | 32 | 2 | 2 |
| 7,059 | 9 | 6 | 5 |
| 12,035 | 5 | 11 | 8 |
| 32,768 | 2 | **27** | **20** |
| ≥65,536 | 1 | **54**(포화) | **40**(포화) |

★★**구조적 귀결: 이 엔진에서 축 ②는 `output_len < n_layers`(≤40–54)일 때만 뒤집힌다.**
`L`을 아무리 키워도 요청당 prefill step은 40–54에서 멈춘다.
⇒ **vLLM(budget 8192)·Sarathi(2048)의 토큰-청크 축② 논의를 그대로 이식할 수 없다.**
gate #5(i) 설계의 **하드 제약**이다.
⇒ **이 프로젝트 문서에서는 축을 붙여 쓴다**: `decode-step-dominated` / `prefill-token-dominated`
같은 형태. 맨 "길다"는 금지.

---

## 2. 실측 앵커 (G16, 2026-08-17)

출처: `results/slo_sched/g16_blk1_d44_boot1_884336_HI.jsonl`(블록1 · d44 · HI)
· 체류 분포는 `G16_RESULTS_2026-08-17.json` `side_outputs.residency_fraction_by_boot`

**구성**: Zamba2-2.7B · ShareGPT · SGLang PD-mux(`--enable-pdmux`,
`--chunked-prefill-size -1`, `--disable-overlap-schedule`, `split_forward_token_budget=65536`)
· rate 12(HI) · 200 요청 / 36.3 s

| 양 | 값 |
|---|---|
| 입력 토큰 | 68,276 → 요청당 **341** |
| 출력 토큰 | 47,376 → 요청당 **237** |
| **축 ①** | **prefill이 decode의 1.44배** |
| decode forward | ≈ **1,450**회 (mean ITL 25.0 ms ⇒ 40 step/s × 36.3 s) |
| prefill forward | ≈ **200**회 (341 토큰 < budget 65536 ⇒ 요청당 1 chunk) |
| **축 ②** | ★**decode가 prefill의 약 7배** |
| 배치당 토큰(유도) | 32.7 |

**교차 검증**: 유도값 32.7이 claims-auditor가 독립 재구성한 split-state decode batch
**33.04 ± 0.86**과 일치한다(다른 경로·다른 코드).

### 2.1 그 결과 나온 체류 분포 (HI, arm당 4부팅 평균, 시간가중)

| arm | `decode_sms=0`(decode 비었음) | **= 명목 D**(분할 ON) | `=108`(decode 바쁨·미분할) |
|---|---|---|---|
| d16 | 26.0% | **7.7%** | 66.4% |
| d44 | 26.5% | **10.3%** | 63.3% |
| d74 | 26.4% | **20.3%** | 53.3% |

⇒ **PD-mux가 SM 배분상 실제로 개입하는 구간은 7.7–20.3%뿐**이고, 53–66%는 decode가
**미분할로 108 SM 전체**를 쓴다(= 도착과 도착 사이, prefill을 마친 요청들이 decode를 가는 시간).

⚠️**단, 그 구간이 "PD-mux를 안 쓴 것과 같다"는 SM 배분에 한한 진술**이다. `event_loop_pdmux`·
split-prefill 기계는 계속 돌므로 fused와 동일하지 않다.
⚠️**"창이 좁다" ≠ "이득이 작다"** — 좁은 창의 prefill 방해는 ITL **꼬리**에 비선형으로 실린다.
앞의 것은 측정됐고 뒤의 것은 이 데이터로 판정되지 않는다.

---

## 3. 왜 이게 용어 문제가 아닌가

노출 `w`(동거 시간 분율)는 **워크로드와 정책 양쪽의 함수**다 —
대략 `w ~ (요청당 prefill 시간 × 도착률) / (요청당 decode 시간 × 동시성)`이고,
prefill 시간은 `108−D`에 의존하므로 **arm도 `w`를 움직인다.**

### ★★귀속 정정 (2026-08-18, venue-strategist 지적)

**"7.7–20.3%"를 워크로드 성질로 인용하면 안 된다.** G16에서 워크로드·rate는
**전 arm 동일**(ShareGPT NP=200·ROUNDS=3)이고 변한 것은 **decode SM뿐**이다.
⇒ 그 스프레드는 **arm(정책) 성질이며 내생적**이다(§3.2: `D`↑ → prefill SM↓ → 동거 시간↑).

| 인용 목적 | 써야 할 값 |
|---|---|
| **워크로드 특성치**(문헌 대조 등) | ★**운영점 d44의 `10.27 ± 0.89%`** 단일값 |
| 정책 민감도 | "7.67–20.28%" — **arm 밴드**라고 명시 |
| 어느 경우든 | 분모 **`time / all bench span`** 병기 (denominator 4종이 `residency_scope_2026-08-17/tables_2026-08-17.txt`에 있음) |

⇒ **우리 `w(d44) ≈ 10.3%`는 "ShareGPT L≈341 × d44"의 성질**이지 PD-mux 자체의 성질이 아니다.
이 프로젝트의 모든 `w`·`gap_upper`·PD-mux 이득 진술이 **단일 워크로드 점에 묶여 있다**
(4모델 캠페인 전부 ShareGPT).

### 3.1 창을 넓히는 세 가지 방법 — 성질이 전혀 다르다

| 방법 | 기전 | 상태 |
|---|---|---|
| **D축**(arm) | prefill **용량**(`108−D`)을 깎아 동거를 늘림 | ★**자기파괴적 — G16이 반증**(§3.2) |
| **rate축**(부하) | prefill **작업량**을 늘림 | **미검증** — gate #16 **원문 문턱 판본**이 여기 있음 |
| **L축**(long-context) | prefill 시간을 `O(L²)`로 늘림 | **미실행** — `PROJECT_STATUS.md` gate #5(i) |
| (sticky) | 놀 때도 분할 유지 | SM 낭비 ⇒ **측정 도구 전용**(G17) |

### 3.2 D축이 자기파괴적이라는 실측 (G16 HI)

| arm | 노출 w | `M_ttft` | goodput |
|---|---|---|---|
| d16 | 0.077 | 2227.9 | 0.386 |
| **d44** | 0.103 | **1089.8** ← 최소 | **3.797** ← 최대 |
| d64 | 0.157 | 1189.9 | 3.775 |
| **d74** | **0.203** ← 최대 | 1875.6 | 3.131 |

★**노출이 가장 큰 d74가 TTFT 두 번째로 나쁘고 goodput도 중위권 아래**다. 성능은 노출과 함께
오르지 않고 **가운데서 꺾인다** — `D`를 키우면 창이 넓어지는 이유가 *prefill이 느려져서*이기
때문이다(prefill 처리량이 SM에 선형임은 result-analyst가 별도 확인:
`T_split × prefill_SM`이 SM 2.71× 구간에서 거의 상수).

⇒ **"창을 넓히면 PD-mux 이득이 커진다"는 D축에서는 반증된다.** rate축·L축은 창을
*prefill을 굶기지 않고* 넓히므로 **다른 명제이며 아직 열려 있다.**

---

## 4. 열린 항목 / 이 문서가 하지 않는 것

1. **외부 동향은 미조사** — "최근 워크로드가 어느 쪽으로 간다"는 문헌 주장이며 이 문서는
   그것을 판정하지 않는다. venue-strategist 조사 진행 중(2026-08-18 착수);
   결과는 이 문서 §5로 붙인다.
2. **축 ④(roofline)의 수치는 8B 격자 것**이며 Zamba2-2.7B G16에 **이식 금지**.
3. **`w`의 워크로드 의존성은 모형이 아니라 관찰**이다 — §3의 비례식은 직관용이고
   측정된 함수가 아니다.
4. **정본 반영 전 claims-auditor 필요.** 특히 §3.2의 "D축 반증"은 **arm 간 비교**이므로
   G16 결과 문서가 등재한 스코프 술어(`n_indep(placement)=3`, 혼합 지표 등)를 그대로 진다.
