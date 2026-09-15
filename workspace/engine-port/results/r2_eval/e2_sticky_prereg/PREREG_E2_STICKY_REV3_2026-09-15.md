# 사전등록 — **E2 (sticky 분할 대조)** **rev3** (2026-09-15)

> **이력**: rev1 `NO-GO`(死因 F1 + 차단 B1–B6) → rev2 `NO-GO`(死因 **F1′·F2′** + 차단 **B1′–B8′**).
> GPU 지출 **0**. 판정서 `VERDICT_e2_rules_2026-09-15.md`(`145b786e…`) ·
> `VERDICT_e2_rules_rev2_2026-09-15.md`(`3386c807…`, 649행) §8이 rev3의 필수 조건이며 전부 GPU 0이다.
> ★**rev2의 死因 F2′는 rev2 저자(메인 세션)의 사실 오류였고, 이 문서가 그것을 정정한다** — 아래 §4-3.
> 이 문서는 **claims-auditor 재감사 대기**이며 `GO` 이전 **sbatch 금지**.
> 제출에는 **새 범위 한정 OVERRIDE + 사용자 승인**이 추가로 필요하다(기존 2건 소진).
> 기준 트리: HEAD `4d1a0ec1aed2a2963ce94b5bc49926680d7c5f30`.

## 0. rev2 → rev3 변경표 (재감사 색인)

| 지적 | 수리 | 위치 |
|---|---|---|
| **F1′** E-time 규약 미결정(두 읽기가 판정 3곳을 뒤집음) | **F1′-a 문안 채택**(다음 비-startup 스냅샷까지·무캡) + **F1′-b**(§4-2에 초 단위 분자/분모) + **F1′-c**(R4″ 전제 삭제 → **drain 축 민감도 프로브**로 재라벨) | §5·§4-2·§6 |
| **F2′** §4-3 E-iter 비재현 공시가 **거짓** | **정정 문안으로 전면 교체** + 셀프테스트를 **외부 독립 기대값**(판정서 §2 B3 열)으로 복원 + **E2C-8 → E2C-8′** | §4-3·§5·§9 |
| **B1′** R2 추론기 미등록 | 부트스트랩 **리터럴**(통계량·방법·`B`·RNG seed·결측) | §6 |
| **B2′** P13이 전례보다 약함 | **양 arm + byte-identical 리터럴 판정 + 빈 출력 ABORT** | §7 |
| **B3′** R4′ ↔ R3(ii) 10× 모순 | R4′ 밴드를 **≥ 95.0%** 로 눈금 통일 + idx0 표본수/분모 병기 | §6 |
| **B4′** R4″가 술어 아님 | `Y = max(s(OFF), s(ON))`, `d̄`, **`|d̄| > 3·Y`** 리터럴 | §6 |
| **B5′** 문턱 3×→1× 미공시 | ★**3× 복원**(판정서 원문). 조임은 채택하지 않는다 | §6 + 이 표 |
| **B6′** Q1 정의가 E-iter 표본을 배제 | ★**감사자 처방과 다른 방향으로 수리** — E-iter를 **왼쪽 스냅샷 decode-busy 게이트**로 못박아 네 추정량이 **같은 표본 조건**을 공유하게 한다(사유 §5-2) | §1·§5 |
| **B7′** 예산 상수가 실측보다 39.5 s/boot 낮음 | 실측 상수로 교체 ⇒ **1.803 GPU-h**(최악 **2.338**) + §8-2 분해 정정 | §8 |
| **B8′** `--time` vs P7 캡 불성립(13,159 s > 10,800 s) | **P14 전역 시계 가드** 신설 | §7 |
| 라벨 불일치 | **`LAMBDA_WITHIN_CI`** 로 일원화(`WITHIN_SPREAD` 전면 삭제) | 전문 |
| 등록 범위 누락(E2C-14) | **E2-α 채택 근거·E2-β 기각 사유를 본문에 복원** | §0-1 |
| ★신규(이 문서 산출) | **E2C-11의 idx0 수치가 규약 혼용**임을 정정 | §4-4·§9 |

### 0-1. ★설계 결정 — **E2-α를 채택한다** (E2C-14 수리, rev1 본문에서 복원)

대조 arm(sticky OFF)을 **같은 job 안에** 들고 간다.

| | **E2-α (채택)** | E2-β (기각) |
|---|---|---|
| OFF arm | **이 job 안에서 새로 측정** | job 908623 셀 재사용 |
| 노드·물리 GPU 축 | **불변** | 교락(908623 = gpu38) |
| 엔진 트리 | **구성상 동일**(한 번의 sync, 한 manifest) | 트리 변경 시 arm 비동일성 재발 |

**채택 사유 3개(전부 정본 등재 사실)**: ① **게이트 #233이 열려 있다** — 미등록 노드 축이
gpu38→43→40→41로 4회 이동. ② **job 908534(D-none)에서 귀속을 닫은 것이 "같은 job 안의 대조"였다**.
③ 감사자 §4-D가 `multiplexing_mixin.py` **한 파일 차이**를 교차 비교의 병기 의무로 등재했고,
E2-α는 그 조항을 발생시키지 않는다.
**E2-β는 버리지 않되 판정량이 아니다**: 908623 OFF 셀과의 교차 비교는 **GPU 0 부수 산출**(§6 R6).

---

## 1. 질문과 판정량

> **sticky 노브 하나를 ON으로 올리면 shape A의 실현 분할이 D44로 고정되는가, 그리고 그때 같은
> offered에서 achieved가 어떻게 변하는가.**

| 기호 | 판정량 | 등급 |
|---|---|---|
| **Q1** | **decode 실행의 D44(`stream_index == 2`) 점유** — §5의 **네 추정량 전부**. 네 추정량은 모두 **"decode-busy 스냅샷"** 을 표본 조건으로 공유한다(E-iter는 인접쌍의 **왼쪽** 스냅샷이 decode-busy). **"점유"는 추정량 이름 없이 인용하지 않는다**(E2C-2) | **1차** |
| **Q2** | 같은 offered·같은 seed의 **achieved = completed/duration**, **paired n=4** | 2차(범위 한정 기전 측정) |
| **Q3** | 실현 division 전수 히스토그램(idx 0–4) | 한 노브 검증 |
| **Q4** | `split_transition` 이벤트 수 | **제2 기전 계측** — 보고 전용 |
| **Q5** | 정본 술어 goodput 율 | 3차(병기 전용, 계단 함수로만 — E2C-6) |

---

## 2. 등록 arm · 셀 · 순서 (리터럴 — 재량 없음)

**arm 차이는 환경변수 한 개**: OFF = `PDMUX_STICKY_PARTITION` **unset** / ON = **`1`**.

| 셀 | shape | in/out | offered | `--num-prompts` | seeds | arm당 n | 908623 OFF 실측 duration |
|---|---|---|---|---|---|---|---|
| `a_r4` | A | 256/512 | 8.00 | 400 | s1,s2,s3,s4 | **4** | 131.01 s (s1) / 130.23 s (s2) |
| `a_r2` | A | 256/512 | 3.00 | 400 | s1,s2,s3,s4 | **4** | 142.45 s (s1) |
| `b_r3` | B | 8192/64 | 1.15 | 200 | s1,s2 | **2** | 286.85 s (s1) / 287.74 s (s2) |

**seed 리터럴**: **s1 = 4386 · s2 = 4162 · s3 = 251 · s4 = 2630** — λ0 쉬핑 규칙
`lambda0_plan.choose_seeds`의 **λ0 등록 N multiset에 대한 순위 1·2·3·4**(이 문서 재도출:
0.014603 / 0.015100 / 0.017025 / 0.018145). ★**공시**: 같은 규칙을 rev3 자신의 N multiset {200,400}에
적용하면 **1083 / 1262 / 4386 / 3519**가 된다. 쓰지 않는 이유 2개 — (i) s1/s2는 908623 OFF 셀이
**실제로 측정된 seed**이고 바꾸면 R6의 바이트 비교 가능성이 사라진다(★단 R6은 §6이 스스로
"판정 아님"으로 강등한 항목이다 — **E2C-12**), (ii) 재계산은 s1을 3위로 강등해 불일치를 만든다.
rev3 격자에서 네 seed의 `max|Ēbar−1|` = **0.001304 / 0.015100 / 0.017025 / 0.016533**, 전부
`EBAR_TOL = 0.025` 내. ⚠️**네 seed는 잡음이 아니라 실현 offered를 바꾼다**(E2C-13).

**실행 순서(리터럴, 인접쌍 · seed-major)** — **20 boot**:
```
a_r4:  s1/OFF s1/ON  s2/OFF s2/ON  s3/OFF s3/ON  s4/OFF s4/ON
a_r2:  s1/OFF s1/ON  s2/OFF s2/ON  s3/OFF s3/ON  s4/OFF s4/ON
b_r3:  s1/OFF s1/ON  s2/OFF s2/ON
```
⚠️ arm 순서가 항상 OFF→ON인 것은 **등록된 비대칭**(E2C-7).

---

## 3. 통제 요인 (908623에서 한 글자도 바꾸지 않는 것)

`MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` · `BACKEND=flashinfer` ·
`CFG=.../probes/pdmux_homog5.yml` · `CTX=16384` · `MEM_FRACTION=0.82` · `MAX_RUNNING=48` ·
`--disable-radix-cache` · `--disable-overlap-schedule` · `--chunked-prefill-size -1` ·
**cudagraph ON** · `PDMUX_R2_POLICY=fixed` · `PDMUX_R2_FIXED_DSM=44` · `--random-seed 1` ·
warmup 8요청 `--max-concurrency 1 --seed 7` · `--random-range-ratio 1.0` ·
`--dataset-name random --dataset-path <ShareGPT raw> --tokenize-prompt` · `--output-details
--output-file` · **셀당 1 boot** · 셀 간 `kill`→`sleep 5`→`pkill -9`→`sleep 10` · 포트 재확인.
(감사자 전항 대조 결과 `lambda0.sbatch`와 일치 — rev1·rev2에서 연속 승인.)

---

## 4. 한 노브 검증 — 노브는 하나, **기전은 둘**

### 4-1. 코드 독해 (쉬핑 트리, HEAD `4d1a0ec`)

`adjust_stream_groups`(`multiplexing_mixin.py:1163-1211`):
```python
if not self.running_batch.is_empty() and (
    self.split_prefill_batch or self.sticky_partition_enabled):      # :1169-1171
    if self.sticky_partition_enabled and self._sticky_fixed_idx is not None:
        stream_idx = self._sticky_fixed_idx                          # :1172-1179 (= idx 2)
    else: ... manual_divisions 임계 스캔 ...                          # :1180-1197
elif not self.running_batch.is_empty():
    set_current_stream_idx(self.real_sm_group_num - 1)               # :1199-1200 (= idx 4)
```
`_init_sticky_partition`(`:388-455`): `PDMUX_R2_POLICY=fixed` + `FixedPolicy(44)` ⇒
`_sticky_fixed_idx = 2`. 미정의 조합은 init에서 예외로 거부.

**(가) `manual_divisions` 드리프트** — `pdmux_homog5.yml` 세 행 임계가 **전부 0** ⇒ OFF arm의 스캔은
원리상 **idx 3 = D92**로 끝난다. 908623 실측 **11셀 전부 idx3 = 0**(v7 블록이 선점). **가정하지 않고
Q3로 측정**(R3).

**(나) ON에서 실현 인덱스가 `{0, 2}`로 닫힌다** — `:1169-1171`의 논리합이 항상 참이라 idx4 분지는
**도달 불가**, `_r2_decide_idx`(`:670-682` + `controller.py:92-102`)는 **항상 idx 2**. idx 0은 decode
배치가 빈 순간의 read-skew로만 들어온다(`CONSENSUS §3 항목120`). ⇒ **R4′의 근거.**

**(다) ★제2 기전 — 드레인 제거** — OFF는 prefill 경계마다 idx4↔idx2 두 번 전환하고 각 전환이
`prefill_stream.synchronize(); decode_stream.synchronize()`(v7 `:1344-1345`, elif `:1361-1362`).
ON은 v7 쪽이 `_tgt == stream_idx`라 **드레인이 발생하지 않는다** ⇒ **prefill 경계당 드레인 1쌍 제거**.
크기: `a_r4` OFF는 131.01 s에 `split_transition` **204건**(1.56 건/s). 엔진 주석(`:1270-1274`)이
*"2x synchronize at EVERY prefill boundary … ~200ms+ spike vs the true ~40ms TPOT"*.
**부호가 분할 효과와 반대** ⇒ **E2C-1 필수 병기**, R2는 3분지.

**(라) 부수 효과 없음이 확인된 축**: cudagraph 비대칭 없음(`cuda_graph_runner.py:806-817`) ·
admission · KV/mamba 예산 · telemetry 발화율 · prefill 축(두 arm 모두 prefill span에서 idx 2).

### 4-2. 908623 OFF arm 실측 — **분자/분모 병기**(B1·F1′-b 수리)

decode-busy = `event=="runtime_snapshot" ∧ phase!="startup" ∧ decode_running_batch_size>0`.
**E-time은 §5의 리터럴 규약(다음 비-startup 스냅샷까지·무캡)으로 산출**하고 초 단위로 병기한다.

| 셀 | busy n | **E-cnt** | **E-time** (idx2 s / 전체 s) | **E-iter** | **E-qcond** | idx0(개수) | idx3 | `split_transition` |
|---|---|---|---|---|---|---|---|---|
| `a_r0` | 2677 | 4.48 (120/2677) | **4.75** (24.95 / 525.2) | 4.50 | 17.65 (6/34) | 0.04% (1/2677) | **0** | 604 |
| **`a_r2`** | 610 | **8.03** (49/610) | **10.56** (22.59 / 213.9) | **8.11** | 17.65 (18/102) | 0.00% (0/610) | **0** | 301 |
| **`a_r4`** | 555 | **8.65** (48/555) | **13.64** (25.55 / 187.3) | **8.76** | 17.20 (43/250) | 0.36% (2/555) | **0** | 204 |
| `a_r4_s2` | 557 | 8.26 (46/557) | 13.40 (25.07 / 187.1) | 8.36 | **17.00 (43/253)** | 0.00% (0/557) | **0** | 169 |
| `b_r0` | 313 | 43.13 (135/313) | 66.18 (110.6 / 167.1) | 45.58 | 80.00 (36/45) | 4.15% (13/313) | **0** | 43 |
| **`b_r3`** | 441 | **90.70** (400/441) | **97.63** (163.8 / 167.8) | **92.29** | **100.00 (395/395)** | 0.23% (1/441) | **0** | 100 |
| `b_r3_s2` | 441 | 90.70 (400/441) | 97.53 (163.3 / 167.4) | 92.13 | 100.00 (392/392) | 0.00% (0/441) | **0** | 100 |

`P(prefill_active | idx2)` = a_r0 64.2 · a_r2 49.0 · a_r4 66.7 · a_r4_s2 76.1 · b_\* 100.0
(분모 120 / 49 / 48 / 46 / 135 / 400 / 400).
⚠️**스냅샷 간격 중앙값 ≈2 ms는 decode-busy 표본의 격자가 아니다**(E2C-9) — decode-busy 한정 인접
간격 중앙값은 `a_r4` **idx2 532 ms / idx4 208 ms**, `a_r2` 413 / 265 ms, `b_r3` 333 / 209 ms.

### 4-3. ★**정정** — rev2 §4-3의 E-iter 비재현 공시는 **거짓이었다** (F2′ 수리)

rev2는 *"판정서 §2 B3의 E-iter 열이 4규약 전부에서 shape B 재현 실패(b_r0 42.1–43.4 vs 45.6,
b_r3 90.8–90.9 vs 92.3)"* 라고 등록했다. **이 문장은 거짓이다.** rev2의 4규약은 *"수열을 먼저
decode-busy로 필터한 뒤 인접쌍을 만드는"* 형태만 담았고, **"인접쌍을 만든 뒤 왼쪽 스냅샷의
decode-busy로 게이트하는"** 규약을 **실행하지 않았다**. 그 규약이 판정서 열을 **7/7 재현한다**:

> **규약**: 인접한 두 비-startup 스냅샷 쌍 중 `decode_iterations` 차분 **> 0**인 것만, **왼쪽**
> 스냅샷이 **decode-busy**일 것을 요구, **왼쪽** 스냅샷의 `stream_index`에 귀속, 가중 = 차분값.
> **값**: **4.50 / 8.11 / 8.76 / 8.36 / 45.58 / 92.29 / 92.13**
> (이 문서 독립 재산출 = 판정서 §2 B3 열 4.5 / 8.1 / 8.8 / 8.4 / 45.6 / 92.3 / 92.1, **7/7 일치**).

⇒ rev2가 이 거짓 위에 세웠던 두 가지를 **철회**한다: (i) 셀프테스트의 E-iter 기대값을 자기 산출로
바꾼 것 → **외부 독립 기대값(판정서 열)으로 복원**(§5), (ii) **E2C-8을 E2C-8′로 교체**(§9).

**같은 계열에서 여전히 참인 것**: `VERDICT_result_lambda0_908623` §4-A **2열(시간 가중)** 은
재현되지 않는다 — 이 문서 14변형 + 재감사 **160변형** 탐색에서 최소 최대오차 **1.44pp**
(목표 `b_r0 46.1` / `b_r3 91.9`). **그 열은 규약이 복원될 때까지 인용 금지**(E2C-8′).

### 4-4. ★**정정 2** — 판정서 E2C-11의 idx0 수치는 **규약 혼용**이다 (이 문서 산출)

판정서 E2C-11은 *"idx0에 귀속되는 decode-iteration 가중은 `a_r2` 0.82% · `a_r4` 1.32% ·
`b_r0` 9.63%, 그래서 `b_r0`가 남았다면 E-iter ON 예보가 **90.37%**(문턱 위 0.37pp)"* 라고 적었다.
이 세 값은 **왼쪽 busy 게이트가 없는** 산출이고, 판정서 자신이 E2C-8′에서 **게이트 있는** 규약을
E-iter로 확정했다. 같은 원자료를 **등록 규약(게이트 L)** 으로 재산출하면:

| | `a_r2` | `a_r4` | `b_r0` |
|---|---|---|---|
| **등록 규약(게이트 L)** idx0 가중 | **0.00%** | **0.26%** | **4.47%** |
| 게이트 없음(판정서 E2C-11) | 0.82% | 1.32% | 9.63% |
| ⇒ `b_r0` ON E-iter 예보 | — | — | **95.53%**(여유 5.5pp) vs 판정서 90.37%(0.37pp) |

⇒ **E2C-11의 판정 방향(= `b_r0` 제외가 R1의 여유를 넓혔다)은 유지**하되 **수치는 위 표로 정정**해
승계한다(§9). 이것은 E2C-8′을 판정서 자신에게 적용한 첫 사례다.

---

## 5. 추정량 명세 (리터럴)

입력: 셀·arm·seed당 telemetry 한 개. 대상: `event=="runtime_snapshot" ∧ phase!="startup"`.
D44 = `stream_index == 2`; `(prefill_sms, decode_sms)`도 기록하고 **불일치 시 ABORT**.

| 이름 | 표본 · 가중 (리터럴) |
|---|---|
| **E-cnt** | 표본 = decode-busy 스냅샷. 균등(개수). |
| **E-time** | 표본 = decode-busy 스냅샷. 가중 `w_i` = **그 스냅샷의 `timestamp_monotonic_s`부터 파일 내 다음 `event=="runtime_snapshot" ∧ phase!="startup"` 레코드의 `timestamp_monotonic_s`까지의 간격**(그 다음 레코드의 decode-busy 여부 **무관**), **무캡**. **파일 내 마지막 비-startup 스냅샷은 가중이 정의되지 않으므로 제외.** E-time = (idx2 표본 `w` 합) / (전 decode-busy 표본 `w` 합). |
| **E-iter** | 표본 = 인접한 두 비-startup 스냅샷 쌍 중 `decode_iterations` 차분 **> 0** ∧ **왼쪽 스냅샷이 decode-busy**인 것. **왼쪽** 스냅샷의 `stream_index`에 귀속. 가중 = 차분값. |
| **E-qcond** | 표본 = decode-busy ∧ `prefill_queue_depth > 0`. 균등. |
| **E-pact** | 표본 = decode-busy ∧ `prefill_active_batch_size > 0`. 균등. **구성 항등식 가드**(≠100%면 계측 결함 ⇒ ABORT) |

### 5-1. R1은 **E-cnt·E-time·E-iter·E-qcond 넷 모두**에서 동시 성립해야 한다
쉬핑 코드 `:778-800`이 count 격자를 *"NOT comparable across the flag … TIME-WEIGHTED statistics …
are unaffected"* 로 경고하고, `CONSENSUS §3 항목120`이 시간 부과 쪽을 경고한다 ⇒ **양쪽 경고를
동시에 만족하는 유일한 길**이다. ⚠️단 이 강화가 통과 가능한 것은 `b_r0`가 셀 집합에서 빠졌기
때문이며 그것은 **예산 사유의 부수 효과**다(E2C-11, 수치는 §4-4로 정정).

### 5-2. ★B6′를 감사자 처방과 **다른 방향으로** 수리한 사유 (공시)
판정서 B6′는 *"Q1 정의를 추정량별 표본 집합과 일치시켜라(E-iter는 전 비-startup 인접쌍)"* 를
처방했다. 이 문서는 **반대로 E-iter를 왼쪽 decode-busy 게이트로 못박아** 네 추정량이 **같은 표본
조건**을 공유하게 했다. 사유 2개: (i) 게이트 있는 규약이 **판정서 자신의 E-iter 열을 7/7 재현**하는
바로 그 규약이다(§4-3) — 게이트를 빼면 외부 기대값과 어긋난다. (ii) 게이트를 빼면 decode 유휴
구간의 iteration이 idx0에 실려 **`b_r0`에서 5.39%의 비-busy 가중**이 판정에 들어온다(판정서 B6′ 본문의
수치). ⇒ 정의를 느슨히 하는 대신 **추정량을 조인다**.

### 5-3. 필수 셀프테스트 (제출 전, GPU 0)
다섯 추정량 + idx0/idx3/`split_transition`/busy n을 908623 아카이브에 적용해 **§4-2 표를 7/7 셀
바이트 일치**로 재현. 불일치 = **제출 차단**. **기대값의 출처(전부 외부 독립)**:
E-cnt·E-qcond = `VERDICT_result_lambda0_908623` §4-A 1·3열 · **E-iter = `VERDICT_e2_rules_2026-09-15`
§2 B3 열** · E-time·idx0/idx3·`P(pact|idx2)` = `VERDICT_e2_rules_rev2_2026-09-15` §2 F1′ (B)열·§3.
★**어느 칸도 이 사전등록 자신의 산출을 기대값으로 삼지 않는다**(F2′ 수리, 교훈 9).

---

## 6. 결정 규칙 (리터럴)

| 규칙 | 리터럴 내용 | 결과 문안 |
|---|---|---|
| **R1**(1차) | `a_r4`·`a_r2` **전 seed**에서 ON의 **E-cnt·E-time·E-iter·E-qcond 네 값 모두 > 0.90** ∧ 같은 셀·seed OFF의 네 값 **모두 < 0.20** | 참 ⇒ `STICKY_REALIZES_A`; 거짓 ⇒ `STICKY_DOES_NOT_REALIZE_A`(P1 배선과 측정을 분리 기록) |
| **R2**(2차, 3분지) | `a_r4`·`a_r2` 각각: `dᵢ = achieved(ON,sᵢ) − achieved(OFF,sᵢ)`(i=1..4)의 **평균**에 대한 **percentile 부트스트랩**, **`B = 100000`**, **`numpy.random.default_rng(20260915)`**, **2.5/97.5 분위**. n<4면 CI를 내지 않고 **UNRESOLVED**(게이트 #3) | CI < 0 ⇒ `LAMBDA_MOVES_DOWN`; CI ∋ 0 ⇒ **`LAMBDA_WITHIN_CI`**; CI > 0 ⇒ `LAMBDA_MOVES_UP`. ★세 분지 전부에 **Q4와 E2C-1 병기 의무**. ★`LAMBDA_WITHIN_CI`·`LAMBDA_MOVES_UP`에서 *"λ0R-8(iii) 교락 우려가 약화된다"* **금지** |
| **R3**(한 노브) | (i) **OFF arm** 전 셀 idx3 점유 **= 0** (ii) **두 arm** 전 셀 idx0 점유 **≤ 5.0%**(초과 ⇒ 그 seed 짝 **UNRESOLVED**), **관측 idx0 표본수/분모 병기 의무** (iii) `split_transition` 수는 **보고 전용**이며 판정에 쓰지 않는다 — 단 **ON > OFF인 셀이 하나라도 있으면** §4-1(다)의 코드 독해가 깨진 것이므로 **E2C-1을 강한 형태로 승계** | — |
| **R4**(분할 축 음성대조) | `b_r3` **두 arm의 E-qcond 차 ≤ 0.5pp**(OFF 실측 100.00% = 395/395; prefill 대기 중 decode-busy 표본은 두 arm 모두 idx 2 ⇒ **처치가 움직일 수 없는 양**) | 거짓 ⇒ arm 차이가 sticky 아닌 무엇(순서·드리프트·read-skew)에서도 온다 ⇒ **R1/R2 전체 caveat** |
| **R4′**(양성 예보) | `b_r3` ON의 **E-cnt ≥ 95.0%**(= idx0 ≤ 5.0%, **R3(ii)와 같은 눈금**). 미만이면 **노브 배선 또는 계측 가정이 깨진 것** | `ABORT_STICKY_STATE_MISMATCH` 층. idx0 표본수/분모 병기(441 격자에서 1표본 = 0.227pp) |
| **R4″**(**drain 축 민감도 프로브** — 음성대조 아님) | `s(OFF) = \|achieved(OFF,s1) − achieved(OFF,s2)\|`, `s(ON) = \|achieved(ON,s1) − achieved(ON,s2)\|`, **`Y = max(s(OFF), s(ON))`**, **`d̄ = mean(d₁, d₂)`**. **`\|d̄\| > 3·Y` ⇒ `B_RESULT_AXIS_MOVES`**, 아니면 `B_RESULT_AXIS_WITHIN` | 어느 분지든 **n=2 · CI 없음 · 검출력 미등록** 병기. `B_RESULT_AXIS_WITHIN`을 *"차이가 없다"* 로 읽는 것 금지. ★**전제 정정**: `b_r3`의 OFF→ON 이동폭은 규약에 따라 **+2.4pp(E-time) / +9.1pp(E-cnt)** 로 갈리므로 이 셀은 **분할 축의 음성대조가 아니다** |
| **R5**(재현) | `a_r4`·`a_r2`에서 **seed 간 E-cnt 산포가 arm별로 ≤ 5pp** | 거짓 ⇒ `SEED_SPREAD_FAILS` ⇒ 그 셀의 R1/R2 **UNRESOLVED** |
| **R6**(부수) | 908623 OFF 셀과의 교차 비교 | **판정 아님** — 보고만, **노드 축 교락 병기 의무** |

**문턱 공시(B5′ 수리)**: R4″의 문턱은 **판정서 원문의 3×를 복원**했다(rev2는 공시 없이 1×로
조였고, 그 조임은 908623 OFF 값에서 트리거를 0.006496 → 0.002165 req/s로 바꾼다 — 교훈 251).

**UNRESOLVED 규약**: boot 실패·CAPPED(P7)·유효성 위반(P9–P12)·예산 절단(P14)·셀 누락은 **그 seed의
OFF/ON 짝을 통째로** UNRESOLVED로 만들고 짧아진 집합으로 재라벨하지 않는다(D23 승계).

---

## 7. 실패 닫힘 게이트

- **P1 양성 대조**: ON boot 로그에 `PD-mux sticky partition ENABLED ... (fixed target index=2)`가
  **있어야**, OFF 로그에는 **없어야** 한다 ⇒ 위반 시 `ABORT_STICKY_STATE_MISMATCH`(게이트 #237).
- **P2**: ON 로그의 `fixed target index` ≠ 2 ⇒ abort.
- **P3**: 두 arm의 `max_mamba_cache_size` / `max_total_num_tokens` 불일치 ⇒ abort.
- **P4**: manifest ≥ 24 엔트리 + `nemotron_h.py`; 결정경로 파일·**이 문서(rev3)** 의 sha256을 결정
  이전에 `REGISTRATION_SHA256.txt`에 기록. 두 arm은 **한 번의 sync·같은 manifest**.
- **P5 제출 전 커밋**(λ0R-10).
- **P6** boot 실패 2연속 ⇒ abort.
- **P7 셀 벽시계 캡**: `T_cap = 3 × (908623 OFF 실측 duration)` ⇒ 초과 시 **CAPPED** ⇒ 그 seed 짝
  UNRESOLVED(`max()` 합산 금지 — 게이트 #7).
- **P8** arm 라벨 무결성(파일명 ↔ JSON) 불일치 ⇒ abort.
- **P9–P12 bench 유효성**(셀별): `errors == 0` · `completed == num_prompts` · 입출력 길이 정확 ·
  `random_range_ratio == 1.0`. 위반 ⇒ 그 seed 짝 **UNRESOLVED**.
- **P13 correctness 대조**(B2′ 수리 — 전례와 같은 강도): **두 arm 모두**의 첫 boot에서 bench **전**에
  고정 프롬프트 6개를 `--max-concurrency 1 --temperature 0`으로 받아 **텍스트와 토큰 ID**를 기록한다.
  **리터럴 판정**: 두 arm의 6개 출력이 **byte-identical** ⇒ `STICKY_CORRECTNESS_BYTE_IDENTICAL`;
  아니면 `STICKY_CORRECTNESS_DIVERGED`를 라벨로 남기고 **그 자체로 abort하지 않되** 결과 문서가
  *"(Nemotron-9B, `(64,44)`, flashinfer, cudagraph ON)에서 sticky ON의 출력은 OFF와 일치하지 않았다"*
  를 **필수 병기**한다. 어느 arm에서든 6개 중 하나라도 **빈 문자열** ⇒ `ABORT_CORRECTNESS`.
  (부하 없는 순차 프로브로 한정 — 전례 job 872800과 같은 조건. 부하 중 출력 비교는 처방하지 않는다.)
- **P14 전역 시계 가드**(B8′ 수리): 각 boot 시작 전 경과시간을 확인해 남은 예산이
  `T_cap(다음 셀) + 200 s` 미만이면 **그 seed 짝부터 끝까지 실행하지 않고 `UNRESOLVED_BUDGET`** 으로
  라벨한 뒤 정상 종료한다. 부분 실행 집합으로 **재라벨하지 않는다**(D23).

---

## 8. 예산 (실측 상수 — B7′ 수리, 게이트 #113)

### 8-1. 계산
per-boot 부대비용은 **908623 아티팩트 mtime 실측**을 쓴다(rev2의 106 s/69 s는 39.5 s/37.5 s 과소였다):
**A = 145.5 s · B = 106.5 s**(teardown+boot+warmup+후처리), preflight **≥ 400 s**.

| 셀 | boot | per-boot | 소계 |
|---|---|---|---|
| `a_r4` | 8 | 131.0 + 145.5 = 276.5 s | 2212.0 s |
| `a_r2` | 8 | 142.5 + 145.5 = 288.0 s | 2304.0 s |
| `b_r3` | 4 | 286.9 + 106.5 = 393.4 s | 1573.6 s |
| preflight | — | — | 400 s |
| **등록 보통** | **20 boot** | | **6489.6 s = 1.803 GPU-h** |
| **최악 코너**(ON에서 A bench ×2.5 · B ×1.5) | | | **8417.5 s = 2.338 GPU-h** |
| `--time` | | | **03:00:00** (10,800 s; **P14가 P7 캡 합계 13,159 s를 먼저 차단**) |

### 8-2. 분해 정정 (B7′ 부수 지적)
rev1(1.421) → rev2/rev3 셀 집합의 참 분해는 **−0.5408(`a_r0`·`b_r0` 제거) + 0.6775(`a_r4` +2 seed ·
`a_r2` +3 seed) = +0.1367** ⇒ 1.558(rev2 상수) → **1.803(실측 상수)**. rev2 §8-2가 적은
*"b_r3 2 seed 유지 +0.20"* 은 **기준선 혼동**이었다(rev1도 이미 2 seed) — 철회한다.
`b_r3` 2 seed를 유지하는 **사유는 그대로 유효**: 판정서가 신설한 R4″가 **B측 achieved 산포**를
요구하는데 seed 1개면 그것을 908623(다른 노드·다른 job)에서 빌려와야 하고, 그것은 λ0R-1·E2C-7이
금지한 형태다.
비교: λ0 job 908623 실지출 **1.391 GPU-h**.

---

## 9. 승계 — 인용 금지 / 필수 병기

**E2C-1 … E2C-7**(rev1 판정서 §9) · **E2C-8′ · E2C-9 · E2C-10 · E2C-11(수치 정정) · E2C-12 ·
E2C-13 · E2C-14**(rev2 판정서 §9)를 **문자 그대로 승계**한다. 요지:

- **E2C-1**(최대 위험) — 노브는 하나, **기전은 둘**(decode SM 108→44 **+** 드레인 1쌍 제거, 부호 반대).
  ⇒ achieved 변화를 "분할 민감도"로 단독 귀속 금지, **변화 없음을 "분할 무관"으로 읽는 것도 금지**.
- **E2C-2** — E-cnt는 시간 점유가 아니다(`_dual_worker_sync`의 32분의 1 count-subsample). **"D44 점유
  X%"는 추정량 이름 없이 인용 금지.**
- **E2C-3**(인용 금지) — *"ON에서 D44 ≈100% ⇒ sticky 효과 실증"*. ON의 인덱스는 엔진이 {0,2}로 강제.
  R1의 ON 절은 **양성 대조**. 정본 전례(`CONSENSUS §3 항목27`)는 **Ha8·`(92,16)`** 이고 이번과 다르다.
- **E2C-4**(인용 금지) — E-pact 100%로 §1-26(B)를 확증하지 마라(양 arm 항등식). 경험량은
  `P(prefill_active | idx2)` = shape A **49.0–76.1%**.
- **E2C-5** — **이 회차는 λ0R-8(iii)을 해소하지 않는다**(두 arm의 prefill 축 동일).
- **E2C-6** — `a_r2`의 Q5는 **절벽 위**(OFF TTFT p95 1.806 s vs SLO 3.0 s) ⇒ **계단 함수로만**.
- **E2C-7** — 실행 순서 OFF→ON의 등록된 비대칭. `LAMBDA_WITHIN_CI`를 *"차이가 없다"* 로 읽기 금지.
- **E2C-8′** — 판정서가 발표한 추정량 열은 **규약을 함께 적지 않는 한 인용 불가**.
  **참인 사례**: λ0 §4-A **2열(시간 가중)** — 두 회차 합계 **174변형**에서 재현 실패(최소 최대오차
  1.44pp) ⇒ **인용 금지**. **거짓 사례(철회)**: `VERDICT_e2_rules_2026-09-15` §2 B3의 E-iter 열은
  **재현된다**(규약·값 §4-3). ⇒ *"재현되지 않는다"를 규약 전수 탐색 없이 쓰지 마라.*
- **E2C-9** — "스냅샷 간격 중앙값 ≈2 ms"를 추정량 표 옆에 단독 인용 금지(decode-busy 격자는 ≈331 ms).
- **E2C-10** — 이 회차의 achieved는 **`--max-running-requests 48`이 구속한 처리율**이다(λ0R-3).
  paired 차분은 유효하나 **수준값을 "시스템 용량"으로 인용 금지**.
- **E2C-11**(★수치 정정, §4-4) — R1이 네 추정량 동시 성립으로도 통과 가능한 것은 **`b_r0`가 예산
  사유로 빠졌기 때문**이다. ⇒ *"네 추정량 동시 성립"을 추정량 집합의 강건성 증거로 인용 금지.*
  ★단 판정서가 적은 idx0 가중(0.82 / 1.32 / 9.63%)과 `b_r0` ON 예보 90.37%는 **게이트 없는 규약**의
  값이며, **등록 규약(왼쪽 busy 게이트)에서는 0.00 / 0.26 / 4.47%**, `b_r0` ON 예보 **95.53%**다.
- **E2C-12** — seed1/seed2 유지의 첫째 사유(R6 바이트 비교 가능성)는 §6이 스스로 "판정 아님"으로
  강등한 항목을 지키기 위한 것이다(선택 자체는 적법·결과 의존성 0).
- **E2C-13** — **네 seed는 실현 offered를 바꾼다**: `Ēbar(N=400)`로 `a_r2`의 실현 offered =
  **2.996 / 2.955 / 2.950 / 2.951 req/s**(1.5% 폭). `a_r2`는 λ\*(A)=3.05 바로 아래 **무릎**이라 이 폭이
  achieved에 직접 실린다. **paired 차분(R2)에서는 seed 안에서 상쇄되지만** seed 간 비교(R5·R4″)와
  수준값 인용에는 실린다.
- **E2C-14** — E2-α의 채택 근거는 **§0-1에 복원**됐다(sha로 봉인되는 것은 rev3 자신이다).

**기존 승계 전부 유효**: λ0R-1…λ0R-10 · λ5C-1…8 · NPC-I(shape A 반증) · N-7·N-8·N-9·N-11 ·
게이트 #13/#16 "닫았다" 금지 · C2 인용정지 (a)(b) · HE0 · layer-type 死 · 정책 순위 · stake #1.

**Q2가 정책 주장이 아닌 이유**: 이 격자는 **정지 rate 사다리**이므로 **게이트 #2**에 의해 정책 비교
벤치가 아니다. n=4 + paired CI는 Q2를 *"이 셀에서 실현 분할을 D44로 고정하면 achieved가 X% 변한다"*
까지만 올린다. **정책 서열 주장 금지.**

---

## 10. 이 회차가 하지 못하는 것 (2회 연속 감사 승인 — 불변)

1. **게이트 #6(λ\*_SLO)을 닫지 않는다** — E1a/E1b가 한다.
2. **P2 블로커 ②(W4 λ\* 실측 부재)를 해소하지 않는다**(②③은 같은 블로커의 두 이름).
3. **Claim D/E 등급 불변**(둘 다 미검증). **새 성능 판정 0건.**
4. **shape A와 B를 서로 비교하지 않는다.**
5. **λ0 라벨을 재계산하지 않는다**(908623 아티팩트는 읽기 전용).
6. **true-dual(B4)에 대해 아무것도 말하지 않는다**(R2C-2 미충족).
7. **E3(48-구속)을 포함하지 않는다.**

## 11. 제출 선행조건 (순서대로 — 전부 미충족)

1. **rev3 재감사**(claims-auditor) → `GO` 또는 `GO-with-caveats`.
2. `GO` 이후 **하네스 작성**(`e2_sticky.sbatch` + `e2_realized_mix.py`) + §5-3 셀프테스트 통과(GPU 0).
3. **하네스 층 감사**(2단, 교훈 34).
4. **새 범위 한정 OVERRIDE + 사용자 승인**(기존 2건 소진; presubmit M4R·TC1 차단 2건 유효).
5. **커밋**(P5) → 그 다음에만 `sbatch`.
