# 사전등록 — **E2 (sticky 분할 대조)** **rev2** (2026-09-15)

> **이력**: rev1 **`NO-GO`**(GPU 지출 0) — 死因 **F1**(R4 음성대조가 처치가 반드시 움직이는 양 위에
> 서 있어 확률 1로 FAIL하고, 그 FAIL이 거짓 귀속을 정본에 넣는다) + 차단 **B1–B6**.
> 판정서 `VERDICT_e2_rules_2026-09-15.md`(`145b786e…`, 433행) §8이 rev2의 필수 조건이고 **전부 GPU 0**이다.
> **감사자가 승인한 것은 그대로 둔다**: E2-α 채택 근거 3개 · 통제 목록 §3 · 인용 승계 · 예산 산술 ·
> 양성대조 P1 · §10(하지 못하는 것)의 자기 진술 · §4-3의 정직 공시.
> 이 문서는 **claims-auditor 재감사 대기**이며 `GO` 이전 **sbatch 금지**. 하네스(2단)는 `GO` 이후.
> **GPU 지출 현재 0.** 제출에는 **새 범위 한정 OVERRIDE + 사용자 승인**이 추가로 필요하다.
> 기준 트리: HEAD `4d1a0ec1aed2a2963ce94b5bc49926680d7c5f30`.

## 0. rev1 → rev2 변경표 (재감사 색인)

| 지적 | 수리 | 위치 |
|---|---|---|
| **F1** R4가 처치가 움직이는 양 위 | R4를 **F1-a(E-qcond 음성대조)** 로 교체 + **R4′(양성 예보)** + **R4″(결과 축 음성대조)** **셋 다 채택** | §6 |
| **B1** `a_r4_s2` E-qcond 17.6 오기 | **17.00(43/253)** 로 정정, **전 칸 분자/분모 병기** | §4-2 |
| **B2** 한 노브지만 **두 기전** | §4-1에 **(다) 드레인 제거** 등재 + R2를 **3분지**로 + `WITHIN_SPREAD`/`MOVES_UP`에서 "교락 우려 약화" **금지** + Q4·E2C-1 병기 의무 | §4-1·§6 |
| **B3** Q1 이름(시간) ≠ 1차 추정량(count) | Q1 개명 + **R1을 전 추정량 동시 성립으로 강화** + **E-iter 신규 등록**(규약 리터럴) | §1·§5·§6 |
| **B4** E-pact를 확증으로 사용 | **구성 항등식 가드**로 재라벨, 기전 문장은 `P(prefill_active \| idx2)` 경험량으로 교체 | §4-3·§5 |
| **B5** R3가 술어가 아님 | R3를 **리터럴 3항**으로 재작성(OFF-only idx3 · idx0 ≤ 5.0% · `split_transition` 보고 전용) | §6 |
| **B6** bench 유효성 게이트 부재 | **P9–P12** 신설 | §7 |
| 권고(§8-8) 예산 재배분 | **채택**(수정 1건: `b_r3` 2 seed 유지 — 사유 §8-2) | §2·§8 |
| 권고(§8-9) correctness 기록 | **P13** 신설 | §7 |

---

## 1. 질문과 판정량 (B3 수리 — 이름을 추정량과 일치시킨다)

> **sticky 노브 하나를 ON으로 올리면 shape A의 실현 분할이 D44로 고정되는가, 그리고 그때 같은
> offered에서 달성률(achieved)이 어떻게 변하는가.**

| 기호 | 판정량 | 등급 |
|---|---|---|
| **Q1** | decode-busy **표본의** D44(`stream_index==2`) 점유 — **네 추정량 전부**(§5). ~~"시간 점유"~~ 라고 부르지 않는다 | **1차** |
| **Q2** | 같은 offered·같은 seed에서의 **achieved = completed/duration**, **paired n=4**(a_r4·a_r2) | 2차(**범위 한정 기전 측정**, §9) |
| **Q3** | 실현 division 전수 히스토그램(idx 0–4) | 한 노브 검증(§4) |
| **Q4** | `split_transition` 이벤트 수 | **제2 기전 계측**(§4-1(다)) — **보고 전용**, 판정 아님 |
| **Q5** | 정본 술어 goodput 율(TTFT ≤ 3000 ms ∧ 요청내부 token-ITL p95 ≤ 60 ms) | 3차(병기 전용, **계단 함수로만**, E2C-6) |

---

## 2. 등록 arm · 셀 · 순서 (리터럴 — 재량 없음)

**arm 차이는 환경변수 한 개**: OFF = `PDMUX_STICKY_PARTITION` **unset** / ON = **`1`**.

**등록 셀 3종 × 20 boot** (판정서 §5 대안 설계 채택 + `b_r3` 2 seed):

| 셀 | shape | in/out | offered | `--num-prompts` | seeds | arm당 n | 908623 OFF 실측 duration |
|---|---|---|---|---|---|---|---|
| `a_r4` | A | 256/512 | 8.00 | 400 | s1,s2,s3,s4 | **4** | 131.01 s (s1) / 130.23 s (s2) |
| `a_r2` | A | 256/512 | 3.00 | 400 | s1,s2,s3,s4 | **4** | 142.45 s (s1) |
| `b_r3` | B | 8192/64 | 1.15 | 200 | s1,s2 | **2** | 286.85 s (s1) / 287.74 s (s2) |

**seed 리터럴**: **s1 = 4386 · s2 = 4162 · s3 = 251 · s4 = 2630**.
출처 = λ0 쉬핑 규칙 `lambda0_plan.choose_seeds`(`max|Ēbar−1|` 최소 순)의 **λ0 등록 N multiset
{80,110,140,200,400,660,1260}에 대한 순위 1·2·3·4**(이 문서가 재도출: 0.014603 / 0.015100 /
0.017025 / 0.018145, 전부 `EBAR_TOL = 0.025` 내).
★**공시(자유 표면 차단)**: 같은 규칙을 **rev2 자신의 N multiset {200,400}** 에 적용하면 순위가
**1083 / 1262 / 4386 / 3519**로 달라진다. 그것을 **쓰지 않는 이유**를 미리 등록한다 — (i) s1/s2는
908623 OFF 셀이 **실제로 측정된 seed**이고 바꾸면 R6 교차 비교의 바이트 비교 가능성이 사라진다,
(ii) 그 재계산은 s1을 3위로 강등해 "이미 쓴 seed가 최적이 아니다"라는 불일치를 만든다.
rev2 격자에서 네 seed의 `max|Ēbar−1|`는 **0.001304 / 0.015100 / 0.017025 / 0.016533**으로
**전부 `EBAR_TOL` 내**이므로 어느 읽기에서도 적법하다.

**실행 순서(리터럴, 인접쌍 · seed-major)** — 20 boot:
```
a_r4:  s1/OFF s1/ON  s2/OFF s2/ON  s3/OFF s3/ON  s4/OFF s4/ON
a_r2:  s1/OFF s1/ON  s2/OFF s2/ON  s3/OFF s3/ON  s4/OFF s4/ON
b_r3:  s1/OFF s1/ON  s2/OFF s2/ON
```
⚠️ arm 순서가 항상 OFF→ON인 것은 **등록된 비대칭**(E2C-7). 단 rev1과 달리 **seed 4쌍에 걸쳐
드리프트가 평균**되고, paired 차분이 seed별로 계산된다.

**rev1 대비 잃는 것**(등록): `a_r0`(비포화 regime의 Q5) · `b_r0`(중간 점유 43% 대조) · R5[B].
**얻는 것**: **게이트 #3(n≥4) 충족** · R2 문턱이 "3× 산포"라는 임의 상수에서 **paired bootstrap CI**로
교체 · 순서 효과 부분 상쇄.

---

## 3. 통제 요인 (908623에서 한 글자도 바꾸지 않는 것)

`MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` · `BACKEND=flashinfer` ·
`CFG=.../probes/pdmux_homog5.yml` · `CTX=16384` · `MEM_FRACTION=0.82` · `MAX_RUNNING=48` ·
`--disable-radix-cache` · `--disable-overlap-schedule` · `--chunked-prefill-size -1` ·
**cudagraph ON** · `PDMUX_R2_POLICY=fixed` · `PDMUX_R2_FIXED_DSM=44` · `--random-seed 1` ·
warmup 8요청 `--max-concurrency 1 --seed 7` · `--random-range-ratio 1.0` ·
`--dataset-name random --dataset-path <ShareGPT raw> --tokenize-prompt` ·
`--output-details --output-file` · **셀당 1 boot**(시간창 슬라이싱 금지 — `CONSENSUS §3 항목120`) ·
셀 간 `kill` → `sleep 5` → `pkill -9` → `sleep 10` · 포트 재확인.
감사자 전항 대조 결과 `lambda0.sbatch`와 **일치**(판정서 §6).

---

## 4. 한 노브 검증 — 노브는 하나, **기전은 둘**이다 (B2 수리)

### 4-1. 코드 독해 (쉬핑 트리, HEAD `4d1a0ec`)

`adjust_stream_groups`(`multiplex/multiplexing_mixin.py:1163-1211`)에서 sticky가 바꾸는 것:
```python
if not self.running_batch.is_empty() and (
    self.split_prefill_batch or self.sticky_partition_enabled
):                                                            # :1169-1171
    if self.sticky_partition_enabled and self._sticky_fixed_idx is not None:
        stream_idx = self._sticky_fixed_idx                   # :1172-1179  (= idx 2)
    else:
        ... manual_divisions 임계 스캔 ...                     # :1180-1197
elif not self.running_batch.is_empty():
    set_current_stream_idx(self.real_sm_group_num - 1)        # :1199-1200  (= idx 4, (0,108))
```
`_init_sticky_partition`(`:388-455`)은 `PDMUX_R2_POLICY=fixed` + `FixedPolicy(decode_sms=44)`에서
`sm_counts=[(108,0),(92,16),(64,44),(16,92),(0,108)]` 중 `decode_sms==44` ⇒ **`_sticky_fixed_idx = 2`**.
미정의 조합은 init에서 **예외로 거부**(조용한 기본값 없음).

**(가) `manual_divisions` 드리프트** — `pdmux_homog5.yml`의 세 행 임계가 **전부 0**이라 OFF arm의
`:1183-1187` 스캔은 원리상 **idx 3 = `(16,92)` = D92**로 끝난다. 908623 실측 **11셀 전부 idx3 = 0**
(도달 경로가 `_slo_on`(`:1256`) True에서 v7 블록(`:1313-1352`)에 선점됨). **가정하지 않고 Q3로
매 셀 측정**한다(§6 R3).

**(나) 실현 인덱스 집합이 arm에 따라 닫힌다** — sticky ON이면 `:1169-1171`의 논리합이 **항상 참**이라
**idx 4 분지(`:1199-1200`)는 도달 불가**이고, `_r2_decide_idx`(`:670-682` + `controller.py:92-102`,
legacy는 `true_dual_worker_runtime is None` ⇒ `safe_boundary=True`)는 **항상 idx 2**를 돌려준다.
⇒ ON arm의 decode-busy 표본 인덱스는 **{0, 2}로 닫힌다**(idx 0은 decode 배치가 빈 순간의
read-skew로만 들어온다, `CONSENSUS §3 항목120`). **이것이 死因 F1의 근거이며, R4′로 등록한다.**

**(다) ★제2 기전 — 드레인 제거** (rev1이 빠뜨린 것):
- OFF: decode-only span은 idx 4, prefill span은 idx 2 ⇒ **prefill 경계마다 두 번 전환**, 각 전환이
  `prefill_stream.synchronize(); decode_stream.synchronize()`(v7 쪽 `:1344-1345`, elif 쪽 `:1361-1362`).
- ON: v7 쪽은 `_tgt == stream_idx`(둘 다 2)라 **드레인이 발생하지 않는다**; elif 쪽만 남는다
  ⇒ ON은 **prefill 경계당 드레인 1쌍을 제거**한다.
- 크기: `a_r4` OFF는 131.01 s에 `split_transition` **204건**(= 1.56 건/s). 엔진 자신의 주석
  (`:1270-1274`)이 *"the adjust block does 2x synchronize (drain) at EVERY prefill boundary …
  ~200ms+ spike vs the true ~40ms TPOT"*.
- **부호가 분할 효과와 반대**다(분할은 느리게, 드레인 제거는 빠르게) ⇒ **E2C-1 필수 병기**,
  R2는 3분지(§6).

**(라) 부수 효과 없음이 확인된 축**(감사자 독립 확인, 판정서 §7-2·§7-3): cudagraph 비대칭 없음
(`cuda_graph_runner.py:806-817`이 전 stream group에 같은 `capture_bs`를 `f"{stream_idx}_{bs}"` 키로
캡처) · admission · KV/mamba 예산 · telemetry 발화율 · prefill 축(두 arm 모두 prefill span에서 idx 2).

### 4-2. 908623 OFF arm 실측 (GPU 0, 분자/분모 병기 — B1 수리)

decode-busy = `event=="runtime_snapshot" ∧ phase!="startup" ∧ decode_running_batch_size>0`.

| 셀 | busy n | **E-cnt** | **E-time** | **E-iter** | **E-qcond** | idx0 | idx3 | `split_transition` |
|---|---|---|---|---|---|---|---|---|
| `a_r0` | 2677 | 4.5 (120/2677) | 4.8 | 4.5 | **17.65 (6/34)** | 0.0 | **0** | 604 |
| `a_r2` | 610 | 8.0 (49/610) | 10.6 | 8.0 | **17.65 (18/102)** | 0.0 | **0** | 301 |
| **`a_r4`** | 555 | **8.6 (48/555)** | 13.6 | 8.7 | **17.20 (43/250)** | 0.4 | **0** | 204 |
| `a_r4_s2` | 557 | 8.3 (46/557) | 13.4 | 8.3 | ★**17.00 (43/253)** | 0.0 | **0** | 169 |
| `b_r0` | 313 | 43.1 (135/313) | 66.2 | 43.1 | 80.00 (36/45) | 4.2 | **0** | 43 |
| **`b_r3`** | 441 | **90.7 (400/441)** | 97.6 | 90.9 | **100.00 (395/395)** | 0.2 | **0** | 100 |
| `b_r3_s2` | 441 | 90.7 (400/441) | 97.5 | 90.9 | 100.00 (392/392) | 0.0 | **0** | 100 |

스냅샷 간격 중앙값 1.99–2.04 ms. `P(prefill_active | idx2)` = **a_r0 64.2 · a_r2 49.0 · a_r4 66.7 ·
a_r4_s2 76.1 · b_\* 100.0** (B4 수리 — 아래 4-3).

### 4-3. 추정량 대조 — **어디가 맞고 어디가 안 맞는지** (교훈 80)

- **E-cnt는 감사자 §4-A 1열을 5/5 재현**(4.5 / 8.0 / 8.6 / 43.1 / 90.7). 감사자 재검증도 일치.
- **E-qcond(= `prefill_queue_depth > 0` 조건부)는 감사자 3열을 5/5 재현**(17.65 / 17.65 / 17.20 /
  80.00 / 100.00). ★단 `a_r4_s2`는 감사자 표에 **행이 없고** rev1이 17.6으로 적은 것은 **오기**였다
  ⇒ **17.00 (43/253)** 으로 정정(B1).
- ★**감사자 판정서 §4-A 2열(시간 가중)은 재현되지 않는다** — 이 문서 4변형 + 감사자 추가 10변형
  (전방·후방·중앙 × 무캡·캡 5종) 전부 실패(`b_r0 46.1` · `b_r3 91.9`를 동시에 내는 규약 없음).
- ★★**새 공시(이 회차 산출) — 같은 일이 판정서 §2 B3의 E-iter에서도 일어났다.** 판정서가 처방한
  E-iter 값(`a_r0 4.5 · a_r2 8.1 · a_r4 8.8 · a_r4_s2 8.4 · b_r0 45.6 · b_r3 92.3 · b_r3_s2 92.1`)을
  4규약(좌/우 귀속 × decode-busy 필터 유무) 전부로 재현 시도한 결과 **shape A는 ±0.1pp로 일치하나
  `b_r0`는 42.1–43.4(vs 45.6) · `b_r3`는 90.8–90.9(vs 92.3)로 어긋난다**. ⇒ E-iter는 **채택하되
  규약을 이 문서가 리터럴로 못박고**(§5), 셀프테스트 기대값은 **이 문서의 재산출값**을 쓴다.
  판정서의 E-iter 열은 **인용하지 않는다**(E2C-8).
- **E-pact(= `prefill_active_batch_size > 0` 조건부)가 7/7 셀 100.0%인 것은 새 사실이 아니라
  구성 항등식이다** — v7 블록이 prefill span마다 idx 2를 설치하므로 **두 arm 모두에서 100%**다
  (B4 수리). 경험적 내용은 역방향 `P(prefill_active | idx2)`에 있고 그 값은 **49.0–76.1%**(shape A)다.
  rev1의 *"`CONSENSUS §1-26(B)`의 가장 순수한 형태"* 와 *"ON에서 E-cnt ≫ E-pact면 노브가 발화"*
  는 **둘 다 철회**한다(후자는 ON에서 두 값이 모두 ≈100%라 성립하지 않는다).

---

## 5. 추정량 명세 (리터럴 — 하네스가 구현할 것)

입력: 셀·arm·seed당 telemetry 한 개. 대상: `event=="runtime_snapshot" ∧ phase!="startup"`.
D44 = `stream_index == 2`; 하네스는 `(prefill_sms, decode_sms)`도 기록하고 **불일치 시 ABORT**.

| 이름 | 표본/가중 | 역할 |
|---|---|---|
| **E-cnt** | decode-busy 표본, 균등(개수) | 1차 |
| **E-time** | 〃, `t[i+1] − t[i]` (**전방차분·무캡**, 마지막 표본 제외) | 1차 |
| **E-iter** | **전 비-startup 스냅샷** 인접쌍의 `decode_iterations` 차분 **> 0**인 것만, **왼쪽 스냅샷의 `stream_index`에 귀속**, 가중 = 차분값 | 1차 |
| **E-qcond** | decode-busy ∧ `prefill_queue_depth > 0`, 균등 | 1차 |
| **E-pact** | decode-busy ∧ `prefill_active_batch_size > 0`, 균등 | **구성 항등식 가드**(≠100% 이면 계측 결함 ⇒ ABORT) |

★**E-cnt·E-time·E-iter·E-qcond는 "1차 추정량 집합"이며 R1은 넷 모두에서 동시 성립해야 한다**
(B3 수리). 어느 하나를 골라 헤드라인으로 쓰지 않는다 — 쉬핑 코드 `:778-800`이 count 격자에 대해
*"NOT comparable across the flag … TIME-WEIGHTED statistics … are unaffected"* 라고 경고하고,
`CONSENSUS §3 항목120`이 시간 부과 쪽을 경고한다 ⇒ **양쪽 경고를 동시에 만족하는 유일한 길은
전 추정량 동시 성립이다.**

**필수 셀프테스트(제출 전, GPU 0)**: 다섯 추정량 + idx0/idx3/`split_transition`/busy n을 908623
아카이브에 적용해 **§4-2 표를 7/7 셀 바이트 일치**로 재현. 불일치 = 제출 차단.
기대값의 출처: E-cnt·E-qcond = 감사자 §4-A(독립) · E-iter·E-time·idx0/idx3·`P(pact|idx2)` =
**이 문서의 재산출**(판정서 E-iter 열은 재현 불가이므로 기대값으로 쓰지 않는다).

---

## 6. 결정 규칙 (리터럴 PASS/FAIL)

승계 예보(감사자 §7, 거짓 가능): *sticky ON에서 shape A의 D44 점유가 >0.90으로 오르고, 그 arm의
λ\*(A)는 현 3.05보다 낮아진다. 이 예측이 거짓이면 λ0R-8(iii)의 교락 우려가 약화된다.*

| 규칙 | 리터럴 내용 | 결과 문안 |
|---|---|---|
| **R1**(1차) | `a_r4`·`a_r2` **전 seed**에서 ON의 **E-cnt·E-time·E-iter·E-qcond 네 값이 모두 > 0.90** ∧ 같은 셀·같은 seed OFF의 네 값이 **모두 < 0.20** | 참 ⇒ `STICKY_REALIZES[A]`; 거짓 ⇒ `STICKY_DOES_NOT_REALIZE[A]`, 이때 **P1(배선)과 측정을 분리해 기록** |
| **R2**(2차, **3분지** — B2 수리) | `a_r4`·`a_r2` 각각 **paired n=4**(seed별 ON−OFF 차)의 **paired bootstrap 95% CI**가 0을 포함하는가 / 어느 쪽에 있는가 | CI < 0 ⇒ `LAMBDA_MOVES_DOWN`; CI ∋ 0 ⇒ `LAMBDA_WITHIN_CI`; CI > 0 ⇒ `LAMBDA_MOVES_UP`. ★**세 분지 전부에 Q4(`split_transition` 수)와 E2C-1 병기 의무.** ★**`LAMBDA_WITHIN_CI`와 `LAMBDA_MOVES_UP`에서 *"λ0R-8(iii) 교락 우려가 약화된다"* 를 쓰는 것을 금지**한다(두 기전 상쇄와 구별 불가) |
| **R3**(한 노브, 리터럴 3항 — B5 수리) | (i) **OFF arm** 전 셀 idx3 점유 **= 0** (ii) **두 arm** 전 셀 idx0 점유 **≤ 5.0%**(초과 ⇒ 그 셀 **UNRESOLVED**) (iii) `split_transition` 수는 **보고 전용**, 단 **ON > OFF인 셀이 하나라도 있으면** §4-1(다)의 코드 독해가 깨진 것이므로 **E2C-1을 강한 형태로 승계** | — |
| **R4**(음성대조, F1-a) | `b_r3` **두 arm의 E-qcond 차 ≤ 0.5pp**(OFF 실측 100.00%, sticky가 움직일 수 없는 양 — prefill 대기 중 decode-busy 표본은 두 arm 모두 idx 2) | 거짓 ⇒ arm 차이가 sticky 아닌 무엇(순서·드리프트·read-skew)에서도 온다 ⇒ R1/R2 전체 caveat |
| **R4′**(양성 예보, F1-b) | `b_r3`의 E-cnt는 OFF 90.7% → ON **99.5–100.0%** 가 **코드상 강제**된다(§4-1(나)). 이 범위를 벗어나면 **노브 배선 또는 계측 가정이 깨진 것** | 벗어남 ⇒ `ABORT_STICKY_STATE_MISMATCH`와 같은 층 |
| **R4″**(결과 축 음성대조) | `b_r3`는 실현 분할이 거의 안 움직이는 셀(E-time 97.6 → ~100) — `\|achieved(ON) − achieved(OFF)\|`의 **paired n=2 차이**가 같은 job의 **seed 간 산포를 초과**하면 그 차이는 분할이 아니라 **순서·드리프트·드레인 수**에서 온 것 | 초과 ⇒ R2 해석 전체에 caveat |
| **R5**(재현) | `a_r4`·`a_r2`에서 **seed 간 E-cnt 산포가 arm별로 ≤ 5pp** | 거짓 ⇒ `SEED_SPREAD_FAILS` ⇒ 그 셀의 R1/R2 **UNRESOLVED** |
| **R6**(부수) | 908623 OFF 셀과의 교차 비교 | **판정 아님** — 보고만, **노드 축 교락 병기 의무** |

**UNRESOLVED 규약**: boot 실패·CAPPED(P7)·유효성 위반(P9–P12)·셀 누락은 **그 seed의 OFF/ON 짝을
통째로** UNRESOLVED로 만들고, 짧아진 집합으로 재라벨하지 않는다(D23 승계). paired n이 4 미만으로
줄면 R2는 그 셀에서 **UNRESOLVED**(n=3 이하로 CI를 내지 않는다 — 게이트 #3).

---

## 7. 실패 닫힘 게이트

- **P1 ★양성 대조**: ON boot 로그에 `PD-mux sticky partition ENABLED ... (fixed target index=2)`가
  **있어야** 하고 OFF boot 로그에는 **없어야** 한다. 위반 ⇒ `ABORT_STICKY_STATE_MISMATCH`.
  (게이트 #237 — 노브가 unset 루프에 먹히는 실패를 이 게이트가 실제로 잡는다.)
- **P2 인덱스 해석**: ON 로그의 `fixed target index` ≠ **2** ⇒ abort.
- **P3 배너 동일성**: 두 arm의 `max_mamba_cache_size` / `max_total_num_tokens` 가 다르면 abort.
- **P4 manifest·digest**: `sync_engine_tree.sh` manifest ≥ 24 엔트리 + `nemotron_h.py` 포함;
  결정경로 파일·이 문서의 sha256을 **결정 이전에** `REGISTRATION_SHA256.txt`에 기록. 두 arm은
  **한 번의 sync·같은 manifest**.
- **P5 제출 전 커밋**(λ0R-10).
- **P6 boot 실패 2연속** ⇒ abort.
- **P7 셀 벽시계 캡**: `T_cap = 3 × (908623 OFF 실측 duration)`. 초과 ⇒ **CAPPED** ⇒ 그 seed 짝
  UNRESOLVED(`max()` 합산 금지 — 게이트 #7).
- **P8 arm 라벨 무결성**: 파일명과 JSON의 arm 문자열 불일치 ⇒ abort.
- **P9–P12 bench 유효성**(B6 수리, 셀별): **P9** `errors == 0` · **P10** `completed == num_prompts` ·
  **P11** 입력/출력 길이 정확(`--tokenize-prompt` 기준) · **P12** `random_range_ratio == 1.0`.
  위반 ⇒ 그 seed 짝 **UNRESOLVED**(조용히 짧아진 duration이 achieved를 올리는 것을 막는다).
- **P13 correctness 기록**(판정서 §8-9): **ON arm 첫 boot**에서 고정 프롬프트 6개를 greedy로
  받아 텍스트를 아티팩트에 기록한다. 사유: `CONSENSUS §3 항목27`의 sticky correctness 실증은
  **Ha8 · division `(92,16)`** 에서 나온 것이고 **(Nemotron-9B, `(64,44)`, flashinfer, cudagraph ON)
  조합의 sticky ON은 서빙 부하에서 한 번도 돌지 않았다**. 비용 ≈0(같은 boot). ★이 기록은
  **correctness 게이트가 아니라 증거 보존**이다 — 통과/실패 판정을 붙이지 않는다.

---

## 8. 예산 (실산 — 게이트 #113)

### 8-1. 계산
per-boot = bench + warmup(A 54 s / B 17 s) + boot·teardown 52 s.

| 셀 | boot | per-boot | 소계 |
|---|---|---|---|
| `a_r4` | 8 | 131.0+54+52 = 237.0 s | 1896.0 s |
| `a_r2` | 8 | 142.5+54+52 = 248.5 s | 1988.0 s |
| `b_r3` | 4 | 286.9+17+52 = 355.9 s | 1423.6 s |
| 프리플라이트 | — | — | 300 s |
| **등록 보통** | **20 boot** | | **5607.6 s = 1.558 GPU-h** |
| **최악 코너**(ON에서 A bench ×2.5 · B ×1.5) | | | **7535.8 s = 2.093 GPU-h** |
| `--time` | | | **03:00:00** (3.0 GPU-h 하드 캡; P7이 개별 셀을 먼저 자른다) |

### 8-2. 감사자 대안(1.359 GPU-h, 18 boot)에서 **한 가지만 수정**했다
`b_r3`를 **seed 2개**로 유지해 **+2 boot(+0.198 GPU-h)** 했다. 사유: 판정서가 `b_r3_s2`를
*"R5[B]가 검사하는 E-cnt 산포가 이미 0.0pp라 정보 ≈0"* 으로 평가했는데, **같은 판정서가 신설한
R4″(결과 축 음성대조)는 `b_r3`의 achieved 산포를 요구한다**. seed 1개면 그 산포를 **908623(다른
노드·다른 job)에서 빌려와야** 하고, 그것은 λ0R-1·E2C-7이 막 금지한 형태다. ⇒ B측 산포를
**이 job이 스스로 만든다**.
(rev1 1.421 → rev2 **1.558 GPU-h**. 증분의 내역: `a_r0`·`b_r0` 제거 −0.53, n=4 확장 +0.47, `b_r3`
2 seed 유지 +0.20.) 비교: λ0 908623 실지출 **1.391 GPU-h**.

---

## 9. 승계 — 인용 금지 / 필수 병기 (문자 승계)

**판정서 §9의 E2C-1 … E2C-7을 전문 그대로 승계한다**(요지만 재기재; 결과 문서는 판정서 원문을 인용):

- **E2C-1**(최대 위험) — *"노브는 하나지만 기전은 둘"*: decode SM 108→44 **+** prefill 경계 드레인
  1쌍 제거. 부호 반대. ⇒ **achieved 변화를 "분할 민감도"로 단독 귀속 금지**, **변화 없음을
  "분할은 무관"으로 읽는 것도 금지**.
- **E2C-2** — E-cnt는 시간 점유가 아니다(`_dual_worker_sync`의 32분의 1 count-subsample,
  `:777-778, 802-806`). **"D44 점유 X%"는 추정량 이름 없이 인용 금지.**
- **E2C-3**(인용 금지) — *"ON에서 D44 ≈100%로 측정됐다 ⇒ sticky 효과 실증"* 금지. ON의 인덱스는
  엔진이 {0,2}로 **강제**한다. R1의 ON 절은 **양성 대조**이지 측정이 아니다. 정본 전례
  (`CONSENSUS §3 항목27`, 0.0839→1.0000)는 **Ha8·`(92,16)`** 이고 이번과 다르다.
- **E2C-4**(인용 금지) — E-pact 100%로 `§1-26(B)`를 확증하지 마라(양 arm 항등식). 경험량은
  `P(prefill_active | idx2)` = shape A **49.0–76.1%**, 분모 24–77 표본.
- **E2C-5**(필수 병기) — **이 회차는 λ0R-8(iii)을 해소하지 않는다**(두 arm의 prefill 축이 동일).
- **E2C-6**(필수 병기) — Q2/Q5 제한: `a_r2`의 Q5는 **절벽 위**(OFF TTFT p95 **1.806 s** vs SLO 3.0 s,
  여유 1.66배) ⇒ **계단 함수로만** 읽는다.
- **E2C-7**(필수 병기) — 실행 순서 OFF→ON의 **등록된 비대칭**. 908623 인접 셀 산포는 A +0.604% ·
  B −0.310%(n=2). `LAMBDA_WITHIN_CI`를 *"차이가 없다"* 로 읽는 것 금지(검출력 미등록).
- ★**E2C-8**(신규, 이 문서 산출 — 필수 병기) — *"판정서가 발표한 추정량 열을 규약 없이
  인용하지 마라."* 이 트랙에서 **두 번** 일어났다: `VERDICT_result_lambda0_908623` §4-A **2열
  (시간 가중)** 과 `VERDICT_e2_rules_2026-09-15` §2 B3의 **E-iter 열**이 각각 14변형·4변형
  재현 시도에서 **shape B에서 재현되지 않았다**(전자 `b_r0 46.1`/`b_r3 91.9`, 후자 `b_r0 45.6`/
  `b_r3 92.3` vs 재산출 42.1–43.4 / 90.8–90.9). ⇒ **추정량 열을 발표하는 문서는 규약을 리터럴로
  적거나 재현 코드를 남겨야 한다**(교훈 80의 계열; 새 방법론 게이트 후보).

**기존 승계 전부 유효**: λ0R-1…λ0R-10 · λ5C-1…8 · NPC-I(shape A 반증) · N-7·N-8·N-9·N-11 ·
게이트 #13/#16 "닫았다" 금지 · C2 인용정지 (a)(b) · HE0 · layer-type 死 · 정책 순위 · stake #1.

**Q2가 여전히 정책 주장이 아닌 이유**(등록): 이 격자는 **정지 rate 사다리**이므로 **게이트 #2**에
의해 정책 비교 벤치가 아니다. n=4 + paired CI는 Q2를 *"이 셀에서 실현 분할을 D44로 고정하면
achieved가 X% 변한다"* 까지만 올린다. **정책 서열 주장 금지.**

---

## 10. 이 회차가 하지 못하는 것 (감사자 승인 — 불변)

1. **게이트 #6(λ\*_SLO)을 닫지 않는다** — E1a/E1b가 한다.
2. **P2 블로커 ②(W4 λ\* 실측 부재)를 해소하지 않는다**(②③은 같은 블로커의 두 이름).
3. **Claim D/E 등급 불변**(둘 다 미검증). **새 성능 판정 0건.**
4. **shape A와 B를 서로 비교하지 않는다.**
5. **λ0 라벨을 재계산하지 않는다**(908623 아티팩트는 읽기 전용).
6. **true-dual(B4)에 대해 아무것도 말하지 않는다**(R2C-2 미충족).
7. **E3(48-구속)을 포함하지 않는다.**

## 11. 제출 선행조건 (순서대로 — 전부 미충족)

1. **이 문서(rev2)의 재감사**(claims-auditor) → `GO` 또는 `GO-with-caveats`.
2. `GO` 이후 **하네스 작성**(`e2_sticky.sbatch` + `e2_realized_mix.py`) + §5 셀프테스트 통과(GPU 0).
3. **하네스 층 감사**(2단, 교훈 34).
4. **새 범위 한정 OVERRIDE + 사용자 승인**(기존 2건 소진; presubmit M4R·TC1 차단 2건 유효).
5. **커밋**(P5) → 그 다음에만 `sbatch`.
