# 사전등록 — **E2 (sticky 분할 대조)** rev1 (2026-09-15)

> **출처**: `VERDICT_result_lambda0_908623_2026-09-14.md`(`089daed5…`) **§7 E2** 처방 +
> §4-A(**λ\*(A)는 D44 측정이 아니다**) + §6 λ0R-8. **사용자 결정(2026-09-15)**: 다음 세션 순서
> **b → a**, 즉 **E2를 E1보다 먼저**(감사자 §7 권고는 E1a 우선이었고 사용자가 뒤집었다 —
> 근거: 4-A가 λ\*(A)의 인용 범위와 **true-dual 정규화 설계 전체**를 교락시키므로 먼저 닫으면
> E1b(shape A) 설계가 달라진다).
> 이 문서는 **claims-auditor 규칙층 감사 대기**이며 `GO` 이전 **sbatch 금지**. 하네스(2단)는
> 규칙층 `GO` 이후에 작성한다(교훈 34: 대형 캠페인은 하네스 이전에 결정 규칙을 먼저 감사받는다).
> **GPU 지출 현재 0.** 제출에는 **새 범위 한정 OVERRIDE + 사용자 승인**이 추가로 필요하다
> (기존 2건 — D-none 908534 · λ0 908623 — 은 **소진**).
> 기준 트리: HEAD `4d1a0ec1aed2a2963ce94b5bc49926680d7c5f30`.

---

## 0. ★이 문서가 **문자로 고정하는** 설계 결정 1건 — **E2-α를 채택한다**

감사자 §7의 E2 문안은 *"E1a/E1b의 **부분집합**을 반복"* 이고 판정량 (ii)가 *"**같은 offered에서의**
λ\*/goodput 변화"* 인데 **E1이 아직 없다** ⇒ sticky-OFF 대조를 어디서 얻을지가 미정이었다.
**이 문서는 (E2-α)를 등록한다: 대조 arm(sticky OFF)을 같은 job 안에 들고 간다.**

| | E2-α (**채택**) | E2-β (기각) |
|---|---|---|
| OFF arm | **이 job 안에서 새로 측정** | job 908623의 셀을 재사용 |
| 노드·물리 GPU 축 | **불변**(두 arm 같은 노드·같은 boot 절차) | 교락(908623 = gpu38, 새 job은 미지정) |
| 엔진 트리 | **구성상 동일**(한 번의 `sync_engine_tree.sh`, 한 manifest) | 908623 이후 트리 변경 시 **arm 비동일성** 재발 |
| 예산 | ≈**1.42 GPU-h**(§8) | ≈0.7 GPU-h |

**채택 사유 3개(전부 정본 등재 사실)**:
1. **게이트 #233이 아직 열려 있다** — 미등록 노드 축이 gpu38→43→40→41로 4회 움직였다.
   E2-β는 그 열린 축 위에서 arm을 비교하게 된다.
2. **D-none 회차(job 908534)에서 귀속을 닫은 것이 바로 "같은 job 안의 대조"였다** — 같은 설계를
   쓴다.
3. 감사자 §4-D가 `multiplexing_mixin.py` **한 파일 차이**를 907959↔908623 교차 비교의 **병기
   의무**로 이미 등재했다. E2-α는 그 조항을 발생시키지 않는다.

**E2-β를 완전히 버리지는 않는다**: 908623 OFF 셀과의 교차 비교는 **GPU 0 부수 산출**로 보고하되
**판정량이 아니다**(§6 R6). 판정은 언제나 **이 job 안의 OFF↔ON**.

---

## 1. 질문 (한 문장) 과 판정량

> **sticky 노브 하나를 ON으로 올리면 shape A의 실현 분할이 D44로 고정되는가, 그리고 그때
> 같은 offered에서 달성률(achieved)이 어떻게 변하는가.**

| 기호 | 판정량 | 축 | 등급 |
|---|---|---|---|
| **Q1** | decode-busy 시간의 **D44(stream_index=2) 점유** | telemetry `runtime_snapshot` | **1차** |
| **Q2** | 같은 offered에서의 **achieved = completed/duration** (클라이언트 측) | `bench_*.jsonl` | 2차(**성능 판정 아님**, §9) |
| **Q3** | 실현 division **전수 히스토그램**(idx 0–4) | telemetry | **한 노브 검증**(§4) |
| **Q4** | `split_transition` 이벤트 수 | telemetry | **한 노브 검증**(§4) |
| **Q5** | 정본 술어 goodput 율(TTFT ≤ 3000 ms ∧ 요청내부 token-ITL p95 ≤ 60 ms) | `bench_*.jsonl` per-request | 3차(**병기 전용**, §9) |

**Q2/Q5는 이 회차에서 성능 판정으로 승격될 수 없다** — 이유는 §9에 등록. Q5는 **게이트 #6(λ\*_SLO)을
닫지 않는다**: 이 회차 셀은 λ\*_SLO를 브래킷하도록 설계되지 않았고, 재스코어로 λ\*_SLO를 얻는 것은
감사자 §7 "하지 말 것 (a)"·게이트 #8이 금지한다.

---

## 2. 등록 arm · 셀 · 실행 순서 (리터럴 — 재량 없음)

**arm은 둘, 차이는 환경변수 한 개뿐이다.**

| arm | `PDMUX_STICKY_PARTITION` | 그 밖의 전부 |
|---|---|---|
| **OFF** | **unset**(엔진 기본값 "0") | 동일 |
| **ON** | **`1`** | 동일 |

**등록 셀 7개**(908623의 셀 이름·offered·num_prompts·seed·shape를 **그대로** 승계 — 새 셀을
만들지 않는다. 감사자 §7 "하지 말 것 (d)"의 정신):

| # | 셀 | shape | in/out | offered | `--num-prompts` | seed | 908623 OFF 실측 duration | 선정 사유 |
|---|---|---|---|---|---|---|---|---|
| 1 | `a_r0` | A | 256/512 | 1.10 | 660 | seed1 | 615.32 s | **D44 점유 최저(4.5%)** + 유일하게 여유 있는 비포화 셀(§4-C goodput 100%) |
| 2 | `a_r2` | A | 256/512 | 3.00 | 400 | seed1 | 142.45 s | λ\*(A)≈3.05 **근방**(무릎) |
| 3 | `a_r4` | A | 256/512 | 8.00 | 400 | seed1 | 131.01 s | ★**λ\*(A) 공급 셀** |
| 4 | `a_r4_s2` | A | 256/512 | 8.00 | 400 | seed2 | 130.23 s | 같은 job 안 **재현 산포** 측정(§6 R2의 문턱을 이 job이 스스로 만든다) |
| 5 | `b_r0` | B | 8192/64 | 0.45 | 80 | seed1 | 183.08 s | B 중 D44 점유 최저(43.1%) |
| 6 | `b_r3` | B | 8192/64 | 1.15 | 200 | seed1 | 286.85 s | ★**λ\*(B) 공급 셀**(D44 90.7% — **거의 null 예상**, 음성대조) |
| 7 | `b_r3_s2` | B | 8192/64 | 1.15 | 200 | seed2 | 287.74 s | 〃 재현 산포 |

seed1 = **4386**, seed2 = **4162**(908623 `plan.json` 전사, 서버 `--random-seed 1`).

**실행 순서(리터럴, 인접쌍)** — 셀마다 OFF와 ON을 **연속 boot**로 붙여 느린 드리프트(열·노드 상태)가
두 arm에 거의 같게 실리게 한다:

```
a_r0/OFF  a_r0/ON   a_r2/OFF  a_r2/ON   a_r4/OFF  a_r4/ON   a_r4_s2/OFF  a_r4_s2/ON
b_r0/OFF  b_r0/ON   b_r3/OFF  b_r3/ON   b_r3_s2/OFF  b_r3_s2/ON
```
**14 boot.** 순서를 바꾸는 운영자 스위치는 없다(D18 규율). ⚠️ arm 순서가 항상 OFF→ON인 것은
**등록된 비대칭**이다: 잔여 순서 효과는 §6 R4의 대조로만 측정하고, 해소됐다고 주장하지 않는다.

---

## 3. 통제 요인 (908623에서 **한 글자도 바꾸지 않는 것**)

`MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` · `BACKEND=flashinfer` ·
`CFG=.../probes/pdmux_homog5.yml` · `CTX=16384` · `MEM_FRACTION=0.82` ·
`MAX_RUNNING=48` · `--disable-radix-cache` · `--disable-overlap-schedule` ·
`--chunked-prefill-size -1` · **cudagraph ON**(`--disable-piecewise-cuda-graph` 미전달) ·
`PDMUX_R2_POLICY=fixed` · `PDMUX_R2_FIXED_DSM=44` · `--random-seed 1` ·
warmup 8요청 `--max-concurrency 1 --seed 7` · `--random-range-ratio 1.0` ·
`--dataset-name random --dataset-path <ShareGPT raw> --tokenize-prompt` ·
`--output-details --output-file` · 셀당 **1 boot**(한 telemetry 파일 = 한 boot; 시간창 슬라이싱
금지 — `CONSENSUS §3 항목120`).

**셀 간 정리**: `kill` → `sleep 5` → `pkill -9 -f "launch_server.*--port $PORT"` → `sleep 10`
(908623 하네스 그대로). 포트는 셀마다 새로 잡고 사용 전 비어 있음을 확인한다.

---

## 4. ★한 노브 검증 — "sticky만 바꾼다"가 **이 config에서 참인지** 먼저 대조했다

> 교훈 **250**: *감사자 자신의 처방도 다중 노브를 움직일 수 있다 — 처방을 등록하기 전에 빌드의
> 노브 결합 조건을 대조하라.* (E3 초안이 `max_mamba_cache_size = max_running_requests`로 죽은 바로 그 사유.)

### 4-1. 코드 독해 (쉬핑 트리, HEAD `4d1a0ec`)

`adjust_stream_groups`(`multiplex/multiplexing_mixin.py:1163-1211`)에서 sticky가 바꾸는 것은
**분지 조건의 논리합 한 개**와 **그 분지 안의 인덱스 선택**뿐이다:

```python
if not self.running_batch.is_empty() and (
    self.split_prefill_batch or self.sticky_partition_enabled      # :1169-1171
):
    if self.sticky_partition_enabled and self._sticky_fixed_idx is not None:
        stream_idx = self._sticky_fixed_idx                        # :1172-1179
    else:
        decode_bs = ...  # manual_divisions 임계 스캔                :1180-1197
elif not self.running_batch.is_empty():
    set_current_stream_idx(self.real_sm_group_num - 1)             # :1199-1200  ← (0,108)
```

`_init_sticky_partition`(`:388-455`)은 `PDMUX_R2_POLICY=fixed`+`FixedPolicy(decode_sms=44)`에서
`sm_counts=[(108,0),(92,16),(64,44),(16,92),(0,108)]` 중 `decode_sms==44` ⇒ **`_sticky_fixed_idx = 2`**
로 해석되고, 미정의 조합(`PDMUX_LA_COORD`/`PDMUX_SLO_SCHED`/`PDMUX_FIXED_DECODE_SM_FILE`/
`r2_policy ∉ {"", "fixed"}`/`real_sm_group_num < 3`)은 **init에서 예외로 거부**한다(조용한 기본값 없음).

**이 회차의 잠재적 제2 효과 2개와 그 처리**:

- **(가) `manual_divisions` 드리프트**: `pdmux_homog5.yml`의 세 행 임계가 **전부 0**
  (`[92,16,0] [64,44,0] [16,92,0]`)이므로 OFF arm의 `:1183-1187` 루프는 항상 마지막 행까지
  진행해 **idx 3 = `(16,92)` = D92**를 고르게 된다 — 이는 **D44가 아니다**. ON arm은 idx 2로
  고정되므로 이 상태에서 두 arm은 **다르다**.
  ★**그러나 그 상태는 908623에서 한 번도 실현되지 않았다**: 전 셀 decode-busy 스냅샷에서
  **idx 3 점유 = 0**(11셀 전부, 아래 4-2 표). 기전은 `_slo_on`(= `r2_policy is not None`, `:1256`)
  이 참이라 prefill span에서는 v7 블록(`:1313-1352`)이 `_r2_decide_idx` = **idx 2**를 먼저 설치하고
  `adjust_stream_group`을 **False**로 내려 `adjust_stream_groups` 호출 자체를 건너뛰기 때문이다.
  ⇒ **등록**: idx 3 점유는 **Q3로 매 셀 측정**하고, **어느 arm에서든 > 0이면 "한 노브" 주장에
  caveat를 붙인다**(§6 R3). 0이라고 **가정하지 않는다**.
- **(나) 드레인/스위치 수**: 코드 주석은 sticky가 *"adds no switch point, removes no drain"* 이라고
  주장하지만, 그것은 **선택기가 같은 인덱스를 골랐을 때만** 참이다. ⇒ **등록**: `split_transition`
  수(Q4)를 arm별로 병기하고, **주석의 주장을 인용하지 않는다 — 측정치를 인용한다**.

### 4-2. 908623 실측으로 본 OFF arm의 현재 상태 (GPU 0, 이 문서가 직접 재집계)

decode-busy(`decode_running_batch_size > 0`, `phase != "startup"`) 스냅샷 기준:

| 셀 | busy n | **E-cnt** D44% | E-time D44% | **E-qcond** D44% | **E-pact** D44% | **idx3** | `split_transition` |
|---|---|---|---|---|---|---|---|
| `a_r0` | 2677 | **4.5** | 4.8 | **17.6** | **100.0** | **0** | 604 |
| `a_r2` | 610 | **8.0** | 10.6 | **17.6** | **100.0** | **0** | 301 |
| `a_r4` | 555 | **8.6** | 13.6 | **17.2** | **100.0** | **0** | 204 |
| `a_r4_s2` | 557 | **8.3** | 13.4 | 17.6 | **100.0** | **0** | 169 |
| `b_r0` | 313 | **43.1** | 66.2 | **80.0** | **100.0** | **0** | 43 |
| `b_r3` | 441 | **90.7** | 97.6 | **100.0** | **100.0** | **0** | 100 |
| `b_r3_s2` | 441 | **90.7** | 97.5 | 100.0 | **100.0** | **0** | 100 |

스냅샷 간격 중앙값 **1.99–2.04 ms** (전 셀).

### 4-3. ★추정량 대조 — 감사자 표와 **어디가 맞고 어디가 안 맞는지**

정직 공시(교훈 80: 출처 허위는 규칙 정본 안이 가장 위험하다):

- **E-cnt는 감사자 §4-A 1열을 5/5 셀에서 정확히 재현한다** (4.5 / 8.0 / 8.6 / 43.1 / 90.7).
- **E-qcond(= `prefill_queue_depth > 0` 조건부)는 감사자 3열을 5/5 셀에서 정확히 재현한다**
  (17.6 / 17.6 / 17.2 / 80.0 / 100.0). ⇒ 감사자의 *"prefill·decode 동시 대기 조건부"* 는
  **대기열 깊이 조건**이었음이 확정.
- ★**감사자의 2열(시간 가중)은 문서의 서술만으로는 재현되지 않는다.** 시도한 4개 규약
  (전방차분/후방차분 × 무캡/캡)의 결과는 shape A에서 ±0.1pp로 근접하나 `b_r0`에서
  44.3–66.2 vs 감사자 **46.1**, `b_r3`에서 91.4–97.6 vs **91.9**로 갈린다. ⇒ **등록**:
  이 회차의 E-time은 **전방차분·무캡**으로 **문자 고정**하고(아래 §5), 감사자 2열과는
  **직접 비교하지 않는다**. 시간 가중이 1차 판정량이 아닌 것은 이 비재현 때문만이 아니라
  `CONSENSUS §3 항목120`(구간 부과 아티팩트 — 순간 상태에 표본 간격을 부과하는 것이
  `C-R` confound를 만들었다) 때문이다.
- ★**신규(이 문서 산출)**: **E-pact(= `prefill_active_batch_size > 0` 조건부)는 7/7 셀에서
  정확히 100.0%** — *"decode가 D SM에서 돌았다 ⟺ prefill이 동시에 in-flight였다"*
  (`CONSENSUS §1-26(B)`)의 **가장 순수한 형태**이며, 이 회차의 **기전 검사**로 등록한다.
  sticky ON은 이 동치를 깨는 것이 목적이므로, **ON arm에서 E-cnt ≫ E-pact 가중이 되는 것이
  곧 노브가 발화했다는 뜻**이다.

---

## 5. 추정량 명세 (하네스가 구현할 것 — 리터럴)

입력: 셀당 telemetry `tel_<cell>_<arm>.jsonl` 한 개. 대상 레코드:
`event == "runtime_snapshot" ∧ phase != "startup"`. 시간축 `timestamp_monotonic_s`.
D44 = `stream_index == 2`(≡ `(prefill_sms, decode_sms) == (64, 44)`; 하네스는 **둘 다** 기록하고
불일치 시 **ABORT**).

| 이름 | 표본 집합 | 가중 |
|---|---|---|
| **E-cnt** (**1차**) | `decode_running_batch_size > 0` | 균등(개수) |
| E-time (병기) | 〃 | `t[i+1] − t[i]` (**전방차분·무캡**; 마지막 스냅샷 제외) |
| E-qcond (병기) | 〃 ∧ `prefill_queue_depth > 0` | 균등 |
| E-pact (기전) | 〃 ∧ `prefill_active_batch_size > 0` | 균등 |

**필수 셀프테스트(하네스 제출 전, GPU 0)**: 위 네 추정량을 **908623의 아카이브 telemetry**에
적용해 §4-2 표의 **E-cnt·E-qcond·E-pact·idx3 열을 7/7 셀에서 바이트 일치**로 재현해야 한다.
불일치 = **제출 차단**. ★이 셀프테스트는 항등식이 아니다(교훈 9): 기대값의 출처는 **이 코드가
아니라** 감사자 판정서 §4-A 표(E-cnt/E-qcond)와 이 문서의 독립 재집계(E-pact/idx3)이다.

---

## 6. 결정 규칙 (사전등록 — 리터럴 PASS/FAIL, 예측은 거짓 가능하게 적는다)

감사자가 §7에 **등록해 둔 예보**(그대로 승계): *sticky ON에서 shape A의 D44 점유가 4.5–13.5% →
**>0.90**으로 오르고(정본 전례 0.0839→1.0000) 그 arm의 λ\*(A)는 현 3.05보다 **낮아진다**.
이 예측이 거짓이면 λ0R-8(iii)의 교락 우려가 약화된다.*

| 규칙 | 내용 | 결과 문안 |
|---|---|---|
| **R1**(1차) | shape A **3 셀 전부**(`a_r0`,`a_r2`,`a_r4`)에서 ON arm의 **E-cnt > 0.90** ∧ 같은 셀 OFF arm의 E-cnt < 0.20 | 참 ⇒ `STICKY_REALIZES[A]`; 거짓 ⇒ `STICKY_DOES_NOT_REALIZE[A]` + **노브가 발화했는지**(§7 P1)와 **발화했는데 안 올랐는지**를 구분해 기록 |
| **R2**(2차) | `a_r4`에서 ON의 achieved가 OFF보다 **낮고**, 그 차이의 절대값이 **같은 job의 seed 재현 산포**(= `|a_r4 − a_r4_s2|` 를 arm별로 계산한 뒤 **큰 쪽**)의 **3배를 초과** | 참 ⇒ `LAMBDA_MOVES[A]`(**분할 민감도 실재, 성능 판정 아님**); 거짓 ⇒ `LAMBDA_WITHIN_SPREAD[A]` ⇒ **λ0R-8(iii) 교락 우려가 약화된다**(감사자 문안 그대로) |
| **R3**(한 노브) | 두 arm 전 셀에서 **idx3 점유 = 0** ∧ Q4 `split_transition`의 arm 차가 §4-1(나)로 설명 가능 | 거짓 ⇒ **"sticky만 바꿨다"에 caveat 부착**(무효화가 아니라 범위 축소) |
| **R4**(순서) | `b_r3`와 `b_r3_s2`는 **음성대조**(OFF에서 이미 90.7%) — 두 arm의 E-cnt 차가 **≤ 3pp**여야 한다 | 거짓 ⇒ **arm 간 차이가 sticky 아닌 무엇(순서·드리프트)에서도 나온다**는 뜻 ⇒ R1/R2 결과 전체에 caveat |
| **R5**(재현) | `a_r4` vs `a_r4_s2`, `b_r3` vs `b_r3_s2`의 E-cnt 차가 **arm별로 ≤ 5pp** | 거짓 ⇒ `SEED_REPEAT_FAILS` ⇒ 그 shape의 R1/R2는 **UNRESOLVED** |
| **R6**(부수) | 908623 OFF 셀과의 교차 비교 | **판정 아님** — 보고만, 노드 축 교락을 **반드시 병기** |

**UNRESOLVED 규약**: boot 실패·CAPPED(§7)·셀 누락은 그 셀의 짝을 통째로 **UNRESOLVED**로
만들고, 짧아진 셀 집합으로 **재라벨하지 않는다**(D23 승계).

---

## 7. 실패 닫힘 게이트 (전부 abort 또는 명시 라벨 — 조용한 통과 없음)

- **P1 ★양성 대조 — ON arm이 실제로 ON인가**: ON boot의 서버 로그에
  `PD-mux sticky partition ENABLED ... (fixed target index=2)` 가 **있어야** 하고,
  OFF boot의 로그에는 **없어야** 한다. 위반 ⇒ `ABORT_STICKY_STATE_MISMATCH`.
  (교훈 9: 조용히 OFF인 ON arm은 항등식 게이트다.)
- **P2 인덱스 해석 확인**: ON boot 로그의 `fixed target index` 가 **2**가 아니면 abort
  (config가 바뀌어 D44가 다른 인덱스로 가면 전 판정이 무효).
- **P3 배너 동일성**: 두 arm의 `max_mamba_cache_size` / `max_total_num_tokens` 가 **같아야** 한다
  (E3 회차의 교훈 — 노브 하나가 KV 예산을 움직이면 그건 다른 실험이다). 다르면 abort.
- **P4 manifest·digest**: `sync_engine_tree.sh` manifest ≥ 24 엔트리 + `nemotron_h.py` 포함,
  결정경로 파일·이 사전등록 문서의 sha256을 **결정 이전에** `REGISTRATION_SHA256.txt`에 기록.
  두 arm은 **같은 manifest**를 쓴다(한 번만 sync).
- **P5 제출 전 커밋**: sbatch 제출 **전에** 사전등록·하네스·판정서가 전부 커밋돼 있어야 한다
  (λ0R-10 — 지난 회차가 이 순서를 어겼다).
- **P6 boot 실패 2연속** ⇒ `ABORT_2_CONSECUTIVE_BOOT_FAILURES`.
- **P7 셀 벽시계 캡**: 셀별 `T_cap = 3 × (908623 OFF 실측 duration)`. 초과 시 그 셀은
  **CAPPED**로 표시하고 짝과 함께 **UNRESOLVED**(조용히 짧은 런을 비교하지 않는다 —
  duration은 절대 `max()`로 합치지 않는다, 게이트 #7).
- **P8 arm 라벨 무결성**: 각 셀 아티팩트에 arm 문자열이 파일명과 JSON 양쪽에 기록되고
  두 곳이 다르면 abort.

---

## 8. 예산 · 벽시계 (실산 — 게이트 #113)

908623 **실측** duration + warmup(A ≈ 54 s, B ≈ 17 s) + boot·teardown ≈ 52 s/셀 기준.

| 항목 | 값 |
|---|---|
| OFF arm 7셀 | 1776.6 s(bench) + 267 s(warmup) + 364 s(boot·teardown) = **2407.6 s** |
| ON arm 7셀 (변화 없다고 가정) | **2407.6 s** |
| 프리플라이트(셀프테스트·manifest·digest) | ≈ 300 s |
| **등록 보통** | **5115 s ≈ 1.421 GPU-h** |
| **최악 코너**(ON에서 A 셀 bench ×2.5, B ×1.5 — achieved가 떨어지면 고정 `num_prompts`가 길어진다) | ≈ 7023 s ≈ **1.95 GPU-h** |
| `--time` | **03:00:00** (= 3.0 GPU-h 하드 캡, P7이 그 안에서 개별 셀을 자른다) |

비교: λ0 908623 실지출 1.391389 GPU-h. **E2는 그와 같은 규모다** — 감사자 §7의 "≈0.49 GPU-h"는
5 boot(대조 arm 없음, 짧은 셀만) 기준이었고, **E2-α 채택 + 재현 산포를 이 job이 스스로 만들도록
한 결정**(R2의 문턱이 외부 상수가 되지 않게)이 그 차이의 전부다. 줄이려면 `a_r0`(단독 1230 s,
전체의 26%)이나 `*_s2` 2쌍(1036 s)을 빼야 하며, **그 경우 잃는 것**은 각각
"여유 있는 비포화 regime에서의 Q2/Q5"와 "R2 문턱의 내생성"이다 — 어느 쪽도 이 문서가
독단으로 버리지 않는다(사용자·감사자 판단 사항).

---

## 9. 승계 — 인용 금지 / 필수 병기 (문자 승계)

이 회차는 아래를 **전부 그대로 진다**. 결과 문서·정본은 문자 그대로 승계한다.

- **λ0R-1**: λ\*(B)와 인용금지 λ_inf(B)의 0.2377% "일치"는 **N-8을 되살리지 않는다**.
  ⇒ 이 회차의 어떤 수치도 N-8 자격 조항을 충족시키지 않는다(교훈 248: 실격은 자격 조항이라
  사후 독립 측정으로 충족되지 않는다).
- **λ0R-3**: λ\*(A)는 `--max-running-requests 48`이 구속한 값 — ON arm에서도 **같은 조건**이므로
  이 회차의 achieved 역시 **설정 상한 48에서의 처리율**이다.
- **λ0R-7**: 17자리 인용 금지. 모든 수치는 **(창 길이, N, arm)** 과 함께.
- **λ0R-8**: λ\*(A)를 *"decode_sm=44 용량"* 으로 인용 금지 · shape A에 905835 계열 5× 민감도
  이식 금지 · **λ\*(A)로 B4(true-dual)를 B1에 정규화하는 설계는 교락**.
  ★이 회차가 R1/R2를 어떻게 내든 **(iii)은 자동으로 해소되지 않는다** — E2는 sticky 축을 닫을 뿐,
  true-dual이 바꾸는 prefill·decode **동시성** 축은 측정하지 않는다.
- **λ0R-9**: 게이트 #6 재채점의 "비 2.3–3.4×"는 **shape B 한정**, A에 일반화 금지.
- **게이트 #13/#16 "닫았다" 금지** · **C2 인용정지 (a)(b)** · **HE0** · **layer-type 死** ·
  **정책 순위** · **stake #1 구조 판정** — 전부 불변, 이 회차가 건드리지 않는다.
- **Q2/Q5가 성능 판정이 아닌 이유**(등록): (1) arm당 셀당 **n=1 boot**이고 재현 산포는
  2 셀에서만 측정된다 — 게이트 #3(n≥4)을 충족하지 않는다. (2) arm 순서가 항상 OFF→ON인
  **등록된 비대칭**이 있다. (3) 이 격자는 **변화 trace가 아니라 정지 rate 사다리**다
  (게이트 #2). ⇒ 결과는 *"sticky가 실현 분할을 바꾸는가"* 와 *"그때 achieved가 산포 밖으로
  움직이는가"* 까지만 말한다.

---

## 10. ★이 회차가 **하지 못하는 것** (미리 등록)

1. **게이트 #6(λ\*_SLO)을 닫지 않는다** — E1a/E1b가 한다.
2. **P2 블로커 ②(W4 λ\* 실측 부재)를 해소하지 않는다** — ②③이 같은 블로커의 두 이름이라는
   정본 판정은 불변.
3. **Claim D/E 어느 쪽도 등급이 바뀌지 않는다**(둘 다 미검증 유지). **새 성능 판정 0건**이 목표다.
4. **shape A와 B를 서로 비교하지 않는다**(in·out 동시 변경 — 감사자 §7 "하지 말 것 (c)").
5. **λ0 라벨을 재계산하지 않는다** — 908623 아티팩트는 **읽기 전용**이다.
6. **true-dual(B4)에 대해 아무것도 말하지 않는다** — R2C-2 선행 미충족(E4는 처방되지 않았다).
7. **E3(48-구속)을 포함하지 않는다** — 감사자가 초안을 자기 철회했고 수정안조차 수행 가능성이
   미확인이다.

## 11. 제출 선행조건 (전부 미충족 — 순서대로)

1. **이 문서의 규칙층 감사**(claims-auditor) → `GO` 또는 `GO-with-caveats`.
2. `GO` 이후 **하네스 작성**(`e2_sticky.sbatch` + `e2_realized_mix.py`) + §5 셀프테스트 통과(GPU 0).
3. **하네스 층 감사**(2단; 교훈 34).
4. **새 범위 한정 OVERRIDE 문서 + 사용자 승인** — 기존 2건 소진, presubmit의 **M4R·TC1 차단
   2건은 여전히 살아 있다**.
5. **커밋**(P5) → 그 다음에만 `sbatch`.
