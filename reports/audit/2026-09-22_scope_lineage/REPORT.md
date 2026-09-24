# PD-mux 좌표계 v2 — 코드 근거 감사 (sticky · 계보 · 선행 대조)

> **감사 산출물 · 새 성능 판정 0건 · GPU 지출 0 · 정본 무수정.**
> 순위·우열·goodput·SLO·운영점·인과 귀속은 이 문서에 없다. 집계는 기술 집계다.
> 작성 2026-09-22 (1차 통과), **2026-09-22 2차 통과에서 §0·§4·§7·§8·§9 갱신** — 사용자가 논문 camera-ready를
> `/home/wonho/Experiments/KISTI/Papers/`에 제공해 §4 전체를 preprint 근거에서 게재본 근거로 교체하고
> **M5·M6의 1차 판정 2건을 정정, M4를 한 단계 더 좁혔다**(§4.0에 판본 이력).
> 인용 fingerprint는 `citations.json`(`line_citations.json` 규약).
> 표기 규약: `[READ]` = 로그·설정·코드에서 직접 읽은 값, `[DERIVED · 규약명]` = 집계 산출값.
> target(설정에 적힌 값)과 realized(로그/telemetry로 확인된 값)를 구분해 적었다.

---

## 0. 항목별 한 줄 판정

| 항목 | 상태 | 한 줄 |
|---|---|---|
| **전제** 엔진 버전 특정 | **확인** | sticky 3개 job 전부 `multiplexing_mixin.py` sha256 `59eaafb4…` = git `f60a128`(2026-08-03) 판. byte-exact 재현 확인. **멈춤 조건 미발동.** |
| **S1** 캠페인 설정 | **확인** | 3 job·22 boot·3 cell(d16/d54/d92)+smoke. OFF arm은 `sticky_smoke` 한 boot 뿐, `s2_sticky`는 **전 셀 ON**(대조 arm 없음). |
| **S2** sticky 실제 적용 | **확인** | 21/21 ON boot에서 `ENABLED … fixed target index=1` 1줄, OFF boot 0줄. `PDMUX_GREEN_READOUT`은 **이 빌드에 없음**(2026-08-25 도입) → green read-out 대조 불가. |
| **S3** 집계기 PART 정의 | **확인(+반박 1건)** | PART ⟺ realized `prefill_sms>0 AND decode_sms>0`. **prefill-idle PART는 PART로 센다.** 재집계: sticky ON PART 시간의 **90.4%(s2_sticky)·96.3%(smoke ON arm)** 가 prefill-idle. |
| **S4** 03절 비용 수치 출처 | **확인** | `p1_gates/gate2` HOLB 캠페인(job 874602/874633/874635) — **sticky OFF**(플래그가 그 job에 존재하지도 않았음). ⇒ **prefill-idle PART 혼입 없음.** 단 그 캠페인은 **자동 격자 경로**다(아래 I-1). |
| **S5** SM 효과/동거 효과 분리 재료 | **없음(확인)** | sticky 아카이브에 per-step decode forward 지속시간 필드가 **0건 채워짐**(`decode_last_tpot_ms`·`decode_step_count`·`worker_overlap_ratio` 전부 0/8541). 유일 후보 `measured_itl_ewma_ms`는 α=0.85 EWMA + 전환 경계 표본 배제 필터라 상태 귀속 불가. |
| **M1** MuxWise = 경쟁자 vs 기판 | **(a) 확인** | upstream pdmux는 MuxWise 저자들이 올린 MuxWise의 engine 부분. 추적 이슈가 MuxWise arXiv를 "Related resources"로 명시. |
| **M2** residency 선행 공백 | **부분(좁혀야 함)** | MuxWise의 GPU utilization = Nsight "active SM 비율"(residency 아님). **Bullet Fig. 20a는 SM 구성별 지속시간 막대 타임라인** — 집계 통계는 아니나 "체류 시간" 시각화다. "아무도 보고하지 않았다"는 **정의된 집계 추정량 한정**으로 좁혀야 한다. |
| **M3** 선행이 체류를 암묵 조절 | **지지** | MuxWise는 `N_PL = ⌈(T_d × N_T)/T_P⌉`로 **prefill 층 수를 decode 1 iteration을 덮도록** 정한다(원문 명시). Bullet은 prefill SM 수를 시간축에서 직접 변경. |
| **M4** "PART가 느린 건 SM 수가 아니라 동거" | **부분(선행 2편 모두 보고)** | MuxWise §3.3.1(dense Llama, ~0–30% slowdown) **그리고** Bullet §3.2.3("isolated SM에서도 memory/network 경합은 남는다", "decode가 prefill보다 경합에 민감") 둘 다 이미 보고. 우리 주장은 **hybrid 재확인 + 크기 차이**로 좁혀야 한다. |
| **M5** Bullet 분할 기구·재구성 비용 | **확인(★1차 판정 정정)** | libsmctrl `set_stream_mask`(TPC 입도) + MPS. **Green Context 명시적 기각**(4 정책에 700MB+). ★camera-ready Table 5에 **`Resource Re-config` Mean 4.1 μs / P99 5.9 μs 측정값이 있다** — "측정값 없음"은 preprint 기준 오류였다. 단 그 값은 **CPU-side**, 우리 `≤0.04 ms`는 **device-level 상한** ⇒ 같은 자릿수·다른 양. |
| **M6** 평가 조건 비교 | **확인(★1차 판정 정정)** | 표 §4.5. ★**Bullet은 A100 1장 평가를 갖고 있다**(§4.2.1) — "우리만 단일 GPU"는 오류였다. 여전히 우리만인 것은 **hybrid 모델군**이고, 가장 큰 축 차이는 **SLO regime**(우리 ITL p95 60 ms · MuxWise TBT 50/100 ms · Bullet TPOT 150–200 ms + 정규화 TTFT)이다. |
| **D1** `adjust_stream_groups` 입력·drift | **부분** | 입력은 맞다. 단 **drift는 sticky OFF(또는 `_sticky_fixed_idx is None`)일 때만** — sticky ON+fixed에서는 상수. 문장에 단서가 빠져 있다. |
| **D2** "한 셀에서 4–19%" | **확인** | job 872077. 출처는 **코드 주석 자신**(`multiplexing_mixin.py:361-373`)과 `sticky_smoke` 결과(OFF arm `frac=0.0805`). |
| **D3** R2 admission hold | **부분(중대한 단서 누락)** | 배치 형성 보류는 맞다. 그러나 `r2_admission_limited`는 **`CoarseGrainedController`(generic/hybrid)만** 세운다 — `FixedPolicy`는 절대 세우지 않으므로 **sticky/E1/S2 계열 캠페인에서는 한 번도 발화하지 않았다.** |
| **D4** "dual-worker도 admission에 개입" | **반박** | `begin_admission`/`finish_admission`은 **스톱워치**다(`admission_latency_ms` 기록). 보류·지연·순서 변경 없음. |
| **D5** "prefill SM 몫이 체류 길이를 정한다" | **부분(지지 데이터 존재, 인용 금지 캠페인)** | `s8p_prefill`(decode 16 고정, prefill 16→92)에서 평균 PART 체류가 4 arm 전부 단조 감소. 단 그 캠페인은 **claims-auditor 미통과 = 정본 인용 금지**. |
| **정정** "hybrid layer 단위 prefill 발사 성립 여부" | **닫힘(열린 질문 아님)** | 4모델 전부 `forward_split_prefill` 패치 존재. |

**부수 발견(범위 밖이나 정본에 영향):** 아래 §6 I-1(자동 격자 경로 55/59)·I-2(census 줄 인용 표류)·I-3(ENABLED 로그 문구 오류).

---

## 1. 전제 — 캠페인 당시 설치된 엔진 버전

### 1.1 현재 트리 대조

| 파일 | sha256 | 비고 |
|---|---|---|
| `sglang_engine_dev/python/sglang/srt/multiplex/multiplexing_mixin.py` | `e2a97b423f93ff6d…` | [READ] 설치 트리 |
| `workspace/engine-port/src/multiplex/multiplexing_mixin.py` | `e2a97b423f93ff6d…` | [READ] 소스 오버레이 |

**동일하다.** 다만 이것은 *지금* 이야기다.

### 1.2 캠페인 당시 — 다르다

sticky 3개 job의 `runtime_source_manifest*.sha256`이 기록한 mixin은 **`59eaafb4ac61cc09…`** 로, 현재와 **다르다** [READ].

| job | 날짜(서버 로그 첫 타임스탬프) | manifest 파일 | manifest sha256 | mixin sha256 |
|---|---|---|---|---|
| 872800 (`sticky_smoke`) | 2026-08-03 16:56:36 | `runtime_source_manifest.sha256` | `2e448ad4ddebb7c4…` | `59eaafb4…` |
| 873015 (`s2_sticky` d16/d54) | 2026-08-04 13:56:54 | `runtime_source_manifest_s2.sha256` | `2e448ad4ddebb7c4…` | `59eaafb4…` |
| 873921 (`s2_sticky` d92) | 2026-08-05 11:38:42 | `runtime_source_manifest_s2d92.sha256` | `25d961fd1f09769c…` | `59eaafb4…` |

**manifest는 입력이 아니라 출력이다.** `sync_engine_tree.sh`는 설치 후 `sha256sum … > "${manifest_tmp}"; mv -f` 로 manifest를 **쓴다**(`sync_engine_tree.sh`, sha256 `09f6ea69321e5d50…`, "MANIFEST (2026-09-13: 17 → 24 → 25 entries)" 블록). 즉 manifest의 sha는 **그 job이 실제로 설치한 바이트**의 기록이다. (반대로 그것은 검증이 아니므로, manifest가 *기대값*과 맞는지는 아무도 확인하지 않았다 — 이 점도 그대로 적어 둔다.)

### 1.3 `59eaafb4…`의 정체

git 이력에서 `src/multiplex/multiplexing_mixin.py`의 모든 판본 sha256을 계산해 대조했다 [DERIVED · git 전수 대조]:

```
59eaafb4ac61cc09a…  f60a128  2026-08-03  feat(engine): add PDMUX_STICKY_PARTITION to hold target SM split
25d170e0135b285b…  27bbae7  2026-08-25  feat(pdmux): green read-out …            <- 다음 변경
b0c92f2a2e69f3df…  ed2a410  2026-07-31  feat(engine): opt-in force-trace …       <- 직전 판본
```

`git show f60a128:…` 로 복원한 파일이 `59eaafb4…` 와 **byte-exact 일치**한다. 같은 manifest의 나머지 5개 multiplex 파일(`dual_worker`/`profile`/`controller`/`telemetry`)도 `f60a128` 판과 전부 일치한다 [DERIVED].

- sticky를 **도입한 바로 그 커밋**이고, 그 다음 mixin 변경은 2026-08-25다.
- 3개 job은 2026-08-03 ~ 08-05 → **세 job 모두 `f60a128` 판**이다.
- ⇒ **멈춤 조건 "캠페인 당시 설치된 엔진 버전을 특정할 수 없다"는 발동하지 않는다.**

### 1.4 그 뒤 sticky 코드가 바뀌었는가 — 아니다

`f60a128` 판과 현재 판의 diff에서 sticky 관련 블록(`_init_sticky_partition` 본문, `adjust_stream_groups`의 disjunct·fixed-idx 분기·decode-empty 분기)은 **문자 단위로 동일**하고 **줄 번호만 이동**했다 [DERIVED · diff]:

| 블록 | `f60a128` | 현재 |
|---|---|---|
| `_init_sticky_partition` | 238–305 | 388–455 |
| sticky disjunct + fixed idx | 881–894 | 1169–1182 |
| decode-empty → idx 0 | 910–923 | 1198–1211 |

따라서 **아래의 코드 판단은 캠페인 판본과 현재 판본 양쪽에 동일하게 성립**하고, `citations.json`은 현재 트리(실행 트리) 기준으로 fingerprint를 잡았다.

### 1.5 provenance 구멍 1건 (기록만)

`sticky_smoke`(Zamba2-7B-Instruct로 서빙)의 manifest 11행에 **`models/zamba2.py`가 없다** [READ]. zamba2는 2026-08-04에야 sync+manifest에 편입됐다(`sync_engine_tree.sh`의 "Brought under sync + manifest on 2026-08-04" 주석). 즉 872800의 **모델 구현 provenance는 기록되지 않았다** — `dev_tree_edits.md` 24번이 NemotronH/Falcon-H1/Granite에 대해 지적한 것과 같은 계열의 사례다. mixin provenance에는 영향이 없다.

---

## 2. 작업 A — sticky 캠페인

### 2.0 코드 의미 (전제로 확인할 것)

#### `_init_sticky_partition`

| 주장 | 판정 | 근거 |
|---|---|---|
| 기본값 OFF | **확인** | `os.environ.get("PDMUX_STICKY_PARTITION", "0") in ("1","true","True")` — `multiplexing_mixin.py:388-400` |
| 함께 쓸 수 없는 모드는 **거부**(무음 폴백 아님) | **확인** | `PDMUX_LA_COORD`·`PDMUX_SLO_SCHED`·`PDMUX_FIXED_DECODE_SM_FILE`·`r2_policy ∉ {"", "fixed"}`·`real_sm_group_num < 3` 전부 `raise RuntimeError` — `multiplexing_mixin.py:401-428` |
| `_sticky_fixed_idx` 결정 규칙 | **확인** | `FixedPolicy`일 때만: `sm_counts`에서 `decode_sms == target_d`인 인덱스를 찾고, **0번(plain prefill)·마지막(plain decode)을 제외**(`1 <= index <= real_sm_group_num - 2`)한 뒤 **첫 번째**를 취한다. 없으면 `raise`. `FixedPolicy`가 아니면 `None`(= 엔진 기본 decode-bs 선택기로 폴백) — `multiplexing_mixin.py:430-455` |

#### `adjust_stream_groups`

| 주장 | 판정 | 근거 |
|---|---|---|
| sticky ON이면 분기 조건이 `split_prefill_batch or sticky_partition_enabled`로 바뀐다 | **확인** | `if not self.running_batch.is_empty() and (self.split_prefill_batch or self.sticky_partition_enabled):` — `multiplexing_mixin.py:1169-1182` |
| decode 배치가 비었을 때 division을 놓는다 | **확인** | `else: … set_current_stream_idx(0)` — 주석이 이유까지 적어 둠("no decode work to protect… `E1_DECODE_REALIZED`의 decode-active 가중이 이 구간에 0 가중") — `multiplexing_mixin.py:1198-1211` |
| **따름정리(신규)**: sticky ON이면 중간 `elif`(`real_sm_group_num-1`, plain `(0,108)`)는 **도달 불가**다 | **확인** | 첫 조건이 "decode 非공백"을 전부 흡수한다. **telemetry로도 확인**: smoke ON arm의 realized index map은 `{0:'108/0', 1:'92/16'}` 로 `(0,108)`이 **한 번도 나타나지 않고**, OFF arm은 `{0:'108/0',1:'92/16',2:'0/108'}` 다 [READ] |

#### "배칭·admission·telemetry를 건드리지 않는다"는 주석

**확인.** sticky 관련 심볼(`sticky_partition_enabled`, `_sticky_fixed_idx`)은 `_init_sticky_partition`와 `adjust_stream_groups` **두 곳에서만** 읽힌다(전수 grep). `update_split_prefill_batch`·`get_new_batch_prefill`·telemetry emit 경로에 sticky 분기가 없다. 다만 **인과적으로 중립이라는 뜻은 아니다** — 분할 상태를 유지하면 decode/prefill 속도가 달라지고, 그것이 배치 구성과 admission 타이밍을 *간접적으로* 바꾼다. 주석이 말하는 것은 "직접 분기가 없다"까지다.

### 2.1 S1 — 캠페인 설정

공통(세 job 전부, 서버 인자 [READ]): `--disable-piecewise-cuda-graph`(= **cudagraph ON**, `--disable-cuda-graph`는 **없음**) · `--disable-radix-cache`(radix OFF) · `--chunked-prefill-size -1` · `--disable-overlap-schedule` · `--max-running-requests 48` · `--mem-fraction-static 0.80` · `--attention-backend triton` · ShareGPT rate 2 · `PDMUX_TRACE_FORCE_PREFILL=0`.

| cell | `PDMUX_STICKY_PARTITION` | `PDMUX_R2_POLICY` | target D | SM_COUNTS(realized 로그) / config | model | cudagraph | radix | chunked_prefill | max_running | mixin sha |
|---|---|---|---|---|---|---|---|---|---|---|
| `stk0_Ha8_d16` (872800) | **0** | `fixed` (`FIXED_DSM=16`) | 16 | `[(108,0),(92,16),(0,108)]` / `pdmux_e1_d16.yml` `67f03090…` | Zyphra/Zamba2-7B-Instruct | ON | OFF | −1 | 48 | `59eaafb4…` |
| `stk1_Ha8_d16` (872800) | **1** | `fixed` (16) | 16 | 동일 | 동일 | ON | OFF | −1 | 48 | `59eaafb4…` |
| `T8_d16` b1–b8 (873015) | **1** | `fixed` (16) | 16 | `[(108,0),(92,16),(0,108)]` / `pdmux_e1_d16.yml` | Qwen/Qwen2.5-7B | ON | OFF | −1 | 48 | `59eaafb4…` |
| `T8_d54` b1–b8 (873015) | **1** | `fixed` (54) | 54 | `[(108,0),(54,54),(64,44),(0,108)]` / `pdmux_e1_d54.yml` `f3adec3f…` | Qwen/Qwen2.5-7B | ON | OFF | −1 | 48 | `59eaafb4…` |
| `T8_d92` b1–b4 (873921) | **1** | `fixed` (92) | 92 | `[(108,0),(16,92),(64,44),(0,108)]` / `pdmux_e1_d92.yml` `13db746b…` | Qwen/Qwen2.5-7B | ON | OFF | −1 | 48 | `59eaafb4…` |

`[DEFAULT]`로 결정된 항목: `PDMUX_SLO_DWELL`·`PDMUX_TPOT_SLO_MS`(60)·`PDMUX_TTFT_SLO_MS`(3000)·`PDMUX_SLO_EMA`(0.85)는 스크립트가 설정하지 않아 코드 기본값. `PDMUX_DUAL_WORKER`/`PDMUX_TRUE_DUAL_WORKER`는 **미설정** → `architecture: "legacy"` [READ, telemetry].

**OFF arm**: `sticky_smoke`의 `stk0` **한 boot 뿐**. `s2_sticky`(20 boot)에는 **OFF 대조 arm이 없다** — `s2_sticky.sbatch:159` 주석 자신이 "★ the ONE substantive flag change vs `e1_m3_control.sbatch`"라고 적어 OFF 대조를 **다른 job(872077, `s8_frontier/e1_m3_control.sbatch`)에 위임**했음을 명시한다. 즉 s2의 OFF 대조는 **job 간 대조**다.

주의 2건:
- **T8 = Qwen/Qwen2.5-7B는 dense Transformer**다. hybrid 4종이 아니다. sticky의 서빙 증거 대부분이 dense에서 나왔다.
- d92는 **다른 job(873921)·다른 날짜·4 블록**이다. d16/d54(873015, 8 블록)와 **cell ≡ job 앨리어스**가 있다.

### 2.2 S2 — sticky가 실제로 적용됐는가

`STICKY_LOG_LINES` [READ]:

| 집합 | boot 수 | `ENABLED` 줄 | `fixed target index` |
|---|---|---|---|
| `s2_sticky` 873015 (d16×8, d54×8) | 16 | 전부 **1** | 전부 **1** |
| `s2_sticky` 873921 (d92×4) | 4 | 전부 **1** | 전부 **1** |
| `sticky_smoke` `stk1` | 1 | **1** | **1** |
| `sticky_smoke` `stk0` | 1 | **0** | — |

**"sticky 미확인" 셀은 0건이다.** `boot_ok=1`도 22/22.

`PDMUX_GREEN_READOUT`: **이 빌드에는 존재하지 않는다.** `green_readout.py`는 `27bbae7`(2026-08-25)에 들어왔고, 세 job의 manifest(11행)에도 없다 [READ]. ⇒ **realized green-context SM 수를 읽은 셀은 0건**이다. 본 감사가 "realized"라고 부르는 것은 전부 telemetry의 `(prefill_sms, decode_sms)` 필드이며, 그 값은 **엔진이 선택한 stream index의 `sm_counts` 항목**이다 — 드라이버가 실제로 준 SM 수를 읽은 값이 **아니다.** (이 구분은 E2C-38 계열 caveat과 같은 축이다.)

`s2_sticky`의 자체 게이트 `S2_GATE_DECODE_REALIZED`(≥0.90)도 참고로 기록한다 — smoke에서 OFF `frac=0.0805`(whole-file) / `0.0839`(probe-only), ON `frac=1.0000` [READ].

### 2.3 S3 — 집계기의 PART 정의

**05절 재집계에 쓰인 도구는 `results/residency_census_2026-09-21/residency_census.py`**(sha256 `6a0cb938cd40cacb…`)이고, `e2_realized_mix.py`는 그것이 **표본 술어를 import 해 오는 상류**다(`_is_snapshot`, `_is_decode_busy`). `e2_realized_mix.py` 자체는 λ0 아카이브의 D44(`stream_index == 2`) 전용 추정기이며 캠페인 횡단 재집계기가 아니다.

**Q1. PART를 무엇으로 정의하는가?**

> `residency_census.py:96-100`
> ```python
> def is_part(rec):
>     p, d = realized_sm(rec)
>     return bool(p) and bool(d)
> ```
> 그리고 `realized_sm(rec) = (rec.get("prefill_sms"), rec.get("decode_sms"))`.

**stream index가 아니라, 스냅샷에 기록된 realized `(prefill_sms, decode_sms)` 쌍이 양쪽 다 0이 아닌가**이다. prefill-busy 여부는 **들어가지 않는다.** 모듈 docstring도 이를 명시하고, 그 이유(교훈 246 / Stage 0 "pin은 realized로 검증")를 적어 둔다.

**Q2. sticky ON 셀의 prefill-idle & division 유지 step은 PART인가 FULL인가?**

**PART로 센다.** sticky ON에서 `_sticky_fixed_idx`는 decode가 바쁜 동안 유지되므로 `(prefill_sms, decode_sms)`는 `(92,16)`/`(54,54)`/`(16,92)`로 남고, `prefill_active_batch_size == 0`이어도 `is_part`는 `True`다. 집계기는 이 둘을 구별하지 않는다 — 구별은 `cohab_P_act`(PART∧decode-busy 중 prefill-active 비율)가 **별도 열로만** 제공한다.

**Q3. KILLED된 변이 `PART ≡ stream_index != 4`가 배제한 것**

> `residency_census.py:428-440` (Tier-1 변이 arm M4)
> `M4 PART := stream_index != 4 (ignores 108/0)` → **KILLED** (`R_cnt_all` 50.0 → 83.33)

합성 fixture에서 index 0의 realized 쌍은 `(108, 0)`이다. M4는 그것을 PART로 오분류한다. 즉 **M4가 배제한 것은 "인덱스(=target)로 PART를 정의하는 것", 특히 full-prefill `108/0` 상태를 PART에 넣는 오류**다. **M4는 prefill-busy/prefill-idle 구별에 대해서는 아무것도 말하지 않는다** — 그 축은 어떤 변이 arm도 건드리지 않았다.

⇒ Q2의 답이 "섞인다"이므로, 지시대로 분해 재집계를 했다.

### 2.4 S3 재집계 — PART∧prefill-busy vs PART∧prefill-idle

스크립트: `scratch/part_cohab_split.py`(맨 위에 "기술 집계 · 판정 아님" 명시). 방법은 **기존 판정 함수 import + `part_pred` 주입점 사용**이고, 새로 구현한 술어는 없다:

```
PART       ⟺ residency_census.is_part(rec)                                    (그대로)
PART_pbusy ⟺ is_part(rec) AND prefill_active_batch_size > 0
PART_pidle ⟺ is_part(rec) AND prefill_active_batch_size == 0
```

`prefill_active_batch_size`의 출처: `dual_worker.py:566-574`
`observe_scheduler`가 `self.prefill.active_batch = getattr(scheduler, "split_prefill_batch", None)` 로 채운다 ⇒ **스냅샷 시점에 prefill 배치가 in-flight였는가**의 직접 프록시이며, `adjust_stream_groups`가 보는 바로 그 값이다. ⚠ 스냅샷 격자(중앙값 gap **2.0 ms** [READ]) 위에서만 관측되므로 그보다 짧은 상태는 해상하지 못한다.

**자기검사** (`--selftest`, 전부 통과):
- 상류 `residency_census` Tier-1(M0 SURVIVED / M1–M4 KILLED) + Tier-1b(sum-not-max) 재호출
- **가법성**: 5 규약 전부에서 `PART_pbusy + PART_pidle == PART`(분모 동일)
- **해상력**: fixture에서 두 갈래 모두 비영(한쪽이 0이면 분해가 아무것도 해상하지 못한다는 뜻 — 교훈 232)
- **음성 대조군**: `p1_gates/gate2`의 no-split 파일 4건에서 **세 값 모두 정확히 0** (realized map `{0:'108/0', 3:'0/108'}`)

**[DERIVED · R-time-all]** 캠페인 집계 (분모 = 전 non-startup 스냅샷의 forward-gap 합, 초):

| 캠페인 | files | span 합(s) | PART% | PART∧prefill-busy% | PART∧prefill-idle% | PART 중 idle 비중 |
|---|---|---|---|---|---|---|
| `s2_sticky` (전부 ON) | 20 | 2663.30 | 78.53 | **7.57** | **70.96** | **90.4%** |
| `sticky_smoke` (OFF+ON 합) | 2 | 308.66 | 57.57 | 3.15 | 54.42 | 94.5% |

**[DERIVED · R-time-all]** 파일별 — OFF/ON 대조가 가장 선명한 곳:

| 파일 | realized index map | PART% | ∧pbusy% | ∧pidle% |
|---|---|---|---|---|
| `stk0_Ha8_d16_872800` (**sticky OFF**) | `{0:'108/0', 1:'92/16', 2:'0/108'}` | **5.57** | **3.07** | 2.50 |
| `stk1_Ha8_d16_872800` (**sticky ON**) | `{0:'108/0', 1:'92/16'}` | **86.22** | **3.19** | **83.03** |

> sticky가 더한 것은 거의 전부 **prefill-idle PART**다. **prefill-busy PART는 3.07 → 3.19%로 사실상 불변**이다. (이것은 두 arm·한 boot씩의 기술 집계이며 어떤 순위·인과 진술도 아니다.)

**[DERIVED · R-time-all]** `s2_sticky` 셀별(블록 범위):

| cell | prefill SM (target) | PART% | ∧pbusy% | ∧pidle% |
|---|---|---|---|---|
| d16 (b1–b8) | 92 | 68.97 – 84.14 | 0.91 – 2.85 | 67.50 – 83.05 |
| d54 (b1–b8) | 54 | 76.26 – 81.63 | 4.07 – 6.79 | 71.14 – 76.80 |
| d92 (b1–b4) | 16 | 69.38 – 79.82 | 21.13 – 27.23 | 46.77 – 58.69 |

다섯 규약 전부 및 파일별 전수는 `scratch/part_cohab_split_out.json`.

**이 결과가 의미를 바꾸는가.** 05절 재집계 표의 PART 수치는 **계산이 틀린 것이 아니다** — 정의대로다. 바뀌는 것은 **해석 라벨**이다: sticky ON 캠페인에서 "PART 체류"는 **동거 체류가 아니라 대부분 분할 유지 체류**다. `residency_census.py`가 이미 `cohab_P_act` 열로 같은 사실을 옆에 적어 두었다(s2_sticky **11.80%**, sticky_smoke **5.44%**, 균일 가중) — 본 재집계는 그것을 **시간 가중 축으로 옮겨** 같은 방향을 재확인한 것이다. E2C-21("shape A에서 sticky ON이 바꾸는 decode-busy 가중 시간의 86–89%가 prefill 유휴")과도 같은 방향이다.

### 2.5 S4 — 03절 비용 수치의 출처

| 수치 | 출처 문서 | 원 캠페인 | sticky |
|---|---|---|---|
| PART 체류 decode forward **granite 1.53–1.66× · zamba2 1.73–1.94×** (매칭 후 1.53–1.54 / 1.79–1.90) | `results/kernel_mech/PROMOTION_DRAFT_SWITCH_2026-08-22.md` 문장 5 (sha256 `b5c4ae3eec4491d3…`), 정본 `CONSENSUS.md` rev44 문단 | `p1_gates/gate2` HOLB 프로브 (jobs **874602 / 874633 / 874635**, granite·zamba2) | **OFF** |
| `other_stream_fw_ms` **PART 13.7–23.2 ms vs FULL 0.2–0.6** | 같은 문서 문장 5 | 같음 | **OFF** |

**sticky는 ON이 아니었다** [READ]:
1. `g2_holb_observer.sbatch`(sha256 `bb3b24cb8aee2498…`)에 `PDMUX_STICKY_PARTITION`이 **없다**(전수 grep).
2. HOLB 프로브 자체(`holb_probe.py`)는 `dev_tree_edits.md` 18–19번, **2026-08-06** 도입이다. sticky 3개 job(08-03~08-05)보다 **뒤**이고, 반대로 HOLB 캠페인은 sticky 플래그를 쓰지 않았다.
3. 결정적으로, 정본 문장 자신이 기전을 그렇게 적어 두었다 — "PART 진입 조건이 `split_prefill_batch is not None`이라 **prefill 동거가 구조적**".

⇒ **"기전이 동거"라는 해석에 prefill-idle PART는 섞이지 않았다.** sticky OFF에서 PART ⟺ prefill 배치 in-flight 이기 때문이다. 이 결론은 §2.4의 OFF arm 수치와도 정합한다(OFF arm의 PART 5.57% 중 3.07%p가 prefill-busy — 나머지 2.50%p는 격자 해상도 한계로 설명되는 잔차이며, ON arm의 83%p와 자릿수가 다르다).

**단, 세 가지 단서를 함께 적어야 한다:**
- (a) 그 캠페인은 `pdmux_a100_smoke.yml`(**`manual_divisions` 없음 → 자동 격자**)을 쓴다. target `74/34`는 `divide_sm(108, (8,·), 2)`의 산물이다. §6 I-1 참조.
- (b) 정본이 이미 적은 대로 `74/34`는 **target이지 realized가 아니다**(그 20개 파일에 realized telemetry 0건).
- (c) 정본이 이미 적은 대로 **두 모델 군집(granite/zamba2)을 합친 풀링값은 인용 금지**(모델≡노드≡job 앨리어스), **C2 값과 대조 금지**. 금지된 수치 자체는 여기 옮겨 적지 않는다 — `citation_stops.tsv` 참조.

### 2.6 S5 — SM 수 효과와 동거 효과를 분리할 재료가 있는가

**없다.** sticky 아카이브(22 파일, decode-busy 스냅샷 8,541건)에서 [DERIVED · 필드 비영률]:

| 필드 | 비영 비율 (s2_sticky / sticky_smoke) |
|---|---|
| `decode_last_tpot_ms` | **0.000% / 0.000%** |
| `decode_step_count` | **0.000% / 0.000%** |
| `worker_overlap_ratio` | **0.000% / 0.000%** |
| `decode_idle_ratio` · `prefill_idle_ratio` · `active_worker_leases` | **0.000% / 0.000%** |
| `measured_itl_ewma_ms` · `measured_itl_p95_ms` | 100% / 100% |

이유는 코드에 있다: 이 필드들의 기록자(`DecodeWorker.begin_step`/`finish_step`, arbiter lease)는 `dual_worker_enabled` 경로에서만 호출되는데, 세 job은 `PDMUX_DUAL_WORKER`를 설정하지 않았다(telemetry `architecture: "legacy"`).

유일한 후보 `measured_itl_ewma_ms`는 **쓸 수 없다**:
- α=0.85 EWMA(≈6–7 iteration 창)라 **상태 경계를 가로질러 섞인다**;
- 갱신부가 `0.0 < _dt < max(3·ema, 90ms)` 밴드 밖 표본을 **버린다** — 버려지는 것이 바로 전환/경계 표본이다;
- 측정 대상이 decode **forward 지속시간**이 아니라 **pdmux 루프 iteration 벽시계**다.

따라서 "PART∧prefill-idle step의 decode forward 지속시간"을 **FULL step·PART∧prefill-busy step과 나란히 놓을 수 있는 데이터는 이 아카이브에 존재하지 않는다.** 기술 통계를 내지 않았다.

**GPU 없이는 불가능한 것 — 질문형으로만:**
- sticky ON·`PDMUX_HOLB_PATH` 동시 활성 상태에서 PART 구간을 `prefill_active_batch_size`로 층화하면, PART∧prefill-idle의 decode forward 지속시간이 FULL과 같아지는가, 아니면 PART∧prefill-busy와 같아지는가?
- 같은 질문을 `PDMUX_GREEN_READOUT`으로 realized SM을 읽으면서 물으면, "SM 수"와 "동거"가 분리되는가?

(설계는 제시하지 않는다. 실행 여부·설계는 experiment-runner + claims-auditor + 사용자 결정 사항이다.)

---

## 3. 작업 B — M1: MuxWise는 경쟁자인가, 기판인가

### 3.1 upstream 이력

upstream `sgl-project/sglang`을 blobless clone(`b01961e295`, 2026-09-22 기준)해 확인했다.

`python/sglang/srt/multiplex/` 최초 도입: **`05ad28f25e`, 2025-10-28, 작성자 `ykcombat`, PR #11592 "[Feature] PD-Multiplexing Context and Scheduler."** → `ea39952797`로 revert → **`41efcaeb45`, 2025-11-01, PR #12275**로 재랜딩. `git tag --contains 41efcaeb45` ⇒ **`v0.5.10` 포함** [READ].

| 심볼 | 도입 PR | merge 커밋 | 날짜 | 작성자 | 도입 파일 |
|---|---|---|---|---|---|
| `ForwardMode.SPLIT_PREFILL` | **#7634** Layer-wise Prefill | `570d33437b` | 2025-07-16 | **jason-fxz** | `schedule_batch.py`, `forward_batch_info.py` |
| `split_forward_count`, `split_index` | **#7634** | `570d33437b` | 2025-07-16 | jason-fxz | `model_runner.py` (+4/+7 occurrence) |
| `forward_split_prefill` (dense 9종) | **#7634** | `570d33437b` | 2025-07-16 | jason-fxz | `models/{llama,qwen,qwen2,qwen2_moe,qwen3,qwen3_moe,gemma,gemma2,gemma3_causal}.py` |
| `create_greenctx_stream_by_value` | **#7649** CUDA Green Context Support | `1ebec1a8b0` | 2025-07-14 | ykcombat | `sgl-kernel/csrc/spatial/greenctx_stream.cu`, `sgl_kernel/spatial.py` |
| TP group switching (`pdmux_role_is_thread_local`의 upstream 대응부) | **#7653** | `d4d0c7c367` | 2025-07-14 | ykcombat | `distributed/parallel_state.py`, `server_args.py` |
| `initialize_stream_groups`, `divide_sm`, `get_arch_constraints` | **#11592** | `05ad28f25e` | 2025-10-28 | ykcombat | **`multiplex/pdmux_context.py` (신규 163줄)** |
| `event_loop_pdmux`, `adjust_stream_groups`, `split_prefill_batch`(스케줄러측) | **#11592** | `05ad28f25e` | 2025-10-28 | ykcombat | **`multiplex/multiplexing_mixin.py` (신규 209줄)** |
| PD-mux CUDA graph | **#11595** | `dd192a55f4` | 2025-11-13 | ykcombat | `cuda_graph_runner.py` |

**★독립 교차검증 (2026-09-22, `git log -S` 전수 추적, upstream clone)** — 위 표는 GitHub API의
PR 메타데이터로 만들었는데, 저장소 커밋 이력을 심볼별로 직접 파낸 결과가 **전부 일치**한다
[DERIVED · `git log -S<symbol> --pickaxe-regex`]:

- `event_loop_pdmux` · `split_prefill_batch` · `initialize_stream_groups` · `adjust_stream_groups`
  → 네 심볼 모두 `05ad28f25e`(#11592) → revert `ea39952797`(#12267) → `41efcaeb45`(#12275),
  작성자 `ykcombat`.
- `ForwardMode.SPLIT_PREFILL` · `split_forward_count` · `forward_split_prefill`
  → `570d33437b`(#7634), **git 커밋 작성자 문자열이 `Xiaoze Fan`** 이다.
  ★이것은 §3.1의 GitHub 프로필 조회보다 **강한 증거**다 — 신원이 프로필이 아니라
  **upstream 저장소의 커밋 author 필드 자체**에 박혀 있다.
- `create_greenctx_stream_by_value` → `1ebec1a8b0`(#7649, `ykcombat`)에 더해 후속 2건:
  **`603f5ce020`(#8701) "fix green context's incompatibility with `cuda < 12.4`"** ·
  **`445f9dca6e`(#9021) "Runtime check CUDA driver version to avoid unresolved green context symbols"**.
  ⇒ green context는 **CUDA ≥ 12.4**를 요구하고 upstream에 **드라이버 버전 런타임 검사**가 있다
  (작업 루트 `CLAUDE.md`의 "드라이버 CUDA ≥ 12.4" 조건의 upstream 출처 — 대여 GPU 사전점검 항목).

(날짜 표기 주의: 위 표의 날짜는 **API의 merge 시각(UTC)**, 이 목록은 `git log --date=short`의
커밋 날짜라 하루 어긋나 보이는 항목이 있다 — 같은 사건이다.)

**결정적 증거** — 이 기능들의 상위 추적 이슈 **#10813 "[Feature] Support PD-Multiplexing"**(작성자 GitHub `Raphael-Hao`)의 마지막 줄 [READ]:

> ### Related resources
> [Arxiv: Optimizing SLO-oriented LLM Serving with PD-Multiplexing](https://arxiv.org/abs/2504.14489)

그리고 arXiv `2504.14489` 의 현재(v3, 2026-02-07) 제목은 **"Towards High-Goodput LLM Serving with Prefill-decode Multiplexing"** = **MuxWise** 다(ASPLOS '26). [READ, arXiv API]

**인물 대조** (GitHub 공개 프로필 [READ] ↔ MuxWise 저자 목록):

| GitHub | 프로필 이름 | 소속 | MuxWise 저자 |
|---|---|---|---|
| `jason-fxz` | **Xiaoze Fan** | SJTU / UC Berkeley | **예 (5번째 저자, jasonfxz@sjtu.edu.cn)** |
| `Raphael-Hao` | **Weihao Cui** | SJTU | **예 (2번째 저자, weihao@sjtu.edu.cn)** |
| `ykcombat` | (비공개) | (비공개) | 미확인 — **추정하지 않는다** |

추가로, 우리 트리에 그대로 남아 있는 upstream 주석 `# TODO(jason-fxz): This is a temporary demo` 가 `adjust_stream_groups` 바로 위에 있다 — 즉 **우리가 "agnostic 정책"이라 부르는 decode-bs 임계 표 조회는, MuxWise 저자가 upstream에 "임시 데모"라고 적어 둔 코드**다.

또한 이슈 #10813의 preview 벤치(H200 + CodeLlama-34b-Instruct-hf, ShareGPT + LooGLE, ITL SLO 60 ms)는 로컬 아카이브 `/home/wonho/Experiments/muxwise/sglang-slo_config`(출처 `~/Downloads/muxwise.zip`, **출처 미검증**)의 `bench_pdmux.sh`·`loogle.yml`과 정확히 일치한다.

### 3.2 MuxWise 3 구성요소 vs upstream v0.5.10

| MuxWise 구성요소 | upstream v0.5.10 | 근거 |
|---|---|---|
| **bubble-less multiplex engine** (layer-wise prefill 발사) | **있음** | `ForwardMode.SPLIT_PREFILL` + `split_forward_count`/`split_index`(#7634) + `event_loop_pdmux`의 split-prefill 루프(#11592) + green-context 스트림(#7649) + pdmux cudagraph(#11595). 단 **query-based synchronization(§3.2 "periodically polls CUDA events")** 은 upstream 코드에서 확인하지 못했다 — **불충분** |
| **contention-tolerant estimator** (solo-run predictor + contention guard, offline profiling) | **없음** | upstream `multiplex/` 2파일(`multiplexing_mixin.py` 221줄 + `pdmux_context.py` 164줄) 전체에 predictor·estimator·slowdown factor·offline profile 관련 심볼이 **0건**. |
| **SLO-aware dispatcher** (`N_PL` 계산, 병합 후 재분할, 선점) | **없음** | `adjust_stream_groups`는 `decode_bs >= threshold` 표 조회뿐이고 위에 `TODO(...): temporary demo`가 붙어 있다. `N_PL`·SLO·선점·재분할 심볼 0건. |

**MuxWise 공개 코드 저장소**: 공식 저장소를 이 감사에서는 **확인하지 못했다**. 로컬 아카이브 `~/Experiments/muxwise/sglang-slo_config`(sglang **v0.5.3rc0** 기반, `multiplex/{multiplexing.py 195줄, pdmux_context.py 163줄}`)가 있으나 **출처 미검증**이다. 그 아카이브를 upstream v0.5.10과 diff한 결과 [DERIVED]:
- `pdmux_context.py` 차이 = 3줄(import 위치·에러 문구) — **기능 동일**
- `multiplexing.py` vs `multiplexing_mixin.py` 차이 = 타입 힌트·`init_pdmux` 분리뿐 — **정책 로직 동일**
- 이름이 `slo_config`임에도 **SLO·estimator·contention·N_PL 심볼이 0건**

⇒ 이 아카이브도 **engine 부분만** 담고 있다. estimator/dispatcher는 공개 코드에서 확인되지 않는다.

### 3.3 engine-port가 upstream 위에 더한 것

`dev_tree_edits.md`(sha256 `fa983e563734bf21…`) 항목 번호로:

| 항목 | 내용 |
|---|---|
| 1–5, **P1.5** | Zamba2 config/model 신규 + **`Zamba2ForCausalLM.forward_split_prefill`**(hybrid SPLIT_PREFILL 활성화) + race-safe decode 루프 |
| **6** | `models/nemotron_h.py` — `forward_split_prefill` 추가 |
| 7 | `triton_backend.py` v_head_dim (Zamba2 부팅) |
| **8** | `models/falcon_h1.py` — `forward_split_prefill` 추가 |
| **9** | `models/granitemoehybrid.py` — `forward_split_prefill` + green-ctx decode-SM pin + per-layer-type timing |
| 10 | `PDMUX_LA_SM_MAP` (layer-aware, 死) |
| **11–12** | `multiplex/dual_worker.py` 신규 + mixin 배선 (`PDMUX_DUAL_WORKER`) |
| 13–16 | pure-Mamba2 음성 대조군 |
| 17 | Zamba2 per-layer-type timing 재작업 |
| 18–19 | HOLB probe + scheduler hook |
| 20–21 | chunked-prefill probe + hook |
| **22–23** | R2 admission latch 수정 · true-dual split-prefill ownership 수정 |
| **24** | hybrid 3종 모델 파일을 sync+manifest에 편입 |
| (항목 번호 없음) | **`PDMUX_STICKY_PARTITION`**(f60a128), **R2 정책층**(`controller.py`, `profile.py`), **SLO 컨트롤러**(`_slo_decide_idx`), **telemetry**(`telemetry.py`), **green read-out**(27bbae7) |

`pdmux_context.py`는 **손대지 않았다** — 설치본 sha256 `b4e6ff1aa10a19f9…` 가 upstream `v0.5.10` 원본과 **byte-exact 일치** [DERIVED].

**02절 "agnostic PD-mux가 fused를 이긴다"는 어느 코드 경로에서 측정됐는가** [READ]:

| 캠페인 | 실행 스크립트 | pdmux config | 경로 |
|---|---|---|---|
| P1.7 4모델 (no-cudagraph) | `triage/p1_7_bench_one.sbatch:35` (sha256 `7b3451eba3a78395…`) | `pdmux_a100_smoke.yml` | **자동 격자 (`divide_sm`)** |
| P1-opint 운영점 (cudagraph-ON, jobs 873944/873945) | `results/p1_opint/p1op_run.sbatch:47,55` | `pdmux_a100_smoke.yml` | **자동 격자 (`divide_sm`)** |

즉 02절 주장을 떠받치는 **두 캠페인 모두** upstream `adjust_stream_groups`의 **`manual_divisions`가 아닌 else 분지**(`decode_bs * (n-2) // decode_bs_divisor`)에서 측정됐다 — `multiplexing_mixin.py:1183-1197`. §6 I-1 참조.

### 3.4 M1 결론

> **(a) upstream pdmux는 MuxWise의 engine 부분이다. 우리의 agnostic은 MuxWise engine에서 estimator와 dispatcher를 뺀 것 + hybrid 패치다.**

근거: §3.1(도입 PR·작성자·추적 이슈의 MuxWise arXiv 명시·저자 신원 2/2 일치) + §3.2(3 구성요소 중 engine만 존재) + §3.3(engine-port가 더한 것은 hybrid `forward_split_prefill`·R2/SLO/sticky/telemetry이며 MuxWise의 estimator/dispatcher가 아니다).

**이 판정의 한계(명시):** (i) `ykcombat`의 신원은 미확인이며 추정하지 않았다 — 판정은 `jason-fxz`/`Raphael-Hao` 2명과 추적 이슈의 명시적 인용만으로도 성립한다. (ii) "upstream 코드가 논문 코드에서 유래했다"가 아니라 **"같은 저자들이 같은 시스템의 engine 층을 upstream에 올렸다"** 까지가 확인된 것이다.

⇒ **M2–M6의 판정 문구는 MuxWise를 "경쟁 시스템"이 아니라 "기판의 출처"로 쓴다.**

---

## 4. 작업 C — 선행 대조 (M2–M6)

### 4.0 판본 — ★2차 통과, 지정 PDF 확보

1차 통과에서는 지정된 두 PDF를 머신에서 찾지 못해 **멈춤 조건이 발동**했고, 공개 preprint를 명시적 대체 출처로 써서 잠정 판정했다. 사용자가 `/home/wonho/Experiments/KISTI/Papers/`에 원본을 놓아 주어 **camera-ready로 전수 재확인했다.** 아래 §4.1–§4.6은 전부 **camera-ready 기준**이다.

| 논문 | camera-ready 파일 | sha256 | 대조한 preprint | 판본 차이 |
|---|---|---|---|---|
| MuxWise | `Papers/Muxwise.pdf` | `0b2062bf9cbd90b0…` | arXiv:2504.14489v3 (`9198bd6c…`) | **인용한 8개 문장 전부 동일**(§3.3.1 경합 범위·§3.4.2 `N_PL` 식·§4.2.3 utilization 정의·프로파일 1주일·SGLang 0.4.10post2·8×A100·TBT 50/100 ms·§3.4.1 "prefill 층 지연 > decode iteration") ⇒ **1차 판정 변경 없음** |
| Bullet | `Papers/Bullet.pdf` | `6a0605e037b2a60d…` | arXiv:2504.19516v4 (`e0b9274a…`) | **제목이 다르고 내용도 달라졌다.** CR 제목 = "Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration". ★**Table 5에 `Resource Re-config (μs)` 행 신설**(preprint엔 없음) · **§4.2.1 Single GPU Performance 신설** · §3.2.3 Contention Modeling 확장 · 그림 번호 17→20, 18→21 ⇒ **M4·M5·M6 판정이 바뀐다** |

★ **1차 통과에서 내가 preprint 근거로 쓴 다음 두 문장은 camera-ready 기준으로 틀렸고, 아래에서 정정한다:**
1. "Bullet은 재구성 오버헤드 **측정값이 없다**(negligible 서술뿐)" → **틀림.** CR Table 5에 `Resource Re-config (μs) Mean 4.1 / Std. 0.79 / P90 4.2 / P99 5.9`가 있다.
2. "우리만 단일 GPU이고 선행 둘은 8-GPU TP다" → **Bullet에 대해 틀림.** CR §4.2.1이 **A100 1장·Llama3.1-8B** 평가를 담는다.

### 4.1 M2 — "PD-mux 선행 중 residency를 측정량으로 보고한 논문은 확인되지 않는다"

| 좌표계 문장 | 원문 위치 | 원문 요지 | 판정 |
|---|---|---|---|
| 선행은 residency를 **측정량으로** 보고하지 않았다 | MuxWise §4.2.3, Table 5 | GPU utilization의 정의가 *"an aggregated metric reported by NVIDIA Nsight Systems, that reflects the fraction of active SMs as well as the utilization of intra-SM resources"* — **SM 활성도**이지 co-running 시간 비율도, 분할 상태 체류 시간도 아니다 | **지지** |
| 〃 | MuxWise §3.3.1 / Fig. 11 | 축이 (decode에 준 SM 수) × (decode slowdown)이다. 분할 **구성**별 느려짐이지 각 구성의 **체류 시간**이 아니다 | **지지** |
| 〃 | MuxWise 전문 grep | `fraction/portion/percentage of time`·`time spent`·`residency` 0건. 유일한 `co-run` 문장은 *"[the prefill] typically co-runs with tens of [decode iterations]"* 로 **정성 서술** | **지지** |
| 〃 | **Bullet §4.3.1 / Fig. 20a** | *"the number of SMs provisioned for the prefill phase, **with each bar showing the SM count and duration**"* — **구성별 지속시간 막대의 타임라인**. 정의된 집계 통계·분모·분포는 없으나 **"분할 상태 체류 시간"을 그림으로 보여준다** | **반박(부분)** |
| 〃 | Bullet §4.3.2 / Fig. 21 | *"From 0s to 27s, when the system concurrently handles prefill and decode requests, Bullet sustains an average of 86.2% active SM cycles"* — **co-run 구간을 전제로 잡은 뒤** 그 안의 SM 활성도를 보고한다. co-run이 전체의 몇 %인지는 보고하지 않는다 | **지지** |

**판정: 부분 (1차와 동일, 그림 번호만 갱신).**
- **지지되는 형태**: "**정의·분모·규약을 갖춘 집계 추정량으로서** residency를 보고한 PD-mux 선행은 확인되지 않는다."
- **쓰면 안 되는 형태**: "체류 시간과 같은 양을 아무도 보여주지 않았다" — **Bullet Fig. 20a가 반례**다.
- 04절 공백 주장은 위의 좁은 형태로만 쓴다.

### 4.2 M3 — "선행은 layer 단위 prefill 진행도를 제어하므로 체류 지속을 암묵적으로 조절한다"

| 논문 | 결정하는 변수 |
|---|---|
| **MuxWise** | (1) **partition 크기** — decode에 best-fit SM, 나머지 prefill (§3.4.2) · (2) **prefill 진행도** — `𝑁_PL = ⌈(𝑇_d × 𝑁_T)/𝑇_P⌉` 층 발사 (§3.4.2) · (3) **병합 시점** — query-based sync가 CUDA event를 폴링, 완료 즉시 decode batch에 merge (§3.2) · (4) **선점** — P2가 P1을 선점, **비재귀**, 선점 시에만 SLO 확인, **optional** (§3.4.2) · admission은 제어 변수로 등장하지 않는다 |
| **Bullet** | (1) **partition 크기** — `SetBalancedSM`/`ReduceDecodeSM`/`ReducePrefillSM` (Algorithm 1 line 15–19), libsmctrl 마스크로 즉시 반영 · (2) **prefill 진행도** — 시스템 상태 `PS`에 **실행 층수 `L_exe`** 가 들어가고 Algorithm 1이 `next_tasks, _, L_exe + L_step` 을 **반환**한다 = **이번 step에 돌릴 층수를 정한다** (§3.3.2) · (3) **admission 순서** — `SortByLeastEstimLatency(Q)`(TTFT SLO를 깨지 않는 한도에서 재정렬) + `ArithInten(...) < peak` 까지 배치 (line 5, 12–13) · (4) **선점** — 고부하 시 **decode를 일시 중단**(Fig. 12-➁), 단 TPOT SLO를 지키는 한도 |

**판정: 지지 (1차보다 강해짐).**
- MuxWise 원문이 직접 근거를 준다 — *"It only requires that the predicted latency of the launched prefill layers exceeds that of the corresponding decode iteration, ensuring full utilization of the allocated compute resources"* (§3.4.1). **prefill 작업 단위를 decode 1 iteration을 덮도록 고르는 것** = 동거 구간 길이를 직접 정하는 행위.
- Bullet도 camera-ready에서 **층수를 반환값으로 내놓는다**(Algorithm 1 line 20) ⇒ 두 시스템 **모두** layer 단위 진행도를 제어 변수로 갖는다.
- 단 **어느 쪽도 그것을 residency라는 이름의 양으로 다루지 않는다** — 이 점이 M2와 함께 04절의 논지를 만든다.

### 4.3 M4 — "PART가 느린 기전은 SM 수가 아니라 동거다"

| 좌표계 문장 | 원문 위치 | 원문 요지 | 판정 |
|---|---|---|---|
| 동거가 기전이라는 것은 우리 발견이다 | **MuxWise §3.3.1 / Fig. 11** | *"While GreenContext supports precise compute resource allocation, it cannot manage memory or network bandwidth"* + 분할 구성을 고정한 채 Llama-8B/70B를 8×A100·8×H100에서 프로파일(**1주일** 소요)한 결과 *"contention-induced slowdown ranges from nearly zero to about 30% across different partition configurations and GPUs"* | **반박** |
| 〃 | MuxWise §3.3.2·§3.4.1 | 그 예측 불가능성 때문에 **contention guard**(decode **전용** 최대 slowdown factor)를 도입 — 즉 "동거로 느려지는 쪽은 decode"라는 것이 설계 전제다 | **반박** |
| 〃 | **Bullet §3.2.3 (camera-ready 확장)** | *"When kernels execute on **isolated SMs** to prevent compute contention, **memory subsystem and network contention persist**"* + *"decode kernels exhibit **higher sensitivity to contention than prefill kernels**"* + sole-run 예측기(contention 무시)는 **decode 지연 예측 정확도를 크게 떨어뜨린다**(Fig. 11) | **반박** |
| 〃 | Bullet §3.4.2 | MPS+Green Context 기각 사유는 **메모리 오버헤드**(700MB/4 정책)이고 경합 기전 보고가 아니다 | **무관** |

**판정: 부분 — 좁혀야 한다. 1차보다 더 세게 좁혀야 한다.**

1차 통과에서는 MuxWise 한 편만 선행 보고로 잡았으나, **camera-ready 기준으로는 두 편 모두** "SM을 고정/격리해도 동거 때문에 decode가 느려진다"를 **dense 모델에서 이미 보고한다.** 따라서 우리 문장은 다음으로 좁아진다:

1. **hybrid(attention+SSM)에서의 재확인** — 선행의 측정은 dense Llama(+Qwen MoE)뿐이다.
2. **크기 차이** — MuxWise 보고 범위는 **≈0–30% 느려짐**인데 우리 03절 값은 **53–94%**(1.53–1.94×)다.
3. ★**두 수를 직접 비교하지 말 것**: 기판(1 GPU vs 8 GPU TP8)·모델·분할 격자·측정 대상(우리 `other_stream_fw_ms` vs 그들의 iteration slowdown)이 전부 다르고, 우리 값에는 정본이 이미 건 단서(target≠realized, 풀링값 인용 금지, C2 대조 금지)가 붙어 있다.
4. 선행이 **하지 않은 것**은 남는다 — 두 편 모두 그 경합을 **예측·회피 대상**으로 다루고, **분할 상태에 얼마나 오래 앉아 있는가**를 양으로 세지 않는다.

### 4.4 M5 — Bullet의 분할 메커니즘과 재구성 오버헤드

| 항목 | camera-ready 확인 내용 | 위치 |
|---|---|---|
| 분할 기구 | **libsmctrl SM 마스킹** — *"we utilize the `libsmctrl_set_stream_mask` API to modify the metadata of CUDA stream to constrain all subsequent kernel executions to a specified subset of SMs"*; prefill/decode 엔진이 각자 CUDA stream 하나를 만들고, 스케줄러가 repartition 명령을 내리면 **즉시 그 스트림을 재설정**한다. 구현은 SGLang **v0.4.6** + PyTorch 2.6.0 + 4100줄 Python + modified libsmctrl, **MPS enabled for spatial sharing** | §3.4.2, §3.5.1 |
| 입도 | 코드상 **TPC 단위**(`libsmctrl_get_tpc_info_cuda`, `TOTAL_TPCS`) — A100 기준 TPC = 2 SM | `BulletServe/python/sglang/srt/bullet/sm_controller.py` |
| Green Context | **명시적 기각** — *"While MPS with CUDA Green Context supports SM partitioning, its memory overhead exceeds 700MB for only 4 static policies in LLM serving, rendering it impractical for fine-grained, dynamic control"* | §3.4.2 |
| 이식성 언급 | AMD `hipExtStreamCreateWithCUMask` 를 대안으로 들며 *"which can be utilized to **pre-create multiple streams with masks** to mitigate SM partitioning overhead"* — ★**우리 green-context stream group 사전 생성 설계와 같은 발상** | §3.4.2 |
| "SM 부분 공유"의 의미 | *"a smooth transition between **co-running prefill/decode** and **decode-only**. During reconfiguration, Bullet [eliminates] inter-phase synchronization by **partially sharing SMs between phases** rather than idling unused resources"* — 두 phase의 SM 집합이 **의도적으로 겹칠 수 있다**(완전 분리가 아니다). layer-wise prefill이 그 간섭을 *"minimal, predictable regions"* 로 가둔다 | §3.4.2 근방 |
| **재구성 오버헤드 측정값** | ★**있다** — Table 5 `Resource Re-config (μs)`: **Mean 4.1 / Std. 0.79 / P90 4.2 / P99 5.9 μs**. 본문도 *"Section 4.3.3 validates that this on-demand setting adds only **microsecond-level runtime overhead and zero additional memory footprint**"* (§3.4.2). 같은 표의 다른 두 행: Metadata Send/Recv **0.21 ms** mean(P99 1.54), Performance Predict **10.2 μs** mean(P99 25.8) | Table 5, §4.3.3, §3.4.2 |

**03절 "전환 ≤ 0.04 ms"와 비교 가능한가 — 조건부로만.**

| | 우리 `s ≤ 0.04 ms` | Bullet `Resource Re-config 4.1 μs` |
|---|---|---|
| 측정 층 | **device-level** 간극 귀속 상한 | **CPU 측 오버헤드**(§4.3.3 본문이 *"quantifies the CPU overhead"*라고 명시) |
| 대상 행위 | **사전 생성된** green-context stream group 사이의 **인덱스 교체**(`set_current_stream_idx`) | 기존 스트림의 **마스크 메타데이터 쓰기**(`libsmctrl_set_stream_mask`) |
| 도출 방식 | **가법성 가정** + agnostic `adjust_stream_groups` 경로 한정 | 직접 계측(Mean/Std./P90/P99) |
| 크기 | ≤ 40 μs | 4.1 μs (P99 5.9) |

⇒ **"비교할 수가 상대편에 없다"는 1차 판정은 철회한다.** 두 값은 **같은 자릿수 범위(μs)** 이지만 **다른 양**이다 — 우리 값은 device-level 상한, 그들 값은 CPU-side 계측이며, 기구 자체도 다르다. 따라서:
- **쓸 수 있는 문장**: "두 시스템 모두 전환/재구성 비용을 **μs 규모**로 보고한다 — 기구는 다르다(green-context 인덱스 교체 vs libsmctrl 마스크 쓰기), 측정 층도 다르다(device-level 상한 vs CPU-side)."
- **쓰면 안 되는 문장**: "우리 전환이 Bullet보다 비싸다/싸다" (층이 다르다), 그리고 정본 금지문장 #18 — **`s`를 2자리 이상으로 표기하지 말 것**.

### 4.5 M6 — 평가 조건 비교표 (camera-ready 기준)

| | **우리 (engine-port)** | **MuxWise (ASPLOS '26)** | **Bullet (ASPLOS '26)** |
|---|---|---|---|
| GPU | **A100-80GB** | A100-80GB (주), H100-SXM5-80GB, H200-SXM5-141GB | A100-80GB(108 SM), H100(132 SM), **H20(78 SM)** |
| 개수 | **1장 (108 SM)** | **8장** (NVLINK 600 GB/s) | **1장 (§4.2.1) 과 8장 (§4.2.2) 둘 다** |
| TP | **1** | **8** (고정) | 단일 GPU는 1, multi-GPU는 TP |
| 모델 | **hybrid 4종**(NemotronH·Zamba2·Falcon-H1·Granite-4) + dense 대조(Qwen2.5-7B) | Llama-8B, Llama-70B (**dense**), Qwen3-235B-A22B (**MoE**) | Llama3.1-8B/70B (**dense**), Qwen3-235B-A22B-FP8 (**MoE**), Qwen3-32B(경합 프로파일) |
| 아키텍처 | **dense + SSM hybrid** | dense + MoE. **hybrid 없음** | dense + MoE. **hybrid 없음** |
| 엔진·버전 | **SGLang v0.5.10** + engine-port 패치 | SGLang **0.4.10post2**, PyTorch 2.6.0, CUDA 12.8, driver 570.124.06 | SGLang **v0.4.6** + FlashInfer v0.2.7, PyTorch 2.6.0, CUDA 12.4, +4100줄 |
| 분할 기구 | **CUDA Green Context** | **CUDA Green Context** (§3.3.1이 명시) | **libsmctrl TPC 마스크 + MPS** (Green Context 명시 기각) |
| 워크로드 | ShareGPT (변화 trace rate 3↔12 포함) | Conversation / Tool&Agent 트레이스, Poisson 재타이밍 | ShareGPT, Azure-Code, arXiv-Summary, Poisson |
| SLO regime | TTFT ≤ 3 s ∧ **요청내 token-ITL p95 ≤ 60 ms** (게이트 #4); tight chat 300/50 ms | **TBT 99%-ile**, **50 ms**(Llama3-8B) / **100 ms**(Llama3-70B). TTFT엔 SLO 없이 P99만 보고 | **normalized TTFT 3.0/1.5/1.5 ms** (입력 토큰당) ∧ **TPOT 150/200/175 ms**, P90 기준 (Table 4) |
| 베이스라인 | fused(plain), chunked, PD-mux 정책들 | chunked-prefill(SGLang), NanoFlow, LoongServe, SGLang-PD | SGLang-1024/2048, vLLM v0.8.5, Nanoflow, xPyD 분리형(+MoonCake) |

**판정: 확인 — 단 1차 판정의 한 줄은 정정한다.**
- **정정**: "우리만 단일 GPU다"는 **거짓**이다. Bullet은 **A100 1장 평가를 갖고 있다**(§4.2.1, Llama3.1-8B, 세 워크로드).
- **여전히 우리만인 것**: **hybrid(attention+SSM) 모델군**. 두 선행 다 dense+MoE만 평가한다.
- **가장 큰 축 차이는 SLO regime이다**: Bullet의 decode SLO는 **TPOT 150–200 ms**로, 우리 **ITL p95 60 ms**·MuxWise **TBT 50/100 ms**보다 훨씬 느슨하다. 그리고 TTFT 술어의 정의 자체가 다르다(우리·MuxWise = 절대 시간, Bullet = **입력 토큰당 정규화 시간**). ⇒ **세 시스템의 goodput·SLO attainment 수치는 어느 방향으로도 가로질러 비교할 수 없다.**

### 4.6 ★범위 밖 신규 — Bullet §4.4가 HE0의 정면 대조 주장을 담고 있다 (기록만)

과제가 지시한 M2–M6에는 없지만, camera-ready를 읽는 동안 **정본 HE0와 직접 맞닿는 선행 주장**을 발견했으므로 **기록만** 한다. 판정하지 않는다.

**Bullet §4.4 Sensitivity Studies** [READ]:
> *"we run the workloads under fixed SM configurations for prefill and allow decode to use all SMs, which mimics static sharing systems like MuxServe. … The SM-108 configuration (no partitioning) demonstrates severe imbalance in the Azure-Code workload. … SM-108 suffers 1.20× higher TTFT on average and 1.19× worse P90 tail latency compared to Bullet, ultimately reducing throughput and SLO attainment by 13%. Smaller static partitions, like 84 SMs, prove even more problematic … **Therefore, there is no optimal fixed SM allocation**, as smaller partitions improve TPOT but degrade TTFT and introduce tail latency violations, and vice versa."*

정본 **HE0**는 "단일-GPU 동적 제어는 best decode-heavy static을 못 넘는다"(n≥4, 5.4σ, 관대·tight SLO 양쪽)이다. 문면상 **정면으로 반대 방향**이다. **그러나 다음 차이들이 확인되지 않은 채로 남아 있으므로 어느 쪽 손도 들어 줄 수 없다** — 이것이 이 항목을 "기록만"으로 두는 이유다:

| 축 | 우리 HE0 | Bullet §4.4 |
|---|---|---|
| static arm의 정의 | **예산 구속 분리 격자**(prefill_SM + decode_SM = 108) | **prefill만 고정하고 decode는 全 SM 사용 허용** — 겹치는 할당(부분 공유) |
| 비교 기준 | **best** decode-heavy static (튜닝된 앵커) | SM-108·SM-84 등 **열거된 고정점** (best static을 찾는 절차가 서술되지 않음) |
| 판정 술어 | TTFT ≤ SLO ∧ 요청내 token-ITL **p95** ≤ SLO | 정규화 TTFT ∧ TPOT, **P90** |
| SLO 엄격도 | ITL p95 60 ms (tight는 50 ms) | **TPOT 150–200 ms** |
| 동적 제어의 내용 | **분할 인덱스만** 바꾸는 단일 레버 | 분할 + **요청 재정렬** + **decode 일시중단** + 예측기 (§4.5 ablation이 `w/ Partition` 단독은 TTFT를 **악화**시킨다고 적는다) |
| 기전 | positioning + entanglement (decode 굶김 → ITL↑ → batch 정체 → admission 차단) | 명시 안 됨 |

★특히 Bullet **§4.5 ablation**의 `w/ Partition`(분할만, 스케줄러 없음) arm이 *"improves TPOT for Azure-Code but suffers unacceptable TTFT degradation from its inability to reorder pending requests"* 라고 적은 것은, **"분할 레버 단독으로는 안 된다"** 는 우리 결론과 **같은 방향**이다. 즉 두 결과가 충돌하는지 아니면 "동적 분할 단독 vs 동적 분할+스케줄링"이라는 **다른 것을 재고 있는지**가 미해결이다.

⇒ **이 항목은 claims-auditor 회부 대상이다.** 본 감사는 새 성능 판정을 내지 않으므로 여기서 멈춘다. HE0의 등급·문구는 **불변**이다.

## 5. 작업 D — v2 01절 코드 인용 문장 재확인

### D1. "`adjust_stream_groups`는 decode batch size와 prefill in-flight 여부를 입력으로 받는다. decode_bs가 커지면 다른 division으로 drift한다."

**부분.** 앞 문장은 맞다(`self.running_batch`의 공백 여부·`batch_size()`와 `self.split_prefill_batch`를 읽는다 — 인자가 아니라 스케줄러 속성이라는 점만 부정확). 뒤 문장에는 **빠진 단서가 있다**:

> `multiplexing_mixin.py:1183-1197`
> ```python
> else:
>     decode_bs = self.running_batch.batch_size()
>     manual_divisions = self.pdmux_config.manual_divisions
>     if manual_divisions:
>         for i in range(len(manual_divisions)):
>             _, _, threshold = manual_divisions[i]
>             if decode_bs >= threshold:
>                 stream_idx = i + 1
>     else:
>         stream_idx = max(1, min(self.real_sm_group_num - 2,
>                                 decode_bs * (self.real_sm_group_num - 2)
>                                 // self.pdmux_config.decode_bs_divisor))
> ```

이 `else`는 **`sticky_partition_enabled and _sticky_fixed_idx is not None`이 아닐 때만** 실행된다(`:1169-1182`). 즉:
- **sticky OFF** → decode_bs drift **있음** (우리 캠페인 대부분)
- **sticky ON + `PDMUX_R2_POLICY=fixed`** → `stream_idx = self._sticky_fixed_idx` **상수, drift 없음**

코드 주석이 그 이유까지 적어 둔다 — guard-satisfier 행을 가진 config(`pdmux_e1_d54.yml`의 `[64,44,48]`)가 `decode_bs`가 48을 넘는 순간 guard 행으로 표류하는 것을 막기 위함. **01절 문장은 sticky 단서를 붙여야 한다.**

### D2. "한 셀에서 실제 division 점유가 decode-active 시간의 4–19%에 그쳤다" (job 872077)

**확인.** 출처가 두 곳에서 일치한다.

1. **코드 주석 자신** — `multiplexing_mixin.py:361-373`(현재) / `:209-221`(f60a128):
   > *"the cell's decode division was realized over only 4-19% of decode-active time in job 872077 (T8 d16 0.038 -> d54 0.093, Ha8 0.104 -> 0.187); the remaining 81-96% ran unpartitioned at 108 SM."*
2. **`sticky_smoke` 결과 파일**(`stksmoke_Ha8_d16_872800_result.txt`, sha256 `e57b1a98192c80de…`) [READ]:
   ```
   E1_DECODE_REALIZED sticky=0 scope=ALL        D=16 frac=0.0805  t_decode_active=75.9s  hist=[D108:69.8s, D16:6.1s]
   E1_DECODE_REALIZED sticky=0 scope=PROBE_ONLY D=16 frac=0.0839  t_decode_active=72.8s  hist=[D108:66.7s, D16:6.1s]
   E1_DECODE_REALIZED sticky=1 scope=ALL        D=16 frac=1.0000  t_decode_active=171.6s hist=[D16:171.6s]
   ```

**단서 2건:** (a) 872077 자체의 원 산출물은 `results/s8_frontier/` 쪽이고 본 감사는 **인용 문서와 재현 셀**까지만 확인했다. (b) 4–19%는 `E1_DECODE_REALIZED`(decode-active **시간 가중**) 규약의 값이다 — 규약명 없이 인용하면 E2C-8′ 위반이다.

### D3. "`update_split_prefill_batch` 앞의 R2 admission hold(`r2_admission_limited`)는 prefill 배치 형성 자체를 보류한다. 이것은 스케줄링층 노브다."

**부분 — 중대한 단서가 빠져 있다.**

*보류한다*는 맞다:
> `multiplexing_mixin.py:1218-1231`
> ```python
> def update_split_prefill_batch(self, sm_count):
>     if self.split_prefill_batch: return False
>     if getattr(self, "r2_admission_limited", False) and self._r2_admission_holds():
>         return False
>     ...
>     batch = self.get_new_batch_prefill()
> ```
> `get_new_batch_prefill()`에 도달하기 **전에** 반환하므로, 배치는 **만들어지지 않는다**(대기열에서 뽑히지도 않는다).

`_r2_admission_holds()`가 보류를 유지하는 조건 (`:2045-2070`): **`running_batch`가 비어 있지 않으면 `True`(계속 보류)**. 비면 래치를 풀고 컨트롤러의 사본까지 `release_admission_limit()`으로 풀며 `r2_admission_released` telemetry를 낸다. 즉 "decode가 빠질 시간을 주기 위한 한도"이고 **decode가 다 빠지면 해제**다.

**빠진 단서 — 누가 래치를 세우는가:**
- `r2_admission_limited`는 `_r2_decide_idx`가 `decision.admission_limited`로부터만 쓴다.
- **`FixedPolicy.decide`는 `admission_limited`를 넘기지 않는다** — `SplitDecision`의 dataclass 기본값 `False`가 그대로 나간다 (`controller.py:88-101`).
- 세우는 것은 **`CoarseGrainedController`(= generic / hybrid 정책)** 뿐: `admission_limited = self.overload_streak >= 2` (`controller.py:320-329`).

⇒ **sticky / E1 / S2 / λ0 등 `PDMUX_R2_POLICY=fixed` 캠페인 전부에서 이 노브는 한 번도 발화하지 않았다.** (`pdmux_e1_d92.yml` 주석이 이미 같은 사실을 적어 두었다.)

**"스케줄링층 부 개입"이라는 표현은 맞는가.** 정본 `CONSENSUS.md`에서 R2는 **정책층 런타임 모드**(`PDMUX_R2_POLICY=fixed|generic|hybrid`, `controller.py`의 `SplitDecision`을 통해 **분할 인덱스**를 정하는 층)로 정의된다. admission hold는 그 `SplitDecision`이 **분할 인덱스 외에 하나 더 들고 나오는 부수 효과**이며, 그것이 닿는 곳은 prefill 배치 형성 = 스케줄링층이다. 그러므로 **"R2 정책층이 스케줄링층에 내는 부수 개입"이라는 서술은 정확하다.** 다만 **"R2 = 스케줄링층 노브"로 읽히게 쓰면 부정확**하고, **dynamic 정책 전용**이라는 단서가 반드시 붙어야 한다.

### D4. "dual-worker 경로도 admission 구간에 개입한다"

**반박.**

> `dual_worker.py:87-97`
> ```python
> def begin_admission(self, now=None):
>     self._admission_started_at = time.perf_counter() if now is None else now
>
> def finish_admission(self, now=None):
>     if self._admission_started_at is None: return
>     now = time.perf_counter() if now is None else now
>     self.snapshot.admission_latency_ms = max(0.0, (now - self._admission_started_at) * 1000.0)
>     self._admission_started_at = None
> ```

이 두 메서드는 **`get_new_batch_prefill()`을 감싼 스톱워치**다(`multiplexing_mixin.py:1224-1229`). 하는 일은 `admission_latency_ms` 기록 하나뿐이고, **반환값도 없고, 아무것도 막거나 미루거나 순서를 바꾸지 않는다.** 게다가 둘 다 `if getattr(self, "dual_worker_enabled", False):` 안에 있으므로 `PDMUX_DUAL_WORKER`가 꺼진 캠페인에서는 **호출조차 되지 않는다**.

**정확한 문장:** "dual-worker 경로는 admission 구간을 **계측한다**(`admission_latency_ms`). 개입하지 않는다."

(관련: `observe_scheduler`가 `prefill.snapshot.admission_blocked` / `admission_block_reason`을 **채우는데**, 이것도 **관측 라벨**이고 제어가 아니다.)

### D5. "PART에서 prefill이 받는 SM 몫이 체류 길이를 정한다"

**부분 — 이것은 코드 사실이 아니라 기전 가설이라는 지적이 맞다. 지지 데이터는 아카이브에 존재한다. 단 인용 금지 캠페인이다.**

**왜 코드 사실이 아닌가.** sticky OFF에서 PART 체류의 **시작과 끝**은 `split_prefill_batch`의 생애주기가 정하고(`:1169-1182`), prefill SM 몫은 그 생애주기의 **길이**에 영향을 줄 뿐이다. 코드는 SM 몫과 체류 길이 사이에 어떤 식도 갖고 있지 않다.

**지지 데이터 — `results/s8p_prefill/`** (prefill 축 SM 스윕: **decode 16 SM 고정**, prefill 16→24→44→92; `PDMUX_R2_POLICY=fixed`, `PDMUX_STICKY_PARTITION` **없음**(job 865973, 2026-07-29 — 플래그 도입 이전)):

**[DERIVED · `mean_part_dwell_s_gridlimited` = R_time_all 분자 / 단조시간축 PART run 수]** (출처: `residency_census_2026-09-21/census_groups.csv`)

| arm | prefill 16 SM | 24 | 44 | 92 |
|---|---|---|---|---|
| Hs8 | **17.30 s** | 12.35 | 6.13 | **3.13 s** |
| M8 | **30.70 s** | 13.52 | 10.31 | **7.05 s** |
| Ha8 | **17.07 s** | 16.91 | 11.96 | **10.23 s** |
| T8 | 7.43 s | 8.49 | 4.16 | **2.51 s** |

prefill SM이 커질수록 평균 PART 체류가 짧아진다 — 4 arm 중 3개가 완전 단조, T8만 16→24에서 국소 역전. 같은 방향으로 `cohab_P_act`도 96.3→81.9%(Hs8)로 떨어진다.

**그러나 이 데이터로 가설을 닫을 수 없다:**
- `results/s8p_prefill/`는 **claims-auditor 미통과**이고 `CONSENSUS.md:3358`이 그렇게 적어 두었다 ⇒ **정본 인용 금지**.
- 셀당 파일이 대부분 **1개**(T8 p16/p92만 2개) — n≥4 게이트 미충족.
- `mean_part_dwell_s_gridlimited`는 census가 **파생값**으로 계산한 평균이며(분포·IQR 없음), 스냅샷 격자(2 ms)에 제한된다.
- 같은 문장은 **sticky ON에서는 성립하지 않는다** — ON에서 PART 체류를 끝내는 것은 prefill이 아니라 **decode 배치가 비는 사건**이다(§2.0).

**새 실험을 설계하지 않았다.**

### D-정정. "hybrid에서 layer 단위 prefill 발사가 성립하는가"는 **열린 질문이 아니다**

4모델 전부 `forward_split_prefill`이 구현돼 있다 [READ]:

| 모델 | 파일 | 줄 | 파일 sha256 | dev_tree_edits |
|---|---|---|---|---|
| Zamba2 | `src/models/zamba2.py` | **752** | `035a7d758ada757a…` | 항목 2 + **P1.5** |
| NemotronH | `src/models/nemotron_h.py` | **859** | `713333e87b4a697a…` | **항목 6** |
| Falcon-H1 | `src/models/falcon_h1.py` | **503** | `3fb851ad84cdd22b…` | **항목 8** |
| Granite-4 | `src/models/granitemoehybrid.py` | **622** | `b97f83121ae215b1…` | **항목 9** |

그리고 `dev_tree_edits.md` **항목 24**가 이 세 개(6·8·9)를 sync+manifest에 편입하면서 그것들을 *"the `forward_split_prefill` methods that are the reason PD-mux SPLIT_PREFILL works on those models at all"* 이라고 명시한다. (참고: upstream v0.5.10은 dense 9종에만 이 메서드를 갖고 있다 — #7634. hybrid 4종은 engine-port가 더한 것이다.)

**남는 좁은 질문이 기존 기록에서 다뤄진 적이 있는가:**

| 좁은 질문 | 기존 기록 | 상태 |
|---|---|---|
| 효율적인 **span 단위** | `pdmux_context.py`의 `split_forward_token_budget`(기본 65536)이 `forward_count = max(1, budget // extend_num_tokens)`로 층 수를 정한다. 우리 59개 config **전부 65536 고정** — **스윕된 적이 없다** [DERIVED · 전수 yaml 파싱] | **미탐색** |
| **span 경계의 Mamba state 비용** | `dev_tree_edits.md` P1.5가 Zamba2에서 `forward_batch.hidden_states` + `forward_batch.zamba_original`(복제 임베딩)을 split window 사이로 **잇는다**고 적는다 — 즉 상태 운반 **메커니즘**은 문서화돼 있다. 그 **비용을 측정한 기록은 찾지 못했다** | **미측정** |

⇒ 정정 문장: "성립 여부는 닫혔다(4모델 구현 존재). 열려 있는 것은 **span 입도의 최적값**과 **span 경계 상태 운반 비용**이며, 둘 다 **미측정**이다."

---

## 6. 부수 발견 (범위 밖이나 정본에 영향)

### I-1. "이 프로젝트는 전적으로 `manual_divisions` 경로를 쓴다"는 **거짓이다 — 55/59**

`CONSENSUS.md` rev80 배너(`:10-12`)와 §4 living-doc 행(`:7277`), 그리고 작업 루트 `CLAUDE.md`가 공통으로 **"현행 `*pdmux*.yml` 59/59 전부 `manual_divisions`"** 라고 적는다.

**YAML 파서로 전수 확인한 결과 [DERIVED · `yaml.safe_load` 59개 파일]: 55/59다.** 나머지 4개는 **`manual_divisions` 키가 없다**:

```
workspace/engine-port/triage/pdmux_a100_smoke.yml
workspace/engine-port/results/p1_opint/pdmux_a100_smoke.yml
workspace/engine-port/results/p1_gates/gate1/pdmux_a100_smoke.yml
workspace/engine-port/results/p1_gates/gate2/pdmux_a100_smoke.yml
```

네 파일은 **byte-identical**(sha256 `8e99131810fe1136…`)이고, 내용은 `sm_group_num: 4` + `decode_bs_divisor: 36` + `split_forward_token_budget: 65536` 뿐이며, 파일 주석 자신이 이렇게 적는다:

> *"No manual_divisions -> engine auto-computes partitions via divide_sm() using the device-queried SM count (108) and cc8 arch constraints (min_per_part=4, multiple=2)."*

**왜 전수 조사가 이것을 놓쳤는가 — 재사용 가치 있는 함정:** 이 파일은 `grep -q "manual_divisions"`에 **걸린다**. 걸리는 것은 위 **주석 문장**이다. 키 존재를 grep으로 물으면 "없음을 설명하는 주석"이 "있음"으로 읽힌다.

**영향 (스코프를 정확히):**
- **로컬 미실행 결론은 불변.** A100은 major 8이라 `get_arch_constraints`가 정상 반환하고, 로컬 미실행의 실효 사유 3건(sm120 `sgl_kernel` 빌드 부재·16 GB·게이트 1)은 그대로다.
- **바뀌는 것은 "이 프로젝트의 실행 경로" 서술이다.** 자동 격자 config를 쓰는 job script가 **27개**이고, 그 안에 **02절 헤드라인을 떠받치는 두 캠페인(P1.7 4모델, P1-opint 운영점 jobs 873944/873945)** 과 **03절 비용 수치의 원 캠페인(p1_gates/gate2 HOLB)** 이 들어 있다.
- ⇒ `get_arch_constraints`는 **우리 실행 경로에서 발화한 적이 있다.** "발화하지 않는다"는 서술은 **`manual_divisions` 계열 캠페인(E1/S2/λ0/E2 등) 한정**으로 좁혀야 한다.
- ⇒ 대여 GPU 이식성 논의도 갈린다: 자동 격자 계열 캠페인을 재현하려면 **`divide_sm`의 아키텍처 상수가 직접 격자를 바꾼다**(H100이면 8 SM 단위·132 SM). `manual_divisions` 계열만 "상수를 타지 않는다".

### I-2. `residency_census` README의 줄 인용이 표류했다

README와 모듈 docstring이 PART 술어의 엔진 대응부를 **`multiplexing_mixin.py:880-893`** 으로 인용한다. 그 범위는 **캠페인 판(f60a128)의 줄 번호**이고, **현재 트리에서 880–893은 `_dual_worker_prefill_ready`/`_dual_worker_start_decode` 본문**이다 [READ]. 현재 트리의 대응부는 **1169–1182**다. sha 없는 줄 인용이라 `check_line_citations.py`가 잡지 못한다. (정본 문서가 아니므로 정정 diff에는 넣지 않고 기록만 한다.)

### I-3. sticky `ENABLED` 로그 문구가 실제 동작과 다르다

로그 [READ]:
> *"PD-mux sticky partition ENABLED: decode holds its division while decode is busy (fixed target index=1); **the unpartitioned `(0, total_sm)` group is used only when the decode batch is empty**"*

코드상 decode 배치가 비면 `set_current_stream_idx(0)`이고, index 0의 `sm_counts`는 **`(total_sm, 0)`**(plain **prefill** 스트림)이다 — `pdmux_context.py:104-137`. `(0, total_sm)`은 마지막 인덱스이고, sticky ON에서는 **도달 불가**다(§2.0). telemetry가 이를 확증한다(smoke ON arm에 `0/108` 0건). **동작은 의도대로이고 로그 문구만 두 plain 그룹을 뒤바꿔 적었다.** 영향: 이 로그를 근거로 "sticky ON에서도 `(0,108)`로 폴백한다"고 읽으면 틀린다.

---

## 7. 좌표계 페이지에 미치는 영향

### 01절

| 현재 문장 | 필요한 변경 |
|---|---|
| D1 "decode_bs가 커지면 다른 division으로 drift한다" | **단서 추가**: "…**sticky OFF일 때**. `PDMUX_STICKY_PARTITION=1`+`R2_POLICY=fixed`에서는 인덱스가 상수이고 drift가 없다(`multiplexing_mixin.py:1169-1182`)." |
| D3 "R2 admission hold … 스케줄링층 노브다" | **단서 추가**: "…단 `admission_limited`를 세우는 것은 **`CoarseGrainedController`(generic/hybrid)** 뿐이고 **`FixedPolicy`는 절대 세우지 않는다**(`controller.py:88-101`, `:320-329`) ⇒ **fixed 계열 캠페인 전부에서 미발화**." 그리고 "R2 = 스케줄링층 노브"가 아니라 "**R2 정책층이 스케줄링층에 내는 부수 개입**"으로. |
| D4 "dual-worker 경로도 admission 구간에 개입한다" | **삭제/교체**: "dual-worker 경로는 admission 구간을 **계측한다**(`admission_latency_ms`); 개입하지 않는다(`dual_worker.py:87-97`)." |
| D5 "PART에서 prefill이 받는 SM 몫이 체류 길이를 정한다" | **가설로 명시 + 근거 상태 병기**: "기전 **가설**. 같은 방향의 아카이브 관측이 `results/s8p_prefill/`에 있으나 그 캠페인은 **claims-auditor 미통과 = 정본 인용 금지**이고 셀당 n=1이다. 그리고 이 문장은 **sticky OFF 한정**이다." |
| "hybrid에서 layer 단위 prefill 발사가 성립하는가"(열린 질문) | **닫힌 항목으로 이동**. 남는 열린 질문은 **span 입도**(`split_forward_token_budget` 59/59 전부 65536, 스윕 0건)와 **span 경계 Mamba state 운반 비용**(메커니즘은 P1.5에 문서화, 비용 미측정). |
| (신설) | **sticky가 바꾸는 것**: 분기 조건 하나(`split_prefill_batch or sticky_partition_enabled`)뿐이고, 그 결과 `(0,108)` 폴백 분지가 **사문화**된다. |

### 02절

| 현재 | 필요한 변경 |
|---|---|
| upstream pdmux를 배경/베이스라인으로 서술 | **계보를 명시**: upstream `multiplex/`는 **MuxWise 저자들이 올린 MuxWise의 engine 층**이다(PR #11592/#12275, 추적 이슈 #10813이 MuxWise arXiv를 Related resources로 명시). 우리 **agnostic = MuxWise engine − estimator − dispatcher + hybrid `forward_split_prefill`**. `adjust_stream_groups` 위의 `TODO(jason-fxz): temporary demo` 를 인용할 것. |
| "agnostic PD-mux가 fused를 이긴다" | **측정 경로 명시**: P1.7·P1-opint 둘 다 `pdmux_a100_smoke.yml` = **자동 격자(`divide_sm`)** 경로다. 기존 단서(cudagraph 유무·술어 의존·정상상태 셀 한정)는 그대로 유지. |

### 03절

| 현재 | 필요한 변경 |
|---|---|
| PART 체류 decode forward 1.53–1.94× 를 "동거 기전"으로 | **유지 가능**(출처 캠페인 sticky OFF ⇒ PART ⟺ prefill in-flight). **단 선행 대조를 추가**: **두 선행 모두** 같은 기전을 dense에서 이미 보고했다 — MuxWise §3.3.1(**≈0–30%** slowdown), Bullet §3.2.3(*"isolated SM에서도 memory/network 경합은 남는다"*, *"decode가 prefill보다 경합에 민감"*) ⇒ 우리 몫은 **"hybrid에서의 재확인 + 자릿수 차이"**. ★두 수를 직접 비교하지 말 것(기판·TP·측정량 상이). |
| "전환 ≤ 0.04 ms" | ★**1차 판정 정정 반영**: Bullet camera-ready는 **재구성 비용을 측정해 보고한다**(Table 5 `Resource Re-config` **Mean 4.1 μs / P99 5.9 μs**). 쓸 수 있는 문장은 "두 시스템 모두 **μs 규모**로 보고한다 — 기구도(green-context 인덱스 교체 vs libsmctrl 마스크 쓰기) 측정 층도(device-level 상한 vs **CPU-side**) 다르다"까지다. 우열 문장 금지, 정본 금지문장 #18(`s` 2자리 이상 표기 금지) 유지. |
| (신설) | 그 캠페인은 **자동 격자 경로**이며 target `74/34`는 `divide_sm(108, cc8, 2)`의 산물이다. |

### 04절

| 현재 | 필요한 변경 |
|---|---|
| "PD-mux 선행 중 residency를 측정량으로 보고한 논문은 확인되지 않는다" | **좁힐 것**: "**정의·분모·규약을 갖춘 집계 추정량으로서** residency를 보고한 PD-mux 선행은 확인되지 않는다." **Bullet Fig. 20a는 구성별 지속시간 막대 타임라인을 보여준다**를 각주로 인정. MuxWise의 GPU utilization은 Nsight **active-SM 비율**(≠ residency), Bullet §4.3.2는 **co-run 구간을 전제한 뒤**의 SM 활성도. |
| "선행은 layer 단위 prefill 진행도로 체류를 암묵 조절한다" | **강화 가능 — 이제 두 편 모두**. MuxWise `𝑁_PL = ⌈(𝑇_d × 𝑁_T)/𝑇_P⌉` + §3.4.1 원문(*"predicted latency of the launched prefill layers exceeds that of the corresponding decode iteration"*), **Bullet은 Algorithm 1이 `L_exe + L_step`(이번 step에 돌릴 층수)을 반환**하고 시스템 상태 `PS`에 실행 층수를 둔다(§3.3.2). |
| (신설·권고) | **평가 조건 비교표를 04절에 넣을 것**(§4.5). 특히 **SLO regime이 세 시스템에서 전부 다르다**(우리 ITL p95 60 ms / MuxWise TBT 50·100 ms / Bullet TPOT 150–200 ms + **입력 토큰당 정규화 TTFT**) ⇒ goodput·SLO attainment 수치의 횡단 비교 금지를 명시. 그리고 "우리만 단일 GPU"라고 쓰지 말 것 — **Bullet §4.2.1이 A100 1장 평가다.** |
| (신설·회부) | **Bullet §4.4가 HE0의 정면 대조 주장을 담고 있다**(*"there is no optimal fixed SM allocation"*). 04절에서 이것을 **무시하면 안 된다.** 단 static arm 정의·best-static 탐색 절차·술어·SLO 엄격도·동적 제어의 내용이 전부 달라 **본 감사는 판정하지 않았다** — §4.6 참조, **claims-auditor 회부 대상**. |

### 05절

| 현재 | 필요한 변경 |
|---|---|
| 재집계 표의 PART 수치 | **수치는 정정 불필요**(정의대로 계산됐다). **라벨을 바꿀 것**: sticky ON 캠페인의 "PART 체류"는 **동거 체류가 아니라 대부분 분할 유지 체류**다. |
| (신설) | 분해값 병기: `s2_sticky` R-time-all PART **78.53% = pbusy 7.57 + pidle 70.96**; smoke ON arm **86.22 = 3.19 + 83.03**, OFF arm **5.57 = 3.07 + 2.50**. sticky가 **prefill-busy PART를 거의 바꾸지 않는다**(3.07→3.19)는 점을 명시. |
| (신설) | **realized 단서**: 이 캠페인들에 `PDMUX_GREEN_READOUT`이 없었으므로, 표의 "realized SM"은 **엔진이 고른 인덱스의 `sm_counts` 항목**이지 드라이버가 준 SM 수가 아니다. |

---

## 8. 하지 않은 것

- **GPU 0.** SLURM 제출 0건, 서버 기동 0건, 벤치 실행 0건. 모든 수치는 디스크에 이미 있던 telemetry·로그·설정·코드의 재집계이거나 원문 인용이다.
- **정본 무수정.** `PROJECT_STATUS.md`·`reports/paper/`·`reports/CONSENSUS.md`·`RESUME.md`를 편집하지 않았다. 수정은 `consensus_patch_proposal.diff`로만 냈고 **적용하지 않았다**.
- **읽기 전용.** `sglang_engine_dev/`·`src/`·`results/`·`reports/`의 기존 파일을 수정하지 않았다. 새 파일은 전부 `reports/audit/2026-09-22_scope_lineage/` 아래다. (§6 I-2·I-3의 결함도 **기록만** 했고 고치지 않았다.)
- **판정 없음.** 순위·우열·goodput·SLO·운영점·인과 귀속 진술 0건. HE0·layer-type negative result·정책 순위·Claim D/E 등급 **불변**.
- **새 실험 설계 없음.** §2.6의 두 항목은 **질문형**으로만 적었다.
- **추측으로 메우지 않은 것**: `ykcombat`의 신원, MuxWise 공식 코드 저장소의 존재 여부, `~/Downloads/muxwise.zip`의 출처, MuxWise query-based synchronization의 upstream 대응부, job 872077의 원 산출물 내부, 그리고 **§4.6의 HE0 대 Bullet §4.4 관계**(충돌인지 다른 것을 재고 있는 것인지 — 판정하지 않았다).
- **§4.6은 과제 범위 밖이다.** 지시받은 M2–M6에 없었으나 정본 HE0와 직접 맞닿아 있어 **기록만** 했고, 어떤 등급·문구도 바꾸지 않았다.
- **멈춤 조건 1건 발동 → 해소**: 1차 통과에서 지정 PDF를 찾지 못해 멈춤 조건이 발동했고 arXiv preprint를 명시적 대체 출처로 썼다. 사용자가 `/home/wonho/Experiments/KISTI/Papers/`에 원본을 제공해 **2차 통과에서 camera-ready로 전수 재확인했다**(§4.0). 그 결과 **M5·M6의 1차 판정 2건을 정정했고**(재구성 비용 측정값 존재, Bullet 단일 GPU 평가 존재) **M4를 한 단계 더 좁혔다**(선행 1편 → 2편). MuxWise는 인용 문장 전부가 preprint와 동일해 판정 변경이 없다.
- **외부 반출 없음.** 두 preprint와 upstream 저장소를 공개 URL에서 **가져왔을 뿐**, 이 프로젝트의 어떤 내용도 외부로 보내지 않았다.

---

## 9. 산출물

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 이 문서 |
| `consensus_patch_proposal.diff` | `reports/CONSENSUS.md` 제안 diff (**적용 안 함**, `git apply --check` 통과). 배너 스코프 정정 + §3 항목 **280–283** + §5 열린 항목 **11**(Bullet §4.4 vs HE0, claims-auditor 회부) |
| `citations.json` | 이번 인용의 anchor/anchor_offset/sha (21건, `line_citations.json` 규약) |
| `scratch/part_cohab_split.py` | S3 분해 재집계기 (기술 집계 · 판정 아님) |
| `scratch/part_cohab_split_out.json` | 위 스크립트 출력 (22 파일 × 5 규약 × 3 갈래) |
| `scratch/make_citations.py` | `citations.json` 생성기 |

**재현:**
```bash
cd reports/audit/2026-09-22_scope_lineage/scratch
python3 -B part_cohab_split.py --selftest        # -B: results/ 아래에 __pycache__ 를 만들지 않기 위해
python3 -B part_cohab_split.py --campaign s2_sticky --campaign sticky_smoke --out part_cohab_split_out.json
python3 -B make_citations.py
```
(`-B`는 읽기 전용 규율 때문이다 — 없이 돌리면 CPython이 `results/residency_census_2026-09-21/__pycache__/`에
바이트코드를 쓴다. 이 감사 중 한 번 발생했고 되돌렸다.)
