> ## ⛔ P4 예측 2 판정 정정
> `P4_LABEL.json`의 *"sec2 code-reading withdrawn"*은 **무효(VOID)**로 정정됐다 —
> 등록 문턱 ≥3×가 이 운영점에서 **산술적으로 도달 불가**(기준 0.9722, 최대 가능 1.029×).
> 상세·정정된 판정표·후속 P4-B: [`ADJUDICATION_P4_2026-09-09.md`](ADJUDICATION_P4_2026-09-09.md)

# RESULT — 계측 프로브 P1–P4 + P3-C (실행 2026-09-09)

사전등록: `../PREREG_PROBES_2026-09-08.md` §0–§4 (P3-C는 2026-09-09 §3-C로 추가등록).
**성능·정책 해석 없음 — 라벨과 수치만.** 해석은 메인 세션/result-analyst 몫.

## 0. 실행 요약

| 순서 | 프로브 | job id | State | Elapsed | ExitCode | 라벨/판정 |
|---|---|---|---|---|---|---|
| 1 | P3 `BOOT_SANITY` | 905701 | COMPLETED | 00:06:18 | 0:0 | **`BOOT_OK`** (3/3) |
| 2a | P1 `SPLIT_FIRES` | 905707 | COMPLETED | 00:21:08 | 0:0 | **`SPLIT_FIRES`** |
| 2b | P3-C `DIVISION_COST` (병렬 제출, P1과 독립) | 905710 | COMPLETED | 00:01:26 | 0:0 | **`HOMOG_FREE`** |
| 3 | P2 `FLOOR` | 905712 | COMPLETED | 00:08:14 | 0:0 | 판정없음(캘리브레이션); `any_outside_30pct=False` |
| 4 | P4 `SKEW` | 905713 | COMPLETED | 00:11:59 | 0:0 | 등록예측 1/2/3, 아래 §4 |

**GPU-h 실측**: 378+1268+86+494+719 = 2945 s = **0.818 GPU-h**
(등록 예산: 최초 ≈0.75 → P3-C 추가 후 ≈0.80 GPU-h; 실측은 갱신 예산 대비 +2.3%).

**순서 선택 보고**: P3-C는 P1과 독립이라는 코디네이터 지시에 따라 P1(905707, RUNNING)과
**동시에** 제출했다(gpu39이 당시 `sinfo`상 `mix`로 여유가 있어 보였음). 실제로는 두 잡 모두
gpu37에서 순차 실행됐으나(파티션 혼잡), 이 선택이 P1→P2→P4 순서 자체를 바꾸지는 않았다.

**실패**: 0건. `BOOT_FAILURES.txt`가 5개 job 디렉터리 어디에도 생성되지 않음(전수 확인).
2회 연속 부팅 실패로 인한 중단 없음.

## 1. 규약 준수

- **`--comment` 게이트**: 5/5 `.sbatch` `check_sbatch_comment.py` conformant, `sbatch --test-only` 전부 accepted.
- **`sync_engine_tree.sh` manifest**: 5개 job 전부 `runtime_source_manifest.sha256` 보존, 전부 2517 bytes
  (동일 소스 트리 상태 — 캠페인 전체가 같은 엔진 커밋 위에서 실행됐음을 확인).
- **Telemetry 무결성**: P1(d44/d92)·P4(sticky_off/on) 4개 summary 전부 `dropped_events=0`,
  `parse_errors=0`. (P2/P3/P3-C는 telemetry env는 §0대로 세팅했으나 라벨 판정에 telemetry를
  쓰지 않는 프로브라 `dropped_events`를 표로 뽑지 않았음 — 파일 자체는 각 job 디렉터리에 존재.)
- **cudagraph**: 전 프로브 **ON**(`--disable-piecewise-cuda-graph`만 사용, `--disable-cuda-graph`
  는 어디에도 없음) — 운영점 정합.
- **backend**: 전 프로브 `flashinfer`(게이트 #83).
- **n / seed**: 이 프로브 세트는 계측 프로브이며 정책 비교가 아니다(사전등록 §5, "정책 순위 없음").
  각 셀 1 boot. Seed는 프로브 내부 재현성 목적으로 고정: P1·P4 동일 셀 `seed=17`(P4는 P1과
  워크로드 실현을 맞추기 위해 동일 seed 사용), P2 `seed=23`, P3·P3-C smoke `seed=7`.
  → 프로젝트 실험 규약의 `n≥5` 정책비교 요건은 **이 프로브 세트에 적용 대상이 아님**
  (사전등록 §5 "성능·goodput·정책 순위 — 재지 않는다"가 그 이유를 명시).

## 2. P1 `SPLIT_FIRES` (job 905707) — ★가장 중요

설정: Nemotron-Nano-9B-v2-Base, flashinfer, ctx 8192, out 96, `--request-rate 1.0`, 160 요청, seed 17.

| arm | target D | n_decode_active_at_target | n_decode_active_snapshots | decode_hist_mode | decode_hist (dsm:sec) | f_time | f_count |
|---|---|---|---|---|---|---|---|
| d44 | 44 | 474 | 483 | **44** | {44: 238.73, 108: 1.91} | 0.9921 | 0.9814 |
| d92 | 92 | 480 | 488 | **92** | {92: 861.32, 108: 1.56} | 0.9982 | 0.9836 |

**라벨: `SPLIT_FIRES`** — 두 arm 모두 split 스냅샷 > 0, `decode_hist_mode`가 각각 타깃(44/92)과 정확히 일치.
`n_split_pab0=0`(양쪽 다) — TRACE_EVERY=32(기본)에서는 record-skew 신호가 관측되지 않았다.

## 3. P3-C `DIVISION_COST` (job 905710, ★2026-09-09 추가등록)

한 변수만 다름: P3의 `d44` 부팅(homog5.yml, 5 groups, 3 division) vs 이 부팅(`stage0_xctrl/pdmux_d44.yml`, 3 groups, 1 division). 그 외 모델·백엔드·ctx·전 플래그·env·스모크 동일.

| | reference (homog5, job 905701) | new (1-division, job 905710) |
|---|---|---|
| n_stream_groups | 5 | 3 |
| max_total_num_tokens | 2,618,868 | 2,620,200 |
| capture list | [1,2,4,8,12,16,24,32,40,48] (len 10) | [1,2,4,8,12,16,24,32,40,48] (len 10) |
| boot_seconds | (미기록, P3에서는 등록 안 됨) | 49 |
| smoke rc | 0 | 0 |

Δmax_total_num_tokens = **0.0509%**(< 5% 문턱) · capture list 길이 동일 **및** 완전 동일.

**라벨: `HOMOG_FREE`** — 균질화(3-division homog5) config는 이 조건에서 사실상 무상.
단 1회 시도(attempt=1)로 boot_ok=1, 재시도 불필요.

★주장하지 않음: 이 Δ가 성능(goodput/ITL/TTFT)에 미치는 영향 — 재지 않았다.

## 4. P2 `FLOOR` (job 905712)

설정: 동시성 1(`--max-concurrency 1`, `--request-rate` 미지정), d44, out 96, 22요청(앞 2개 워밍업 제외), seed 23.

| ctx | floor_ref (TTFT p50, s) | ttft_p95 (s) | itl_solo p50 (s, [9,32) 창, n샘플) | 실현 입력 토큰(mean/min/max, n=20) | 오류(n_errors) | 사전 외삽값 (s) | ±30% 이내 |
|---|---|---|---|---|---|---|---|
| 1024 | 0.1237 | 0.1261 | 0.01294 (n=460) | 1024/1024/1024 | 0 | 0.13 | **예** |
| 4096 | 0.4638 | 0.4662 | 0.01300 (n=460) | 4096/4096/4096 | 0 | 0.49 | **예** |
| 8192 | 0.9104 | 0.9142 | 0.01304 (n=460) | 8192/8192/8192 | 0 | 0.96 | **예** |
| 16384 | 1.8216 | 1.8293 | 0.01311 (n=460) | 16384/16384/16384 | 0 | 1.90 | **예** |

**판정 없음(캘리브레이션, 사전등록 §2)**. 등록 예측 결과: **`any_outside_30pct=False`, `any_missing=False`**
→ `RATIO` 셀 좌표·k·예산 재계산 트리거 **미발동**(4셀 전부 ±30% 이내로 관측됨 — 이 자체가 산출물).
실현 입력 토큰수가 4셀 전부 정확히 요청 ctx와 일치(min=mean=max=ctx) — `--tokenize-prompt`+
`--random-range-ratio 1.0`가 의도대로 정확한 토큰 길이를 만들었음을 확인.
★L-10 승계: 동시성 1이므로 이 값들은 arm 무관 물리량으로 등록됨(재확인은 하지 않았음 — d44만 부팅).

## 5. P4 `SKEW` (job 905713)

설정: P1과 동일 셀(d44, ctx 8192, rate 1.0, 160요청, seed 17) × `PDMUX_DUAL_WORKER_TRACE_EVERY=1`, 2 boot(sticky OFF / `PDMUX_STICKY_PARTITION=1`).

| | sticky OFF | sticky ON |
|---|---|---|
| f_time | 0.9841 | 0.9921 |
| f_count | 0.9722 | 0.9894 |
| n_decode_active_at_target / n_decode_active_snapshots | 15012 / 15442 | 15278 / 15442 |
| n_split_total | 15012 | 15278 |
| n_split_pab0 | 4 | 270 |
| frac_split_pab0_of_split | 0.000266 | 0.017672 |
| t_decode_active_s | 236.21 | 236.61 |

### 등록 예측 결과 (사전등록 §4, 그대로 인용 — 판정 아니라 관측)

| # | 비교 | 수치 | 결과 |
|---|---|---|---|
| 1 | `f_count`(TRACE_EVERY=1, sticky OFF) vs `f_count`(P1 d44, TRACE_EVERY=32) | 0.9722 vs 0.9814, rel_diff=0.94% | **±20% 이내 → "부표집 무해"** |
| 2 | `f_count`(sticky ON) / `f_count`(sticky OFF) | 0.9894/0.9722 = **1.018배** | **3배 미만 → "`FINDING_DUTYCYCLE §2`의 코드-독해 주장 철회"** |
| 3 | `frac_split_pab0_of_split`: ON(0.01767) vs OFF(0.000266) | ON > OFF | **"record-skew 기전 설명 지지"** |

## 6. P3 `BOOT_SANITY` (job 905701) — 상세

| D | boot_ok | sm_counts(5행) | max_total_num_tokens | capture list len | smoke rc | Median TTFT/ITL (ms, 참고용 원시값) |
|---|---|---|---|---|---|---|
| 16 | 1 | [(108,0),(92,16),(64,44),(16,92),(0,108)] | 2,618,868 | 10 | 0 | 596.89 / 14.96 |
| 44 | 1 | 〃 | 2,618,868 | 10 | 0 | 767.24 / 14.98 |
| 92 | 1 | 〃 | 2,618,868 | 10 | 0 | 2543.39 / 14.92 |

**라벨: `BOOT_OK`**(3/3 부팅 + 캡처 완료 + sm_counts 5행 확인). `N_CAPTURE_FAILURES=0`(전 D).

## 7. 아티팩트 절대경로

```
workspace/engine-port/results/longctx_conflict/probes/
  pdmux_homog5.yml
  p3_boot_sanity.sbatch      p3_905701/    (BOOT_OK, telemetry×3, banner×3, manifest)
  p1_split_fires.sbatch      p1_905707/    (SPLIT_FIRES, summary_d44.json, summary_d92.json, P1_LABEL.json, manifest)
  p3c_division_cost.sbatch   p3c_905710/   (HOMOG_FREE, banner_d44.txt, P3C_LABEL.json, manifest)
  p2_floor.sbatch            p2_905712/    (floor_c*.json ×4, P2_LABEL.json, manifest)
  p4_skew.sbatch             p4_905713/    (summary_sticky_off.json, summary_sticky_on.json, P4_LABEL.json, manifest)
  analyze_probes.py  p2_floor_analyze.py   (분석 코드; e1_pin_check.compute_decode_realized 재사용, 재구현 없음)
  RESULT_PROBES_2026-09-08.md  (본 문서)
```

## 8. 이 결과가 주장하지 않는 것 (사전등록 §5 승계)

- 성능·goodput·정책 순위 — 재지 않았다(P1의 rate 1.0은 임의 선택, 운영점 주장 아님).
- `f`의 절대값을 물리량으로 승격하지 않는다.
- HE0·정본 결론에 대한 어떤 함의도 없다.
- P1이 `SPLIT_FIRES`인 것은 "처치가 존재한다"까지이며 **실현 정도가 아니다**.
- P3-C의 Δ가 성능(goodput/ITL/TTFT)에 미치는 영향 — 재지 않았다(§3-C 명시).

## 9. 쓰면 안 되는 문장 (사전등록 §6 승계)

- *"P1이 분할 실현을 검증했다"* — 발화 여부만.
- *"`f`가 처치 걸린 시간 비율이다"* — 32-sync 부표집 격자의 지속시간 몫이다.
- *"P2 바닥이 다른 모델·부하에 이식된다"*.
- *"프로브가 규칙층을 대체한다"* — 규칙층은 여전히 `NO-GO`이고 캠페인은 미제출이다.
- *"P3-C가 division 비용을 성능 측면에서 닫았다"* — `max_total_num_tokens`·capture list만 쟀다.
