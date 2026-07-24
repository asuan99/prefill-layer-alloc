# Sweep Consistency Audit (Phase 0)

작성일: 2026-06-11
대상: prefill-layer-alloc / Stage 1 · decode sweep · figure 파이프라인
방법: 코드·CSV grep/파싱 (코드 수정 없음). 각 항목에 판정과 근거 `파일:라인` 기록.

판정 범례: **[확인됨]** = 불일치 실재 · **[오탐]** = 불일치 없음 · **[이미 수정됨]** = 과거 수정 흔적 있고 현재 정합.

요약 표:

| # | 항목 | 판정 | Phase 1+ 조치 |
|---|------|------|---------------|
| 1 | SM 격자 | **[확인됨]** (부분) | sm_grid 단일화. snapping 흔적은 없음 |
| 2 | seq/batch 격자 | **[확인됨]** | batch_grid/seq_grid_prefill 통일 |
| 3 | 측정 granularity | **[확인됨]** | Phase 2 attn chunked 경로 |
| 4 | BW 상수 | **[확인됨]** | hbm_bw_GBs=2039 단일화 |
| 5 | CSV 스키마 | **[확인됨]** | 통일 스키마 |
| 6 | 파일명 규약 | **[확인됨]** | canonical_filename 언더스코어 |
| 7 | 포화 판정 | **[확인됨]** | saturation_point() 단일 구현 |
| 8 | n_blocks | **[확인됨]** | n_blocks() 단일 구현. `//4` 1건 잔존 |
| 9 | DEMO 데이터 | **[확인됨]** (부분) | `_DEMO` suffix 강제 |

---

## 1. SM 격자 — [확인됨] (부분: 비-A100 config가 다른 격자, snapping 흔적 없음)

**Green Context preset (정본 후보):** `[14, 27, 40, 54, 68, 81, 94, 108]`

소비 경로는 모두 `hw_cfg["sm_sweep_steps"]`를 읽음 (단일 출처처럼 보이나 yaml 내부에서 격자가 갈림):
- `workspace/shared/configs/hardware.yaml:20` `a100_80gb` → `[14,27,40,54,68,81,94,108]` ✓
- `workspace/shared/configs/hardware.yaml:29` `a100_sxm4_80gb` → `[14,27,40,54,68,81,94,108]` ✓
- `workspace/shared/configs/hardware.yaml:37` `a100_80gb_pcie` → `[14,27,40,54,68,81,94,108]` ✓
- **`workspace/shared/configs/hardware.yaml:12` `a100_40gb` → `[11, 22, 33, 44, 54, 65, 87, 108]` ✗ (preset과 불일치, "~10% increments" 주석)**
- `hardware.yaml:45/53/61` h100/4090/5060ti → 각 하드웨어 고유 격자 (정상)

하드코딩 리터럴(yaml 외부):
- `workspace/characterization/stage1_sm_scaling/run_chunked_ssm_sweep.py:95` `default=[14,27,40,54,68,81,94,108]`
- `workspace/characterization/decode_sm_scaling/run_decode_sweep.py:49` `SM_PRESETS = [14,27,40,54,68,81,94,108]`
- `workspace/characterization/decode_sm_scaling/_decode_worker.py:305` `preset_sm_counts=[14,27,40,54,68,81,94,108]`
- `workspace/characterization/decode_sm_scaling/free_zone_phase.py:50` `PARAM_D = [14, 27, 40, 54]` (부분 격자, 의도적)
- `workspace/characterization/tests/test_green_ctx_controller.py:315` `a100_steps = [...]` (테스트 — 제외 대상)

**snapping 흔적:** 측정된 모든 A100-SXM4 CSV의 `sm_count` 고유값 집합은 정확히
`{14,27,40,54,68,81,94,108}` — 요청값과 preset이 달랐던 행 없음.
(`results/stage1/chunked/*`, `workspace/characterization/results/stage1/*_sxm4_80gb.csv`,
`.../decode_scaling/*_sxm4_80gb.csv` 12개 파일 전수 확인). Green Context 내부 snapping
(`green_ctx_controller.py:340` bisect)은 동작하나, preset과 sweep 격자가 일치하므로
CSV에 snap된 흔적이 남지 않음.

**조치:** Phase 1 `sm_grid` 단일화. yaml 외부 리터럴(run_chunked/run_decode_sweep/_decode_worker)을
spec 참조로 교체. `a100_40gb` 격자는 A100-SXM4 측정과 무관하므로 본 태스크 범위 밖(보고만).

---

## 2. seq/batch 격자 — [확인됨]

각 sweep 스크립트의 기본 격자 (argparse default):

| 스크립트 | seq/context default | batch default |
|----------|---------------------|---------------|
| `run_ssm_prefill_sweep.py:387/391` | `[512,1024,2048,4096,8192,16384,32768]` | `[1,4,16,32,64]` |
| `run_attn_prefill_sweep.py` (호출부 `_sweep_worker`) | (CLI 전달) | (CLI 전달) |
| `run_mlp_prefill_sweep.py:151/153` | `[512,1024,2048,4096,8192,16384]` | `[1,4,16,32,64]` |
| `run_chunked_ssm_sweep.py:100/105` | `[512,1024,2048,4096]` | `[1,4,16]` |
| `run_ssm_two_pass_sweep.py:280/289` | `[512,1024,2048,4096,8192,16384,32768]` | `[1,4,16,32,64]` |
| `run_decode_sm_sweep.py:497/499` | `[512,2048,4096,8192]` | `[1,4,16]` |
| `decode_sm_scaling/run_decode_sweep.py` | (context) | — |
| `run_ncu_profile.py:437/441` | `[256,512,1024,2048,4096,8192,16384,32768]` | `[1,4,16,32,64]` |

기존 CSV의 (seq, bs) 고유 조합 (A100-SXM4):
- `ssm_chunked_*`: seq `{512,1024,2048,4096,8192,16384}` × bs `{1,4,16,32}`
- `ssm_scaling_zamba2`: seq `{512,1024,2048,4096,8192}` × bs `{1,4,16,32,64}`
- `ssm_scaling_falcon_h1`: seq `{...,16384}` × bs `{1,4,16,32,64}`
- `attn_scaling_*`: seq `{512..16384}` × bs `{1,4,16,32}`
- `mlp_scaling_*`: seq `{512..16384}` × bs `{1,4,16,32,64}`
- `decode_decode_attn_*`: context `{512,2048,8192,32768}` × bs `{1,4,8,16,32}`
- `decode_decode_ssm_*`: context `{0}` (decode seq_len=1) × bs `{1,4,8,16,32}`

**차집합 (불일치 핵심):**
- prefill 격자에 **`bs=8` 부재** (chunked/scaling 모두). decode에는 존재. → 4셀 교차비교 불가.
- prefill 격자에 **`bs=64` 존재** (ssm/mlp). decode·chunked에는 부재.
- chunked는 `bs∈{1,4,16}`만, scaling은 `bs∈{1,4,16,32,64}` — chunked vs scaling 직접 비교 시 bs=32/64 결측.
- zamba2 ssm_scaling은 seq 8192 상한, 다른 파일은 16384까지 — seq 교집합 `{512,1024,2048,4096,8192}`.

spec 제안값 `batch_grid=[1,4,8,16,32]`, `seq_grid_prefill=[512,2048,4096,8192]`,
`context_grid_decode=[512,2048,8192,32768]`와 대조: decode 격자는 이미 일치, prefill 격자는
`bs=8` 추가·`bs=64`/`seq∈{1024,16384,32768}` 제외 필요.

**조치:** Phase 1 격자 단일화 후 Phase 2/3 신규 측정은 spec 격자로. 기존 CSV는 보존(차집합은 신규 측정으로 채움).

---

## 3. 측정 granularity — [확인됨] (SSM=chunked, Attn=full-seq 단일 호출)

- **SSM (chunked):** `chunked_ssm_runner.py`는 `prefill_chunk_tokens` 단위로 `mamba_chunk_scan_combined`를
  반복 호출 (`run_chunked_ssm_sweep.py:8` "kernel 호출당 토큰 수 sweep", `chunked_ssm_runner.py:121`
  `n_blocks_per_call = batch * nchunks_per_call * n_heads`, CSV에 `n_kernel_calls` 컬럼 존재).
- **SSM (scaling):** `run_ssm_prefill_sweep.py` → `layer_runner.run_ssm_layer` → full-seq 단일
  `mamba_chunk_scan_combined` (내부 chunk_size=256은 커널 내부 분할, 외부 chunk 루프 없음).
- **Attn:** `run_attn_prefill_sweep.py` → `_sweep_worker.py:263` `runner.run_attn_layer(...)` →
  `layer_runner.py:559/651` `F.scaled_dot_product_attention(q,k,v,is_causal=True)` (또는 flashinfer
  단일 호출). **chunk 루프 부재 확인.** `attn_scaling_*.csv`에 `n_chunks`/`n_kernel_calls` 컬럼 없음.
  `context_len` 컬럼은 있으나 단일 full-seq 호출용.

→ SSM(chunked) vs Attn(full-seq) granularity 비대칭 = 현재 비대칭 결과의 confound 확정.

**조치:** Phase 2 `run_attn_chunked_sweep.py`로 attn을 512-token chunk·누적 KV로 재측정.

---

## 4. BW 상수 — [확인됨] (1555 / 1935 / 2000 / torch-query 혼용, 2039 정본 미사용)

- `hardware.yaml:14` a100_40gb `memory_bw_GBs: 1555`
- `hardware.yaml:22` a100_80gb `memory_bw_GBs: 2000`
- `hardware.yaml:31` **a100_sxm4_80gb `memory_bw_GBs: 2000`** ← 측정 대상
- `hardware.yaml:39` a100_80gb_pcie `memory_bw_GBs: 1935`
- `plot_srm.py:61` `PEAK_BW_GBS = 2000.0   # HBM2e ... A100 SXM4 80GB` (line 74에서 yaml 값으로 override)
- `metrics.py:159-174` `_query_theoretical_bw()` — `theoretical_bw_GBs` 미지정 시 torch device
  property (`2 × mem_clock × bus_width / 8`)로 계산. 이 값이 yaml의 2000과 다를 수 있음.
  실제 runner는 `hw_cfg.get("memory_bw_GBs")`를 전달 (`run_ssm_prefill_sweep.py:275` 등)하므로
  보통 2000 사용, 단 `--bw-gbs` 미전달 경로/decode에서 None→torch query 가능.

spec 정본 `hbm_bw_GBs: 2039` (A100-SXM4-80GB HBM2e spec sheet)는 **현재 어디에도 미사용** —
전 경로가 2000.

**spec sheet 재확인 (Phase 0 확정):** A100-SXM4-80GB의 공식 HBM2e 대역폭은 NVIDIA datasheet
기준 **2039 GB/s** (= 2.039 TB/s). 기존 2000은 반올림값. → spec `hbm_bw_GBs: 2039` 채택, 전 경로
이 값 참조로 통일. (단 bw_utilization은 derived 라벨 유지.)

**조치:** Phase 1 spec `device.hbm_bw_GBs=2039`, Phase 4.3에서 모든 bw_util 계산을 spec 참조로 교체.

---

## 5. CSV 스키마 — [확인됨] (3종 스키마 분기)

컬럼 합집합 대비 파일별 결측:

| 컬럼 | scaling (ssm/attn/mlp) | chunked | decode |
|------|:--:|:--:|:--:|
| sm_count | ✓ | ✓ | ✓ |
| sm_ratio | ✓ (`sm_ratio`) | ✓ (`sm_ratio_pct`) | ✗ |
| seq_len | ✓ | ✓ | `context_len` |
| batch_size | ✓ | ✓ | `batch` |
| latency_ms | ✓ | ✓ (`latency_ms`/`latency_std_ms`) | `latency_per_step_ms` |
| latency_p99_ms | ✓ | ✗ | ✗ |
| achieved_bandwidth_GBs | ✓ | **✗** | `achieved_bw_GBs` |
| theoretical_bw_GBs | ✓ | **✗** | ✗ |
| bw_utilization_pct | ✓ | **✗** | `bw_util_pct` |
| model_name | ✓ | `model` | `model` |
| layer_type | ✓ | ✗ (파일명으로 구분) | `scope` |
| n_blocks | ssm만 ✓ | `n_blocks_per_call` | ✗ |
| cooperative_safe | ✗ | ✓ | ✗ |
| n_kernel_calls | ✗ | ✓ | ✗ |
| prefill_chunk_tokens | ✗ | ✓ | ✗ |
| state_passing_active | ✗ | ✓ | ✗ |
| status | ✗ | ✗ | ✓ |
| error | attn/mlp ✓ | ✗ | (status로 대체) |

근거 헤더:
- scaling: `run_ssm_prefill_sweep.py:354`, `run_attn_prefill_sweep.py:38`, `run_mlp_prefill_sweep.py:36`
- chunked: `run_chunked_ssm_sweep.py:174-176` 및 실측 헤더
- decode: `decode_decode_ssm_*.csv` 1행 주석(`# value_kind: ...`) + 2행 헤더 `model,cell,scope,sm_count,batch,context_len,latency_per_step_ms,bytes_analytic,bytes_analytic_w_weight,achieved_bw_GBs,bw_util_pct,backend,status`

3종이 컬럼명(예: `model` vs `model_name`, `sm_ratio` vs `sm_ratio_pct`, `seq_len` vs `context_len`),
bw 컬럼 유무, 라벨 주석 유무에서 모두 다름. chunked는 bw 컬럼 자체가 없음.

**조치:** Phase 2 attn_chunked CSV는 ssm_chunked 스키마 + `n_chunks` + (derived 라벨) bw 컬럼으로 통일.
Phase 1 통일 스키마 정의.

---

## 6. 파일명 규약 — [확인됨] (하이픈 vs 언더스코어 혼용)

device tag 출현 빈도 (results 전체):
`a100-sxm4-80gb` 22건 · `a100_sxm4_80gb` 15건 · `a100_80gb` 20건 · `a100-80gb` 2건.

두 생성 경로가 다른 정규화:
- **언더스코어:** scaling sweep은 `shared/loaders.py:36-42` `_make_tag()` → `.replace(" ","_").replace("-","_")`
  → `a100_sxm4_80gb`. (`run_ssm_prefill_sweep.py:276` `tag=device_tag(hw_cfg)`, `:350` 파일명)
- **하이픈:** chunked sweep은 `run_chunked_ssm_sweep.py:182` 자체 정의
  `dev_tag = hw_cfg["name"].lower().replace("nvidia ","").replace(" ","-")` → `a100-sxm4-80gb`.

glob이 가정하는 쪽:
- `plot_motivation.py:294-297` `ssm_chunked_{model}_*.csv` / `attn_scaling_{model}_*.csv` — `*`로 양쪽 흡수.
- `plot_saturation.py:529/535` `*.csv` / `ssm_chunked_*.csv` — `*`로 흡수, `:599` 모델명만 파싱.

→ 현재 glob이 `*`라 깨지진 않으나, tag 표기 자체가 두 갈래라 정렬·중복 위험.

**조치:** Phase 1 `canonical_filename()` 언더스코어(`a100_sxm4_80gb`) 단일화.
Phase 4.2 신규 쓰기는 정본만, 구 하이픈 파일은 읽기 전용 호환 glob.

---

## 7. 포화 판정 — [확인됨] (3개 중복 구현, 임계값 동일 0.03이나 입력·가드 상이)

| 위치 | 함수 | 입력 컬럼 | <2점 가드 | 비고 |
|------|------|-----------|:--:|------|
| `plot_saturation.py:82` | `find_saturation_sm` | `normalized_throughput` (사전계산) | ✗ | THRESHOLD `:45`=0.03 |
| `decode_sm_scaling/analyze_decode_sat.py:101` | `find_saturation_sm` | `normalized_throughput` | ✓ (None 반환) | THRESHOLD `:44`=0.03 |
| `serving-eval/plot_motivation.py:103` | `_find_saturation_sm` | `1/latency_ms` 내부 정규화 | ✗ | `_SAT_THRESHOLD`=0.03 |

세 구현 모두 "marginal throughput gain per 10% SM < 3%" 정의는 같으나:
- plot_motivation은 throughput을 `1/latency_ms`에서 직접 만들고 max로 정규화.
- 나머지 둘은 외부에서 만든 `normalized_throughput` 컬럼에 의존.
- analyze_decode_sat만 `len<2 → None` 가드. → 2점 미만 그룹에서 동작 차이.

기타 saturation 언급(검출 로직 아님): `run_ssm_prefill_sweep.py`, `g1_gate_check.py`,
`compute_decision_matrix.py` 등은 위 함수 결과 소비/게이트만.

**조치:** Phase 1 `saturation_point(df)` 단일 구현. plot_motivation의 자체 `_find_saturation_sm`(Fig B
검출 로직) 제거하고 import로 교체 (Phase 3/4).

---

## 8. n_blocks — [확인됨] (단일 공식 합의 없음, `batch×seq//4` 1건 잔존)

n_blocks 공식 출현:
- `wave_estimator.py:133` (SSM full): `n_blocks = batch * n_heads * ceil(seq_len/chunk_size)` — 모델별 n_heads 사용 ✓
- `wave_estimator.py:169` (SSM chunked): `n_blocks = batch * n_heads * ceil(prefill_chunk/ssd_chunk)` ✓
- `wave_estimator.py:31` (matmul): `ceil(M/128) × ceil(N/128)` (별도)
- `chunked_ssm_runner.py:121/183` `n_blocks_per_call = batch * nchunks_per_call * n_heads` ✓ (모델 n_heads)
- **`run_ssm_two_pass_sweep.py:231` `row["n_blocks"] = max(1, row["batch_size"] * row["seq_len"] // 4)`**
  ← **구 `batch × seq // 4` 공식 잔존 (모델-agnostic, 부정확).**
- `run_ssm_prefill_sweep.py:56` 주석: "The old simplified formula (batch × seq_len // 4) was
  model-agnostic and incorrect." → 이 파일은 **[이미 수정됨]**, 정정 공식 사용.

→ 체크리스트 "`batch × seq / 4` 잔존 0건" 목표 대비 **1건 잔존** (`run_ssm_two_pass_sweep.py:231`).
two_pass는 deprecated 계열이나 검증 체크리스트가 grep 0건을 요구하므로 Phase 1 `n_blocks()` 호출로 교체 대상.

**spec 제안 공식 검토:** `n_blocks(model, batch, seq, chunk) = batch × ceil(seq/256) × ssm_n_heads`.
기존 `wave_estimator`는 `ceil(seq/chunk_size)`이고 chunk_size=256(zamba2/fh1/nemotron 공통)이므로
`ceil(seq/256)`와 동치. → 정합. (단 모델별 chunk_size가 256이 아닐 경우 차이 — 현재 3모델 모두 256.)

**조치:** Phase 1 `n_blocks()` 단일 구현, two_pass:231 교체, wave_estimator/chunked_runner는 호출로 정리.

---

## 9. DEMO 데이터 — [확인됨] (부분: Nemotron 전용 패널 없음, 범용 --demo synthetic 존재)

- `plot_motivation.py:512-529` Fig B(roofline/SRM): `demo or (ssm_df.empty and attn_df.empty)` 분기에서
  `ssm_ai_demo/ssm_tfl_demo/attn_ai_demo/attn_tfl_demo` synthetic 점을 동일 figure에 scatter
  ("SSM chunked (DEMO)", "Attention (DEMO)"). 데이터 결측 시 **자동으로 demo로 폴백** (`:386 demo=True`).
- `plot_motivation.py:369-388` Fig A도 demo 분기 존재.
- `:681` `--demo` 플래그 (synthetic). `:777` `--demo` 시 Fig D skip.

→ 핵심 위험: 실측 결측 시 **자동 demo 폴백**으로 synthetic이 실측 figure와 같은 출력 경로/파일명에 섞일 수 있음
(`:386`, `:512` `ssm_df.empty` 조건). DEMO 라벨은 범례에만, 파일명엔 없음.

**Nemotron 전용 synthetic 패널:** `plot_motivation.py`에 `nemotron` 문자열 없음 — 모델 전용 demo 패널은
**없음**. Nemotron은 `models.yaml`/CLI choices/serving에만 등록 (값은 TODO, `models.yaml:27-50`).
프롬프트 Phase 3.4의 "Nemotron-H DEMO 패널 제거"는 *현 시점 별도 패널 부재*로, 실제로는 (a) Nemotron을
실측 모델로 편입하고 (b) 범용 demo 폴백이 Nemotron 실측 figure를 오염시키지 않도록 `_DEMO` suffix 강제가
적용 대상.

**조치:** Phase 3.4 — demo 출력 파일명에 `_DEMO` suffix 강제(`:386`/`:512` 폴백 포함), 자동 폴백이
실측 디렉토리에 무라벨 파일을 쓰지 않게 차단. Nemotron 실측 CSV 생성(Phase 3.1-3.3).

---

## Phase 1+ 진입 시 제외/주의

- **[이미 수정됨]:** `run_ssm_prefill_sweep.py`의 n_blocks 정정 공식 — 재수정 불필요(참조 모델).
- **범위 밖(보고만):** `a100_40gb`의 `[11,22,...]` 격자 — A100-SXM4 측정과 무관(타 하드웨어).
  H100/4090/5060ti 격자도 하드웨어 고유값이므로 통일 대상 아님.
- **deprecated 경로:** `run_ssm_two_pass_sweep.py:231`는 deprecated이나 체크리스트 grep 0건 요구로 교체.
- **회귀 검증 메모(Phase 4.4용):** scan 대표 수치 재산출은 `ssm_chunked_*` / `ssm_scaling_*` 중
  어느 것이 "scan @sm=108 seq=4096 bs=4 = 1.820ms"의 출처인지 Phase 4에서 확정 후 비교.

---

## Phase 4.4 회귀 검증 결과 (통일 saturation_point() 재산출)

기존 SXM4 CSV에 대해 `shared.sweep_spec.saturation_point()` (단일 구현)로 정정 보고서
(`stage1_corrected_vs_hm_thesis.md`)의 대표 수치 3개를 재산출. **GPU 불필요** (기존 CSV 분석).

**값 출처 확정:**
- "Zamba2 scan @sm=108 seq=4096 bs=4 = 1.820ms"의 출처는 `ssm_chunked_zamba2_a100-sxm4-80gb.csv`,
  `prefill_chunk_tokens == seq_len`(= 단일 kernel call) 행. (`ssm_scaling_*`의 동일 셀은 1.651ms로 별개 경로.)
- 정정 보고서 §2.3/§2.5의 "scan-only saturation"은 **(seq,bs)별 최대 pct**(= 최소 kernel 호출,
  full-scan에 최근접) 행에 대해 sm_count를 sweep한 결과. seq≤4096은 pct==seq, seq>4096(8192/16384)은
  data상 최대 pct=4096(데이터에 pct==seq 행 없음). 이 해석으로만 보고서의 seq≥8192 행이 재현됨.

**재산출 결과 (전부 일치):**

| 검증 항목 | 보고서 | 재산출(통일 함수) | 판정 |
|-----------|:--:|:--:|:--:|
| Zamba2 scan @sm=108 seq=4096 bs=4 (latency) | 1.820 ms | 1.820 ms | ✅ |
| Zamba2 bs=16 sm_sat (seq 2048/4096/8192/16384) | 94 | 94 / 94 / 94 / 94 | ✅ |
| FH1 bs=4 sm_sat (seq 1024/2048) | 81 | 81 / 81 | ✅ |

부수 확인: Zamba2 (8192,1)=81, (16384,1)=81 등 §2.3 개별 셀도 일치.

→ **불일치 0건.** 통일 `saturation_point()`가 기존 분석 결과를 보존함을 확인. 코드를 수치에 맞추는
조정은 없었으며, scan-only의 pct 선택 규칙(최대 pct)을 명문화한 것이 유일한 해석 확정 사항.

---

## 구현 현황 및 측정 블로커 (작업 환경 한계)

### 측정 블로커 (현 머신에서 신규 측정 불가)

이 작업 머신에서 Phase 2/3의 **실측 CSV 생성은 불가**하며, 그 이유 2가지:

1. **GPU 불일치:** 현 머신은 **A100 80GB PCIe** (`auto` 감지 → `a100_80gb_pcie`, BW 1935).
   기존 결과는 전부 **A100-SXM4-80GB**에서 측정됨. PCIe에서 새로 측정하면 다른 디바이스·다른 성능이라
   SXM4 비대칭 표와 직접 비교 불가 → 혼입 금지 원칙 위반. (PCIe로 찍으면 `*_a100_80gb_pcie.csv`로
   분리 저장되긴 하나, matched-granularity/회귀 비교의 기준이 SXM4임.)
2. **mamba_ssm 미동작:** `selective_scan_cuda` 임포트가 `libcudart.so.13: cannot open shared object file`로
   실패 (커널이 CUDA 13 빌드, 현 env는 cu128). SSM chunked 커널(`mamba_chunk_scan_combined`) 자체가
   현 머신에서 실행 불가.

→ Phase 2/3의 **코드는 완성**하여 SXM4 + 정상 mamba_ssm 환경에서 그대로 실행 가능. 합성/예측값을
측정 컬럼에 넣는 것은 금지 원칙이라 **실측 CSV는 생성하지 않음** (해당 GPU에서 실행 필요).

**SLURM 실행 경로 (`slurm/sweep_stage1_v2.sbatch`):**
- 파티션 `amd_a100nv_8` (A100 NVLink = SXM4-80GB). PCIe 파티션(`amd_a100_4`)으로 잘못 제출해도
  runner의 **device guard**가 `torch.cuda.get_device_name`을 정규화해 `a100_sxm4_80gb`이 아니면 abort →
  잘못된 디바이스 데이터가 SXM4 트리에 섞이는 것을 코드 레벨에서 차단 (현 PCIe 박스에서 abort 동작 확인).
- **mamba_ssm 블로커 해소:** prebuilt `selective_scan_cuda`가 현 torch와 ABI 불일치
  (`undefined symbol: c10::cuda::c10_cuda_check_implementation`)라 `import mamba_ssm` 자체가 실패.
  chunked 경로는 Triton 커널(`mamba_chunk_scan_combined`)만 쓰므로 `_mamba_compat.ensure_mamba_importable()`가
  깨진 compiled ext를 stub 처리해 패키지 임포트를 통과시킴 (Triton 커널은 정상, `initial_states` 지원 확인).
  sbatch는 `LD_LIBRARY_PATH`에 cuda13 `libcudart.so.13`도 추가(보조).
- 제출: `sbatch slurm/sweep_stage1_v2.sbatch` (3모델) 또는 `sbatch slurm/sweep_stage1_v2.sbatch zamba2`.
  산출: `results/stage1_v2/{ssm,attn}_chunked_{model}_a100_sxm4_80gb.csv` + `reports/matched_granularity_{model}.md`.

### 완료 (GPU 불필요, 본 환경에서 검증됨)

- **Phase 0** 감사 9항목 (본 문서).
- **Phase 1** `shared/configs/sweep_spec.yaml` + `shared/sweep_spec.py`:
  `load_spec`, 격자 accessor, `n_blocks`(모델별 ssd_chunk 반영 — Nemotron 128), `saturation_point`(단일),
  `canonical_filename`/`canonical_device`(언더스코어), `assert_on_sm_grid`(snapping 예외),
  `CHUNKED_FIELDNAMES`/`chunked_row`(통일 스키마). 스모크 테스트 통과.
- **Phase 2 코드** `chunked_attn_runner.py`(512 chunk·누적 KV·CUDA event 전체 구간·SDPA 동일 백엔드),
  `_attn_chunked_worker.py`(subprocess isolation), `run_attn_chunked_sweep.py`(spec 격자·stage1_v2 출력),
  `run_ssm_chunked_v2_sweep.py`(기존 SSM core 재사용·통일 스키마), `analyze_matched_granularity.py`.
  ssm/attn v2 행 **스키마 diff 0 검증**.
- **Phase 3 코드/설정** `models.yaml`·`sweep_spec.yaml` Nemotron-H 실측값 확정 기입
  (52 layers, pattern 24M/4Attn/24FFN, mamba_num_heads=128, head_dim=64, d_state=128,
  **chunk_size=128**, n_groups=8, attn 32/8 heads head_dim128, intermediate 21504).
  chunked v2 경로는 `get_model_config` 직접 사용이라 extractor 없이 Nemotron 지원.
  `plot_motivation.py` `_DEMO` suffix 강제(범용 demo + 자동 폴백 모두) + 단일 `saturation_point` import.
- **Phase 4 부분**: saturation 단일화 — `plot_saturation`, `analyze_decode_sat`, `plot_motivation`,
  `compute_decision_matrix` 4곳 전부 `shared.sweep_spec.saturation_point` 위임 (인라인 marginal 로직 0건).
  `n_blocks` `batch×seq//4` 잔존 제거(`run_ssm_two_pass_sweep.py`). BW 정본 2039로 통일
  (sweep_spec + hardware.yaml sxm4). **Phase 4.4 회귀검증 3/3 일치** (위 절).

### 미완 (선택: 대규모 기계적 리팩터 / 또는 SXM4 실행 필요)

- **Phase 2/3 실측 CSV 생성** — SXM4 + mamba_ssm 환경에서 아래 실행:
  `run_ssm_chunked_v2_sweep.py`, `run_attn_chunked_sweep.py` (각 모델), 이어 `analyze_matched_granularity.py`.
- **Phase 4.1** `decode_sm_scaling/run_decode_sweep.py`의 `SM_PRESETS`/context 격자를 sweep_spec 참조로.
- **Phase 4.2** 분석 스크립트 glob을 `canonical_filename` 기반으로 (구 하이픈은 읽기전용 호환).
- **검증 체크리스트 "grid 리터럴 sweep_spec 외부 0건"** — 다수 runner의 argparse default 격자를
  sweep_spec 참조로 교체하는 기계적 작업 (run_ssm/attn/mlp/decode_sm sweep 등). 본 환경에서 실행 검증
  불가하여 미적용 — 기존 정상 runner 회귀 위험 대비 가치 낮아 보류.

