# s8p_prefill — 7-8B prefill-SM 스윕 설계 (2026-07-28, 정본 아님)

이 캠페인은 **주장 job이 아니다.** `s0_deconfound/DESIGN.md` §5 사전등록이 계속
적용된다 — claims-auditor 통과 전엔 어떤 결론도 정본 인용 금지.

## 0. 목적과 지위 (오독 방지 — 반드시 인용 시 이 문단 함께)

`results/s8_scaleup/`가 4개 7-8B arm에서 **decode**-SM 민감도를 측정했다
(prefill 16 SM 고정, decode 16→92 스윕, matched-batch 2.36–2.91×). 본 캠페인은
그 **거울상**으로 **prefill**-SM 민감도를 같은 4 arm에서 측정한다.

**셀 = `[prefill, decode, idle]`**: `[16,16,76]`, `[24,16,68]`, `[44,16,48]`,
`[92,16,0]`. decode는 **모든 셀에서 16 SM 고정**, prefill만 움직인다. 남는 SM은
의도적으로 논다.

⇒ **이것은 prefill 쪽 등량곡선이지 정책 비교가 아니다.** 실제 정책에서는
prefill이 안 쓰는 SM을 decode가 가져가므로 `prefill_SM + decode_SM ≤ 108`
제약이 걸린다. 여기서는 그 제약을 의도적으로 어긴다(저-prefill 셀은 대부분의
SM을 놀린다). 따라서 본 결과는 **레버의 존재**만 확립하며, **레버를 움직여
서빙 goodput이 나아진다는 근거가 아니다**(정본 방법론 게이트 #1). s8_scaleup
FINDINGS §0의 동일 caveat과 정확히 같은 취지다.

이 두 캠페인(s8_scaleup 축=decode, s8p_prefill 축=prefill)을 합치면 다음
8B 프론티어 캠페인이 필요로 하는 `ITL(D) vs TTFT(108−D)` 프론티어의 **절반**
(TTFT축)이 채워진다. 프론티어 자체(양쪽 동시에 108 제약 하에 트레이드오프)는
여기서 측정하지 않는다.

## 1. arm (s8_scaleup과 동일)

| arm | 모델 | 계열 | backend |
|---|---|---|---|
| M8 | `hf_cache/mamba-codestral-7b-sglang` (Mamba-Codestral-7B, 7.3B) | pure SSM | triton |
| Ha8 | `Zyphra/Zamba2-7B-Instruct` | additive hybrid | triton |
| Hs8 | `nvidia/Nemotron-H-8B-Base-8K` | substitutive hybrid | **flashinfer 필수**(server_args.py:1959가 triton 금지) |
| T8 | `Qwen/Qwen2.5-7B` | pure Transformer | triton |

backend 비대칭은 s8_scaleup과 동일 이유로 불가피(Zamba2-7B는 flashinfer에서
cudagraph capture가 죽음). Hs8이 낀 cross-arm 비교는 이 caveat을 승계한다.

## 2. 텔레메트리 스키마 확인 (측정 지표 결정)

`PDMUX_TELEMETRY_PATH`의 `runtime_snapshot` 이벤트 필드를 실제 로그
(`s8_scaleup/s8_coupled_Ha8_C4096_d16_865312_telemetry.jsonl`)에서 확인했다:

```
active_decode_sequences, active_worker_leases, architecture, context_max,
context_p50, context_p95, decode_batch_size, decode_idle_ratio,
decode_iterations, decode_last_tpot_ms, decode_ready_queue_depth,
decode_running_batch_size, decode_sms, decode_step_count, event,
expected_remaining_output_p90, itl_slo_ms, kv_*_occupancy,
measured_itl_ewma_ms, measured_itl_p95_ms, oldest_prefill_age_ms, phase,
prefill_active_batch_size, prefill_admission_block_reason,
prefill_admission_blocked, prefill_admission_latency_ms,
prefill_chunk_progress, prefill_idle_ratio, prefill_queue_age_ms,
prefill_queue_depth, prefill_sms, running_batch_occupancy, sample_index,
stream_index, timestamp_monotonic_s, timestamp_s, ttft_risk,
worker_overlap_ratio, ...
```

**결과: per-prefill-forward-duration 필드는 존재하지 않는다.** 소스
(`sglang/srt/multiplex/dual_worker.py:604-625`, `multiplexing_mixin.py:195-254`)를
직접 대조해도 prefill 쪽 유일한 시간 필드는 `prefill_queue_age_ms`(대기 시간,
포워드 시간 아님)·`prefill_admission_latency_ms`(admission 지연)뿐이고, decode
쪽에만 `decode_last_tpot_ms`가 있다. `prefill_sms`/`decode_sms`는
`dual_worker.py:608,622-623`에서 `arbiter.sm_counts[stream_index]`를 그대로
찍은 것이라 **realized 파티션 판정에는 직접 쓸 수 있다**(스펙의 게이트 1은
이 필드로 바로 구현했다 — sm_counts를 다시 끌어올 필요 없음).

⇒ **1차 지표 = client TTFT를 realized 파티션에 귀속**(스펙 §fallback 2 채택).
`max_new_tokens=1`이므로 client round-trip latency ≈ TTFT(1개 토큰만 있으므로
스트리밍 없이도 요청 전체 latency가 곧 prefill-dominated latency; 소수 ms의
디코드 스텝 1회분 오차는 있으나 s8의 decode ITL 절대값(수십 ms)보다 훨씬
작지는 않을 수 있음 — 한계로 기록).

### 2.1 개정(2026-07-28, coordinator 지시) — 헤드라인을 TTFT 절대비에서 **기울기 비**로

`max_new_tokens=1` 요청의 TTFT는 `admission wait + prefill forward(L) + decode
step 1회 + detok/network`의 합이고, prefill 컴퓨트가 아닌 항들은 **SM 축과
무관한 큰 상수**로 들어온다. decode 캠페인(s8_scaleup)의 steady-state ITL은 이
오염이 무시할 수준이었지만, 여기선 크다 — 파티션 간 TTFT **절대비**는 이
상수 때문에 민감도를 **체계적으로 과소평가**한다(합성 데이터로 직접 확인:
진짜 기울기비 3.33×를 주입했을 때 절대비는 L=512에서 1.67×, L=4096에서도
2.77×까지만 수렴 — §9 검증 참고).

⇒ **헤드라인 = 셀별 TTFT(p50) ~ L 회귀의 기울기 비**(`s8p_analyze.py`
"PRIMARY" 절). 절편이 상수 오염항을 흡수하고 기울기(µs/token)가 prefill 컴퓨트
처리율의 역수다. TTFT 절대비 표는 **AUXILIARY(보조)**로 강등해 유지한다.

- 회귀는 **rep별로 별도 수행**(rep 풀링 금지 — claims-auditor가 이전 분석기의
  rep 풀링·CI 부재를 잔존 confound로 지적함). rep마다 (L, TTFT p50) 4점으로
  OLS 적합 → 기울기·절편 1쌍. reps≥4개면 **기울기 표본 n≥4**, 셀별 mean±sd
  병기. n<4인 셀은 인용 금지로 표시.
- **곡률(L-의존성) 별도 보고**: 4개 L점에서 나오는 3개 구간별 기울기
  (512–1024, 1024–2048, 2048–4096)를 rep별로 구해 같은 방식(n≥4, mean±sd)으로
  집계한다. attention이면 구간 기울기가 L에 따라 오르고, SSD면 평평해야
  한다 — 이 캠페인 고유의 모델 판별 신호. **4096까지로 곡률이 안 잡히면
  그 사실을 한계로 기록**한다(전체 2차항 적합은 4점으로는 자유도 1뿐이라
  보류 — L_LIST를 8192까지 확장하는 옵션을 §7에 적어둔다).
- **절편도 셀별로 보고**하고, 같은 arm 안에서 셀 간 절편이 크게(스프레드
  >30%) 움직이면 그건 admission/스케줄링 경로가 SM 축에 따라 오염됐다는
  뜻이므로 `s8p_analyze.py`가 자동으로 경고를 찍는다 — **한계로 기록**하고
  기울기비 헤드라인을 그 caveat과 함께 읽는다.

## 3. 워크로드 (s8_scaleup과 역할 반전)

- **공존 부하 = decode**: closed-loop, 짧은 프롬프트(8 rep phrase, ~64 토큰) +
  긴 생성(`ignore_eos`, `--decode-out-tokens 512`), `--decode-conc 8`로 상시
  in-flight 유지, 완료 시 즉시 재발사(s0dc_client.py `worker()`와 동일 패턴).
  s8 §4-2 교훈: `multiplexing_mixin.adjust_stream_groups`는 `split_prefill_batch`와
  `running_batch`가 **동시에** 빈 상태에서만 무분할 (0,108)로 되돌아간다
  (`multiplexing_mixin.py:726,745-748`) — decode를 상시 채워두면 그 경로 자체는
  피할 수 있지만, prefill probe가 직렬(동시성 1)이라 요청 사이 간극에서
  `split_prefill_batch`가 잠깐 `None`이 되는 순간은 남는다(§7 한계).
- **측정 대상 = prefill probe**: `max_new_tokens=1`, **동시성 1**(직렬) 반복
  발사. 부하 민감도 확인용으로 `--probe-conc 2`도 1 rep 찍는다(스펙 요구).
- **L 축**: 각 셀 안에서 prompt 길이 L ∈ {512, 1024, 2048, 4096}을 프로브마다
  순환(`s8p_client.py:probe_worker`, thread offset/stride로 다중 스레드 시
  위상을 어긋나게 함). s8과 동일한 "L/8회 반복 phrase" 방식으로 생성했고
  Qwen2.5-7B 토크나이저로 L/8회 반복 → L+1 토큰임을 직접 검증했다(512→513,
  1024→1025, 2048→2049, 4096→4097 토큰). 부팅 비용은 셀당 1회뿐이므로 L 축은
  거의 공짜다.

## 4. config 관례

`pdmux_pf{16,24,44,92}_d16.yml` — `sm_group_num: 3`, 단일
`manual_divisions: [[prefill_sm, 16, idle_sm]]`. decode가 모든 셀에서 16
SM으로 고정되므로 `decode_states={16}`이 `_build_r2_policy`의
"decode_states must include one of {16,24,34,44}" 가드를 그 자체로 만족한다
(`multiplexing_mixin.py:127-142`) — s8의 d92 셀이 필요로 했던
guard-satisfier용 2번째 division이 **필요 없다**(decode가 92가 아니라 항상
16이므로). `PDMUX_R2_POLICY=fixed` + `PDMUX_R2_FIXED_DSM=16`.

p16/p92 셀은 s8_scaleup의 기존 `pdmux_p16_d16.yml`([16,16,76])·
`pdmux_c_d16.yml`([92,16,0])과 **값이 동일**하지만, 이 캠페인 디렉터리를
자기완결적으로 만들기 위해 사본을 새로 두었다(심볼릭 링크 대신 — 재현성
문서화 원칙상 값을 config 파일 자체에 주석으로 명시하는 편을 택함).

## 5. 사전등록 게이트 (측정 전 확정 — claims-auditor가 이 기준으로 판정)

1. **realized prefill-SM 게이트 ≥ 0.80**: `prefill_active_batch_size > 0`인
   `runtime_snapshot` 샘플 중 `prefill_sms`가 셀의 target과 일치하는 비율.
   target(정책 요청값)이 아니라 **realized** 필드로 판정
   (`prefill_pin_check.py`) — Stage 0가 이 구분을 놓쳐 무효화된 전례
   (`results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md`)를 반복하지 않는다.
2. **co-residency 게이트 ≥ 0.80**: prefill-active 샘플 중
   `decode_running_batch_size > 0`인 비율(`prefill_pin_check.py`의
   `CO_RESIDENT_frac`). 미달 셀은 결과 인용 금지로 표시.
3. **n ≥ 4 reps** (`REPS=4`), 셀·L별 표본 수를 표에 병기. 희박 칸(특히
   attribution 후 `n<50` 슬라이스, s8의 §2.1 교훈과 동일) 인용 금지.
4. **공변 확인**: L 분포가 셀(=realized SM)과 무관한지, `prefill_active_batch_size`가
   셀마다 안정적인지(즉 probe_conc=1이 실제로 그렇게 realize되는지)를
   `s8p_analyze.py`의 gate-#4 절에서 **측정해서 보고**(가정 금지).

게이트 실패 셀은 그 셀의 숫자를 헤드라인/정본 인용에서 제외하고 "as-run"
기록으로만 보존한다(s8 NOTES_865311_865312.md와 동일 정책).

## 6. 운영

- 부팅 전 `scripts/bootstrap/sync_engine_tree.sh "$HERE/runtime_source_manifest.sha256"`
  로 manifest를 이 캠페인 디렉터리에 자체 보관.
- QOS(제출 4/실행 2)에 맞춰 arm을 job 내부에서 순차 실행(s8 방식), **2 job**으로
  분할 제출: `ARM_IDS="0 1"`(M8, Ha8) / `ARM_IDS="2 3"`(Hs8, T8). 두 job이
  동시에 뜰 수 있으므로 `TRITON_CACHE_DIR`을 **job별**로 분리했다
  (`$ROOT/.../triage/.triton_cache_s8p_${SLURM_JOB_ID}`) — s8은 job을 순차
  제출해 이 문제를 겪지 않았지만 이 캠페인은 병렬 제출이라 캐시 경합 위험이
  실재한다(프로젝트 정본 gotcha: "triton cache race").
- fail-fast: `PIPESTATUS[0]`로 client 종료코드를 확인(tee가 삼키지 않게,
  865006/§4 교훈 승계).
- 루트에 `.out/.err` 남기지 않도록 `#SBATCH --chdir`과 `--output/--error`에
  캠페인 디렉터리 **절대경로**를 명시했다(제출 cwd와 무관하게 항상 이
  디렉터리로 떨어짐).

## 7. 알려진 한계 (인용 시 필수)

- **direct forward-time 계측 없음**(§2) — client round-trip latency가 유일한
  1차 지표. TTFT-proxy이지 순수 prefill forward 시간이 아니다(디코드 요청
  admission·큐잉·network round-trip 오차 포함, guard=3s 창으로 극단값만 배제).
- 직렬 probe와 상시 decode 사이의 완전한 co-residency는 **보장이 아니라
  측정 대상**(§5 게이트 2)이다. 특히 probe 간 짧은 간극에서
  `split_prefill_batch`가 일시적으로 비어 legacy `adjust_stream_groups` 경로로
  넘어갈 가능성이 이론상 있다 — 매 회귀 필터링(§5 게이트 1)이 이를 걸러낸다는
  전제이며, 게이트가 FAIL이면 그 셀은 인용하지 않는다.
- backend 비대칭(Hs8만 flashinfer) 승계, 크기는 s8 §1의 실측 offset(+2.3%
  attention 경로, Qwen d44 occ-matched)과 같은 자릿수로 가정하되 여기서
  재측정하지 않았다.
- `--disable-piecewise-cuda-graph`, `--disable-overlap-schedule`,
  `--chunked-prefill-size -1`, `--disable-radix-cache`, `max-running-requests 48`,
  A100 80GB 1장, TP1 — s8과 동일 운영점.
- goodput·SLO·정책 비교는 측정하지 않는다(§0).
- **곡률 해상도 한계(§2.1)**: L_LIST={512,1024,2048,4096}는 3개 구간 기울기만
  주고 독립적인 2차항 적합(자유도 1)은 신뢰하지 않는다. attention의 O(L²)
  곡률이 이 범위에서 노이즈와 구분 안 되면(구간 기울기가 통계적으로 평평),
  그건 "SSD와 동일"이 아니라 **"이 L 범위에서 곡률 미검출"**로 기록해야
  한다. **미실행 옵션**: `L_LIST="512 1024 2048 4096 8192"`로 확장하면 4개
  구간(6점)이 되어 곡률-노이즈 분리가 개선된다 — 부팅 비용은 셀당 1회뿐이라
  거의 공짜지만, `--context-length`/`--max-running-requests` 여유·decode
  background와의 KV 경합을 재검토해야 하므로 이번 스윕에서는 바꾸지 않는다.

## 8. 스모크 테스트 계획 (본 스윕 제출 전 필수)

단일 arm(T8, 가장 가벼운 순수 Transformer라 부팅이 빠름) × 2 셀(p16, p92 —
양 극단) × 짧은 window(WARMUP_S=10, MEASURE_S=20, REPS=1)로 확인:
  (i) 서버 부팅 성공,
  (ii) `prefill_pin_check.py`가 realized prefill pin PASS/FAIL을 산출(FAIL이어도
      무방 — 파이프라인이 정상 작동하는지가 목적),
  (iii) `s8p_analyze.py`의 귀속 표본 수가 0이 아님(클라이언트 시간원점 버그
      재발 방지, s8 §4-4).
스모크가 (i)-(iii)을 통과하면 본 스윕(4 arm × 4 cell × 4 L × REPS=4)을
2 job으로 제출한다. 스모크 실패 시 원인 규명 전까지 본 스윕 제출 보류.

## 9. 산출물 매핑

- 설계: 이 파일.
- 하네스: `s8p_sweep.sbatch` (`s8_sweep.sbatch` 미러).
- config: `pdmux_pf{16,24,44,92}_d16.yml`.
- 클라이언트: `s8p_client.py` (`s0dc_client.py` 미러, 역할 반전).
- 게이트: `prefill_pin_check.py` (`realized_pin_check.py` 미러, 역할 반전).
- 분석: `s8p_analyze.py` (`s8_analyze.py`+`s8_batch_matched.py` 미러, 2026-07-28
  개정 — PRIMARY 절이 §2.1의 rep별 TTFT~L 회귀·기울기비·구간별 곡률·절편안정성
  을 내고, 구 절대비 표는 AUXILIARY로 강등). 회귀 로직은 합성 데이터
  (기울기비 3.33× 주입, base 80ms/78ms 절편, 노이즈 2–3ms)로 단위검증했다 —
  복원된 헤드라인 기울기비가 주입값과 소수점까지 일치(199.79/59.99=3.33×),
  AUXILIARY 절대비는 예상대로 L=512에서 1.67×까지만 과소평가되고 L=4096에서도
  2.77×로 3.33×에 못 미침(§2.1 서술과 일치, 합성 스크립트는 재현용으로 보존하지
  않음 — 필요 시 이 문단의 파라미터로 재생성 가능).
- manifest: `runtime_source_manifest.sha256` (부팅마다 자동 갱신).
- 결과 로그: `s8p_<arm>_<job>_result.txt`, `s8p_<arm>_<cell>_<job>_telemetry.jsonl`,
  `s8p_<arm>_<cell>_<job>_rep<r>_{raw.jsonl,clocks.csv}`, `s8p_<arm>_<cell>_<job>_srv.log`.
