# 구현 결함 B1·B2 크기 사전 점검 — Phase 0 (C1 HE0, C2 VESSL), 2026-10-28

> 작성: result-analyst, main 트랙. CPU 재집계만 했다(GPU 0, 엔진 수정 0, 정본·tracks 수정 0, 커밋 0).
> **새 성능 판정 0건.** 이 문서의 수치는 결함 크기의 사전값(prior)을 잡기 위한 기술값이다. HE0·N3·N2 등급을 바꾸지 않는다.
> 라벨: [측정] 원자료에서 직접 센 값 · [유도] 측정값을 식·코드 규칙에 넣어 만든 값 · [코드] 소스 사실 · [가정] · [인용] 기존 문서 값(재계산 안 함).
> 대상 코드: `src/multiplex/multiplexing_mixin.py` `event_loop_pdmux`(현 HEAD `3858f129` 기준 `:1240-1545`).
> drain 구조(`_tgt != stream_idx` 분기의 2× synchronize `:1341-1345`, `elif adjust_stream_group` 분기 `:1358-1362`, decode-empty 트리거 `:1304-1306`)는
> HE0 트리 `fdbf5d1`, N3 `38ed3f4`/`93c3cca`, N2 `7e09cc3`, smoke `71f5006`에서 같다 [코드, `git show <c>:…` grep으로 확인].

## 0. 코드 독해 검증 (호출자 메모 대조)

| 호출자 메모 | 판정 | 근거 |
|---|---|---|
| B2 drain은 **이미 발행된** chunk만 기다리고, 발행 안 된 나머지 층은 idx 0(108 SM)에서 돈다 | 맞음 | [코드] drain 뒤 `prefill_stream = stream_groups[0][0]`; 다음 반복부터 chunk가 그 stream에 발행됨 |
| 입장(admission)·merge 직후 adjust drain은 prefill stream에서 기다릴 게 없다 | 맞음, **실측 확인** | [코드] merge는 `prefill_exe_done.query()`가 참일 때만, 입장은 `split_prefill_batch is None`일 때만. [측정] gate truth: 입장 drain 23건 prefill_sync p50 0.022 / max 0.153 ms, merge drain 28건 max 0.021 ms(§C2-A) |
| in-flight prefill을 기다릴 수 있는 drain = SLO 전환(B1) + split 진행 중 decode-empty(B2) 둘뿐 | 맞음 | [코드] 위 두 경우 외에는 prefill stream이 비어 있다 |
| chunk가 반복마다 하나씩 쌓인다(prefill 미대기, decode stream만 sync) | 맞음, 실측 확인 | [측정] N3 B2 사건 728/865건 중 717/850건이 이미 **모든 chunk 발행 완료**(`prefill_kernel_done_wait`) 상태에서 decode가 비었다 |

**추가로 확인된 코드 사실(호출자 메모에 없던 것)**:
1. [코드] SLO/R2 컨트롤러는 `not wait_prefill_kernel_done`일 때만 평가된다. 즉 **span의 입장 반복(chunk 0개 발행)과 반복 1…n_chunks−1에서만** 전환이 일어날 수 있다. 마지막 chunk를 발행한 뒤(WAIT 구간, N3에서 span 시간의 약 90% [인용 DIAG §3-1])에는 B1이 구조적으로 불가능하다. → **B1이 in-flight chunk를 기다리려면 span이 2 chunk 이상이어야 한다.**
2. [코드] 결정 53의 `switch_count`(컨트롤러 결정 변경 = `SLO-BIND`/`SLO-SCHED` 줄, R2 `split_transition`)는 **drain 횟수가 아니다.** 비sticky에서는 merge 뒤 decode-only 그룹(`real_sm_group_num−1`)으로 갔다가 다음 입장에서 다시 분할로 오므로, static·dynamic arm 모두 **거의 매 span 입장마다 drain**을 한다(빈 prefill stream). gate truth: 4 boot에서 drain 53건 / span 50개 [측정].
3. [코드] B2(decode-empty drain)가 일어나는 동안은 (a) decode batch가 비어 있어 ITL 표본이 생기지 않고 (b) `update_split_prefill_batch`가 `split_prefill_batch`가 있으면 즉시 False를 돌려 새 입장도 막혀 있다. 즉 **drain이 없어도 그 시간 동안 할 수 있는 일이 없다.** drain이 바꾸는 것은 (i) host가 polling 대신 GPU 완료를 기다림(span_end 관측 지연 ≤1 반복이 오히려 줄어듦) (ii) 발행 안 된 나머지 chunk가 108 SM으로 감(가속) 두 가지뿐이다.

---

## C1. HE0 (KISTI, Zamba2-2.7B, rate 3↔12, `sharegpt_vary_bench.sbatch`)

### 질문
dynamic arm 전환 수, 그중 prefill이 진행 중(in-flight chunk 있음)이던 전환의 비율, 전환 직후 decode 간격 증가.

### 방법
- **원자료 전수**: `sharegpt_vary_bench.sbatch`에 `PDMUX_TELEMETRY_PATH`·`PDMUX_PHASE_EVENTS` 설정이 없다 [코드, `:39-43`]. 저장소 `results/slo_sched/`와 이양 번들
  `~/Experiments/KISTI/_migration_2026-09-17/C_results_untracked/{telemetry_jsonl/slo_sched.tar.zst, results_nontelemetry.tar.zst}` 목록을 HE0 33 job 번호·`sgptv`로 grep →
  번들에 있는 것은 client `sgptvHi/Lo_*.jsonl`과 `.out/.err`뿐이고 이미 로컬에 풀려 있다. **server telemetry jsonl은 0개** [측정]. 남은 서버 측 자료는 server log뿐이다
  (`Prefill batch` 배치당 1줄·1 s 해상도·`#new-token`, `Decode batch` 40 decode 반복마다, `SLO-BIND/SCHED a->b`, `SLO-CTLCOST n=` 200 평가마다).
- **in-flight 판별 규칙**(코드 사실 1 이용): 전환 줄은 파일 순서상 직전 `Prefill batch` 줄의 span에 속한다(스케줄러 단일 프로세스 → 줄 순서 = 실행 순서 [코드]).
  그 배치의 `n_chunks = ceil(54 / max(1, 65536 // #new-token))`(L = 54 [인용, Zamba2-2.7B], 예산 65536 [코드, `pdmux_{d16..d44,slo}.yml`]).
  `n_chunks = 1`이면 컨트롤러 평가는 입장 반복 한 번뿐 → 그 전환의 drain은 **빈 prefill stream**(in-flight 0, 확정). `n_chunks ≥ 2`면 "in-flight 가능"(입장 반복인지 k ≥ 1 반복인지는 로그로 못 가름 → 상한).
- **모형 검증**: 예측 평가 수 Σ_{running>0 배치} n_chunks 대 `SLO-CTLCOST` 마지막 n.
- **전환 직후 decode 간격 대리값**: 전환 뒤 첫 `Decode batch` 창의 반복당 시간(#running-req / gen throughput [코드, `scheduler_metrics_mixin.py:542-544`])에서 앞뒤 3창 중앙값을 빼고 ×40 = 창 초과 ms. 대조군 = 같은 run에서 전환 없는 multi-chunk 배치 직후 창. 전환 창의 대조군 내 순위를 낸다.
- **B1 크기 상한**: k번째 반복 전환의 drain ≤ (k/n)·W ≤ ((n−1)/n)·W, W ≤ 해당 arm의 요청별 (입장→첫 토큰) 최대값(`he0_realized_2026-10-01.json` q4) [유도]. 정지 s는 이후 TTFT를 최대 s 늘리므로, HI 요청 중 TTFT ∈ (3000 − s_run, 3000] ms인 비율 = TTFT 쪽 goodput 손실 상한(보수적) [유도].

### 결과

**표 C1-1. arm별 전환 수(결정 53 정의)와 in-flight 가능성** — 각 run = 6 라운드(LO/HI × 3), HE0 정본 33 job(896565/896689/896690 제외)

| arm | n | 전환/run [측정] mean ± SD (run별) | 입장 반복 확정(n_chunks=1, in-flight 0) [유도] | multi-chunk(in-flight 가능, 상한) [유도] | multi-chunk 배치 비율 [유도] |
|---|---|---|---|---|---|
| d44 / d34 / d24 / d16 | 4 각 | 0 ± 0 | — | — | 4.5–6.3% |
| bind+GATE | 9 | 1.56 ± 1.67 ([2,2,0,0,1,5,1,3,0]) | 12/14 | **2/14** (2 run에 1건씩) | 5.2–5.7% |
| bind no-gate | 4 | 11.0 ± 4.76 ([18,8,10,8]) | 39/44 | **5/44** | 5.2–5.6% |
| slo (v7b) | 4 | 15.0 ± 2.0 ([18,14,14,14]) | 43/60 | **17/60** (run당 4–5) | 5.4–5.8% |

- multi-chunk 전환 span은 전부 2–3 chunk(#new-token 1,322–3,089) [측정]. 따라서 in-flight chunk 수 ≤ 2.
- 모형 검증: CTLCOST 줄이 있는 5 run(bind+GATE 857111·857112, slo 859037·859038·859059)의 마지막 n = 1000, 예측 평가 수 1,084–1,102 → `[1000, 1200)` 구간과 일치 [유도]. 나머지 12 dynamic run에는 CTLCOST 줄이 없다(검증 불가).
- 정적 arm도 매 span 입장에서 drain한다(코드 사실 2). 정적 arm의 drain은 전부 빈 prefill stream이다 [코드].

**표 C1-2. 전환 직후 decode 창 초과 시간(대리값)** [측정·유도]

| arm | multi-chunk 전환 창 초과 ms | 같은 run 대조군(전환 없는 multi-chunk 창) 내 순위 |
|---|---|---|
| bind+GATE | 589, 909 | 0.865, 1.00 |
| bind no-gate | 89, 194, 306, 471, 627 | 0.35–0.79 |
| slo | 388–1,042 (9개 창, 8개 창에 전환 2건이 겹침) | 0.78–0.98 |

- 대조군 자체가 넓다: 전환 없는 multi-chunk 창 초과 p50 92–322 ms, p95 381–6,586 ms; single-chunk 창 p50 61–117 ms [측정].
- slo 전환 창은 대조군 상위에 몰린다. 그러나 **SLO-SCHED 1→2는 TPOT-EMA > 51 ms일 때만 발화**한다(`_hi_frac·60` [코드]). 즉 전환 창은 이미 느린 창에서 골라진 것이어서(선택 편향) 이 순위를 drain 비용으로 귀속할 수 없다.

**표 C1-3. B1 크기 상한(보수적)** [유도]

| arm | run당 정지 상한 s_run | HI 요청 중 TTFT ∈ (3 s − s_run, 3 s] | ITL 쪽: 정지가 닿는 요청 수 상한 |
|---|---|---|---|
| bind+GATE (W ≤ 489 ms) | 0.33 s·0.25 s(2 run), 0(7 run) | 3.3%·3.2%(2 run), 0(7 run) → arm 평균 ≤ 0.72% | ≤ 48/run(2 run) |
| slo (W ≤ 538 ms) | 1.26–1.61 s | 10.2–11.2% | ≤ 192–240/run |
| bind no-gate (W ≤ 7,212 ms, 이상치 1건) | 4.8–18.0 s | 45–67% (상한이 정보 없음) | — |

- 이 상한은 "모든 multi-chunk 전환이 마지막 비종료 반복에서 일어났고, W가 arm 최댓값이며, 정지가 HI 구간 모든 요청을 밀었다"는 최악 가정이다. 실제 값은 0일 수도 있다(입장 반복 전환이면 0).
- HE0 판정 방향과의 관계 [유도]: B1은 **dynamic arm만** 벌한다. 따라서 B1을 고치면 dynamic이 좋아질 수만 있다. 정본 dynamic arm(bind+GATE)의 상한은 HI 요청의 평균 ≤ 0.72%로 3% headline 문턱보다 작다. slo arm은 상한 10–11%라 이 데이터로 배제할 수 없다.

### 판별 가능 여부
- **전환 수**: 판별 가능 [측정].
- **in-flight 비율**: **부분 판별.** 하한 0, 상한은 bind+GATE 2/14, no-gate 5/44, slo 17/60. 정확한 값은 **판별 불가**.
  빠진 것은 전환이 span의 몇 번째 반복(split_index)에서 일어났는지다. 필요한 것은 `drain` 이벤트(`split_index`, `prefill_chunks_unresolved`, `prefill_sync_ms`) 또는 최소한 반복 단위 `decode_iteration.stream_idx`인데 HE0에는 둘 다 없다.
- **전환 직후 decode 간격 증가**: **판별 불가.** 빠진 것은 반복 단위 host 시각(`t_launch/t_sync`)이다. 40반복 창 대리값은 해상도가 수십–수백 ms이고 전환 조건이 선택 편향을 만든다.
- **B2(HE0)**: 재구성상 "prefill 진행 중 + decode 비어 있음" 시간이 라운드당 0.021–0.072 s, run당 0.32–0.37 s다. 7개 arm 모두 같은 범위다 [인용 `he0_realized_2026-10-01.json` frac.pt.prefill_only_s]. 이 시간은 라운드 첫 요청(단독 prefill 57–63 ms)과 크기가 맞는다 [인용 q4]. → HE0에서 **span 중간에 decode가 비는 경우는 사실상 없고 arm 대칭이다** [유도].

### 한계
- L = 54는 문서 인용이며 HE0 서버 로그에서 직접 확인한 값이 아니다. 예측 평가 수와 CTLCOST의 일치(5 run)는 간접 확인이다.
- `#new-token`은 radix OFF라 extend 토큰과 같다고 가정했다 [가정, HE0 재구성에서 `sum(input_lens)==#new-token` 33,416 배치 일치 [인용]].
- W 상한은 요청별 (입장→첫 토큰)의 arm 최댓값이다. 클라이언트 시계 재구성 오차를 그대로 물려받는다. no-gate의 7.2 s 이상치 때문에 그 arm의 상한은 정보가 없다.
- 단일 기판(KISTI A100), 단일 모델(Zamba2-2.7B, triton), cudagraph ON. VESSL 수치와 섞지 않는다.

---

## C2. VESSL (N3, N2 Job-EA, N2 Job-PROFILE, T3, engine_gate / policy_gate_smoke)

### 질문
adjust 경로(decode-empty / merge / 입장) drain이 in-flight prefill을 기다린 횟수와 지속 상한. drain 이벤트가 있는 run이면 drain 시간 분포와 in-flight chunk 수.

### 방법
- **A. 실측 truth**: `PDMUX_PHASE_DIAG` `drain` 이벤트가 있는 파일 = `results/engine_gate/pdmux-policy-gate-smoke-71f5006-20261028064810/{on,on_slo,static_ns}/tel.jsonl`, 3개뿐이다
  (저장소 `*.jsonl` 전수 grep [측정]; stage2 smoke `cap_char/policy_runs/…/smoke/{s16,s64,dyn}`에는 없다). 분류: `reason`, `split_present`, `split_index`, `running_bs`, `prefill_kernel_done_wait`, `prefill_chunks_unresolved`.
- **B. 재구성**(drain 이벤트 없는 run): `decode_iteration`은 decode가 돈 반복에서만 나온다 [코드]. 그래서 "stream_idx → 0 전이"는 직접 보이지 않는다(idx 0 = decode 빈 상태라 기록 자체가 없다).
  대신 span 단위 서명을 쓴다: span 시작 시 running_bs > 0, span 안에 decode 반복이 ≥ 1개, span 종료 뒤 첫 decode 반복의 bs ≤ n_done(옛 요청이 merge 전에 모두 끝남).
  지속 상한 = `span_end − t_sync(span 안 마지막 decode 반복)`이다. 여기에는 drain + 빈 루프 + 남은 chunk 발행 + span_end 관측 지연이 들어간다.
  span 안 decode 반복이 0개면 "입장부터 decode 빔"(빈 stream drain)으로 따로 센다. `span_start`는 `update_running_batch` 전에 찍혀 끝난 요청이 running_bs에 남기 때문이다 [코드].
  B1형 대리값: (i) R2 `split_transition`의 시각을 span의 첫 decode 반복과 비교해 입장 반복 / span 중간 / span 밖으로 나눈다. (ii) span 안 연속 decode 반복 사이 `stream_idx` 변화(둘 다 `prefill_in_flight`)를 센다. (ii)는 `PDMUX_SLO_SCHED` arm처럼 전이 이벤트를 내지 않는 경로에도 쓸 수 있다.
- n_chunks = ceil(56 / max(1, 65536 // extend)) [인용 `lockstep_diag.py:56-57`].

### 결과

**표 C2-A. drain truth (gate Job `71f5006`, VESSL A100, Nano-9B-v2, PHASE_DIAG ON, 4 boot, span 50개)** [측정]

| 분류 | n | prefill_sync ms p50 / p95 / max | decode_sync ms max | prefill_chunks_unresolved | split_index |
|---|---|---|---|---|---|
| adjust · 입장(split_index 0) | 23 | 0.022 / 0.127 / 0.153 | 0.096 | 전부 0 | 전부 0 |
| adjust · merge 뒤(split 없음) | 28 | 0.016 / 0.019 / 0.021 | 0.008 | 전부 0 | — |
| slo_switch | 2 | 0.010 / — / 0.077 | 0.006 | 0, 0 | 0, 0 |
| adjust · decode-empty **in-flight** | **0** | — | — | — | — |

→ 이 Job의 drain 53건은 모두 빈 prefill stream이었다. prefill_sync 합계 1.87 ms, decode_sync 합계 0.49 ms [측정]. in-flight drain의 양성 사례는 **0건**이라, 재구성 추정기를 양성 쪽에서 검증하지 못했다(음성 쪽만 일치: 추정기도 0).

**표 C2-B. 재구성** [측정·유도]

| 데이터 | span | 입장 시 decode 빔 | 입장부터 빔(빈 drain) | **B2 in-flight** (span 대비) | 모든 chunk 발행 후 빔 | 지속 상한 ms p50 / p95 / max | 상한/span 시간 p50 / p95 |
|---|---|---|---|---|---|---|---|
| N3 Job 3 `38ed3f4` (43 파일) | 1,626 | 860 | 38 | **728 (44.8%)** | 717 | 849 / 1,549 / 1,640 | 0.46 / 0.68 |
| N3 Job 4 `93c3cca` (48) | 1,889 | 1,022 | 2 | **865 (45.8%)** | 850 | 888 / 1,559 / 1,571 | 0.53 / 0.67 |
| N3 `79999aa` (1) | 28 | 27 | 1 | 0 | — | — | — |
| N2 Job-EA `7e09cc3` (16) | 1,360 | 256 | 0 | 40 (2.9%) | 40 | 0.48 / 0.59 / 0.61 | 0.0003 / 0.0005 |
| N2 Job-PROFILE `a5de2d6` (28) | 1,469 | 581 | 1 | 0 | — | — | — |
| policy smoke stage2 `71f5006` s16/s64/dyn | 328 | 14 | 0 | 0 | — | — | — |
| gate `71f5006` (4) | 50 | 18 | 0 | 0 | — | — | — |

**B1형(span 중간 전환)** [측정]:
- R2 `split_transition` 전수: N3 Job 3 953건(입장 반복 706, span 밖 247, **중간 0**), Job 4 1,110건(865 / 245 / **0**), N2-EA 112건(112 / 0 / **0**), N2-PROF 215건(212 / 3 / **0**), smoke 166건(166 / 0 / **0**), gate 1건(입장).
- span 안 in-flight `stream_idx` 변화: 모든 데이터에서 **0건**(smoke dyn = `PDMUX_SLO_SCHED` arm 포함).
- 이 run들은 fixed 정책이거나 anchor에 머문 dyn뿐이다. 따라서 "B1이 VESSL에서 0"은 **B1이 일어날 기회가 거의 없던 구성에서의 0**이다.

**T3(cap_char costterms)**: 로컬에는 `t3_iterations.json`(동거 간격 배열 `co_ms`)만 있고 span·빈 decode 정보가 없다. 원자료는 클러스터 볼륨 `/data/costterms/*`에 있다(`RAW_LOCATION.txt`). → **판별 불가.**
**engine_gate decomp `d1f2ddb` TEL**: `tel.jsonl`이 m12에만 있다 [인용 DIAG §2]. → **판별 불가.**

### 판별 가능 여부
- gate Job: **판별 가능**(직접 측정). in-flight drain 0건, drain 전체 ≤ 0.153 ms.
- N3·N2-EA·N2-PROF·smoke: **B2 빈도와 지속 상한은 판별 가능**(재구성, 상한). drain 자체의 실제 길이(상한 안의 몇 %가 synchronize였는지)는 **판별 불가**다. 빠진 것은 `drain.prefill_sync_ms`다.
- T3, decomp: **판별 불가**(원자료 원격).

### 한계
- 추정기의 양성 검증이 없다(truth Job에 B2 in-flight 사례가 0). 지속 상한에는 drain이 없었어도 생겼을 빈 루프 시간이 들어 있다. 그래서 이 값은 "**drain이 막을 수 있었던 최대 시간**"이지 "drain이 만든 지연"이 아니다.
- "bs ≤ n_done" 서명은 merge 직후 즉시 끝나는 요청(출력 1토큰)이 있으면 과소 계수한다. N3·N2는 출력 64·고정 길이라 영향이 작을 것으로 본다 [가정].
- 단일 모델(Nano-9B-v2, flashinfer), VESSL A100-SXM4, 예산 65536. KISTI HE0와 한 풀로 묶지 않는다.

---

## B2 크기 사전값 (정직한 한계 포함)

| 항목 | 값 | 라벨 |
|---|---|---|
| 빈도 — 저부하·짧은 출력(N3: 8k/64, rate 0.10–0.77) | span의 44.8–45.8% | [측정, 재구성] |
| 빈도 — 동거 부하(N2-EA) | 2.9%, 지속 ≤ 0.61 ms | [측정] |
| 빈도 — HE0(ShareGPT, rate 3↔12) | span 중간 발생은 사실상 0. "prefill만" 시간 run당 0.32–0.37 s, arm 대칭 | [유도·인용] |
| 지속 상한(N3) | p50 0.85–0.89 s, p95 1.55 s, span 시간의 0.46–0.53배(p50) | [측정, 상한] |
| 사건 중 "나머지 chunk가 108 SM으로 옮겨진" 비율 | N3 11/728, 15/865(1.5–1.7%) | [측정] |
| **ITL·goodput 효과** | **≈ 0**: B2 동안 decode가 비어 ITL 표본이 없고, split prefill이 진행 중이라 입장도 막혀 있다. drain이 없어도 그 시간에 할 일이 없다. 효과는 나머지 chunk 가속(사건의 ≤ 1.7%)과 span_end 관측 지연 제거뿐이며 둘 다 TTFT를 **줄이는** 방향이다 | [코드 유도] |
| arm 비대칭 | 가속분이 prefill SM이 적은 arm(D 큰 arm)에서 더 크다. 크기는 미측정이지만 사건 비율 ≤ 1.7% × 남은 층 비율이라 작다 | [유도] |

**한 줄 사전값**: B2는 **흔하지만(저부하 span의 약 45%) 비용이 거의 없는 결함**이다. goodput 효과의 사전값은 0이고, 부호는 0 또는 약간의 이득이다. 이 판단의 근거는 코드 구조(decode 빔 + 입장 차단)다. 실측 drain 시간으로 검증한 것은 아니다(truth 양성 0건). **B1은 반대다**: VESSL 구성에서는 기회가 없어 0이었고, HE0에서는 slo arm 상한(HI 요청 10–11%)을 배제하지 못한다. 정본 bind+GATE의 상한은 ≤ 0.72%다.

다음에 필요한 것: B1 실측. dynamic arm이 multi-chunk span(extend > 65536/L)에서 전환하는 구성으로, PHASE_DIAG `drain` 이벤트(`split_index`, `prefill_chunks_unresolved`, `prefill_sync_ms`)를 켠다. `PREREG_DIAG_stall_drain` D-3 블록이 정확히 이것을 잰다. B2는 같은 Job의 `reason=adjust, running_bs=0, split_index>0` 행으로 양성 검증만 하면 된다.

---

## 재현 (정확한 명령)

```bash
cd ~/Experiments/KISTI/prefill-layer-alloc
python3 workspace/engine-port/results/impl_defects/c1_he0_switch_inflight.py   # ~10 s → c1_he0_switch_inflight.json
python3 workspace/engine-port/results/impl_defects/c2_vessl_adjust_inflight.py # ~30 s → c2_vessl_adjust_inflight.json
```
산출 sha256(이번 실행): `c1_he0_switch_inflight.json 62dc6cbc…7b9`, `c2_vessl_adjust_inflight.json 3029c28d…487`.
C1은 `pdmux_eval.analyze.load_bench_serving_rounds`(라운드 duration 합산 규약)로 client를 읽는다. 분위는 단순 순위 분위(진단용)이며 goodput 판정에는 쓰지 않았다.

## cost model 기여

- **CM-4(동거 간섭 / lockstep 정지 항, `COSTMODEL.md` 기준)** — 알게 된 것: drain 항은 경로에 따라 둘로 갈린다. 입장·merge drain은 **≤ 0.153 ms/건**(VESSL Nano, gate truth 51건)으로 비용 모델에서 0으로 둔다. decode-empty drain(B2)은 시간은 길지만(p50 0.85–0.89 s) **decode·입장이 모두 정지된 구간 안에** 있어 latency 항에 들어가지 않는다. 따라서 drain 비용 항은 "in-flight chunk가 있는 상태의 **전환**(B1)"에만 존재하고, 그 크기 ≤ ((n−1)/n)·W_span이다. 측정값은 아직 없다(HE0 상한만, VESSL은 기회 0).
- 깨진 가정: "switch_count = drain 수"가 아니다. 비sticky 경로는 arm과 무관하게 거의 매 span 입장마다 drain한다(gate 53 drain / 50 span). 전환 비용을 switch_count에 곱하는 모형은 잘못이다.
