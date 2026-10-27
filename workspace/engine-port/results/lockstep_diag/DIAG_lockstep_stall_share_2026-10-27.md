# lockstep 정지가 ITL·TTFT에서 차지하는 몫 — 보관 telemetry 진단 (2026-10-27)

> 작성: result-analyst, 2026-10-27, main 트랙. CPU 재집계만 했다(GPU 0). **새 성능 판정 0건. 정본 결론을 바꾸는 문장도 0건이다.**
> 이 문서의 수치는 전부 진단용 기술값이다. 라벨 규약: `measured` = 원자료(telemetry·client 기록)에서 직접 센 값,
> `derived` = 측정값을 조합한 값(비, 회귀, 순서 대응), `cited` = 기존 문서를 재계산 없이 인용한 값.
> 정본(PROJECT_STATUS·CONSENSUS·COSTMODEL)과 tracks는 수정하지 않았다. 반영 여부는 감사 뒤 promote가 정한다.

## 0. 한 줄 요약

- **VESSL A100-SXM4 / Nemotron-Nano-9B-v2-Base / flashinfer / cudagraph ON** run들(N3 76셀, N2 Job-EA 64셀, 비용 항 T3 73 cell-rep)에서는
  **lockstep 정지가 ITL 꼬리를 지배하지 않는다.** 꼬리는 동거 중 decode step(분할 + 동거 간섭)이 만든다. decode 루프는 prefill
  chunk의 GPU 완료를 기다리지 않는다. host는 chunk를 decode 박자로 모두 발행한 뒤 span 시간의 약 90%를 `WAIT`(GPU 잔여 대기)로 보내고,
  그동안 decode는 정상 박자로 계속 돈다.
- **KISTI A100 / Zamba2-7B(Ha8)·Codestral-7B(M8) / triton** run들(C2 s8_scaleup, M4 s8_frontier)에는 **0.17–1.8 s 정지가 실제로 있다.**
  다만 PHASE_EVENTS가 없어서 기전은 판단할 수 없다. 기판·모델·backend·엔진 트리가 위 run들과 모두 달라 한 풀로 묶지 않는다.
- HE0(Zamba2-2.7B, KISTI)에는 phase 계측 run이 **하나도 없다**. 따라서 HE0가 lockstep 산물인지는 **판단 불가(데이터 없음)**다.

## 1. 계측이 기록하는 것 (코드 사실)

`src/multiplex/multiplexing_mixin.py` `PDMUX_PHASE_EVENTS` 블록(`:2129-2302`)의 이벤트 3종은 기존 telemetry writer로 나간다(비동기 JSONL).

| 이벤트 | 방출 지점 | 필드 | 시각 기준 |
|---|---|---|---|
| `prefill_span_start` | `update_split_prefill_batch` 안(`:870`, 새 split-prefill 배치 입장) | `span_seq, n_new, extend_tokens, running_bs, queue_len, cap, stream_idx, prefill_sms, decode_sms` | `timestamp_monotonic_s` = host CLOCK_MONOTONIC(방출 시각) |
| `prefill_span_end` | `_dual_worker_prefill_ready`(`:883`): 마지막 chunk의 `prefill_exe_done.query()`가 TP 전 rank에서 참이 된 루프 반복에서, merge **직전** | `span_seq, n_done, running_bs, queue_len, stream_idx` | host 관측 시각. GPU 완료 시각 이후 **최대 1 루프 반복 늦게** 관측된다 |
| `decode_iteration` | decode 발행 직후 `t_launch`(`:1405`), `decode_stream.synchronize()` 직후 `t_sync`(`:1494`) | `iter_seq, t_launch, t_sync, bs, stream_idx, prefill_sms, decode_sms, prefill_in_flight, prefill_kernel_done_wait, queue_len` | `time.perf_counter()` = CLOCK_MONOTONIC(`SUBSTRATE_N3.txt`: `perf_counter_clock=clock_gettime(CLOCK_MONOTONIC)`) |

- **모든 시각은 host 벽시계다. GPU 타임스탬프는 없다.** `t_sync − t_launch` 구간에는 이번 반복의 prefill chunk host 발행
  (`forward_count = max(1, 65536 // extend_tokens)` 층)과 decode stream의 GPU 잔여 시간이 섞여 있다.
- `decode_iteration`은 decode가 실제로 돈 반복에서만 나온다. decode batch가 비어 있던 구간은 기록되지 않는다(`iter_seq`는 연속으로 남는다).
- 요청 ID는 없다(`request_id: ""`). 그래서 span과 요청의 대응은 순서로만 추정할 수 있다.
- 루프 구조(코드): 한 반복은 recv → 입장(span_start) → [필요하면 drain: `prefill_stream.synchronize()` + `decode_stream.synchronize()`] →
  decode 발행 → chunk 1개 발행 → **`decode_stream.synchronize()`만 수행** → 결과 처리 → [마지막 chunk가 끝났으면 span_end·merge] 순이다.
  **매 반복 동기화되는 것은 decode stream뿐이다.** prefill stream은 `prefill_kernel_done_wait`가 켜진 동안 `query()`로 확인만 한다.
  그래서 "decode가 span GPU 작업을 기다린다"(lockstep-GPU)는 코드상 필연이 아니다. 성립 여부는 데이터로 확인해야 한다.
- `src/multiplex/decode_probe.py`(I-4·I-6, `PDMUX_DECODE_PROBE_PATH`)는 decode forward마다 `decode_fwd_gpu_ms`(decode stream 위
  CUDA event 쌍 = **GPU 시간**), `decode_fwd_host_ms`, `bs_real/bs_padded/graph`를 기록한다. phase 이벤트와는 독립이고 PD-mux·fused 양쪽에 있다.
  **이 진단이 쓸 수 있는 run 중에는 I-4와 PHASE_EVENTS가 함께 켜진 것이 로컬에 없다**(§2).

## 2. 원자료 전수 (PHASE_EVENTS가 켜진 run)

방법: 저장소 안의 `*.jsonl` 7,616개를 전부 `"decode_iteration"`으로 grep해 123개 파일을 찾았다. 그다음 boot·design 파일의 `PDMUX_PHASE_EVENTS=1`로
원자료가 로컬에 없는 run까지 대조했다.

| run (경로 `workspace/engine-port/results/…`) | 기판 | 모델 / backend | arm (D = decode SM) | 부하 | PHASE | 원자료 위치 | 이 진단에서 |
|---|---|---|---|---|---|---|---|
| `n3_capacity/n3b_…-38ed3f4-20261006092813` (N3 3회차) | VESSL A100-SXM4, `GPU-c36b3c0c…`, 드라이버 580.105.08 | Nano-9B-v2-Base / flashinfer, cudagraph ON | target-D44 sticky OFF | random 8192/64, rate 0.10–0.77 + floor(inf) + bridge 1.15 | ON(grid 36 + floor 1) | 로컬 | 분석 |
| `n3_capacity/n3b_…-93c3cca-20261007064124` (N3 4회차) | VESSL, `GPU-024bced7…`, 580.105.08 | 같음 | 같음 | 같음(grid 40 + floor 2) | ON | 로컬 | 분석 |
| `n3_capacity/n3b_…-79999aa-20261006081638` | VESSL | 같음 | 같음 | floor_head 1 | ON | 로컬 | 분석(1셀) |
| `n3_capacity/probe_6group_2026-10-05` | VESSL | 같음 | 프로브 3 boot | warm-up만 | ON | 로컬 | 제외(벤치 창 없음) |
| `n2_static_grid/n2p_…-7e09cc3-20261013022149` (N2 Job-EA) | VESSL `GPU-3be4c7da…`, 580.105.08, 400 W | 같음 | D{16,44} × sticky{ON,OFF} × b{8,32} × c{2048,8192}, round 4 | co-load K = 33(D16)/15(D44) × 8192 토큰 일괄 송신 | ON | 로컬 + **client 토큰 절대시각** | 분석(핵심) |
| `n2_static_grid/n2p_…-a5de2d6-20261012021243` (N2 Job-PROFILE) | VESSL | 같음 | D{16,24,34,44} × sticky ON/OFF 프로파일 | 단독 프로파일 격자 | ON | 로컬 | 서버 측만(보조, §3-5) |
| `cap_char/costterms_runs/…ja1/ja2/jh2` (비용 항 T3) | VESSL | Nano 실가중치·더미 변형 a4/a13/a22/a31 | D{24,44}, bs{8,32}, L{8k,32k} | 통제 동거 | T3 boot ON | **클러스터 볼륨 `/data/costterms/…`(m12, 로컬 없음)**. 로컬에는 파생 `t3_iterations.json`만 | 파생 배열만(§3-4) |
| `engine_gate/pdmux-engine-gate-decomp-d1f2ddb-20261021022225` `b{1..4}p*_TEL` | VESSL | Nano / flashinfer, agnostic arm | HI 4.784 · LO 1.794 req/s | 서빙 | ON(TEL 수준) | **`tel.jsonl` 17 MB가 raw_index에만 있고 로컬 없음(m12)**. 로컬은 `cell_metrics.json`뿐 | 미분석(원자료 부재) |
| 같은 Job `*_I46` | VESSL | 같음 | 같음 | 같음 | OFF(대신 I-4 GPU events ON) | m12 | 미분석 |
| `cap_char/probe_128k`, `costterms_smoke_*` | VESSL | — | — | — | 없음(grep 0) | — | 해당 없음 |
| `s8_scaleup` (C2), `s8_frontier` (M4 = Figure 1 출처) | **KISTI** A100 | Ha8 Zamba2-7B·M8 Codestral(triton)·Hs8 Nemotron-H-8B(flashinfer)·T8 Qwen2.5-7B | C2: `[16, D, idle]`, D{16,24,44,92}+np | keepalive 동거 / ShareGPT rate 2 | **없음**(runtime_snapshot 32 sync마다) | 로컬 | client 간격만(§3-4) |

- 같은 비교 안의 arm은 같은 Job·같은 GPU에서 나왔다(N3는 Job별, N2-EA는 16 boot가 한 Job). **Job 사이에는 풀링하지 않았다.**
- HyMux Figure 1의 "관측 decode 정지 167.7–865.6 ms"는 `s8_frontier/DESIGN.md` §4.3.8(a)(`:719-726`)와 CONSENSUS §1 행 24(KISTI, Ha8 seed 1,
  ShareGPT rate 2, D24→92)까지 추적된다. "span 165→858 ms"와 "decode step 44.0→23.9 ms"는 저장소 grep에서 찾지 못했다 → **정본 출처 미확인 유지**.

## 3. 결과

### 3-1. (a) 겹친 반복의 간격은 decode step인가, span(chunk GPU) 시간인가

정의: 서버 측 토큰 간격 I_k = t_sync_k − t_sync_{k−1}(`iter_seq` 연속일 때만). 반복은 `SOLO`(prefill 없음), `CHUNK`(chunk 발행),
`WAIT`(chunk를 다 발행하고 GPU 완료 대기), `ADMIT`(간격 안에 span_start 있음), `MERGE`(간격 안에 span_end 있음)로 나눈다.
가중치는 두 반복 모두에 있던 요청 수(= bs − 이번에 merge된 n_done)이고, decode가 비어 있던 공백은 제외된다.
**검증**: N3 서버 가중 p50/p95/p99와 client ITL 분위의 차이는 Job 4에서 −1.35…+0.08 ms, Job 3에서 −1.66…+0.09 ms다(셀 42·37개) `measured`.

**lockstep-GPU 가설의 예측과 실측 비교**:

| 데이터 | 가설 예측: 동거 간격 ≈ W / n_chunks | 실측 동거 간격 p50 | 실측: span당 요청별 decode 토큰 | 가설 예측 토큰/span |
|---|---|---|---|---|
| N3 Job 4 (D44, 8k span 7 chunk / 16k span 14 chunk), span 865개 | 중앙 206 ms [137–217] `derived` | CHUNK 20.13–20.78 ms, WAIT 20.08–20.75 ms(셀 40) `measured` | (span당 반복 ≈ 63 = CHUNK 7 + WAIT 56, 예시 셀) `measured` | 7 |
| N3 Job 3, span 728개 | 중앙 210 ms [148–428] | CHUNK 20.36–20.94, WAIT 20.33–21.01(셀 36) | 같음 | 7 |
| N2-EA D16(prefill 92 SM), 14 chunk span | 151–155 ms `derived` | CO 50.6–89.3 ms(b8/b32) `measured`(client 절대시각) | 24–47 `measured` | 14 |
| N2-EA D44(prefill 64 SM), 14 chunk span | 218–223 ms | CO 23.8–40.7 ms | 75–130 | 14 |

- N3 회귀(span 1,587개, x = W/n_chunks 157–428 ms): 동거 간격 중앙 = 0.0015·x + 20.2 ms, 기울기 95% CI [0.0005, 0.0027](bootstrap 10,000, seed 1) `derived`.
  lockstep-GPU 가설은 기울기 ≈ 1을 예측한다. 실측 기울기는 0에 가깝다.
- N2-EA는 (D, b) 층마다 x의 산포가 약 3%뿐이라 층별 기울기(−1.05…+0.72)는 해석하지 않는다. 판별은 위 표의 **수준 차**(간격 3–9배, 토큰 3–9배)로 했다.
- **CPU-launch 가설도 지지되지 않는다.** chunk를 발행하는 반복(CHUNK)과 발행하지 않는 반복(WAIT)의 간격이 같다.
  N3 Job 3: CHUNK p50 중앙 20.61 대 WAIT 20.56 ms, Job 4: 20.32 대 20.31 ms. N2-EA 64셀: 44.27 대 43.88 ms(중앙) `measured`.
  chunk 발행의 host 비용은 decode GPU step 뒤에 숨는다(≤ 0.1–0.4 ms 수준) `derived`.
- span 구조(N3, `derived`): host가 마지막 chunk를 발행한 뒤 span 끝까지 걸린 시간이 W의 중앙 0.90배다(Job 4 0.89–0.96, Job 3 0.44–0.96).
  기대 chunk 수와 실측 발행 반복 수가 일치한 span은 Job 3 718/728, Job 4 854/865다 `measured`. → host는 GPU보다 앞서 달린다.
- 남는 것은 동거 감속이다. N2-EA client 기준 CO p50 / 같은 D의 단독(sticky ON) p50 = **1.156–1.193**(32 boot-cell, n = 4 round/셀) `derived`. 같은 Job의 정본 ρ_on(S-per 기반)과 같은 자릿수다.
  N3의 CHUNK/SOLO 1.50–1.56은 D 108 → 44 변경과 동거가 섞인 값이라 분해하지 않는다.

### 3-2. (b) ITL p95/p99 중 span 겹침 토큰의 비율, 정지 길이

| 데이터 | p95 꼬리 토큰 중 비-SOLO | p99 꼬리 토큰 중 비-SOLO | 정지(> 3× 단독 p50 또는 > 40 ms) | 꼬리 간격의 크기 |
|---|---|---|---|---|
| N3 Job 4 (셀 40) | 1.00 [1.00–1.00] `derived` | 1.00 [1.00–1.00] | client > 40 ms: 6 / 144,396 토큰(4.2e−5), 최대 43 ms `measured` | client p99 21.6–24.9 ms = CHUNK/WAIT 수준 |
| N3 Job 3 (셀 36) | 1.00 [0.98–1.00] | 1.00 [0.92–1.00] | client > 40 ms: 106 / 127,008(8.3e−4), 최대 1.37 s. 서버에서 보인 18건 중 13건이 ADMIT(첫 chunk 반복의 H = 0.66–1.34 s) `measured` | 같음 |
| N2-EA (64셀) | — | CO+CO_ADMIT+CO_MERGE 중앙 0.90–1.00 `derived` | 동거 창 최대 간격 = CO p50의 1.07–1.84배(28.7–100.7 ms) `measured`. 전체 client 간격 > 100 ms = 1,329 / 2,798,433(4.7e−4), 셀당 최대 63개 — 대부분 창 밖 경계 간격 150–345 ms | p99(all) ≈ CO p99 |
| 비용 항 T3 (73 cell-rep, 28,354 동거 간격, L 8k·32k) | — | — | > 5× 단독 0건, max/단독 중앙 1.27(최대 2.89) `derived`(커밋된 파생 배열) | 동거 p50/단독 1.134–1.186 |

- N2-EA의 창 밖 150–345 ms 간격 3개 셀을 열어 보았다. 그 구간 안에서 서버 decode 반복은 정상 박자(14–84 ms)로 여러 번 돌았다.
  즉 이 간격은 **스케줄러 정지가 아니라 client 전달 공백**이다(co-load 송신 순간과 겹침, front-end 기원으로 보이나 미확인) `measured`.
- N3 Job 3의 드문 1 s급 정지 일부는 서버 쪽에서도 보인다. 새 span의 첫 chunk 반복에서 host가 decode 발행부터 sync까지 0.66–1.34 s 막혔고,
  이것은 lockstep형 사건이다. 같은 설정의 Job 4(다른 GPU)에서는 0건이었다. 기전과 GPU·boot 의존성은 **미해결**이고 토큰 비중은 ≤ 0.1%다.
- **정지 길이 = f(span 예산, 108−D)**: VESSL Nano run들에는 정지 모집단이 없어 형태를 적합할 수 없다. 예산은 전 run에서 65536 하나뿐이었다.

### 3-3. (c) TTFT 분해

- **N2-EA co-load(정확 분해, client 절대시각 + span 순서 대응 33/33·15/15 일치)** `derived`:
  TTFT = 대기(송신 → span 입장) + prefill 실행 W(span 시간) + 마지막 span 뒤 병합·전달(span_end → client 첫 토큰).
  - D16(prefill 92 SM): 대기 0.887–0.888 / W 0.112 / 병합 ≤ 0.0007, W(2요청 span) 2.09–2.14 s.
  - D44(prefill 64 SM): 대기 0.772–0.774 / W 0.226 / 병합 ≤ 0.0017, W 2.97–3.00 s.
  - 병합·전달 1.4–31 ms(b와 함께 증가). **TTFT는 직렬 split-prefill(한 번에 1배치) 대기열이 지배한다.**
- **N3(근사)**: 요청 ID가 없고 span 수가 요청 수 +1이라 정확한 대응은 불가능하다. Σ(n_new·W)/ΣTTFT는 rate 0.10에서 0.91–0.99,
  0.53에서 0.65–0.71, 0.77에서 0.28–0.34다 `derived`(±5% 근사; floor 셀은 1.04로 1을 넘어 근사 오차를 드러낸다).
  8k span W: decode가 없을 때(비분할 108) 870–891 ms, decode와 동거할 때(D44 → prefill 64) 1,463–1,816 ms(중앙 ≈ 1.49 s) `measured`.
  **TTFT 안의 W 증가는 분할(prefill SM 108 → 64)과 동거에서 오고, lockstep 반복 수와는 무관하다**(host가 chunk를 140 ms 안에 다 발행).

### 3-4. (d) C2의 ITL은 span 겹침을 포함했는가, 통제가 정지를 배제했는가

- **문서**: `s8_scaleup/FINDINGS_8B_2026-07-28.md` §5와 감사 부록은 "채택 구간 prefill 동거 98.8–100%"라고 적었다 `cited`.
  즉 **C2 ITL은 16-SM prefill과 동거하는 상태에서 잰 것이다.** 채택 규칙은 양끝 runtime_snapshot의 (실현 SM, decode batch) 일치와
  GUARD 3 s(스냅샷까지 거리)다. **간격 길이나 정지에 대한 필터는 없다**(`s8_batch_matched.py:72-75`).
- **데이터**(`c2_tail_check.py`, 같은 채택 규칙을 재현하되 모든 간격 보존) `measured`, headline Job 865533:

| arm | 채택률 | 채택 p50 | 채택 p99 | 채택 > 3×p50 | 기각 p99 | 기각 max |
|---|---|---|---|---|---|---|
| Ha8 d16 / d92 | 0.90 / 0.91 | 33.6 / 33.9 ms | 92.2 / 40.7 | 0.75% / 0.16% | 760 / 773 ms | 1.81 s / 0.91 s |
| M8 d16 / d92 | 0.87 / 0.93 | 20.2 / 20.0 | 62.5 / 26.1 | 1.22% / 0.16% | 669 / 826 | 1.71 / 0.87 |
| Hs8 d16 / d92 | 0.81 / 0.92 | 15.1 / 15.1 | 46.0 / 22.2 | 1.21% / 0.06% | 50 / 25 | 1.32 / 0.04 |
| T8 d16 / d92 | 0.76 / 0.87 | 11.9 / 12.0 | 34.8 / 19.4 | 0.64% / 0.03% | 37 / 21 | 0.43 / 0.41 |

(셀 전체 분포이며 batch는 매칭하지 않았다. headline의 batch 매칭 값과 같은 수치가 아니다.)

- 해석(`derived`): (i) C2 run에는 **0.7–1.8 s 정지가 실재**하고, Ha8·M8(Zamba2·Codestral)에서 뚜렷하다. Hs8·T8은 드물다.
  (ii) 채택 규칙이 정지를 **설계상** 배제하지는 않았다. 다만 정지가 분할 복귀나 batch 변화와 겹쳐 대부분 기각됐다(기각 쪽 > 3×p50 3–13%, 채택 쪽 0.2–1.2%).
  (iii) headline은 **p50**이고 채택 집합의 정지 비중이 ≤ 1.2%라 p50에는 영향이 없다. 또 prefill이 16 SM로 고정된 채 p50이 D에 따라
  2.4–2.9배 움직이므로, C2 p50은 span 시간이 아니라 decode step(동거 포함)을 잰 양이다. **C2 채택 집합의 꼬리 분위는 정지에 오염돼 있어 인용하면 안 된다.**
  (iv) Job 865493의 Ha8 d24–d92 채택 집합은 > 3×p50 비중이 4–12%다(이 다리들은 감사상 headline에 쓰이지 않았다, `AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md`).
- M4(같은 KISTI·Ha8): 정지 크기가 D24 167.7 → D92 865.6 ms로 prefill SM(= 108 − D)이 줄수록 커진다 `cited`. 이것은 lockstep-GPU 형태
  (정지가 prefill GPU 시간을 따라감)와 **부합하지만**, PHASE_EVENTS가 없어 decode가 무엇을 기다렸는지는 확인되지 않았다.

### 3-5. N2 Job-PROFILE(보조, 서버 측만)

288 cell-window. 단독 프로파일 설계라 동거는 부수적이다. 서버 측 간격 > 3×SOLO는 692건이고 최대 418 ms다. 이 SOLO 기준에는
sticky OFF의 108 SM 단독이 섞여 있어 "정지"로 읽지 않는다. 본문 판정에는 쓰지 않았다 `derived`.

## 4. 판정 (기술값만)

| 범위 | 판정 | 근거 |
|---|---|---|
| VESSL A100-SXM4 · Nano-9B-v2-Base · flashinfer · cudagraph ON · D16/24/44 · 프롬프트 8k(N3·N2-EA), 8k·32k(T3) · split 예산 65536 | **lockstep 정지가 ITL 꼬리를 지배하지 않는다** | N3 Job 4: client > 40 ms 4.2e−5, p99 꼬리 100%가 동거 decode step. N2-EA: 동거 간격이 lockstep 예측의 1/3–1/9, span당 토큰이 chunk 수의 2–9배. T3: > 5× 0건. host가 GPU보다 앞서 달리는 것이 직접 보임(WAIT ≈ span의 90%) |
| 같은 설정의 N3 Job 3 | 드문 lockstep형 사건(첫 chunk host 차단 0.66–1.34 s)이 있으나 **꼬리를 지배하지 않는다**(≤ 8.3e−4 토큰) | 같은 설정 Job 4에서는 0건, 원인 미해결 |
| KISTI A100 · Ha8 Zamba2-7B / M8 Codestral(triton) · prefill 16 SM | 정지가 존재한다. **그것이 ITL 꼬리를 지배하는지·기전이 무엇인지는 판단 불가**(phase 데이터 없음) | 채택 집합 p99가 Ha8 d16에서 p50의 2.7배, 기각 집합 p99 0.67–0.83 s |
| HE0(Zamba2-2.7B, KISTI, rate-swing) | **판단 불가(데이터 없음)** | phase 계측 run 0, 해당 모델 VESSL run 0 |
| C2 headline p50(2.36–2.91×) | 이 진단 범위 안에서는 **lockstep 정지가 p50을 만든 흔적이 없다** | 채택 정지 비중 ≤ 1.2%, p50이 prefill 고정 아래 D에 따라 움직임. 꼬리 분위는 오염됨 |

- n: N3 = Job 2개(rate 10단계 × seed 4, 3회차 0.53–0.77은 seed 3), N2-EA = 셀당 round 4, T3 = Job 3개 × rep 2.
  **모두 단일 모델(Nano-9B) · 단일 기판 계열(VESSL A100 SXM 400 W)이다.** 다른 모델이나 KISTI로 일반화하지 않는다.
- 이 판정은 기전 판정(lockstep-GPU 대 CPU-launch)에 대한 **기술 관측**이다. SPEC S-6/K8에 따르면 P4 판별 예측은 동등성 여유가 등록되지 않아
  판별 예측으로 인용할 수 없다. 여기서 쓴 것은 등록된 P4가 아니라 사후 진단이다.
- **정본 결론(HE0, C2, ITL 기반 goodput)의 등급이나 문안은 이 문서로 바뀌지 않는다.** 다만 COMPARE 메모(`handoff_review_2026-10-15/COMPARE_span_sync_muxwise_bullet_2026-10-15.md`)
  §2.2의 "lockstep에서 ITL ≈ span 시간(decode step의 10–20배)" 유도는 VESSL Nano 데이터와 맞지 않는다. 이 불일치는 claims-auditor에 회부할 후보다.

## 5. 데이터 공백과 다음 GPU Job의 최소 계측

공백:
1. **KISTI 정지 모델(Zamba2·Codestral, triton)에 phase 이벤트가 하나도 없다.** HE0 모델(Zamba2-2.7B)은 VESSL run이 0이다.
2. chunk별 GPU 시간이 없다. W/n_chunks는 상한 대리값이다(span_end는 관측 지연을 포함).
3. drain(`adjust` 경로의 `prefill_stream.synchronize()`) 소요 시간이 기록되지 않는다.
4. 요청 ID가 없어 N3류 서빙 run에서 TTFT를 정확히 분해할 수 없다.
5. engine_gate decomp TEL·I46 원자료와 costterms T3 원자료가 m12 볼륨에만 있다(회수하면 I-4 GPU decode 시간과 대조할 수 있다).

최소 계측(다음 GPU Job, 1 Job, 측정 = Job 규약):
- **필드**: (i) `prefill_chunk` 이벤트 = `span_seq, split_index, forward_count, t_issue_start, t_issue_end`(host) + prefill stream CUDA event 쌍의
  GPU ms(query 비차단 회수, decode_probe와 같은 방식). (ii) `drain` 이벤트 = 사유(`adjust`·SLO 전환), 두 synchronize 각각의 host ms.
  (iii) `prefill_span_start`에 `rids`(요청 ID 목록). (iv) I-4 `decode_fwd_gpu_ms`를 같은 boot에서 함께 켠다. (v) client 토큰 절대 CLOCK_MONOTONIC.
- **arm**: 모델 {Nano-9B-v2(flashinfer), Zamba2-2.7B 또는 7B(triton)} × D {24, 44, 92}(prefill 84/64/16) × 프롬프트 {2k, 8k} × 예산 {8192, 65536}
  (SPEC R13-3 격자의 양 끝), 저부하 고정 rate(정지 관측용, goodput 판정 아님) + PHASE OFF 대칭 arm 1개(관측자 비용).
  **판별 기준**: 정지 반복에서 H(host)가 chunk GPU 시간을 따라가는가(lockstep-GPU), host 발행 시간을 따라가는가(CPU-launch),
  decode GPU 시간 자체가 늘어나는가(간섭). P4 동등성 여유는 PREREG에 먼저 등록한다.
- 원자료 회수: m12의 `engine_gate_decomp` `*_TEL/tel.jsonl`·`*_I46/decode_probe.jsonl`과 `/data/costterms/*` T3 telemetry를
  `fetch.sh`로 받으면 GPU 비용 없이 (iv) 대조의 일부를 할 수 있다(Nano 한정).

## 6. 재현

```bash
cd ~/Experiments/KISTI/prefill-layer-alloc
python3 workspace/engine-port/results/lockstep_diag/lockstep_diag.py --with-profile   # ~90 s, → lockstep_diag_values.json
python3 workspace/engine-port/results/lockstep_diag/c2_tail_check.py                  # ~8 s,  → c2_tail_check.json
```
분위수는 `pdmux_eval.analyze.percentile`, bootstrap은 10,000회 seed 1이다. 산출물 sha256: `c2_tail_check.json 8adba621…`(재실행 시 동일),
`lockstep_diag_values.json`은 재실행 때마다 새로 쓴다(내용은 결정적).

## 7. cost model 기여

- **CM-4(동거 간섭 `δ(p,d)`)** — 알게 된 것: VESSL A100 / Nano-9B-v2 / flashinfer / cudagraph ON / 예산 65536 범위에서는 동거 decode 간격을
  `I_co = T_dec(D, bs, ctx) · ρ`로 두고, **별도의 lockstep 정지 항 `L_stall(span, 108−D)`은 해상도 안에서 0이다**(N3 Job 4 > 40 ms 4.2e−5,
  T3 > 5× 0/28,354, 동거 간격이 W/n_chunks에 대해 기울기 0.0015 [0.0005, 0.0027]). 따라서 DISCUSSION_INDEX Q20("lockstep 정지를 CM-4와 분리된 항으로 둘지")에
  대한 첫 데이터는 **이 범위에서는 그 항이 비어 있다**는 것이다. ρ_on은 1.156–1.193(N2-EA client, 이 계산)이고 전력 교락이 있어 총효과로만 읽는다.
- **깨진 가정**: "lockstep 기판에서 ITL ≈ span 시간(decode step의 10–20배)"(COMPARE 메모 §2.2, 유도값)이 이 범위에서는 성립하지 않는다.
  host는 chunk를 decode 박자로 다 발행하고 GPU 완료를 기다리지 않는다(WAIT ≈ W의 0.90).
- **범위 밖**: KISTI Zamba2-7B·Codestral(triton)에는 0.17–1.8 s 정지 항이 있고, 크기는 prefill SM이 줄수록 커진다(M4, cited).
  함수 형태와 기전은 미측정이다. 같은 항이 모델·backend 의존인지 기판 의존인지는 분리되지 않았다(교락).
  부산물(CM-2·CM-6): 8k span W는 비분할(108)에서 870–891 ms, D44 동거에서 약 1.49 s다. co-load TTFT는 대기 77–89% / W 11–23% / 병합 ≤ 0.2%다(N2-EA).
