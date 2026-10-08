# RESULT — power cap 수동 재분석(Rung A, 0 GPU-h) — 2026-10-07 (claims-auditor 감사 반영판)

**등급: 기술(記述)·탐색적. 정본(CONSENSUS/PROJECT_STATUS) 인용 금지. 결정 입력 아님.**
감사: `AUDIT_RESULT_passive_2026-10-07.md`(claims-auditor, 2026-10-07, 치명 0·문장 수정 다수 — 전부 이 판에 반영).

- 목적: `reports/power_cap_viability_2026-10-07.md`(개입여지 평가)의 근거 자료. 기존 VESSL `meta/clk.csv`(100 ms)와
  KISTI 5 s `*_clocks.csv`만 재집계했다. **새 측정 0건, GPU 0 h, 처치(`-pl`) 0건.**
- 코드: `analysis_2026-10-07/pc_common.py`(공통), `pc_energy.py`, `pc_split_power.py`, `pc_itl_throttle.py`.
  세 스크립트 모두 `--selftest` PASS. **selftest의 한계(감사 2c)**: `pc_itl_throttle` selftest는 5 s 합성 에피소드·25 % 효과·전체 라벨 반전
  변이만 시험한다 — 실제 에피소드 길이(p50 0.9 s)에서의 부분 오정렬(0.3–1 s)이 ~3 % 소효과를 희석하는 민감도는 **미검증**.
  표 원본은 `analysis_2026-10-07/out/pc_*.md`.
- 회귀 대조: `pc_energy.py --regress` — N3 분석기(`N3_RESULT.json substrate.clk`) 43유닛·λ0 분석기(`lam0v_descriptive.json`) 15셀과
  0x4 %·SM MHz·p50 W·샘플 수 차이 0.0. **같은 파서 규약의 재구현이므로 항등 검사(교훈 9)** — 파서 드리프트만 잡는다.
  **에너지 적분·J/req·J/good·J/tok 열은 외부 대조가 없다.**
- window 정의(감사 1): N3·E2는 runner `TIMING.tsv`의 `bench_start/end_epoch`(bench_serving `duration_s`보다 **18–21 s 길다** —
  bench 프로세스 기동·종료 포함), λ0는 srv.log 마지막 시각 − `duration_s`(= bench 길이). 두 규약의 J/*는 서로 비교하지 않는다.
- 기판: VESSL A100-SXM4-80GB(power.limit 400 W, max SM 1410 MHz), KISTI A100-SXM4-80GB(400 W).

## 0. 한 줄

처치가 없으므로 어느 수치도 power cap의 효과가 아니다. 지금 데이터가 말해 주는 것: (1) 한 Job 안에서 J/req는 부하에 따라 2.7× 움직이고
J/good은 SLO cliff에 노출된다, (2) 기존 VESSL 데이터에서 cap 중 SM 클럭 하락은 −3…−4 %뿐(1250 MHz 아래로 간 적 없음) —
**power cap 처치를 알려 주는 데이터는 없다, `-pl` 기판이 필요하다**, (3) shape B에서 0x4는 거의 항상 prefill 동거((64,44))와 겹치며,
유일한 비동거 층(bs1, (0,108), n=1,035, 전부 0x4 전환 ≤0.5 s 이내)에서 SM 클럭 −2.8 % 동안 step 중앙값은 **+0.2 %**(3 % 문턱 아래, "변화 없음" 아님).

## 1. 에너지 지표 (`pc_energy.py`, 표 `out/pc_energy.md`)

정의: E = Σ P·Δt(window, 100 ms 샘플, 샘플 간격 0.5 s 초과분은 적분 제외); J/req = E/completed;
J/good = E/good(good = TTFT ≤ SLO ∧ 요청 내 token-ITL p95 ≤ SLO, `analyze.RequestResult.passes` 그대로);
J/out_tok = E/total_output_tokens. SLO: N3 = T_5(=5×floor, 4.52 s)/50 ms, E2·λ0 = 3000/50 ms.

### 1-1. N3 r3 (shape B 8192→64, D44 sticky-OFF, grid 4 seed 평균, TIMING window)

| rate(req/s) | mean W | 0x4 % | J/req | J/out_tok | good_frac |
|---|---|---|---|---|---|
| 0.10 | 127 | 9.1 | 1410 | 22.0 | 1.00 |
| 0.22 | 166 | 17.8 | 932 | 14.6 | 1.00 |
| 0.30 | 192 | 23.9 | 793 | 12.4 | 0.96 |
| 0.45 | 244 | 37.1 | 660 | 10.3 | 0.93 |
| 0.61 | 289 | 46.1 | 579 | 9.0 | 0.74 |
| 0.69 | 304 | 46.0 | 550 | 8.6 | 0.42 |
| 0.77 | 303 | 42.9 | 531 | 8.3 | 0.13 |

- 한 Job 안에서 J/req는 0.10→0.77 req/s에서 **2.7× 감소**(유휴 전력 ~66 W 상각). 0x4 노출은 9→46 % — 부하에 내생적이며,
  N3 grid는 높은 rate부터 내려오므로 **rate는 Job 내 실행 순서와 교락**(N3C-16): rate 순 추세를 cap 효과로 읽지 않는다.
- J/good는 good_frac이 무너지는 0.61 이상에서 발산(0.77에서 2.8k–20k J/good) — **metric cliff(게이트 6)에 그대로 노출되는 지표**.
  J/good을 쓰려면 용량 먼저 측정하고 cliff 밖에서만 비교해야 한다. "부하를 올리면 효율↑"는 운영 권고가 아니다(정성적 주의만).

### 1-2. E2 r2 (shape A 256→512, 같은 Job·같은 seed 쌍, sticky OFF vs ON)

| cell | seed | 창 평균 W OFF→ON | 0x4 % OFF→ON | 창 E/완료요청 OFF→ON | 비 |
|---|---|---|---|---|---|
| a_r2 | 251/2630/4162/4386 | 323–331 → 260–271 | 22.7–26.7 → 12.6–21.3 | 131–135 → 137–143 | 1.03–1.08 |
| a_r4 | 251/2630/4162/4386 | 328–335 → 265–268 | 23.9–25.8 → 16.0–17.5 | 124–127 → 138–140 | 1.08–1.13 |

- shape A, 같은 Job·seed 쌍: ON boot의 창 평균 전력은 OFF보다 55–70 W 낮고, 창 에너지/완료요청은 높다(**부호 8/8**; 크기 1.03–1.17은
  창 정의[TIMING vs bench 길이]에 의존). **귀속 없음** — E2 금지 문안 16건(`CONSENSUS §1` 행 35, `VERDICT_result_e2_vessl_r2_2026-10-06.md`) 적용.
  읽을 것은 하나뿐: **전력 ≠ 에너지** — 정규화 전 부호가 뒤집힌다(정규화 교훈).
- shape B 쌍(n=2)은 보고하지 않는다(R4″ n=2 "차이 없음" 금지; 실현 분할 e_time은 이 쌍에서 산출돼 있지 않다).

### 1-3. λ0 (D44, 11+4셀, bench-길이 window)

- a_r4(8.0 req/s 제공, 48-cap 구속 하 2.99 달성): 353 W, 0x4 27 %, 118 J/req, 0.23 J/out_tok. b_r3(1.1 제공, 0.69 달성): 345 W, 0x4 49 %, 500 J/req, 7.8 J/out_tok.
- shape B의 J/out_tok는 A의 34×(8192 입력 토큰 prefill 비용을 64 출력 토큰이 나눠 갖기 때문; J/total_tok로는 0.061 vs 0.153로 역전).
  **토큰당 에너지는 분모 정의에 따라 순위가 바뀐다** — 분모 없이 인용 불가. λ0·E2·N3의 J/* 수준은 **Job 간·규약 간 비교 금지**(E2VR-3/LVR 계열).

## 2. 전력/클럭 vs SM 분할 (`pc_split_power.py`, 표 `out/pc_split_power.md`) — 캠페인별 기술

KISTI(2026-07~08) `*_clocks.csv` 800파일, busy = util ≥ 90(작업량 정규화 아님), near-cap = P ≥ 380 W. **타임스탬프 없음, 실현 분할 미검증.**
s8p(CONC 8 + 직렬 prefill 프로브)와 s8 deconf(CONC 16, C1024)는 **다른 캠페인·다른 부하**다 — 두 축을 한 문장에서 비교하지 않는다.

### 2-1. s8p (decode 16 고정, prefill 16→92; busy 평균 W)

| arm | P16/D16 | P24/D16 | P44/D16 | P92/D16 | near-cap @P92 | min MHz @P92 |
|---|---|---|---|---|---|---|
| Ha8 Zamba2-7B | 182 | 215 | 281 | 354 | 0.65 | 1350 |
| Hs8 Nemotron-H-8B | 230 | 264 | 326 | 388 | 0.79 | 1305 |
| M8 Mamba-Codestral-7B | 194 | 230 | 279 | 367 | 0.72 | 1365 |
| T8 Qwen2.5-7B | 253 | 290 | 353 | 392 | 0.82 | 1275 |

이 캠페인 안에서 P16→P92는 +139…+173 W, P92에서만 near-cap 0.65–0.82. (샘플당 작업량이 SM 수에 따라 다르므로 "SM당 전력"이 아니다.)

### 2-2. s8 deconf (prefill 16 고정, decode 16→92, C1024; busy 평균 W)

| arm | P16/D16 | P16/D24 | P16/D44 | P16/D92 | np(108) | near-cap: 분할 / np |
|---|---|---|---|---|---|---|
| Ha8 | 213 | 223 | 235 | 230 | 332 | 0.00 / 0.44 |
| Hs8 | 256 | 264 | 288 | 284 | 357 | 0.00 / 0.42 |
| M8 | 216 | 225 | 238 | 249 | 353 | 0.00 / 0.47 |
| T8 | 272 | 286 | 327 | 323 | 353 | 0.00 / 0.44 |

- 이 캠페인 안에서 D16→D92는 +17…+51 W. **deconf 분할 arm(P16)은 near-cap 0.00·클럭 1410 고정**(deconf 행에 한함 — c2r P16 행은 min 1290–1350 MHz·max 418 W),
  np(비분할)만 near-cap 0.4–0.67·클럭 1260–1380. s8 coupled(P=108−D; yml `c_d44=[64,44]`, `c_d16=[92,16]`)에서 P64/D44·P92/D16은 near-cap 0.21–0.71.
- 2026-07-29 handoff의 "clock axis 실패(d16–d92 전부 1394–1410 MHz)"는 "deconf 분할 arm이 cap에 닿지 않았다"와 **정합한다(인과 아님)**.
- s0_deconfound(2026-07, 3B급 H/M/T, 명목 P16/D16…D92): near-cap 0.32–0.48·클럭<1400 비율 ~0.4가 P16/D16에서도 나타난다. Stage 0 판정
  (`PARTITION_RESIDENCY_STAGE0.md`)은 이 셀들이 (16,D)를 decode-active 시간의 60–97 % 유지했다고 적는다 ⇒ **분할을 대부분 유지한 셀에서도
  near-cap이 나타난다** — 전력은 실현 분할의 증인/대리지표로 **부적합할 수 있다**(이전 판의 "값싼 증인" 서술 철회).

### 2-3. VESSL λ0 — 실현 분할과 전력

- λ0 shape A(e_time D44 3–14 %): 330–356 W, 0x4 1–31 %(rate 따라). shape B(e_time 70–97 %): 271–347 W, 0x4 42–53 %. (E2와의 교차 비교는 하지 않는다 — #14 기판/Job 혼합 금지.)

## 3. cap 중 decode step 시간 (`pc_itl_throttle.py`, 표 `out/pc_itl_throttle.md`) — 탐색적

- 데이터: N3 r3 PHASE_EVENTS=1 유닛 37개(grid 36 + floor), `decode_iteration` 55,793회(edge guard 후).
- 정렬: srv.log "Decode batch" 줄(`decode_log_interval=40`, `server_args.py:418`; 초 단위 벽시계) ↔ telemetry `iter_seq=40k`의 `t_sync` 짝 →
  유닛별 O=epoch−mono. bracket 반폭 ≤ 52 ms(로그 지연 d만큼 상향 편향, ms 단위), Job 전체 드리프트 54 ms, marks 교차검사 (−0.61, +0.35) s ∈ (−1, 1).
  독립 교차 증거: telemetry의 `prefill_in_flight`와 clk의 0x4가 bs1에서 95 %/99 % 일치(두 계측기 간 정렬 성립).
- 0x4 에피소드: 1,660개, p50 0.9 s, p90 2.1 s, 55 %가 1 s 미만.
- **교락(감사 3, 변수 동시 변화)**: 이 arm에서 동거 플래그는 실현 분할과 완전히 겹친다 — 동거 = (64,44) 42,091회, 비동거 = (0,108) 33,522회.
  "동거 vs 비동거"는 **"(64,44)+prefill 대 (0,108) 전체-GPU decode"**이며 SM 수와 동거를 분리할 수 없다.

| 층(bs × 상태) | n(0x4=0 / 1) | step 중앙값 ms (0 / 1) | 비 | SM MHz (0 / 1) | 해석 가능? |
|---|---|---|---|---|---|
| bs1, (0,108) 비동거 | 20,212 / 1,035 | 12.10 / 12.12 | **1.0017** | 1410 / 1371 | 유일한 깨끗한 층 — 단 0x4=1 표본 전부가 전환 ≤0.5 s 이내(guard 0.5 s면 n=23, 1.0 s면 0) |
| bs1, (64,44)+prefill | 136 / 18,734 | 19.20 / 18.79 | 0.978 | 1410 / 1361 | 불가 — 0x4=0 셀(n=136)은 prefill 시작 직후 iteration |
| bs2–4, (64,44)+prefill | 123 / 14,296 | 18.78 / 19.16 | 1.021 | 1410 / 1350 | 불가(동일 이유) |
| 참조: (64,44)+prefill / (0,108) @0x4=0 | bs1 1.59×, bs2–4 1.51× | | | | SM 수·동거 교락 |

- bs1 (0,108) 층: step IQR 12.093–12.106 ms(≈0.1 %)로 측정은 정밀하고, 비 1.0017(iteration 단위 bootstrap CI ≈[1.0016, 1.0019], 유닛 군집 미보정)은
  **측정 가능한 +0.2 %**이지 "변화 없음"이 아니다. 3 % 문턱 아래. 적용 범위: bs1·(0,108)·prefill 종료 후 ~0.5 s·SM −2.8 %뿐 —
  운영 분할(64,44)의 decode, bs>1, 더 낮은 cap에 대해 아무것도 말하지 않는다.
- pooled 비 1.001(rate 0.10–0.69 전부 1.001–1.002); 0.77은 1.021이나 **rate는 Job 내 순서와 교락(N3C-16)** — 수치 부착 금지.
- 이전 판의 "클럭 채널은 검출되지 않는다"는 **철회**(감사 3 REFUTED as worded). 남는 서술: 위 표 1행의 범위 한정 수치뿐.

## 4. caveat

1. 0x4는 처치가 아니라 부하·prefill 활동의 함수다(내생). 어떤 Δ도 cap에 귀속하지 않는다(LVR-2/VC-2/N3C-16 유지).
2. NVML power.draw는 A100에서 ~100 ms마다 갱신되는 순간값 스냅샷(반영 지연 ~120–175 ms; 2026-10-08 센서 프로브 `sensor_bw/`). 이 문서의 에너지는 스냅샷 적분이며, 주기적 부하에서는 aliasing으로 편향될 수 있다(사각파 50 ms 주기에서 −27 %). 정상 부하 대조는 ±2 %. 이후 Job은 NVML 에너지 카운터(`meta/energy.csv`)로 대체.
3. 전력은 전체 GPU 값 — prefill/decode 파티션별 귀속 불가.
4. KISTI 파일은 타임스탬프·실현 분할 검증·작업량 정규화가 없다(busy 마스크 util ≥ 90, 파일당 17–20행). 캠페인 간 비교 금지.
5. 3절은 관측 연구: bs와 prefill_in_flight(=실현 분할)만 통제. 깨끗한 층은 전환 꼬리 표본뿐.
6. J/good은 SLO cliff에 노출되며(1-1), 토큰당 에너지는 분모 정의에 의존한다(1-3). J/* 수준의 Job·기판 간 비교 금지.
7. 기존 VESSL 데이터에서 cap 중 SM 클럭 하락은 −3…−4 %(최저 ~1250 MHz)뿐 — **power cap 처치의 효과를 알려 주는 데이터가 없다.**

## 5. 금지 문안

- "power cap이 arm Δ의 원인" · "sticky ON이 에너지를 절감/비효율" · E2 A/B 에너지를 한 문장에 비교(#7) · "SM 분할이 전력을 분할한다(power partition)".
- "cap은 decode에 무해" / "클럭 채널 없음" — bs1·(0,108)·−2.8 % 외 어떤 decode 상태에도 금지.
- "co-residency(동거) 간섭 1.5–1.6×" — SM 수와 동거가 교락(위 §3).
- "전력은 실현 분할의 증인/대리지표".
- "prefill 축이 decode 축보다 전력에 민감하다"(캠페인 간 비교) · "cap은 prefill이 돌 때 걸린다"를 일반 명제로(N3 shape B 한정 관측).
- J/* 수준의 Job·기판 간 비교(λ0 118 vs E2 124 등) · "부하↑⇒효율↑ 운영 권고" · rate 순 추세(0.77 포함)를 cap 효과로.
- 2-2의 "분할 arm은 cap 미도달"을 정책 근거로. 어떤 수치도 정본에 올리지 않는다.

## 6. 재현

```bash
cd ~/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/power_cap/analysis_2026-10-07
python3 pc_energy.py --selftest && python3 pc_split_power.py --selftest && python3 pc_itl_throttle.py --selftest
python3 pc_energy.py --all --regress && python3 pc_split_power.py --all && python3 pc_itl_throttle.py --all
```
(입력: `results/{n3_capacity,e2_sticky_vessl,lambda0_vessl}/vessl/<job>/meta/clk.csv`, 각 캠페인 out 디렉터리, KISTI `results/{s0_deconfound,s8_scaleup,s8p_prefill,s8_frontier}/*_clocks.csv` — git 밖 원시 자료는 이양 번들 `C_results_untracked/` 참조.)
