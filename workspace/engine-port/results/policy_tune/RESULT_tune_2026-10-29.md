# TUNE 결과 (2026-10-29/30) — 기술값, 미감사

Job `pdmux-policy-tune-074ffc3-20261029140207` (job-9t469viisz2i), PREREG rev1.6 §4-3·§21. 구성 {anchor idx 2,3,4}×{dwell 3,6} × 튜닝 seed {101,102} = 12 boot·12 run, HE0 셀(LO 2.7356 / HI 10.9424 req/s, 구간당 200 req, 3 라운드), SLO TTFT 3000 ms / 요청 내부 ITL p95 60 ms. **수치는 측정 기술값이며 성능 판정이 아니다.** 인덱스↔D: 1=D16 … 5=D64 (idx2=D24, 3=D34, 4=D44).

## 1. 무결성·등록 준수
- `RAW_SHA256SUMS` 148줄 전부 OK, `RAW_EXPORT_OK`. run 12 전부 `done`, correctness 12 boot n_bad=0.
- run 구조 등록대로: run당 세그먼트 6(3 라운드×LO/HI), 요청 1200(200×6). trace 동일성: seed별로 6 구성이 같은 trace_hash(101, 102 모두 True).
- 시간: 측정 run 합 77.6 min(run당 약 380–400 s = 6.4–6.7 min). 등록 보수값 run≈11 min보다 짧은 이유: 등록 가정은 HI drain 67–100 s였는데 실측 HI 구간 완료율 4.2–4.4 req/s·drain 약 28 s, LO 73–80 s(§21-6의 HI drain 가정이 과대). 즉 run 수·길이는 등록(12 run, 3 라운드, 200 req)대로이고 짧아진 것은 구간 drain 때문이다.

## 2. 구성별 TRUE goodput (duration 합산, req/s) 및 switch_count
| 구성 | seed101 | seed102 | 평균 | LO | HI | switch_count(101/102) | 체류 |
|---|---|---|---|---|---|---|---|
| dyn_a4_w6 | 2.121 | 2.240 | **2.1804** | 2.424 | 1.744 | 0 / 0 | D44 100% |
| dyn_a4_w3 | 2.116 | 2.224 | 2.1698 | 2.422 | 1.719 | 0 / 0 | D44 100% |
| dyn_a3_w3 | 2.103 | 2.189 | 2.1458 | 2.422 | 1.658 | 4 / 0 | D34 79–100%, D44 0–21% |
| dyn_a3_w6 | 2.102 | 2.179 | 2.1408 | 2.423 | 1.643 | 4 / 0 | 〃 |
| dyn_a2_w3 | 2.036 | 2.075 | 2.0556 | 2.421 | 1.414 | 9 / 5 | D24 약 2/3, D34, D44 일부 |
| dyn_a2_w6 | 1.996 | 2.036 | 2.0158 | 2.417 | 1.320 | 6 / 5 | 〃 |

## 3. 선택(등록 규칙 §4-3)
- 최고 평균 goodput = dyn_a4_w6(2.1804). **동률 규칙 발동**: dyn_a4_w3(2.1698)가 1% 이내(0.49% 아래). 동률 집합의 평균 switch_count는 둘 다 0 → 이름 순으로 **선택 = `dyn_a4_w3` (anchor idx 4 = D44, dwell 3)**. 즉 최고 goodput 구성이 아니라 이름 순 규칙의 결과다.
- 관찰(해석 아님): 선택된 구성(anchor idx 4)은 TUNE의 4 run 전부에서 switch_count 0이고 체류가 D44 100%이다 — 이 trace에서 컨트롤러가 한 번도 움직이지 않았다(D44 고정과 같은 분할 상태). 따라서 EVAL의 동적 arm은 이 구성 기준으로는 사전지정 static D44와 같은 분할을 쓸 수 있고, "동적 대 static D44" 차이가 컨트롤러 효과인지 노이즈인지는 EVAL에서 분리되어야 한다. 구성 간 goodput 차(a4 > a3 > a2)는 HI 구간 차에서 오며 LO는 모든 구성이 2.42로 같다.

## 4. 게이트 5 (TTFT/ITL 분포, run 합산 구간별 중앙 서술)
- LO TTFT p50 83–89 ms, p95 310–442 ms, p99 476–678 ms (모든 구성; a4가 p95 412–442로 약간 높음, a2·a3 310–372).
- HI TTFT p50 4.8–5.9 s, p95 14–19 s, p99 15–20 s (seed 101 < seed 102 대체로 14 s vs 18–19 s).
- HI 토큰 ITL p50 25.7–26.6 ms, p95 49.1(a4) / 54–57(a3, a2) ms, p99 50–60 ms.
- CLIFF_ADJACENT: 구간별 TTFT ±5% 요청 비율 최대 2.7%(HI), LO 0 → 해당 없음. HI에서 TTFT p50이 SLO 3 s를 넘는다(과부하 구간).

## 5. 공변량 (기술값; 판정 무영향, §21-5)
- 기판: 단일 Job·노드 `job-9t469viisz2i-vjrkv`, GPU UUID GPU-90b3ce6c-…, 드라이버 580.178.04(Job 전체 동일). 이 GPU는 CAP1/1-R/CAP2와 다른 UUID(caveat 4: Job 간 차이를 표류로 해석 금지).
- run 창 throttle(SW power cap) 5.6–10.0%, 전력 중앙 310–319 W, 온도 중앙 52–58 ℃, SM clk 중앙 1410 MHz(전 run).
- 구간별 완료율: LO 2.48–2.68 req/s(drain 9.8–12.4 s), HI 4.17–4.41 req/s(drain 27.5–30.5 s). HI 구간 throttle 10–16% > LO 5–7%.
- 기준 셀(D44, 7.29491 req/s, 120 req, seed 201, 같은 Job): start attainment 0.983(TTFT p50 400 ms, p95 2819 ms, throttle 6.9%, 32.0 s) → end 0.908(p50 417 ms, p95 3122 ms, throttle 5.5%, 32.7 s). 기술 서술만: Job 시작→끝 사이 attainment가 0.983에서 0.908로 낮아졌고(120 req에서 118→109 양호) TTFT p95가 2.8→3.1 s로 SLO 경계(3 s)를 넘었다. 이 두 셀은 120-요청 한 번씩이라 같은 분산 범위인지는 이 자료로 알 수 없으며, 원인(기판·시각·누적)은 서술하지 않는다.

## 6. EVAL addendum 초안 (미봉인·미제출)
`addendum_eval_DRAFT.json`: tuned_config = dyn_a4_w3(anchor 4, dwell 3), HE0 rate LO 2.7356 / HI 10.9424, SLO 3000/60, 튜닝 seed 101·102 비중첩, 동률 규칙 발동·동적 arm 무이동 관찰을 병기.

## 7. 산출물
`tune_analysis.json`(run별 전수), `tune_goodput.png`(구성별 goodput·tie 밴드, 구간별 goodput, switch_count), `addendum_eval_DRAFT.json`, `analyze_tune.py`·`plot_tune.py`.

## cost model 기여
CM 칸 지정 없음. 기술값: 이 기판·HE0 셀에서 앵커가 낮을수록(D24→D44) HI 구간 goodput이 단조 증가(1.32–1.41 → 1.64–1.66 → 1.72–1.74 req/s)하고 LO는 모든 구성이 2.42로 같다 — 이 rate에서 LO는 분할 무관, 차이는 HI 과부하 구간에서 나온다. 가장 큰 D(44) 고정에서 컨트롤러 전환이 0회인 구성이 최고 goodput이며, 낮은 앵커 구성은 전환(5–9회)에도 더 낮은 HI goodput을 보였다.
