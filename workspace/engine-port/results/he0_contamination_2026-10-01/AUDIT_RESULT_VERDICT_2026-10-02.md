# HE0 정량 재집계 — 결과 감사 판정서 (claims-auditor, 2026-10-02)

> 읽기 전용 감사, GPU 0. 대상: `RESULT_2026-10-01.md`, `SYNTHESIS_2026-10-01.md`, `he0_realized.py`,
> `he0_realized_2026-10-01.json`, `he0_headline_unpaired_ci_2026-10-01.json`. 감사자에게 쓰기 도구가 없어 메인 세션이
> 감사 보고 본문을 그대로 옮겨 기록했다. 감사자 계산 스크립트(ladder/phase/nullval/latest/mut/rr.py)는 세션 scratchpad에만 있고 저장소 밖이다.
> C-1 잔차·C-4 cap 시간비·856964 귀속 계산은 result-analyst 별건(`FOLLOWUP_2026-10-02.md`)이며 이 판정서는 그 전제만 점검한다.

**전체 판정: `MIXED`.** 재구성은 대체로 건전하다. 결정적 오류 1건 — **"headline 효과크기 < 3% 게이트(−2.7%)"는 효과크기
해석으로서 `REFUTED`**(도착에 묶인 LO phase 희석; 처치가 작용하는 HI phase의 cliff 밖 효과 ≈ −7%). SYNTHESIS §2-1·§4의
"크기 과장" 서술은 정정 대상. 정본 술어 −19.9%의 metric cliff 증폭은 수치로 `CONFIRMED`.

## 1. 주장별 판정

| # | 주장 | 판정 | confound / 근거 |
|---|---|---|---|
| 1 | legacy −2.7% < 3% 게이트 ⇒ headline 자격 없음 | **산술 CONFIRMED / 해석 REFUTED** | (i) 상대 CI [−3.29%, −2.21%]가 −3%를 걸침(Welch [−3.46, −2.03]%). (ii) 희석: LO는 d44 vs bind+GATE −0.1% [−0.4, +0.2], LO 라운드 duration 전 arm 69.9 s 동일 = 도착 묶임. HI legacy **−6.8% [−8.0, −5.5]%**. (iii) ITL SLO 50/55 ms에서 COMBINED −5.6/−5.8%. 3% 아래는 legacy·COMBINED 규약 하나뿐. |
| 2 | 정본 술어 −19.9%는 ITL cliff 증폭 | **CONFIRMED (재스코어 민감도로만)** | ITL SLO 사다리(TTFT 3 s, COMBINED): 50 → −5.6%, 55 → −5.8%, 58 → −15.9%, **60 → −19.9%**, 62 → −10.1%, 65 → −2.8%, 70~∞ → −2.7~−2.8%. HI: −49.8%(60 ms) → −6.9%(65 ms·∞). 부호는 전 사다리·TTFT 1–6 s에서 음수. 게이트 8에 따라 크기 민감도 진단일 뿐, 다른 SLO의 정책 주장 아님. |
| 3 | 부호 견고 | **CONFIRMED (scoped)** | 전 사다리·HI·COMBINED 음수. n=4 vs 9 unpaired, arm×노드×날짜 교락 잔존. |
| 4 | duration 합산(max 아님) | **CONFIRMED** | `analyze.load_bench_serving_rounds` 합산, COMBINED = dL+dH. goodput 차 일부는 HI 배출 시간 차(d44 36.0 s vs bind+GATE 36.9 s) — 정의상 정당, 기술 필요. |
| 5 | unpaired CI | **PLAUSIBLE** | rep 짝 없음 ⇒ unpaired 정당. n=4 percentile bootstrap은 반보수적(canonical Welch [−0.961, −0.279] > bootstrap [−0.877, −0.344]). Welch 병기 필요. |
| 6 | TTFT 차이 95%+가 공유 대기열 | **산술 CONFIRMED / 기전 NOT-YET-SUPPORTED** | `queue = s_pt − a_i`는 입장 전 전 시간(선행 prefill 직렬 대기·슬롯 대기·배치 형성 혼합). in-flight ≈100 ms vs TTFT 차 219–945 ms라 ≥95%는 거의 산술. in-flight는 prefill SM에 비단조(d44 64SM 110 ms, d16 92SM 106, d24 98, d34 100) — 계측기가 자기 split 효과를 거의 해상 못 함(상한 ~1.4×여도 결론 유지). "cap 48 backlog 경로와 정합"은 정합이지 증거 아님 → C-4로 이관. |
| 7 | HI 동거율 범위 31–59% | **PLAUSIBLE(조건부) — "범위" 표현 강등** | 세 값은 규약 점이지 상·하한 아님. 로그만으로의 최만기 시작 규약에선 HI 동거 **0.1–0.8%**(852342/857051/859008). 하한은 running-count 정합 모형에 전적 의존. 첫 토큰 e_k는 클라이언트 수신 시각이라 detok·HTTP 지연이 전 규약을 위로 밈(합성 엔진 e_k +30 ms → 0.187 → 0.312). 재구성 무관(로그만) 값은 개수 가중 99.3–99.5%뿐. |
| 8 | arm 차이는 동거 구간에만 | **CONFIRMED (scoped, 간접)** | `adjust_stream_groups` 바이트 해시 f27fac7·81aab84·fdbf5d1·ed2a410 동일(`4cffa1552a7033a2`); static groups `[(108,0),(64-or-…,D),(0,108)]`; bs 맞춘 decode-only ITL 7 arm 동일(19.5–19.7 ms), C2 민감도(SM16→92 2.4–2.9×) 감안 시 검출력 있음. 단 확인은 "같은 SM"까지, "108"은 코드 함의. 출처 정정: fdbf5d1은 HE0 이후 문서 커밋, 실행 시점 코드는 f27fac7(07-15)·81aab84(07-16~) — 바이트 동일로 결과 불변. 원격 dev tree ↔ 저장소 src 동일성 미검증. |
| 9 | static 동거 100% 라벨 SM / dynamic 분포 | **항등식(측정 아님)** | `sm["static"] = {STATIC_D[arm]: length(CO)}` 구성상 100%. dynamic은 컨트롤러 목표 idx 배치 — 코드상 목표=실현이나 판독값 아님. "동거 구간 d44=44"는 코드 함의 + ITL 단조 서명으로만. |
| 10 | self-test PASS | **부분 항등식** | 변이 3종 조악(intersect→union 0.997, +1 s 0.868). 합성 엔진이 R을 재구성과 같은 규칙으로 생성(정합 59/59), 클라이언트 지연 0, prefill 수 ms ⇒ pt와 최조기 구별 불가(0.1878 vs 0.1878; 실데이터 HI 0.40 vs 0.56). **변이 "s_pt := 최조기"가 self-test를 SURVIVE** — HI 범위를 정하는 핵심 규약이 시험되지 않음. |
| 11 | 검증 지표(R 정합 98.1%, Decode 줄 적중 95.6–99.8%) | **R 정합 = 적합 통계(독립 검증 아님); Decode 줄은 초 단위로만 유효 CONFIRMED** | 귀무 검정(타임라인 이동): +0.1 s 0.99, +0.3 s 0.94–0.97, +1 s 0.62–0.74, +3 s 0.27–0.41, +10 s 0.12–0.27. 정렬은 ~0.3–1 s 수준만 실증, ~100 ms prefill 구간 위치는 미검증. |
| 12 | gate/no-gate 분류 | **PLAUSIBLE(강함)** | FEAS 줄 gate 9런 58–299, no-gate 4런 0, 경계 사례 없음. 소속 목록은 정본 유래(독립 도출 아님). |
| 13 | SYNTHESIS §2-4: 856964 = 클라이언트 정지 아티팩트 | **과장** | RESULT는 미확정으로 적었는데 SYNTHESIS가 아티팩트 쪽으로 기울임. 다수 요청 동시 ITL 갭은 서버측 정지로도 생김. |

## 2. 정정 문장 (인용 금지 문장 승계용)

- "HE0의 효과크기는 집계 규약 의존이다. COMBINED legacy −2.7%는 도착률에 묶인 LO phase(arm 간 −0.1%)의 희석 값이고,
  HI phase에서 cliff 밖 술어(legacy·ITL ≥65 ms·ITL 제약 없음)로 보면 −6.8~−6.9%(unpaired 95% CI 상대 약 [−8, −5.5]%)다.
  '3% 게이트 미만'이라고 쓰지 말 것."
- "정본 술어(ITL p95 60 ms) 효과 −19.9%(COMBINED)·−49.8%(HI)는 ITL p95 ~55–62 ms 절벽의 증폭이다. 같은 런을 65 ms로
  재채점하면 −2.8%·−6.9%다(재스코어 민감도, 정책 주장 아님)."
- "HI 시간 가중 동거율 31/39/57%는 세 규약 점이지 구간이 아니다. 로그만으로 정할 수 있는 하한은 약 0이다."

## 3. result-analyst 계산 3건의 전제 점검

- **C-1**: bind+GATE 지배 성분 24 SM, 9런 중 7런 gpu38인데 d24 static엔 gpu38 런 없음(gpu43 1, gpu37 3) — "같은 노드
  static 가중 혼합"이 주 성분에서 불성립. canonical HI goodput 런 간 CV 55%(legacy 1.4%) — 선형 혼합이 cliff 비선형에 깨지므로
  **HI phase + cliff 밖 술어**로 수행, LO는 구조적 null. r=0.96 자체는 견고(legacy 0.947, ITL65 0.868, LOO 최소 0.75–0.92).
- **C-4**: `Prefill batch` 줄은 입장 순간 샘플이라 포화 과소추정. `Decode batch` 줄로 Δt 시간가중. 구속 정의 =
  **running = cap ∧ queue 비어있지 않음**, 문턱 46/47/48 모두 보고.
- **856964**: 클라이언트 ITL>1 s 갭 동안 서버 `Decode batch` 줄이 정상 간격이면 클라이언트측, 끊기면 서버측 + 송신 지연 계단 동시성.

## 4. 확정에 필요한 것

1. (GPU 0) 동거 구간 독립 추정량: 느린 decode 토큰(ITL > k × decode-only 중앙값) 클러스터로 prefill in-flight 구간 직접 추정
   (running-count 정합과 독립), pt / +1 iter와 대조, k 격자 사전 고정.
2. (GPU 0) self-test 보강: (a) 클라이언트 수신 지연·prefill 길이 실데이터 규모(~100 ms) 합성 (b) 변이 "s_pt := 최조기" (c) 변이 "e + 30 ms".
3. (GPU 0) SYNTHESIS §2-1·§4와 정본 반영 초안을 §2 문장으로 정정(사용자 결정 + doc-steward).
4. (GPU, 사용자 결정) 동거율 폭을 실제로 좁히는 길은 per-iteration telemetry(`decode_sms`, `timestamp_monotonic_s`) 재실행뿐.
