# 결과 감사 r1 — SLO attainment 문헌 조사 · `sq_maxitl.py` 사후 정지 지표 (claims-auditor, 2026-11-08, 읽기 전용)

> 판정문은 claims-auditor 최종 보고를 main 세션이 그대로 저장한 것이다.

## 0. 회차·범위·절차 공시

- **회차: 두 대상 모두 r1.** 전용 감사 파일 없음(`tracks/main.md:11` "감사 필요(미반영)" ⑤·⑥, `reports/CONSENSUS.md:8822-8823`이 미감사로 적음). `RESULT_stock_quick_2026-11-02.md`의 결과 감사 r1은 **결과 문서**를 감사했고, 1 s 문턱은 그 감사자가 원자료를 본 뒤 꺼낸 값이다. 스크립트·지표 자체의 감사는 처음이므로 r1이다. 게이트 #110에 따라 직전 감사가 꺼낸 "1 s"를 등록 상수로 보지 않고 독립 검증했다.
- **1차 출처 대조**: alphaXiv 도구는 이 감사 세션에 없었다. 대신 4편을 직접 열어 표본 대조했다.
  - 로컬 PDF: `~/Experiments/KISTI/Papers/Bullet.pdf`, `Muxwise.pdf`
  - arXiv PDF: Sarathi-Serve 2403.02310, DistServe 2401.09670. scratchpad에 격리하고 `pdftotext`로 읽었다. Bullet Fig.15d와 Sarathi Fig.10은 페이지 이미지로 그려 눈으로 판독했다.
  - **미대조**: SLOs-Serve, PolyServe, Niyama, LoongServe, Mooncake, QLM, NanoFlow, §3 목록. 이들 행은 "작성자 주장"으로 남는다.
- **보고 Doc 원문은 읽지 못했다**(artifact 도구 없음). 판정 대상 문장은 호출자가 인용한 "W3에서 최대 정지 1초 조건이면 순서 역전"이다.
- **재계산**: 일회용 스크립트 2개를 scratchpad에 두고 `sq_analyze`를 import해 원자료에서 계산했다. 저장소 변경은 0이다(`sq_maxitl.json` mtime 2026-11-02 그대로, `git status`에 새 변경 없음).

---

## 문서 1: `reports/SURVEY_slo_attainment_published_2026-11-02.md`

### 1-A. 주장별 판정

| # | 주장(위치) | 판정 | 근거 |
|---|---|---|---|
| S1 | §1 W3 출력 "자연 출력, 평균 203, 상한 1024" | **REFUTED** | CAP2 원자료 전수(2,880요청)에서 실현 출력은 평균 **1,007.3**, 중앙 **1,024**, 상한 미만은 52건뿐이다. stock_quick SG W3도 셀당 출력 합이 49,152 = 48 × 1024다. "203"은 `cap_char/COMPARE_shapeC_workloads_2026-10-20.md`의 옛 측정값이고, `PREREG_HD_HE0cell_rev0` Q-1에도 같은 값이 남아 있다. 이 기판·모델에서 W3는 사실상 **1024토큰 고정 출력**이다. |
| S2 | §1 W1·W3 attainment 곡선, λ\* 7.29491·0.3375, seed별 40/48·47/48, D24 0.506 = 0.885 | **CONFIRMED** | 원자료 재계산과 일치한다. D44 W1: 240·240·233·168·…·182(11.09)·130(19.4)/240. D16 W3: 95·95·87·52·8/96. |
| S3 | §0.3 비 계산(7.29/18.3 ≈ 0.4, 0.3375/0.72 ≈ 0.47, /0.62 ≈ 0.54) | **CONFIRMED**(산술) | 단 "Bullet 그림의 vLLM 약 10"은 축 왼쪽 끝 첫 점(x = 10)이 이미 약 0.9 선에 걸쳐 있다. 정확한 표기는 "≤ 약 10(축 끝, 교차점 미관측)"이고, "약 0.7배"는 하한 쪽 비다. |
| S4 | §0.3 "ITL 술어는 우리가 조이고(60 ms 대 150–175 ms)" | **OVERBROAD** | 같은 문서 §6의 "어느 쪽이 항상 더 엄격하지 않다"와 충돌하는데, §0.3에는 그 단서가 없다. 반례: stock_quick W3에서 SG는 요청 최대 정지가 **1.8–9.5 s**인데도 등록 술어(요청 내부 p95 ≤ 60 ms)를 통과한다. 출력이 1024토큰이면 p95는 간격 51개를 크기와 무관하게 흡수한다. 정지 축에서는 우리 술어가 Sarathi의 P99 TBT(토큰 합산)보다 **느슨할 수 있다**. |
| S5 | §0.1 "attainment 문턱 0.90은 문헌의 주류 형식", "99%는 Niyama, MuxWise(토큰 백분위), DistServe 부록" | **OVERBROAD** | 표의 14편 중 90% attainment로 용량을 읽는 것은 5편(DistServe, Bullet, SLOs-Serve, PolyServe, LoongServe)이다. 나머지는 P99/P90 지연형(Sarathi, MuxWise, Mooncake), 평균형(NanoFlow), 문턱 없음(QLM)이다. MuxWise의 "P99 TBT ≤ SLO"는 **지연 백분위 SLO**이지 99% 요청 attainment가 아니다. 범주가 섞였다. 원문: "we use tail latency (e.g., P99) to assess SLO attainment". |
| S6 | §0.2 "가장 가까운 비교 대상 Bullet·Sarathi", §5 "같은 16k 필터" | **PLAUSIBLE(조건부)** | Sarathi의 필터는 **총 길이**(prompt + output) ≤ 16384이고(원문 "removing requests with total length more than 8192 and 16384"), 우리 필터는 입력 ≤ 16,384다. 출력도 Sarathi 중앙 208 대 우리 1024(S1)로 약 5배 다르다. 그래서 "가장 가깝다"는 입력 중앙(7059 대 6484) 한 축에서만 맞는 말이다. |
| S7 | §5 "W3의 낮은 λ\*는 2.8 s TTFT 때문일 가능성(가설)" | **NOT-YET-SUPPORTED** | 가설 라벨은 정직하다. 그러나 더 1차적인 대안이 빠졌다. 요청당 decode 일이 Sarathi보다 약 5배(1024 대 208) 많고, 이것만으로도 req/s 용량은 크게 낮아진다. |
| S8 | §2 Bullet 행: Table 4(norm. TTFT 3.0/1.5 ms, TPOT 150/175 ms), Fig.15d 판독(ShareGPT 90%: 18.3·17.2·15.1·12.2, arXiv: 0.72·0.64·0.52, 곡선값), "SGLang-1024 대비 1.49×", "Nanoflow 대비 5.2%" | **CONFIRMED**(표본 대조) | PDF와 그림 이미지로 확인했다. 단 Fig.15d 캡션 "P90 SLO attainment"가 "attainment 곡선 + 90% 기준선"인지 "P90 지연 기준 attainment"인지는 원문에서 확정되지 않는다. "평균 TPOT형, 요청 단위 결합"으로 읽은 것은 §4.1 "satisfy both the constraints listed in Table 4"와 정합하지만 1차 문언으로 확정되지는 않는다. 정규화 분자를 미확정으로 둔 것은 정직하다. |
| S9 | §2 MuxWise 행: TBT 50 ms(8B)·100 ms(70B), ShareGPT 1.9·1.73·9.5·1.46×, LooGLE 1.71·2·1.33·2×, A100 8B 1.2× | **CONFIRMED**(본문) | 그림 판독값(19 req/s, 0.24)은 대조하지 않았다. "P99 TBT가 토큰 합산"은 원문이 명시하지 않는다(원문은 "TBT accounts the latency of each individual token"까지) → **PLAUSIBLE**. |
| S10 | §2 Sarathi 행: Table 2 길이, Table 3 SLO, Fig.10 값(sharegpt4 strict 2.0·0.77·0.72, relaxed 2.3·1.55·1.05, arxiv strict 0.62·0.44·0.34, relaxed 0.77·0.47·0.39, Yi 1.07·0.27), 배수 라벨 = Orca 대비, 중앙 스케줄링 지연 ≤ 2 s | **CONFIRMED** | 원문·그림 모두 일치한다. 토큰 합산 P99 TBT도 원문에 있다: "since every decode token results in a TBT latency value". |
| S11 | §2 DistServe 행: Table 1 SLO(0.25/0.1, 2.5/0.15, 4.0/0.2, LongBench 15/0.15), ShareGPT 입력 평균 755.5, 90% 문언, 99% 부록 | **CONFIRMED**(본문·표) | Fig.8·9 판독값은 대조하지 않았다. |
| S12 | §2 SLOs-Serve, PolyServe, Niyama, LoongServe, Mooncake, QLM, NanoFlow 행, §3 | **NOT-YET-SUPPORTED**(감사 미대조) | 작성자는 1차 확인했다고 적었으나 감사는 확인하지 않았다. 반증은 발견하지 못했다. |
| S13 | §6 TPOT 평균 대 요청 내부 p95: "어느 쪽이 항상 더 엄격하지 않다" | **CONFIRMED**(원리) | 가장 강한 근거는 S4의 stock_quick W3 사례다. 이 사례를 메모에 추가하면 §0.3과의 충돌도 풀린다. 인용된 Phase A M3 J1 사례에는 원 문서의 "순위 서술 금지" 라벨이 승계돼 있다. |
| S14 | §0.4·§0.5(지속 블록 0, 0.225–0.506 민감도) | **PLAUSIBLE** | 감사 r1을 거친 문서의 승계값이다. 이번에 재계산하지 않았다. |
| S15 | §0.6, §5 "static arm은 SLO를 입력으로 받지 않아 재채점이 게이트 8에 걸리지 않는다" | **PLAUSIBLE(조건부)** | 컨트롤러가 없는 static arm이라 게이트 8의 취지(컨트롤러 행동)는 비켜 간다. 그러나 Bullet 채점에는 정규화 분자와 TPOT 정의의 원문 확정이 필요하다. 확정 전 재채점은 "Bullet SLO 근사"까지만 말할 수 있다. |

### 1-B. confound

- **11 (기판 혼합)의 문헌판**: 기판, 모델(hybrid 대 attention), 런 길이, 출력 길이가 모두 다르다. 메모는 대체로 공시했지만 S1(출력 5배)을 놓쳤다.
- **5 (metric cliff)**: W3 0.3375의 통과 여유가 1요청이라는 점은 메모가 공시했다.
- **6 (re-score) 변형**: §5에 Bullet SLO로 재채점하자는 제안이 있다. static arm이라 컨트롤러 문제는 없으나, SLO 정의가 미확정이면 비교량 자체가 정해지지 않는다(S15).
- **범주 혼동**(카탈로그 밖, 이 문서 고유): P99 지연 SLO와 99% attainment를 같은 축에 놓았다(S5).

### 1-C. 허용 문안

1. "이 메모가 1차 출처로 읽은 ShareGPT·arXiv 요약류 평가 논문 중 attainment 곡선으로 용량을 읽는 5편(DistServe, Bullet, SLOs-Serve, PolyServe, LoongServe)은 모두 90% 문턱을 쓴다. Sarathi-Serve·MuxWise·Mooncake는 attainment 문턱 대신 P99/P90 지연 SLO를 쓴다."
2. "A100 1장에서 가장 가까운 문헌 좌표는 Bullet(ShareGPT·arXiv-Sum, Llama3.1-8B)과 Sarathi-Serve(arxiv_summarization, Mistral-7B)다. 다만 술어, 출력 길이(우리 W3 실현 출력 약 1024토큰 대 Sarathi 중앙 208), 모델 구조가 달라 크기 자릿수 이상의 비교는 하지 않는다."
3. "요청 평균 TPOT와 요청 내부 ITL p95는 서로 다른 양이며, 어느 쪽도 항상 더 엄격하지 않다. 예: 출력 1024토큰이면 p95는 수 초짜리 정지를 최대 51개까지 흡수한다(stock_quick W3, 기술값)."

### 1-D. 인용 금지 문안

1. "우리 W3 출력은 자연 출력 평균 203토큰이다."(실현 평균 1,007, 중앙 1,024)
2. "우리 ITL 술어는 문헌(150–175 ms TPOT)보다 엄격하다." / "60 ms는 150 ms보다 2.5배 엄격하다."(방향이 정지 분포에 따라 바뀜)
3. "90% attainment는 문헌의 주류 형식이다." / "MuxWise는 99% attainment를 쓴다."(P99 지연 SLO와의 범주 혼동)
4. "우리 용량은 문헌의 절반 수준이다." / "Bullet 대비 0.4배."(메모 스스로 금지한 순위 해석. 숫자는 문헌 좌표 참고값까지만)
5. "W3의 낮은 λ\*는 TTFT SLO 2.8 s 때문이다."(가설. 출력 5배라는 대안 미배제)

### 1-E. 해소 방법

- **S1**: §1 W3 출력 칸을 "자연 출력(`ignore_eos` 없음), 상한 1024, 실현 평균 1,007·중앙 1,024(CAP2 전수 2,880요청)"로 고친다. §5 Sarathi 비교 행과 가설 단락에 출력 5배 차이를 넣는다. `PREREG_HD_HE0cell_rev0` Q-1의 같은 오류는 promote 때 별도 정오로 처리한다(이 메모 범위 밖).
- **S4·S5**: §0.3·§0.1을 허용 문안 1·3으로 바꾼다. MuxWise·Sarathi는 "P99 지연 SLO"로 따로 분류한다.
- **S12**: 미대조 7편을 alphaXiv나 원문 PDF로 표본 대조한 뒤 "감사 대조" 표기를 붙인다. 그 전까지는 "작성자 주장"이다.
- **S15**: Bullet 재채점은 정규화 분자(입력 토큰 수인지 등)를 원문이나 코드 저장소에서 확정한 뒤에 한다.

### 1-F. 종합: **GO-with-edits**

4편 표본 대조에서 옮겨 적기 오류는 0이다(그림 판독 오차 안). 다만 우리 쪽 기준값 하나(S1)가 틀렸고, 결론 두 줄(§0.1, §0.3)이 과잉이다. 메모가 스스로 순위 주장을 금지했으므로, 위 수정 후에는 참고 문서로 둘 수 있다. 정본(CONSENSUS §1/§2/§6)과 모순하는 서술은 없다. 인용 금지는 5개로 상한 안이다.

---

## 문서 2: `workspace/engine-port/results/stock_quick/sq_maxitl.py`와 "W3에서 최대 정지 1초 조건이면 순서 역전"

### 2-A. 코드 검증

| 항목 | 판정 | 근거 |
|---|---|---|
| 정의 | **CONFIRMED** | `good = req_good(TTFT ≤ S ∧ 요청 내부 ITL p95 ≤ 60) ∧ max(token_itl_ms) ≤ 1000`. 등록 술어에 정지 조건 하나를 AND로 붙였다. |
| 단위 | **CONFIRMED** | `token_itl_ms` = 클라이언트 토큰 도착 시각 차 × 1000(`sb_loadgen.py:117`, `ph_loadgen` 계열과 같은 형식), `LIM = 1000.0` ms. |
| 첫 토큰 처리 | **CONFIRMED** | ITL 목록이 `zip(tt, tt[1:])`라 TTFT 구간을 포함하지 않는다. 첫 토큰 뒤 간격만 본다. |
| 요청 단위 집계 | **CONFIRMED** | 요청마다 최대값을 구하고, 셀 attainment는 두 seed를 합친 요청 비율이다(seed별 값은 `by_seed`). 오류 요청은 `req_good`에서 탈락한다. |
| 경계 사례 | 공시 필요(결함 아님) | 빈 ITL(출력 1토큰)은 `max([0.0])`로 자동 통과하며 `req_good`의 빈 ITL 규약과 같다. 다중 토큰 청크는 간격 0으로 기록되지만 최대값에는 영향이 없다. |
| 출력 길이 노출 | 공시 필요 | 최대값은 출력 길이에 따라 커지는 극값 통계다. W3에서는 모든 arm이 1024토큰에 붙어 이번에는 교락이 아니다. 출력 길이가 다른 W1에서는 짧은 출력이 유리하다. |
| 출처 기록 | 사소 | `sq_maxitl.json`에 `LIM`이 기록되지 않는다. 문턱을 바꿔 다시 돌리면 같은 파일명에 덮어써져 구분할 수 없다. |

**코드 판정: CONFIRMED.** 재계산이 결과 문서 수치를 재현했다(W3 1 s, 두 seed 합: SG 68·54·43·13·7·5/96, best D 95·95·95·87·85·59/96).

### 2-B. "W3에서 최대 정지 1 s 조건이면 순서 역전"

**판정: 수치는 CONFIRMED, 서술로서는 OVERBROAD.** L = 1 s에서는 SG < best D가 6개 rate 모두, seed별로도 성립한다(예: 0.1에서 SG 32·36 대 D24 47·48). 그러나 역전은 문턱의 함수다. 반전 시험(같은 CAP2·stock_quick raw, sq_maxitl과 같은 추정량, L만 변경):

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치(SG / best static, 96요청, 2 seed 합) |
|---|---|---|---|
| 정지 문턱 L | 0.5 s | 역전 강화 | 0.1: 44/95, 0.3375: 7/87, 0.759: 5/59 |
| 〃 | 1 s(사용값) | 전 rate 역전 | 68/95 · 54/95 · 43/95 · 13/87 · 7/85 · 5/59 |
| 〃 | 2 s | 부호는 유지. 저부하는 2–6요청 차로 잡음 수준(seed 202·0.1, seed 201·0.15는 동률) | 93/95 · 93/95 · 89/95 · 65/87 · 45/85 · 19/59 |
| 〃 | 3 s | 0.225 동률(95/95), 0.506은 4요청 차 | 93/95 · 93/95 · 95/95 · 72/87 · 81/85 · 33/59 |
| 〃 | 5 s | **반전**: 0.1–0.225·0.506에서 SG ≥ best D로 복귀. 남는 것은 0.3375(84/87, 잡음)와 0.759(35/59, D 과부하 셀)뿐 | 96/95 · 96/95 · 96/95 · 84/87 · 88/85 · 35/59 |
| 〃 | 10 s | 등록 술어 순서로 완전 복귀 | 96/95 · … · 96/85 · 77/59 |

구조는 이렇다. static arm의 최대 간격은 전 셀에서 ≤ 131 ms라 L ≥ 0.15 s에서 D 쪽 값은 L과 무관한 상수다. 그러므로 "역전"은 SG 한 arm의 최대 정지 분포가 L 위에 몇 개 있는가라는 단일 arm 지시함수다. L = 1 s는 SG 저부하 최대 정지 중앙값(545–1,187 ms)에 거의 정확히 놓인다. 문턱이 한 arm 분포의 중앙에 있는 **절벽**이다. SG 0.1 rate에서 L을 0.5 → 1 → 1.5 → 2 s로 바꾸면 44 → 68 → 82 → 93이 된다.

**confound**
- **5 (metric cliff), 문턱 축 변형**: L이 SG 최대 정지 분포의 중앙에 있다. 2배(1 → 2 s)만 바꿔도 저부하 차가 27요청에서 2요청으로 줄고, 5배에서는 부호가 돌아온다.
- **6 (re-score)/사후 기준 선택**: 1 s는 SG 최대 ITL 중앙값(644 ms → 5,160 ms)을 본 **뒤** 감사자가 제안한 값이다. 게이트 8의 문자(컨트롤러 재스코어)는 static과 SG에 해당하지 않지만(둘 다 SLO를 입력으로 받지 않음), 결과를 본 뒤 판정 술어를 고른 오염(forking path)은 남는다. 등록 술어에서는 "SG ≥ 모든 static"이었고, 그 순서를 뒤집은 것은 사후 술어다.
- **3 (small-n)**: n = 2 seed, CI 없음. 1 s에서도 저부하 셀의 seed 간 편차가 크다(SG 0.15: 22 대 32/48).
- **11 (기판 혼합)**: SG는 stock_quick Job(`GPU-ab79c915…`), static은 CAP2 Job이다. 다른 GPU·시점·트리다. SG Job의 W3 전력 상한 노출은 16.2%(run별 8–44%, rate와 함께 증가)이고, 최대값은 꼬리의 꼬리 통계라 throttle에 가장 민감하다.
- **8 (비-운영점 인용) 변형**: SG arm은 `--disable-piecewise-cuda-graph`로 돌았다(`sq_plan.py:83`, raw argv 확인). upstream 기본이 아니다. Phase B 감사 E-①과 같은 범위 제한이다.
- **2 (untuned) 변형**: W3의 SG 정지 길이 = prefill이 decode와 겹치지 않는 구간의 길이이며, SG 노브 `chunked-prefill-size`(기본 8192)의 직접 함수다. 청크를 줄이면 정지가 짧아질 것으로 예측된다(미측정). 그러므로 말할 수 있는 것은 "SG 기본 청크 8192가 역전된다"까지다. W1 진단 B3(청크 −1)은 trace 최대 입력 3126 < 8192라 처치가 발화하지 않은 null이어서 W3 근거가 될 수 없다(Phase B 감사 ②).
- **기전**: "decode를 prefill과 겹치지 않는 엔진에서는 도착한 긴 프롬프트의 prefill 시간만큼 진행 중인 decode가 멈춘다"는 설명은 정합한다. SG 최대 정지는 rate와 함께 커지고, static은 100 ms를 넘는 정지가 0이다. 그러나 W3에서 정지와 도착 prefill을 1:1로 대응시킨 측정이 없어 **NOT-YET-SUPPORTED**다.
- 과부하 셀: 0.506 이상에서 PD-mux 셀은 과부하이므로 크기를 인용할 수 없다(결과 감사 r1 승계).

**허용 문안**
> "사후(미등록) 보기: W3(arXiv 요약, 실현 출력 약 1024토큰)에서 요청 내부 ITL p95 술어에 '요청 최대 토큰 간격 ≤ 1 s'를 더하면, 다른 Job·GPU의 n = 2 기술값에서 SGLang 기본 구성(piecewise CUDA graph OFF, 청크 8192)의 attainment가 static PD-mux arm보다 낮아진다. static arm의 최대 간격은 전 셀 ≤ 131 ms였고, SG 최대 간격 중앙값은 0.1 req/s에서 0.55–0.69 s, 0.759 req/s에서 2.7–7.2 s였다. 이 순서는 문턱에 의존한다 — 2 s에서는 저부하 차이가 잡음 수준이고, 5 s에서는 0.1–0.225·0.506 req/s에서 원래 순서로 돌아온다."

**인용 금지 문안**
1. "정지까지 보면 PD-mux가 SGLang보다 낫다." / "W3에서 순서가 역전된다."(문턱·사후·n = 2·다른 Job 단서 없이)
2. "요청 내부 p95 술어는 SGLang을 과대평가한다."(등록 술어 판정을 사후 술어로 뒤집는 재채점)
3. "SGLang은 W3에서 수 초 정지한다."(piecewise OFF·청크 8192·이 Job 단서 없이)
4. "정지 원인은 prefill과 decode의 직렬화다."(W3 1:1 대응 미측정)
5. 1 s 값을 rate별 attainment 크기로 인용하는 것(절벽 위의 값)

### 2-C. 다음 PREREG 등록 후보로 쓸 수 있는가 (근거만, 결정은 사용자)

**찬성 근거**
- 요청 내부 p95는 긴 출력에서 수 초짜리 정지를 구조적으로 흡수한다(1024토큰이면 51개). 사용자가 체감하는 끊김을 놓치는 실제 맹점이다.
- 문헌도 같은 맹점을 다른 방식으로 막는다. MuxWise는 TPOT 대신 TBT를 쓰며 그 이유로 "평균이 개별 토큰을 가린다"를 든다. Sarathi는 토큰 합산 P99 TBT를 쓴다.
- 코드가 단순하고 정확하다(2-A).

**반대 근거와 등록 조건**
1. **N2 위험**: L을 등록 범위 안의 자유 표면으로 두면 반전 시험이 이미 판정을 뒤집는다(1 s 대 5 s). 등록하려면 L을 이 원자료를 보지 않은 외부 근거로 사전 고정하거나, 단일 L 대신 고정 격자 곡선을 등록하고 읽기 규칙을 미리 박아야 한다(예: 격자 {0.25, 0.5, 1, 2, 4} s에서 "5개 중 4개 이상에서 부호가 같아야 부호 라벨"). 이번 데이터로 L을 고르면 그 선택이 새 자유 표면이 된다(P6 전례).
2. **N3 위험**: "PD-mux ≥ SG"를 정지 지표의 부호 예측으로 등록하면 출력공간에 사실상 거짓이 없다. static arm은 설계상 prefill을 별도 SM에서 돌려 최대 간격이 ≤ 131 ms이므로, L ≥ 0.15 s에서 D 쪽 값은 변하지 않는다. 등록 대상은 부호가 아니라 크기(SG 최대 정지 분포와 L의 교차점)여야 측정이 된다.
3. 같은 Job 안 arm, n ≥ 4 seed, paired CI가 필요하다(게이트 3, confound 11). 이번 2-seed·교차 Job 수치는 검출력 예보의 입력으로만 쓴다.
4. SG 구성 범위를 정해야 한다. 최소한 piecewise 상태를 공시하고, 가능하면 청크 크기를 하나 이상 추가 arm으로 둬서 "SG 본질"과 "기본 청크"를 가른다(confound 2). 추가 arm을 못 두면 "SG 기본 청크 8192 한정"을 등록 caveat로 고정한다.
5. 출력 길이 정규화 규약: 출력 길이가 다른 워크로드에서는 최대값이 긴 출력에 불리하다. W1 같은 가변 출력 셀이라면 "요청당 L 초과 간격 수"나 "요청 내부 p99" 같은 대안을 함께 등록하거나, W3형 고정 출력으로 한정한다.
6. cliff 회피: 사전 예보 단계에서 등록 L이 어느 arm의 최대 정지 분포 중앙 근처(±2배)에 놓이는지 공시한다(게이트 6).

**의견**: 조건 1–4를 갖추면 보조(secondary) 지표로 등록할 수 있다. 주 판정 라벨로 쓰려면 조건 1·2를 둘 다 만족해야 하는데, 지금 근거로는 그 설계가 서 있지 않다.

### 2-D. 해소 방법(확정 실험)

- 같은 Job 안에서 W3 고정 출력 1024, rate {0.1, 0.225, 0.3375}(용량 아래, 과부하 셀 제외)로 잰다.
  - arm: SG 기본(piecewise 상태 공시), SG 청크 2048, PD-mux best static(D16·D24)
  - 조건: seed ≥ 4, 같은 trace hash, 같은 커밋, cudagraph 상태 동일, clk·power 노출 기록
  - 주 지표: 사전 고정 L 격자에서의 attainment 곡선과 SG 최대 정지 분포(p50·p90)
  - 기전 확인: 정지 구간과 도착 prefill 시각을 `token_t_mono`·`t_sent_mono`로 1:1 대응시키는 사전 등록 분석
- 로그 공시: 결과 문서에 "`sq_maxitl.py` = 사후 지표, `LIM` 미기록"을 적는다. 재실행한다면 출력 파일명이나 JSON에 L을 기록한다(코드 수정 제안이며 실행은 호출자 결정).

### 2-E. 종합

- **코드**: **GO**(결함 없음, 사소 1건 `LIM` 미기록).
- **"1 s면 역전" 서술**: **GO-with-edits**. 허용 문안으로 범위를 좁히고 문턱 의존성(2 s 잡음, 5 s 복귀)·사후·n = 2·교차 Job·piecewise OFF·청크 8192를 병기한다.
- **정본 영향**: 없음. CONSENSUS §1/§2/§6과 모순하지 않고 등급 변경 근거도 아니다. Phase B 감사 E-①(piecewise OFF 범위)과 같은 방향의 제한이 붙는다. 인용 금지는 5개로 상한 안이다.

## 관련 파일
- 대상: `reports/SURVEY_slo_attainment_published_2026-11-02.md`, `workspace/engine-port/results/stock_quick/sq_maxitl.py`, `workspace/engine-port/results/stock_quick/RESULT_stock_quick_2026-11-02.md`
- 근거:
  - `workspace/engine-port/results/stock_quick/sq_plan.py:83`(piecewise OFF)
  - `workspace/engine-port/results/cap_char/policy_harness/ph_verdict.py:40-46`(`req_good`)
  - `workspace/engine-port/results/stock_baseline/sb_loadgen.py:117`(ITL 정의)
  - `workspace/engine-port/results/stock_baseline/AUDIT_result_phaseB_claims_r1_2026-11-07.md`(confound 8 변형 선례)
  - `workspace/engine-port/results/cap_char/COMPARE_shapeC_workloads_2026-10-20.md`(옛 "평균 203" 출처)
- 1차 출처:
  - `~/Experiments/KISTI/Papers/Bullet.pdf`(10쪽 Fig.15d, Table 4)
  - `~/Experiments/KISTI/Papers/Muxwise.pdf`(§4.1 Metrics, §4.3, §4.3.1)
  - arXiv 2403.02310(Table 2–3, Fig.10, 11쪽)
  - arXiv 2401.09670(Table 1, §6.1 "particularly interested in an SLO attainment of 90%", 부록 C)
