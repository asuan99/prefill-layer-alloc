# Phase B 사전 예측 — static PD-mux가 손대지 않은 SGLang·vLLM 대비 얼마나 좋을 것인가 (2026-11-02, 봉인 전 작성)

> 지위: **사전 예측**(봉인 대상, PREREG rev2 §8에 포함). 측정 전에 쓴 기대치이며 채점 뒤 맞고 틀림만 기록한다.
> **rev1(2026-11-02, 재감사 r2 반영)**: (1) 등록 채점 행 = PREREG §8 P-1…P-5뿐. 아래 표의 나머지 행(TTFT·ITL 띠, D34 대 D44, 지속/버스트 비, mixed-chunk, Q8 지연)은 **추정량·arm이 등록되지 않은 기술 예측**이며 측정 뒤 "사전등록 적중"으로 인용하지 않는다(인용 금지 문장 9). (2) "둘 다 포화" 예측은 §5에 따라 채점 불가이므로 적중 주장에 쓰지 않는다 — 그 예측의 쓸모는 "포화가 나오면 설계가 예상한 대로였다"는 기록뿐이다. (3) 용량 띠는 P-1 [6, 10]으로 통일(아래 7–10 삭제). (4) 지속/버스트 비는 격자 {1.0, 0.8, 0.5} 위의 값만 가능하므로 띠가 아니라 격자점으로 적는다. 라벨: [측정] 기존 데이터 · [유도] 그 데이터에서 계산 · [가정] 검증 안 된 해석.
> 근거 데이터: KISTI P1-opint(Granite·Zamba2, in2000/out96, cudagraph ON, n=5) · VESSL M3 J1(Nano, N=1) · VESSL H Job(Nano, PD-mux D44, W1 LO 2.74 req/s, n=4) · CM-2 prefill floor(9 ms + 0.111 ms/토큰 @108 SM) · CAP1-R W1 프롬프트 길이 분포(p50 36–51, p90 724–1092, 최대 3.5–9.2k 토큰) · `policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`.

## 0. 한 줄 예측

**이 워크로드(ShareGPT)·이 SLO(TTFT 3 s, 요청 내부 ITL p95 60 ms)·용량 아래 셀에서는 PD-mux가 goodput으로 stock을 이기지 못할 가능성이 크다(둘 다 포화, 차이 0). PD-mux는 TTFT에서 뚜렷이 불리하고(+30~+100%), ITL 꼬리에서만 유리하거나 비슷하다.** KISTI의 +27%(Granite)는 "모든 요청이 2k prefill, 출력 96토큰"이라는 PD-mux에 가장 유리한 조합에서 나온 값이며 ShareGPT로 옮겨지지 않을 것으로 본다.

## 1. 왜 그렇게 보는가 — 기전 [유도]

PD-mux가 이기는 유일한 경로는 **stock의 prefill 정지가 decode 요청의 ITL p95를 SLO 밖으로 밀어낼 때**다. 요청 내부 ITL p95 술어는 한 요청의 토큰 중 5%까지 SLO 초과를 허용한다.

| 양 | KISTI Granite (이긴 셀) | Phase B W1 (예측) |
|---|---|---|
| prefill 1회 정지 길이 | 2000토큰 ≈ 105 ms (T_dec 7 ms의 15배) [측정 r_ser p50≈15] | p50 40토큰 ≈ 13 ms(정지 아님), p90 1k ≈ 120 ms, 최대 9k ≈ 1 s [유도, CM-2 floor] |
| 정지 빈도 | 모든 요청이 2k → rate 4에서 4회/s | SLO 초과(>60 ms)급 정지는 500토큰 이상 ≈ 상위 10–15% → 80% 셀(≈4 req/s)에서 0.4–0.6회/s |
| 요청 수명 | 96토큰 × 10 ms ≈ 1 s | 출력 ~200토큰 × 20 ms ≈ 4 s |
| 요청당 정지 노출 | ≈ 4회 / 96토큰 = **4%** → p95 경계 | ≈ 2회 / 200토큰 = **1%** → p95 안 |
| 결과 | stock ITL p95 52→69 ms(SLO 60 횡단) → goodput 손실 | stock ITL p95가 SLO 안에 머무를 것 → goodput 손실 없음 |

즉 ShareGPT는 "짧은 프롬프트·긴 출력"이라 정지가 드물고 요청이 길어 p95가 흡수한다. 이것이 PD-mux에 가장 불리한 조합이다. [가정: 정지는 prefill 길이에 비례하고 overlap이 없다 — Phase B 계측 boot가 r_ser로 확인]

## 2. 질문별 예측 (채점 띠; 채점은 "부호 + 띠 안/밖")

| 질문 | 비교 | 50% 셀 | 80% 셀 | 근거 |
|---|---|---|---|---|
| Q2 goodput | PD-mux D44 vs stock SGLang | **둘 다 포화**(확률 ~0.8), 아니면 ±3% 안 | 둘 다 포화(~0.6); 포화 아니면 PD-mux **+0~+10%**(stock에 1k 이상 정지가 겹칠 때만) | §1 |
| Q2 TTFT p50 | 〃 | PD-mux **+30~+70%** | +40~+100% | prefill SM 64/108 → 프롬프트당 ×1.5–1.7 [유도]; M3 A LO: 152 vs 123 ms p95(+24%) [측정 N=1] |
| Q2 ITL p95(요청 중앙) | 〃 | PD-mux **+20~+60% 나쁨** (31 vs 20 ms, M3 A LO) — 저부하에서는 decode SM 44가 108보다 느린 효과가 정지 회피 효과보다 큼 | **−30~+20%**(부호 불확실): 정지가 늘면 stock이 나빠져 역전 가능 | CM-1 decode SM 민감도 2.4–2.9× [측정]; H Job D44 LO ITL p95 35.5 ms [측정] |
| Q3 goodput | vLLM vs SGLang | 둘 다 포화(~0.8) | 둘 다 포화(~0.6); 아니면 ±10% 안 | 같은 chunked prefill 구조(vLLM V1은 prefill+decode 혼합 배치, SGLang은 overlap) [가정] |
| Q3 TTFT p50 | 〃 | **±30% 안**, 부호 미정 | 〃 | Nano에서 vLLM 데이터 0 |
| Q3 ITL p95 | 〃 | ±30% 안 | vLLM 혼합 배치가 정지를 step 안으로 옮겨 **p95는 비슷하거나 약간 높음** [가정] | Phase A3 r_mix ≈ r_ser 관찰(KISTI mixed ON) |
| Q4 | PD-mux vs vLLM | Q2와 같은 모양 | 〃 | Q2 + Q3 합성 |
| Q5 | mixed-chunk vs stock SGLang | goodput 동일; ITL p95 **−10~+10%**; TTFT +0~+20% | 〃 | 정지가 step 안으로 이동, 총량 불변 |
| Q8 goodput | 패치 트리 OFF vs 손대지 않은 트리 | 둘 다 포화 → 지연 지표로 | 〃 | — |
| Q8 TPOT 평균 | 〃 | **±3% 안(EQUIV)** 확률 ~0.7 | 〃 | host 오버헤드 ≈ 0.1 ms/forward(<1%); conv1d int64 커널은 prefill만 |
| Q8 TTFT p50 | 〃 | 패치 OFF **+0~+5%**(conv1d int64) | 〃 | 부록 D |
| Q7 W3 goodput | PD-mux vs stock, 7k 프롬프트 | 둘 다 포화(~0.6). 정지 0.8 s/회지만 출력 1024토큰·rate ≈0.2 → 요청당 ≈5회/1024 = 0.5% | — | §1 계산 |
| Q7 W3 TTFT p50 | 〃 | PD-mux **+40~+80%**(7k prefill 64 SM ≈ 1.3 s vs 0.8 s) | — | CM-2 floor |
| D34 vs D44 | PD-mux 두 점 | EQUIV(고원) | EQUIV | R Job D34–64 1.5% 안 [측정] |
| 용량(P-1 등록) | stock SGLang 버스트 λ\* | **[6, 10] req/s**(PD-mux 7.29와 같거나 높을 것으로 봄: 108 SM 전부 사용) | — | CAP1-R [측정] |
| 용량 | vLLM 버스트 λ\* | SGLang의 0.8–1.25배 | — | — |
| 지속(기술 예측) | 지속 격자점 / 버스트 | PD-mux D44: **0.5**(포화 완료율 5.06–5.34 < 0.8×7.29 = 5.84이므로 0.8점은 탈락 예상) · stock SGLang: **0.8**(버스트 용량이 더 높아 0.8점이 5.3 아래일 가능성) | — | RECHECK [측정]; 격자 {1.0, 0.8, 0.5} |

## 3. 어디서 PD-mux가 이길 수 있는가 (이번 측정 범위 밖, 다음 설계 입력)

1. **정지 노출이 요청당 5%를 넘는 워크로드**: 긴 프롬프트 + 짧은 출력(RAG·요약·분류). KISTI in2000/out96이 정확히 이것. ShareGPT는 반대.
2. **ITL SLO가 더 엄격할 때**(예: 30 ms): stock의 평범한 정지(500토큰 ≈ 65 ms)도 전부 위반이 됨.
3. **용량 근처(90–100%)**: 큐가 생겨 정지가 TTFT로도 번짐. 이번 설계는 cliff 때문에 80%까지만.
4. **TTFT 손해가 작은 분할**(D34→prefill 74 SM): 고원 안이라 goodput은 같고 TTFT만 개선될 수 있음 — PM-ref34가 이를 봄.

## 4. 이 예측이 틀리면 무엇을 뜻하나

- 80% 셀에서 stock SGLang goodput이 PD-mux보다 **5% 이상 낮게** 나오면 → §1의 "정지 노출 1%" 계산이 틀린 것(정지가 더 길거나 overlap 경로가 decode를 더 막음). 계측 boot의 r_ser 분포로 확인.
- PD-mux TTFT가 stock보다 **좋게** 나오면 → stock 스케줄러의 prefill 큐잉이 SM 분할보다 큰 비용 — 예상 밖, 기전 재검토.
- Q8에서 TPOT가 3% 이상 다르면 → 패치 트리의 host 오버헤드가 무시 불가 → 지금까지 모든 OFF 참조의 범위 축소.

## 5. 채점 규칙

각 행을 측정 후 HIT(부호·띠 안) / MISS / UNSCORABLE(포화·INVALID)로만 기록. 종합 판정 없음. 식 수정은 채점 뒤 별도 절차.
