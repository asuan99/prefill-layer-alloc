# 변화 trace의 의의와 실 워크로드 근거 (2026-10-26, main 세션)

> 지위: 정본 아님, 새 측정 0. venue-strategist 조사(공개 trace 원본을 내려받아 직접 집계 + 논문 원문 확인)의 전사와 main 세션의 정리. [직접 계산] = 공개 원본 직접 집계(스크립트 scratchpad `scripts/{az,az2,bg,mc}.py`), [원문 확인] = 논문·데이터셋 페이지, [기억] = 미확인.

## 1. 우리 변화 trace의 실제 구조
`workspace/engine-port/results/slo_sched/sharegpt_vary_bench.sbatch:66-75`: 한 라운드 = `bench_serving` 두 번(LO `--num-prompts 200 --request-rate 3`, HI 같은 설정 rate 12), 3라운드 6구간. LO 도착 약 67 s, HI 약 17 s, **각 호출이 모든 응답을 기다려 구간 사이 큐가 비워짐(drain)**. 즉 "용량을 넘는 짧은 버스트의 반복"이며 일주기 변동의 모사가 아니다. (정본 HE0 수치가 정확히 이 스크립트로 나왔는지는 확인 필요.)

## 2. 공개 trace의 시간 변동
| trace | 도착률 변동 | 길이·혼합 변동 | 변화의 시간 척도 |
|---|---|---|---|
| Azure 2024 code(1주, 16.8M) [직접 계산] | 하루 안 시간당 max/min 5.85–11.3×, UTC 08–09시 6.7 → 19시 59.9 req/s | 긴 요청(입력 ≥ 2048) 10분 창 0.306–0.627, 1시간 창 안 변동 p95 0.12, 6시간 창 max 0.24 | 인접 1분 창 2× 변화 비율 0.000; 1분 rate max/min 1시간 창 p95 2.69, 3시간 5.75, 6시간 11.8 → **4×는 수 시간** |
| Azure 2024 conv(1주, 27.3M) [직접 계산] | 하루 안 1.68–2.19× | 긴 요청 10분 창 0.271–0.363, 6시간 창 변동 max 0.066 → **30↔60% 없음** | 6시간 창에도 max 2.9× |
| DynamoLLM 원문 [원문 확인] | Coding peak/avg 2.8×, peak/valley 34.6×; Conversation 1.7×/3.3× | "the popularity of request types changes over time" | 재구성 주기 클러스터 약 30분·병렬도 약 5분 |
| BurstGPT(공개분 61일) [직접 계산·원문 확인] | 1분 peak/mean 68.5, 1분 max/1시간 이동평균 중앙값 3.8·p95 20; API는 "aperiodic and bursty" | API 시간당 긴 요청 비율 p5 0.00–p95 0.999(소수 클라이언트 배치), 대화 ≥1024 p5 0.15–p95 0.49, API 비중 p5 0.00–p95 0.99 | **분 단위 급증 근거는 이 계열(버스트 클라이언트)** |
| Mooncake/Kimi(약 1시간) [직접 계산] | 1분 창 max/min 1.72(정상 상태에 가까움) | 긴 요청 비율 5분 창 ±5%p | MuxWise 원문 "up to 13× spike within 1min"은 미재현(창 정의 미확인) |
| Alibaba ServeGen(NSDI'26, 4개월·12모델) [원문 확인] | CV > 1 버스트, 오후 피크·새벽 저점 | 평균 입력 시간 변화 최대 1.63×, 출력 1.46×, 주로 상위 클라이언트 rate 변동 때문; 순진한 합성은 "misleadingly easier to serve", 프로비저닝 50% 과소추정 | 멀티모달 약 9시간 시점 step 급증 |
| Azure 2023(Splitwise, 약 1시간) | — | 1시간 안 거의 일정 | 시간 변화 분석 불가 |
에이전트·장문 RAG의 시간 변화를 수치로 특성화한 공개 자료는 찾지 못했다.

## 3. 다른 서빙 논문의 시간 변화 평가
정상 상태 Poisson rate 스윕이 PD 계열 top-venue 표준(DistServe·Sarathi [기억], MuxWise·Nexus·Llumnix [원문 확인]). 동적·적응 정책 주장 논문은 **실 trace 재생**(DynaServe: BurstGPT 42분 원 도착, Arrow: Azure·BurstGPT·Mooncake 1시간/10분 재생 + 타임스탬프 배율, DynamoLLM: 1시간·1일 재생). **실 trace 근거 없는 사각파 교대 trace를 주 평가로 쓴 top-venue 사례는 찾지 못함**(없다는 증명 아님).

## 4. 판정
- **존재 여부**: 변화 양상(도착률·혼합의 시간 변동)은 실재한다. 문제는 **우리 파라미터**다.
  - rate 4× 진폭: 실 범위 안(conv 하루 약 2× < 4× < code 하루 6–11×). 그러나 17–67 s 구간 + drain 교대는 집계 trace에는 없고(인접 1분 2× 변화 0.000), 버스트 클라이언트 근거만 있다 → "일주기 모사"가 아니라 "용량 초과 버스트 반복"으로만 정직하게 서술 가능, 그것도 drain 때문에 실제 버스트보다 쉬움.
  - 긴 요청 30↔60%: Azure conv에선 불가, code 단독도 주 단위 극단값 수준; 분 단위면 BurstGPT API 같은 단일 클라이언트 배치 유입만 근거.
- **신빙성 영향(어떤 주장이 흔들리나)**:
  - 흔들리는 것: HE0("동적은 best static 못 넘음")를 실 워크로드로 일반화하는 서술. 특히 실 trace처럼 수 시간에 걸쳐 천천히 변하면 "시간마다 static을 다시 고르는" 동적 정책은 전환 비용이 거의 0인데, 그 영역의 증거는 없다. DP 길이 라우팅의 30↔60% 교대 결과를 실 환경으로 일반화하는 것도.
  - 흔들리지 않는 것(정상 상태 측정 근거): PD-mux 대 fused 우위(P1-opint), layer-aware 정책 死(TPOT 42→124 ms), entanglement 기전, 해석적 비용 수식.
  - **비용 모델 기반 static 권고는 변화 trace 없이 세울 수 있다**: 변화가 느리면 시간 변화 goodput ≈ Σ_t(그 시점 rate·혼합의 정상 상태 goodput) → rate·혼합 격자마다 정상 상태 best split을 재고 "피크 기준 split이 모든 격자점에서 그 점의 최적과 ε 이내"를 보이면 그 범위의 어떤 느린 trace에도 성립. 단 게이트 2가 정상 상태 r8을 폐기한 이유(static끼리 ±1.3)가 cliff·잡음일 수 있으므로 cliff 밖·n ≥ 4 재측정 필요(미측정).
  - 결론: **변화 trace가 무의미해도 논문 전체가 무너지지는 않지만**, 동적·라우팅 정책 비교와 HE0의 일반화는 지금 근거로는 약하고 리뷰에서 지적받을 위험이 높다.

## 5. 보강 권고
1. 서술 정정: 기존 변화 trace를 "용량 초과 버스트 반복(drain 포함)"으로 부르고 일주기 대표성을 주장하지 않는다(정본 HE0 범위 문안 보강 후보 — promote 몫).
2. **실 trace에서 파라미터 유도**: Azure 2024 code/conv 분 단위 rate·10분 단위 길이 혼합 시계열에서 교대 진폭·주기(예: code 1시간 창 p95 2.7×, 3시간 5.75×), BurstGPT API에서 분 단위 버스트 진폭.
3. **실 trace 재생 1종 추가**: thinning(시간 척도 보존)으로 단일 GPU·DP 용량에 맞춘 Azure code 상승 구간(08→19 UTC 약 9×) 또는 BurstGPT 구간 재생, 압축 배율 공시. Azure는 토큰 수만 주므로 길이 맞춤 합성 프롬프트(radix OFF), 컨텍스트 초과분 절단 비율 공시.
4. **정상 상태 격자 논증**을 static 권고의 1차 근거로(§4), 변화 trace는 분 단위 과도 효과(제어 지연·backlog 이월) 보조 실험으로 위치 변경.
5. DP 라우팅 PLAN의 변화 trace(30↔60%)는 Azure code·BurstGPT 혼합 시계열에서 진폭·주기를 다시 정한다(결정 36 개정 후보).
모두 새 Job이 필요한 항목은 사용자 결정.
