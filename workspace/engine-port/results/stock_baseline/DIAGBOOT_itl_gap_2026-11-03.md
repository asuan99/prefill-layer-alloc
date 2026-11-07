# 진단 boot 결과 — 손대지 않은 SGLang의 ITL 정지는 어디서 오나 (2026-11-03, Job `pdmux-stock-baseline-diag-4d1b9da-20261103112203`, job-yv0ltzdugf2k)

> **결과 감사 r1(2026-11-07) 정정** (`AUDIT_result_phaseB_claims_r1_2026-11-07.md` E7·E8, `AUDIT_result_phaseB_recompute_r1_2026-11-07.md` 지적 7): (1) 아래 "같은 GPU 안 비교"는 **B 변형끼리(B1–B5)만** 같은 GPU(GPU-27097514)다. 표의 기준 두 행("J1 손대지 않은 SGLang"·"J1 PD-mux D44")은 **첫 번째 Job(GPU bc0be109)** 값이고 진단 Job 안에 PD-mux ON boot는 없다. B5(OFF 69 ms) 대 PD-mux ON(45 ms) 대비는 GPU를 가로지른 n=1 값이다. (2) B3은 단일 변수가 아니다: `--chunked-prefill-size -1`이 mem 자동값을 0.836 → 0.686으로 함께 바꿨다(B4가 mem 무관을 따로 보임). (3) "`--enable-pdmux`만 뺐다"는 argv 수준 단일이며 코드 경로 수준에서는 묶음(이벤트 루프·SM 분할·split-prefill·stream별 graph)이다. (4) "정지 비율 ≈10%"는 5.516 req/s 한정(3.15에서 4–5%). 본문 수치는 그대로 둔다.

> **DIAGNOSTIC — 측정 아님.** 봉인 Phase B 밖의 Job(캠페인 `stock_baseline_diag`), 셀 1개(W1 ShareGPT 120요청, 5.516 req/s, seed 201, trace hash `1bda5cbb…7483` = 첫 번째 Job과 동일), seed 1, 같은 GPU(GPU-27097514…) 안 비교 [★감사 r1 E7: B1–B5끼리만; 기준 두 행은 GPU bc0be109]. 수치는 기술값이며 정본 인용 금지. exit 0, raw sha 검증, 엔진 digest `2836112b…` 일치, pristine 보존 OK. 원자료 `results/stock_baseline/runs/<job>/raw/`(미추적). 진단 설계 `DIAG_sglang_default_itl_2026-11-03.md` §5.

| 변형 | 바꾼 것 하나 | 트리 | ITL p50 | ITL p95 | 토큰 20–25 ms | 토큰 >40 ms | TTFT p50 / p95 | attainment |
|---|---|---|---|---|---|---|---|---|
| J1 손대지 않은 SGLang (기준) | — | pristine | 21.3 | 63.2 | 0.525 | 0.103 | 90 / 741 | 0.217 |
| J1 PD-mux D44 (기준) | — | patched | 23.0 | 45.2 | 0.431 | 0.142 | 90 / 598 | 1.000 |
| **B5** | PD-mux argv 그대로, `--enable-pdmux`·config만 제거 | patched | 23.0 | **68.6** | 0.463 | 0.093 | 63 / 1225 | **0.192** |
| B1 | SG-stock + `--disable-overlap-schedule` | pristine | 23.1 | 67.7 | 0.497 | 0.092 | 62 / 1109 | 0.217 |
| B3 | SG-stock + `--chunked-prefill-size -1` [★mem 자동값 0.836→0.686 동반] | pristine | 21.4 | 63.1 | 0.526 | 0.105 | 90 / 813 | 0.225 |
| B4 | SG-stock + `--mem-fraction-static 0.6 --max-mamba-cache-size 48` | pristine | 21.3 | 63.7 | 0.508 | 0.100 | 90 / 673 | 0.208 |
| **B2** | SG-stock에서 `--disable-piecewise-cuda-graph` 제거 | pristine | **부팅 실패** — `PiecewiseCudaGraphRunner.__init__`: `AttributeError: 'typing.Union' object has no attribute '__module__'`(py3.14 이미지, multigpu 2026-10-12와 동일) | | | | | |

## 읽기 (기술)
- **B5가 가른다**: PD-mux와 같은 argv(mem 0.6·chunked −1·overlap OFF·mamba 48)에서 `--enable-pdmux`만 빼면 ITL p95가 45 → 69 ms로 돌아가고 attainment 1.00 → 0.19. 정지는 SGLang 기본값 차이가 아니라 **PD-mux의 prefill/decode 동시 실행이 없을 때 생기는 것**이다. [★감사 r1 E8: "동시 실행" 귀속은 미분리 — 올바른 읽기는 "`--enable-pdmux`가 켜는 경로 묶음이 없을 때"; n=1·교차 GPU] 진단 §5 B5 예측과 일치.
- B1·B3·B4는 정지 비율(토큰 >40 ms ≈ 9–10%)·p95를 바꾸지 못한다 → overlap·chunk 크기·메모리 설정은 배제. (B1·B5는 overlap OFF 때문에 TTFT p50이 62 ms로 낮아지고 p95가 커지는 부수 효과만.)
- **B2 미해결**: 이 이미지에서는 손대지 않은 SGLang이 piecewise CUDA graph를 켤 수 없다(py3.14 환경 문제, SGLang 코드 문제 아님). 따라서 prefill은 eager로 돌고, 정지의 ~42 ms 고정 비용 중 eager 발행 CPU 몫은 **분리되지 않았다**. vLLM은 piecewise가 켜져 있으므로 vLLM 대 SGLang 비교에는 "SGLang은 이 환경에서 piecewise 불가"라는 범위가 붙는다.
- 두 번째 봉우리 길이(60–65 ms)와 비율(≈10%)은 네 변형에서 안정적이라, 첫 번째 Job의 `LADDER_BOTTOM_FAIL`은 설정 우연이 아니다.

## 함의 (판정 아님, 다음 등록 입력)
1. 손대지 않은 SGLang의 SLO(ITL p95 60 ms) 준수 용량은 이 셀 격자(≥ 5.516)에 없다. 값을 알려면 아래쪽 사다리(새 등록)가 필요하다.
2. PD-mux 대 stock SGLang의 차이는 "정지를 decode와 겹치느냐"이고, 그 정지는 요청 도착마다 1회·길이 ~42 ms(짧은 프롬프트에서도)다. 이것이 Phase B가 재려던 기전 그 자체다.
3. 공정성 범위: SGLang eager prefill(piecewise 불가) 대 vLLM piecewise ON. 해소하려면 py3.12 기반 SGLang 이미지가 필요하다(별도 결정).
