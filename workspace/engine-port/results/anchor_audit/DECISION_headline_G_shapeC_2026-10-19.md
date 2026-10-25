# 사용자 결정 2026-10-19 — headline 비중·공통 워크로드·claim 확장 (main 세션, 측정 0)

전제: `VERDICT_claimE_first_arrow_2026-10-19.md`(claims-auditor 1회차). 선택지 원문은 같은 세션 대화의 경로 비교표(G / E-a / E-α / E-β / shape C).

1. **(가) headline = Claim G 중심** — "PD-mux 이득의 특성화·일반화"를 뼈대로 둔다. 구체 범위(E의 위치, G의 hybrid 고유성 근거, M3 역할)는 이어서 논의(미정).
2. **(나) shape C(긴 입력 + 긴 출력)를 공통 워크로드로 편입** — 구체값·`max_prefill_tokens`·G0/M3 반영 방식은 미정(G0 PREREG·M3 rev13 몫).
3. **(다) Claim E 문안을 memory/admission 예측(Q18, CM-5)까지 넓힐 필요는 인정** — 확정 문안·허수아비 위험 처리는 (가) 논의 뒤 이어서 결정.

금지: 이 결정을 "Claim E 반증"으로 서술하지 않는다(VERDICT 금지 문안).

## 후속 결정 (같은 날, (가) 범위 — 권고안 채택)
4. **G0에 순수 attention 대조 모델 편입**(예: Qwen2.5-7B, 같은 백엔드) — "G가 hybrid와 transformer에서 다른가"를 1차 비교로. α(hold-out) 재사용 가능. 모델·백엔드 확정은 G0 PREREG 몫.
5. **shape C = G0 두 번째 표준 셀**(arXiv-Sum은 긴 입력·짧은 출력이라 shape C 아님 — 출력 mean 203 [SPEC v0.2 :164]). 데이터셋 후보 OpenThoughts/Mooncake(10-15 P-C 논의), 확정은 G0 PREREG.
6. **M3는 G0 뒤로 연기**, J1(N=1) 결과 분석(CPU)만 먼저. J1 throttle(GPU0 61%)은 G0 계측 공시 항목(F-1)으로 처리.
7. E의 위치 = 기본 "열린 질문 + 시험 설계 공시", β(memory/admission)는 (다) 문안 확정 뒤 cap 특성화 Job(R16-3)에 계측 얹기.

## (다) 결정 (같은 날, 권고안 채택)
8. **Claim E 확장 문안 = 후보2 본문 + 후보1 anchor** (허수아비 점검: venue-strategist, 엔진 층 Q18 = 허수아비, Bullet/MuxWise = 변수 부재).
   - 본문(후보2): "SGLang에서 hybrid의 cap 하나가 admission·decode bs 상한·state 예약을 결합한다. 그 결합의 goodput 비용을 특성화한다."
   - anchor(후보1): "Bullet/MuxWise의 예측기는 메모리 변수 없이 엔진 admission을 상속한다. hybrid로 이식하면 decode 배치 상한이 KV가 아니라 고정 state 슬롯(cap)에 묶이고, ctx < [실측 전환점]에서 구속한다."
   - E(원안, decode floor 예측) = 열린 질문. 후보3(외부 KV 전용 용량 모델 X)은 X 실명 식별 전 금지.
   - 금지: "generic 스케줄러는 mamba state를 무시한다" · "Bullet/MuxWise가 C_mem을 오예측한다" · "~52k에서 구속 전환"(유도값).
   - `handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md` §3-2의 "generic 예측기(KV만 보는 모델)는 hybrid의 C_mem을 과대평가" 문장은 이 결정으로 낡았다(promote 때 정정 포인터).
9. 실험 = cap 특성화 Job(R16-3)에 I-5 거절 사유 계측 + cap×ctx 스윕(설계 초안 `../cap_char/`).

## 결정 8 개정 → 8′ (같은 날, 사용자 질의 후 승인)
경위: (1) SGLang mamba pool은 부팅 시 정적 할당, 슬롯 고갈 = assert(프로세스 사망), radix OFF에서 슬롯 수 = cap → cap≡슬롯 결합은 **SGLang 정적 풀의 구현 결과**(Jenga/vLLM V1 통합 풀이면 소멸). 할당기는 바꾸지 않는다(사용자). (2) I-5 조사(engine-porter): radix OFF에서 슬롯 사유와 cap 사유는 항등 → 거절 사유는 KV / cap(≡슬롯) 2분류. (3) "~52k 경계"는 cap 48 아티팩트지만 long-context 구간 자체는 hybrid 의의가 큰 곳이라 유지(사용자).
- **8′ E′ 문안**: "hybrid의 요청당 메모리는 길이 무관 고정 state s를 가져, 짧은 ctx(< s/k, Nano-9B-v2 ~9k [유도: 0.1435 GB ÷ 16 KiB = 8.8k(GB)–9.4k(GiB), 단위 확인 필요])에서는 s가, 긴 ctx에서는 (작은) KV가 지배한다. 이 비용 구조와 goodput 영향을 양 구간에서 특성화한다. SGLang 정적 풀에서는 cap이 admission·decode bs·state 예약을 함께 정한다(실험 설정 공시)."
- 이전 8의 후보2·후보1 문안은 대체됨("슬롯 상한"은 할당기 종속이라 기여 문장에서 뺀다).
- 실험: cap 블록 {48, C_mem} × 표준 셀 + long-ctx 블록 ctx {32k, 64k, 96k}(+128k 프로브 결과에 따라), cap = C_mem(L), ctx 비례 TTFT SLO. 설계 `../cap_char/DESIGN_cap_longctx_v1_2026-10-19.md`.
- 금지 추가: "~52k가 hybrid 경계" · "9k 경계 실측"(유도값, config 산술).

## 후속 결정 (2026-10-20, 설계 v1.1 §5 권고안 채택)
10. **attention 대조 모델 = Llama-3.1-8B-Instruct**(128k 네이티브, `--attention-backend flashinfer` 명시). 결정 4의 예시 Qwen2.5-7B를 대체. 교락(토크나이저·k 8×·attn 32층 RoPE·gated 라이선스·`ignore_eos` 필수)은 설계 v1.1 §5-2대로 공시.
11. **fused_tuned arm 추가**(cap 블록·long-ctx 블록): G0 튜닝 argmax config(mixed-chunk 포함). 근거 = J1 감사 1번(미조율 분모).
- shape C 데이터셋은 미정(LongProc의 서빙 선례 질의 중).

## 결정 12 (2026-10-20, shape C 재정의 — 사용자 결정)
경위: LongProc은 서빙 선례가 없고, 출력 길이를 특정 과제군에 묶어 연구 범위를 임의로 좁힌다(사용자). 비교 메모 `../cap_char/COMPARE_shapeC_workloads_2026-10-20.md`.
- **LongProc 제외** — 측정에도, 수치 근거로도 쓰지 않는다.
- **입력 원천 = arXiv-Sum(≤16k 표준 셀) + LongBench v1/v2(길이 구간별)**. 구간 표본이 부족하면 InfiniteBench로 보완한다. long-ctx 블록은 합성 고정 길이 대신 **실제 문서를 길이 구간 {32k, 64k, 96k} ±10%로 선별**한다(표본 실측 후 확정).
- **출력 길이 = 강제 출력 요인 {자연(~200), 1k, 4k}**. 근거는 데이터셋이 아니라 비용 구조(생성 중 L\* 횡단, Little 동시 수)다. 블록별 수준은 PREREG에서 GPU-h로 정한다.
- **shape C = (입력 구간 × 출력 길이) 격자의 한 영역**이며 단일 데이터셋이 아니다. 결정 5("shape C = G0 두 번째 표준 셀")는 "arXiv-Sum 입력 × 강제 출력 {1k, 4k}"로 구체화된다.
- 선행: LongBench v1/v2·InfiniteBench 길이 분포 CPU 실측(Nano·Llama 토크나이저).

## 결정 12′ (2026-10-20, 결정 12의 long-ctx 입력 부분 개정 — 사용자 결정)
경위: "목표 길이 T 절단"은 벤치마크를 임의로 수정하는 것으로 읽힌다(사용자). 한도 초과분을 거르면 v2는 38–42%, InfiniteBench는 46–50%가 빠진다[측정, `../cap_char/longctx_stats/`].
- **ctx 통제 축 {32k, 64k, 96k, ~124k} = 합성 고정 길이 입력**(bench_serving random 계열, 합성임을 공시). 목표 길이 절단·±10% 선별은 쓰지 않는다.
- **실제 분포 셀 = LongBench v1 원본 그대로**(Nano·Llama 모두 한도 안 100% [측정]).
- **LongBench v2·InfiniteBench = 보류**(공식 가운데 절단 프로토콜 사용 여부는 나중에 결정).
- 프로브에 G7 추가: 같은 길이에서 합성 입력 대 실제 문서의 prefill 시간 비교("내용 무관" 가정 확인).
- Llama 가중치 적재: 로컬 HF 토큰(계정 확인, gated 접근 OK)을 적재 시점에 SSH 표준입력으로 `pdmux-build`에서 일시 로그인 → 고정 리비전으로 받기 → 즉시 logout. VESSL org secret은 만들지 않는다.

## 결정 10′ (2026-10-20) — 대조 모델 Base로 정렬
- 대조 모델 = **meta-llama/Llama-3.1-8B (base)**, rev `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. 주 모델이 `Nemotron-Nano-9B-v2-Base`(rev `dc0661c…`)이므로 짝을 맞춘다. 아키텍처·비용은 Instruct와 같다. 로컬 HF 토큰으로 gated 접근을 확인했다.
- 사실: `Nemotron-Nano-9B-v2-Base`와 `-v2`의 `tokenizer.json` sha가 같다(`3277c00f…`). ARXSUM_STATS·LONGCTX_STATS의 Nano 길이는 Base 모델에 그대로 적용된다(M3 PREREG rev12 :127의 미확인 대조 해소).
- 128k 프로브 실행 승인(사용자, 2026-10-20): pdmux-build 적재 → pdmux-probe 실행, 하드캡 120분.

## 결정 13 (2026-10-20) — long-ctx 범위·mem-fraction (권고안 채택)
- **long-ctx 통제 축 상한 = 64k**({16k, 32k, 64k}), **`--mem-fraction-static` 0.6을 모든 arm·블록(cap 블록·G0 포함)에 통일**. 근거 = 128k 프로브 1·2회차(`../cap_char/probe_128k/RESULT_probe_128k_2026-10-20.md`): 0.82는 chunk -1 활성화 OOM, 0.6에서 Nano chunk -1이 약 95k 이상에서 Triton causal\_conv1d int32 오프셋 넘침 추정(감사: PLAUSIBLE 강함). ★서술 범위(감사 6번): 64k는 측정 한계가 아니라 격자 선택이고, 한계는 chunk -1 Nano arm 공통(PD-mux 고유 아님). "PD-mux는 64k를 넘을 수 없다"·"OOM 때문" 금지. 0.6에서 Llama KV 풀 263,900 토큰 → shape C KV 입장 제한 비대칭 점검 필요, 0.82 결과와 혼합 금지.
- 3회차 짧은 프로브: G4·G7(≤64k) + 커널 오류 임계 확인(L = 94,500 / 95,100, `CUDA_LAUNCH_BLOCKING=1`).

## 결정 14 (2026-10-20) — 연구 프레이밍 (권고안 채택)
- **주 목적 = hybrid 서빙 비용 모델(C)**: Nano 대상, 단일 GPU 기준 + N·TP 항 확장.
- **PD-mux 이득(G) = 비용 모델의 검증 대상**(G0·M3 셀이 검증 데이터).
- **"왜 hybrid인가"(H) = 비용 모델의 hybrid 고유 항을 transformer와 대비**해 답한다. Llama는 그 항의 최소 대비 측정에만 쓰고, 모델 간 goodput 이중차분(rev0 Q-B)은 폐기한다.
- 설계: 항별 고정 배치 마이크로 측정 → 조립 → λ\*\_SLO 검증. 계획 `../cap_char/PLAN_costmodel_terms_2026-10-20.md`(약 25–33 GPU-h [유도]). `PREREG_CAP_LONGCTX_rev0_2026-10-20.md`는 대체됨(이력).

## 결정 15 (2026-10-20) — 선행 조건 (권고안 채택)
- **I-5 엔진 동기화와 Triton causal\_conv1d int64 패치를 한 번의 ENGINE\_DIGEST 변경으로 묶는다**. GPU correctness gate도 1회로 하고, 패치 반증 실험(감사 2번 ①–③)을 그 게이트에 포함한다.
- **fused\_tuned config는 검증(V)보다 먼저 G0 튜닝으로 정한다**. 튜닝은 Nano cap = C\_mem(mem-fraction 0.6)에서 한다. 명세 R8-2의 "cap = G0 cap" 조항 개정이 필요하다(PREREG에서 명시하고 감사로 넘긴다).
- 다음 순서: 비용 항 PREREG 작성 → 규칙층 감사. 병행해서 engine-porter가 I-5 + int64 패치를 준비한다(CPU).

## 결정 16 (2026-10-20) — PREREG rev1 열린 결정 (권고안 채택)
- **O-1** R8-2′ 개정 승인: fused\_tuned 튜닝은 cap\_V = C\_mem(출력 예약 항 반영 산식), mem-fraction 0.6, 자연 출력 셀에서만 한다. T4가 cap 특성화 Job(R16-3)을 대체한다.
- **O-2** I-4(decode 커널 GPU 시간)와 I-6(cudagraph 패딩)을 I-5·int64 엔진 묶음에 포함한다. digest는 1회만 바꾼다.
- **O-3** 96k·124k 확장은 int64 패치 gate 통과를 조건으로 넣는다.
- **O-5** 검증 V의 PD-mux arm은 2개다.
- **O-7** 검증 Job은 분할한다.
- **GPU gate는 Job 1개(약 8.5 h)가 아니라 비교 단위로 분할**한다(사용자 지적: 장시간 단일 Job과 의도적 crash가 섞이면 위험). 각 Job은 자기 비교를 Job 안에 완결한다. 앞 단계가 PASS일 때만 다음 단계를 제출한다.

## 결정 17 (2026-10-20) — 감사 상한 도달 후 처리 (권고안 (a) 채택)
- PREREG rev2 2회차 NO-GO(`../cap_char/VERDICT_prereg_costmodel_rev2_rules_r2_2026-10-20.md`): **항 측정(T1–T4, J-A·J-B·J-C)만 caveat를 붙여 봉인용으로 분리**한다. 검증 V는 항 데이터를 받은 뒤 **별도 PREREG로 새로 감사**받는다. 데이터를 본 뒤 구조를 고르는 위험은 그 PREREG에서 공시한다.
- GPU gate 진행 순서: experiment-runner가 G-α/β/γ 하네스를 작성하고, 병행해서 claims-auditor가 gate 설계 감사 3건(I-4 허용 범위의 SPEC ±5% 이탈, TOST 짝당 seed 1개, ITL p95 짝 분산 미측정)을 본다. 이후 G-α dry-run → 제출 승인.

## 결정 18 (2026-10-20) — rev3-MEAS 열린 결정·gate 하네스 이탈 (권고안 채택)
- rev3-MEAS(`../cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md`):
  - O-12: shape C용 T1 셀 10개를 J-A에 추가한다((88, 8k) 등; (88, 9k)가 INFEASIBLE이면 (80, 9k)로 대체).
  - O-4: TP2 96k는 gate 단위 시험으로 경계 이동을 보고, 서빙 96k는 정상 점으로 둔다.
  - O-6: TP2 앵커는 순차 1장.
  - O-8: T4 fused\_sym 대칭 점검을 포함한다.
  - O-9′: 적합 기준은 [제안]값 그대로 봉인한다.
  - O-10′: 라운드 유지(기본 2, 108v·Llama 4).
  - O-13: T2′ 범위는 확장하지 않는다.
- gate 하네스(engsync `9561f867`)가 설계에서 벗어난 점을 수용한다:
  - REG sha `f3310545…` → `1c20b953…`.
  - TOST CI를 α 분할(초기·확장 각 단측 0.025, 95%)로 한다.
  - 확장 짝은 등록된 1회 별도 Job으로 돌리고 pool-gamma로 합산한다.
- 실행 승인: G-α 제출 → PASS면 G-β → PASS면 gamma-A·F 동시 제출(짝 수는 ITL SD 예보 뒤 확정). 각 제출 직전에 dry-run을 확인한다.

## 결정 19 (2026-10-20) — G-α FAIL 처리·gamma 추가 (권고안 채택)
- G-α 1회차 FAIL 1건(`i5i6_B2`: PD-mux stair에서 bs=40 미방문)은 검사 기대값 결함으로 보고 **검사를 수정한 뒤 G-α를 재실행**한다.
  - PD-mux arm의 기준: bs\_real이 1부터 최댓값까지 연속이고, 그 최댓값이 I-5 admission의 동시 decode 최댓값과 일치한다.
  - 수정은 "1회차 데이터를 본 뒤의 수정"으로 공시한다. 판정은 재실행 데이터로만 내린다.
  - 수정 diff는 하네스층 재감사(마지막 1회)를 거친 뒤 제출한다.
- gamma에 **계측 OFF/OFF 짝 2–3개**(같은 Job 안 흔들림 측정)와 **stall 경계 진단 라벨**(stall 비율 5%±1%p인 seed 표시, 판정 불변)을 추가한다.
- cooperative barrier 우려(격리 문서 `deprecated/reports/quarantine_root/ssm_cooperative_barrier_context_corruption.md`): 현 엔진 SSD·conv 커널에 cooperative launch·grid sync·smid 사용이 0건이고 서빙 실측도 정상이라, 별도 감사는 하지 않는다(사용자가 외부 출처를 주면 재검토).

## 결정 20 (2026-10-20) — G-β 하네스 수정·재실행, gamma (a) (권고안 채택)
- G-β 1회차 FAIL(engsync `afe94ba2`)은 하네스 기대값 결함 3건으로 판단한다. 반증 증거 자체는 예측대로 나왔다. 수정은 **하네스층 감사 상한 도달 후 사용자 승인**으로 하며, 1회차 데이터를 본 뒤의 수정임을 공시한다. 판정은 재실행 데이터로만 한다.
  - ① 단위 시험: torch 참조와 atol로 비교하던 것을 "패치본 view 출력 sha = 원본 커널 contig 출력 sha"(유효 길이) 기준으로 바꾼다. 붕괴 판정은 "원본 view ≠ 원본 contig 또는 crash"로 한다.
  - ② w11328 사례의 offset·dim을 TP2 폭에 맞게 고친다.
  - ③ 비차단 boot는 "illegal 서명 + 사망"만 요구한다. traceback 위치는 차단(blocking) boot에서만 요구한다.
- gamma: **(a) OFF/OFF 진단 짝을 제외**한다(r1 gamma 설계). OO는 확장 Job으로 옮긴다. STALL\_BOUNDARY 라벨은 stage가 아닌 셀 기술값이므로 유지하되, 판정 경로(run\_gate 1521)에 들어가지 않음을 테스트로 확인한다.
- G-β 재실행이 PASS하면 gamma-A·gamma-F를 동시에 제출한다(사전 승인).

## 결정 21 (2026-10-21) — gamma 관측자 효과 분해 (권고 (가) 채택)
- gamma-A(agnostic)가 OBSERVER\_EFFECT(λ −2.6%, ITL p95 +2.0%)로 나왔다. 원인 계측을 찾기 위해 **계측별 분해 Job**을 돌린다: agnostic에서 {OFF, I-4만, I-5만, I-6만} × 4회, A100 1장, 약 3 h. 판정 Job이 아니라 진단(기술값)이며, 수준 비교는 Job 안에서만 한다.
- fused\_sym의 확장 Job(+6짝)은 하지 않는다(ITL SD 4%로 확장해도 판정 불가 예상). 분해 결과를 보고 다시 판단한다.
- export 절단이 재발했다(gamma-A·F). 분해 Job에서는 큰 원자료를 클러스터 볼륨에만 두고, export 대상은 요약·판정 파일로 줄인다.

## 결정 22 (2026-10-21) — 계측 정책 (권고안 채택)
- 근거: 분해 Job(engsync `results/engine_gate/pdmux-engine-gate-decomp-d1f2ddb-20261021022225/DECOMP_RESULT.json`, agnostic·n=4)에서 TEL 묶음(`PDMUX_TELEMETRY_PATH` + `PDMUX_PHASE_EVENTS=1`, 두 knob 몫 미분리)이 λ −1.7%, I-4 증분이 ITL p95 +1.0%로 FLAGGED. I-5·I-6은 CI∋0(저검정력).
- **판정 측정**(arm 비교·SLO·goodput 판정)은 모든 arm에서 계측 OFF로 잰다. 계측 효과는 진단 boot와 **같은 Job 안 ON/OFF 짝**으로 재서 공시한다(보정 계수로 쓰지 않는다).
- gamma는 **"관측자 효과 특성화 완료"로 닫는다**. G-γA·G-γF는 PASS가 아니며, 엔진 묶음 봉인 조건은 G-α·G-β PASS + 이 결정으로 대체한다.
- 항 측정(rev3-MEAS)은 계측 레코드가 estimand의 정의라 계측을 끌 수 없다. 이 정책의 적용 형은 rev3-MEAS §2-2′(계측 매트릭스 + Job 내 OFF 진단 boot)에 둔다.

## 결정 23 (2026-10-21) — 병합 방식·rev3.1 O-14·O-15·감사 후속 (권고안 채택)
- engsync(`engine-sync-i5-int64`) → main은 **squash 병합**한다. `results/engine_gate*`의 원자료(`.so`, Triton 캐시, SGLang 소스 사본, `power_samples.csv`, 대형 `.gz`·`.csv`)는 커밋하지 않고 저장소 밖 `_vessl_fetch/<job>`에 두며, 빠진 파일 목록·sha256 manifest만 추적한다. `.gitignore`를 보강한다. 병합 전후 CPU 회귀를 대조한다.
- rev3.1 O-14: J-C에 OFF 진단 boot를 포함한다. O-15 (b): J-B를 J-B1(H 대비)·J-B2(T4)로 나눈다.
- telemetry 비대칭 감사(`../engine_gate_prep/AUDIT_telemetry_asymmetry_past_results_2026-10-21.md`) 후속: CPU 아카이브 확인 1건만 한다(P1-opint 엔진의 trace-path 조기 return 분기). J1 보정 GPU Job은 하지 않는다.

## 결정 24 (2026-10-21) — H("왜 hybrid인가") 근거를 같은 골격 조성 스윕으로 (권고안 채택, 결정 14의 H 부분 개정)
- 근거 논증: 구성요소(attention 대 SSM) 차이만으로는 서빙 결과가 달라진다는 화살표를 보일 수 없고(층별 분해 측정은 유효 0건·프로파일러 불가·격리 커널은 게이트 1), Llama 대비는 크기·층 수·RoPE·토크나이저 교락으로 귀속이 불가하다.
- **H의 주 근거 = 같은 골격 조성 스윕**: Nemotron-Nano-9B-v2 골격(hidden·MLP·층 수 고정)에서 `hybrid_override_pattern`으로 Mamba 층 일부를 attention으로 바꾼 변형을 `--load-format dummy`(무작위 가중치)로 띄워, attention 비율 3–4점에서 비용 항(T1 decode ctx 기울기, T4 요청당 메모리·동시 수 중심)을 잰다. 바뀌는 변수 = 층 조성(공시: 층당 파라미터가 달라 가중치 바이트·KV 풀이 함께 변한다).
- **Llama-3.1-8B는 실재 Transformer 참조점 1–2셀로 축소**한다(J-B1 축소).
- 선행: 변형·dummy 가중치의 **부팅 프로브(비측정)**. dummy 대 실가중치 비용 동일성(원본 골격)을 측정 Job 안에서 gate로 확인한다.
- 정확성 검사는 원본 모델에서만 한다(dummy 출력은 무의미). goodput은 원본에서만 잰다.

## 결정 25 (2026-10-21) — 조성 스윕의 빈칸 두 개 추가 (사용자 지시 "추가 해야겠는데")
- **동거 간섭(T3)의 조성 의존**을 J-H에 추가한다(PREREG rev3.3 §4′: PD-mux D {24, 44} boot에 T3 4셀, J-H1·J-H2 분할). 기술값이며 변형 간 차이를 간섭 크기로 쓰지 않는다(ρ 합성값 규칙 승계).
- **goodput·G의 조성 의존**을 G0 범위에 넣는다: `ignore_eos`로 출력 길이를 고정하면 goodput은 타이밍만의 지표라 dummy 변형에서도 잴 수 있다. 실가중치 Nano에서 기록한 출력 길이 분포를 변형에 재생한다. 구체 설계(변형 수·arm별 튜닝 범위·변형별 SLO 기준)는 G0 PREREG에서 정한다. 결정 24의 "goodput은 원본에서만"을 이 범위에서 개정한다.

## 결정 26 (2026-10-21) — rev3.3 열린 결정·프로브 실행 (권고안 채택)
- O-16: J-H에 D 축(D 24·44의 T1 2셀) 포함. O-17: 순수 SSM 극한 a0 제외.
- 조성 프로브에 TEL ON 확인 단계(변형에서 phase event 방출 여부, J-H2 T3의 σ 전제)를 추가한 뒤 Workspace `pdmux-probe`에서 GPU 실행을 승인한다(상한 90분 + TEL 단계, 끝나면 Workspace 정지). 수치는 인용 불가 프로브 값.

## 결정 27 (2026-10-22) — 해석적 비용 수식을 정본으로 고정 (사용자 지시)
- Attention / Mamba2(SSD) / MLP 층의 FLOPs·상주 메모리·decode 트래픽 닫힌 식(Marconi 부록 A 형식을 Nemotron-H 구조로 일반화)을 **정본에 항상 반영하고 고정적으로 쓴다.** 논문에 들어갈 수식이다.
- 정본 문서 `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1), 계산 단일 출처 `workspace/engine-port/scripts/costmodel/analytic_terms.py`, 시험 `workspace/engine-port/tests/test_analytic_terms.py`. 식 변경은 코드·문서 동시, 버전 증가, 이력 기록.
- 비용 모델은 이 수식을 사전(prior)으로 두고 측정(rev3.3 T1–T4, J-H)이 효율 η·PD-mux 항·간섭·할당 구조를 덧붙이는 방식으로 정립한다.

## 결정 28 (2026-10-22) — 하네스 1단계 모호점 처리 (권고안 채택)
- **A-1**: T1 창은 ramp 인식 방식으로 바꾼다 — 출력을 넉넉히 주고(`ignore_eos`), 엔진 `decode_forward`의 bs가 등록 bs에 처음 도달한 시점부터 300 step을 창으로 쓰며, 창 뒤 요청을 취소한다. 셀 문맥은 명목 c가 아니라 **창 안 실제 배치 평균 ctx**로 기록하고 그 값으로 적합한다(정본 수식상 decode 비용 ∝ Σctx). 창 안 prefill 동거 iteration은 제외·계수한다.
- **A-2**: decode가 비면 prefill은 (108,0)에서 돈다(코드 사실) → T2는 P=108(vanilla·108 boot)에서만 잰다. prefill SM 축(P = 108−D)은 **T2′(bs=1 decode 동거 prefill)를 주 측정**으로 하고 L 범위를 T2와 같게 넓힌다(estimand에 decode 1개 동거 포함을 공시).
- **A-6**: T2-ext(96k·124k) 삭제(ctx 73,728에서 불가, int64 정확성은 G-β로 확인). **A-11**: J-H T4의 거절 사유 `NOT_COLLECTED` 수용(P2는 동시 수만 필요). **A-3…A-23**: 하네스 해석을 PREREG rev3.4에 명시하고 하네스층 감사로 검토.
- 셀 시간 증가에 따라 Job 시간·GPU-h를 다시 계산한다.

## 결정 29 (2026-10-22) — rev3.4 열린 결정 (권고안 채택)
- O-18: shape C 보간 셀 손실은 Workspace smoke에서 실제 ramp를 재고 봉인 전에 ramp 상한을 실측 기반으로 재설정해 복구를 시도한다(b). 복구되지 않는 셀은 외삽 공시.
- O-19: J-C(TP2)는 2단계에서 재설계(ctx·96k 점·T1 실행 가능성).
- [제안] 상수(제외 상한 30, 합류 여유 0.25 s, σ=θ=1, vanilla ramp 64, 취소 319토큰, T2′ 확인 구간)는 하네스 감사·smoke 후 봉인 때 확인한다.
- 진행 순서: 2단계 구현 → Job entry → 하네스층 감사 → Workspace smoke → ramp 상한 재설정 → 봉인 → 제출 승인.

## 결정 30 (2026-10-22) — 장문 측정점 복구와 측정점 삭제 원칙 (권고안 채택)
- **복구**: N=1 T2-ext(P = 108, L ∈ {98,304, 126,976})를 J-A1 맨 끝 vanilla boot(context 131,072)로, TP2 장문 점(96k, 가능하면 124k)을 J-C 맨 끝 boot(context 131,072)로 둔다. 실패는 `SERVER_DEATH` 등 라벨로 결과에 남긴다. 근거: 96k 점의 목적 중 ① 경계 이동 시험은 G-β로 대체되나 ② 장문 비용 측정점(γ 이차항 검증 구간, hybrid 장문 동기, TP2 장문 서빙 실증)은 대체되지 않는다. 결정 28의 A-6 삭제를 이 범위에서 개정한다.
- **원칙(고정)**: 측정점은 물리적으로 불가능하거나 더 강한 증거로 대체될 때만 뺀다. 설정 제약(context, mem, 출력 길이 등) 때문이면 별도 boot나 보정으로 살린다. 빼는 경우 이유와 잃는 범위를 PREREG에 공시한다.
- rev3.5 초안의 나머지 권고(O-21 σ′ 분모 = 같은 T3 boot 기준 span, O-20 J-H2 공통 비교 축소 공시, J-C 앵커 (32,32k) 불가 공시, F4 stride = 배너 유도값 + G-β, O-22 J-B2 하드캡 smoke 후 재계산)를 함께 채택한다.

## 결정 31 (2026-10-26) — 비용 모델→goodput 실험 구체화 (권고안 채택, 사용자 "진행")
- 근거: `reports/hymux_review/EXPERIMENT_CROSSWALK_2026-10-26.md` §4(HyMux 원고 v4·검증 계획 문서 검수에서 도출, 사슬 ①–⑤).
- **A(GPU 0)**: 비용 항 PREREG에 판정 4개를 봉인 전 추가한다 → rev3.6 §1-9. A-1 BW(D) = W/a\_D, A-2 시간 knee b\_D/c\_D 대 2s/k, A-3 T2′ θ′의 P 선형성 라벨, A-4 조성 간 회귀(층 종류 차 t\_A − t\_M). 하네스·Job·GPU-h 불변(분석 코드는 fetch 전 커밋, §8-6).
- **C-1 V(oracle 격차)·C-4 동적 여지(static arm만)를 다음 PREREG로 우선** 작성한다. 설계 초안 `cap_char/PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md`. V의 구조는 항 Job fetch **전에** 봉인하는 것을 권고(결정 17의 "데이터 뒤 구조 선택" 위험을 제거).
- **HyMux 구현은 C-4가 `DYNAMIC_HEADROOM_PRESENT`일 때만**. Falcon-H1-34B TP2/4·replica·Bullet/DuetServe 재구현·vLLM은 보류.
- 열린 채로 둔 것: B-1(층별 span 시간 boot), B-2(J-H2 D16 추가) — smoke 뒤 사용자 결정.

## 결정 32 (2026-10-26) — V-O·H-D CI 추정량 O-S1 (감사 r2 권고안 채택, 사용자 "진행")
- 근거: `cap_char/AUDIT_rev36_planV_r2_2026-10-26.md` §사용자 결정(seed 4 percentile bootstrap 피복률 0.806, fold 단위 0.493 → 정본 §3 항목56(A) 금지 경우).
- **V-O 축 N = (c)**: `R(N2) − R(D̂) = ln(λ*(D̂)/λ*(N2))`로 oracle 항이 상쇄되므로 교차적합 없이 전체 seed(4)의 seed 수준 짝차에 t(3) 95% CI.
- **H-D = (a)**: seed 8, 교차적합 fold 대신 seed 수준 짝차(선택 seed 4 / 평가 seed 4, 역할 교대)에 t(n−1) 95% CI. 비용 약 2배 — 축소안(O-H4)은 열린 결정.
- V-O 축 O는 CI를 쓰지 않으므로(문턱 라벨) 변경 없음. PLAN_V rev3에 반영.
