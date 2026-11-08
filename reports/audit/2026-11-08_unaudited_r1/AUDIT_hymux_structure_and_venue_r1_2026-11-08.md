# 결과 감사 r1: HyMux 구조 검증 보고서와 학회 검토 (claims-auditor, 2026-11-08, 읽기 전용)

**회차.** 두 대상 모두 **r1**이다. 이전 감사를 저장소 전역에서 찾았지만 없었다. `reports/audit/`, `results/*/VERDICT*`, `tracks/main.md`의 "감사 필요(미반영)" ①②가 이 두 건이다. 따라서 대상당 1회가 남는다. 이후 재감사는 이 판정 뒤의 diff만 본다.

**한 줄 요약**
- 문서 1(STRUCTURE): 코드 사실은 거의 전부 맞다. 인용 줄 번호 대조에서 불일치 1건, 누락 1건이 나왔다. 줄 번호 기준점도 낡았다. 문서 수정 후 쓸 수 있다(**GO-with-edits**).
- 문서 2(VENUE_C): 정정 후보 2건 가운데 1건(Bullet venue 미확인)은 **로컬 1차 자료로 반증된다.** 정본 쪽에 놓친 정정 1건(Drift 중복 등재)도 있다(**GO-with-edits**). 정정 후보 ①은 반영하면 안 된다.

---

## 문서 1. `reports/hymux_review/STRUCTURE_batching_admission_2026-10-27.md` (+ `structure_derived_2026-10-27.py`, 커밋 `7a7a2c08`)

### 0. 내가 직접 확인한 것

**트리 동일성**
- 로컬 `~/Experiments/KISTI/sglang_engine_dev`는 v0.5.10 `1519acf`이고, 수정 파일 14개는 문서 서술과 같다.
- 128k 프로브 serving manifest(`results/cap_char/probe_128k/longctx_probe_20261020T034615Z/meta/runtime_source_manifest.sha256`) 27항목을 로컬 트리에 `sha256sum -c` 하니 **27/27 OK**였다.
- 커밋 `bb1d936d`와 `7a7a2c08` 시점의 미러 `multiplexing_mixin.py`는 로컬 트리와 바이트가 같다.
- **현재 HEAD 미러는 로컬 트리와 다르다**(`cmp` 결과 686행부터 다름). 원인은 2026-10-28 이후 커밋 `96ae3afa`(diag 계측·span rids)와 `3858f129`다. 그래서 "줄 번호는 HEAD 미러 기준(로컬 트리와 동일)"이라는 문장은 **오늘 시점에는 거짓**이다. 줄 번호는 `bb1d936d`(= 2026-10-20 serving tree) 기준으로 고정해야 한다.

**줄 인용 전수 대조 (로컬 트리 = `bb1d936d` 미러 기준)**
- `multiplexing_mixin.py`: :1219–1221, :1227, :1248–1249, :1404, :1416–1424, :1443, :1468, :1494, :1518, :1531–1534, :1341–1345, :1358–1362, :388–391, :1163–1168, :1179–1196, :1199–1211, :484, :626–682, :2045–2071
- `controller.py`: :88–103, :151–166, :281–292, :311–328, :410–457
- `profile.py`: :303–360
- `pdmux_context.py`: :20
- `scheduler.py`: :681–684, :902–904, :2309–2313, :2316–2336, :2366, :2446, :2490, :2572, :2592
- `schedule_policy.py`: :429, :457–476, :764–773, :767–768, :791–795
- `common.py`: :22, :297–325, :318–324, :351
- `memory_pool.py`: :530–565, :543–545
- `schedule_batch.py`: :1395–1397
- `server_args.py`: :363
- `model_runner_kv_cache_mixin.py`: :177–192, :218–243, :390–392, :857–876
- `nemotron_h.py`: :859–891
- `tp_worker.py`: :537–541
- `llama.py`: :544
- upstream v0.5.10 `multiplexing_mixin.py`: :82, :86, :166, :194, :215

위 항목은 전부 문서 서술과 일치했다. 불일치·누락은 아래 표 F3·F6에 적었다.

**derived 재계산**
- `python3 -I structure_derived_2026-10-27.py`를 돌린 값이 문서와 같다: D1 8,763.1 / L\* 8,761.5, D2 230.19 / 49, D3 33,260.9 / 33,078.4, D4 1,251.2 / 1,251.6 / 1,298.0, c_mem 5점 일치.
- analytic 대조는 `analytic_terms.py`(analytic-v1)를 import해서 했다. 손계산이 아니다(결정 27 준수).
- 단 Llama 형태는 `Arch(...)`를 스크립트 안에서 직접 정의했다. KV 바이트에 영향을 주는 필드(n_A 32, n_kv 8, d_h 128)는 공개 구성과 맞는다.

**원고 위치**
- 원고 v4 본문은 claude.ai artifact라 로컬에 없다.
- 대신 다른 세션 scratchpad의 Revision Letter 발췌(`/tmp/claude-1000/-home-wonho-Experiments-KISTI/c3714a91-2d4a-48cf-9bc2-fef6e1988cbb/scratchpad/hymux_letter_excerpts.md`)와 대조했다. R1–R4·R7–R11의 문단 번호와 요지는 발췌와 맞는다.
- **R5·R6은 발췌에 없다.** 출처가 "지시서 §3"뿐이다.

### 1. 주장별 판정

| # | 판정 | 근거·단서 |
|---|---|---|
| §1.1 트리 동일성 | CONFIRMED(2026-10-20 serving tree 기준) | 27/27 OK. 단 manifest는 패치 파일 27개만 덮는다. 미수정 파일(`schedule_policy`·`common`·`schedule_batch`·`tp_worker`)이 serving tree에서도 v0.5.10 원본이라는 것은 "같은 v0.5.10 체크아웃 + sync" 절차에서 나온 추론이다. 바이트로 확인한 것이 아니다. 오늘 HEAD 미러와는 다르다(위 0절). |
| F1 | CONFIRMED | 스트림 그룹 구성(`pdmux_context.py` 0 = (108,0) 일반, 1…k = green, 마지막 = (0,108))까지 확인했다. |
| F2 | CONFIRMED | :1219–1221, upstream :82. |
| F3 | CONFIRMED, **인용 1건 오류** | `longctx_probe.sh:110`은 **llama arm**이다. nano fused arm은 :108이다. "fused arm(:110)"을 "llama arm(:110, fused_sym 노브), nano_fused(:108)"로 고쳐야 한다. chunk −1 → None(:903–904) 서술은 맞다. |
| F4 | CONFIRMED | `sm_count`를 쓰지 않는 것 확인, 65536 확인, `max(1, …)` 단서 맞음. |
| F5 | CONFIRMED | |
| F6 | CONFIRMED(정정 서술 포함), **누락 1건** | 매 반복 `decode_stream.synchronize()`(:1494)와 prefill `query()`(:1518)를 확인했다. drain 두 곳(:1343–1344, :1361–1362)도 맞다. 다만 `adjust_stream_group`을 켜는 지점은 입장(:1292)·merge(:1541) 외에 **decode가 빈 경우(:1304)**도 있다. "입장과 merge 뒤에 켜진다"는 불완전하다. 또 prefill `run_batch` 내부의 암묵 host 동기화(예: split_index 0의 `ForwardBatch.init_new`, 마지막 span sample)는 정적으로 조사하지 않았다. "decode가 prefill GPU 작업을 기다리지 않는다"는 루프 구조 수준의 서술로 한정해야 한다(DIAG 문서의 WAIT 약 90% 관측과는 정합). |
| F7 | CONFIRMED(범위 단서 정확) | `pdmux_r2.yml` 6개 / `pdmux_agnostic_manual.yml` 4개 확인. |
| F8 | CONFIRMED | |
| F9 | CONFIRMED | `PROJECT_STATUS.md:27`은 7a7a2c08 시점 기준이다. 지금은 줄이 밀렸다. |
| F10 | CONFIRMED | grep 0건 주장은 재현하지 않았다. 반증도 못 찾았다. |
| F11 | CONFIRMED | |
| F12 | CONFIRMED(예외 포함) | PrefillAdder의 mamba 분기는 `is_hybrid_ssm_cache` 플래그 하나다. 쓰이는 곳은 `rem_total_tokens`·`cur_rem_tokens` 두 property이고 둘 다 evictable 계산이다. `--enable-prefill-delayer` 예외는 :2316–2336·`server_args.py:363`에서 확인했다. |
| F13 | CONFIRMED | 3분기 :218–243, clamp :857–876 확인. 단 `longctx_probe.sh`는 `--max-mamba-cache-size`를 명시하므로 분기 ①(명시값)이다. `mg_boot.py:96–97`은 docstring 줄이다(정책 서술로는 유효). |
| C1–C5 처리 | PLAUSIBLE(문서 자신이 "미감사·채택 아님") | 처리 방식(수정해서 올림, 일부만 올림)은 정본 금지 서술과 충돌하지 않는다. C3에서 "hybrid 고유" 서술을 뺀 것은 올바르다. |
| C5 입력 "출력 평균 203 [measured]" | **라벨 정정 필요** | `ARXSUM_STATS`의 203은 `kept out(abstract)`, 즉 **참조 초록 길이**다. 모델 생성 출력의 측정이 아니다. "출력 길이 대용(참조 초록)"으로 적어야 한다. |
| C6 / D4 "8.8k는 손익분기가 아니라 s/k" | CONFIRMED | 손익분기의 정의는 s + kL = k_T·L, 즉 L = s/(k_T − k)다. 8.8k = s/k = L\*는 다른 양이다. |
| D4 "손익분기 ≈1,251 토큰 [derived]" | **PLAUSIBLE(조건부)** — 값은 맞고, 대조 모델 선택과 가정 공시가 부족하다 | ① analytic_terms 기반 계산이 맞다(boot 1,251.2, analytic 1,251.6). ② k_T = Llama-3.1-8B KV 131,095 B/토큰이다. 같은 Job·같은 mem 0.6에서 배너 GiB(소수 둘째 자리)로 유도했고, analytic 131,072와 +0.018% 차이다. ③ **대조 모델 선택**: 결정 24는 hybrid 귀속 근거를 "같은 골격 조성 스윕 a4…a31, Llama는 참조점 1–2셀"로 정했다. 그러면 같은 골격 대조는 **a31(1,298)**이고 Llama(1,251)는 참조점이다. 문서는 1,251을 정정값의 대표로 올렸고 a31은 부수값으로 적었다. 두 값을 나란히 놓아야 한다. ④ 값은 대조의 KV 형태에 종속된다. GQA n_kv 8·d_h 128·bf16 대조에서만 약 1.25–1.30k다. MHA 대조였다면 자릿수가 바뀐다. ⑤ D4는 D2와 같은 **가변 풀(fungible pool) 가정**을 물려받는다. SGLang은 부팅 때 풀을 고정 분할하므로, 이 값은 실현 동시성의 교차 길이가 아니다. "가정 공시" 단락은 최대 점유 가정만 적었고 이 가정은 빠졌다. |
| D1–D3, c_mem 표 | CONFIRMED(derived) | "c_mem은 식의 출력이며 측정이 아니다"라는 정정은 올바르다. D2의 "49 슬롯"은 slot = (conv+ssm)/(cap+1)이라는 프로브 가정에서 나온 값이지만, analytic s와 −0.012% 차이라 정합한다. |
| §2 R1–R3 | CONFIRMED(baseline 서술 = 구현됨) | |
| R4 미구현 | CONFIRMED | 시간·SM·층 종류 입력이 없다. `PDMUX_SLO_SPAN_TYPE`은 잔재다. |
| R5 미구현 | **NOT-YET-SUPPORTED(출처)** | 코드 사실(:1443은 인라인 대입이고 호출이 없음)은 CONFIRMED다. 그러나 대조 대상 "원고 서술"의 출처가 지시서 §3뿐이고, 응답서 발췌에도 없다. "원고 미구현" 집계에 넣으려면 원고 본문 대조가 필요하다. |
| R6 미구현 | **NOT-YET-SUPPORTED(출처)**, 코드 쪽 CONFIRMED | latch 서술 (i)–(iv)는 `controller.py:311–328`·mixin :2045–2071과 일치한다(generic도 `stabilize`를 거쳐 latch가 걸리는 것 확인). 출처 문제는 R5와 같다. |
| R7 부분 | CONFIRMED | "P = 108 − D는 설정 항등식"이 맞다(`pdmux_r2.yml` 행 합 108, P는 행에서 읽음). D_floor 닫힌 식은 코드에 없다. 사소한 서술 문제: 프로파일 보간은 D마다 (b, c) 위의 쌍선형 보간이다. D 방향으로는 단조 투영(더 큰 D의 값을 하한으로 씀)이다. "{16,24,34,44} 가운데 보간"이라는 표현은 부정확하다. |
| R8 대응 코드 있음 | CONFIRMED | hybrid도 공용 `_live_underprediction` 즉시 상향을 쓴다(`HybridInformedPolicy`가 `controller.stabilize`를 호출). |
| R9 부분 | CONFIRMED(범위 단서 정확) | |
| R10 해당 없음 | CONFIRMED | D4 조건이 그대로 붙는다. |
| R11 미구현 | **OVERBROAD(분류)** | 응답서가 이미 "측정 아닌 개념도"라고 공시했다. 이것을 "미구현 서술"로 세는 것은 범주가 섞인 것이다. "해당 없음(개념도), HyMux 런타임 부재"가 공정하다. |
| §2 집계 "미구현 4·부분 2" | **OVERBROAD** | 원고 본문 대조로 확정되는 미구현은 R4 하나뿐이다. R5·R6은 출처가 지시서이고, R11은 개념도다. |
| §4 Q1 "요청 ID·요청별 대기열 진입 시각은 기록 안 함" | **낡음(2026-10-28 이후)** | `96ae3afa`가 `PDMUX_PHASE_DIAG=1`(opt-in)에서 `prefill_span_start.rids`를 추가했다. "2026-10-27 트리 기준"을 명시하거나 갱신해야 한다. |

### 2. Confound 대조

- **#1·#4·#5·#6·#8·#12·#13**: 해당 없음. 성능 수치가 없다.
- **#2 미조율을 정책 탓으로(변형)**: 이 문서가 피해야 할 함정이다. 원고의 HyMux "미구현"이 곧 "HyMux 설계가 틀렸다"는 뜻이 아니다. 반대로 SGLang 공개 루프의 성질(F2·F4·F6)이 MuxWise 논문 시스템의 성질인 것도 아니다. 문서는 C2에서 이것을 지켰다.
- **#11 기판 혼합(코드 변형)**: 줄 번호가 기준 트리(2026-10-20 serving = `bb1d936d`)에 묶여 있는데, 문서는 "HEAD 미러"라고 적었다. HEAD는 이미 움직였다. 고정하지 않으면 다음 인용자가 다른 줄을 가리키게 된다.
- **#10 변수 동시 변경(유사)**: D4의 기준 모델 선택(Llama 대 a31)이 정정값을 3.7% 움직인다. 반올림 "≈1.25k" 안에서는 판정이 안 바뀌지만, "원고 M3와 맞는다"는 Llama를 골랐을 때만 성립한다.

### 3. 허용 문안

- "PD-mux 루프의 좌석 1개(F2)·고정 token×층 예산 span(F4)·매 반복 decode stream 동기화(F6)는 SGLang v0.5.10 원본 `multiplexing_mixin.py`에 이미 있으며, 이 프로젝트의 선택이 아니다(2026-10-20 serving tree = 커밋 `bb1d936d` 미러 기준 줄 번호)."
- "매 반복 동기화되는 것은 decode stream뿐이고, prefill 완료는 `query()`로 확인만 한다. 두 스트림을 함께 drain하는 곳은 R2/SLO 전환과 `adjust_stream_group` 경로 두 곳이며, 후자는 입장·merge·decode 비움 때 켜진다. 이것은 루프 구조에 대한 서술이며 run_batch 내부의 암묵 동기화는 조사하지 않았다."
- "SGLang admission 경로에는 mamba 슬롯 검사가 없다. 슬롯은 부팅 때 `max_running_requests` clamp로만 admission에 닿고, 고갈되면 프로세스가 죽는다(`--enable-prefill-delayer` opt-in은 예외, 프로젝트 미사용)."
- "요청당 메모리가 같아지는 길이 s/(k_T − k)는 Nano-9B-v2 기준으로 Llama-3.1-8B 참조점 1,251 토큰, 같은 골격 all-attention 변형 a31 1,298 토큰이다[derived, 가변 풀·최대 점유 가정, GQA n_kv 8·d_h 128 대조 한정]. 채팅 값 8.8k는 s/k(= L\*)로 다른 양이다."
- "코드에는 HyMux span scheduler(θ_stall)가 없다(R4). partitioner의 P = 108 − D는 설정 표의 항등식이며, D_floor 닫힌 식은 구현되지 않았다. hybrid 정책은 측정 프로파일 조회로 최소 D를 고른다(R7)."

### 4. 인용 금지 문안

- "1,251 토큰이 hybrid와 Transformer의 손익분기다" (대조 모델과 가변 풀 가정 없이).
- "원고 M3의 ≈1.25k가 검증됐다" (Llama 참조점 하나와 맞을 뿐이고, a31 기준은 1,298이다).
- "원고의 HyMux 서술 6건이 미구현이다" 또는 "원고에 미구현 서술 4건" (원고 본문으로 확정된 것은 R4 하나다).
- "lockstep은 GPU 수준 결합이 아니다" (코드 구조상 필연이 아니라는 것까지만 말할 수 있다. 이 문서는 측정하지 않았다).
- "PD-mux 계열 공통 비용" (SGLang 하나만 읽었다).
- "줄 번호는 HEAD 미러 기준" (2026-10-28 이후 거짓).
- "arXiv-Sum 출력 평균 203 [measured]" (참조 초록 길이다).
- "cap 48이 KV보다 먼저 구속한다"를 범위 없이 쓰는 것 (최대 점유 가정·이 부팅·L+O ≲ 33k라는 조건을 붙여야 한다).

### 5. 해소 방법 (전부 문서 편집이고 GPU는 필요 없다)

1. 머리말과 §1.2 기준 문장을 "줄 번호 = 커밋 `bb1d936d` 미러 = 2026-10-20 serving tree(manifest 27/27)"로 바꾼다. HEAD가 이후 움직였다는 사실을 한 줄 덧붙인다.
2. F3: :110을 llama arm으로 고치고, nano_fused는 :108로 적는다.
3. F6: `adjust_stream_group` 트리거 목록에 :1304(decode 비움)를 넣고, 암묵 동기화를 조사하지 않았다는 단서를 붙인다.
4. D4·R10: Llama 1,251과 a31 1,298을 나란히 둔다. 결정 24에 따라 a31을 같은 골격 대조로, Llama를 참조점으로 명시한다. 가변 풀 가정과 GQA 대조 한정을 공시한다.
5. C5·스크립트 주석: "출력 평균 203"을 참조 초록 길이로 다시 라벨링한다.
6. §2: R5·R6 출처 열에 "지시서 §3 [cited], 원고 v4·응답서 미대조"를 적는다. R11은 "해당 없음(개념도)"으로 옮긴다. 집계 문장을 "원고 대조로 확정된 미구현 R4, 출처 미대조 R5·R6, 부분 R7·R9"로 고친다.
7. R7: 보간 서술을 "D별 (b,c) 쌍선형 보간 + D 방향 단조 투영"으로 고친다.
8. Q1: "2026-10-27 트리 기준. 2026-10-28 `96ae3afa` 이후 `PDMUX_PHASE_DIAG=1`에서 rids 기록"을 덧붙인다.
9. (선택, 확정용) R5·R6을 원고 v4 본문 해당 절과 직접 대조한다. 사용자가 artifact를 열어 문장을 옮기면 된다.

**종합: GO-with-edits.** 성능 판정은 0건이고 정본과 모순되는 곳도 없다. 위 1–8을 반영하면 promote 대상이 된다. 인용 금지 문장은 8개지만 대부분 편집으로 사라진다. 편집 뒤에 남는 caveat는 3개(D4 대조 한정·가변 풀, 루프 구조 한정, SGLang 단일 구현 한정)로, 5개 상한 안이다.

---

## 문서 2. `reports/hymux_review/VENUE_C_positioning_2026-10-27.md`

### 0. 직접 확인한 것

**`~/Experiments/KISTI/Papers/Bullet.pdf`** (camera-ready, 2026-09-22부터 로컬에 있음)
- 1쪽: "In Proceedings of the 31st ACM International Conference on Architectural … (**ASPLOS '26**), March 22–26, 2026, Pittsburgh" → Bullet의 게재처는 **ASPLOS'26으로 확인된다.**
- 같은 PDF 관련연구에 "Drift [25] explores lock-step execution…"이 있고, [25]는 Cui et al., "Optimizing SLO-oriented LLM Serving with PD-Multiplexing", **arXiv:2504.14489**다.

**`Papers/Muxwise.pdf`**
- "Towards High-Goodput LLM Serving with Prefill-decode Multiplexing … (**ASPLOS '26**)". 즉 Drift는 MuxWise(2504.14489)의 옛 제목이다.

**DuetServe**
- `reports/SURVEY_slo_attainment_published_2026-11-02.md:91`에 "PDF 각주상 ICML 2026"이 있다. 이 문서는 별개 세션의 1차 PDF 판독이지만 역시 미감사다.
- PMLR v306 URL은 이 세션에서 열 수 없었다(웹 도구 없음).

**정본 대조**
- CONSENSUS §6 E(미검증 핵심 가설)·G, §9(C 감사 r1 허용·금지 문안), COSTMODEL CM-1("hybrid 고유 항 미입증")·CM-5("cap≡슬롯은 할당기 종속"), 결정 24·38·43, `tracks/main.md:21`(V-O·H-D 동결 설계)과 대조했다.

### 1. 주장별 판정

| 주장 | 판정 | 근거 |
|---|---|---|
| **정정 후보 ① "venue_positioning: Bullet venue 미확인"** | **REFUTED** | 로컬 camera-ready가 ASPLOS'26을 명시한다. 이 정정을 반영하면 **맞는 사실을 틀린 쪽으로 강등**하게 된다. 반대 방향 정정, 즉 venue_positioning의 "ASPLOS'26 추정/검증 필요"(:120, :473) 단서를 지우고 출처로 `Papers/Bullet.pdf`를 적는 것이 맞다. 같은 문서 표 14행의 "**미확인**"도 틀렸다. |
| **정정 후보 ② "DuetServe ICML'26 확인"** | **PLAUSIBLE(조건부)** | 두 독립 판독(venue 문서의 PMLR URL, SURVEY의 PDF 각주)이 일치한다. 그러나 둘 다 미감사이고 나는 직접 열지 못했다. promote 전에 PMLR v306 페이지나 PDF 각주를 열어 한 줄로 확인해야 한다. |
| **놓친 정정(과소 주장)**: venue_positioning은 Drift를 MuxWise와 **별개 선행으로 등재**한다(:410, :423, :509, :528, :559) | 정정 필요(문서 2가 표 15행 "MuxWise/Drift 2504.14489"로 올바르게 묶었지만 정정 후보로 올리지 않음) | Bullet [25]가 Drift = arXiv 2504.14489(Cui et al.) = MuxWise임을 보인다. 정본 쪽 선행 목록에 이중 집계가 있다. |
| "현재 A·B 측정값 0(V-O·H-D 동결, 결정 38)" | CONFIRMED(2026-10-27 시점, 오늘도 유효) | `tracks/main.md:21` V-O·H-D는 동결 설계다. 결정 43 이후 돈 CAP1-R·CAP2·TUNE·Phase A/B는 V-O·H-D 측정이 아니다. 날짜를 박아 두어야 한다. |
| "C의 형태는 DuetServe가 transformer에서 이미 수행 → C 신규성은 hybrid 귀속에만 달림" | PLAUSIBLE(문헌 판단) | C 감사 r1 허용 문안("Bullet/MuxWise형 최소-SM의 hybrid 확장은 서빙 측정 0건")과 정합한다. "형태 동일"은 venue-strategist의 판단이며 재현은 0건이다(lockstep 진단 r1 :50이 같은 단서를 달았다). |
| "CM-1 hybrid 고유 항은 미입증" | CONFIRMED | COSTMODEL:18과 일치한다. |
| "CM-5 cap≡슬롯은 할당기 종속" | CONFIRMED | COSTMODEL CM-5와 :92(결정 8′)와 일치한다. |
| "웹 검색상 hybrid SM 분할 PD 스케줄러는 부재" | NOT-YET-SUPPORTED | 부재는 검색 완전성을 보증할 수 없다. 문서가 "빈자리 ≠ 자명하지 않음"이라고 스스로 단서를 단 것은 좋다. "확인한 범위에서 찾지 못했다"로 써야 한다. |
| 학회 fit 표(MLSys "강" 등) | PLAUSIBLE(의견, claim 아님) | 조건부 칸("C: hybrid 귀속 승리")은 Claim E가 성립할 때의 가정이다. Claim E는 §6에서 미검증이다. 표를 성과 예측으로 읽으면 안 된다. |
| "순수 negative(Elastic KV)는 미게재 ⇒ constructive 장치가 최소" | **OVERBROAD** | 미게재 preprint 사례 하나에서 일반 규칙을 끌어냈다. POLCA·Vidur 선례만으로 "constructive 장치 동반"이 유리하다고 말하는 것까지는 의견으로 허용된다. |
| 마감표(MLSys 2027 10-30 등) | NOT-YET-SUPPORTED, 일부는 시간상 지남 | 문서 스스로 "추정·미검증"을 달았다. 2026-11-08 현재 MLSys 2027 마감은 지났다. 투고 결정에 쓰려면 CFP 원문으로 다시 확인해야 한다. |
| "C가 negative일 때 가치 … HE0 범위를 예측형까지 확장" | PLAUSIBLE(조건부) | 문서가 조건(같은 하네스의 순수 attention 재현)과 lockstep 아티팩트 위험(결정 F4)을 함께 적었다. 정본의 2026-10-29 루프 범위 정정(SGLang 공개 PD-mux 루프 한정)을 그대로 이어받아야 한다. |
| 권고 "C 게이트: H-D 여지 > 3% 또는 V-O 분기 시 착수" | PLAUSIBLE(결정 권고, 판정 아님) | C 감사 r1 조건 7("H-D ABSENT는 C 기각 아님", PLAN_V:88)이 빠졌다. 게이트를 넘지 못하는 것은 자원 배분상 보류이지 반증이 아니다. 이 문장을 덧붙여야 한다. |
| 금지 목록("C가 이긴다", "hybrid가 예측기를 틀리게 만든다", MuxWise 2.20× 배율 비교) | CONFIRMED | §9 C 감사 금지 문장과 같은 방향이다. |
| 기타 venue 표기(DistServe·Sarathi-Serve OSDI'24, Splitwise ISCA'24, Vidur MLSys'24, Marconi MLSys'25, POLCA ASPLOS'24) | 반증 실패(직접 확인 안 함) | promote에 쓰지 않는다면 표기 그대로 두어도 된다. |

### 2. Confound 대조

- 측정 confound(#1–#13)는 대부분 해당이 없다. 측정이 0건이다.
- **#11 기판 혼합(문헌 변형)**: 확인되지 않은 2차 판독(웹 검색)이 로컬 1차 자료(camera-ready)를 덮어썼다. 그래서 정정 ①이 반대 방향으로 나왔다.
- **#2(문헌 변형)**: "MuxWise 공개 엔진의 문턱표"를 MuxWise 논문 시스템과 동일시하면 안 된다. 문서는 "공개 코드는 문턱표"로 구분했다. 정본(CONSENSUS §6 G, CLAUDE.md "MuxWise 논문 시스템과 동일시 금지")과도 정합한다.

### 3. 허용 문안

- "Bullet(arXiv 2504.19516)과 MuxWise(arXiv 2504.14489, 옛 제목 'Optimizing SLO-oriented LLM Serving with PD-Multiplexing', Bullet이 'Drift'로 인용)는 ASPLOS'26에 게재됐다(로컬 camera-ready 확인)."
- "C(비용 모델 구동 iteration 단위 예측형 SM 배분)의 형태는 Transformer 선행(DuetServe, Bullet/MuxWise형 최소-SM)에 이미 있다는 것이 문헌 판단이며, 이 저장소의 신규성은 hybrid 귀속이 성립할 때만 생긴다. hybrid 확장은 서빙 측정 0건으로 미검증이다(CONSENSUS §6 E)."
- "확인한 범위의 문헌에서 hybrid 모델용 SM 분할 PD 스케줄러는 찾지 못했다. 이것은 빈자리의 존재이지 확장이 자명하지 않다는 증거가 아니다."
- "C 착수 게이트(H-D 여지 > 3% 또는 V-O phase별 oracle 분기)는 자원 배분 규칙이다. 게이트 미통과는 C의 반증이 아니다."

### 4. 인용 금지 문안

- "Bullet의 게재처는 미확인이다" 및 이것을 근거로 venue_positioning을 고치는 모든 문장 (REFUTED).
- "DuetServe는 ICML'26에 확정 게재됐다" (1차 확인 한 줄 전까지는 "ICML'26으로 보고됨(PMLR v306 / PDF 각주, 미감사)").
- "MuxWise, Drift, Bullet 세 선행" 또는 Drift를 MuxWise와 별개로 세는 문장.
- "hybrid SM 분할 PD 스케줄러는 존재하지 않는다".
- "순수 negative 결과는 게재되지 않는다".
- "C가 hybrid 귀속에 성공하면 MLSys에 강하다"를 기대 성과로 쓰는 것 (Claim E 미검증).
- "H-D 여지가 3% 미만이면 C는 기각된다".
- "C가 negative면 HE0가 예측형까지 확장된다"를 조건 없이 쓰는 것 (순수 attention 재현과 SGLang 공개 lockstep 루프 범위가 함께 붙어야 한다).

### 5. 해소 방법

1. 정정 후보 ①을 **철회**하고 반대 방향으로 바꾼다. venue_positioning의 Bullet ASPLOS'26 단서(:120 "추정", :473 "검증 필요")를 지우고 출처로 `Papers/Bullet.pdf` 1쪽을 적는다. 문서 2 표 14행도 "ASPLOS'26(camera-ready)"로 고친다.
2. 정정 후보 ②: promote 전에 PMLR v306 `gao26m` 페이지나 arXiv v2 PDF 각주를 열어 한 줄로 확인한다. 실행자는 venue-strategist다. 웹이 필요하고 GPU는 필요 없다.
3. 새 정정 후보 ③을 추가한다. venue_positioning의 Drift 별도 등재를 "Drift = MuxWise 옛 제목(arXiv 2504.14489, Bullet ref [25])"으로 합친다.
4. 본문의 "부재" 문장, "순수 negative 미게재" 문장, C 게이트 문장을 §3 허용 문안으로 바꾼다. 마감표에는 "2026-10-27 시점·CFP 미확인"을 유지하고, 지난 마감은 지운다.
5. 이 정정들은 venue_positioning(동결 이력, `reports/paper/`)을 바꾸는 일이다. 정본 위계상 promote 경로로만 반영한다.

**종합: GO-with-edits.** 정본 §6·COSTMODEL과 모순되는 곳은 없다. 다만 promote 몫 정정 2건 가운데 1건은 사실이 반대이고, 정본 쪽 오류 1건(Drift 이중 집계)을 놓쳤다. 위 1–3을 반영하지 않으면 promote하지 않는다. 편집 뒤 남는 caveat는 2개(DuetServe venue 1차 확인 대기, 문헌 판단 = 재현 0건)로, 상한 안이다.

---

## 관련 파일 (절대경로)

- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/hymux_review/STRUCTURE_batching_admission_2026-10-27.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/hymux_review/structure_derived_2026-10-27.py`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/hymux_review/VENUE_C_positioning_2026-10-27.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/paper/venue_positioning.md` (정정 대상: :120, :473, Drift :410/:423/:509/:528/:559)
- `/home/wonho/Experiments/KISTI/Papers/Bullet.pdf`, `/home/wonho/Experiments/KISTI/Papers/Muxwise.pdf` (ASPLOS'26 1차 근거, Bullet ref [25] = Drift = 2504.14489)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/SURVEY_slo_attainment_published_2026-11-02.md:91` (DuetServe ICML 2026 각주 판독, 미감사)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/cap_char/probe_128k/longctx_probe_20261020T034615Z/meta/runtime_source_manifest.sha256` (27/27 OK)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/scripts/vessl/longctx_probe.sh:106-112` (F3 arm 줄)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/handoff_review_2026-10-15/arxsum/ARXSUM_STATS_2026-10-15.txt:17-18` (203 = 초록 길이)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/anchor_audit/VERDICT_claimC_online_predictive_2026-10-27.md` (C 게이트 조건 7, 허용·금지 문안 근거)
- 원고 R 대조에 쓴 발췌: `/tmp/claude-1000/-home-wonho-Experiments-KISTI/c3714a91-2d4a-48cf-9bc2-fef6e1988cbb/scratchpad/hymux_letter_excerpts.md` (다른 세션 scratchpad, 휘발성. 원고 v4 본문은 claude.ai artifact라 로컬에 없음)

> 저장: main 세션이 claims-auditor 최종 보고 원문을 그대로 옮김(2026-11-08).
