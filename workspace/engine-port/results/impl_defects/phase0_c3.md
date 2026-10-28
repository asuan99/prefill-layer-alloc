# Phase 0 C3 — PD-mux prefill span 영역(미발화 / 적체 / 박자) 지형도 (2026-10-28)

> 작성: result-analyst, main 트랙. CPU 재집계와 모형 계산만 했다(GPU 0). **새 성능·정책 판정은 0건이다. 정본·tracks는 수정하지 않았다.**
> 라벨: [측정] 원자료에서 직접 센 값 · [유도] 측정값과 정본 식(analytic-v1)을 조합한 값 · [코드] 코드나 설정 파일의 사실 · [가정] 이 문서가 둔 가정(§2 A1–A9) · [인용] 기존 문서를 재계산 없이 옮긴 값.
> 스크립트: `workspace/engine-port/results/impl_defects/c3_regimes.py`(sha256 `99b5813f…`) → `c3_regimes.json`(`107ad226…`).
> 실행 명령(저장소 루트에서): `python3 workspace/engine-port/results/impl_defects/c3_regimes.py --out workspace/engine-port/results/impl_defects/c3_regimes.json`

## 1. 질문

prefill 배치 하나는 PD-mux 루프에서 다음 셋 중 어느 영역에서 실행되는가. 캠페인별로 그 비율은 얼마인가.

- **미발화**: `forward_count ≥ num_hidden_layers`. prefill 전체가 span 1개로 끝나므로 사실상 교차 실행(interleaving)이 없다.
- **적체**: span이 2개 이상이고, 어떤 span의 GPU 시간이 decode step보다 길다. host가 GPU 실행 속도보다 빨리 span을 발행해 prefill stream에 쌓인다.
- **박자 지배**: span이 2개 이상이고, 모든 span의 GPU 시간이 decode step 이하다. decode step 한 번에 span 하나가 실제로 prefill 진행 속도를 정한다.

코드 근거 [코드, `src/multiplex/multiplexing_mixin.py` @3858f129]:
- `forward_count = max(1, split_forward_token_budget // extend_num_tokens)`(`:1416-1424`).
- 루프 한 번에 chunk 하나를 발행하며, 이전 chunk가 끝났는지 확인하지 않는다(`:1410-1414`, `:1477-1480`).
- 매 반복마다 동기화되는 것은 decode stream뿐이다(`:1494`). `wait_prefill_kernel_done`은 마지막 chunk를 발행한 뒤에만 켜진다(`:1517`).
- ★`extend_num_tokens`는 **배치 전체의 합**이다. 프롬프트 하나의 길이가 아니다. 2026-09-22 census(`reports/synthesis_2026-09-22/`)는 요청 단위 입력 길이로 미발화를 예측했다. 이 문서는 배치 단위로 센다.

## 2. 방법

1. **budget**은 실제로 쓴 config에서 확인했다 [코드]. 이 문서가 다루는 캠페인의 config는 전부 `split_forward_token_budget: 65536`이다.
   - `results/slo_sched/pdmux_d*.yml` 9/9
   - `results/p1_opint/pdmux_a100_smoke.yml`, `triage/pdmux_a100_smoke.yml`
   - `n3_capacity/configs/*`
   - `cap_char/costterms_harness/pdmux_costterms_D64_candidate.yml`(정책 하네스 `ph_spec.py:26`이 같은 파일을 쓴다)
   - 엔진 기본값도 65536이다(`sglang_engine_dev/.../pdmux_context.py:20,50`). 다른 budget 값은 lockstep 진단 PREREG 격자의 8192뿐이다.
2. **층 수**는 config.json에서 읽었다 [코드].
   - Zamba2-2.7B 54
   - Nano-9B-v2 56. pattern 길이도 56이고, 스크립트가 assert로 확인한다.
   - Nemotron-H-8B 52, Falcon-H1-3B 32
   - Granite-4.0-h-micro 40 [인용, `triage/notes.md:229` "36 mamba + 4 attn"]
   - 미발화 임계 `ext ≤ 65536 // L`: Zamba2 1,213 · Nano 1,170 · Granite 1,638 · Falcon 2,048.
3. **배치 extend 분포 [측정]**는 서버 로그의 `Prefill batch, #new-token`(= 배치 extend 토큰)에서 셌다. 대상은 HE0 33 run(`he0_realized.HE0_JOBS`)과 P1-opint 10 run(agnostic arm, 100토큰 미만 warm-up은 제외)이다.
4. **실측 span 분류 [측정]**: N3·N2-EA는 `lockstep_diag/lockstep_diag_values.json`의 span 표(PHASE_EVENTS)를 썼다. 규칙은 다음과 같다.
   - `n_chunks = 1` → 미발화.
   - CHUNK·WAIT 반복이 0 → decode 부재. 비교 기준이 없으므로 분류하지 않는다.
   - 마지막 chunk를 발행한 뒤 순수 WAIT 반복이 2회 이상 → 적체.
   - 그 밖의 경우 → 박자 또는 관측 지연(span_end는 최대 1반복 늦게 관측된다).
5. **span 시간 모형 [유도]**: Nano만 정본 식으로 계산할 수 있다.
   - 전체 시간: W(L, P) = T2′(32k, P) · T2(L) / T2(32k). T2 = P108 이차 적합(J-A1 108u), T2′ = decode 1개 동거 중 단일 prefill 시간이다 [인용, `cap_char/RESULT_costterms_2026-10-26.md` §3·§4].
   - W를 analytic-v1의 층별 FLOPs(`analytic_terms.py`: mamba proj+conv+SSD, MLP, attn proj+core, 마지막 chunk에 LM head) 비율로 chunk에 나눈다.
   - 이로부터 나오는 효율 η는 0.43–0.46이다 [유도].
   - **검증**: 모형 W(8k, P64) = 1,542 ms, N3 실측은 1,463–1,816 ms(중앙 약 1.49 s). 모형 W(8k, P108) = 905 ms, 실측은 870–891 ms [측정, DIAG §3-3].
6. **decode step [측정]**: Nano는 N2 PROFILE의 sticky-ON 단독 간격 p50(D, b, c)에 ρ 1.13–1.19를 곱했다 [측정, T3 총효과]. HE0·P1-opint는 client ITL p50(phase·rate별)을 썼다.
7. **Zamba2·Granite는 정본 식(analytic-v1, Nemotron-H 계열)에 없다.** 그래서 하한만 계산했다.
   - FLOPs ≥ 2·(비임베딩 파라미터)·토큰, η = 1, 312 TFLOP/s·P/108.
   - 평균 chunk가 하한이고, 가장 긴 chunk는 평균 이상이다.
   - 하한이 step보다 크면 "적체(확정)"로 둔다. 그렇지 않으면 Nano의 η 0.45를 옮겨 쓴 추정치를 띠로 분류한다 [가정 A8].
8. **영역 띠** [가정 A5]: r = (가장 긴 span)/(동거 step). r ≤ 0.67이면 박자, r ≥ 1.5이면 적체, 그 사이는 **경계(판별 불가)**다.

가정 목록(JSON `assumptions`에 같은 문구가 있다):
- A1: T2의 L 형태가 P < 108로 옮겨진다.
- A2: chunk 시간은 FLOPs에 비례한다(층 종류별 효율은 같다고 둔다).
- A3: T2′·P가 일정하다. 실측 407k–414k ms·SM(P44–92)을 P16으로 외삽한다.
- A4: D92 step은 108 SM 단독 값을 하한으로 쓴다.
- A5: 띠 ×1.5.
- A6: Little 법칙으로 diag 셀의 decode 배치를 잡는다. 2k → b≈8, 8k → b≈1.
- A7: Zamba2 파라미터 = safetensors 2.689e9 − 2·임베딩.
- A8: η 0.45를 Zamba2·Granite에 옮겨 쓴다.
- A9: Granite 파라미터 = 로그 weight 5.98 GiB/2 B − 0.9e9.

## 3. 결과 표

영역 비율의 분모는 prefill 배치 수(또는 span 수)다. budget은 표에 따로 적지 않은 한 전부 65536 [코드]이다.

| 캠페인 | 기판 | 모델 · L | extend 분포(배치 단위) | budget | 미발화 | 적체 | 박자 지배 | 근거 등급 |
|---|---|---|---|---|---|---|---|---|
| **HE0** (static d16/24/34/44, bind±gate, slo; 33 run, 33,680 배치) | KISTI A100, triton, cudagraph ON | Zamba2-2.7B · 54 | ShareGPT ctx-len 4000, rate 3↔12. 배치 extend p50 252 · p90 957 · p99 ≈3,000 · max 12,290 [측정] | 65536 | **94.0–95.2%** [측정] | 4.8–6.0%: LO step(10.9 ms) 기준 전부 하한으로 확정. HI step(20.0 ms) 기준 확정 0.3–3.3% + 추정 2.4–3.3% + 경계 0–2.6% [유도] | **0%** (추정 기준에서도 0) [유도] | 미발화 [측정] / 적체 [유도, 하한] |
| **P1-opint** (873944/873945, n=5, rate 2–6) | KISTI A100, cudagraph ON, 자동 격자 P ∈ {54, 74, 108} [코드] | Zamba2-2.7B · 54 / Granite-h-micro · 40 | random 2000/96 고정. 배치 extend p50 ≈2,106–2,109 · p90 4.2k–6.3k · max ≈16.8k [측정] | 65536 | **0%** [측정] (1,522 / 1,919 배치 전부 span ≥ 2) | Zamba2: rate 2–3(ITL 10.5–15.5 ms)에서 100% 확정. rate 6(48 ms)에서는 P54 추정 88% + 경계 12%, P108에서는 경계 100%. Granite: rate 2–6 전부 하한으로 확정(span ≥ 14.8 ms > ITL 7.6–13.0 ms) [유도, A9] | 0% (관측되지 않음; Zamba2 rate 6은 경계) | 미발화 [측정] / 적체 [유도] |
| **P1.7** 4모델 | KISTI A100, **cudagraph OFF** [인용 CONSENSUS] | Zamba2 54 · Granite 40 · Falcon-H1-3B 32 · Nemotron-H-8B 52 | random 2000/96 고정. 서버 로그가 로컬에 없다 | 65536 | 1-요청 배치: Falcon만 미발화(fc 32 ≥ 32), 나머지 3모델은 2 span [코드 산술]. 배치 분포는 미측정(P1-opint 분포를 대리로 쓰면 Falcon도 대부분 다중) | 판별 불가 | 판별 불가 | 로그 부재 |
| **N3** (Job 3·4 + floor, 80 cell, 2,663 span) | VESSL A100-SXM4, flashinfer | Nano-9B-v2 · 56 | random **8192** 고정. max_prefill 16384이므로 span extend는 8,192(7 chunk) 또는 16,384(14 chunk) [측정] | 65536 | 0% | decode가 있는 span의 **98.1%**(1,562/1,593). WAIT 반복 중앙 49/span. (W/n_chunks)/CHUNK p50 중앙 **10.1** [측정] | 1.9%(31 span, 박자 또는 관측 지연) [측정] | **[측정]** (decode 부재 1,070 span은 분류 밖) |
| **N2 Job-EA** (64 cell, 1,128 span) | VESSL, 400 W | Nano · 56 | co-load K×8192를 일괄 송신 → 대부분 16,384 span(14 chunk). 상주 요청 입장 span(c_in, 8192 배수 아님)도 포함 [측정] | 65536 | 1.4%(16) | **93.6%**(1,056). WAIT 중앙 28, (W/n)/CHUNK 중앙 **2.96**(최대 9.3) [측정] | 0% | **[측정]** (decode 부재 56은 분류 밖) |
| **T3 비용 항** (J-A1·A2·H2, 73 cell-rep = 29 셀) | VESSL | Nano 실가중치 + 조성 변형 a4/13/22/31 · 56 | L **8k**(7 chunk) / **32k**(28 chunk), bs 8/32, D24/44 | 65536 | 0% | **100%**. span 171–227 ms 대 동거 decode p50 22–63 ms, r = 2.7–10.2 [유도: 모형 span ÷ 측정 co_p50] | 0% | [유도] (span 원자료는 m12 볼륨) |
| **lockstep diag PREREG**(D-1a/b) 12셀 | VESSL(계획) | Nano · 56 | 2k·8k 고정, 저부하 | **{8192, 65536}** | **0/12셀** | 8/12셀: r 5.0–59 (65536 6셀 전부 + 8192의 D92 2셀) | 0/12가 확정. 4/12가 **경계**: 8192×D24(2k r 0.67, 8k r 0.75)·8192×D44(2k r 1.26, 8k r 1.44) [유도] | [유도] |
| 〃 추가 셀 2k × **512** × D44 | 〃 | Nano · 56 | 2,048 | 512 | 0 | 0 | **박자 지배**: fc 1 → 56 span, span 최대 7.6 ms 대 step 20.5×1.16 = 23.8 ms, **r = 0.32**(ρ 범위 전체에서도 박자) [유도] | [유도] |
| **정책 Job PREREG**(HD_HE0cell rev1.2) | VESSL(계획) | Nano · 56, static D {16, 24, 34, 44, 64} | W1 ShareGPT(ctx 4000, rate 0.375/1.5 λ\*) · W3 arXiv-Sum ≤16,384(p50 6,484 [인용]) | 65536 | W1: 요청 단위 96.8% [인용 census]. 배치 단위는 미측정(HE0 대리값 약 94–95% [가정]). W3: 약 0%(p50 6,484 → 6 span, 16,384 → 14 span) | W1 다중 span 배치와 W3 전부: r = 3.0–14 (L 1,171–16,384 × D16/44/64) [유도] | **0%** [유도] | [유도] / W1 배치 분포는 [가정] |

## 4. 판별 가능 여부와 질문별 답

1. **"현 diag 격자 {2k, 8k} × {8192, 65536}은 전부 미발화 또는 적체다" → 문자 그대로는 거짓이다. 부분적으로만 맞다.**
   - **미발화 셀은 0개다.** 2k × 65536도 fc = 32 < 56이라 span이 2개다.
   - 65536 6셀과 8192 × D92 2셀은 적체가 견고하다(r ≥ 5).
   - 8192 × D24(r 0.67–0.75)와 8192 × D44(r 1.26–1.44) 4셀은 **경계**다. 모형 오차(A1–A5)가 측정되지 않아서 적체인지 박자인지 가르지 못한다. 8192 × D24는 ρ 상단에서 박자 쪽으로 넘어간다.
   - 따라서 "박자 지배 셀 0개"는 **확정이 아니다**.
2. **2k × budget 512 × D44는 박자 지배다** [유도].
   - fc = max(1, 512 // 2048) = 1이므로 층 하나가 span 하나이고 56 span이다.
   - 가장 긴 span(mamba 또는 MLP 층)은 7.6 ms, 동거 decode step은 약 23.8 ms다. r = 0.32이고 경계 띠(0.67)에서 2배 넘게 떨어져 있다.
   - 이 판정을 뒤집으려면 span 모형이 3배 넘게 과소추정해야 한다.
   - 결과로 따라오는 것 [유도]: 박자 영역에서는 prefill 완료가 약 56 × 23.8 ≈ 1.3 s다. GPU 일 W = 0.40 s보다 길다. TTFT가 decode step 수에 묶인다.
3. **실측(N3·N2-EA)은 적체를 직접 보여 준다** [측정].
   - 마지막 chunk 발행 뒤 WAIT 반복이 중앙 49회(N3)·28회(N2-EA)다.
   - span당 chunk 시간이 CHUNK 반복 간격의 약 10배(N3)·약 3배(N2-EA)다.
   - 진단 문서의 "host가 다 발행한 뒤 약 90% WAIT"와 같은 그림이다.
4. **HE0는 거의 미발화 영역이다** [측정].
   - 94–95%의 배치가 span 1개다.
   - 나머지 5–6%는 하한 또는 추정 기준으로 적체이고, 박자는 0이다.
   - 따라서 HE0 결과는 "층 단위 교차가 거의 일어나지 않는 PD-mux"에서 나온 값이다.
   - 이 결론은 phase 계측 없이 로그 extend만으로 얻은 것이다. 다중 span 배치의 실제 GPU 시간은 미측정이다.
5. **P1-opint 운영점 headline(Zamba2 +40.5%/+185.8%, Granite +27.0%)이 나온 run은 미발화 0%, 박자 0%, 사실상 적체 영역이다.** HE0와 영역이 다르다.

## 5. 한계

- **HE0·P1 계열(Zamba2·Granite·Falcon)의 span 시간은 정본 식 밖이다.** 하한(η = 1, 파라미터 FLOPs)과 Nano η 이식 추정뿐이다.
  - 박자 판정은 낙관 쪽으로 기운다. 평균 chunk ≤ 최장 chunk이기 때문이다.
  - Granite 파라미터는 로그 GiB에 임베딩 여유분 0.9e9를 둔 가정이다(A9).
- HE0·P1-opint의 decode step은 phase·rate 전체의 client ITL p50이다. 배치별 phase·실현 P는 로그에서 얻지 못한다.
  - 동적 arm은 P92(하한이 가장 작은 쪽)로 계산했다.
  - P1-opint는 P ∈ {54, 108} 양끝으로 계산했다.
- Nano 모형은 8k에서만 실측으로 검증했다(오차 +3.5% 이내). 2k와 P16(외삽 A3), 층 하나 span(fc = 1)의 효율은 미측정이다. 경계 셀 4개는 이 공백 때문에 판별되지 않는다.
- N3의 "박자 또는 관측 지연" 31 span에는 decode가 span 중간에 생기거나 사라진 경우가 섞여 있다. 박자의 증거로 쓰지 않는다.
- P1.7은 서버 로그가 로컬에 없다. 영역 비율을 측정하지 못한다.
- 기판이 섞여 있다(KISTI 행과 VESSL 행). 행끼리 수치를 비교하지 않는다. 이 표는 영역 지형을 기술할 뿐이고 성능 판정이 아니다.
- **다음에 필요한 것**: diag 격자 경계 4셀(8192 × D24/D44)과 512 × D44 셀에서 `prefill_chunk.gpu_ms`(event 쌍, 계측 `3858f129`)를 재야 한다. 그 값과 같은 반복의 `decode_fwd_gpu_ms`를 비교하면 영역이 직접 판정된다. 이 셀들이 박자 영역이 실재하는지 보는 유일한 후보다.

## 6. cost model 기여

- **CM-2(prefill span 시간 `T_span(L, SM)`)**: 알게 된 것은 다음과 같다.
  - T2′(32k, P)·P가 P44–92에서 407k–414k ms·SM로 일정하다. 곧 T_span ∝ 1/P 형태다 [유도]. P16은 외삽이다.
  - T2(L) 형태 × T2′(32k, P) 모형이 N3의 8k span W를 +3.5% 이내로 재현한다.
  - 함축 η는 0.43–0.46이다(L 2k–8k, P 16–108).
  - 범위는 Nano·VESSL·flashinfer이고, 층 하나 span의 효율은 미측정이다.
- **CM-1/CM-4 접점(span 대 step 비 r)**: 새 무차원 항 r = T_span,max / (T_dec·ρ)가 PD-mux 실행 영역을 정한다. 기존 캠페인은 전부 budget 65536에서 r ≫ 1(적체) 또는 span 1개(미발화)였다. **박자 영역(r < 1)은 한 번도 측정되지 않았다.** CM-4의 "동거 간격이 span 시간에 반응하지 않음"(기울기 0.0015)도 적체 영역에서만 성립하는 관측으로 범위를 한정해야 한다(깨진 가정 후보).
