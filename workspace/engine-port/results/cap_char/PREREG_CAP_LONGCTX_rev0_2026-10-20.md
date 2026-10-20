# 사전등록 rev0 (DRAFT) — cap 블록 + long-context 블록 (Q-A…Q-D), 2026-10-20

> ★**대체됨(2026-10-20, 결정 14)**: 연구 프레이밍이 "비용 모델(C) 주 목적"으로 정해져 `PLAN_costmodel_terms_2026-10-20.md`로 대체된다. Q-B 이중차분은 처치 방향이 모델마다 반대라 폐기. 이력용.

> 지위: **DRAFT rev0. 미봉인, 감사 0회, 측정 0, 성능 판정 0.** 문서 작업(CPU)만 했다. 제출·GPU·커밋 없음.
> 수치 라벨: [측정](출처 명시) · [유도](식 대입·모형 예보, 측정 아님) · [코드] · [결정](사용자 결정 문서 명시) · [제안](이 초안이 고른 값, 감사 대상) · **[미등록]**(봉인 전에 사용자 또는 addendum이 채울 칸).
> 근거: 설계 `DESIGN_cap_longctx_v1_2026-10-19.md`(§2·§5는 아래 결정이 우선) · 결정 `../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 결정 1–13 · 프로브 `probe_128k/RESULT_probe_128k_2026-10-20.md`·`ANALYSIS_triton_96k_failure_2026-10-20.md`·감사 정정(coordinator 전달, 원문 `probe_128k/AUDIT_probe_128k_2026-10-20.md`는 이 초안 작성 시 아직 없었음 → 봉인 전 대조 의무) · J1 `../multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md`·`AUDIT_J1_N1_analysis_2026-10-19.md` · 명세 `../handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`(이하 SP, v0.2+) · 형식 선례 `../multigpu/PREREG_MULTIGPU_rev12_2026-10-15.md`.
> 기판: VESSL A100 SXM 80GB(108 SM), 모든 수치는 이 기판·**mem-fraction 0.6** 범위다. 0.82로 잰 이전 결과(M3 J1, 프로브 1회차 등)와 섞지 않는다(§4 금지).

## 0. 이 문서가 하지 않는 것 / 선행 조건

- G0 PREREG가 아니다. Q-C가 G0 cap을 낸다(R16-3). fused_tuned config는 G0 튜닝 argmax에 의존한다(§2-3).
- 선행 조건(전부 충족 전 제출 금지, 순서대로):
  1. I-5(`src/multiplex/admission_probe.py`) **엔진 트리 동기화**(현재 CPU 검증만, 미동기화) → engine-porter correctness gate(SP §9: ON/OFF 출력 일치, 관측자 TOST ±1.5%, 변이 시험) → **GPU gate**. 사유는 KV / cap(≡슬롯) 2분류다(결정 8′). 이 gate 전에 계측 필드를 쓰는 판정은 없다.
  2. 하네스 이식: per-model 사전 토큰화 파일(Nano 기존 `arxsum_x_tokens_v1.bin` sha `4c32f1af…` [측정] + Llama 토크나이저판 신규), 합성 길이 생성기, LongBench v1 로더, 출력 강제(`ignore_eos`), 모델 둘을 한 Job에서 순차 부팅하는 드라이버. 이식은 이 PREREG의 일부다(운영 모델 §4).
  3. Llama-3.1-8B(rev `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`)의 **agnostic(PD-mux) 부팅·단일 요청 정확성**. 프로브 G5는 Llama fused_sym만 확인했다. PD-mux가 Llama에서 도는지는 **미확인**이다. 실패하면 대조 모델은 fused_sym·fused_tuned만 갖고 Q-B는 PD-mux 항 없이 쓴다(범위 문장).
  4. 합성 생성기 CPU 검증(§5-3), Job-0 보정(§8-1).
  5. 규칙층 감사 → 하네스층 감사 → 봉인 → CPU preflight → `launch.sh` dry-run → 사용자 승인 → `--submit`.

## 1. 질문과 estimand (SP R8-3 형식)

공통: 효과는 **같은 Job 안** 짝 차분. λ\*\_SLO는 SP R9-1(seed 합산 attainment 0.90 교차의 log-rate 선형보간, 외삽 금지), Job 안 seed 짝 bootstrap은 **기술값**(n=4 → 서로 다른 재표집 35개, R5-5 공시), 판정은 "이 Job에서" 범위(R5-4). goodput 정의는 게이트 4(요청 TTFT ≤ SLO ∧ 요청 내부 token-ITL p95 ≤ 50 ms). |효과| < 3%는 방향을 인용하지 않는다(게이트 3). `CLIFF_ADJACENT`·R9-6 해상도 > 3%인 비교는 크기 문장 금지, 부호는 R9-2 조건 안에서만. primary k = 5 [제안, §10 O-3], k=3·10은 기술값.

| Q | 1차 estimand | 판정 규칙(사전 등록) | 허용 문장 | 금지 문장 |
|---|---|---|---|---|
| **Q-A** (cap 블록) | 모델 m·arm a·셀 c마다 R\_A = λ\*\_SLO(cap=C\_mem) / λ\*\_SLO(cap=48), 같은 Job·같은 공통 격자. 부속: 사유별 몫(KV / cap≡슬롯, **iteration 가중**), `CAP_BINDING` 구속률 b(SP R16-2), retract 건수, decode bs 히스토그램 | 두 cap의 λ\* 모두 정의되고 |log R\_A| ≥ 3%, `CLIFF_ADJACENT` 아님, 해상도 ≤ 3%이면 부호 인용. 자연 출력 셀은 **음성 대조**(b < 10% 예상, [유도]) — 여기서 R\_A가 3% 밖이면 cap 외 요인(cudagraph 캡처 목록 I-6)을 의심하고 구속 귀속 금지 | "이 Job의 (모델, arm, 셀)에서 cap을 48에서 C\_mem으로 푼 것의 λ\*\_SLO 비는 …이다. 구속률 b = …(`CAP_BINDING` 라벨)" | "cap이 goodput을 막는다/푼다"(b < 10% 셀에서), "슬롯 상한이 hybrid 고유", "generic 스케줄러는 mamba state를 무시한다", "Bullet/MuxWise가 C\_mem을 오예측한다" |
| **Q-B** (cap 블록) | 이중차분 Δ(a,c) = [log R\_A]\_Nano − [log R\_A]\_Llama, arm·셀마다 | 부호까지만. 두 항 모두 3% 규칙 통과 + 두 항의 `CAP_BINDING` 상태 공시. **C\_mem 방향이 모델마다 다르다**(아래 ★) → Δ는 "각 모델이 자기 C\_mem으로 옮겼을 때의 차"이지 "같은 처치의 차"가 아니다 | "각 모델을 48에서 자기 C\_mem으로 옮긴 효과의 차의 부호는 …이다(모델 크기·가중치·토크나이저·k 8× 교락)" | "hybrid 고유", "hybrid에서 cap 효과가 더 크다/작다"(크기), G 크기의 모델 간 비교 문장 |
| **Q-C** (cap 블록) | G0 cap 후보: 두 표준 셀(자연·4k) × **세 arm 모두**에서 `CAP_BINDING`이 아닌 최소 cap ∈ {48, C\_mem} (Nano 기준; C\_mem/2는 선택 [미등록]) | 규칙: 모든 (arm, 셀)에서 b < 10%인 cap 중 작은 쪽, 없으면 C\_mem. `CAP_BINDING_UNRESOLVED`(필드 부재)이면 C\_mem이 아니라 **판정 불가 → G0 cap 미결정(사용자 결정)**. 한 arm 데이터만 보는 규칙 금지(대칭). 규칙은 이 PREREG로 **데이터 전에** 등록한다 | "G0 cap = c (Q-C 규칙)" | 창식 K §2 출력을 cap 규칙으로 쓰는 것, 사후에 cap 선택 |
| **Q-D** (long-ctx 블록) | (i) 합성 L ∈ {16k,32k,64k}마다 G\_SLO(L) = λ\*\_agnostic / λ\*\_fused\_sym 와 G\_tuned(L) = λ\*\_agnostic / λ\*\_fused\_tuned, log G ~ log L 기울기(L 3점, seed 공동 재표집은 기술값) (ii) 실현 동시 수 대 C\_mem(L)(모델별 부팅 실측) (iii) decode step GPU 시간(I-4)의 ctx 의존 (iv) 정지 ≥200 ms 횟수·최대 정지(§6-4) | (i)은 L마다 Q-A와 같은 3% 규칙. 기울기는 점 3개라 **부호·단조성만**, 크기·"선형" 문장 금지. 대조 모델과는 (ii)·(iii)의 **형태 차이**까지만 | "이 Job에서 L = …(합성, ≤ 64k)에서 G\_SLO = …", "실현 동시 수는 C\_mem(L)의 …%였다(모델별)" | 모델 간 G 크기 비교, "~52k hybrid 경계", "9k 경계를 실측했다"(config 산술 [유도]), "hybrid의 긴 L 메모리 우위 실측"(유도 가설 L\_x ≈ 1.2k는 실측 전 인용 금지), "hybrid PD-mux는 64k 불가"(§4) |

- ★ **C\_mem 방향 비대칭 [유도, 프로브 부팅 로그 + 산식]**: mem 0.6에서 Nano 풀 = 33.04 GB(KV 1,587,762 토큰 + mamba 48슬롯 6.46 GiB; 슬롯 143.5 MB/요청, KV 16,379 B/토큰), 산식 C\_mem(L) = floor(풀 / (슬롯 + k·L)). Nano arXiv 자연 L̄ ≈ 7.3k → ≈ 125, 4k 강제 L̄ ≈ 11.2k → ≈ 100, 1k L̄ ≈ 8.1k → ≈ 119. Llama(슬롯 0, KV 263,900 토큰, k 131,095 B/토큰) → 36, 24, 33으로 **48보다 작다**. 즉 Llama의 "cap = C\_mem"은 cap을 푸는 것이 아니라 **낮추는** 것이고, cap 48에서 Llama는 KV 고갈·retract 영역에 든다. 그래서 Q-B의 두 항은 방향이 다르다. 이 PREREG는 {48, C\_mem}을 [결정]대로 등록하되 Δ 문장에 위 범위를 붙인다. 공통 cap 점 추가는 §10 O-2. [CS-OK] 33.04는 풀 GB 값(우연 일치)
- ★ **L\_ref 규칙 [제안]**: C\_mem의 L = 그 셀 유지 요청의 (입력+출력) 평균. p50·p90 기준 C\_mem도 부팅 표에 **기술값으로 병기**. 가변 길이이므로 평균 기준 cap에서도 retract는 남을 수 있다(I-5 retract 기록).
- ★ **C\_mem 등록 절차**: cap 값은 **데이터 전에 고정**한다. 부팅 로그(`max_total_num_tokens`, 슬롯 수, KV·가중치 바이트)로 (모델, 셀)마다 산식을 대입한 값을 봉인 전 부팅 프로브(Workspace, 측정 아님)로 확인해 LADDER\_REG addendum에 sha와 함께 등록한다 **[미등록 — 값]**. Job에서는 부팅 로그를 셀마다 기록하고 `max_total_num_tokens`가 등록 기대와 2% 넘게 다르면 `C_MEM_DRIFT`(공시, cap 불변). 풀이 cap과 독립이라는 가정(KV 토큰 수 + mamba 바이트 = 상수)은 [유도]이며 두 번째 cap의 부팅에서 검증한다.

## 2. Arm

### 2-1. 정의 (SP R8-1 바이트 단위 승계, 모든 arm 공통 노브)
cudagraph ON, radix OFF(F8), attention backend flashinfer, `--mem-fraction-static 0.6`(**모든 arm·블록·G0 통일** [결정 13]), `--max-running-requests = cap`, `--max-mamba-cache-size = cap`(Nano; Llama는 해당 없음), `max_prefill_tokens` 16384 명시, `mamba_ssm_dtype` SP 규칙, 로그 knob `PDMUX_HEARTBEAT_S=0`, 요청 logprob 없음.

| arm | 정의 |
|---|---|
| `fused_sym` | PD-mux off + `--chunked-prefill-size -1 --disable-overlap-schedule` |
| `agnostic` | PD-mux on, R2 정책 없음, `pdmux_agnostic_manual.yml`, `split_forward_token_budget` 65536 |
| `fused_tuned` | PD-mux off, **G0 튜닝 argmax config**(SP R8-2d: 11 config 합집합, k별 엄격 argmax; mixed-chunk 후보 포함) |

### 2-2. fused\_tuned config 결정 경로
G0 튜닝 Job(SP R8-2: 별도 튜닝 Job에서 argmax, 판정 Job에서 재측정)의 `fused_best11(k)` 중 **k = 5 [제안] 의 argmax**를 이 PREREG의 fused\_tuned로 쓴다. 튜닝은 A′·ITL 50 ms로 했으므로 이 PREREG의 long-ctx SLO(ctx 비례)와 다른 SLO에서 고른 config다 — **게이트 8 위반 방지: 이 PREREG에서 fused\_tuned를 "long-ctx에서 튜닝됐다"고 쓰지 않는다**(허용: "G0 A′ 튜닝 argmax config를 그대로 쓴 arm"). 재스코어 금지. mixed-chunk가 argmax이면 SP R8-2c 운용 규약(logprob 없음, 작은 예산의 prefill 몫 C − 실행 중 decode 수, PD-mux arm assert)과 correctness gate G-M1–G-M4를 선행한다.

### 2-3. G0보다 먼저 이 PREREG를 돌릴 때의 선택지
| 선택지 | 내용 | 비용·위험 |
|---|---|---|
| (a) **권고: G0 튜닝 Job 뒤에 제출** | argmax가 확정된 뒤 이 PREREG를 봉인(fused\_tuned config 칸을 채움) | 일정 의존. 가장 깨끗함. 단 Q-C가 G0 cap을 주므로 **G0 튜닝은 cap 48로 이미 돌았거나** 튜닝 Job이 이 cap 블록 뒤여야 한다는 순환이 있다 → 아래 (c) |
| (b) fused\_tuned를 빼고 돌린 뒤 별도 후속 Job | arm 2개(fused\_sym, agnostic)로 진행, fused\_tuned는 후속 Job에 fused\_sym·agnostic 재측정 포함 | arm이 Job 둘로 갈려 같은 Job 원칙상 후속 Job 안에서만 G\_tuned 계산. GPU-h +(agnostic 중복) |
| (c) **순환 해소(권고 보강)**: cap 블록의 fused\_tuned config는 "G0 튜닝 Job을 cap 48로 먼저" 확정 | G0 cap 후보가 바뀌어도 tuned config는 cap 48에서 고른 값이라는 범위 문장을 붙인다. SP R8-2의 "cap = G0 cap 고정" 조항과 어긋나므로 **SP 개정 또는 사용자 결정 필요**(O-5) | 규칙 변경 → 규칙층 감사 대상 |
| (d) G0 이전에 임시 대리 config(chunk 2048/8192 중 하나) | 프로브 G3-chunked가 쓴 대리 | **금지**: 사후에 argmax와 다르면 arm 정의가 사라진다. 쓰려면 "대리 arm(비 argmax)"로 명명하고 headline 불가 |

권고: (a)+(c). 어느 경우에도 fused\_tuned config는 데이터 전에 칸이 채워져야 하고 채워지지 않으면 해당 arm은 [미등록]으로 실행 불가(`ABORT_ARM_UNREGISTERED`).

## 3. 모델

| | Nemotron-Nano-9B-v2-Base | Llama-3.1-8B (base) |
|---|---|---|
| revision | `dc0661c…` | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` [결정 10′] |
| 구조 | hybrid(attention 4층·NoPE 추정 + SSM), s ≈ 143.5 MB/요청 [측정], k = 16,379 B/토큰 [측정] | attention 32층 RoPE, k = 131,095 B/토큰 [측정] |
| mem 0.6 KV | 1,587,762 토큰 [측정, 프로브] | 263,900 토큰 [측정, 프로브] |
| 토크나이저 | `tokenizer.json` sha `3277c00f…`(Instruct와 Base 동일 [측정, 결정 10′]) | **Base rev의 `tokenizer.json` sha 미대조** [미확인]. 길이 통계(LONGCTX/ARXSUM)는 Instruct 원본 rev `0e9e39f2…`(sha `79e3e522…`)·unsloth 미러(`6b9e4e7f…`)로 쟀다 → Base rev 토크나이저로 재측정 선행 |

- **같은 Job 안 두 모델** [결정(사용자 지시)]: Q-B 이중차분이 같은 Job의 짝 차분을 요구한다. 부팅은 모델·arm·cap마다 순차(GPU 1장 점유), 한 Job 안 순서는 블록 무작위(seed 등록). 가중치 적재는 `/data/hf` 오프라인.
- 입력 문서: 모델별 토큰열을 쓰되 **문서 ID 집합은 두 토크나이저 유지분의 교집합**(D-12). 교집합 크기는 Llama 토큰 파일 생성 뒤 등록 [미등록].
- 교락 공시(의무): 토크나이저 상이(합성 고정 토큰 길이면 무해, 실문서 셀은 모델별 토큰 수 병기), 가중치 바이트(Nano 16.6 GiB vs Llama ~15 GiB), k 8×, 층 구조. 모델 간 크기 비교 문장 금지.

## 4. 고정 노브 mem-fraction 0.6과 범위 문장

- 근거(프로브, 사실만): 0.82에서 chunk −1 단일 장문 prefill이 활성화 메모리로 OOM(Nano PD-mux 64k, Nano fused\_sym 96k, Llama 124k). 0.6에서 Nano chunk −1은 96k에서 Triton 오류, Llama 124k까지 OK.
- **96k 실패의 기술(정정 반영)**: "**Triton causal\_conv1d의 int32 토큰 오프셋 넘침으로 추정**한다(비연속 in\_proj view의 stride 22,656, 경계 T ≥ 94,793). 1회 관측이며 수리 반증 실험 전이다." SSD 커널 오류가 아니다(보고 지점일 뿐). 이 한계는 **chunk −1을 쓰는 Nano arm 전부**(agnostic·fused\_sym)에 공통이며 PD-mux 고유가 아니다. chunked 8192(fused 계열 chunk 있는 config)는 124k까지 됐다.
- **64k 상한은 측정된 한계가 아니라 격자 선택이다.** 96k·128k는 커널 수리(후보: 오프셋 int64 캐스트, `.contiguous()`) 전까지 제외한다. 범위 문장(의무): "이 PREREG의 long-ctx 결과는 합성 입력 ≤ 64k, SGLang v0.5.10, mem-fraction 0.6, Triton causal\_conv1d 현 구현에서의 값이다."
- 0.82에서 PD-mux가 fused보다 먼저 OOM 난 원인은 **미확인**이다(부팅 시 활성화 여유가 1.57 GB 적었던 것만으로도 설명됨). split-prefill로 귀속 금지.
- 금지: "PD-mux는 64k를 넘을 수 없다/hybrid PD-mux는 64k 불가"(범위 없이), "OOM 때문에 64k"(0.6의 96k는 OOM 아님), 0.82 결과와 0.6 결과의 혼합·비교·같은 표 병기.
- 0.6 부작용 점검(§6-5): Llama KV 풀이 263,900 토큰이라 KV 입장 제한이 **arm 비대칭**을 만들 수 있다 — 점검 항목으로 등록.

## 5. 입출력 셀과 context\_length

### 5-1. cap 블록
- 입력: arXiv-Sum 표준 셀(SP §10, ≤ 16,384 토큰 필터, 비복원 seed 추출, Poisson 도착).
- 출력: 강제(`ignore_eos`) {**자연**(abstract 길이, mean ≈ 203), **4k**(4,096), **1k** (예산이 허용할 때만, §8 Tier)}. 출력 < 2토큰 제외·건수 공시.
- **context\_length = 20480 유지**(D-18, G0 공유) [제안]. 확인 [측정, `arxsum_x_tokens_v1.bin` 헤더, 로컬 CPU]: Nano 유지분 5,906건 최대 입력 **16,379**, 최대 입력+자연 출력 16,744. 4k 강제 시 최대 16,379 + 4,096 = **20,475 ≤ ctx − 2 = 20,478**(여유 3토큰), 제외 건수 0. 스케줄러의 `max_new_tokens` 절단(`max_req_len − 입력 − 1`)도 4,096 이상 유지(최대 4,099). 1k는 여유 큼.
  - 위험: 여유 3토큰. **Llama 토큰열은 최대 입력이 16,384까지 갈 수 있어**(필터 ≤ 16,384) 4k에서 20,480 > 20,478이 되는 문서가 생길 수 있다 → 규칙: `fit_problems`(입력+출력 ≤ ctx−2)를 모델별로 적용해 위반 문서를 **교집합에서 제외**하고 건수 공시. 제외가 유지분의 0.5%를 넘으면 ctx를 24,576으로 올리는 안을 사용자에게 올린다(O-4; G0와 값이 달라지므로 G0 PREREG 동기 필요).
- 원 입력 필터를 전 출력 수준에서 동일 문서 집합으로 유지한다(출력 수준 간 표본 교락 방지, 교집합 − 위반 문서의 합집합 제외를 4k 기준으로 고정 [제안]).

### 5-2. long-ctx 블록
- **통제 축(합성 고정 길이)**: L ∈ {16k, 32k, 64k} = 16,384 / 32,768 / 65,536 토큰, 무작위 토큰(bench\_serving random 계열), radix OFF, 출력 {자연, 1k}. 자연 출력 = 200토큰 고정 [제안] (합성에는 abstract가 없음, `ignore_eos`). 품질 지표는 보고하지 않는다.
- **실제 분포 셀**: LongBench v1 **원본 그대로**(4,750 인스턴스, 21과제, 중국어 과제 포함, 입력 = context + "\n" + question, 과제 prompt 템플릿·chat 템플릿 없음; 길이는 LONGCTX\_STATS와 동일 정의) [결정 12′]. Nano 입력 p50 9,374·p90 19,599·max 68,980, Llama max 65,287 [측정]. 출력: 자연 = 참조 답 토큰 길이(≥ 2, 과제별 상한은 LongBench 자체 max\_gen), 또는 1k 강제. 표본: 순열 prefix(seed 등록), 과제 층화 여부 [미등록, O-6]. LongBench v2·InfiniteBench는 **보류**(범위 문장에 명시). 라이선스(과제별 원천) 확인은 공개 전 필요.
- **context\_length = 73,728**(72k) [제안]: 필요 하한 = LB v1 최대 68,980 + 1,000 + 2 = 69,982, 합성 64k + 1k + 2 = 66,538. KV 풀·슬롯은 context 무관 [코드·유도]. cap 블록(20480)과 다른 Job이므로 한 Job 안 arm 비교는 영향 없음.
- 상한 64k, 96k·128k 제외 사유와 범위 문장은 §4.
- cap = **C\_mem(L)** 하나(L마다 모델별 부팅 실측): Nano 16k ≈ 80, 32k ≈ 48, 64k ≈ 27; Llama 16k ≈ 16, 32k ≈ 8, 64k ≈ 4 [측정 부팅 로그의 `cmem.by_L`, 프로브 3회차]. `max-running-requests`·`max-mamba-cache-size` = 이 값.
- **선행 항목(G7 정정 반영)**:
  1. G7은 "단일 요청 prefill 시간이 내용과 무관하다"까지만 인용한다(n=2/arm, 첫 요청 제외 시 차이 거의 없음, 기술값). 결정 12′의 **bench\_serving random 생성기는 시험하지 않았다.**
  2. 봉인 전 CPU: 합성 생성기가 낸 토큰 수가 목표 L과 **정확히 일치**하는지(Nano·Llama 토크나이저 재인코딩 아님 — 생성은 토큰 id 직접) 전수 실측, 중복·특수 토큰 비율 기록.
  3. 봉인 전 GPU(Job-0): 합성 대 실제 문서(LB v1 장문 ≥ 4개, 가능하면 ≤ 64k 점 2개 이상)를 **ABBA/BAAB 교대**로 단일 요청 prefill 시간 비교. 차이가 3% 넘으면 통제 축을 "합성 ≠ 실문서"로 공시하고 합성 셀 결과의 범위 문장에 반영.

## 6. SLO, 계측, correctness gate

### 6-1. SLO
- 짧은 셀(cap 블록): **A′**(SP R6-1: T\_i = k·floor(L\_i), k ∈ {3,5,10}, primary 5 [제안], floor는 같은 Job의 fused\_sym 무경합 단일 요청 중앙값 16회, log–log 보간, L < 1k는 floor(1k)).
- long-ctx 셀: **ctx 비례 TTFT SLO** T = k·floor(L), floor는 **같은 Job 안**에서 모델별로 실측(L ∈ {1k,2k,4k,8k,12k,16k,32k,64k} 16회 중앙값 — 16k 초과 점은 SP R6-2 목록에 없어 이 PREREG가 추가 등록). J1의 0.111 ms/토큰·9 ms 절편은 **≤ 16k 선형** 범위이고 프로브(기술값)는 Nano 64k ≈ 8.5 s, Llama 64k ≈ 9 s로 초선형 조짐이 있다 → **외삽 금지**, 측정 점만 쓴다. LB 실분포 셀의 요청별 T는 보간(범위 밖 `None` → 그 요청 판정 불가가 아니라 L > 64k 요청 없음이 전제이며 LB v1은 ≤ 69k라 69k까지 floor 점 하나 추가 [제안: 68,980 상한 점, O-6]).
- ITL: 요청 내부 token-ITL(첫 토큰 제외) p95 ≤ 50 ms(`numpy.percentile` linear), mean-ITL은 secondary(같은 표 칸 금지).
- **D-7 주의(오류 요청)**: 서버 측 오류(5xx, abort, SSE error, `OUTPUT_SHORT`)는 **불합격**이고 셀 무효가 아니다. 클라이언트 측(송신 전 연결 실패, 발송 지연 > 0.5 s, 분류 불가) > 1%이면 비교 짝 무효·λ\* `UNRESOLVED`. **서버 프로세스가 죽으면**(슬롯 assert `memory_pool.py:544`, 커널 오류) 이후 요청이 연결 실패로 기록되어 "클라이언트 측"으로 오분류된다 → `SERVER_DEATH` 별도 분류를 등록한다: 죽은 (arm, cap, L)은 `UNRESOLVED(arm_death)`로 두고 죽음 시각·로그 시그니처(트레이스백 마지막 줄)를 보존, 다른 arm 결과를 지우지 않으며 "그 arm이 불가능"으로 일반화하지 않는다. client timeout ≥ 25·floor(최대 L): 64k floor ≈ 8.5 s [기술값] × 25 = 212 s < `sock_read` 600 s [유도].
- **ITL 50 ms의 원리적 도달성**: 긴 ctx decode 자체가 50 ms에 근접할 수 있다(Llama 64k KV/요청 8.6 GB [유도]). 셀마다 **ITL-only 실패, TTFT-only 실패, 둘 다**(R7-5)를 병기하고, 최저 rung에서 모든 arm의 요청 ITL p95 통과율 < 0.5이면 셀을 `ITL_INFEASIBLE`로 표시해 TTFT-only attainment를 대신 기술값으로 낸다(판정 문장 금지). 이 규칙은 데이터 전에 등록.

### 6-2. 계측 (SP §9)
I-1(클럭·전력·throttle, 수집 생존 검사; 0행이면 `SUBSTRATE_UNCHECKED`) · I-4(decode 커널 GPU 시간, Q-D 필수) · I-5(`pdmux.admission/v1`: 사유 KV/cap≡슬롯, sticky 상속 분리, decode bs 히스토그램, cap 이상 iteration 수, retract, `cap_slot_relation`, EXTEND running\_bs) · I-6(cudagraph 패딩; cap에 따라 캡처 목록이 바뀜) · 부팅 로그(`max_total_num_tokens`, 슬롯 수, KV·가중치 바이트).
- `CAP_BINDING`(SP R16-2, D-10): λ\*\_SLO 교차 rung 셀과 LO rung에서만, 구속률 b = running ≥ cap인 decode iteration 비율, b ≥ 10% → 라벨, 짝 규칙(한 arm이라도 묶이면 "cap c 조건부"), 사다리 {5,10,20}% 병기. **원천 필드(I-6 `bs_real` 또는 I-5 추가 필드) [미등록 — 계측 패치 몫]**, 없으면 `CAP_BINDING_UNRESOLVED`(Q-C 판정 불가).
- 몫은 **iteration 가중**(PD-mux가 결정 기회를 더 갖는다)과 시간 가중을 둘 다 낸다.
- I-5·I-1은 fused 스케줄러도 고치므로 관측자 TOST(±1.5%, n ≥ 8 짝)를 PD-mux 계열·fused 계열 각각 채운다. 판정 Job마다 ON/OFF 짝 블록.
- I-2(시계 정렬) 없음 → SP R11-2 공변 검사는 적용하지 못한다(M3와 같은 범위 문장: "R11-2 범위 밖, 클럭·전력은 기술값"). 클럭 고정 불가이므로 arm별 SW power cap 노출률·SM 클럭 중앙값을 셀마다 기술값으로 낸다. J1에서 B·X의 arm 비 8개가 `SUBSTRATE_COVARIES`였던 선례 → 크기 문장은 "400 W 한도에서 클럭 하락을 포함한 총효과"로만(F-1 문안). **긴 ctx 셀은 부하가 낮아 노출률이 다를 수 있음** — 예상하지 않고 공시만.

### 6-3. correctness gate (★ 항목 공통, SP §9)
(a) 계측 ON/OFF 출력 일치(greedy, 등록 prompt 64개) (b) 관측자 TOST (c) 변이 시험(I-3/I-4 호스트 시간 변이본 실패) (d) 스키마 버전 증가. 추가(이 PREREG): (e) 4k 강제 출력에서 `OUTPUT_SHORT`=0 (f) Llama agnostic 단일 요청 정확성(§0-3) (g) 64k 단일 요청이 세 arm 모두 OK(프로브 Nano chunk −1 64k OK, 0.6).

### 6-4. 정지 지표 (프로브 G4 정정 반영)
G4는 "ITL SLO 충족"의 근거가 아니다(요청 내부 p95가 1회 정지를 가린다; 32k 창 안 최대 1.13 s, 64k 0.80 s는 기술값). 모든 셀에 **정지 ≥ 200 ms 횟수와 최대 정지(ms)를 요청·셀 단위 기술값으로 병기**한다(decode token 간격, prefill 창 안/밖 분리). 판정 지표가 아니며 "정지 없음" 문장 금지.

### 6-5 점검 항목 — KV 입장 제한과 arm 비대칭 (mem 0.6)
Llama KV 풀 263,900 토큰(C\_mem(124k)=2, 64k 4, 16k 16) → shape C(긴 입력 + 긴 출력)에서 KV 입장 제한이 arm마다 다르게 작동하는지 점검한다. 점검: (i) 거절 사유 KV 몫과 retract 건수를 arm별로 기록해 arm 간 차를 본다 (ii) PD-mux의 split-prefill가 KV 예약 시점이 달라 retract 빈도가 다른지(가설, 귀속 금지) (iii) fused\_tuned의 chunked prefill이 KV를 점진 점유하는 것과 chunk −1 arm의 일괄 점유 차이. 이 점검에서 arm 간 KV 사유 몫 차가 10%p를 넘으면 해당 셀은 `KV_ADMISSION_ASYMMETRIC`으로 라벨하고 G를 "KV 입장 조건부"로만 쓴다 [제안, O-7]. 0.82 결과와 섞는 것은 금지.

## 7. 사다리·반복·판정 규칙(공통)

- **Job-0(보정 Job, 비판정)**: 모델·arm·셀마다 거친 사다리(ρ = 1.6, 4 rung, seed 1, 짧은 창)로 교차 위치 추정, floor 블록, 합성 대 실문서 ABBA(§5-2), C\_mem 부팅 표, 계측 생존 검사. **Job-0 수치는 판정·인용 불가**(게이트 6의 "용량 먼저"). 출력은 `LADDER_REG` addendum(셀별 rung 절대값, sha, **판정 Job 데이터 전에 봉인**) **[미등록 — 값]**.
- 공통 격자: (모델, 셀)마다 모든 arm·cap 공유, 인접 비 ρ = 1.2, 기본 7 rung, 예상 λ\*(Job-0 arm별 추정) 위·아래 각 2 rung(SP R9-3). arm 간 추정 폭이 1.44를 넘으면 `ceil(log(폭/1.44)/log 1.2)`개 추가.
- **확장 규칙(기본 시나리오로 예산에 포함)**: 교차가 격자 밖(`BELOW/ABOVE_LADDER`)이면 높은 쪽(또는 낮은 쪽) 2 rung 추가(seed 4), 최대 1회 [제안]. J1에서 확장이 시간의 46%였다 → 예산은 확장 발화를 기본으로 둔다.
- seed 4(판정 하한), 운영 목표 5는 +25%(O-8). 부팅은 (모델, arm, cap)마다 **2 pass**(seed {1,2}/{3,4}), pass 2는 arm 순서 역순(ABBA), rung 순서는 boot마다 seed 등록 무작위. 셀 사이 drain 완전(queue 0·running 0)+5 s. 모든 pair는 동일 trace hash·workload seed·server seed. warm-up = 예상 e2e 1배(자연 20 s, 1k 45 s, 4k 120 s, long은 floor 기반) 동안 보낸 요청은 전송하되 attainment에서 제외 [제안].
- n/rung/seed: cap 자연·1k 60, 4k 45, long 24–30, LB 48 [제안]. R9-6 해상도(2·√(0.09/n)/|Δa|·log ρ)를 셀마다 등록·공시, > 3%면 크기 문장 금지.
- 판정 범위: Job 1개 → "이 Job에서"(R5-4). Job ≥ 3 반복(SP R5-1)은 **본 예산에 넣지 않았다**(O-1). Job 간 분산은 B·X에서 미측정이므로 prior 상수 고정 금지.
- λ\*\_thr는 기술값으로 병기(처리율↔SLO 순서 반전 선례, J1 감사 1번: 둘이 갈리면 기술값으로 보고).
- 수치 비교는 모두 **같은 Job**. 다른 Job의 arm 값끼리 비교 금지(기판 교락).

## 8. 셀 수·GPU-h·Job 분할

### 8-1. 모형 [유도 — 측정 아님, Job 크기 결정용]
J1 셀 시간 모형 승계: 셀 = warm-up + max(n/x, n/λ\*\_arm) + drain tail + 5 s (x = rung 속도, 과포화 rung은 n/λ\*로 늘어남). boot 71–97 s, floor·xcheck·계측 포함 시 189–297 s → 여기서는 boot 200 s 가정. priors는 **임의 중심값**(Job-0 전 알 수 없음, arm 상대 1.0 / 0.8 / 1.1 = J1 B·X 패턴): Nano 자연 0.7, Llama 1.0, 4k Nano 0.45·Llama 0.15 req/s(Llama는 KV 동시성 ≈ 24/e2e 135 s), 1k 0.6/0.5, long 16k 0.4/0.5, 32k 0.19/0.23, 64k 0.09/0.075. TTFT 기술값(프로브): Nano 16k 1.8–2.2 s, 32k 3.7–4.6 s, 64k 7.8–8.4 s, Llama 64k ≈ 9 s. 스크립트 `scratchpad/bud.py`(비영속; 재현 시 이 문단 파라미터로 재작성).

### 8-2. 결과 (GPU 1장, 사용률 ≈ 100% — 아래 8-4)
| 구성 | 셀 수 | GPU-h (확장 없음 → 기본 확장) |
|---|---|---|
| **Full**: cap {자연,1k,4k} × 2모델 × 3arm × 2cap × 7+2 rung × 4 seed | 1,296 | 77.5 → **97.9** |
| Full-b: cap {자연,4k} | 864 | 57.2 → **72.2** |
| **Full long**: L{16,32,64}×{자연,1k} + LB 2출력, 7+2 rung | 1,728 | 113 → **142** |
| **Lean cap** {자연,4k}, 5+1 rung | 768 | **47.2** |
| Lean cap +1k | 1,152 | **63.9** |
| **Lean long**: 16k·32k·64k 자연 + 32k·64k 1k + LB 자연, 5+1 rung, n 24 | 1,152 | **63.9**(셀당 분해: 16k 3.9, 32k 7.0, 64k 16.1, 32k-1k 9.9, 64k-1k 20.1, LB 7.0 h) |
- 부팅·floor 오버헤드는 위에 포함하지 않았다: cap 24 부팅 ≈ 1.6 h, long 36 부팅 ≈ 2.4 h, Job-0 ≈ 6–8 h [유도].
- **총계(Lean): 약 47 + 64 + 4 + 8 ≈ 123 GPU-h; Full ≈ 250+ GPU-h.** 이 규모는 M3(예약 190 GPU-h)와 같은 급이다 → **범위 결정은 사용자 몫(O-1)**.

### 8-3. Job 하드캡 16 h와의 충돌 (중요)
같은 Job에 두 모델·세 arm·두 cap을 넣어야 한다는 제약 때문에 **4k 셀 하나가 Job 하나에 안 들어간다**: 4k 셀 Lean 기준 Nano 13.6 + Llama 24.2 = **37.8 h**(3 arm), 2 arm(fused\_sym, agnostic)만 25.7 h. Llama 4k는 λ\*가 낮고(≈0.15 req/s) e2e가 길다. 완화안(모두 [제안]): (i) Job-0 이후 **arm·cap별 4 rung 좁은 격자**(공통 격자 규칙 완화 → SP R9-3 편차, 감사 필요)로 ≈ 17–18 h (ii) n 45→30 (iii) 4k를 arm 쌍으로 나눔 — Job C2 = {fused\_sym, agnostic}, Job C3 = {fused\_tuned, agnostic 재측정}: GPU-h +33%이나 agnostic이 Job 효과 진단값을 겸한다 (iv) 하드캡 16 h 상향은 규칙 변경(운영 한도) → 사용자 결정. 64k 1k(20.1 h)도 같은 문제.

### 8-4. Job 분할 제안 (Lean, 모두 GPU 1장, 사용률 계산)
GPU 1장 Job은 `resourcespec-a100x1`을 쓰면 예약 = 사용(측정 중 GPU 100%, 부팅·drain 포함 평균 ≈ 95–98% [유도]; M3 J1이 x2에서 49%였던 것과 대비). "GPU Job 적정 크기" 규칙에 따라 x1.

| Job | 내용(모두 두 모델 × 3 arm, 해당 cap) | 예상 h [유도] | 비고 |
|---|---|---|---|
| J0 | 보정·gate·floor·합성/실문서 ABBA·C\_mem 표 | 6–8 | 비판정 |
| C1 | cap 자연(음성 대조) | ≈ 11 (9.4+부팅) | |
| C2 | cap 4k {fused\_sym, agnostic}, 완화안 (i)(ii) | ≈ 16–18 | **캡 초과 위험 → O-1/O-9** |
| C3 | cap 4k {fused\_tuned, agnostic} (§2-3 (a) 뒤) | ≈ 16–18 | fused\_tuned config 확정 후 |
| (C4) | cap 1k | ≈ 17 | 선택(Lean +1k) |
| L1 | 합성 16k 자연·1k + 32k 자연 | ≈ 16 | |
| L2 | 합성 32k 1k + LB 자연 | ≈ 17 | 캡 근접 |
| L3 | 합성 64k 자연 | ≈ 16 | |
| L4 | 합성 64k 1k | ≈ 20 | **캡 초과** → 64k 1k를 Tier-2로 |
| L5 | LB 1k | ≈ 8 | 선택 |
- 모델 둘은 **항상 같은 Job**. cap 블록과 long-ctx 블록은 context\_length가 달라 **Job을 섞지 않는다**.
- 각 Job 하드캡 = 예상 × 1.2, 상한 16 h [제안]. 재개 가능(`DONE` 마커·셀 단위), 자동 재시도 없음.

## 9. 판정 라벨·금지 문안

### 9-1. 라벨
`KNEE`-계열 대신 SLO 계열: `LAMBDA_RESOLVED` / `BELOW_LADDER` / `ABOVE_LADDER` / `UNRESOLVED_J` / `CLIFF_ADJACENT` / `RESOLUTION_COARSE`(>3%) · 구속: `CAP_BINDING` / `CAP_BINDING_UNRESOLVED` · 기판: `SUBSTRATE_COVARIES` / `SUBSTRATE_UNCHECKED` / `SUBSTRATE_WARNING`(드라이버 변경) · 오류: `SERVER_DEATH` / `CLIENT_ERROR` / `SERVER_ERROR_HI`(합산 1%, rev12 §19-1b 승계) · 이 PREREG 신규: `ITL_INFEASIBLE`, `KV_ADMISSION_ASYMMETRIC`, `C_MEM_DRIFT`, `ARM_UNREGISTERED`, `OBSERVER_EFFECT`, `FLOOR_JOB_EFFECT`.

### 9-2. 사전 등록 결정 규칙 요약
Q-A·Q-B·Q-D(i): §1 표. Q-C: 세 arm 대칭 최소 비구속 cap. 모든 비교: 같은 Job, 3% 규칙, `CLIFF_ADJACENT` 시 부호도 보류. 제외·무효는 데이터 전에 등록된 사유만. 사후 rung 라벨·양자화 값으로 t·CI 금지(R9-5).

### 9-3. 금지 문안 (VERDICT·AUDIT 승계 + 이 PREREG)
- 승계: "generic 스케줄러는 mamba state를 무시한다" · "Bullet/MuxWise가 C\_mem을 오예측한다" · "Bullet/MuxWise 오예측" · "~52k가 hybrid 경계"/"~52k에서 구속 전환" · "9k 경계를 실측했다" · "슬롯 상한이 hybrid 고유" · "Claim E 반증"(결정 1–3 금지) · J1 비의 크기 문장 · 대조 모델 대비 G 크기 문장 · "hybrid는 KV가 작아 배치 여유가 크다 REFUTED"의 구간 확장(1.0–1.5k 측정점·per-seq 범위) · CM-6 "처리율↔ITL 역순"을 쓰인 그대로 · "PD-mux가 클럭을 낮춘다" 일반 가정 · fused 정지 꼬리를 PD-mux 구조 이득으로 귀속(미조율 노브 효과일 수 있음, J1 허용 문안 1).
- 범위 없이 금지: "hybrid PD-mux는 64k 불가" · "PD-mux는 64k를 넘을 수 없다" · "OOM 때문에 64k" · "96k 실패는 SSD 커널 오류"(정정: causal\_conv1d int32 오프셋 추정) · "PD-mux 고유 한계"(chunk −1 Nano arm 공통) · "0.82에서 PD-mux가 먼저 OOM 난 것은 split-prefill 때문"(미확인) · "G4는 ITL SLO 충족" · "G7이 합성 생성기 동등성을 보였다" · 0.82 결과와 0.6 결과의 혼합.
- fused\_tuned: "long-ctx에서 튜닝된 fused", "fused를 어떻게 조율해도"(T2).
- Q-B: "hybrid 고유", 모델 간 크기 비교. Q-C: 창식 출력을 규칙으로.
- 선행 gate 전: "I-5가 거절 사유를 측정했다"(동기화·GPU gate 전), "preflight PASS가 이 하네스를 검증했다".
- 허용(J1 감사 5): F-1 문안(throttle 노출 + 총효과 범위).

## 10. 열린 결정 (사용자 몫)
- **O-1** 범위: Full(≈250 GPU-h) / Lean(≈123) / 더 작은 안. Job 반복(J ≥ 3, SP R5-1)은 예산에 없음 — 넣으면 ×3.
- **O-2** Q-B 방향 비대칭: 공통 cap 점(예: 96) 추가 여부(+GPU-h) 또는 Llama 항 설명 문장만.
- **O-3** primary k(5 제안)와 Holm 가족 여부(Job 안 판정이면 M3처럼 비적용).
- **O-4** context\_length 20480 유지(Nano 여유 3토큰, Llama 위반 문서 제외) vs 24,576(G0 동기 필요).
- **O-5** fused\_tuned 순환(§2-3 (c)): SP R8-2 "cap = G0 cap" 조항 개정 여부, G0 튜닝 Job 순서.
- **O-6** LB v1 표본(과제 층화, 중국어 과제 포함 여부, 69k floor 점).
- **O-7** `KV_ADMISSION_ASYMMETRIC` 문턱 10%p.
- **O-8** seed 4 vs 5.
- **O-9** 하드캡 16 h를 넘는 셀(4k·64k-1k)의 처리: 완화안 (i)–(iv) 중 선택.
- **O-10** 1k 출력의 포함 범위(cap 블록 1k, long 64k 1k).

## 11. [미등록] 목록과 감사 필요 항목

### 11-1. 규칙층
[미등록]: fused\_tuned config 값 / C\_mem·rung 절대값(`LADDER_REG`) / `CAP_BINDING` 원천 필드 / Llama 교집합 문서 수 / LB 표본 규칙 / primary k / 완화안 선택 / C\_mem/2 점 포함 / 정지 지표의 정의 문턱 외 사다리.
감사 쟁점: (1) **Q-B 이중차분의 처치 비대칭**(C\_mem 방향) — estimand가 의미 있는가 (2) Q-C 규칙이 세 arm을 대칭으로 쓰는가, `CAP_BINDING_UNRESOLVED` 처리 (3) 확장 규칙·Job-0 addendum이 "데이터를 본 뒤 사다리 선택"이 되지 않는가 (4) fused\_tuned를 A′로 튜닝하고 ctx 비례 SLO로 채점하는 것이 게이트 8 위반인가 (5) 공통 격자 완화(4 rung 좁은 격자) 허용 여부 (6) warm-up 요청 제외 규칙 (7) `SERVER_DEATH`의 D-7 정합 (8) 합성 입력의 대표성(G7 정정) (9) C\_mem L\_ref 선택이 arm 비대칭 경로를 만드는가 (10) 단일 Job 판정 범위에서 Q-D 기울기(3점)의 인용 한계.

### 11-2. 하네스층
이식 목록: 모델별 토큰 파일·합성 생성기·LB 로더·출력 강제·두 모델 순차 드라이버·pass 2 역순·`SERVER_DEATH` 분류·정지 지표 계산·C\_mem 표 기록·`C_MEM_DRIFT`·floor 확장 점(32k·64k·69k)·λ\*\_SLO 분석기(SP R9-1, 봉인 밖이면 fetch 전 커밋). 변이 시험 후보: "cap=48인데 C\_mem으로 부팅"·"Llama를 cap 슬롯으로 취급"·"warm-up 요청을 채점에 포함"·"서버 사망 이후를 클라이언트 오류로 셈"·"합성 길이 off-by-one"·"모델별 교집합 대신 Nano 집합 사용"·"4k 출력이 `max_new_tokens` 절단으로 짧아졌는데 성공으로 셈"·"I-5 필드 누락 시 CAP_BINDING 계산 통과". 선행 gate: §0 순서.
