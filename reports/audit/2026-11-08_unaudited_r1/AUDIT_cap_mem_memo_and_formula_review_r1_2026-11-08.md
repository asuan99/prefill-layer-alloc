# 결과 감사 r1: cap·mem 메모와 수식 선행연구 검토 (claims-auditor, 2026-11-08, 읽기 전용, GPU 0)

## 0. 회차와 감사 범위

**회차.** 두 대상 모두 **r1**입니다. 이전 감사는 없습니다.
- `tracks/main.md:11, :29, :30`에 두 문서가 "감사 필요(미반영)"로 올라 있습니다.
- 저장소 전체에서 `AUDIT_*`·`VERDICT_*` 중 두 파일을 대상으로 한 것은 찾지 못했습니다. `reports/audit/2026-09-28_contributions/AUDIT_1_prior_art_and_arm_2026-09-28.md`는 대상이 다릅니다.
- 상한은 대상당 2회이므로 r2가 한 번 남습니다. r2는 이 판정 이후의 diff만 봅니다.

**대조에 쓴 원자료**
- **엔진 트리** `~/Experiments/KISTI/sglang_engine_dev`, HEAD `1519acf3`. 로컬 수정 14파일이 있지만, 인용된 `memory_pool.py` 수정분은 무관합니다(`HybridLinearKVPool.get_v_head_dim` 패치).
- **ReplaySSM 저장소** `73a9592` 사본: `/tmp/claude-1000/-home-wonho-Experiments-KISTI/36eb3fc2-…/scratchpad/replayssm_repo/src`
- **ReplaySSM 논문** `2610.13778v1`(9 Oct 2026) 텍스트·PDF 두 사본. sha256 앞자리 `749c459895ed93a3`로 서로 같습니다.
- **1차 출처 PDF·텍스트**: `/tmp/claude-1000/-home-wonho-Experiments-KISTI/2a1fffe6-…/scratchpad/dl/`. Marconi `v3`, FA2, SSD, Kaplan, Chinchilla, DistServe, Priming 등이 있습니다.
- **로컬 vLLM 0.12.0**: `~/miniconda3/.../vllm`, `~/.local/.../vllm`
- **부팅 로그**: `cap_char/probe_128k/longctx_probe_20261020T034615Z/nano_fused/server.log`
- **결정문**: `anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md:59-75`

**공시**
- alphaXiv 도구는 이 세션에서 쓸 수 없었습니다. 1차 출처 대조는 위 `/tmp` 사본으로 했습니다.
- 이 사본들은 **다른 세션의 휘발성 scratchpad**입니다. 저장소 안에는 Bullet·MuxWise(`~/Experiments/KISTI/Papers/`)만 있습니다.
- 감사자가 직접 계산한 값은 두 종류입니다. 하나는 `analytic_terms.py` 호출이고, 다른 하나는 부팅 로그 보정이 들어간 **손계산**입니다. 뒤의 것은 아래에 [감사자 손계산]으로 표시했고, 해소 기준으로만 씁니다. 인용하지 마십시오.

---

## 문서 1. `workspace/engine-port/results/cap_char/MEMO_cap_memfraction_vs_replayssm_2026-11-07.md`

### 주장 1. "cap 48은 근거 없이 승계된 값이다"

**판정: CONFIRMED(최초 도입에 한정). 현재 근거로 확장하면 OVERBROAD.**

독립 재확인한 사실:
- `git log -S"max-running-requests 48"`로 보면 최초 등장은 `40d68699`(2026-07-04, P1.4 Task2)입니다. diff에는 `--mem-fraction-static 0.82 --max-running-requests 48 --context-length 8192`만 있고, 커밋 메시지·보고서 diff에 선택 근거가 없습니다.
- 직전 커밋들은 값이 달랐습니다. `ed6f2275`은 0.55·32, `d17ce46e`·`be13ba12`는 0.6·8입니다.
- 메모는 이 사실을 `COSTMODEL_running_cap_2026-10-15.md` §1에서 가져왔는데, 이번에 이력으로 직접 확인했습니다.

과장인 부분:
- **"이후 모든 캠페인이 상속"은 사실이 아닙니다.** 스크립트 집계로 cap 48이 732회로 압도적이지만, 다른 값도 있습니다.
  - compo_probe·비용 항 J-A는 cap 128
  - 비용 항 T4 셀은 cap = C_A 114
  - 그 밖에 8·16·32·40
- 승계 뒤에 cap 48은 **비교 가능성용 통제 상수**라는 역할을 얻었습니다. 예: STOCK-BASE `PM-ref`가 "CAP/TUNE/implfix 수치와 잇는 다리 arm"이고, multigpu는 48·N입니다. 비용 항 `CAP_INVARIANT`(D44에서 cap 48 대 128, |ln 비| ≤ 0.51%)도 있습니다.
- 따라서 "근거 없는 값이니 바꿔도 잃는 게 없다"는 따라 나오지 않습니다.

**confound**: 해당 없음. 이력 사실입니다. 다만 "근거 기록 없음"은 부재 증거라서 범위를 한정해야 합니다.

**허용 문안**: "cap 48은 2026-07-04 커밋 `40d6869`에서 처음 쓰였고, 그 커밋과 보고서에 선택 근거가 기록돼 있지 않다. 이후 다수 캠페인이 이 값을 통제 상수로 승계했다(cap 128·114 boot도 존재)."

**인용 금지 문안**
- "모든 캠페인이 cap 48을 썼다."
- "cap 48은 근거가 없으므로 비교 가능성 손실 없이 바꿀 수 있다."

**해소**: 문장을 "최초 도입 시 근거 기록 없음"으로 좁힙니다. cap을 바꾸는 캠페인에서는 `PM-ref`(cap 48) 다리 arm을 함께 두도록 등록합니다.

### 주장 2. "radix OFF에서 cap은 정적 mamba 슬롯 수와 같고, 고갈되면 assert"

**판정: CONFIRMED(조건부). 인용 위치·식은 수정 필요.**

맞는 코드 인용:
- `model_runner_kv_cache_mixin.py:218-230`. `max_mamba_cache_size`가 명시돼 있으면 그 값이 우선이고, 아니면서 `disable_radix_cache ∧ max_running_requests`이면 `max_mamba_cache_size = cap // dp`입니다. 따라서 "cap 지정" 조건과 "max_mamba_cache_size 미지정" 조건이 모두 필요합니다. PM arm은 두 값을 모두 48로 줘서 결과는 같습니다.
- `memory_pool.py:543-545`의 assert(`Not enough space for mamba cache…`)가 맞습니다.

수정할 부분:
- **assert는 할당 경로의 방어 장치입니다.** admission이 `max_running_requests` = 슬롯 수로 묶이므로 정상 운영에서는 도달하지 않습니다. "고갈 assert가 운영 제약"이라는 식으로 읽히면 안 됩니다.
- **radix ON 식이 틀렸습니다.** 메모는 "cap = 슬롯/(3+α)"라고 쓰고 `:391-402`를 인용했습니다.
  - `:390-402`는 비율 함수 `_calculate_mamba_ratio`일 뿐입니다.
  - 실제 적용은 `:871-874`의 `max_num_reqs = min(max_num_reqs, max_mamba_cache_size // ratio)`입니다.
  - ratio는 3 + {0, 1, 2}입니다(`:64-66`; extra buffer와 overlap 여부에 따름).
  - 따라서 등식이 아니라 **min**입니다.
- 부팅 배너는 ssm_state 6.46 GiB를 48슬롯으로 보고합니다. 슬롯당 값이 analytic 대비 1.0208배 ≈ 49/48이라, 풀이 cap + 1 슬롯일 가능성이 있습니다(코드 미확인). 결론에는 영향이 없습니다.

**허용 문안**: "SGLang v0.5.10 dev tree에서 radix OFF이고 `--max-mamba-cache-size`를 따로 주지 않으면 mamba 슬롯 수 = `--max-running-requests`이고, 슬롯은 부팅 때 정적으로 잡힌다. radix ON이면 동시 실행 상한은 min(cap, 슬롯 // (3+α))이다."

**인용 금지 문안**
- "radix ON이면 cap = 슬롯/(3+α)이다."
- "슬롯 고갈 assert가 운영 중 cap을 제약한다."

**해소**: 인용을 `:218-230`, `:64-66`, `:871-874`, `memory_pool.py:543-545`로 고치고, min 식으로 바꿉니다.

### 주장 3. "ReplaySSM B=256은 GPU 1장·고정 배치·짧은 프롬프트의 오프라인 측정(vLLM 통합 풀)"

**판정: GPU 1장·고정 배치·오프라인은 CONFIRMED. "짧은 프롬프트"는 OVERBROAD. vLLM 인용 위치 오기. 부속 주장 "용량은 줄이지 않는다"는 OVERBROAD.**

확인된 설정:
- **GPU 1장, TP 없음**: 논문 §7.1 "each on one GPU without tensor parallelism".
- **4B 모델은 H100 80 GB**: 부록 E.1(l.1565).
- `e2e_decode_speedup.py` 기본값을 직접 확인했습니다.
  - 모델 `NVIDIA-Nemotron-3-Nano-4B-BF16`, `--batch-size 256`, `--num-steps 1000`
  - `max_model_len = num_steps + 256 = 1256`(`resolve_max_model_len`)
  - `gpu_memory_utilization 0.9`, `enable_prefix_caching=False`, `enable_chunked_prefill=False`
  - `prompts = [DEFAULT_PROMPT] * batch_size`(짧은 한 문장), `ignore_eos`
  - `LLM.generate()` 일괄 실행
- **추측 디코딩 스크립트**: GSM8K, `max_tokens 256`, `max_model_len 2048`, B 512로 메모와 일치합니다.
- 덧붙임: ms/step 값에는 prefill 시간이 섞여 있습니다(elapsed/num_steps).

과장인 부분:
- **"짧은 프롬프트"**: 스크립트 기본값으로는 맞습니다. 그러나 논문의 B=256 결과 전체에 적용할 수는 없습니다. Fig. 1 하단(§3.1)은 B=256에서 decode 문맥을 약 100K까지 훑은 지연 분해입니다(Super-120B, B300). 메모 스스로 "그림별 실행 명령은 저장소에 없음"이라고 썼으므로, Fig. 2 등을 스크립트 기본값으로 연결하는 것은 추론입니다. 논문은 이 배치를 "serving batch size 256"이라고 부릅니다.
- **vLLM 인용 위치**: `scheduler.py:574-613, 933-940`은 **ReplaySSM fork 트리**(dev 버전, 2,816줄)의 선점과 "cannot schedule → break" 경로와 일치합니다. 그런데 설치된 **vLLM 0.12.0**(1,726줄)에서 `:933-940`은 encoder cache 코드입니다. "로컬 0.12.0"이라는 표기가 틀렸습니다.
- **A100 `max_num_seqs` 기본 256**: fork `arg_utils.py:2514-2533`에서 맞습니다. 다만 H100 기본은 1024이고, ReplaySSM 스크립트는 `max_num_seqs=batch_size`로 명시합니다. 256은 vLLM 기본값이 아니라 저자가 고른 B입니다.
- **"ReplaySSM은 용량을 줄이지 않는다"**: 표준 디코딩·FP32 state에서는 맞습니다.
  - ring buffer 약 1.5 GB를 재현했습니다. shape는 `mamba_utils.py:248-266`의 x (n_h,16,p), dt (n_h,16), B (G,16,N)이고, Nano-4B 21층 × 256에서 ≈ 1.53 GB입니다. dtype(x bf16, dt fp32, B bf16)은 가정이며 미확인입니다.
  - 그러나 추측 디코딩에서는 ReplaySSM이 기존 방식의 T개 snapshot을 없애 동시성 상한을 올립니다(l.560, 660, 1709).
  - §7.4의 저정밀 공유 state는 용량도 줄입니다.

**confound**
- 11(기판 혼합): ReplaySSM은 H100/B300·vLLM이고 우리는 A100·SGLang입니다. 메모는 이 둘을 구분하고 있어 위반은 아닙니다.
- 7·1 계열(오프라인 고정 배치 micro를 서빙 운영점으로 읽는 오류): 메모의 결론이 이 오류를 막는 쪽입니다. 정당합니다.

**허용 문안**: "ReplaySSM 저장소 `73a9592`의 `e2e_decode_speedup.py` 기본 설정은 GPU 1장·TP 없음, 같은 짧은 프롬프트 256개 복제, `max_model_len` 1256, prefix caching·chunked prefill OFF, `ignore_eos` 1000 토큰의 오프라인 일괄 생성이다. 도착 과정·TTFT·ITL 꼬리·SLO는 없다. 논문의 B=256은 이런 고정 배치 측정점이며, 도착 부하 SLO 서빙에서 256으로 운영할 근거가 아니다."

**인용 금지 문안**
- "ReplaySSM의 B=256 결과는 모두 짧은 프롬프트에서 잰 것이다."
- "ReplaySSM은 state 메모리 용량을 줄이지 않는다"(표준 디코딩·FP32라는 한정 없이).
- "vLLM 0.12.0 `scheduler.py:933-940`"

**해소**
- "짧은 프롬프트"를 "스크립트 기본값"으로 한정하고, Fig. 1의 B=256 문맥 스윕(B300)을 병기합니다.
- vLLM 인용을 "ReplaySSM fork `73a9592` `vllm/v1/core/sched/scheduler.py`"로 고칩니다.
- 용량 문장에 "표준 디코딩·FP32 state 한정"을 붙입니다.
- 출처 사본을 저장소에 고정합니다(커밋 해시와 PDF sha256 기록). 지금은 `/tmp`에만 있습니다.

### 주장 4. "mem 0.6은 결정 13의 장문 OOM 대응이며 다수 요청용이 아니다"

**판정: 출처는 CONFIRMED. "다수 요청용 아님"은 PLAUSIBLE이지만, 이후 결정 이력과 함께 써야 합니다.**

확인된 사실:
- 결정 13(`DECISION_headline_G_shapeC_2026-10-19.md:59-62`)의 근거는 128k 프로브 1·2회차입니다.
  - 0.82에서는 chunk −1 활성화 OOM으로 Nano PD-mux 64k가 실패했습니다.
  - 0.6에서는 64k가 통과했고, 96k는 Triton causal_conv1d 경계였습니다(`RESULT_probe_128k_2026-10-20.md:11-12, 43, 56`).
  - KV 2,722,655 → 1,588,248 토큰, 여유 11.72 → 29.0 GiB 수치도 원자료와 맞습니다.
- "통일 이유는 결정문에 명시돼 있지 않음": 결정문에는 프로브 근거와 "0.82 결과와 혼합 금지"만 있고, cap 블록·G0까지 넓힌 별도 이유는 없습니다. 맞습니다. 다만 "혼합 금지"는 비교 가능성 동기를 시사합니다.

빠진 이력:
- **결정 15·16(같은 문서 :69-75)은 cap 특성화를 "Nano cap = C_mem(mem 0.6)"으로 의도적으로 묶었습니다.** 즉 0.6은 나중에 cap 블록의 용량 정의에도 의식적으로 쓰였습니다. "다수 요청 수용과 무관한 값"이라고만 쓰면 이 이력이 빠집니다.
- §4의 "스크립트 115개 정적 집계"는 범위가 정의돼 있지 않아 재현할 수 없습니다. 감사자 grep(`workspace/`·`scripts/`, `*.sh|*.sbatch|*.py`)으로는 `mem-fraction-static`을 담은 파일이 877개(사본 포함)였고, 토큰 분포는 0.82가 503, 0.80이 288, 0.75가 9, 0.8이 6, 0.6이 5, 0.55가 1입니다. "0.82가 지배적, 0.6은 10-20 이후 cap_char 계열"이라는 정성 판단은 이 분포와 모순되지 않습니다. 정확한 개수는 **NOT-YET-SUPPORTED**입니다.
- STOCK-BASE에서 PM arm만 0.6이고 SG arm은 자동이라는 것은 맞습니다(PREREG §2 :79, 구성 묶음 caveat).
- "cap 48에서 0.6은 KV를 구속하지 않는다"(983,040 < 1,586,264)는 산술상 맞습니다. 다만 출력 토큰 예약을 포함했는지는 적어야 합니다.

**confound**: 10(변수 동시 변경) 경고입니다. STOCK-BASE처럼 mem과 다른 노브가 묶인 비교에서 mem 효과를 따로 읽으면 안 됩니다. 메모도 "mem 효과 자체는 측정된 적 없음"이라고 썼습니다. 정당합니다.

**허용 문안**: "mem-fraction 0.6은 결정 13에서 128k 프로브의 chunk −1 장문 활성화 OOM(0.82, 64k)을 피하려고 도입됐고, 같은 결정이 근거를 따로 밝히지 않은 채 모든 arm·블록(cap 블록·G0 포함)으로 넓혔다. 이후 결정 15·16은 cap 특성화의 C_mem을 이 0.6에서 정의했다. mem 자체의 서빙 효과는 측정된 적 없다."

**인용 금지 문안**
- "mem 0.6은 다수 요청과 무관한 값이다"(결정 15·16 병기 없이).
- "스크립트 115개 중 …"(집계 범위·명령 공시 전).

**해소**: 결정 15·16을 병기합니다. 115개 집계는 재현 명령과 범위를 달거나 삭제합니다.

### 주장 5. "cap 256은 mem 0.6에서 state만으로 풀 초과, 평균 1.3k면 mem 약 0.75 필요 [유도]"

**판정**
- **앞 문장은 CONFIRMED.** `analytic_terms.py` 기준 s = 143,548,416 B이고, 256 × s = 36.75 GB입니다. 측정 풀은 30.79 GiB = 33.06 GB입니다(배너 KV 24.24 + ssm 6.46 + conv 0.09). 슬롯이 cap + 1이어도 결론은 같고 여유가 큽니다.
- **뒤 문장의 수치(0.75, "2.6k면 0.82", "7k면 80 GB 초과")는 REFUTED(단위 혼용).** 결론의 방향(0.6보다 크게 올려야 함)은 PLAUSIBLE입니다.

근거:
1. **분모 오류.** SGLang의 정의는 `rest = post_load_avail − pre_load_avail × (1 − frac)`입니다(`model_runner_kv_cache_mixin.py:171`). 분모는 "80 GB"가 아니라 부팅 직전 가용량 **78.74 GiB ≈ 84.5 GB**입니다(`server.log:30`; 0.6에서 풀 30.66 GiB가 나와 배너와 맞음). 메모는 십진 GB 합(17.8 + 36.7 + 5.3)을 80으로 나눴습니다.
2. **감사자 재계산 [감사자 손계산 — 인용 금지, 해소 기준용]**: 가중치 16.58 GiB(배너) + state 34.22 GiB + KV(256 × c × 16,384 B)를 78.74 GiB로 나누면 다음과 같습니다.

   | 평균 문맥 | 메모 값 | 감사자 재계산 |
   |---|---|---|
   | 1.3k | 0.75 | ≈ 0.71 |
   | 2.6k | 0.82 | ≈ 0.77 |
   | 7k | "80 GB 초과" | ≈ 0.99 |

   7k의 0.99는 물리적 초과는 아니지만 활성화·cudagraph 여유가 없어 실현 불가능합니다. 같은 보정으로 0.75는 평균 약 2.1k까지 수용합니다.
3. **KV 값 불일치.** 1.3k의 KV는 `analytic_terms` 기준 256 × 1,300 × 16,384 = 5.45 GB입니다. 메모의 5.3과 맞지 않습니다(손계산 흔적).
4. **모두 하한입니다.** "평균 실제 점유"는 SGLang admission의 출력 예약(`new_token_ratio` × `max_new_tokens`)과 피크 점유를 무시합니다. 1.3k가 어느 워크로드의 값인지도 적혀 있지 않습니다.
5. **수식 정본 밖의 계산입니다.** mem-fraction 환산은 analytic-v1에 없는 항(§8 "메모리 할당 구조", T4 산식 B는 확장 항)이라 `analytic_terms.py`로 계산할 수 없습니다. 메모는 s·k만 코드에서 가져왔고 나머지는 손계산입니다. CLAUDE.md의 "손계산 수치 인용 금지" 규정에 걸립니다.

**반전 시험(§5 권고에 대해)**

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| 분모 80 GB 대 78.74 GiB | 메모 대 실제 | 권고 "0.75에서 cap 256" **유지**(올바른 보정이 오히려 관대함) | 1.3k: 0.75 → 0.71, 수용 한계 ≈ 2.1k |
| 평균 문맥 | 1.3k → 2.6k | 0.75에서 cap 256 **불가로 반전** | 2.6k 필요 ≈ 0.77 > 0.75 |
| 점유 정의 | 평균 → admission 예약 | 필요 mem 상승(미산출) | 출력 예약 미반영 |

→ 권고의 성패는 워크로드 문맥 분포에 달려 있습니다. 유도값으로는 정할 수 없고 부팅 측정이 필요합니다.

**confound**
- 1(sim-vs-serving) 변형: 유도 메모리 산술을 실현 가능성 판정으로 쓰는 것.
- 5(metric cliff) 예비 경고: 실현 가능 경계 근처(0.71–0.77)에서 KV 고갈이 선점이나 admission 차단 절벽을 만들 수 있습니다.

**허용 문안**: "Nano-9B-v2에서 cap 256의 SSM·conv state 예약은 36.7 GB [유도, `analytic_terms.py`]로, mem-fraction 0.6 boot의 측정 풀 33.1 GB [측정, 배너]를 넘는다. 따라서 mem 0.6에서는 cap 256 boot가 불가하다 [유도]."

**인용 금지 문안**
- "평균 1.3k면 mem 약 0.75가 필요하다."
- "평균 2.6k면 0.82가 필요하다."
- "7k면 80 GB를 넘는다."
- "mem 0.75면 cap 256을 수용한다."

**해소(실험)**
- **CPU 단계**: mem-fraction 환산을 스크립트로 만듭니다. 입력은 부팅 로그의 `pre_load avail`, 가중치 사용량, `analytic_terms`의 s·k, 출력 예약 규칙입니다. 이 스크립트는 analytic-v2 §8 확장으로 넣거나 T4 산식 B 코드에 붙입니다.
- **GPU 단계(사용자 결정 뒤)**: 한 Job 안에서 cap 256 boot를 mem {0.70, 0.75, 0.80}으로 각 1회씩 띄우고, 배너의 `max_total_num_tokens`·avail mem·cudagraph 사용량을 기록합니다. 실현 가능성 판정은 boot 성공과 부하 중 최대 메모리로 하고, 서빙 n과 SLO는 그 다음 PREREG에서 다룹니다.
- **기판**: VESSL A100 단일 Job, `substrate.json`·`clk_summary.json` 확인.

### 문서 1 종합: **GO-with-edits**

- 死因(NO-GO 사유) 없음. 이 메모는 정책 주장이 아니라 코드·이력 메모입니다.
- 필수 수정 다섯 가지:
  1. §3의 0.75·0.82·"80 GB 초과"를 삭제하거나 스크립트 값으로 교체하고, 하한이라고 표시
  2. radix ON 식을 min으로 바꾸고 인용 줄 수정
  3. vLLM 인용을 fork 트리로 수정
  4. "짧은 프롬프트"와 "용량 불변"의 범위 한정
  5. 결정 15·16 병기와 115개 집계 범위 공시
- §5 권고의 mem {0.6, 0.75}, cap {48, 128, 256}은 위 반전 시험대로 워크로드 문맥에 의존합니다. 사용자 결정 문구에 "0.75는 평균 ≈ 2k 이하에서만 cap 256 수용 [유도, 하한]"을 붙이십시오.

---

## 문서 2. `reports/cost_model/REVIEW_formula_prior_art_2026-11-02.md`

**1차 출처 대조 가능 여부**: 가능했습니다. 대조한 출처는 ReplaySSM `2610.13778v1`, Marconi `2411.19379v3`, FA2, SSD, Kaplan, Chinchilla, DistServe, Bullet(N_d = 30, Bullet.txt:407), Priming 일부입니다. Pope·Roofline survey·Jenga·Nemotron-H 인용 문구는 대조하지 않았습니다. Priming의 "state = at most 2K tokens of KV" 문구는 감사자 검색으로 찾지 못했습니다(미확인). 사본은 휘발성 `/tmp`에만 있습니다.

### 주장 1. "F_S,dec·2s·c\*는 ReplaySSM 부록 C와 같은 식"

**판정: CONFIRMED(주석 2개 필요).**

원문 대조:
- **C.1**: head당 FLOPs는 d_v d_k(decay) + d_v d_k(외적) + d_v d_k(합) + (2 d_v d_k − d_v)(readout) = 5 d_v d_k − d_v입니다. n_h개 head를 합하면 5 d_in N − d_in이고, 정본 §3의 5 d_in N과 −d_in만 다릅니다.
- **traffic**: "loads S_{t−1} and stores S_t, moving 2b d_v d_k bytes" → 2s입니다.
- **C.2**: crossover = 층 비 × (state/KV 토큰) × 2(load + store)이고, "Batching does not move the crossover"라고 씁니다 → c\* = 2s/k, b와 무관합니다.
- 날짜: arXiv v1이 9 Oct 2026이고 analytic-v1이 2026-10-22이므로 13일 앞선다는 것도 맞습니다.

필요한 주석:
- 우리 s에는 **conv state가 들어 있지만**(Nano에서 s_M의 1.39%) ReplaySSM C.2는 SSM state만 셉니다. 또 KV dtype이 다릅니다(ReplaySSM FP8, 우리 bf16). 같은 것은 **식의 형태**이고 상수는 다릅니다.
- "이 식은 우리 유도가 아니다"는 우선권(인용 의무) 문장으로 쓰는 게 정확합니다. 독립 유도였는지는 판정할 수 없습니다.

**허용 문안**: "decode state 갱신 FLOPs 5·d_in·N(−d_in 생략), state read + write 트래픽 2s, state–KV decode 트래픽 교차 문맥 c\* = 2s/k와 그것이 b에 무관하다는 관찰은 ReplaySSM(arXiv 2610.13778v1, 2026-10-09) 부록 C.1–C.2에 같은 형태로 있다. 우리 식은 conv state를 포함한다는 점만 다르다."

**인용 금지 문안**
- "c\*·2s는 우리가 처음 도출했다."
- "우리 식은 ReplaySSM과 상수까지 같다."

### 주장 2. "causal 계수는 Marconi와 관례가 다르다"

**판정: CONFIRMED.**

원문 대조:
- Marconi Table 1의 attention FLOPs는 `8LD² + 4L²D`이고 causal 절반을 반영하지 않습니다(2411.19379v3 l.978).
- FA2 §4.1은 "4·seqlen²·head dim·heads … with causal mask, we divide this number by 2"라고 씁니다(l.497-499). 그러면 2·n_q d_h L²로 정본 §2와 같습니다.
- Chinchilla 부록 F는 logits 2·L²·(key·heads) + reduction 2·L²·(…)로 비causal 4입니다.
- Kaplan Table 1은 Mask `2 n_layer n_ctx d_attn`입니다.
- 정본 머리말은 "형식은 Marconi Table 1을 따르고"라고 쓰면서, §9 차이 목록에 attention 계수가 없습니다.
- L_q 191k, 64k 비중 0.34(계수 4일 때)는 산술이 맞습니다(382k/2, 0.172 × 2).

**과장 주의**: "FlashInfer causal 커널에는 2가 물리적으로 맞다"는 커널 작업량을 프로파일하지 않은 추론입니다. **PLAUSIBLE**입니다.

**허용 문안**: "analytic-v1의 F_A,pre = 2·n_q·d_h·L²는 causal 반삼각을 세는 FA2 §4.1 관례이고, Marconi Table 1(4L²D)·Chinchilla 부록 F의 밀집 관례와 계수 2배 차이가 난다. L_q와 이차항 비중은 이 관례에 비례한다(계수 4이면 L_q ≈ 191k)."

**인용 금지 문안**
- "Marconi의 attention FLOPs는 틀렸다."
- "계수 2가 서빙 커널의 실제 작업량임이 확인됐다."

### 주장 3. "그 밖에 계산 오류 없음"

**판정: 정본 §6·§7 대입값은 CONFIRMED. 검토서 전체 수치로 넓히면 OVERBROAD.**

정본 값의 독립 손검산(코드를 다시 돌린 것이 아님):
- W_A = 55,050,240, W_M = 147,374,080, W_F = 140,492,800
- s_M = 5,316,608이고 × 27 = 143,548,416, k = 16,384
- F_S,pre = 262,144 + 2,621,440 + 5,242,880 = 8,126,464, F_S,dec = 6,553,600
- α = 15,645,310,976, γ = 40,960, L_q ≈ 381,966
- a₀ = 8.14 ms, b₀ = 0.141 ms, c₀ = 8.04 ns, c\* = 17,523

전부 일치합니다. 검토서가 SSD §6.3 본문의 "BMM(T/Q, Q, P, N)" 차원 오기를 지적한 것도 원문으로 확인했습니다.

과장인 부분:
- **검증 방식이 항등식에 가깝습니다(교훈 9 계열).** 검토서는 "`analytic_terms.py`로 다시 계산해 문서와 일치"라고 했습니다. 그러나 정본 수치는 같은 코드가 만든 것이고 `test_analytic_terms.py`가 이미 재현하므로, 이것은 전사(transcription) 검사일 뿐 식 검증이 아닙니다. 독립 검증은 감사자 손검산과 배너 대조뿐입니다.
- **머리말 "수치는 전부 `analytic_terms.py`로 다시 계산했다(손계산 인용 없음)"는 거짓입니다.** 9.3k, 191k, 0.34, 4.6배, 2.408e8, 9.18e6, 9.6%, 1.31e6, 0.23%, 0.6%, 20,480, 295/281/110 MB, <0.2% 등은 코드 출력이 아니라 본문 산술입니다. 감사자 검산으로 값 자체는 대체로 맞습니다.
- **B-7 "ReplaySSM 커널이면 c\* ≈ 1.06·L\* ≈ 9.3k"는 부정확합니다.**
  - checkpoint 항만 쓴 (L+1)/(2L) = 0.531을 썼는데, 같은 문장에서 buffer를 포함한 0.58을 인용합니다.
  - conv state는 ReplaySSM이 줄이지 않는데 그것도 반영하지 않았습니다.
  - Nano 상수(p = 80, N = 128, group 공유 B)로 buffer와 conv를 넣으면 ≈ 9.4–10.1k입니다 [감사자 손계산].
  - 부호와 순서는 바뀌지 않습니다.
- **A-2 "≈ 1.31e6 FLOP"**: BLOCK_K를 0으로 둔 상한입니다. 실제 차이는 (Q − BLOCK_K)/(2Q) × 2,621,440입니다.
- **§3(b)의 신규성 "u(D)를 SM 축으로 낸 것은 문헌에서 확인하지 못했다"**: **NOT-YET-SUPPORTED**입니다. 부재 증거이고, 검토서 스스로 SketchSSM·DUET·HYDRA 등을 열지 않았다고 공시합니다.

**confound**
- 교훈 9(검증 도구가 항등식): 코드 재실행으로 코드 출력을 검증함.
- 신규성 판정의 부재 증거.
- 1(sim-vs-serving): 검토서가 §3(b)에서 "사후 재현뿐, EVAL 사전 예측 전에는 '모형이 D\*를 예측한다' 금지"라고 이미 막아 두었습니다. 정당합니다.

**허용 문안**: "analytic-v1 §6·§7의 대입값은 식에서 손으로 다시 계산해도 일치하고, 파라미터·s·k는 부팅 배너와 일치한다."

**인용 금지 문안**
- "검토서의 모든 수치는 `analytic_terms.py`로 계산했다."
- "ReplaySSM 커널이면 c\*는 9.3k다."
- "u(D)의 SM 축 특성화는 문헌에 없다."

**해소**
- 머리말을 "정본 §6·§7 값은 코드로 재현했고, 검토서 본문 수치는 손산술이다"로 고칩니다.
- B-7은 "checkpoint 항만이면 ≈ 1.06 L\*, buffer와 conv를 넣으면 더 크다"로 바꾸거나, 이 값을 analytic-v2 코드 함수로 만듭니다.
- 신규성 문장은 열지 않은 문헌을 확인할 때까지 "확인 범위 안에서"로 한정합니다.
- 출처 사본 경로와 sha256을 저장소에 기록합니다.

### 문서 2 종합: **GO-with-edits**

- 死因 없음. 핵심 두 주장(ReplaySSM 동일 형태, causal 관례 차)은 1차 출처로 확인됐습니다.
- 필수 수정 네 가지:
  1. 머리말의 "손계산 인용 없음" 정정
  2. B-7의 9.3k 한정
  3. "계산 오류 없음"을 정본 §6·§7로 한정하고, 독립 손검산이 감사에서 이뤄졌다고 병기
  4. 1차 출처 사본을 저장소에 고정(지금은 `/tmp`에만 있음)
- 정본 v2 반영(§9 차이 목록에 계수 2 대 4, ReplaySSM 출처, 2s의 "eager read + write 커널 기준" 한정)은 검토서 §4 표대로 정본 변경 규칙을 따릅니다. 이 감사는 그 표의 방향에 이의가 없습니다.

---

## 주요 파일
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/cap_char/MEMO_cap_memfraction_vs_replayssm_2026-11-07.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/cost_model/REVIEW_formula_prior_art_2026-11-02.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/cost_model/ANALYTIC_COST_FORMULAS.md`
- `/home/wonho/Experiments/KISTI/sglang_engine_dev/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`(:64-66, :171, :218-230, :390-402, :871-874)
- `/home/wonho/Experiments/KISTI/sglang_engine_dev/python/sglang/srt/mem_cache/memory_pool.py`(:543-545)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/cap_char/probe_128k/longctx_probe_20261020T034615Z/nano_fused/server.log`(:30-44)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`(:59-75)
- 휘발성 출처 사본: `/tmp/claude-1000/-home-wonho-Experiments-KISTI/36eb3fc2-250c-457b-8b64-0de2c5693dc0/scratchpad/{replayssm,replayssm_repo}`, `/tmp/claude-1000/-home-wonho-Experiments-KISTI/2a1fffe6-16fb-47f8-9d0f-e0809f5a5416/scratchpad/dl/`

> 저장: main 세션이 claims-auditor 최종 보고 원문을 그대로 옮김(2026-11-08).
