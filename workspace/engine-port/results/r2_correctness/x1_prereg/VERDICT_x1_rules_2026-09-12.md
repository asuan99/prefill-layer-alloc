<!-- claims-auditor 반환문 원문 전사 (2026-09-12, 메인 세션이 스크립트로 추출;
     재타이핑·요약 아님). 대상 = PREREG_X1_SENSITIVITY_2026-09-12.md(rev1).
     이 판정서의 조건 D1-D4/D6-D9를 반영한 판본이 rev2다. -->

# X1 사전등록 규칙층 감사 판정서 (claims-auditor, read-only, GPU 0)

- 일자: 2026-09-12. 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/x1_prereg/PREREG_X1_SENSITIVITY_2026-09-12.md` + `x1_preflight_splits.py` + `x1_cross_job_compare.py` + 두 stdout.
- 방법: 등록 문서의 산술을 **믿지 않고 커널 소스에서 다시 계산**했다. 비교 코드는 **원자료로 독립 재계산해 교차 확인**했다. 파일 수정 0건, GPU 0.
- 근거 코드(직접 읽음): `triton_backend.py:110-143, 185-235, 286-300, 447-468, 665-717, 807-1038, 1039-1103, 1223-1267, 1270-1319`, `triton_ops/decode_attention.py:35, 80-110, 200-250, 520-560, 610-635`, `server_args.py:647-649, 1344-1359, 2561-2563, 5546-5557`, `cuda_graph_runner.py:454-480, 1068-1140`, `r2_correctness_check.py` 전문 헤더 + 330-470, `r2_correctness.sbatch` 전문, Zamba2-2.7B `config.json`.

---

## §0 판정: **`GO-with-caveats`**

- **死因 0건.** N1(estimand 부재/항등식)·N2(반전 확인)·N3(예측 도달 불가)·N4(제1원칙 위반) 전부 **미발화**. 반전 시험 7개 표면 중 **등록 판정을 실제로 뒤집은 것은 0개**다(§2 표).
- 단 **등록 문서의 핵심 수치가 거짓**이다. 자유 표면 "교란됨의 정의"를 커널이 실제로 쓰는 granularity로 밀면 `6 → 무교란 0/24`가 **`6/24`**로 바뀐다. 이 변화는 *이탈 결정(8 대신 6)*과 *GO/NO-GO*를 뒤집지 않으므로(초안 8은 오히려 `22/24` 무교란으로 더 나빠진다) **死因이 아니라 등록 caveat + 제출 전 수정 조건**이다.
- 따라서 **실행 승인**이다. 단 D1–D4·D6–D9는 **데이터 생성 전 반드시 반영**해야 한다(§5의 "데이터 생성 이후 수정 금지"는 사후 수정만 금지하므로, 지금 고치는 것이 규율에 맞다).
- ⚠️ `GO-with-caveats`는 "주장 범위가 좁아진다"는 뜻이다. **X1은 어떤 게이트도 닫지 않는다.** Claim D 선결 #1·#2·#3·#4a·#4b·#5와 R2C-1…16·P-1…7은 X1 실행 후에도 **문자 그대로 불변**이다.

---

## §1 死因 검사 (요청 1(a)–(d))

### 1(a) 양성대조 자신이 항등식인가 — **부분적으로 그렇다(24/96). 전체 항등식은 아니다.**

등록 pre-flight는 `get_num_kv_splits_triton`이 돌려주는 **행별 split 수**만 비교한다. 그러나 실제 축약 구간을 정하는 양은 커널 안에서 한 단계 더 **양자화**된다:

```
decode_attention.py:35   _MIN_BLOCK_KV = 32
decode_attention.py:98   kv_len_per_split = cdiv(cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
decode_attention.py:553  (stage-2 combine가 같은 식을 다시 계산)
```

즉 `split 수`가 달라도 `kv_len_per_split`이 같으면 split 경계·비어있지 않은 split 개수·combine 순서가 **전부 동일**하고(빈 split은 `if split_kv_end > split_kv_start:`로 건너뛰며 store도 그 안에 있다), `num_warps=4`·`BLOCK_N`·`MIN_BLOCK_KV`도 불변이므로 결과는 **비트 단위로 같다**(`MAX_KV_SPLITS` constexpr와 `att_out` stride가 바뀌어도 값은 불변 — FP 산술은 주소 무관).

실효 기준으로 907100의 실제 `prompt_tokens`에 대입한 재계산:

| 강제값 | 교란되는 비교 단위(실효) | 무교란(구성상 항등식) |
|---|---|---|
| **8**(판정서 초안) | **2/24** (S05, S06만) | 22/24 — O 8/8 전부 포함 |
| 7 | 17/24 | 7 |
| **6**(등록값) | **18/24** = 72/96 unit-pair | **6/24** = 24/96 — S00,S01,S02,S03,S04,S13 |
| 5 | 18/24 | 6 |
| 4 | 17/24 | 7 |
| 3 | 20/24 | 4 |
| **2** | **24/24** | **0** |
| 시나리오 B(cap=6, static 무효) | 17/24 | 7 — 위 6개 + S05 |

⇒ 결론 세 가지.
1. 등록 문서의 "**6이면 24/24가 교란된다 · 무교란 0**"은 **거짓**이다. 올바른 값은 **18/24 교란, 72/96 unit-pair**다. `x1_preflight_splits.py`의 docstring이 스스로 선언한 목적("does the planned perturbation actually change the decode attention **reduction**")을 자기 코드가 달성하지 못했다 — de-confound 교훈 9의 한 변종(항등식 검출 도구 자신이 granularity를 틀림)이다.
2. **이탈 결정은 반전되지 않는다.** 초안(8)은 실효 기준으로 2/24만 교란(O는 0/8)이므로, "초안대로면 양성대조가 사실상 항등식"이라는 문서의 판단은 **더 강하게 옳다**.
3. 그러나 **무교란 6개의 정체가 치명적으로 나쁘다.** 판정서 §1.6은 S를 "긴 프롬프트 9/16이 4-gram ≥0.98(복사형), 짧은 7/16이 ≤0.04"로 나눴다. 실효 무교란 6개(S00–S04, S13)는 **짧은 7개 중 6개**이고, 시나리오 B(env var 무효)에서는 **7/7 전부**다. 즉 교란은 argmax 마진이 큰 복사형 17–18단위에만 떨어지고, 뒤집힐 확률이 가장 높은 비복사형 단위는 **구성상 비트 동일**이다. F2의 검정력이 구조적으로 가장 나쁜 곳에 배치됐다.

### 1(b) 산술 pre-flight가 자기 결론을 가정하는가 — **아니다(전제는 모두 입력). 단 라벨 한 줄이 부정확하다.**

| 전제 | 실제 출처(파일로 확인) | 판정 |
|---|---|---|
| `CORE_COUNT=108` | `get_device_core_count` = `torch.cuda.get_device_properties(id).multi_processor_count` (`utils/common.py:1882-1884`) — 장치 속성. green ctx 무관(판정서 §2.6 재확인). 907100 노드 = `A100-SXM4-80GB`(provenance) | 입력 ✓ |
| `NUM_HEAD=NUM_KV_HEAD=32` | `hf_cache/.../Zamba2-2.7B/config.json`: `num_attention_heads=32, num_key_value_heads=32` ⇒ `kv_group=1` ⇒ `token_grid = num_seq*32` 분기 맞음 | 입력 ✓ |
| 길이 = 907100의 `prompt_tokens` | **서버 응답 `meta_info`에서 읽은 값**(`r2_correctness_client.py:176, 219`). 단 입력 텍스트(`prompt_sha256` 56/56 동일)와 토크나이저의 결정함수이고 4 boot 합계가 전부 동일(11007/15942/28053) | 실질 입력 ✓ / **라벨 부정확** |
| bs 도메인 | capture_bs = `[1,2,4,8,12,16,…]`(`server_args.py:1354-1359`) ⇒ bs 1·2는 **패딩 없음**. 패딩이 있으면 `seq_len_fill_value=1`(`triton_backend.py:791-792`)이 `min_seq_len`을 1로 끌어내려 휴리스틱이 완전히 달라지는데, S·O는 그 경로를 타지 않는다 | 입력 ✓(문서에 미기록) |
| `bg_lo=15` 하드코딩 | 기록된 bg `prompt_tokens`는 **17** | 무해(둘 다 `mks1`이 8로 포화) / **수정 권고** |

문서의 "길이 = … (입력값, 결과 아님)"은 "**응답에서 읽었으나 입력 텍스트와 토크나이저의 결정함수이며 4 boot 동일**"로 고쳐야 한다. 그리고 비교기는 `prompt_sha256`만 보고 `prompt_tokens`는 **보지 않으므로**(cross-job), 산술 전제가 조용히 깨질 통로가 남는다(D11).

### 1(c) 강제값 6이 **다른 것까지** 바꾸는가 — **바꾼다. 문서의 "fill_뿐"은 거짓.** 단 양 arm 대칭이라 게이트 의미는 보존된다.

`--triton-attention-num-kv-splits 6`은 `self.max_kv_splits`(`:114`)를 바꾸고, 그 값은 `fill_` 말고도 다음을 바꾼다(코드 확인):

- `attn_logits` shape `(bs, 32, max_kv_splits, 160)`, `attn_lse` `(bs, 32, max_kv_splits)` (`:290, :295`) — **버퍼 shape**.
- cudagraph 버퍼 `(max_num_tokens, 32, max_kv_splits, 160)` (`:450, :455`), `cuda_graph_num_kv_splits = torch.full(..., max_kv_splits)` (`:461-463, :1223-1225`).
- stage-1 **커널 grid z축** `grid = (batch, head_num, MAX_KV_SPLITS)` (`decode_attention.py:206`) 8→6, stage-2 루프 상한 `MAX_KV_SPLITS` (`:555`).
- `r2_correctness_check.py`의 **O1 문턱** `probe_ptok > (K-1)*(bg_ptok+bg_max_new)`에서 K=6 ⇒ 문턱 1239→885(전 probe 통과, 더 느슨해짐).

무해성은 확인됐다: 그래프는 6-shape로 **처음부터 capture**되므로 재캡처가 아니고(문서의 "재캡처 없음"은 결론만 맞다), 빈 split은 store하지 않으며, `triton_attention_split_tile_size=None`(`server_args.py:649`)이라 `:139-142`의 덮어쓰기는 발동하지 않고, `:2563`의 `=16` 덮어쓰기는 `is_hip()` 전용이다. `enable_deterministic_inference`도 off다. 다만 문서의 근거 문장은 **"`fill_`뿐이므로"**가 아니라 "**MAX_KV_SPLITS constexpr·grid z·버퍼 shape가 함께 6으로 고정되며, 그 구성으로 1회 capture된다**"로 고쳐야 한다(D3). 그러지 않으면 사후에 "X1은 fill 값 하나만 바꿨다"는 **거짓 축소 서술**이 정본에 승계된다.

부작용 중 하나는 실질적이다: **prefill(extend)은 전혀 교란되지 않는다.** `forward_extend`(`:807-1038`)에는 `num_kv_splits`/`max_kv_splits` 참조가 0건이고, extend 모드에서 `num_kv_splits=None`으로 설정된다(`:357, :376, :424`). 판정서 §2.3이 "비교 대상 중 D44 64-SM에서 돈 유일한 계산 = probe prefill"이라고 못 박았으므로, **X1이 F2를 성공시켜도 그 계산의 민감도는 여전히 미보정**이다. 또 `output_ids[0]`은 extend 산출이므로 **first_divergence=0인 불일치는 교란으로 설명되지 않는다**(D2의 귀속 규칙).

### 1(d) 교란이 arm 비대칭을 만드는 경로 — **S·O에는 없다. C에는 있지만 C는 등록상 판정 불가다.**

- env·CLI 모두 4 boot에 동일 주입(`server_cmd`가 warmup까지 공유). `SAME_ACROSS_BOOTS`에 `triton_attention_num_kv_splits`가 포함되어(`r2_correctness_check.py:119-122`) **boot 간 불일치는 자동 FAIL**로 잡힌다. 907100 기록값은 `['8']` — X1에서 `['6']`로 바뀌는 것이 보고서에 남는다.
- S: bs=1 고정(decode_host_tasks 16×63=1008, 판정서 §1.1) ⇒ 배치 구성 변동 0.
- O: O1이 "bg가 probe보다 늦게 끝난다"를 **probe별·boot별로 강제**하므로 probe decode 47 step 전부 bs=2 ⇒ GEMM M 불변, 패딩 버킷 2 고정. 게다가 강제 split 하에서 probe 행의 실효 구간은 **자기 길이만의 함수**가 되어 bg와 완전히 독립이다.
- C: 교란이 decode attention을 **행 단위 배치 불변**으로 만들어 arm-특이 경로 하나를 제거한다(§1.7의 C01 기전). 그러나 bs 시퀀스 자체는 arm마다 다르게 남고, GEMM의 M 의존 타일링 등 **다른 배치 의존 경로는 그대로**다. 그래서 C 층 결과는 어느 방향이든 귀속 불가다(R2C-3 유지).

### 제1원칙·confound 카탈로그 대조

| # | 해당 | 내용 |
|---|---|---|
| 1 sim→serving | 아님 | 실엔진 서빙 측정 |
| 2 미조율 오귀속 | 아님 | 정책 주장 0건 |
| 3 small-n | **해당** | boot 2개/arm. F3의 "arm-분리 소멸"은 균등 귀무에서도 P=(2/3)^4≈0.20로 발생 |
| 4 max 분모 | 아님 | |
| 5 metric cliff | 아님 | 지시함수 아님 |
| 6 재스코어 | 아님 | rule v2 무수정 + 새 데이터 |
| 7 micro/GIL | **부분** | S/O는 마이크로 프로토콜. 서빙 부하 동치 주장 불가(기존 R2C-11 유지) |
| 8 비운영점 | **해당(새 축)** | static KV splits=6은 **운영 수치 구성이 아니다**(기본 8 + 휴리스틱). 단 성능·정책 주장이 0건이라 N4는 미발화 |
| 9 positioning | 아님 | |
| 10 변수 동시 변경 | **해당** | env var + CLI arg 동시 주입. 실효 기준 env var의 한계 기여는 **1/24 단위(S05)뿐** ⇒ D4 |

---

## §2 반전 시험 표 (필수)

원자료 = `job_907100/gen_*.json`(실제 `prompt_tokens`), `job_907032/gen_L{1,2}.json`. 격자 = 강제값 2–8 × 단위 24개 × 전 decode step. 추정량 = 커널이 쓰는 `kv_len_per_split = cdiv(cdiv(L,s),32)*32`(소스 인용). 반전 계산에 체커·비교기 코드는 쓰지 않았고, 비교기 출력은 **독립 재계산으로만** 검증했다.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| "교란됨"의 정의(행별 split 수 → 커널 실효 축약 구간) | `MIN_BLOCK_KV=32` 양자화 적용 | **없음**(양성대조는 여전히 항등식 아님, 이탈 결정도 유지) · 단 등록 수치는 거짓 | 강제 6: 무교란 **0/24 → 6/24**(24/96 pair가 구성상 비트 동일) · 초안 8: **16/24 → 22/24** |
| 강제값(등록 허용 범위 2…8) | 2까지 밀기 | **없음**(F2 도달성 개선) | 무교란 단위: 8→22, 7→7, 6→6, 5→6, 4→7, 3→4, **2→0** |
| env var 유효성(시나리오 A vs B) | 완전 무효로 밀기 | **없음** | 교란 단위 18 → **17**. 차이는 S05 **1개**뿐 ⇒ 두 노브의 한계 기여가 1/24 |
| 민감도 표적의 성질 | "argmax 마진 작은 단위"로 제한 | **없음**(판정 불변) · 검정력 경고 | 교란 17–18개 전부 4-gram ≥0.98 복사형. ≤0.04 비복사형 7개 중 **6개(시나리오 B에선 7/7)가 무교란** |
| O 층 불일치의 기원 위치 | first_divergence=0으로 밀기 | **F2의 귀속이 붕괴**(교란은 prefill 미접촉) → 死因 아님, **등록 진단 규칙으로 전환** | `forward_extend:807-1038`에 `num_kv_splits` 참조 0건. `output_ids[0]`은 extend 산출 |
| 비교 분모(총 compared) | 라벨/단위 결손으로 96 미만 | **없음**(F2 문장이 96을 전제할 뿐) → **분모 재기술 조건으로 전환** | 907032↔907100 실제 실행: `compared=32`(라벨 2개 MISSING)인데도 동일한 "mismatch==0 → 민감도 미시연" 문장이 그대로 출력됨 |
| C 층 등가류 분모(boot 결손) | gen 파일 1개 결손 | F3 분기1이 **허위 발화** · 그러나 그 시나리오는 F1에서 `BOOT_FAILED`=측정 실패로 폐기되므로 **등록 판정 반전은 아님** | 907032(L1·L2만 존재)에서 스크립트가 `units differing = 0 []` 출력. **실제 L1-L2 C 불일치는 7/32**(독립 재계산) |
| §1.7 C01 기전의 실효 검증 | 실효 축약 구간으로 재계산 | **없음**(기전 유지 ⇒ F3 전제 성립) | split 수 차이 4/48 step 중 **실효 차이 2/48**(k=3,4: L 96 vs TD 64). 강제 6에서는 C01 행이 전 구간 96 고정 ⇒ 그 경로 제거 |

**반전 0건** ⇒ N2 미발화. 위 두 건(prefill 기원, 분모 결손)은 반전을 만들지 못했으므로 **등록 caveat + D-조건**으로 전환했다.

---

## §3 예보 F1–F4의 반증 가능성 (요청 2)

- **F1 — 반증 가능. 단 "두 번째 수치 구성에서도 유지"는 *강화*가 아니라 *약화된 복제*다.** 강제 split은 decode attention을 **행 단위 배치 불변**으로 만들고, O1의 존재 이유("probe split이 bg에 의존하지 않음")를 **구성상 자동 성립**으로 바꾼다. 즉 arm-특이 타이밍이 토큰을 바꿀 수 있는 통로 하나를 X1이 **제거한 뒤** 동치를 재확인하는 설계다. PASS는 907100보다 **더 쉬운 시험의 통과**이고, 반면 FAIL은 매우 정보적이다(비대칭은 정직하다). 등록 문구에 이 비대칭을 명시해야 한다.
- **F2 — 반증 가능하고, 실패 분기가 프로젝트 이익에 **반**한다(정직).** "0/96 → 민감도 상한 정보" 줄은 사후 합리화 통로가 **아니다**: ★금지 줄이 유일한 자기이익 해석("TD가 정말 같다")을 명시적으로 닫았고, 0 분기에서 새로 사는 주장이 0건이다. **다만 두 가지 결함**: (i) 분모 96은 거짓이다(실효 72). (ii) 0 분기의 읽기가 과대하다 — **C 층 8/32 불일치(복사형 C10·C02 포함)는 이미 이 엔진·프롬프트 계열에서 ULP급 섭동이 argmax를 뒤집을 수 있음을 0 GPU로 보여 주고 있다.** 따라서 0/72는 "측정층이 둔하다"보다 "**이 교란이 이 복사형 단위들의 마진을 넘지 못했다**"로 읽는 것이 정확하다. 이 문장을 0 분기에 등록해야 한다.
- **F3 — 반증 가능하나 실질 내용은 거의 없다. 등록상 주장 0건이므로 용인된다.** "정합이지만 확증 아님"은 정직한 라벨이지만, (i) small-n(boot 2/arm)에서 "arm-분리 소멸"은 균등 귀무로도 **P≈0.20**, (ii) 강제 split은 §1.7 기전 외에 **GEMM M 의존 등 다른 배치 의존 경로를 제거하지 못하므로** "남았다"도 가설을 반증하지 못한다, (iii) 제3분기("L-L도 불일치 → 재확인만")가 사실상 흡수 분기다. 세 분기 전부가 **등록상 어떤 claim도 생산하지 않을 때에만** 수용 가능하다. 반대로 제2분기("TD 고유 원인 후보 이관")는 R2C-3과 충돌할 수 있으므로 bs 시퀀스 증거 요구를 붙여야 한다(D9).
- **F4 — 반증 가능. 단 현재 하네스로는 *자동 충족되지 않는다*.** `provenance.txt`는 env를 **기록하지 않는다**(현 sbatch는 job/node/date/commit/src_dirty/nvidia-smi/thread_local_role_patch/harness sha/model·dsm·seed·order·config만 기록). `verdict_rule.txt`·`runtime_source_manifest.sha256`·harness sha는 자동이다. 또 **`x1_prereg/`는 현재 git 미추적**(`git ls-files` 0건)이므로 "이 문서와 같은 커밋"은 아직 거짓이고, `provenance.txt`의 `commit=`은 등록 문서를 핀하지 못한다.

---

## §4 스코프 누출 (요청 3)

누출 위험 문장과 필요한 차단:

1. **F1 PASS 문장**("907100의 scoped 결론이 두 번째 수치 구성에서도 유지된다") — Claim D 선결 #2의 "부분 해소"를 "더 해소"로 읽히게 할 수 있다. 차단: ① X1 수치 구성은 **운영점이 아니다**(운영 기본 = `num_kv_splits=8` + 휴리스틱), ② X1은 동치 시험을 **약화**시킨다(위 F1), ③ §4에 이미 있는 "아무 선결도 닫지 않는다"를 결과 문서에 **문자 승계**.
2. **F2 ≥1 분기**("907100 PASS의 정보량이 '대형 손상 부재'보다 강해진다") — 가장 위험하다. R2C-1/R2C-11/R2C-4·5를 침식할 수 있다. 차단: 민감도는 (a) 불일치가 난 **층·단위**에 대해서만, (b) **decode attention 수치**에 대해서만 시연된다. **prefill은 전 구간 비트 동일**이므로 §2.3이 지목한 "D44 64-SM에서 돈 유일한 비교 대상 = probe prefill"의 민감도는 **X1 후에도 미보정**이다.
3. **F3 분기**("arm-분리가 사라진다", "TD 고유 원인 후보") — R2C-2/R2C-3 침식 경로. 추가로 **C 층 불일치 총수가 8에서 줄어드는 것은 교란의 예상된 부작용**(decode attention이 행 단위 배치 불변이 됨)이며 TD에 대한 어떤 증거도 아니다. 이 문장을 금지 목록에 넣어야 한다.
4. **"성능 판정 0건"은 구조적으로 보장되지 않는다.** 하네스는 요청별 `t_start/pc_start/pc_end`와 텔레메트리를 남기고, 교란은 **성능에 영향을 주는 설정**이다. 누군가 907100↔X1 wall/ITL을 비교하면 n=1·비페어·교차-잡·두 수치 구성이 동시에 다른 **무의미한 성능 수치**가 생산된다. 명시적 금지가 필요하다.
5. **R2C-13 인접**: X1은 ctx 4096·관측 플래그 ON·decode-log-interval 1을 유지하므로 러너 설정 인증과 무관하다(불변).

---

## §5 비교 코드 검증 (요청 4)

- **selftest는 항등식이 아니다**(자기가 검증할 로직을 복사하지 않고 실제 `compare()`를 호출한다). 변이본 4종(`first_div`를 항상 None, `units()`를 빈 dict, sha 검사 제거, phase_o 누락)에 대해 assert가 깨지는 것을 논리적으로 확인했다. **그러나 한 변이본은 통과한다**: `units()`에서 **phase_s만 조용히 누락**시키면 세 assert(0 / 1 / 1)가 모두 성립한다 — 1-토큰 변이는 O에, 입력 불일치는 S에 심어 놓았기 때문이다. 이 구멍을 막는 것은 selftest가 아니라 **실데이터 실행의 `compared=24/label` 값**이다. 그래서 분모 고정이 필수다(D2).
- **`prompt_sha256` 제외 규칙은 조용하지 않다**(`INPUT_MISMATCH=[...]`와 `excluded(input)=N`을 출력한다). 그러나 **분모를 말없이 깎는다**: 전 단위 제외 시 `TOTAL compared = 0 mismatching = 0`이 찍히고, **스크립트가 인쇄하는 해석 문장은 total_cmp==0을 구분하지 않는다.** 실제 교차-잡 실행(`../job_907032 ../job_907100`)에서 라벨 2개가 `MISSING`이어도 같은 해석 문장이 그대로 출력되는 것을 확인했다.
- **C 층 출력에는 허위 0 경로가 있다.** `rows = c_classes(gens) if len(gens)==4 else {}` 때문에 gen 파일이 4개 미만이면 "`units differing = 0 []` / `arm-separated = 0 []`"이 인쇄된다. 907032(L1·L2만 존재)에서 스크립트는 0을 찍었지만 **실제 L1-L2 C 불일치는 7/32**다(독립 재계산). boot 1개가 빠지면 F3 분기1이 파일 결손만으로 발화한다. (등록상 그 시나리오는 F1에서 측정 실패로 폐기되므로 반전은 아니나, 반드시 `NOT-RUN` 표기로 막아야 한다.)
- **C 층이 판정으로 승격될 통로**: 스크립트 자체에는 없다(출력은 진단 문자열). 통로는 **문서 쪽**이다 — F3 제2분기와 "C 불일치 감소" 서술. §6 금지 문장으로 닫는다.
- **`x1_selftest_stdout.txt`의 "baseline vs baseline" 블록은 판정서 §1.7과 일치한다 — 원자료 독립 재계산으로 확인:**

| 항목 | stdout | 내 독립 재계산(gen 파일 직접) | 판정서 §1.7 |
|---|---|---|---|
| C 층 불일치 단위 | 8 `[C00,C01,C02,C10,C14,C16,C19,C24]` | **동일** | 동일 |
| arm-분리 | 3 `[C01,C10,C19]` | **동일** | 동일 |
| first divergence | — | C01@12, C10@1, C19@6 | 12 / 1 / 6 **일치** |
| pairwise | — | L1-L2 **4**, TD1-TD2 **3**, cross 7/7/4/4 | "L-L 4, TD-TD 3, cross 4–7" **일치** |
| S/O 층 | compared 24/label, 0 mismatch | S·O 6쌍 전부 0 | 일치 |

  단 이 4번째 블록은 **자기 자신과의 비교**라 0이 항등식이다(비교기가 실제 교차-잡 차이를 본다는 증거가 아님). **진짜 귀무대조는 이미 존재한다**: `x1_cross_job_compare.py ../job_907032 ../job_907100` → `compared=32, mismatch=0`(노드 gpu42→gpu38, 커밋 `02918e8`→`38c1aca`, 교란 없음). 이것을 등록해야 한다(D7). O 층에는 이 귀무대조가 **없다**(907032에 phase_o 부재).

---

## §6 제출 전 필수 수정 조건 (D1–D11)

**차단(데이터 생성 전 반드시):**

- **D1 — pre-flight를 `MIN_BLOCK_KV=32`로 재계산하고 §1 표·"24/24가 교란된다"를 정정.** 등록 분모를 "**교란 단위 18/24 = 72/96 unit-pair, 구성상 무교란 6/24 = 24/96(S00,S01,S02,S03,S04,S13)**"으로 고정. 초안 8의 칸도 22/24로 정정(이탈 정당성은 유지·강화). `x1_preflight_splits_stdout.txt` 재생성. 판정 기준은 `kv_len_per_split`(= `cdiv(cdiv(L,s),32)*32`, `decode_attention.py:35,98,553`)로 명기.
- **D2 — F2 판정 규칙 고정 3항**: (i) `TOTAL compared unit-pairs == 96`이고 gen 4/4가 있어야 등록 F2를 읽는다. 미만이면 등록 문장은 무효이고 분모·제외 목록을 명시 재기술한다. (ii) **불일치는 `first_divergence ≥ 1`이어야 교란에 귀속된다** — `output_ids[0]`은 extend 산출이고 prefill은 비트 동일(`forward_extend:807-1038`에 `num_kv_splits` 0건)이므로 index 0 불일치는 교란으로 설명되지 않는다(해석 보류 + 원인 조사). (iii) 불일치 수를 **S/O 층별로 분리 보고**한다 — S는 bs=1 고정이고 907032↔907100 귀무대조가 있어 귀속 가능, O는 O1 8/8 confirmed일 때만(그때 probe decode 전 구간 bs=2) 귀속 가능.
- **D3 — 교란 서술 정정**: "`fill_`뿐이므로 재캡처 없음" → "**`MAX_KV_SPLITS` constexpr·stage-1 grid z(8→6)·`attn_logits`/`attn_lse`/cudagraph 버퍼 shape가 함께 6으로 고정되며, 그 구성으로 1회 capture된다. 재캡처는 없고 양 arm 동일하다.**" 또 "**prefill(extend) 경로는 교란되지 않는다**"를 변인 절에 명시.
- **D4 — 두 노브 중 하나를 고른다.** (a) **권고**: `--triton-attention-num-kv-splits 6`만 쓰고 env var를 **뺀다** — 관측 가능(서버 args 덤프 + `SAME_ACROSS_BOOTS` 검사)하고 confound #10이 사라지며, 비용은 **교란 단위 18→17(S05 1개)**뿐이다. (b) 둘 다 유지하려면 "env var의 한계 기여는 1/24 단위이고 시나리오 A/B는 23/24 단위에서 구별 불가"를 등록하고, provenance에 CPU-only 파싱 확인(`get_bool_env_var`는 "true"/"1" 수용, `utils/common.py:336-348`)을 남긴다.
- **D6 — provenance 보강**(현 sbatch는 전부 미기록): `env | grep -E '^(SGLANG|PDMUX|R2C)_'`, `torch.cuda.get_device_properties(0).multi_processor_count`, `nvidia-smi -L`(기존), **PREREG·`x1_preflight_splits.py`·`x1_cross_job_compare.py`·stdout 2개의 sha256**, 2차 진단 실행 명령과 stdout를 job 디렉터리에 저장. **`x1_prereg/`를 제출 전에 커밋**(현재 미추적 ⇒ `commit=`이 등록을 핀하지 못함).
- **D7 — 교차-잡 귀무대조 등록**: `907032 → 907100` S 층 `compared=32, mismatch=0`(노드·커밋 상이, 교란 없음)을 F2의 귀무대조로 **사전 등록**. O 층에는 귀무대조가 없음을 같이 등록.
- **D8 — C 층 허위 0 차단**: gen 4/4 미만이면 `NOT-RUN`을 인쇄하도록 고치고(현재 `0 []` 인쇄, 907032 실증), **F3은 gen 4/4 + F1=PASS일 때만 읽는다**(F1 PASS는 B2의 `request_errors==0`을 통해 C 32/32 완주를 보장한다 — `r2_correctness_check.py:390-398`).
- **D9 — F3 제2분기 제약**: "TD 고유 원인 후보 이관"은 **arm 간 bs 시퀀스 동일성 증거**를 함께 요구한다(강제 split은 decode attention만 배치 불변으로 만들고 GEMM M 의존 등은 남긴다). 그리고 **C 불일치 총수 감소는 교란의 예상 부작용**임을 등록.

**비차단(정확성·권고):**

- **D5(선택, 검정력 상향)** — 강제값을 **2**로 하면 실효 기준 **24/24**가 교란되어 4-gram ≤0.04 비복사형 7단위 전부가 포함된다(3은 20/24). 비용: attention 커널 z-grid 2, 긴 행의 블록당 작업 ~4배 — Zamba2-2.7B는 54층 중 attention이 9층(`hybrid_layer_ids`)뿐이고 wall 상한 1.25 GPU-h에 실측 0.16이므로 **실현 가능**하다. 단 운영점에서 더 멀어지고 타이밍 변화가 커져 C/O 체류가 더 달라진다 — 6을 유지하고 D1의 정정된 분모를 받아들이는 것도 허용 가능한 선택이다. **둘 중 무엇을 택해도 결과 문서에는 "무교란 단위 수와 그 단위들의 4-gram 구간"을 병기**한다.
- **D10** — pre-flight의 `bg_lo=15`를 기록값 17로(결론 불변: `mks1`이 양쪽 다 8로 포화).
- **D11** — "`prompt_tokens`(입력값, 결과 아님)"을 "응답 `meta_info`에서 읽었으나 입력 텍스트(sha256 동일)와 토크나이저의 결정함수이며 4 boot 동일"로 정정하고, **cross-job 비교에 `prompt_tokens` 동일성 검사를 추가**(현 비교기는 sha만 본다; in-job 체커의 `INPUT_IDENTITY`는 boot 간만 본다).

---

## §7 실행 후 등재 시 필수 병기 / 인용 금지 (문자 그대로 승계)

**필수 병기 X1P-1…7**

- **X1P-1** "X1의 수치 구성(decode KV-split 강제 고정, `triton_attention_num_kv_splits=6`)은 **운영점이 아니다**. 운영 기본은 8 + 커널 휴리스틱이다. X1의 within-job PASS는 운영 수치 구성의 correctness를 확장 인증하지 않는다."
- **X1P-2** "X1의 교란은 **decode attention에만** 걸린다. prefill(extend) 경로는 비트 동일이므로(`forward_extend`에 `num_kv_splits` 참조 0건), 판정서 §2.3이 '비교 대상 중 D44 64-SM에서 돈 유일한 계산'으로 지목한 **probe prefill의 민감도는 X1 이후에도 미보정**이다."
- **X1P-3** "X1의 교란은 비교 단위 24개 중 **18개(실효 기준, `MIN_BLOCK_KV=32` 양자화 반영)**에만 도달한다. S00·S01·S02·S03·S04·S13 **6개(= 96 unit-pair 중 24개)는 구성상 비트 동일**이어서 불일치를 만들 수 없다. 교란된 단위는 전부 4-gram ≥0.98 복사형이고, ≤0.04 비복사형 7개 중 6개가 무교란이다."
- **X1P-4** "X1의 within-job 동치 시험은 907100보다 **약하다**. 강제 split은 decode attention을 행 단위 배치 불변으로 만들고 O1의 전제를 자동 성립시켜, arm-특이 타이밍이 토큰을 바꿀 수 있는 경로 하나를 제거한 뒤 동치를 확인한다."
- **X1P-5** "교차-잡 비교는 **진단 전용**이다. 귀무대조는 `907032→907100` S 층 `compared=32 / mismatch=0`(노드·커밋 상이, 교란 없음)이며, **O 층에는 귀무대조가 없다**(907032에 O 층 부재). 불일치는 `first_divergence ≥ 1`일 때만 교란에 귀속한다."
- **X1P-6** "C 층 수치는 등가류 서술뿐이고 어떤 게이트·기전·정책 판정에도 쓰지 않는다. boot 2개/arm에서 'arm-분리 소멸'은 균등 귀무로도 P≈0.20이다. 강제 split은 decode attention의 배치 의존성을 제거하므로 **C 불일치 총수 감소는 교란의 예상된 부작용**이다."
- **X1P-7** "X1은 성능을 측정하지 않는다. 새 성능 판정 0건. Claim D 등급 불변(미검증), 선결 #1·#2·#3·#4a·#4b·#5 상태 불변(#2는 여전히 S/O 프로토콜 부분 해소). R2C-1…16·P-1…7·HE0·정책 순위·stake #1 불변."

**인용 금지 X1C-1…9**

- **X1C-1** "X1이 correctness를 더 넓게 인증했다" / "두 수치 구성에서 확인됐으므로 TD≡legacy가 강해졌다". 허용형: X1P-1 + X1P-4.
- **X1C-2** "비교기(측정층)의 민감도가 보정됐다". 허용형: "불일치가 난 층·단위에서 decode attention 수치 교란에 대한 민감도의 **하한**이 시연됐다"(+X1P-2, X1P-3).
- **X1C-3** "X1 0 mismatch는 TD와 legacy가 정말 같다는 추가 증거다". (교란이 걸렸는데 안 움직인 것과 검출력 부족은 이 설계로 구별되지 않는다. 더해서 C 층 8/32 불일치는 이 엔진·프롬프트 계열에서 ULP급 섭동이 argmax를 뒤집을 수 있음을 이미 보여 준다 ⇒ 0은 "이 복사형 단위들의 마진을 넘지 못했다"로 읽는다.)
- **X1C-4** "X1이 24/24 비교 단위를 교란했다" / "무교란 단위는 없다". (실효 기준 18/24, 무교란 6/24.)
- **X1C-5** "X1은 `fill_` 값 하나만 바꿨다". (`MAX_KV_SPLITS` constexpr·grid z·버퍼 shape가 함께 바뀐다.)
- **X1C-6** "C01 arm-분리 소멸이 §1.7 기전을 확증한다" / "X1이 C 층 불일치 원인을 규명했다". (R2C-3 유지. 배치 의존 경로가 남는다.)
- **X1C-7** "C 층 결과가 동시 부하 동치(R2C-2)를 완화한다".
- **X1C-8** "X1이 Claim D 선결을 닫았다/진전시켰다" 또는 "r2_eval 설정·D16/24/34·generic/hybrid·다른 모델·TP≥2로 확장됐다".
- **X1C-9** X1 아티팩트(또는 907100↔X1 교차)에서 **어떤 지연·스루풋·GPU-h 비교 수치도 인용 금지**. n=1, 비페어, 교차-잡, 두 수치 구성이 동시에 다르다.

---

## §8 자기 적용 (게이트 #113)

- D1·D2·D3·D9·D11은 **문서/스크립트 편집만**(GPU 0). D6·D7·D8은 sbatch·비교기 편집 + **이미 존재하는 데이터 재실행**(GPU 0; 907032↔907100 교차-잡은 내가 실제로 돌려 출력까지 확인했다). D4(a)는 인자 1개 삭제. D5는 스칼라 1개 변경 + pre-flight 재실행. **모든 처방이 0 GPU이거나 등록된 0.16 GPU-h 안에서 수행 가능하다.**
- 직전 회차(`audit_r2corr_2026-09-11/VERDICT.md`)의 지적을 상수로 승격하지 않았다: §1.6의 민감도 미보정, §1.7의 C01 기전, §2.3의 D44 귀속, §7.3의 X1 초안을 **각각 원자료·커널 소스로 독립 재검증**했고 그 과정에서 §7.3 초안의 값 8이 실효 기준으로 **2/24만 교란**(초안 자신의 진단보다 더 나쁨)이라는 점, §1.7 기전이 실효 기준으로 **2/48 step만** 성립한다는 점을 새로 확정했다.
- 등급 인플레 점검: 死因을 만들려고 7개 표면을 끝까지 밀었고 **반전 0건**이었다. 따라서 `NO-GO`를 내지 않는다. 반대로 거짓 수치·허위 0·미기록 provenance가 실재하므로 무조건 `GO`도 내지 않는다. **`GO-with-caveats` + 차단 조건 D1–D4·D6–D9**가 정확한 판정이다.

---

## §9 관련 파일 (절대경로)

- 등록 문서: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/x1_prereg/PREREG_X1_SENSITIVITY_2026-09-12.md`
- 정정 대상 산술: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/x1_prereg/x1_preflight_splits.py` (+ `x1_preflight_splits_stdout.txt`)
- 정정 대상 비교기: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/x1_prereg/x1_cross_job_compare.py` (+ `x1_selftest_stdout.txt`)
- 판정 규칙(무수정 유지, sha 현재 `ec355e17…`로 907100 provenance와 일치): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness_check.py`
- provenance/env 기록을 고칠 곳: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness.sbatch` (provenance 블록 · `server_cmd`)
- 결정적 커널 근거: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/layers/attention/triton_ops/decode_attention.py:35,98,206,553` 및 `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/layers/attention/triton_backend.py:110-143,185-235,286-300,447-468,717,807-1038,1039-1103,1270-1319`
- 귀무대조 데이터: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907032/`, `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907100/`

단일 문장 요약: **실행해도 된다(`GO-with-caveats`). 단 "24/24 교란"은 거짓이고 실제는 18/24이며 무교란 6개가 하필 가장 민감한 단위들이다 — D1–D4·D6–D9를 데이터 생성 전에 반영하지 않으면 X1은 "검출력 하한"과 "구성상 항등식"을 구별할 수 없는 결과를 낳는다.**
