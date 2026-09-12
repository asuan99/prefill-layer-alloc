# 사전등록 **rev2** — X1: R2 correctness 측정층 민감도 양성대조 (+ C01 기전 판별)

작성 2026-09-12 (메인 세션). **GPU 미지출 상태에서 고정**한다(이 트랙 누적 0.28 GPU-h, X1 전 0).
rev1([`PREREG_X1_SENSITIVITY_2026-09-12.md`](PREREG_X1_SENSITIVITY_2026-09-12.md))을 **대체**한다 —
rev1은 제출 전에 claims-auditor 규칙층 감사를 받아 `GO-with-caveats` + 차단 조건 **D1–D4·D6–D9**를
받았고, rev2가 그 조건을 반영한 판본이다. 감사 판정서 = [`VERDICT_x1_rules_2026-09-12.md`](VERDICT_x1_rules_2026-09-12.md).
근거 판정서(직전 회차) = `../audit_r2corr_2026-09-11/VERDICT.md` §1.6·§1.7·§2.3·§7.3.

## 0. rev1에서 무엇이 틀렸는가 (기록)

rev1은 "교란됨"을 **행별 split 수**로 판정해 강제값 6에서 **24/24 교란**이라고 적었다. **거짓이다.**
커널은 split 수를 한 단계 더 양자화한다:

```
decode_attention.py:35   _MIN_BLOCK_KV = 32
decode_attention.py:98   kv_len_per_split = cdiv(cdiv(seq_len, kv_splits), 32) * 32
decode_attention.py:553  stage-2 combine이 같은 식을 다시 계산
:110, :560               두 단계 모두 `if split_kv_end > split_kv_start`로 빈 split을 건너뛴다
```

⇒ split 수가 달라도 `kv_len_per_split`이 같으면 경계·비어있지 않은 split 개수·combine 순서가
전부 같고 **비트 단위로 동일**하다. 실효 기준으로 재계산한 결과(`x1_preflight_splits.py` rev2,
907100의 실제 `prompt_tokens`):

| 설정 | 교란되는 비교 단위(실효) | 무교란(구성상 비트 동일) |
|---|---|---|
| 판정서 §7.3 초안(static env var만 ⇒ cap 8) | **2/24** | 22 — **O 탐침 8/8 전부 포함** |
| rev1 계획(static + cap 6) | **18/24** | 6 — S00–S04, S13 |
| **rev2 계획(cap 2, CLI만)** | **24/24** | **0** |
| cap 3 / 4 / 5 / 6 / 7 | 20 / 17 / 17 / 17 / 15 | 4 / 7 / 7 / 7 / 9 |

rev1의 "초안대로면 양성대조가 사실상 항등식"이라는 판단 자체는 **더 강하게 옳았다**(16/24 → 실제 22/24).
그러나 rev1이 고른 6도 **무교란 6개가 하필 4-gram ≤0.04 비복사형 7개 중 6개**여서, 뒤집힐 확률이
가장 높은 단위에 검정력이 0이었다. rev2는 이를 **cap 2**로 해소한다(감사 조건 D1·D5).

## 1. 실험 설명 (통제 / 변인 / 예상 / 진행 가부)

### 통제 요인 (907100과 **같게**)
모델 Zamba2-2.7B · TP=1 · `pdmux_r2.yml` · `PDMUX_R2_POLICY=fixed` · `R2C_DSM=44` ·
greedy(T=0) · `ignore_eos` · 고정 `max_new` · `--random-seed 1` · ctx 4096 · cudagraph-ON(decode,
prefill eager) · `--attention-backend triton` · `--disable-radix-cache` · `--chunked-prefill-size -1`
· `--disable-overlap-schedule` · `--decode-log-interval 1` · boot 순서 `L TD L TD` + warmup ·
프롬프트 집합/seed/stagger(client 불변) · **판정 규칙 = `r2_correctness_check.py` 무수정**
(rule v2, sha `ec355e17…` = 907100 provenance와 동일).

### ★통제 요인 추가 (2026-09-12, 데이터 생성 전 · 통제를 **강화**하는 추가)
**엔진 소스는 907100이 돌린 것과 바이트 동일해야 한다.** 같은 세션에서 engine-porter가
`src/multiplex/{controller,multiplexing_mixin,profile}.py`를 수정 중이고(Claim E 불일치 3건 반영),
하네스는 job 시작 시 `sync_engine_tree.sh`로 **저장소의 현재 파일**을 dev tree에 설치한다. 그
상태로 제출하면 교란 노브와 **엔진 소스 변경이 동시에** 바뀌어(confound #10) F2의 불일치를
교란에 귀속할 수 없다 — F1(within-job)은 양 arm이 같은 소스라 영향 없지만 F2는 무효가 된다.

그래서 제출 절차를 다음으로 고정한다:
1. engine-porter 작업을 먼저 끝내고 커밋한다(X1 제출 전 워킹트리 확정).
2. `git checkout 38c1aca -- workspace/engine-port/src/multiplex`로 **907100이 돌린 소스**를 트리에
   복원한 뒤 제출한다(job 완료까지 이 경로를 건드리지 않는다).
3. 결과 검증: X1의 `runtime_source_manifest.sha256`의 multiplex 항목이 `job_907100/
   runtime_source_manifest.sha256`과 **전부 일치**해야 한다. 불일치하면 **F2를 읽지 않는다**
   (측정 실패로 기록, 게이트 실패 아님).
4. 검증 후 `git checkout HEAD -- workspace/engine-port/src/multiplex`로 최신 소스를 되돌린다.

### 변인 요인 — **단 하나의 노브**
네 boot 전부에 `--triton-attention-num-kv-splits 2`를 준다. **env var는 쓰지 않는다**(감사 조건 D4(a)).
- 이유: CLI 인자는 **관측 가능**하다 — 서버 args 덤프에 찍히고, checker의 `SAME_ACROSS_BOOTS`
  (`r2_correctness_check.py:119-122`)가 boot 간 불일치를 자동 FAIL로 잡는다. env var는 효과가
  계측 없이 관측 불가였고, 두 노브 동시 주입은 confound #10(변수 동시 변경)이었다.
- cap 2에서는 휴리스틱 경로만으로도 **모든 비교 행의 split 수가 2**가 된다(`mks1`·`mks2` 둘 다
  cap에 걸린다) ⇒ static env var는 **불필요**하고, 교란은 24/24로 최대가 된다.
- ★**이 노브가 바꾸는 것의 정확한 목록**(감사 조건 D3 — "`fill_`뿐"은 거짓이었다):
  `self.max_kv_splits`(`triton_backend.py:114`)가 2가 되어 ① stage-1 커널 **grid z축**
  `(batch, head, MAX_KV_SPLITS)`(`decode_attention.py:206`) 8→2, ② stage-2 combine 루프 상한
  `MAX_KV_SPLITS`(`:555`), ③ `attn_logits (bs,32,K,160)`·`attn_lse (bs,32,K)`
  (`triton_backend.py:290,295`)과 cudagraph 버퍼(`:450,455,461-463`) shape가 함께 2로 고정된다.
  **재캡처는 없다** — 그래프가 처음부터 2-shape로 1회 capture된다. 양 arm에 동일하게 걸린다.
- ★**prefill(extend)은 전혀 교란되지 않는다.** `forward_extend`(`triton_backend.py:807-1038`)에
  `num_kv_splits`/`max_kv_splits` 참조가 0건이고 extend 경로는 `num_kv_splits=None`이다(`:357,376,424`).
- ★**부작용 공시**: checker의 O1 문턱 `probe_ptok > (K-1)*(bg_ptok+bg_max_new)`이 K=2에서
  `> 177`로 느슨해진다(전 probe 1437+ 통과). 그 문턱의 목적("probe split 수가 bg에 의존하지 않음")은
  cap 2에서 **구성상 자동 성립**한다 — 그래서 O1은 X1에서 거의 공허해진다. 이것은 아래 F1의
  비대칭(이 시험이 907100보다 **약하다**)의 일부다.

### 예상 (아래 §2 F1–F4에 문자로 고정)
### 진행 가부
감사 판정 `GO-with-caveats` + 차단 조건 D1–D4·D6–D9 반영 완료 ⇒ **제출 가능**. 단 X1은
**어떤 Claim D 선결도 닫지 않는다**(#1·#2·#3·#4a·#4b·#5 상태 불변, #2는 여전히 S/O 부분 해소).

## 2. 사전 등록 예보 F1–F4와 판정 규칙

- **F1 (1차 게이트, within-job)**: `PASS`.
  - ★**비대칭 공시**: 강제 split은 decode attention을 **행 단위 배치 불변**으로 만들어, arm-특이
    타이밍이 토큰을 바꿀 수 있는 경로 하나를 **제거한 뒤** 동치를 확인한다. 즉 **PASS는 907100보다
    더 쉬운 시험의 통과**이고(`X1C-1` 금지 대상), **FAIL은 매우 정보적**이다(즉시 engine-porter 이관,
    "correctness가 수치 구성에 의존"으로 등재, 907100의 scoped 결론을 **약화**).
  - `NO_VERDICT_INFRA`/`BOOT_FAILED`/`CLIENT_FAILED` = **측정 실패**(게이트 실패 아님, 교훈 21).
    수리 후 1회 재실행까지 허용, 그 이상은 사용자 승인.
- **F2 (양성대조, 2차 진단)**: 96 unit-pair 중 **불일치 ≥ 1**. 판정 규칙 3항(감사 조건 D2):
  1. **분모 고정**: `TOTAL compared unit-pairs == 96` **이고** gen 4/4가 있어야 이 F2 문장을 읽는다.
     미만이면 등록 문장은 **무효**이고 분모·제외 목록을 명시 재기술한다(비교기가 자동 경고한다).
  2. **귀속 규칙**: 불일치는 `first_divergence ≥ 1`일 때만 교란에 귀속한다. `output_ids[0]`은
     prefill(extend) 산출이고 교란은 extend를 건드리지 않으므로, index 0 불일치는 **교란으로
     설명되지 않는다**(해석 보류 + 별도 조사).
  3. **층 분리 보고**: S(bs=1 고정, 교차-잡 귀무대조 있음)와 O(귀무대조 없음, O1–O4 confirmed일
     때만 귀속)를 분리해 적는다.
  - `0/96` 해석: **민감도 미시연**으로만 등재한다. ★**"TD가 정말 같다"의 추가 증거로 쓰는 것 금지**
    (`X1C-3`). 또한 "측정층이 둔하다"로 단정하지도 않는다 — 정확한 문장은 "**이 교란이 이 복사형
    단위들의 argmax 마진을 넘지 못했다**"다(C 층 8/32 불일치가 이미 ULP급 섭동이 argmax를 뒤집을
    수 있음을 보여 주므로).
  - **귀무대조(D7, 사전 등록)**: `907032 → 907100` S 층 **compared=32, mismatch=0**(노드 gpu42→gpu38,
    커밋 `02918e8`→`38c1aca`, 교란 없음) — `x1_selftest_stdout.txt` 블록 B. **O 층에는 귀무대조가
    없다**(907032에 O 층 부재).
- **F3 (C01 기전, 진단 전용)**: C01의 arm-분리가 **사라진다**.
  - 읽기 조건: gen 4/4 **이고** F1=`PASS`일 때만 읽는다(D8).
  - 사라짐 → §1.7 가설과 **정합**, **확증 아님**(강제 split은 여러 기전을 동시에 제거).
  - 남고 L-L=TD-TD=0·cross>0 → TD 고유 원인 **후보**로 이관하되, **arm 간 bs 시퀀스 동일성 증거를
    함께 요구**한다(D9 — 강제 split은 decode attention만 배치 불변으로 만들고 GEMM M 의존 등은 남긴다).
  - small-n 경고: boot 2개/arm에서 "arm-분리 소멸"은 균등 귀무로도 **P≈0.20**.
  - ★**C 불일치 총수 감소는 교란의 예상된 부작용**이며 TD에 대한 증거가 아니다(`X1C-6`).
- **F4 (provenance, D6)**: job 디렉터리에 다음이 남는다 — `env | grep -E '^(SGLANG|PDMUX|R2C)_'`,
  `multi_processor_count`, `nvidia-smi -L`, harness sha256(기존), **이 사전등록·preflight·비교기·
  stdout 2개의 sha256**, `verdict_rule.txt`, `runtime_source_manifest.sha256`, 2차 진단 실행 stdout.
  빠지면 F1을 인용하지 않는다. **이 디렉터리는 제출 전에 커밋한다**(rev1 시점엔 미추적이어서
  `commit=`이 등록을 핀하지 못했다).

## 3. 비교 단위와 분석 코드 (데이터 생성 전 고정)

- 1차: job 내부 `r2_correctness_check.py`(무수정) → `verdict.txt`.
- 2차 진단: `x1_cross_job_compare.py ../job_907100 ../job_<X1>` — 4 label × 24 단위 = 96 pair.
  입력은 **`prompt_sha256` + `prompt_tokens` 둘 다** 검사하고 불일치는 제외·명시 보고(D11).
  C 층은 gen 4/4 미만이면 **`NOT-RUN`**을 인쇄한다(D8 — rev1은 907032에서 `0 []`이라는 **허위 0**을
  인쇄했고, 실제 L1-L2 C 불일치는 7/32였다). selftest 6케이스(동일/ S·O 1토큰 변이/ sha 불일치/
  `prompt_tokens` 불일치/ 부분 job) 통과 기록 = `x1_selftest_stdout.txt` 블록 A.
- 산술 전제 공시: `device_core_count=108`(= `multi_processor_count`, green ctx 무관),
  `num_head=num_kv_head=32`, bg 행 길이 `[17, 97, 177]`(기록값 17 — rev1의 15는 오기, 결론 불변),
  길이는 907100의 `prompt_tokens`(서버 `meta_info`에서 읽었으나 입력 텍스트[sha 56/56 동일]와
  토크나이저의 결정함수이고 4 boot 전부 동일 — 결과값이 아니다).

## 4. 비용·선결

≈0.16 GPU-h(907100 실측 9m34s와 동일 구조, wall 상한 1.25). cap 2는 attention 커널 병렬도를
낮추지만 Zamba2-2.7B는 54층 중 attention이 9층(`hybrid_layer_ids`)이라 wall 여유가 충분하다.
이 트랙 누적 0.28 → **≈0.44 GPU-h**(longctx 15.42 GPU-h 장부와 별개). **Claim D 선결 0건 닫힘.**

## 5. 실행 후 필수 병기 / 인용 금지 (감사 판정서 §7에서 **문자 그대로 승계**)

필수 병기 **X1P-1…7**, 인용 금지 **X1C-1…9** 전문은 [`VERDICT_x1_rules_2026-09-12.md`](VERDICT_x1_rules_2026-09-12.md) §7에
있고, 그 문장들을 결과 등재 시 그대로 옮긴다. 요지:
- X1P-1 X1의 수치 구성은 **운영점이 아니다**(운영 기본 = cap 8 + 휴리스틱).
- X1P-2 교란은 **decode attention에만** 걸린다 ⇒ §2.3이 지목한 **probe prefill의 민감도는 미보정**.
- X1P-3 교란 도달 범위와 무교란 단위 수를 **4-gram 구간과 함께** 병기한다(rev2 계획 = 24/24, 0).
- X1P-4 X1의 within-job 동치 시험은 907100보다 **약하다**.
- X1P-5 교차-잡은 진단 전용, 귀무대조는 S 층만, 귀속은 `first_divergence ≥ 1`.
- X1P-6 C 층은 어떤 판정에도 쓰지 않는다(P≈0.20, 총수 감소는 예상 부작용).
- X1P-7 성능 판정 0건 · Claim D 등급 불변 · R2C-1…16·P-1…7·HE0·정책 순위·stake #1 불변.
- X1C-9 **907100↔X1의 어떤 지연·스루풋 수치도 인용 금지**(n=1·비페어·교차-잡·수치 구성 상이).

## 6. 고정 확인

- `x1_preflight_splits_stdout.txt`(rev2 재생성) · `x1_selftest_stdout.txt`(A selftest / B 귀무대조 /
  C 항등 sanity) · 이 문서 · 감사 판정서 전사본이 **같은 커밋**에 들어간다.
- 데이터 생성 이후 F1–F4·비교 단위·해석표를 수정하지 않는다. 필요하면 rev3을 만들고 이유를 적는다.
