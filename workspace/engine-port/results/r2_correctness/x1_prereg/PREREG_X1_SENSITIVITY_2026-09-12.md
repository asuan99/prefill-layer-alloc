> ⚠️ **SUPERSEDED (2026-09-12, 제출 전)** — claims-auditor 규칙층 감사가 이 판본의 핵심 수치를
> 반증했다(`MIN_BLOCK_KV=32` 양자화를 빠뜨려 "강제 6 ⇒ 24/24 교란"이라고 적었으나 실효값은
> **18/24**, 무교란 6개가 하필 가장 민감한 비복사형 단위였다). 유효 판본은
> [`PREREG_X1_SENSITIVITY_2026-09-12_rev2.md`](PREREG_X1_SENSITIVITY_2026-09-12_rev2.md),
> 판정서는 [`VERDICT_x1_rules_2026-09-12.md`](VERDICT_x1_rules_2026-09-12.md). 이 파일은
> **감사 전 상태의 기록**으로만 보존한다(GPU는 이 판본으로 한 번도 돌지 않았다).

# 사전등록 — X1: R2 correctness 측정층 민감도 양성대조 (+ C01 기전 판별)

작성 2026-09-12 (메인 세션). **GPU 미지출 상태에서 작성·고정**한다. 이 문서가 고정된
뒤에만 job을 제출한다. 근거 판정서 = `../audit_r2corr_2026-09-11/VERDICT.md` §1.6·§1.7·§7.3
(claims-auditor, job 907100 `CONFIRMED(scoped)`).

## 0. 이 실험이 답하려는 질문 (그리고 답하지 않는 것)

job 907100의 `PASS`는 "S16 순차 + O8 중첩에서 true-dual과 legacy의 greedy 토큰이 같다"였다.
판정서 §1.6이 남긴 구멍은 **그 같음이 얼마나 강한 진술인지 모른다**는 것이다 — O 탐침 출력
8/8이 passage를 그대로 베낀 복사형이어서 **비교기의 민감도가 보정되지 않았다**. 민감도가
낮으면 `PASS`는 "argmax를 뒤집을 만큼의 대형 상태 손상이 없었다"만 인증한다.

X1은 **정답이 바뀌어서는 안 되는 수치 교란**(decode attention의 KV-split 축약 분할 수)을
네 boot 전부에 똑같이 걸고,
1. 같은 조건의 within-job 동치 게이트가 **여전히 PASS인지**(= 2번째 수치 구성에서의 correctness),
2. 907100 대비 **토큰이 한 번이라도 뒤집히는지**(= 비교기가 축약순서급 교란을 감지하는가)
를 본다.

**답하지 않는 것**: 성능(아무것도 재지 않는다) · Claim D 등급 · 동시 부하(C) 동치 ·
D16/24/34 · generic/hybrid · 다른 모델 · TP≥2. 인용 금지 R2C-1…16과 필수 병기 P-1…7은
그대로 유효하며 X1은 거기에 아무 완화도 주지 않는다.

## 1. 실험 설명 (통제 / 변인 / 예상 / 진행 가부)

### 통제 요인 (907100과 **같게** 유지)
- 모델 Zamba2-2.7B, TP=1, `pdmux_r2.yml`, `PDMUX_R2_POLICY=fixed`, `R2C_DSM=44`.
- greedy(`temperature=0`) · `ignore_eos` · 고정 `max_new` · `--random-seed 1` · ctx 4096.
- cudagraph ON(decode), prefill eager. `--attention-backend triton`, `--disable-radix-cache`,
  `--chunked-prefill-size -1`, `--disable-overlap-schedule`, `--decode-log-interval 1`.
- boot 순서 `L TD L TD`(= null control 2쌍), warmup boot 1회.
- 프롬프트 집합·seed·stagger: 하네스 client 불변(sha256 동일).
- **판정 규칙**: `r2_correctness_check.py` **무수정**(rule v2). within-job 판정 권한은 전적으로
  이 checker에 있고, 이 문서는 그 규칙을 바꾸지 않는다(checker sha256이 provenance에 기록됨).

### 변인 요인 (단 하나의 축)
네 boot **전부**에 decode attention의 KV-split 수를 **6으로 고정**한다:
- `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true`
  (`triton_backend.py:111-113` 읽기, `:202-205`에서 `num_kv_splits.fill_(max_kv_splits)`)
- `--triton-attention-num-kv-splits 6` (`server_args.py:648`, argparse `:5546-5551`)
- 이 교란은 cudagraph와 호환된다: replay 경로(`triton_backend.py:717`)가 매 replay에 같은
  헬퍼를 호출하므로 고정값이 replay에도 적용된다. `fill_`뿐이므로 그래프 재캡처는 없다.
- arm 간 비대칭이 없다(L과 TD가 같은 값을 받는다) ⇒ within-job 동치 게이트의 의미는 보존된다.

### ★판정서 초안(X1)에서의 **의도적 이탈**과 그 이유
판정서 §7.3 초안은 `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true` **만** 추가하라고 적었다.
그 경우 고정값 = 서버 기본 `max_kv_splits=8`이다. 그런데 커널 휴리스틱을 907100의 실제
`prompt_tokens`에 대입하면(`x1_preflight_splits.py`, 산술 재구성):

| 강제값 | 교란되지 않는 비교 단위 |
|---|---|
| **8**(초안) | **16/24** — `S07–S12,S14,S15` **+ O 탐침 8/8 전부** |
| 7 / 6 / 5 | 0/24 |
| 4 | 5/24 |

즉 초안대로 하면 **D44를 실제로 밟는 유일한 층(O 탐침)이 전부 무교란**이어서 양성대조가
**항등식**이 된다(de-confound 교훈 9의 20번째 재발 후보). 긴 행은 휴리스틱이 이미 8로
포화하기 때문이다. 그래서 중간값 **6**을 쓴다 — 짧은 행(휴리스틱 3–4)과 긴 행(8) **양쪽**이
움직이고, 24/24가 교란된다. 비용·부팅 수·판정 규칙은 초안과 동일하다.

산술 전제(측정 아님): `device_core_count=108`, `num_head=num_kv_head=32`(Zamba2-2.7B
config), 길이 = 907100이 기록한 `prompt_tokens`(입력값, 결과 아님) + decode step.

## 2. 비교 단위와 분석 코드 (데이터 생성 전 고정)

- **1차(within-job, 게이트)**: job 내부 `r2_correctness_check.py` → `verdict.txt`.
  S 16 + O 8의 TD vs L 교차 4쌍, null control L-L·TD-TD 포함. 규칙 v2 그대로.
- **2차(cross-job, 진단 전용)**: `x1_cross_job_compare.py job_907100 job_<X1>`.
  boot label 4개 × 단위 24개 = **96 unit-pair**. 각 단위는 **`prompt_sha256` 일치를 먼저
  검사**하고 불일치면 `INPUT_MISMATCH`으로 **제외**(하네스 결함 H1보다 엄격). 자기검사
  `--selftest` = {동일→0, 1토큰 변이→검출, 입력 불일치→제외} 통과 확인됨.
  C 층은 같은 스크립트가 등가류만 출력하고 **어떤 판정에도 쓰지 않는다**.

## 3. 사전 등록 예보 (F1–F4)와 결과 해석표

- **F1 (1차 게이트)**: within-job 판정 = `PASS`.
  - `PASS` → 907100의 scoped 결론이 두 번째 수치 구성에서도 유지된다(스코프는 여전히 S/O).
  - `FAIL` → **907100의 결론을 약화시키는 결과**다. TD≠L을 교란 하에서 본 것이므로
    engine-porter로 즉시 이관하고, 정본에 "correctness는 수치 구성에 의존"으로 등재한다.
    이 경우 X1을 "실패한 실험"으로 적지 않는다(교훈 21).
  - `NO_VERDICT_INFRA`/`BOOT_FAILED`/`CLIENT_FAILED` → **측정 실패**이며 게이트 실패가
    아니다. 원인 수리 후 1회 재실행(추가 0.16 GPU-h)까지만 허용하고, 그 이상은 사용자 승인.
- **F2 (양성대조, 2차 진단)**: 96 unit-pair 중 **불일치 ≥ 1**.
  - ≥1 → 비교기가 축약순서급 교란에 민감함이 **시연**됐다. 907100 `PASS`의 정보량은
    "대형 손상 부재"보다 강해진다. 단 민감도의 **하한**일 뿐이며 정량 보정은 아니다.
  - 0/96 → **판정서 §1.6 문장을 그대로 유지·강화해 등재**: "PASS가 배제하는 것은 argmax를
    뒤집을 만큼의 상태 손상뿐이다." 이것은 측정 실패가 아니라 민감도 상한 정보다.
    ★0/96을 "TD가 정말 같다"의 추가 증거로 쓰는 것은 **금지**(교란이 걸렸는데 아무것도
    안 움직였다면 검출력이 낮다는 뜻이고, 그 두 해석은 이 설계로 구분되지 않는다).
  - **교란이 실제로 걸렸는지의 한계**: env var의 효과는 계측 없이 관측 불가다. 서버 args
    덤프에 `triton_attention_num_kv_splits=6`이 찍히고 sbatch가 env를 provenance에 남기는
    것까지만 확인된다. env가 무력했을 경우(시나리오 B = cap만 6)에도 예보 대비 **17/24**가
    여전히 교란되므로(`x1_preflight_splits.py` 하단 표) F2의 해석은 유지된다.
- **F3 (C01 기전, 진단 전용)**: C01의 arm-분리(`{L1,L2}≠{TD1,TD2}`)가 **사라진다**.
  - 사라짐 → 판정서 §1.7의 "triton KV-split 휴리스틱 × arm-특이 타이밍" 가설과 **정합**.
    고정 splits는 휴리스틱 자체를 끄므로 여러 기전이 동시에 제거된다 ⇒ **확증 아님**,
    "가설과 정합하며 배타적 확증은 아님"으로만 등재.
  - 남고 L-L=TD-TD=0·cross>0 → TD 고유 원인 후보로 engine-porter 이관.
  - 그 외(L-L도 불일치) → 907032·907100과 같은 구조(이 엔진에서 C 층 토큰 동일성은
    **정의되지 않는 양**)로 재확인만 하고 끝낸다.
  - C 층으로는 어떤 게이트 판정도, 어떤 정책·성능 주장도 하지 않는다(R2C-3).
- **F4 (provenance)**: 새 sbatch sha256·env 기록·`runtime_source_manifest.sha256`·
  `verdict_rule.txt`가 job 디렉터리에 남는다. 빠지면 F1을 인용하지 않는다(게이트 #157 계열).

## 4. 비용·선결

- 4 boot + warmup, 907100 실측 9m34s(0.16 GPU-h)와 동일 구조 ⇒ **≈0.16 GPU-h**, 상한
  1.25 GPU-h(wall). 이 트랙 누적은 0.28 → **≈0.44 GPU-h**가 된다(longctx 15.42 GPU-h 장부와 별개).
- 재실행 허용 = 측정 실패 시 1회. 그 외 추가 구매는 사용자 승인.
- 이 실험은 Claim D 선결 중 **아무것도 닫지 않는다**. #2(부분 해소)의 **해석 강도**와
  C 층 기전 후보에만 기여한다.

## 5. 고정 확인

- `x1_preflight_splits.py` 출력 = `x1_preflight_splits_stdout.txt`(이 문서와 같은 커밋).
- `x1_cross_job_compare.py --selftest` 통과 기록 = `x1_selftest_stdout.txt`.
- 데이터 생성 이후 이 문서의 F1–F4·비교 단위·해석표를 수정하지 않는다. 수정이 필요하면
  **새 문서**로 버전을 올리고 이유를 적는다.
