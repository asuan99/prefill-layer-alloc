# G-2 negative control — pre-registration (2026-08-07, experiment-runner)

## 0. 왜

Gate 2 rev4 본 캠페인(job 875346, Granite `ibm-granite/granite-4.0-h-micro-base`)에서
G-2(동시 정확성 게이트, PREREG_GATE2_2026-08-06.md §2.1)가 **FAIL**했다:

```
n_concurrent=16 n_ok=16 n_match=14 n_mismatch=2  (idx 3, 6 -- 둘 다 같은 alt-sha)
```

Zamba2(job 875344)는 같은 게이트에서 16/16 PASS. G-1(단일 요청, batch=1 self-repro +
A1-vs-A3 byte-identical)은 두 모델 모두 PASS였다 — 즉 **동시성에서만** 드러난다.

**그러나 원 G-2 게이트에는 음성대조가 없다.** 비교 대상이 **A3(chunk512)-16동시 vs
A1(plain)-batch=1**뿐이라 (a) chunked-prefill의 mamba state 이월 결함과 (b) 배치 조성에
따른 통상적 수치 비결정성(모든 arm에 있을 수 있는 성질)을 구분하지 못한다. (b)라면 A1도
같은 비율로 자기 batch=1 기준과 불일치해야 한다 — 그것이 여기서 재는 것이다.

**성능 측정·헤드라인 없음.** 산출은 불일치율과 첫 갈라진 토큰 위치뿐.

## 1. 설계

**두 모델**: `Zyphra/Zamba2-2.7B` (ctx=4096, `--attention-backend triton`),
`ibm-granite/granite-4.0-h-micro-base` (ctx=8192, `--attention-backend flashinfer`).
875344/875346과 **동일 서버 플래그**(arm별 extra flags 동일), **동일 프롬프트 생성기**
(`g2_holb_phaseA_lib.sh`의 `build_g2holb_prompt` — bench_serving `random-ids` 복제,
offset seed 4242 고정, target 2000 토큰 → 실측 재인코딩 길이를 토크나이저로 실측해
`g2ctrl_prompt_meta_<tag>_<jobid>.json`에 로그). 4 arm: `plain` / `plainaux`
(`--chunked-prefill-size -1 --disable-overlap-schedule`) / `chunk512`
(`--chunked-prefill-size 512`) / `agnostic` (`--enable-pdmux
--pdmux-config-path pdmux_a100_smoke.yml --chunked-prefill-size -1
--disable-overlap-schedule`) — `g2_run.sbatch`의 `arm_flags()`와 문자 그대로 동일.

**HOLB 프로브 OFF** (`PDMUX_HOLB_PATH` 미설정) — 875346은 ON이었다. 이 차이는 §4에 명시.

arm마다, 한 번의 서버 부팅 안에서:

1. **자기 batch=1 기준선**: 동일 프롬프트로 greedy(`temperature=0`) 1회 호출, `output_ids`
   + sha256 저장(`g2ctrl_<tag>_<arm>_selfbaseline_<jobid>.json`).
2. **16-동시 greedy 배치**를 **3회 반복**(rep1/2/3), 각 응답을 (1)의 자기 기준선과 대조
   (`output_ids` 배열 완전 일치 = match). `plain`(A1)의 자기 기준선은 별도로
   `chunk512`(A3)의 **교차 기준선**으로도 보존한다(원 G-2와 동일 대조를 이 잡음에서
   재현하기 위함) — A3만 self·cross 두 대조를 모두 낸다.

| arm | 대조 | 무엇을 판별하나 |
|---|---|---|
| A1 `plain` | 16동시 vs 자기 batch=1 | ★핵심 음성대조 — 여기서 불일치가 나면 (b) |
| A2 `plainaux` | 16동시 vs 자기 batch=1 | overlap 끈 상태의 기준 |
| A3 `chunk512` | 16동시 vs 자기 batch=1 **및** vs A1 batch=1 | 875346 재현 + 자기기준 분리 |
| A4 `agnostic` | 16동시 vs 자기 batch=1 | pdmux의 비결정성 수준 |

arm 순서는 고정(`plain plainaux chunk512 agnostic`) — 성능이 아니라 정확성만 재므로
순서 랜덤화의 근거(방법론 게이트 "동일 trace·node 내 실행 순서 randomize")가 적용되지
않는다(A1이 먼저 나와야 A3의 교차 기준선을 확보할 수 있다는 순차적 제약이 있음).

- **HTTP 상태 코드 필수 검사** — 874601 전례(재-토큰화 없이 바이트 길이를 토큰 수로
  오인해 400을 받고도 빈 sha로 "일치"라 오판). 비200 = `UNDETERMINED`, `FAIL` 아님.
  `g2ctrl_gate.py`가 이 규율을 `g2_concurrent_gate.py`/`greedy_call`과 동일하게 따른다.
- **불일치 위치**: 불일치가 난 요청은 자기(또는 교차) 기준선과의 첫 갈라진 생성-토큰
  인덱스(0..95)를 기록한다(`g2ctrl_gate.py`의 `first_diff_index`). 길이가 다르면 그것도
  기록.

## 2. 사전등록 판정 규칙 (제출 전 고정 — 여기서 바꾸지 않는다)

모델별·arm별로 3 rep × 16 = 48 시행 중 불일치 수를 합산해 불일치율 $p$를 낸다.

- **A1의 자기-불일치율이 A3의 자기-불일치율과 통계적으로 구분되지 않으면**
  (2×2 Fisher exact test, α=0.05, 양쪽 모델 합쳐 n=96 vs n=96 우선 검정 — 표본이
  작으므로 모델별로도 개별 보고) ⇒ 875346의 G-2 FAIL은 **(b) 배치 비결정성**이고
  chunk512 고유 결함이 아니다 ⇒ PREREG_GATE2 §2.1의 "불일치 ⇒ A3 폐기" 규칙은 **이
  불일치 패턴에는 적용 대상이 아님**으로 보고한다(다른 이유로 A3를 배제할 수는 있다 —
  여기서 면제되는 것은 이 특정 증거뿐).
- **A1이 3/3 rep에서 48/48 자기-일치인데 A3만 불일치**하면 ⇒ **(a) chunk512 고유** —
  심각, PREREG_GATE2 §2.1 규칙 그대로 적용(A3 폐기 검토).
- 그 사이(A1도 A3도 불일치하지만 비율이 유의하게 다름, 또는 표본 부족으로 Fisher가
  판별 못 함)면 `INDETERMINATE`로 보고하고 필요한 n(검정력 계산)을 `g2ctrl_analyze.py`가
  낸다.
- **불일치 위치가 매 rep·매 모델에서 chunk512에만 존재하고 생성 스텝 0(첫 토큰)에
  고정적으로 몰리면** (a) 쪽 증거를 강화(상태 오염은 첫 생성부터 궤적 전체에 영향).
  **위치가 넓게 퍼져 있거나 A1에도 나타나면** (b) 쪽 증거.
- 이 판정 규칙은 결과를 본 뒤 바꾸지 않는다(방법론 게이트 #8 — SLO/판정 기준 재조정
  금지의 취지를 여기 correctness 판정에도 적용).

## 3. 예산

4 arm × (1 baseline + 3 rep × 16 동시) = 4 × 49 = 196 호출/모델, 2 모델 = 392 호출.
서버 부팅 8회(모델 2 × arm 4), 각 부팅에 cudagraph capture 포함 — 원 g2_run.sbatch와
같은 플래그이므로 부팅 시간도 비슷(~60-90s/arm). 벤치 서빙(`bench_serving`) 호출 없음
(성능 측정 자체가 없음) — Phase 0/Phase 1 생략. 추정 ≈0.1-0.3 GPU-hr, 짧다.

## 4. 원 G-2(875346)와의 차이 — 인용 시 명시할 것

- **HOLB 프로브 OFF**(875346은 ON, PREREG_GATE2 §14.2 addendum B). 875346의 G-2 자체는
  HOLB 프로브가 이미 마운트된 서버에서 측정됐다 — 이 negative control은 그 조건과
  다르므로, "875346과 완전히 동일 조건의 재현"이 아니라 **같은 서버 플래그·같은 프롬프트
  생성기 위에서 자기기준을 추가한 독립 측정**이다.
- **독립 새 boot**(같은 job이 아니라 별도 job) — 원 G-2가 관측한 정확한 불일치(idx 3,6,
  같은 alt-sha)가 이 job에서 재현될 필요는 없다(비결정성이 배치 조성에 의존하면 서버
  재부팅마다 어느 슬롯이 걸리는지가 달라질 수 있다) — 재현 여부 자체도 보고할 가치가
  있는 관찰.
- **arm 순서 고정**(원 캠페인은 rep마다 순열) — §1에서 정당화.

## 5. 하지 말 것 (제출 전 고정)

- 성능 측정·헤드라인 금지 — 산출은 불일치율과 위치뿐.
- `875344`/`875346` 아티팩트 수정 금지(읽기 전용 참조).
- 커밋 금지(이 사전등록 파일 포함 — coordinator/git-committer 소관).
- 제출 후 폴링 금지 — 즉시 job id + 채점 명령 보고, 완료 확인은 코디네이터가 한다.
