# D-none 사전등록 rev2 — 구속 追記 (2026-09-14, `GO-with-caveats` 이후)

> **지위**: `PREREG_DNONE_2026-09-14_rev2.md`(sha `11b630957a7d5bae5b8f9bd44aecd84255b6dc143466d9d6a22a899bea319157`)는
> 재감사에서 **`GO-with-caveats`**(死因 0, `VERDICT_dnone_rev2_2026-09-14.md`,
> sha `ec8502e7666ec9d63e05e1dc497daeb41d3ea7f7d81f799add20d35cf43af8d7`)를 받았다.
> 이 追記는 그 판정의 **비차단 권고 DNA2-1…DNA2-11과 신규 라벨 DN2-1…DN2-5를 구속 조항으로
> 편입**한다. **rev2 본문은 수정하지 않는다**(판정서가 sha로 핀했다) — 충돌 시 **이 追記가 우선**한다.
>
> ★**새 감사 회차를 요구하지 않는 근거(자기 점검)**: 아래 A1–A6의 규칙 변경은 **전부 보수적**이다 —
> 어떤 결과도 "서술 문장 있음" 라벨(`RECOVERED-*`/`NOT-RECOVERED`)에서 **"서술 문장 없음" 라벨로만**
> 이동할 수 있고 그 반대는 불가능하다. 즉 이 追記는 H1을 인정하기 **더 어렵게만** 만든다.
> A7–A9는 사실 정정·공시이며 판정 규칙을 건드리지 않는다. 이 주장이 거짓이면 그것이 死因이다.

## A1 (DNA2-1, 우선순위 최상) — 분지 A에 비-OOM 치명 크래시 필터 추가
rev2 §4-1 분지 A를 다음으로 **대체**한다:
> **분지 A** — `srv_DN1.log`에 `torch.OutOfMemoryError` **0줄** AND
> `Traceback (most recent call last)` **0줄**일 때:
> `#new-token ≥ 10036`인 배치가 1개 이상 완주 ⇒ **`NOT-RECOVERED`**, 아니면 **`UNREALIZED-NO-RISK-BATCH`**.
> `torch.OutOfMemoryError` 0줄 **AND** traceback ≥1줄 ⇒ **`UNREALIZED-OTHER-CRASH`**(신설, 서술 문장 없음).

근거: rev2 분지 A는 **비-OOM 치명 크래시를 걸러내지 않아** 그 경우에도 `NOT-RECOVERED`(= 등록
서술 문장 + 강등 검토 트리거)를 냈다 — 교훈 21의 부호 반대 통로. 감사자 전수 검사에서 그 3중
결합은 30 boot 중 **0건**이고 전례 907032 TD1·TD2는 `max=none`이라 분지 A가 거짓이 되어 정확히
`UNREALIZED`로 빠지지만, **통로 자체는 살아 있으므로 닫는다.**

## A2 (DNA2-4) — `UNREALIZED-*` 우선순위 규약 (RA3-10 승계의 완전 이행)
> `BOOT_FAILURES.txt`에 DN 행(`BOOT_FAILED boot=DN1` 또는 `SERVER_DIED_DURING_CLIENT boot=DN1`)이
> 있으면 **`UNREALIZED-BOOT-FAILED`가 다른 모든 `UNREALIZED-*`보다 우선**한다.
> `NO_KNOB`과 동시 성립하면 **둘 다 보고**한다(rev2 §4-2 유지).
근거: 907032 TD1·TD2에서 `-BOOT-FAILED`와 `-NO-RISK-BATCH`가 동시 참이 되는 것이 실증됐다.

## A3 (DNA2-3, DN2-3) — 검정력 수치 공시 (rev2 §0-d 보강)
> 분지 B에 들어가도 (i) 사다리가 거짓일 사전확률은 **같은 기판 8 boot 실측으로 ≈ 2/8**이다.
> 사다리 보유 **6/8**, 반례 **2개**: 908179 L1 = `55 368 2313 5414 10630 6213` ·
> **907959 L2 = `55 368 1468 3241 7345 12516`**(rev2가 지목하지 못한 두 번째 반례).
> 그때 라벨은 `UNREALIZED-OTHER-ARRIVAL`이며 **어떤 서술 문장도 쓸 수 없다.**
> ⇒ **"OOM이 재현되면 반드시 `RECOVERED-*`가 나온다"는 거짓이다.**

## A4 (DNA2-11) — "회차 중단"을 실행 가능한 문안으로 교체
rev2 §8-2의 *"발화하면 이 회차를 중단하고 별도 사전등록으로 넘긴다"* 를 다음으로 **대체**:
> E5가 발화하면 **하네스는 멈추지 않는다**(`set -uo pipefail`, `set -e` 없음 ⇒ 남은 boot이 완주).
> 처분은 **사후 규칙**이다: ①채점 arm 가드의 2번째 축 이동을 **자동 적용하지 않는다**,
> ②그 회차의 결과를 **`NO_VERDICT`로 보고**하고 별도 사전등록으로 넘긴다.

## A5 (DNA2-7) — 제출 선행조건을 "제출 직전 실측"으로 바꾸고 창을 좁힌다
rev2 §4-4(1)의 *"현재 2 BLOCK"* 을 **제출 직전 실측**으로 대체한다. 그리고:
> ★**`doc_facts` 위반은 OVERRIDE 대상이 아니다** — 후속 spec 없는 타 트랙 설계 결함(M4R·TC1)과
> 달리 **한 숫자로 닫히는 위생 결함**이다(`kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md:382`의
> 자기보고 회귀 수 ↔ 저장소 전역 정적 `def test_` 카운트). 진리원이 전역 카운터라 **동시 작업
> 트랙이 언제든 다시 깨뜨린다** ⇒ **OVERRIDE 작성 · `presubmit` 재실행 · `sbatch`를 하나의 짧은
> 창에서 수행**하고, 그 창의 실측 수치를 결과 문서에 병기한다(**DN2-1**).

## A6 (DN2-5 · RRC-4 승계) — `NOT-RECOVERED`의 병기 의무
> `NOT-RECOVERED`를 인용할 때는 **`srv_DN1.log`의 `Traceback` 계수와 `BOOT_FAILURES.txt`의 DN 행을
> 함께** 적는다. 승계 **RRC-4**(OOM 계수·스택 병기 의무)를 rev2 §8의 본문 인용 목록에 **추가**한다
> (rev2 목록에 누락돼 있었다).

## A7 (DNA2-2) — rev2 §4-1 표의 전사 오류 2행 정정 (데이터 정정, 규칙 불변)
| 대상 | rev2 표기(오기) | **실제 원자료** |
|---|---|---|
| 907959 L1 | `… 3241 3491 6245 10125` | **`… 3241 6245 10125 3491`** |
| 908179 L2 / TD1 / TD2 | `… 3241 3491 6245 10125` | **`… 3241 6245 10125 3491`** |

**`3491`은 `10125` 뒤다.** 실제 원자료에서는 (i)의 순서-부분열/연속-부분열 두 독법이 **일치**하므로
라벨은 바뀌지 않는다(감사자 30 boot 전수 확인). 이 정정은 표를 문자대로 읽을 때 생기는 모호성만 닫는다.

## A8 (DNA2-5 · DNA2-6) — 산술·계수 정정
- 순수 구조적 최악(5 boot 전부 최악) = **7,000 s**(+provenance) < `--time` 9,000 s.
  rev2 §7의 "1,400 s"와 §5-1의 "~1,997 s"는 **DN만 최악·채점 4 boot 정상**의 혼합 모형이다.
  `NO_RUN` 판정은 세 값 전부에서 불변.
- rev2 §0-a의 DNA-1 행 "7개 `R2C_*`" → 본문 §4-4(3)은 **9개**(명령 자체는 옳다).

## A9 (DNA2-8 · DNA2-9 · DNA2-10 · DN2-1 · DN2-2 · DN2-4) — 공시
- **전체 CPU 스위트 실측(2026-09-14 재감사 시점) = 696 / 4 실패**, 전부 `test_lambda0_prereg`
  (`TestMutationHarness.test_control_is_scored_on_the_mutation_path` ·
  `TestNoStaleRules.test_plan_has_no_dead_drain_tolerance` ·
  `TestNoStaleRules.test_plan_no_longer_claims_the_engine_cap_was_reached` ·
  `TestSbatchDiscipline.test_no_phantom_flags_and_no_stale_revision_banner`).
  rev1 감사 시점의 유일 실패 `test_no_escapes`는 **지금 통과**한다 ⇒ 그 실패는 **λ0 트랙 동시
  편집의 산물** 쪽으로 기운다(단 "선재였다가 그 사이 고쳐졌다"는 배제 불가).
  **이 트랙은 98/98 OK**이고 합격 기준은 NPC-H가 등록한 그 범위다. **λ0 관련 실패는 이 회차의
  제출 게이트 판정에 쓰지 않는다.**
- **rev1·rev2 판정서는 둘 다 메인 세션 전사본이며 전사 충실성은 제3자 검증 불가**(양 파일 머리에 공시).
- **rev1 판정서 §2 DNR-1의 역산 문장("198.00 MiB → T = 10137.6 ⇒ 밴드 안")은 거짓이다**(DN2-2) —
  `10137.6 ∉ [10036, 10137]`. 감사자가 독립 재계산으로 확인했고 rev2가 역산을 게이트에서 뺀 것이 옳다.
  인용 시 **거짓으로 표시**한다.
- **"DN arm에서 E5는 구조적으로 발화 불가"는 근거가 거꾸로다**(DN2-4) — 엔진은 worker 가드와
  무관하게 `model_runner.py:2107,:2374`에서 `torch.inference_mode()`에 들어가고, `none`은 worker를
  E5가 요구하는 **"inference mode 밖 소비자"** 로 만든다(`multiplexing_mixin.py:96-105,:700-705`).
  지지되는 문장은 **"pre-repair true-dual 2 boot(907959 TD1·TD2)에서 미발화(n=2, 30 boot 전수 0 hit)"**
  까지다. ⇒ A1의 traceback 필터가 이 위험의 실질 완화책이다.
- 재현되지 않은 항목: 격리 미러의 git 이력 의존 테스트 1개 `skipped`(실질은 별도 확인됨).

## 승계 갱신
rev2 §8의 승계에 **`DNR-V1…DNR-V9`(rev1 판정서 §6) + `DN2-1…DN2-5`(rev2 판정서 §6) + `RRC-4`** 를
추가한다. 절차의무 **15건 열거**(rev2 §8-1)는 그대로 유효하며 감사자가 **집합 일치**를 확인했다.
