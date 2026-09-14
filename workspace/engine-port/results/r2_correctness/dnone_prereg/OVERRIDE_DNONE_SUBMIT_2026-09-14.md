# 범위 한정 제출 OVERRIDE — D-none 회차 (2026-09-14)

> **지위**: `presubmit.py`가 내는 **reachability BLOCK 2건**을 **이 회차 1 job에 한정해** 넘기기 위한
> 문서. 전례 `results/cp_baseline/OVERRIDE_VPROBE_SUBMIT_2026-09-01.md`(`:10-11,:40,:42`) ·
> `results/cp_baseline/OVERRIDE_P1_SUBMIT_2026-08-28.md`(`:44,:81`)의 형식 요건 5개를 따른다.
> ★**발효(2026-09-14, 제출 창 안에서 재실행)**: §1은 **제출 직전 실행의 실제 출력**이다.
> `doc_facts` 위반은 §3대로 **고쳐서 닫았고**(681 → 697, 진리원 = 정적 `def test_` 카운트),
> 그 결과 차단은 **정확히 이 2건**으로 줄었다.

## 1. 넘기는 차단 — 출력 그대로 인용 (요건 ①)

```
=== presubmit ===
  OK     registry_append_only 삭제 0건
  OK     line_citations --- check: 93 compared, 0 violation(s)  OK
  OK     doc_facts      --- 11 fact(s), 14 occurrence(s) compared, 0 violation(s)  OK
  OK     citation_stops --- 0 added line(s), 0 violation(s), 16 rules  OK
  OK     reachability   reach_spec_A.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   reach_spec_B.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   spec_C1_canonical.json -> superseded by .../cp_baseline/spec_cp0_arm_reqactive.json (미검사)
  OK     reachability   spec_cp0_arm_reqactive.json -> DISCRIMINATING
  OK     reachability   spec_cp0_arm_reqinactive.json -> superseded by .../cp_baseline/RETRACTION_reqinactive_2026-08-28.md (미검사)
  OK     reachability   spec_cp0_capacity.json -> DISCRIMINATING
  BLOCK  reachability   reachability_spec.json -> SINGLE_LABEL_FORCED
  BLOCK  reachability   reach_spec_rev3_A.json -> RESTRICTIONS_INERT

제출 금지 — 차단 2건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.
```
★**세 번째 차단(`doc_facts`)은 이 OVERRIDE로 넘긴 것이 아니라 §3대로 고쳐서 사라졌다.**
넘기는 것은 위 **2건뿐**이다.

- `reachability_spec.json` = `results/m4r_confinement/` (M4R confinement 트랙)
- `reach_spec_rev3_A.json` = `results/tc1_model_attrib/` (TC1 model-attribution 트랙, **2026-08-28자**)

## 2. 왜 이 회차와 인과 무관인가 (요건 ②)

- **M4R (`SINGLE_LABEL_FORCED`)**: spec 자신의 측정된 제약이 `RESIDUAL_PRESENT`/`RESIDUAL_ABSENT`를
  배제해 `RESIDUAL_INCONCLUSIVE`만 도달 가능하다. 정본이 이것을 **현재 상태**로 등재하고 있고
  (`PROJECT_STATUS.md:8665` *"presubmit: `SINGLE_LABEL_FORCED`(BLOCK, 제출 금지)"*,
  *"유일한 해소 경로 = `PDMUX_STICKY_PARTITION=1` 신규 측정"*; `:3578` *"두 트랙 모두 캠페인 구매
  불가"*), `m4r_rule_rev2`가 최신이며 rev3는 없다. **해소는 새 측정이고 supersede가 아니다.**
- **TC1 (`RESTRICTIONS_INERT`)**: 제약이 격자를 균일 축소한다(모든 실질 라벨 24/72) — `design_reachability.py`
  TOOL_REV 2가 잡도록 만들어진 형태. 도구 자기감사(커밋 `5d180a6`) 이후 *"rev3 spec은 이제
  `DISCRIMINATING`이 아니라 **`BLOCKED`**(구 판정 철회)"*(`PROJECT_STATUS.md:3502`·`:8664`).
  TC1 rev3는 규칙층 감사 死因 F8–F12로 죽었고 **rev4가 없어** `superseded_by`가 가리킬 후속이 없다.
- **이 회차와의 관계**: D-none은 **R2 correctness 트랙**(`results/r2_correctness/`)이며 두 spec의
  어떤 산출물·규칙·측정도 읽거나 쓰지 않는다. 채점기는 `boots.txt` 라벨의 5개 파일만 열고
  디렉터리 스캔이 없다(감사자 독립 확인: `open(` 지점 `:135 :190 :221 :366 :373 :502`).
  ⇒ **두 차단은 이 회차의 판정에 어떤 경로로도 들어오지 않는다.**

## 3. ★이 OVERRIDE가 **덮지 않는** 것 — `doc_facts`

같은 presubmit 실행의 `BLOCK doc_facts`는 **넘기지 않는다.** 그것은 후속 spec이 없는 타 트랙 설계
결함이 아니라 **한 숫자로 닫히는 위생 결함**이다(`results/kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md:382`의
자기보고 회귀 수 ↔ 저장소 전역 정적 `def test_` 카운트). **고쳐서 통과시켜야 하며 OVERRIDE로
넘기면 안 된다**(rev2 판정서 DNA2-7). 진리원이 전역 카운터라 동시 작업 트랙이 언제든 다시
깨뜨리므로 §6의 **한 창 규율**이 필요하다.

## 4. 하지 않는 것 — 전수 (요건 ③)

1. **두 차단을 해소·완화하지 않는다.** 두 spec의 판정 규칙·임계·제약을 고치지 않는다.
2. **레지스트리에서 spec을 제거하지 않는다**(`registry_append_only` 검사, 전례 커밋 `d11243a`).
3. **`presubmit.py`의 차단 범위를 좁히지 않는다**(전례 2문서가 명시 금지).
4. **선행 `NO-GO`를 무르지 않는다** — D-none rev1의 `NO-GO`, λ0 rev1–rev4의 4연속 `NO-GO`는 그대로다.
5. **감사 없는 정책 판정을 하지 않는다.** 이 회차는 **성능 판정 0건**이며 확정 결론
   (HE0·layer-type 死·정책 순위·stake #1·게이트 #13/#16·C2 인용정지)을 건드리지 않는다.
6. **M4R/TC1을 재개하지 않는다** — `PROJECT_STATUS.md:8664`가 *"모델을 바꾸면 TC1이 재개된다"*를
   명시 금지한다.

## 5. 범위와 GPU 상한 (요건 ④)

- **정확히 1 job**: `R2C_ORDER="L TD L TD DN"` · `R2C_GUARD=none` · 등록 튜플
  (Nano-9B-v2-Base / flashinfer / ctx 16384 / D44). 사전등록 =
  `dnone_prereg/PREREG_DNONE_2026-09-14_rev2.md`(`11b63095…`) + 구속 追記
  `PREREG_DNONE_rev2_ADDENDUM_2026-09-14.md`(`5a0fbde8…`), 판정 `GO-with-caveats`
  (`VERDICT_dnone_rev2_2026-09-14.md`, `ec8502e7…`).
- **GPU 상한 0.197 GPU-h**(예보 708초). 초과 시(구조적 최악 7,000초) 그 사실을 장부에 등재한다.
- **재실행 예산 0**(C3 승계). `NO_VERDICT_SCOPE`·`NO_KNOB`·`UNREALIZED-*`·`NO_RUN`도 소비로 계상.
- 이 OVERRIDE는 **이 1 job으로 소진**된다. 다음 job은 새 OVERRIDE가 필요하다.

## 6. 한 창 규율 (DNA2-7)

제출은 다음을 **하나의 짧은 창에서** 수행한다: ① λ0 트랙 편집 종료 확인 →
② `check_doc_facts.py` 위반 0으로 수리(§3) → ③ `presubmit.py` 재실행, 차단이 **정확히 이 2건인지**
확인하고 그 출력을 §1에 전사 → ④ `git diff HEAD --` 로 `m4r_confinement/reachability_verdict.json`·
`tc1_model_attrib/reach_verdict_rev3_A.json`가 **빈 출력**임을 확인(게이트 #240: presubmit은
read-only가 아니다) → ⑤ 이 트랙 CPU 회귀 **98/98 OK** 확인(NPC-H) + 전체 스위트 실측 기록
(**DN2-1**) → ⑥ `sbatch`. 창이 벌어지면 ②–⑤를 다시 한다.

## 7. 사용자 명시 지시 (요건 ⑤)

2026-09-14, 사용자 선택: **"(a) 승인 — 단 마지막에 몰아서 한 창에서"**
(제시된 선택지: (a) 승인·한 창에서 / (b) 지금 바로 제출 / (c) 제출 안 함).
선택지 (c)에는 *"이 회차는 Claim D 선결을 하나도 닫지 않고 P2 반입도 안 열어주며, 최대 산출은
n=1 스코프 한정 존재 문장 하나이고 OOM이 재현되도 도착열이 다르면(실재 ≈2/8) 아무 문장도 못
쓴다"*는 감사 실산이 함께 제시됐고, 사용자는 그 정보 위에서 (a)를 택했다.

## 8. 제출 창 기록 (§6 절차의 실측 — DN2-1 이행)

| 단계 | 실측 |
|---|---|
| ① λ0 트랙 편집 종료 | 코드 수리 완료(F1·F5·F6), 사전등록 rev5 작성 완료, **감사는 read-only로 진행 중** ⇒ 테스트 수 안정 |
| ② `check_doc_facts.py` | **0 violation OK**(681 → 697 정정, §3) |
| ③ `presubmit.py` | **차단 2건** = §1의 그 2건(등록된 것과 정확히 일치) |
| ④ presubmit 부작용(게이트 #240) | `git diff HEAD --` 두 verdict json **빈 출력** · `git status --porcelain` **빈 출력** ⇒ 바이트 불변 |
| ⑤ 이 트랙 CPU 회귀(NPC-H) | **98/98 `OK`**(dnone 36 + instrument 35 + ctx 16 + scope_guard 11) |
| ⑤′ 전체 CPU 스위트(DN2-1) | 제출 창에서 실측 — 결과는 아래 §8-1에 기록 |
| ⑥ `sbatch` | §5의 리터럴 명령 1회 ⇒ ★**job 908534**(2026-09-14, `amd_a100nv_8`, `--time 2:30:00`). 이 OVERRIDE는 **이 job으로 소진**된다 |

### 8-1. 전체 스위트 실측 (제출 창) — ★`DN2-1`의 등록 수치를 정정한다
```
Ran 697 tests in 778.875s
OK
```
**전체 CPU 스위트 = 697 / 697 `OK`, 실패 0건**(`python -m unittest discover -s workspace/engine-port/tests`,
제출 창 1회 실행). ★**rev2 판정서 `DN2-1`이 등록한 "696 중 4 실패(전부 `test_lambda0_prereg`)"는
그 감사 시각(2026-09-14 17:3x)에 λ0 트랙이 **편집 중이던 상태의 스냅샷**이었다 — λ0 rev5의
F1·F5·F6 수리가 끝난 뒤 그 4건은 전부 통과한다. rev1 감사 시각의 유일 실패
(`TestMutationHarness.test_no_escapes`)도 마찬가지다.** ⇒ **"제출 시점 전체 스위트에 실패가
있었다"고 쓸 수 없다.** 이것이 `DNA2-7`이 "제출 직전 실측"을 요구한 이유의 실물 사례다:
**동시 편집 중인 트랙의 중간 상태를 게이트 사실로 등재하면 거짓이 된다.**
합격 판정 자체는 여전히 **NPC-H가 등록한 이 트랙 98/98**을 기준으로 한 것이다(전체 수치는 병기용).
