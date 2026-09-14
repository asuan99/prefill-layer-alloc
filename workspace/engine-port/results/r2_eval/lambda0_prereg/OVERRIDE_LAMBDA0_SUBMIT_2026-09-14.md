# 범위 한정 제출 OVERRIDE — λ0 0단계 (2026-09-14)

> **지위**: `presubmit.py`의 **reachability BLOCK 2건**을 **λ0 0단계 1 job에 한정해** 넘기기 위한 문서.
> 전례 `results/cp_baseline/OVERRIDE_VPROBE_SUBMIT_2026-09-01.md`(`:10-11,:40,:42`) ·
> `OVERRIDE_P1_SUBMIT_2026-08-28.md`(`:44,:81`)의 형식 요건 5개를 따른다.
> ★**D-none 회차의 OVERRIDE(`OVERRIDE_DNONE_SUBMIT_2026-09-14.md`)는 job 908534로 소진**됐다 —
> 이것은 **두 번째, 별개의** 범위 한정 OVERRIDE다.
> ★**발효(2026-09-14, 제출 창 안에서 재실행)**: §1은 **제출 직전 실행의 실제 출력**이고 digest는
> 追記 §B-2의 재계산표를 인용한다. `doc_facts` 위반은 §3대로 **고쳐서 닫았다**(697 → 710).

## 1. 넘기는 차단 — 출력 그대로 인용 (요건 ①)
```
  OK     registry_append_only 삭제 0건
  OK     line_citations --- check: 93 compared, 0 violation(s)  OK
  OK     doc_facts      --- 11 fact(s), 14 occurrence(s) compared, 0 violation(s)  OK
  OK     citation_stops --- 0 added line(s), 0 violation(s), 16 rules  OK
  OK     reachability   (superseded/DISCRIMINATING 6건 생략)
  BLOCK  reachability   reachability_spec.json -> SINGLE_LABEL_FORCED
  BLOCK  reachability   reach_spec_rev3_A.json -> RESTRICTIONS_INERT

제출 금지 — 차단 2건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.
```
★**차단은 정확히 이 2건뿐이다**(`doc_facts`는 넘긴 것이 아니라 **고쳐서 사라졌다**).
- `reachability_spec.json` = `results/m4r_confinement/`(M4R confinement 트랙)
- `reach_spec_rev3_A.json` = `results/tc1_model_attrib/`(TC1 model-attribution, **2026-08-28자**)

## 2. 왜 이 회차와 인과 무관인가 (요건 ②)
- **M4R (`SINGLE_LABEL_FORCED`)**: spec 자신의 측정된 제약이 두 라벨을 배제해
  `RESIDUAL_INCONCLUSIVE`만 도달 가능하다. 정본이 이를 **현재 상태**로 등재
  (`PROJECT_STATUS.md:8665` *"유일한 해소 경로 = `PDMUX_STICKY_PARTITION=1` 신규 측정"*, `:3578`).
  `m4r_rule_rev2`가 최신, rev3 없음 ⇒ 해소는 **새 측정**이며 supersede가 아니다.
- **TC1 (`RESTRICTIONS_INERT`)**: 제약이 격자를 균일 축소(모든 실질 라벨 24/72). 도구 자기감사
  (커밋 `5d180a6`) 이후 *"rev3 spec은 `BLOCKED`(구 판정 철회)"*(`PROJECT_STATUS.md:3502`·`:8664`).
  규칙층 감사 死因 F8–F12로 죽고 **rev4 부재** ⇒ `superseded_by`가 가리킬 후속이 없다.
- **이 회차와의 관계**: λ0 0단계는 **`results/r2_eval/` 트랙**이며 두 spec의 산출물·규칙·측정을
  읽거나 쓰지 않는다. λ0의 결정 경로는 `lambda0_{lambda_inf,plan,cells,label,analyze,reachability}.py`와
  job 907959의 `instrument/`뿐이다(사전등록 rev5 §1·§5, 판정서 `33dac116…` §2).
  ⇒ **두 차단은 이 회차의 판정에 어떤 경로로도 들어오지 않는다.**

## 3. ★이 OVERRIDE가 **덮지 않는** 것
`presubmit.py`가 내는 **`doc_facts` 차단은 넘기지 않는다.** 그것은 후속 spec 없는 타 트랙 설계 결함이
아니라 **한 숫자로 닫히는 위생 결함**이다(`results/kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md`의
자기보고 회귀 수 ↔ 저장소 전역 정적 `def test_` 카운트). **고쳐서 통과시켜야 하며 OVERRIDE로 넘기면
안 된다.** ★이 세션에서 그 수가 **다섯 번** 움직였다(295→641→645→681→697→…) — 진리원이 전역
카운터라 동시 작업이 언제든 다시 깨뜨리므로 §6의 **한 창 규율**이 필요하다(신규 게이트로 등재됨).

## 4. 하지 않는 것 — 전수 (요건 ③)
1. **두 차단을 해소·완화하지 않는다**(spec의 판정 규칙·임계·제약 불변).
2. **레지스트리에서 spec을 제거하지 않는다**(`registry_append_only`, 전례 커밋 `d11243a`).
3. **`presubmit.py`의 차단 범위를 좁히지 않는다**(전례 2문서 명시 금지).
4. **선행 `NO-GO`를 무르지 않는다** — λ0 rev1–rev4의 4연속 `NO-GO`, D-none rev1의 `NO-GO` 그대로.
5. **감사 없는 정책 판정을 하지 않는다.** 이 회차는 **성능 판정 0건**이고 **게이트 #6을 닫지 않는다**
   (λ5C-8: 어떤 결과에서도 닫지 못한다).
6. **M4R/TC1을 재개하지 않는다**(`PROJECT_STATUS.md:8664`의 명시 금지).
7. **λ0 rev5의 caveat를 완화하지 않는다** — λ5C-1(ANCHORED 분지는 이 shape/cap에서 구조적 사용 불가) ·
   λ5C-5/6(A 창 하단 여유 3.3% · B 창은 타당범위의 43%) · λ5-1(1차 등록은 `FALLBACK`) 전부 유효.

## 5. 범위와 GPU 상한 (요건 ④)
- **정확히 1 job**: λ0 0단계(`lambda0.sbatch`), 사전등록
  `PREREG_LAMBDA0_REV5_2026-09-14.md`(`8367b6ecdb6b4d795fac2e8aa9960805887f2512fe623fb01c1a87020082c214`)
  + 구속 追記 `PREREG_LAMBDA0_REV5_ADDENDUM_2026-09-14.md`(**`632c2d7804f7fef03622f5287c4f7ae2dad019f605511bb896acba061061ab6d`**
  — §B-1 구현 완료·§B-2 digest 재계산표·§B-3 미해결 2건이 추가돼 초안 sha `9af1c141…`에서 이동),
  판정 `VERDICT_lambda0_rev5_2026-09-14.md`(`33dac116c37fe758f39d9b1aa67f55124c0d714a142545e25d544dc45abd45b0`)
  **`GO-with-caveats`**(死因 0).
- **GPU 상한**: FALLBACK 최악 코너 **3.509 GPU-h**(1.00× 1.348 / 0.50× 1.977), 요청 3.60,
  `#SBATCH --time=04:30:00`. 초과 시 그 사실을 **장부에 등재**한다. 초과 시 미시도 셀은 D23으로
  `UNRESOLVED`(fail-closed).
- **재실행 예산 0** — `ABORT_F6`·`ABORT_D18`·`ABORT_MUTATION_*`·`NO_VERDICT`도 소비로 계상.
- **이 OVERRIDE는 이 1 job으로 소진된다.** 다음 job은 새 OVERRIDE가 필요하다.
- ★**digest 재확인 의무**: 판정서 §4 λ5A-8이 결정 경로 10개 digest를 핀하고
  *"어느 것이라도 달라진 상태로 제출되면 이 판정서는 그 파일에 대해 무효"*라고 적었다. λ5A-2·3·4·9·11
  구현으로 일부가 **의도적으로 바뀌었으므로**, 제출 직전 **변경 파일의 새 digest 표**를 追記 §B에
  등재하고 그 표를 이 절에서 인용한다.

## 6. 한 창 규율
① 코드 권고 5건 구현 완료 확인 + 변이 54/54·이 트랙 회귀 통과 → ② `check_doc_facts.py` 위반 0으로
**수리**(§3) → ③ `presubmit.py` 재실행, 차단이 **정확히 §1의 2건**인지 확인하고 출력을 §1에 전사 →
④ `git diff HEAD --`로 `m4r_confinement/reachability_verdict.json`·`tc1_model_attrib/reach_verdict_rev3_A.json`
가 **빈 출력**임을 확인(게이트 #240) → ⑤ 결정 경로 digest 재계산·등재 → ⑥ `sbatch`.
창이 벌어지면 ②–⑤를 다시 한다.

## 7. 사용자 명시 지시 (요건 ⑤)
2026-09-14, 사용자 선택: **"(a) 제출 — 추가 코드 5건 마친 뒤"**
(제시된 선택지: (a) 제출[코드 5건 후] / (b) 보류—오늘은 GPU 0만 / (c) B 사다리 재설계 사전등록 변제 먼저).
선택지와 함께 **"보통 1.348 / 최악 코너 3.509 GPU-h, 1 job, `--time 4:30`"** 과 감사자의 검정력 실산
(λ5C-6: B 창이 타당범위의 43%만 덮음 · λ5C-5: A 창 하단 여유 3.3%)이 제시됐고, 사용자는 그 정보 위에서
(a)를 택했다. ★제출은 presubmit 2 BLOCK을 넘기지 않으면 **구조적으로 불가능**하므로 그 승인은
이 범위 한정 OVERRIDE의 승인을 포함한다 — 단 **덮는 범위는 §1의 2건과 §5의 1 job뿐**이다.

## 8. 제출 창 기록 (§6 절차의 실측)

| 단계 | 실측 |
|---|---|
| ① 코드 권고 5건 | **구현 완료**(追記 §B-1). 변이 6종 되돌림 전부 검출 · 이 트랙 `test_lambda0_prereg` **94 OK** · 전체 **710 OK** · `lambda0_mutation_check.py` **54/54 blocked · escape 0** · `bash -n` rc 0 |
| ② `check_doc_facts.py` | **0 violation OK**(697 → 710 정정, §3 — 이 세션 **다섯 번째** 갱신) |
| ③ `presubmit.py` | **차단 2건** = §1의 그 2건 |
| ④ 부작용(게이트 #240) | `git diff HEAD --`·`git status --porcelain` 두 verdict json **빈 출력** = 바이트 불변 |
| ⑤ digest 재계산 | 追記 **§B-2** — 판정서 §4의 핀 10개 중 **3개 의도적 무효화**(`lambda0.sbatch` `c789af6c…` · `lambda0_plan.py` `2e51280c…` · `lambda0_lambda_inf.py` `2c9c8011…`), 7개 불변. **무효화 사유는 그 판정서 자신이 처방한 수리의 이행**이며 판정량은 불변(변이 검사가 고정) |
| ⑥ `sbatch` | 아래 §9 |

## 9. 제출
`sbatch workspace/engine-port/results/r2_eval/lambda0_prereg/lambda0.sbatch`
(★**운영자 스위치 없음** — D18: anchored/fallback은 **파일이 결정**한다. `PDMUX_LAMBDA0_*`를 주지 않는다.)
⇒ ★**job 908623**(`amd_a100nv_8`, `--time 04:30:00`, 2026-09-14 제출, 제출 시 `PENDING`).
이 OVERRIDE는 **이 job으로 소진**된다 — 다음 job은 새 OVERRIDE가 필요하다.

★**채점 규율(다음 세션)**: 결과는 `PREREG_LAMBDA0_REV5_2026-09-14.md` + 追記로만 채점한다.
1차 등록은 **`FALLBACK`**이며(λ5-1: 앵커는 실격) `ANCHORED`가 나오면 `ABORT_D18`로 죽어야 정상이다.
`lambda_inf = 2.1 / 0.675`는 **사전 기준이고 측정값이 아니다**(λ5C-4). **게이트 #6은 어떤 결과에서도
닫히지 않는다**(λ5C-8). 라벨 해석 시 λ5C-5(A 창 하단 여유 3.3%)·λ5C-6(B 창은 타당범위의 43%)·
λ5C-1(ANCHORED 분지는 이 shape/cap에서 구조적 사용 불가)을 병기한다.
