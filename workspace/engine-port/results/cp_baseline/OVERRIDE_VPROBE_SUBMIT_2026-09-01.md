# OVERRIDE — V-probe 제출 (presubmit 차단 2건은 **타 트랙 소관**)

2026-09-01 · 사용자 명시 지시(*"실험 진행"*) · 범위 **V-probe 캠페인 한정**

## 1. 무엇을 넘기는가

`presubmit.py --registry presubmit_registry.json`은 현재 **차단(exit 1)** 상태다:

```
BLOCK  reachability   reachability_spec.json   -> SINGLE_LABEL_FORCED     (M4R 트랙)
BLOCK  reachability   reach_spec_rev3_A.json   -> RESTRICTIONS_INERT      (TC1 트랙)
```

둘 다 **다른 트랙의 설계 결함**이고 V-probe와 인과적으로 무관하다. 선례는
`OVERRIDE_P1_SUBMIT_2026-08-28.md`(같은 두 차단에 대한 같은 판단, 사용자 승인).

## 2. ★이번에 추가된 이유 — 그 도구를 **돌리는 것 자체가** 부작용이다

5회차 감사가 CP-2 §11의 재현 절차대로 `presubmit.py`를 실행했더니 그 도구가
**타 트랙 인증서 2개를 재작성**했다(`design_reachability.py`가 spec의 `out` 경로에
쓴다). 감사자가 `git checkout --`로 되돌렸는데, 그 두 파일은 **다른 세션의 미커밋
작업**(`TOOL_REV 2` 재발급)이었고 그래서 그 작업이 소실됐다.

이 세션이 확인·복구했다: 두 spec에 `design_reachability.py`를 직접 재실행해
**세션 시작 시점과 같은 상태**(+7/+15줄, 같은 finding 코드·같은 `reading`)로 되돌렸다.

⇒ **게이트가 read-only가 아니다**(감사 L16). 그러므로 V-probe 경로는 그 도구를
**돌리지 않는다.** 대신 이 트랙이 통제하는 검사만 돌린다:

```
python3 cp2_selftest.py            # rc=0
python3 g1b_plateau_selftest.py    # rc=0
python3 cp0_selftest.py            # rc=0
python3 check_version_sweep.py     # rc=0
bash -n vprobe.sbatch              # 구문
```

## 3. 이 override가 **하지 않는** 것

- ❌ M4R `SINGLE_LABEL_FORCED` / TC1 `RESTRICTIONS_INERT`를 해소하거나 완화하는 것.
  두 트랙의 설계는 **그대로 차단 상태**다.
- ❌ 레지스트리에서 spec을 빼는 것(append-only 규율, 커밋 `d11243a` 전례).
- ❌ CP-2 rev1의 5회차 `NO-GO`를 무르는 것. **정책 비교는 여전히 제출 금지**이며,
  V-probe는 판정 규칙이 아니라 측정 단계다(`PREREG_VPROBE_2026-09-01.md` §0·§9).
- ❌ 규칙층 감사 없이 **정책 판정**을 내는 것. V-probe는 판정을 내지 않는다.

## 4. 범위

- 대상: `vprobe.sbatch` 스모크 1 job + 본 캠페인 최대 7 job (총 ≈2.6 GPU-hr).
- 이 범위를 넘는 제출에는 **새 override 또는 감사**가 필요하다.
