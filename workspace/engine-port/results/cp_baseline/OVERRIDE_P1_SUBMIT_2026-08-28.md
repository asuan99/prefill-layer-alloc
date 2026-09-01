# 제출 override 기록 — P1 수용 시험 (2026-08-28)


> ⚠️ **날짜 정정**: 파일명/본문 날짜 2026-08-28은 오기 — 실제 작성/실행일 **2026-09-01**. 상세: `DATE_CORRECTION_NOTE.md`.

**무엇을 넘었나**: `presubmit.py --registry .../presubmit_registry.json`이 `rc=1`
(**제출 금지**)인 상태에서 `p1_accept.sbatch`를 제출한다.

```
OK     registry_append_only 삭제 0건
OK     line_citations   50 compared, 0 violation(s)
OK     doc_facts        11 fact(s), 14 occurrence(s), 0 violation(s)
OK     reachability   spec_cp0_arm_reqactive.json -> DISCRIMINATING
OK     reachability   spec_cp0_capacity.json      -> DISCRIMINATING
BLOCK  reachability   reachability_spec.json  -> SINGLE_LABEL_FORCED   (M4R 트랙)
BLOCK  reachability   reach_spec_rev3_A.json  -> RESTRICTIONS_INERT    (TC1 트랙)
```

**차단 2건은 전부 다른 트랙 소관**이고 CP가 고칠 수 있는 것이 아니다. CP가 유발했던 2건
(`line_citations` 16 · `doc_facts` 1 — 이 세션의 P1 패치가 `scheduler.py`를 13줄 밀고 회귀
테스트를 73개 늘려서 생겼다)은 doc-steward가 **이미 해소**했다.

**누가 승인했나**: 사용자. 세 갈래(①대기 ②도구 범위 조정 ③기록 남기고 P1만 제출)를 제시하고
③을 권고했으며 사용자가 진행을 지시했다.

**왜 이 건에 한해 방어 가능한가**
1. **P1은 이 게이트가 겨냥하는 대상이 아니다.** `presubmit.py`는 *"판정 대상이 자기 판정 범위를
   정할 수 있으면 게이트가 아니다"*를 막으려고 만들어졌고, 차단 술어는 **도달가능성 인증**이다.
   P1은 도달가능성 인증을 소비하지 않는다 — **계측이 등록된 추정량을 재는지**를 확인하는
   배관 스모크이고 `PROJECT_STATUS.md` "방법론 게이트" **#26**의 명시적 형태다
   (*"스모크의 PASS는 성능 판정에 아무 정보도 주지 않는다"*).
2. **차단된 두 spec을 읽지도 쓰지도 않는다.** P1은 M4R·TC1의 규칙 파일·인증·아티팩트와
   접점이 없다.
3. **판정을 산출하지 않는다.** goodput 0건 · arm 순위 0건 · 정본 갱신 0건.
4. 비용이 작고 되돌릴 수 있다(`scancel`).

**이 override가 허가하지 않는 것 (명시)**
- ❌ **CP-0 본체 제출** — 4회차 감사가 `NO-GO`(死因 G1–G8)이고, 남은 순수 규칙 문제 4건
  (G1 rate 축 미식별 · G2 격자의 절대단위 이송 · G5 자기검사 오라클 부재 · G6 접기 규칙 부재)이
  닫히기 전에는 CP-1이 소비할 수 없는 라벨이 나온다.
- ❌ **G1 프로브**(감사 rev4 권고 3번, ≈0.3 GPU-hr) — 그것은 **측정**이고 자체 등록이 필요하다.
  이번 제출에 얹지 않는다.
- ❌ 다른 트랙의 차단을 대신 해소하거나 우회하는 것.
- ❌ `presubmit.py`의 차단 범위를 좁히는 도구 변경(그 변경은 그 자체로 감사 대상이다).

**단계별 플래그 규약(제출 직전 발견, 등록 완료)**: `p1_accept.sbatch`는
`--disable-piecewise-cuda-graph`를 **의도적으로 넘기지 않는다** — P1-g의 존재 이유가 cps가
piecewise 캡처 목록을 좌우하는지 **관측**하는 것이기 때문이다. CP-0 본체는 반대로 그것을
**전 arm에 강제**해 cps를 유일한 레버로 만든다. 두 단계의 차이는 사전등록 §6과
`check_version_sweep.py`의 **S7**이 기계로 고정한다.

**제출 전 통과한 것**: `check_version_sweep.py` 0 violation · `cp0_selftest.py` OK ·
도달가능성 인증 2건 `DISCRIMINATING`(findings 0) · `check_citation_stops.py` 전 파일 0 violation ·
CPU 회귀 `Ran 244 tests OK`.

---

# 범위 확장 (2026-09-01) — P1 **재실행** + **G1 프로브**

**승인**: 사용자. 두 잡을 함께 내도록 명시 지시.

## 1. P1 재실행 — 기존 범위 안
같은 `p1_accept.sbatch`, 같은 근거. 첫 실행(job 899768)이 `P1_BLOCKED_INSTRUMENT`로 끝났고 그것이
드러낸 결함 3종(SIGKILL로 인한 flush 유실 · 채널2의 DECODE 오계수 · P1-e window 겹침)이
수리됐다. ★**재실행이 특히 필요한 이유**: `on_forward`가 이제 forward마다 `classify_forward_mode`를
호출하고 방출 이벤트 집합이 바뀌었으므로 **P1-f(관찰자 효과 ±3%)를 CPU 테스트로 주장할 수 없다.**

## 2. G1 프로브 — ★**이전 override의 제외 목록에서 해제**
원 문서는 *"❌ G1 프로브 — 그것은 **측정**이고 자체 등록이 필요하다"*로 금지했다. **그 사유가
사라졌다** — `PREREG_G1_PROBE_2026-08-28.md`(규칙 `cp0_g1_rule.py` RULE_REV=1, 오라클 등재,
금지 문장 5건)로 등록이 끝났다. 제외의 근거가 없어졌으므로 해제한다.
★**도달가능성 인증은 여전히 발급하지 않는다** — G1엔 증거에 근거한 제약이 없고, 빈 `restrict`는
도구의 `RESTRICTIONS_INERT` 검사를 구조적으로 회피한다(`RETRACTION_reqinactive_2026-08-28.md`가
세운 규율). 그 사실은 사전등록 §3에 적혀 있다.

## 여전히 허가하지 않는 것
- ❌ **CP-0 본체 제출** — 4회차 감사 `NO-GO`의 死因 중 G2(격자의 절대단위 이송)·G3·G4·G7이
  미해소다. (G1·G5·G6은 이 세션에서 닫혔다.)
- ❌ 다른 트랙(M4R `SINGLE_LABEL_FORCED`, TC1 `RESTRICTIONS_INERT`)의 차단을 대신 해소하거나
  우회하는 것 — 여전히 손대지 않는다.
- ❌ `presubmit.py`의 차단 범위를 좁히는 도구 변경.
- ❌ 두 잡의 결과로 **성능 주장**을 하는 것. P1은 계측을 재고 G1은 축의 식별가능성을 잰다.
  둘 다 arm을 재지 않는다.

## 제출 전 상태
`check_version_sweep.py` 0 violation · `cp0_selftest.py` OK(111 검사) · 회귀 **295 tests OK** ·
도달가능성 인증 2건 `DISCRIMINATING`(findings 0) · 커밋 `fe8781b`·`5907a33`·`94fa01a`로 이력 고정.
