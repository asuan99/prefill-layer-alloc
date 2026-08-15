# E-1a 아티팩트 결함 등재 (2026-08-14, doc-steward, Tier 2)

새 성능 판정 아님. `PREREG_E1_BSWEEP_REGIME_2026-08-14.md` 감사(claims-auditor,
E-1a Tier 2)에서 나온 아티팩트 결함 4건을 engine-porter/result-analyst 이관용으로
기록한다. 코드/데이터 대조는 이 문서 작성 세션이 직접 재확인했다(추정 인용 아님).

## 1. `VERDICT`·`T6_PC1_anchor_check`는 어떤 스크립트도 산출하지 않는다

`e1a_preanalysis_2026-08-14.json`의 최상위 키에 `VERDICT`(status/findings
F1–F8)와 `T6_PC1_anchor_check`가 있으나, 저장소의 세 스크립트
(`e1a_extract.py`·`e1a_analyze.py`·`e1a_report.py`) 전부를 `grep -n
"VERDICT\|findings\|T6_PC1"`로 대조한 결과 **무매치 0건**이다. `e1a_analyze.py`
실행이 만드는 키는 `SC1_...`·`T1_...`–`T5_...`뿐이고(코드 611줄 전체 확인),
`e1a_report.py`는 그 키들만 pretty-print한다(114줄 전체 확인, VERDICT/T6
언급 없음). ⇒ 두 키는 **재실행 재현 불가**(수기 편집 또는 별도 미보존 스크립트
산출)이며, `e1a_verdict.py`를 새로 작성해 이 두 키를 코드로 재현하고 diff를
첨부하기 전에는 **본문(PROJECT_STATUS.md/CONSENSUS.md)에서 `VERDICT.findings`나
`T6_PC1_anchor_check`를 직접 인용하지 않는다.**

## 2. SC1 "assert" 서술은 docstring뿐 — 코드는 assert하지 않는다(수동 재현은 10/10 일치)

`e1a_extract.py:13`: "that identity is asserted in `e1a_analyze.py` (self-check
SC1)." 그러나 `e1a_analyze.py:161-166`은 아래처럼 **값을 기록만** 하고 어떤
`assert`도 실행하지 않는다:

```python
sc1 = [{"cell": f"{a}/SM{s}/B{b}", "e1a_pooled_p50": round(pooled[(a, s, b)]["p50"], 2)}
       for (a, s, b) in [...] if (a, s, b) in pooled]
report["SC1_pooled_reproduces_s8_batch_matched"] = sc1
```

**독립 재현(이 문서 작성 세션 직접 실행, 감사자 서술을 신뢰하지 않고 재검증)**:
`s8_scaleup/s8_batch_matched.py`를 직접 재실행해 그 표에서
`(M8,SM16,B12)=58.44 / (M8,SM92,B12)=20.09 / (Ha8,SM44,B8)=38.22 /
(Ha8,SM92,B8)=29.47 / (Ha8,SM44,B16)=51.16 / (Ha8,SM92,B16)=36.08 /
(T8,SM44,B8)=13.95 / (T8,SM92,B8)=12.52 / (T8,SM44,B16)=14.81 /
(T8,SM92,B16)=13.05`를 읽었고, `SC1_pooled_reproduces_s8_batch_matched`의
10개 값과 **10/10 바이트 단위로 일치**한다. ⇒ 실질 동치성 자체는 참으로
확인됐으나, 그것이 **코드가 강제하는 불변량은 아니다** — 다음에 두 스크립트
중 하나가 바뀌면 아무 경고 없이 조용히 어긋날 수 있다. **권고**: docstring의
"asserted"를 철회하거나, `e1a_analyze.py`에 실제
`assert all(abs(v)<1e-6 for v in diffs)` 류 검증을 추가한다.

## 3. "68% of decode-active time ran at 108 SM"은 일반 서술로 쓰였으나 셀 1개 한정값

`e1a_extract.py:16`: "``D=16: FAIL ... frac=0.317`` (68% of decode-active time
ran at 108 SM, i.e. the requested partition was NOT realized)." 이 문장은
docstring 안에서 **일반 서술**로 읽히나, 이는 **T8/d16/job 865533 한 셀**의
값이다. E-1a 자신의 정정 (c)(claims-auditor, 이 세션이 채택)가 이미 지적한
대로 **job 865493**에서는 (T8/d16 조건에서) **90.1%가 16 SM에서 realize**되어
정반대 결과가 나온다(같은 (arm,cell) 라벨, 다른 job). ⇒ docstring 문장 앞에
**"(T8/d16/job 865533 한정, job 865493에서는 90.1%가 16 SM에서 realize됨)"**을
추가해 셀 한정으로 정정한다. 이 정정은 §1-22·§1-25(CONSENSUS.md, realized
partition은 target이 아니다)와 정합적이며 새 발견이 아니라 **기존 정본을
docstring에 반영 누락한 것**이다.

## 4. `t_ppf` 이분탐색 브래킷 `[-400, 400]`은 하드코딩 — 조용히 clip될 수 있다

`e1a_analyze.py:117`: `lo, hi = -400.0, 400.0`. 이 범위 밖의 분위수를 요청하면
`t_cdf`가 단조 함수이므로 이분탐색은 종료되지만 **경고 없이 브래킷 경계값으로
수렴**한다(참 분위수가 그보다 크더라도 발각되지 않는다).

**이번 실행에서는 미발화**: `e1a_preanalysis_2026-08-14.json` 전체를 순회해
`t_b`/`t_k`(Student-t 통계량, `wls()`가 산출) 절대값을 전수 확인한 결과
**최대 |t| ≈ 19.22**(`T2_regression["44->92|kappa2_paddedB"].
WLS_by_measurement_se.t_b`)로 브래킷(±400)에서 한참 안쪽이다. ⚠️**정정**:
이 문서 초안이 인용받은 "최대 |t| = 11.36"은 이 세션이 독립 재계산한 결과와
**일치하지 않는다**(재계산값 19.22, 약 1.7배 차이 — 어느 부분집합·통계량을
지칭한 것인지 원 출처를 특정할 수 없었다). **결론(브래킷 미발화)은 두 수치
모두에서 참**이므로 실질 판정에는 영향이 없으나, 인용 시 **19.22**(이 문서가
직접 산출·재현 가능)를 쓴다. `z_b`/`z_k`(known-variance 정규분포 통계량,
`t_ppf`를 쓰지 않음)는 최대 |z|≈65.96까지 나오지만 이건 애초에 `t_ppf` 경로가
아니라 이 결함과 무관하다.

**권고**: `t_ppf`에 수렴 실패/브래킷 근접 시 경고를 내는 가드
(`if abs(mid) > 390: warnings.warn(...)`류)를 추가하거나, 최소한 브래킷을
상수로 빼내 호출부에서 근접 여부를 사후 점검할 수 있게 한다.
