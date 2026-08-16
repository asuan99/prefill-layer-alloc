# `audit_g16_independent_2026-08-16/` — 감사의 독립 재구현 (원문 보존)

**무엇인가**: G16 사전등록 rev1의 **규칙 감사**(claims-auditor, 2026-08-16) 및 그 후속
질의에서 감사가 작성한 **독립 재구현 스크립트 5조각의 원문**이다. 사전등록
`PREREG_G16_RULES_REV3_2026-08-16.md`의 양성대조 **PC-D·PC-E 표적값의 유일한 출처**다.

**왜 저장소에 있는가**: 이 저장소는 **감사의 독립 재구현이 스크래치에만 남아 소실되는
실패를 2회 겪었다**(2026-08-14 E-1a errata · 2026-08-16 C2-R — 후자는 `CONSENSUS.md`
§3 항목56(C)/메모리 교훈 39에 "재현 경로 미보존"으로 등재됨). **3회차를 만들지 않기 위해**
감사 회신 본문의 소스를 **축자 그대로** 옮겨 보존한다.

**리팩터링 금지.** `pdmux_eval`을 import하지 않는 것이 이 코드의 존재 이유다(양성대조의
독립성). `percentile`은 정본과 같은 **선형보간**을 독립 구현한 것이다.

---

## 실행 (조각마다 cwd가 다르다)

| 조각 | cwd | 명령 |
|---|---|---|
| A | 무관(절대경로) | `python3 fragA_g20h_scan.py` |
| B | `results/g2_0_hard` | `python3 <이 디렉터리>/fragB_g20h_perboot.py` |
| C | `results/slo_sched` | `python3 audit_g16_independent_2026-08-16/fragC_sgptv_recompute.py` |
| D | `results/slo_sched` | `python3 audit_g16_independent_2026-08-16/fragD_sgptv_perboot.py` |
| E | `results/slo_sched` | `python3 audit_g16_independent_2026-08-16/fragE_sitl_sensitivity.py` |

**2026-08-16 메인 세션이 5조각 전부 실제로 실행해 감사 보고값과 일치함을 확인했다**
(아래 §교차검증).

## 출력 ↔ 사전등록 소비처

| 수치 | 조각 | 소비처 |
|---|---|---|
| PC-E 표적 HI/LO `M_itl`·`M_ttft` | **C** | rev3 §7 PC-E |
| ORACLE §3-2 교차검증 pass율 | **C** | 이 코드의 정당성 근거 |
| 처리량 5.115/5.413/5.492/5.556 | **C** | rev3 §3-1 처리량 순위 법칙 |
| per-boot `M_itl` 리스트(검정력) | **D** | rev3 §6 δ(K7)·`ITL_SATURATED` |
| g2_0_hard 확장격자 arm 평균·pass율 | **A** | rev3 §3 도달가능성 · §7 PC-D |
| g2_0_hard per-boot ±SD | **B** | rev3 §3 기준1 표 |
| `S_itl` 밴드·knife-edge | **E** | rev3 §4 1차-A · §6 `S_ITL_UNREACHED` · K8 |

## ⚠️ 보존 시 병기 필수 (감사 자신의 지적 3건)

1. **A·B는 `json.load()`(파일당 JSON 1개)** ⇒ `ROUNDS=1` 아티팩트 전용. sgptv(3라운드/파일)에
   걸면 **깨진다**. **C/D/E만 줄 단위 루프**로 두 형식을 처리하며, **게이트 #7 duration
   합산은 C만 수행**한다.
2. **빈 ITL 요청 규약이 조각마다 다르다**: A `9e9` · B **드롭** · C/D/E `inf`.
   정본 `RequestResult.passes`는 `percentile(())=nan → 실패`이므로 **`inf`(C/D/E)가 정본
   정합**이다. B의 드롭은 이 데이터셋에서 결과를 바꾸지 않았으나(A와 B의 `ITLp95med` 일치)
   **규약 불일치로 명시**한다.
3. 어느 조각도 `pdmux_eval`을 import하지 않는다.

## 교차검증 (2026-08-16 메인 세션 직접 실행)

조각 C의 HI 블록이 `ORACLE_REANALYSIS_2026-08-16.md` §3-2를 **소수 셋째 자리까지 재현**:
TTFT-pass `57.083 / 66.375 / 68.917 / 70.708` · ITL-p95-pass `18.292 / 46.167 / 92.125 /
93.458` · 처리량 `5.115 / 5.413 / 5.492 / 5.556` · LO TTFT-pass `100.000 / 99.500 /
99.833 / 99.958`. ⇒ **독립 구현이 정본 라이브러리와 같은 답을 낸다**는 것이 확인됐고,
이 조각들이 산출한 **신규** 표적(`M_ttft`·`M_itl`·`S_itl` 밴드·g2_0_hard 표)은 같은
코드가 낸 값이므로 PC-D·PC-E의 표적으로 쓸 자격이 있다.
