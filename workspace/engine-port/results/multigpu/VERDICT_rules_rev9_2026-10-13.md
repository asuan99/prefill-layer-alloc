# 규칙층 경량 델타 감사 — PREREG_MULTIGPU rev9 §16 (커밋 5203fe6), 2026-10-13

> 감사자: claims-auditor(읽기 전용, GPU 0). 요지를 메인 세션이 저장(처방·caveat·하네스 조건 문안은 원문 그대로).

## 0. 결론
| 층 | 등급 | 死因 |
|---|---|---|
| M2g | `GO-with-caveats` | 0 (MGP-44·47·48 승계, 새 MGP-51·55) |
| M3 | **`NO-GO`** | **2: D10(N2), D11(N2)** — 등록 문장 2개 + 상수 1개로 해소, 설계 변경 불필요 |
| 종합 | `MIXED` | |
D9는 P9-a·b·d로 닫힘. P9-c 문언이 shape 축을 빠뜨려 D10, 하드캡 14 h 확정으로 미등록 config 순서 유도식이 S3에서 살아남는 mode를 정함(D11). rev8 판정서의 "config 순서 = 정의역만 변화" 분류를 뒤집음(#110, #113).

## 1. P9 검토
P9-a 닫힘(봉인 조건: §4-4 문장 취소선, 우선순위 조항 "§16 > §3·§4·§15"). P9-b 닫힘(부작용 MGP-52). P9-c 부분 → D10. P9-d 닫힘(J1 S3 9.0–9.9 h vs 10 h, 절단돼도 fail-closed). 기준 셀 포화 미요구 → 하네스 H-3.

## 2. 사용자 결정 3건
- 메모리 (iii) 폐지: fused_sym 대비 상대차 — TP2 agnostic 0.957%·d44 0.9989%, TP4 1.341%·1.381%, TP8 1.290%·1.328%, DP/N1 ≤ 0.092%. 분모 d44(min)면 TP2 1.0090% > 1% → 반사실 공시에 두 분모 필요(MGP-51). (i)(ii)만으로 C-5/MGP-12 목적 유지(KV 여유 N1/DP 2.72 M vs 396 K 6.9×, TP2 6.39 M vs 793 K 8.1×, TP4 13.7 M vs 1.59 M 8.7×, TP8 28.5 M vs 3.17 M 9.0×; KV 외 arm별 메모리 경로 미발견). 사후 변경이지만 결과량 아닌 공변량·M3 미관측·정의 가능 여부만 변화·기전 근거 실측 → 死因 아님.
- 하드캡 14 h: MGP-49 적용, 단 순서 무작위화로 문구 수정 → MGP-49′.
- J1 x2 GPU0: J1↔J2 앵커 quota 일치(J4·J8은 상이), x2 quota·throttle 미실측, x2 위상 실패 시 J1(+J2) ABORT 위험 → MGP-43′, H-7. M2g J1 적용 여부 미명시 → MGP-55.

## 3. 반전 표 요지
P-DP shape 간 우선순위 → MGP-13 발화 반전(D10). config 순서 유도식 8조합 → S3에서 TP 추세 인용 ↔ DP 추세 정의(D11; S3 초과 J2 3.9–5.0 h, J4 2.2–3.2 h, J8 0.1–1.2 h vs LO 블록 ≈1.30 h → J2·J4 두 번째 config LO 전부 손실). 메모리 (iii) 존폐·반사실 분모 → caveat. KV 확대·J1 10 h·기준 셀 포화: 미반전. 보정 불가 J_N·x2 quota: 원자료 없음. P9-b NO_TREND+미달 → MGP-52.

## 4. 死因·처방
- **D10**: P9-c가 shape 축 미지정 → 한 shape UNTESTED가 다른 shape의 BROKEN 증거를 억제. **P10**: "P-DP는 shape별로 상태를 낸다. shape s의 G\_sym DP 추세가 정의되고, G\_sym(1, s)가 정의되며, `TREND_JOB_SENSITIVE`가 아닌 방향 라벨(G\_INCREASES/G\_DECREASES)이면 그 shape는 BROKEN이다. P9-c의 세 조건에 해당하면 그 shape는 UNTESTED이고, 두 shape 모두 `NO_TREND_DETECTED`면 HELD다. P-DP 전체 상태는 BROKEN > UNTESTED > HELD 순으로 정한다. 어느 한 shape라도 BROKEN이면 MGP-13이 발화하며, 다른 shape의 UNTESTED가 이를 억제하지 않는다."
- **D11**: 하드캡 14 h 아래 미등록 순서 유도식이 살아남을 mode 결정. **P11** (a) 순서 유도 문자열과 결과 순서(J2/J4/J8)를 봉인 전 등록 문서에 적고 dry-run과 대조. (b) "DP 추세가 `TRUNCATED` 셀 때문에 `TREND_UNDEFINED`이고 그 결과 P-DP가 UNTESTED이면, TP 방향 라벨은 공시를 붙여도 인용하지 않는다(`TP_UNCONTROLLED_TRUNCATION`). 반대로 TP가 절단으로 미정의일 때 DP 라벨은 P-DP 판정에만 쓴다."

## 5. caveat (문자 그대로 승계)
- **MGP-43′**(MGP-43 대체): "rev9에서 J1은 `resourcespec-a100x2` 노드의 GPU 0 단독이다. J1과 J2 앵커는 같은 스펙이고, J4·J8 앵커는 GPU 수에 비례하는 다른 quota에서 잰 N=1 값이다. x2·x4 quota와 x2 셀 throttle 분포는 실행 전 미실측이며(x1 11코어, x8 88코어만 실측), 실측 `cpu.max`를 Job별로 병기한다. J1↔J4/J8 앵커 차이를 순수 'Job 효과'로 쓰지 않는다. J1↔J2 앵커 차이도 quota 일치가 실측으로 확인된 경우에만 'quota를 맞춘 Job 효과'로 쓴다."
- **MGP-49′**(MGP-49 대체, MGP-14 존속): "rev9 하드캡(J1 10 h, J2·J4·J8 14 h)에서 공동 확장(S3)이 일어나면 예보상 J2·J4의 순서상 두 번째 config는 LO 전부를 잃어 G가 미정의이고, J8은 두 번째 config의 LO 일부를 잃는다. 어느 config가 두 번째인지는 등록 순서 유도식이 정한다. 절단된 셀은 `TRUNCATED`다. '모든 Job이 캡 안', 'TP와 DP가 같은 조건에서 비교됐다'로 쓰지 않는다."
- **MGP-50′**: "MGP-50(TP4·TP8 G\_sym 미정의 예보)은 rev9의 메모리 규칙 (iii) 폐지로 효력을 잃었다. 단 MGP-51의 반사실 공시를 함께 적는다."
- **MGP-51**: "rev9는 `MEMORY_MISMATCH` 조건 (iii)(arm 간 `max_total_num_tokens` 상대차 > 1%)을 M2g 2회차 공변량을 본 뒤 삭제했다(사후 변경, M3 결과 미관측 상태). 근거는 KV 비구속이다(필요 대비 가용 6.9–9.0×, M2g 2회차). (iii)가 유지됐다면 TP4·TP8은 G\_sym 미정의였고, TP2는 상대차 분모에 따라 0.9989%(분모 fused\_sym) 또는 1.0090%(분모 d44)로 경계에 있다. 결과 문서는 config별 arm 간 상대차를 두 분모로 모두 적고, TP 추세 라벨 옆에 'rev1–8 규칙이었다면 TP 추세는 `TREND_UNDEFINED`'를 병기한다. 'arm 간 메모리가 동등했다'로 쓰지 않는다. 쓸 수 있는 문장은 'KV 풀이 어느 arm에서도 구속되지 않았다(조건 (ii))'뿐이다."
- **MGP-52**: "P9-b에 따라 `JOB_EFFECT`가 하나라도 발화하면 크기 3% 미만의 NO\_TREND 추세도 `TREND_JOB_SENSITIVE`가 되므로, 그 경우 P-DP는 HELD가 될 수 없다. P-DP UNTESTED를 'P-DP 성립'으로 쓰지 않는다."
- **MGP-53**: "Job 효과 검사의 기준은 J1 boot1(1.6× rung) 단일 Job 값이며 J1 자체의 Job 효과는 검사되지 않는다. `JOB_EFFECT` 미발화의 뜻은 MGP-42에 따른다."
- **MGP-54**: "TP와 DP 추세가 서로 다른 Job 순서·절단 조건에서 정의됐으므로 'TP가 DP보다 잘 스케일한다/못한다'를 이 등록의 결과로 쓰지 않는다."
- **MGP-55**: "§16-2.3(J1 = x2 GPU0 단독)이 M2g J1에도 적용되는지와 M2g J1의 내용은 하네스 dry-run 출력으로 확정된다. 그 전에 M2g J1과 M3 J1을 같은 노드 조건으로 쓰지 않는다."

## 6. 하네스 조건 추가
H-1(D10 변이: "UNTESTED 우선" — {A: TREND_JOB_SENSITIVE, B: 인용 가능 G_INCREASES}, {A: G1_UNDEFINED_NO_JOB_CHECK, B: G_INCREASES}에서 기대 BROKEN·MGP-13 발화, KILLED + 무변이 대조). H-2(D11: 순서 유도 문자열 mg_const 상수, dry-run 순서 출력·문서 일치, "절단 사유 무시 공시 후 인용" 변이 KILLED). H-3(기준 셀 J1 boot1 A fused_sym·agnostic이 HI_SATURATED 아님/HARNESS_LIMITED/CLIENT_ERROR → 모든 J_N JOB_EFFECT_UNTESTED; "확장 rung G(1)로 보정 비" 변이 KILLED). H-4(앵커 UNTESTED J_N 점은 보정 추세에서 제외, 점 <3이면 보정 추세 TREND_UNDEFINED·P9-b 불통과; "원값 대입" 변이 KILLED). H-5(G5(c) (iii) 제거, 상대차 두 분모 기록, `MEMORY_RULE_III_COUNTERFACTUAL` 두 분모 산출; "(iii) 잔존"·"(ii) 제거" 변이 KILLED). H-6(우선순위 조항 §16 > §3·§4·§15, §4-4 폐기 문장 취소선). H-7(권고: M3 전 x2 위상·cpu.max 프로브).

## 7. GO 조건
P10 문장, P11(a) 상수, P11(b) 문장 등록 + H-6 → M3 경량 델타 감사에서 `GO-with-caveats` 가능. 이후 하네스층 H-1–H-5.
