# 결과 감사 r1: 미감사 문서 셋 (2026-11-08, claims-auditor, 읽기 전용)

## 0. 회차와 범위

- **회차는 세 대상 모두 r1이다.** 이 문서들을 대상으로 한 감사 파일을 찾지 못했다. 정본도 미감사로 적고 있다. `PROJECT_STATUS.md:11`은 "그 REPORT의 결함 S1–S12·패턴 P1–P6 진단은 미감사라 판정으로 쓰지 않는다"이고, 두 메모와 PROPOSAL 헤더도 각각 "미감사"·"제안"이다. 대상당 2회 상한이 있으므로 r2는 이 판정 이후의 diff만 본다.
- **CPU 재계산은 세 가지를 했다.**
  - CAP1-R 5 boot와 TUNE 12 boot의 `server.log` `#new-token`에서 multi-chunk 비율을 개수 기준과 토큰 기준으로 계산했다.
  - implfix `pi_analyze.py`·`pi_runner.py`에서 E3 handoff 문턱이 라벨 경로에 들어가는지 추적했다.
  - `dstar_result.json`의 위 벽 입력을 확인했다.
- **전제 하나를 정정한다. PROPOSAL에는 §8b가 없다.** "약 3.0"이 남은 곳은 PROPOSAL §4 l.99 한 줄뿐이다. "§8b"는 축 지도 `reports/anchor/axes_map.html` §8b(D\* 수식 절)다. 이 절은 8차 promote에서 이미 고쳐졌다. `scripts/costmodel/render_dstar_section.py:121`이 "과부하 포화 완료율 5.06–5.34 req/s, 정오 2026-11-02"를 출력하고, 재발행된 HTML도 같다.

---

## 1. `reports/audit/2026-11-01_design_defects/REPORT.md`

### 1-1. 결함별 진단과 근거 대조

| # | 판정 | 근거 / 문제 |
|---|---|---|
| S1 | **PLAUSIBLE(조건부), 일부 OVERBROAD** | 진단(FEAS 가드 bs=47 ≥ 0.85×48, 거부 72–78회, 전환 0) 자체는 TUNE 감사 (2)와 일치한다. 그러나 "TUNE·EVAL·H의 동적 arm 전부"는 거짓이다. 같은 H Job의 `dyn_a4_w3-e12`는 switch 12/6/6/6, dwell D44 0.995 · D34 0.004 · D24 0.001이다(implfix RESULT l.83). EVAL은 실행되지 않았다(결정 60 흡수, dry-run만). 따라서 "루프 수정으로 안 풀림"은 "E1+E2 ON 한 구성에서 D44 체류 ≥0.995"까지만 지지된다. |
| S2 | **진단 CONFIRMED, "영향" 열 OVERBROAD** | 오염 자체는 implfix 결과 감사 1·8과 일치한다. 그러나 P의 UNTESTED_REMAINS는 오염과 무관하게 성립한다(raw 구간 +4.4% [−0.6, +9.5]가 UNRESOLVED_INTERVAL; 감사 3, RESULT l.105). E3의 `e3_handoff_ok`는 `pi_runner.py:187`에서 계산만 된다. `pi_analyze.py:330`은 E3 쌍을 `pc=None`으로 분석하므로 **어떤 라벨도 이 문턱에 걸리지 않는다**. 이번 캠페인에서 S2가 바꾼 판정 결과는 0이다. 심각도의 근거는 "판정 무효"가 아니라 "잠재 결함(다음 설계에서 거짓 라벨 생산)"이다. |
| S3 | **"S1의 직접 원인" REFUTED** | TUNE 감사 (2)의 원문은 "그 조건은 h=1.5 과부하에서 batch가 상한 48에 붙는 것으로 사실상 결정"이다. 가드 문턱은 0.85×cap으로 cap에 비례한다. 과부하에서는 어떤 cap이든 running batch가 상한에 붙으므로 cap을 재결정해도 가드는 열리지 않는다. 직접 원인은 h(λ\* 대비 과부하)다. 같은 저자의 HE0 메모 3도 "h와 cap을 따로 정하면 결과가 그 조합에 묶인다"고 적는다. "D34–64 동률이라 D44는 Nano 근거 없음"은 R Job과 일치한다(상위 3개 차이 1.5% < seed SD 3%). |
| S4 | CONFIRMED | TUNE (4), implfix §4 l.77과 일치한다. |
| S5 | **CONFIRMED, 단 문구 둘 수정** | sd 0.030 > 0.0186은 결과 감사 l.172와 같다. 수정 1: "S2 없이도 UNRESOLVED였을 가능성"이 아니라 **확정**이다(감사 3). 수정 2: 0.0186은 EQUIV 라벨 조건이다. 관측 효과 +0.043의 CI가 0을 넘으려면 sd < 0.027 정도가 필요하다(t(3) 반폭 = 3.182·sd/2). 필요 sd는 라벨 방향에 따라 다르다. |
| S6 | **정오 반영 CONFIRMED, 용어 수정** | 5.06–5.34는 RECHECK §6에 따르면 "7.29 과부하·큐 무한 증가 상태의 포화 완료율"이다. SLO를 지키는 지속 rate가 아니다. "지속 완료율"이라는 표현은 CAP2 감사의 금지 문안("지속 처리율 5.2 req/s")에 접근한다. "버스트 대비 약 1.4배"도 방향이 뒤집혔다. 정확한 문장은 "버스트 λ\*가 포화 완료율의 약 1.4배"다. |
| S7 | **REFUTED(경위)** | H 블록은 rev0 초안부터 `s44-e12` 묶음 arm만 있었다(rev0 l.39). 단일 플래그 arm이 예산 때문에 제거된 적은 없다. 제거된 것은 P 셀의 `e2k1`(정의상 PC_PACED 불가, 설계 사유)과 `e12`(Job 크기)다(rev1 l.116). PREREG는 "E2의 박자 영역 효과"를 답하지 않는 질문으로 §11(a)에 이미 적었다. 결과(−4%를 E1/E2로 귀속 불가, B-8)는 맞다. 경위가 틀렸다. confound 10이다. |
| S8 | PLAUSIBLE, 문구 과장 | RESULT l.156은 "handoff가 일어나는 시간은 run의 작은 부분"이라고 적는다. "기회 없음"은 과장이다. 또 이 arm 선택은 강제였다(튜닝 arm은 전환 0). "계획 오류"로 분류하기보다 대안이 없던 선택으로 적는 편이 정확하다. |
| S9 | **결함 아님** | 봉인 예측이 MISS한 것은 측정이 작동한 증거다. 문서 스스로 "정상"이라고 적는다. 결함 목록에서 빼거나 별도 절로 옮겨야 한다. 넣어 두면 결함 수가 부풀려진다. |
| S10 | CONFIRMED | gate rev0.2 결과 감사 2a와 일치한다. |
| S11 | CONFIRMED | 감사 3·4와 일치한다. |
| S12 | CONFIRMED | CAP2 (1), CAP1-R (1)과 일치한다. |
| **누락** | **S13 필요** | 이 기간 가장 멀리 퍼진 수치 오류는 감사 자신이 만들었다. `audit_r1_sus.py:16`이 `sent_s`를 이중으로 더했다. 오류 값 "마지막 완료 1405–1475 s"는 run 길이 821–866 s보다 길어 물리적으로 불가능하다. 이 값이 CAP2 감사, HE0 메모 2, PROPOSAL §4, `tracks/main.md`, Phase B PREREG §7 시간 모형으로 퍼졌다. REPORT는 S6 안에 정오만 넣었다. 결함 분류는 하지 않았고, claims-auditor 개정에도 "감사자 재계산 스크립트의 물리 상한·단위 검사"가 없다. |

### 1-2. 심각도 순서

- **S1이 1위인 것은 정당하다.** HE0 판정을 불가능하게 만든 유일한 결함이다.
- **S2가 2위인 것은 문서 자신의 기준("판정 무효 > Job 낭비 > 정본 오염")과 맞지 않는다.** 이번 캠페인에서 S2가 바꾼 판정 결과는 0이고, P 결과를 실제로 정한 것은 S5다. S2를 2위에 두려면 기준에 "잠재 위험"을 명시해야 한다. 아니면 S5를 S2 위로 올린다.
- **S3의 3위는 REFUTED된 인과("S1의 직접 원인")에 기대고 있다.**
- **S9는 목록에서 뺀다.**

### 1-3. P1–P6 원인 귀속

모두 **사후 분류**다. 반사실 검증이 가능한 것은 P2 하나뿐이다.

- **P1: S5 포함은 OVERBROAD.** P 셀 HI에서 s44의 TTFT≤SLO 비율은 0.77이고 e1ty 0.62, e1t 0.55다(RESULT l.116–119). 이는 새 "부하 위치" 대역 0.6–0.85 안이다. 새 규칙을 적용했어도 S5는 막히지 않았다. "S5의 sd가 HI 과부하에서 생겼다"는 원문에서도 [가정]이다(l.121).
- **P2: CONFIRMED.** 하네스 감사가 K1–K5 결정 규칙만 봤다는 사실은 기록과 일치한다.
- **P3: 구성원 귀속 OVERBROAD.** S7은 예산 탓이 아니었다. S8은 승계 문제가 아니라 강제 선택이었다.
- **P4·P5: 반증 불가한 동기 귀속.** P4는 2026-09-10 사용자 결정의 "부작용"이라는 반사실이고, P5는 "압력"이라는 심리 서술이다. 기록할 수는 있지만 판정 근거로는 쓸 수 없다.
- **P6: CONFIRMED.**
- **누락: P7(감사 계산 무검증, S13).**

### 1-4. 규약 개정: 진단에서 도출됐는가

규약 3파일은 `.claude/agents/`와 `tools/claude/agents/`가 diff 0이다(커밋 `fe6e3147`).

| 개정 | 판정 | 근거 |
|---|---|---|
| claims-auditor confound 12·13, 파이프라인 모집단 층 | 도출됨 | TUNE 로그, implfix 감사 1·8 |
| `GO-with-caveats(검출력)` | 도출됨 | S5 |
| **caveat 상한 5** | **도출 안 됨, 계수 단위 모호** | 과거 사례가 정확히 문턱 위에 있다. CAP1-R caveat 5개, implfix 인용 금지 5개다. 이 상한은 과거 어느 사례도 잡지 못했을 것이다. implfix 결과에 붙은 B-1·3·5·6·7·8·9·10, C1·5·6·10까지 세면 즉시 초과한다. 무엇을 세느냐에 따라 분기가 뒤집힌다. |
| **experiment-runner "부하 위치"(HI = best static attainment 0.6–0.85)** | **OVERBROAD, 수정 필수** | 세 문제가 있다. (아래 반전 표 참조) |
| **"승계 상수 → 판정 셀 불가"** | **과잉 일반화** | budget 65536은 SGLang 기본값이고, CONSENSUS의 결론 범위("SGLang 공개 PD-mux 구현, budget 65536")를 정의하는 상수다. 이를 재결정하면 연구 대상 자체가 바뀐다. 또 순서 "cap → λ\* → …"는 cap 재결정이 사용자 결정 대기(cap/mem 메모)인 상태에서 그 결정을 앞질러 버린다. |
| 처치 발생 횟수 | 도출됨 | 예시 문장 "기회 없음"만 과장이다. |
| **PREREG 필수 항 1(답하는 질문 목록)** | 규칙은 타당, **예시 REFUTED** | 예시로 든 S7 전례는 사실과 다르다. implfix PREREG는 이미 질문을 줄이고 공시했다(§11(a)). 이 규칙을 지킨 사례를 위반 사례로 인용한 셈이다. |
| PREREG 필수 항 2(검출력) | 도출됨 | 0.0186은 EQUIV용이라는 단서가 필요하다(S5 참조). 사전 sd가 없는 새 셀은 예비 측정(pilot)을 허용해야 실현 가능하다. |
| PREREG 필수 항 3, 분석기 phase 분리 | 도출됨 | 다만 `t_load_start`–`t_load_end` 필드가 runner에 실제로 있는지 확인되지 않았다. |
| result-analyst 게이트 4 차단형(< 0.5), "측정하지 않은 것" 절, cost model 2칸 | 도출됨 | H HI 0.34–0.37, CONSENSUS §1-1 선례, S11 |

**"부하 위치" 규칙의 세 문제**

1. **측정 셀이 정해져 있지 않아 판정이 뒤집힌다.** 반전 시험 표:

   | 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
   |---|---|---|---|
   | attainment를 재는 셀 | 용량 사다리 120요청 셀 ↔ 등록 변화 trace HI 구간 | S1 셀(HI 10.94)이 **허용 ↔ 거부로 반전** | CAP1-R 감사 #5: 11.09 rung에서 상위 D 0.74–0.78(대역 안). H RESULT l.76: HI TTFT≤SLO 34–37%(대역 밖). |
   | "best static" | P 셀 s44 | P 셀 허용. 그러나 판정 쌍은 검출력 미달(S5) | s44 0.77 |

   규칙을 그대로 두면 S1을 만든 바로 그 셀을 통과시킬 수 있다.

2. **기전이 측정되지 않았다.** 가드는 bs ≥ 0.85×cap에 걸린다. attainment 대역과 bs 분포의 관계는 측정된 적이 없다.
3. **사용자 결정을 앞질렀다.** HE0 메모 원칙 (b)의 두 선택지 중 하나를 이미 규약으로 박았다. REPORT §2 P1 자신은 (a)–(d)를 "사용자 결정 남음"으로 적는다. 서로 모순이다.

### 1-5. 이 문서의 confound

3(small-n: 사례 1–2건으로 심각도를 매김), 10(S7), 12·13(정확히 지목됨), 그리고 카탈로그 밖의 **감사자 계산 오류(S13)**.

### 1-6. 허용 문안

- "2026-11-01 결함 정리의 진단 중 S1(과부하 HI에서 FEAS 혼잡 가드가 선택 동적 arm을 D44에 고정)·S2(계측 boot 축약에 warmup·가드 프로브 혼입)·S4·S5·S10·S11은 출처 감사와 일치한다(감사 r1). 심각도 순위와 P1–P6 원인 분류는 사후 분류이며 판정이 아니다."
- "policy_implfix에서 S2 오염은 P 쌍의 결과(UNTESTED_REMAINS)를 바꾸지 않았다. E3 라벨은 오염된 handoff 문턱에 걸리지 않는다."

### 1-7. 인용 금지 문안

1. "승계 상수(cap 48)가 FEAS 가드 고정(S1)의 직접 원인이다."
2. "E2 단독 arm은 예산 때문에 제거됐다."
3. "HI를 best static attainment 0.6–0.85에 두면 동적 arm이 움직인다." (측정 셀 미지정, 기전 미측정)
4. "계측 오염(S2)이 P·E3 판정을 무효로 만들었다."
5. "W1 지속 처리율은 5.06–5.34 req/s다." (과부하 포화 완료율이며 SLO 기준 rate가 아니다)

### 1-8. 해소 방법

- **문서 수정.** S1·S2·S3·S6·S7·S8 문구를 위 판정대로 고친다. S9는 결함 목록에서 뺀다. S13과 P7을 신설한다.
- **규약 수정, 다음 PREREG 전에 필수.**
  - "부하 위치" 규칙에 측정 셀을 명시한다. "등록 변화 trace와 같은 구간 길이·라운드의 HI 구간 TTFT≤SLO 비율"로 쓴다.
  - 원칙 (b) 결정 전까지 이 규칙을 "잠정"으로 표기한다.
  - caveat 상한의 계수 단위를 정한다(인용 금지 문장만인지, 부착 문언 포함인지).
  - S7 예시를 교체한다.
  - "승계 상수" 규칙에서 공개 루프 정의 상수(budget 65536)를 제외한다.
- **실험.** 부하 위치 규칙을 실증하려면 한 Job·한 GPU에서 W1, static D44 + `dyn_a4_w3`를 h ∈ {0.8, 1.0, 1.2, 1.5}×λ\* 변화 trace(구간 200요청 × 3라운드)로 돌린다. 각 n=4 seed, 같은 trace hash로 맞춘다. 기록할 것: HI 구간 TTFT≤SLO 비율, 결정 시점 bs의 0.85×cap 미만 비율, `SLO-FEAS refused` 수, switch_count, dwell. 대역과 가드 개방이 실제로 겹치는지가 확인 대상이다. 이 실험은 원칙 (b)의 입력이다. 실행 여부와 h 격자는 사용자 결정이다.

### 1-9. 종합: **GO-with-edits**

死因은 0이다. 이 문서는 메모이고 판정을 정본에 올리지 않는다. 단 이미 적용된 "부하 위치" 규칙은 위 반전 때문에 **다음 PREREG가 인용하기 전에 고쳐야 한다**.

---

## 2. 두 메모와 implfix 결과 대조

### 2-1. `MEMO_implfix_impact_on_policy_campaign_2026-10-30.md`

| 주장 | 판정 | 근거 |
|---|---|---|
| W1 multi-chunk 5.5–9.4%, p50 36–51, p90 724–1092, 최대 3.5k–9.2k | **CONFIRMED(개수 기준, boot 전체 집계)** | 재계산 결과: 측정 boot 5.5–9.4%, 기준 셀 boot 4.3–5.7%. p90 716–1084(백분위 방식 차이), 최대 3,550–9,244. |
| "W1은 대부분 미발화 → E1/E2 영향 제한적일 것" | **REFUTED(추론), H 결과와 불일치** | 토큰 기준으로는 multi-chunk 배치가 prefill 토큰의 **51–61%**다(CAP1-R 5 boot 0.535–0.613, TUNE 5 boot 0.508–0.530; 재계산). 개수 기준 "드묾"이 prefill 작업량 기준으로는 과반이다. H Job: E1+E2 ON에서 HI −13.1% [−20.8, −5.4], dyn −14.6%, 라벨 `ON_BELOW_REF`. LO는 0으로, "작음"과 일치한다. |
| B1/E3 ≈ 0 | **PLAUSIBLE(조건부)** | E3 Job `EQUIV_WITHIN_3PCT` −0.31% [−0.90, +0.29]와 일치한다. 그러나 측정 arm의 물리 handoff 수는 미관측이다(부하 창 handoff 1건). 전제 "선택 arm 전환 0"은 OFF 루프에서만 성립한다. H2에서 E1+E2 ON은 같은 arm을 6–12회 전환시켰다. |
| E1 편차 가능성 최대 | **NOT-YET-SUPPORTED** | H는 E1+E2 묶음뿐이다(B-8, confound 10). 예상의 방향은 "TTFT↑ 또는 ↓" 양측이라 반증할 수 없다. "ITL 측 여지 작음"은 일치한다(ITL p95 49.2 → 48.8). |
| W3에서 E2 영향 큼 | **NOT-YET-SUPPORTED, 내부 충돌** | W3는 implfix에서 측정되지 않았다. 이후의 [유도] `results/impl_defects/expected_gain.py`(축 지도 §6b, `ac0720fe`)는 E2 효과를 "prefill-heavy 저부하, P(decode 빔)≈e^−b̄"로 한정한다. 그런데 PROPOSAL은 W3 용량점에서 b≈17이라고 적는다. 그 식대로면 E2 ≈ 0이다. 둘 다 미감사이고 서로 충돌한다. |
| OFF 경로 동일성 GPU 미검증 | CONFIRMED(공백 서술) | implfix C6는 여전히 CPU golden뿐이다. |
| §3 "어느 쪽도 쓰지 않는다" | CONFIRMED(절제가 적절했다) | 메모 자체가 측정으로 넘겼다. |

**confound:** 개수 대 토큰(시간) 가중 혼동(카탈로그 밖), 10, 11(H는 digest 0a5d9c3c·다른 GPU라 CAP/TUNE의 65cbb49d에 대한 함의는 C6 조건부), 5·12(HI 크기 인용 불가).

### 2-2. `MEMO_HE0_cell_load_redefinition_2026-10-30.md`

| 주장 | 판정 | 근거 |
|---|---|---|
| 셀 고정의 원인은 컨트롤러 규칙과 셀 부하이며 루프 결함이 아니다 | **기전 CONFIRMED** | TUNE 서버 로그의 FEAS 거부 72–78회. 수정 트리 OFF에서도 재현됐다(H `dyn_a4_w3` 전환 0, 다른 GPU). |
| "루프를 고쳐도 풀리지 않는다" | **엄밀형 REFUTED, 실질형 PLAUSIBLE** | H2(E1+E2 ON): 전환 0 → 6–12, D44 체류 0.995. 루프 변경은 가드 여유를 흔든다. 실질 고정은 유지됐지만, 시험된 수정은 E1+E2 한 구성뿐이다. |
| λ\*는 루프에 의존한다 | **NOT-YET-SUPPORTED(측정 없음)** | ON 루프 λ\*는 측정된 적이 없다. implfix 감사의 확정 실험 목록에도 "ON 루프 λ\* 별도 측정"이 있다. H의 HI 부호는 방향만 시사한다. |
| 항목 2·6 "지속 완료율 ≈3.0", "LO 2.74는 그 ≈0.9배라 경계 근처" | **REFUTED** | RECHECK: 포화 완료율 5.06–5.34, LO는 그 0.51–0.54배. "LO가 지속 처리율 근처라는 점은 h와 무관하게 남는다"도 함께 무너진다. |
| 항목 3 가드 문턱은 cap에 비례 | CONFIRMED | 0.85×cap, TUNE (2) |
| 항목 4 h를 낮추면 수정 영향이 커진다 | NOT-YET-SUPPORTED | 예상이다. H2는 h=1.5에서도 ON이 전환을 만든다는 것을 보였다. |
| 항목 6 HI goodput = 도착률의 16% | CONFIRMED | TUNE (4) |
| 항목 7 "원 HE0의 HI/λ\* 배율 미상" | 이 감사에서 미검증 | 판정 영향은 작다. |

**원칙 (a)–(d)의 근거 사실.** 원칙 자체는 결정하지 않는다.

- **(a) 기준량.**
  - 메모의 수치 입력(3.0, 0.9×)이 틀렸다.
  - "지속"에 대해 SLO를 지키는 측정값이 없다. 있는 것은 7.29 과부하의 포화 완료율뿐이다. RECHECK §6에 따르면 SLO 지속 rate는 5.1 이하라는 것만 알고 하한은 없다.
  - 저장소 안에 "지속"의 정의가 셋 있다. ① 오류값 3.0(폐기), ② 포화 완료율 5.06–5.34(CAP2 Job), ③ Phase B의 600 s 안정성 규칙 경계(PD-mux (3.65, 5.84), 다른 Job·이미지). ②와 ③은 기판이 섞이므로 함께 쓸 수 없다(confound 11).
  - Phase B 결정 ①("판정 셀 rate는 지속 기준")은 Phase B 셀 한정이다. HE0 원칙 (a)를 대신하지 않는다.
- **(b) HI 배율.** 대역 0.6–0.85를 어느 셀에서 재느냐에 따라 현 셀이 허용되기도 거부되기도 한다(§1-4 반전 표: 0.74–0.78 대 0.34–0.37). 대역과 가드 개방의 관계는 미측정이다.
- **(c) cap.** 과부하에서는 cap을 바꿔도 가드가 비례해서 움직인다. cap/mem 메모는 따로 결정 대기 중이다.
- **(d) 비교 방식.** H Job이 측정한 것은 "같은 절대 rate의 OFF 대 ON"이다(RESULT l.80 부착 문언). 각 루프의 자기 λ\* 기준 비교는 측정된 적이 없다.

**confound:** 12(중심), 감사 계산 오류의 전파(카탈로그 밖), 11.

### 2-3. 두 메모 공통

**허용 문안**

- "CAP1-R·TUNE boot 전체 집계에서 W1 prefill 배치의 5.5–9.4%(개수)가 multi-chunk였고, 그 배치가 prefill 토큰의 51–61%를 차지했다(감사 r1 재계산, warmup·기준 셀 포함 근사)."
- "선택 동적 arm의 D44 고정은 HI 과부하에서 FEAS 혼잡 가드가 이동을 거부한 결과다. 수정 트리 플래그 OFF에서도 재현됐고, E1+E2 ON에서는 전환 6–12회가 생겼지만 D44 체류는 0.995였다."

**인용 금지 문안**

1. "W1은 대부분 1 chunk라 E1/E2의 영향이 작다."
2. "HE0 LO 2.74는 지속 처리율 근처(≈0.9배)다." / "W1 지속 완료율은 약 3.0 req/s다."
3. "루프를 고쳐도 셀 고정은 풀리지 않는다." (시험된 것은 E1+E2 한 구성)
4. "W3에서 E2 효과가 크다."
5. "λ\*는 루프에 따라 바뀐다." (측정 없음)

**해소 방법**

- **배너.** 두 메모 모두 미감사 메모이므로 본문은 보존하고 배너를 단다. HE0 메모에는 항목 2·6의 3.0 → 5.06–5.34(포화 완료율) 정오를 단다. implfix 영향 메모에는 "§3 예상 중 'E1/E2 영향 제한적'은 H Job ON_BELOW_REF(HI)와 불일치, 토큰 기준 multi-chunk 51–61%"를 단다.
- **CPU 실험(지금 가능).** multi-chunk 비율을 등록 부하 창 안으로 제한하고, 개수·토큰·추정 prefill 시간 세 가중으로 다시 집계한다. `#new-token` 시각과 run 창을 대조한다.
- **GPU 실험.**
  - **E1/E2 귀속:** 한 Job에서 OFF / e1ty / e2k1 / e12 각 n=4 + 플라시보. 용량 아래 셀 하나와 HI 셀 하나. 부하 창 PC로 dose·r을 잰다. implfix 감사의 확정 실험과 같다.
  - **λ\* 루프 의존:** 같은 Job에서 OFF와 ON 각각 CAP 사다리를 돌린다.
  - **W3 E2:** W3 셀에서 OFF 대 e2k1, n=4. 기록할 것: 실행 전 P(decode 빔) 예보와, 실측 decode-empty 비율.

### 2-4. 종합: **GO-with-edits**

배너 두 개가 필요하다. 원칙 (a)–(d)의 결정은 사용자 몫이고, 위 근거 사실 정정을 입력으로 쓴다.

---

## 3. `reports/cost_model/PROPOSAL_dstar_model_2026-10-30.md` §4 사후 재현

### 3-1. 주장별 판정

| 주장 | 판정 | 정보량과 문제 |
|---|---|---|
| 구속 제약 W1 5/5 | **OVERBROAD** | 5개 중 3개(D34·D44·D64)는 "SLOT은 실측에서 TTFT 실패로 나타남"이라는 사후 사상으로 맞았다. SLOT은 측정상 따로 구별되는 신호가 없어서, ITL이 아닌 실패는 전부 적중으로 셈된다. 실질 정보는 하나다: ITL 벽이 D24와 D34 사이에 있다는 것. D16·D24 구간은 CAP1(다른 Job·GPU)에서 왔다(confound 11). 측정 구간은 S1 UNSTABLE인 120요청 버스트 셀이다(confound 5). |
| 구속 제약 W3 4/5 | **PLAUSIBLE, 정보량 낮음** | 평균 7k 토큰 프롬프트에서 큰 D가 TTFT에 묶인다는 것은 prefill ∝ L/P를 가정하는 어떤 모형도 낸다. λ\*_W3는 통과 여유가 1요청이다. CAP2 감사 반전에 따르면 문턱 0.875에서 best D가 D24로, floor를 평균으로 바꾸면 λ\* 0.225로 바뀐다. W3 쪽 적중과 "틀린 것"(상위 순서)이 모두 이 취약 구간에 있다. §4는 이를 공시하지 않았다. |
| W1 순서 D16 < D24 < D34 < D44 ≤ D64 | **PLAUSIBLE, 정보량 낮음** | 측정상 D44와 D64는 같은 구간 [7.29, 8.39)다. 모형의 D44 < D64(5.03 < 5.30)를 "≤"로 흡수했다. decode-heavy 부하에서 D에 대한 단조성은 사전 기대값이다. D16 < D24 부분은 Job 간 비교다. |
| 워크로드별 D\* 방향 반대 | **CONFIRMED(사실), 정보량 낮음** | 워크로드 구성(decode-heavy 대 prefill-heavy)만으로 나오는 방향이다. 모형을 시험하지 않는다. |
| W3 D64 불가 | **CONFIRMED(사실), 거의 항등식(N1 계열)** | S_TTFT = 3 × floor(D24, 즉 P=84에서 단일 요청 TTFT 중앙값)이다. 위 벽은 사실상 "q90 문서를 P=44로 처리하는 시간 > 3 × 중앙 문서를 P=84로 처리하는 시간"인지를 묻는다. 근사 [유도, 문서 자체 수치]: (12471/6469) × (84/44) ≈ 1.93 × 1.91 ≈ 3.7 > 3. 길이 분포와 SLO 정의의 산술이지 서빙 모형의 예측이 아니다. |

### 3-2. 오염 공시는 충분한가

**불충분하다.** 헤더와 §4 제목의 "사후 재현, 예측 아님" 공시는 정확하다. 다만 빠진 것이 넷 있다.

- 결과를 본 뒤 고정된 자유도 목록이 없다. 해당 후보는 A6(f > 5% 경성 규칙), ρ 띠 1.13–1.19, SLOT 제약의 도입과 SLOT→TTFT 사상, "맞은 것"을 세는 항목 선택과 분모(5·5)다.
- CAP1 대 CAP1-R의 Job·GPU 혼합이 공시되지 않았다.
- λ\*_W3 취약성(여유 1요청, 반전 시 best D 변경)이 공시되지 않았다.
- 실제 예측 시험은 R Job 채점이다(E-1…E-5 HIT, E-6·E-7 MISS, E-8 점추정 MISS; 플래그 OFF 대체물, 1 GPU). §4가 그 시험을 대신하지 않는다는 문장이 없다.

### 3-3. "지속 600 s 완료율 약 3.0" (l.99)

- 3.0은 RECHECK에 따라 단위 오류다. 정오는 5.06–5.34(과부하 포화 완료율)다.
- 정정하면 같은 문장의 "모델 값 5.0–5.3은 지속 과부하 완료율과 버스트 통과 rate **사이에** 있다"도 거짓이 된다. D44 모형 5.03(4.98–5.08)은 측정 5.06–5.23 아래쪽과 겹치고, D64 5.30은 5.15–5.34 안에 있다.
- **새 위험이 생긴다.** 정정된 수치가 모형과 거의 같아지면서 "W1 절대값도 맞았다"라는 두 번째 사후 재현을 부른다. SLOT 제약은 Little 법칙의 slot 처리율 상한(cap / (E[R]·T_eff + E[S] + W))이다. 그래서 포화 완료율과 같은 종류의 양이고, SLO 용량과는 다른 양이다. 일치 여부를 따지려면 사전 등록된 예측이어야 한다.

**배너 권고: 예.** 본문은 고치지 않는다. 봉인 대상(§5)은 커밋 `113de504`로 고정돼 있고, 원문 보존 선례는 CAP2 감사 배너다. §4 바로 아래(또는 헤더)에 배너를 단다. 축 지도 §8b는 이미 정정됐으므로 추가 조치가 필요 없다.

> 배너 문안 제안: "★정오(2026-11-02, `workspace/engine-port/results/policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`): §4의 '지속 600 s D44·D64 완료율 약 3.0 req/s'는 CAP2 감사 스크립트의 `sent_s` 이중 가산 오류다. 올바른 값은 7.29 req/s 과부하의 포화 완료율 5.06–5.34 req/s(goodput·SLO 지속 rate 아님)이며, 같은 문장의 '모델 값이 그 사이에 있다'는 무효다. 정정값과 모델 λ_cap(5.03·5.30)의 근접은 사후 관찰이며 W1 절대값 일치의 근거로 인용하지 않는다. 원문은 보존한다."

### 3-4. 허용 문안

- "D\* 식은 CAP1·CAP1-R·CAP2를 본 뒤 썼고, 그 대조(구속 제약 종류·W1 순서·D\* 방향·W3 D64 불가)는 사후 재현이다. 실질적으로 시험된 것은 W1 ITL 벽의 위치(D24–D34 사이)다. 예측 시험은 R Job 채점(E-1…E-5 HIT, E-6·E-7 MISS, E-8 점추정 MISS; 플래그 OFF 대체물, 1 GPU)뿐이다."

### 3-5. 인용 금지 문안

1. "D\* 모형이 구속 제약을 W1 5/5·W3 4/5로 맞혔다." (사후 사상 포함, 예측 아님)
2. "W3 D64 불가는 모형의 예측력이다." (SLO 정의와 길이 분포의 산술)
3. "모형 λ_cap 5.03·5.30은 W1 지속 포화 완료율 5.06–5.34와 일치한다." (사후 관찰, 다른 양)
4. "D\*가 워크로드에 따라 반대로 움직인다는 것을 모형이 보였다." (워크로드 구성으로 나오는 방향)

### 3-6. 해소 방법

- §4 배너(위 문안).
- §4 오염 공시를 보강한다. 사후 고정 자유도 목록, Job 혼합, λ\*_W3 취약성, "예측 시험은 R Job 채점"을 추가한다.
- **예측 시험.** 데이터 전에 새 봉인을 건다. 예: 한 Job·한 GPU에서 W1 static D34·D44·D64 × 7.29 req/s × 600 s × seed {201, 202}. "D34 포화 완료율 ∈ 모형 SLOT λ_cap 4.82–4.91 ± 오차 띠", "D34/D44 비 ≈ 0.966"을 사전 등록하고 D44를 같은 Job 기준(anchor)으로 둔다. 식 수정(§7-2, A7 등)은 R 채점 뒤 별도 봉인으로만 한다. 이 순서는 문서 자체 절차와 같다.

### 3-7. 종합: **GO-with-edits**

죽은 결론은 없다. 문서가 스스로 예측이 아니라고 공시했기 때문이다. 단 배너와 공시 보강 전에는 §4를 근거로 인용하지 않는다.

---

## 4. 전체 요약

| 대상 | 회차 | 종합 |
|---|---|---|
| `reports/audit/2026-11-01_design_defects/REPORT.md` + 규약 3파일 | r1 | **GO-with-edits**. 단 "부하 위치" 규칙은 다음 PREREG 전에 측정 셀을 명시하고 잠정 표기해야 한다(반전 확인됨). |
| `MEMO_implfix_impact…`, `MEMO_HE0_cell_load…` | r1 | **GO-with-edits**. 배너 2개가 필요하다. 원칙 (a)–(d)는 사용자 결정이며, 위 근거 사실 정정을 입력으로 쓴다. |
| `PROPOSAL_dstar_model_2026-10-30.md` §4 | r1 | **GO-with-edits**. §4 배너와 공시 보강. 축 지도 §8b는 이미 정정됐다. |

## 관련 파일

- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/audit/2026-11-01_design_defects/REPORT.md`
- `/home/wonho/Experiments/KISTI/.claude/agents/experiment-runner.md`, `result-analyst.md`, `claims-auditor.md` (사본 `/home/wonho/Experiments/KISTI/prefill-layer-alloc/tools/claude/agents/`, diff 0)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/policy_tune/MEMO_implfix_impact_on_policy_campaign_2026-10-30.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/policy_tune/MEMO_HE0_cell_load_redefinition_2026-10-30.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/reports/cost_model/PROPOSAL_dstar_model_2026-10-30.md` (l.99)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/scripts/costmodel/render_dstar_section.py` (l.121, §8b 이미 정정)
- implfix(HEAD `96793350`): `workspace/engine-port/results/policy_implfix/RESULT_policy_implfix_2026-11-01.md`, `AUDIT_result_policy_implfix_r1_2026-11-01.md`, `PREREG_POLICY_IMPLFIX_rev0_DRAFT_2026-10-31.md`(l.39, H는 처음부터 묶음 arm), `pi_runner.py:187`, `pi_analyze.py:330`
- 재계산 원자료: `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/cap_char/policy_runs/pdmux-policy-{cap1r-4bef3cf-20261029003539,tune-074ffc3-20261029140207}/**/server.log`

> 저장: main 세션이 claims-auditor 최종 보고 원문을 그대로 옮김(2026-11-08).
