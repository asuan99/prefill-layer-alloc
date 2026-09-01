# PREREG — G1 프로브 (CP-0 용량 축의 **식별가능성**)


> ⚠️ **날짜 정정**: 파일명/본문 날짜 2026-08-28은 오기 — 실제 작성/실행일 **2026-09-01**. 상세: `DATE_CORRECTION_NOTE.md`.

2026-08-28 · 메인 세션 · **사전등록 초안 — 규칙층 5회차 감사 대기** · GPU **미제출** ·
**성능 판정 0건** · 정본 변경 0건 · 정책 순위 변경 0건

> `OVERRIDE_P1_SUBMIT_2026-08-28.md`가 *"G1 프로브 금지 — 그것은 **측정**이고 자체 등록이
> 필요하다"*고 명시했다. 이 문서가 그 등록이다.
> ★**불변 승계**: HE0 · 정책 순위 · gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 ·
> C2 인용정지 (a)(b).

## 0. 이 프로브가 사지 **않는** 것
정책 판정 0건 · arm 순위 0건 · goodput 0건 · chunked prefill에 대한 어떤 주장도 없다.
결정하는 것은 **하나**: *"CP-0이 등록한 포화 축이 이 워크로드에서 식별되는가."*

## 1. 왜 사야 하는가 — 4회차 감사 G1·G2의 산술

- **G1**: `ATTAINMENT_THRESHOLD = 0.80`이 유도됐다는 knee 밴드는 **폭 0.09**인데, 같은 통계량의
  **무부하 seed 변동은 SD 0.108**이다(offered rate 1, completed 60/60, 3 seed:
  attainment **1.075 / 0.957 / 0.859** — ★하나는 **1을 넘는다**, 분자 duration과 분모 nominal
  rate의 시간축이 다르다는 뜻이다). **자기 잡음보다 좁은 밴드에서 유도한 임계는 knee를 식별하지
  못한다.**
- **G2**: 그 유도는 **다른 워크로드**(`random-ids in2000/out96`, 요청당 ≈2096 tok)에서 왔고
  CP-0은 ShareGPT(≈578 tok)를 돈다 — **3.6× 가벼운 부하**인데 격자 {2,4,8}은 **절대 req/s**다.
  비율 임계는 이식 가능해도 **격자는 이식 불가**이고, 라벨을 정하는 것은 격자다.
- **부수**: 정본 하네스가 `--seed 1`을 고정하므로 CP-0의 3 rep은 **한 arrival realization의 3회
  복제**다 ⇒ 그 위의 paired CI는 이 분산을 **원리적으로 담지 못한다**.

⇒ 이 프로브가 사는 것: **① 이 워크로드의 실제 용량 ② attainment의 seed 분산 ③ 임계가 그 분산
밖에 있는지.** 셋을 한 잡에서 산다.

## 2. 설계

| | |
|---|---|
| arm | `fused_default`(cps 플래그 없음 → 8192) · `cp512` — 가장 빠를 것으로 기대되는 것과 가장 느릴 것으로 기대되는 것 |
| rate | **{2, 3, 4, 6, 8}** — 등록 격자 {2,4,8}을 포함하고, 참조 스캔에서 knee가 앉았던 두 점(3·4)을 더한다 |
| seed | **{1, 7, 17}** — ★rep이 아니라 **arrival realization**이다. 1은 정본 하네스 seed, 7·17은 저장소 capscan의 seed 관례 |
| 부팅 | 2 arm × 3 seed = **6 부팅**, 각 부팅이 5 rate를 순차 서빙. 1 job(같은 노드) |
| 워크로드 | ShareGPT, NP=200, `--sharegpt-context-len 4000` — CP-0과 **동일 모집단**(`served_population.json`) |
| 플래그 | CP-0 §6과 동일. ★**전 arm `--disable-piecewise-cuda-graph`** — 이 프로브는 용량을 재지 기전을 관측하지 않으므로 cps가 유일한 레버여야 한다(P1과 반대 방향, 사전등록 §6의 단계별 규약) |

**추출**: rate 구간별 `completed / duration`(duration은 **합산**, `CLAUDE.md` 게이트 **#7**),
그리고 그것을 offered로 나눈 attainment. seed SD는 **(arm, rate) 셀마다 3 seed에 대해** 구한다.
knee 밴드 = "아직 오르는 마지막 rate"와 "plateau 첫 rate"의 attainment 차.

## 3. 결정 규칙

`cp0_g1_rule.py`(`RULE_REV=1`, 18세계). 축: `coverage` · `band_vs_sd` · `grid_brackets`.
실질 라벨 **4개**: `AXIS_IDENTIFIED` · `AXIS_IDENTIFIED_REGRID` · `AXIS_NOT_IDENTIFIED` ·
`NO_KNEE_IN_RANGE`. 비실질: `NOT_MEASURED` · `COVERAGE_INCOMPLETE`(게이트 #21) ·
`IMPOSSIBLE_WORLD`.

라벨 표는 **`cp0_expected_labels.json`에 오라클로 등록**됐고 `cp0_selftest.py`가 세 규칙 전부에
대해 그것과의 일치를 **첫 검사**로 돌린다.

### ★도달가능성 인증을 **발급하지 않는다** (그리고 그것이 규율이다)
G1에는 **증거에 근거한 제약이 없다** — 있으면 프로브가 필요 없다. 제약 없는 spec을 등록하면
도구의 `RESTRICTIONS_INERT` 검사가 `if restrict and _inert(...)` 때문에 **아예 돌지 않아** 통과한다
(4회차 감사 L-c/L1이 실증). 실제로 `coverage`만 제약한 spec을 만들어 돌려 보니
**`VERDICT: RESTRICTIONS_INERT`** — 그 제약이 실질 라벨을 하나도 배제하지 못했기 때문이다.
⇒ `RETRACTION_reqinactive_2026-08-28.md`가 세운 규율대로 **spec과 인증서를 삭제했다.**
G1의 도달가능성은 **전체 격자**(실질 4라벨 전부)이고, 도구 자신의 문구대로 그런 실행은
*"could not have failed"*이므로 인증으로 세지 않는다.

## 4. 예산 — ★추정(실측 근거 없음)
6 부팅 × (부팅 ≈33 s + 5 rate ≈ 275 s) ≈ **31분 ≈ 0.55 GPU-hr**. 부팅 33 s는 실측
(Zamba2-2.7B n=10 중앙값), rate 구간 소요는 **추정**이다. ★**1 부팅의 실소요를 먼저 재고**
나머지를 낸다(`PROJECT_STATUS.md` "방법론 게이트" **#26**).

## 5. 금지 문장
- *"G1이 chunked prefill에 대해 무엇을 보였다"* — 이 프로브는 arm을 재지 않는다.
- *"G1의 용량 수치를 정본 용량으로 쓴다"* — 2 arm · 3 seed · 1 노드다.
- *"`AXIS_NOT_IDENTIFIED`이므로 CP-0은 불가능하다"* — 그것은 **이 축이 이 격자·이 임계로는
  식별 안 된다**는 뜻이고, 다른 축 설계(예: 절대 처리량 plateau 검정)는 별개다.
- *"`AXIS_IDENTIFIED`이므로 CP-0 본체를 제출할 수 있다"* — 4회차 감사의 나머지 死因
  (G2 격자 이송 · G5 오라클[해소됨] · G6 접기[해소됨] · G3 · G4 · G7)이 독립으로 남는다.
- *"seed 3개면 분산을 안다"* — SD는 df=2 추정이다. 이 프로브가 답하는 것은 **밴드가 SD보다
  넓은가**라는 부등호이지 SD의 값이 아니다.

## 6. 이 프로브가 **해결하지 않는** 것
G2(격자의 절대단위 이송)는 **부분만** 닫힌다 — 이 워크로드에서 격자를 다시 고를 수는 있으나
*"워크로드가 바뀌면 격자를 다시 사야 한다"*는 구조는 남는다. G3(인증이 물리 아닌 격자에서
돎)·G4(F3 수리 실재성)·G7(동기의 워크로드 이송)은 이 프로브와 무관하다.
