# 감사 판정서 — P0-A **하네스층** 적대 감사 (2026-08-23, claims-auditor, 게이트 #34 **2단계**)

## ★ 메타 한 줄 (사용자의 "이대로 0.1 GPU-hr을 쓸 것인가" 결정용)

**`GO-with-conditions` — 차단 9건(G1–G9) · 비차단 12건(N1–N12) · 버틴 것 14건(B1–B14).**
**차단 9건 전부 GPU 0 · 코드 수십 줄 + 문서로 닫힌다.** 규칙층 재감사 아님(§2·§4·§5 불변).

* ★**가장 중요한 좋은 소식**: 감사가 먼저 깨라고 지정한 **1번(계약표 전수성)의 위험한 방향**
  — *"어댑터/표가 읽는데 `run()`이 쓰지 않는 필드"*(S-6 F2 형태, 예산 전액 결정론적 소실) —
  은 **없다**. `run()`의 AST에서 leg 키·driver 스트림 키를 뽑아 어댑터 상수와 대조했고
  **전부 일치**한다(B1). 또 **거짓 `PRESERVED`를 만드는 경로도 찾지 못했다**(B4·B5·B6).
* ★**차단 중 판정을 실제로 바꾸는 것은 3건**:
  **G1**(`escape_replicated`가 **서로 다른 라벨**로도 참 ⇒ *스윕당 잡음 1라벨*이 이 프로브의
  **최고가 판정** `LOST (PARTIAL)`+즉시 정본 회부를 만든다. **실측 재현**) ·
  **G2**(`graph_plain`의 캡처/replay 상태를 **아무도 읽지 않아** 도구 실패가 결정량 게이트
  실패 `NULL CHANNEL OPEN`으로 라벨된다. **실측 재현**) ·
  **G3**(`NOCAP`이 서로 다른 세 사건 — 레그 부재 / **평범 side 스트림의 warmup 실패** /
  green 캡처 거부 — 을 한 라벨로 뭉개고, 판정서 `.json`에는 구분할 필드가 **0개**. **실측 재현**).
* ★**나머지 6건은 "가드의 가드"와 정직성**: **G4**(변이 스위트에 `intact != mutated` assert가
  없다 = 4회차 **E6**가 규칙 파일에서 잡아 `T5′`로 고친 그 결함이 **하네스로 이사**. 감사가
  퇴화 witness를 만들어 `MUTANTS ALL PASS`를 **실측**) · **G5**(어댑터 계약이 `run()`에
  기계적으로 묶여 있지 않다 — 오늘은 맞지만 지키는 것이 없다) · **G6**(sbatch가 stage 3 rc를
  안 본다 = 정본이 R0에서 이미 등재하고 미수정으로 남긴 결함의 **복제**) · **G7**(§10 완료
  상태표의 검증 방법 `grep -in physical = 0`이 **재현되지 않는다** — E6/교훈 #67의 형태) ·
  **G8**(§8 "전수 목록"이 전수가 아니다) · **G9**(§9-3·§9-4가 기록만 되고 대조되지 않으며
  참조값이 어디에도 없다).

★**추세 판정**: 규칙층 5회차의 *"수리가 다음 회차 최고 차단을 만든다"* 패턴은 **하네스층에서
약해졌다**. 이번 차단 중 규칙층 수리의 부작용은 **1건(G1)** 뿐이고, 나머지는 **새 코드의
가드 결손**이다. 남은 위험 표면은 사전등록이 말한 대로 **어댑터 한 곳으로 수렴했고**, 그
어댑터의 배선 자체는 **오늘 옳다**(실측). ⇒ ★**3회차 하네스 감사는 권하지 않는다.** G1–G9를
닫고 `--selftest`가 통과하면 **제출한다**.

---

## 판정어: `GO-with-conditions`

GPU 실행 **0** · job 제출 **0건** · **대상 파일 수정 0건**(읽기 전용; 이 판정서와
`/tmp` 스크래치 사본만 신규) · 새 성능 판정 **0건** · 정책 순위 변경 **0건** · **HE0 불변**.

> **대상**: `p0a_graph_sm_confinement.py`(1,304줄, 2026-08-23 전면 재작성) ·
> `p0a_graph_sm_confinement.sbatch`(113줄) · `PREREG_P0A_2026-08-22.md` **rev7**(756줄)
> §10 / §10.1 / §10.2 / §13.
> **참조(감사 대상 아님, 수정 0)**: `p0a_rule_totality.py` · `../smid_census/smid_l0_census.py`.
> **범위**: 하네스층만. 규칙층 결함은 **비차단으로 분리 등재**(N11·N12).
> **합격 기준**(사전등록 §13): *"그대로 0.1 GPU-hr에 실행했을 때 판정이 (a) 어댑터의 잘못된
> 배선 (b) 측정 실패의 결과 라벨링 (c) 항등식 검사 (d) 사후 재해석으로 만들어질 경로가 없다.
> 남는 한계가 정직하게 등재돼 있으면 합격."*
> — **(a)는 열려 있지 않다**(B1 실측). **(b)는 두 곳 열려 있다**(G2·G3 — 다만 어느 쪽도
> **실질 라벨**을 만들지는 않는다: 전부 `UNDETERMINED` 계열이다). **(c)는 가드층에 하나**
> (G4 — 현재 9행 전부 판별하므로 잠복). **(d)는 하나**(G1 — 같은 관측이 두 산문 중 어느 것을
> 읽느냐로 `LOST (PARTIAL)`↔`LOST (UNREPLICATED)`로 갈린다). **"정직한 등재"는 미충족**
> (rev7 헤더의 *"닫히지 않은 것"* 목록에 아래 9건이 하나도 없다).
> ⇒ 그래서 `GO`가 아니고, 전부 GPU 0으로 닫히므로 `NO-GO`도 아니다.

★**감사가 직접 실행한 것**(프로젝트 venv):
* `python p0a_graph_sm_confinement.py --selftest` → **`ALL PASS`**, rc=0,
  **A5 = 516,096 세계 전수 통과**, 변이 9/9, R1–R3 + 판독 변이 2/2. **벽시계 193 s**
  (= §12의 *"stage 1 ≈3분"* 재현 ✅, 로그인 노드).
* 어댑터 AST 대조(`run()`의 leg/stream 리터럴 vs 상수) — **일치**.
* AST 가드 변이 **4종** 실측(직접 누출·신규 헬퍼 세탁·금지 driver 키·`CEN.*` 세탁).
* 라벨 라우팅 실측 **7종**(graph_plain partial/fail, graph_green partial/replay-partial,
  leg 부재, warmup 실패, 서로 다른 라벨 2-스윕 탈출).
* 변이 스위트 **퇴화 witness 주입** → `MUTANTS ALL PASS` 실측(G4).
* 상류 사실 대조: `divide_sm(108,(8,0),2)[0]=(74,34)` · `initialize_stream_groups`의
  `create_greenctx_stream_by_value(prefill_sm, decode_sm)` 순서 · R0 job 889631 원자료.

---

# 차단 조건 (닫기 전 제출 금지)

## **G1 — ★★`escape_replicated`가 "서로 다른 라벨"로도 참이 된다: 스윕당 잡음 1라벨이 이 프로브의 최고가 판정을 만든다**

**위치**
* `p0a_graph_sm_confinement.py:542-554` (`_escape_replicated`) — 특히 `:553-554`
  `per = [((s - base) & pre) for s in _leg_sweep_sets(...)]` / `return len(per) >= 2 and all(bool(x) for x in per)`
* 규칙 `p0a_rule_totality.py:202,206-213` (`E_attrib` → `REPL` → `LOST_FULL/LOST_PART`)
* 사전등록 **§5 표 3행**(`:411`) *"탈출이 **한 sweep에만** 나타남"* ↔ **§8-25**(`:549`)
  *"둘 다 비지 않을 때만 참"* — **같은 술어가 아니다.** 하네스는 §8-25를 구현한다.

**무엇이 문제인가.** 잡음의 **가장 싼 형태**는 "스윕마다 엉뚱한 라벨 하나"다. 그 형태는
`per = [{40}, {41}]`이 되어 `all(bool(x))`가 참 ⇒ `escape_replicated=True` ⇒
`LOST (PARTIAL)` ⇒ §5.2-(b) **즉시 정지 + 정본 재작성 회부**(이 프로브가 만들 수 있는
가장 비싼 결과)를 받는다. 반면 **같은 라벨이 한 스윕에만** 보이면 `LOST (UNREPLICATED)`
(회부 없음)로 훨씬 싸게 처리된다. ⇒ **재현성이 더 약한 관측이 더 비싼 판정을 받는다.**
3~5회차가 세 판 연속 죽인 *"1라벨 잡음 → 최고가 판정"* 이 **replication 층으로 이사했다.**

**실측 재현**
```python
w = RULE.World(RULE.FULL, RULE.FULL, RULE.GREEN, RULE.GREEN|{40,41}, S_egp=set(range(34,108)))
raw = H.raw_from_world(w)
raw["legs"]["graph_green"]["sweeps"] = [H._fake_sweep(RULE.GREEN|{40}),
                                        H._fake_sweep(RULE.GREEN|{41})]
H.score(raw)["verdict"]        # -> CONFINEMENT_LOST_THROUGH_GRAPH_REPLAY (PARTIAL)
H._escape_replicated(raw)      # -> (True, [[40], [41]])
```

**닫는 법(GPU 0)**: 두 술어를 **둘 다 계산해 둘 다 보고**하고, **어느 쪽이 회부를
게이팅하는지 실행 전에 사전등록에 한 문장으로 고정**한다.
권고: `escape_replicated_same := per[0] & per[1] ≠ ∅` 를 **`LOST (FULL|PARTIAL)`의 조건**으로,
현행 `any`(`escape_replicated_any`)는 **서술**로. `p0a_rule_totality.py` 세계 축 1개
(`repl ∈ {same, any_only, none}`) + 변이 1개(`replication_same_to_any`)를 함께 등록.
★그렇게 하지 않을 거라면, **§5 표 3행의 산문을 §8-25 공식에 맞춰 고쳐** 사후 재해석
경로를 없애라. **결과를 본 뒤에 고르면 §9-5 파기다.**

---

## **G2 — ★★`graph_plain` 레그의 `capture_status`/`replay_status`가 생산되지만 어디서도 읽히지 않는다: 도구 실패가 결정량 게이트 실패로 라벨된다**

**위치**
* 생산: `p0a_graph_sm_confinement.py:262-263` (`_leg_graph`가 **모든** graph 레그에 대해 기록)
* 소비: `:583-584` — `_tool_status(legs.get("graph_green"), ...)` **graph_green만**
* 계약표: 사전등록 **§10.2 5행**(`:625`)이 `legs.graph_green.capture_status/replay_status`
  **만** 적는다 ⇒ **계약표가 "존재하는 graph 레그 전체"에 대해 전수가 아니다**(감사 1번 항목).

**무엇이 문제인가.** `graph_plain`도 **replay 레그**다(§3-A: *"graph replay, 2차 대조(P3)
+ `E_null`의 한쪽"*). 그 레그의 캡처가 **부분 실패**하면 `S(gp)`가 좁아지고, 규칙은
`Δ_null = S(gp) △ S(ep) ≠ ∅`을 보고 **`UNDETERMINED (DECISION QUANTITY'S NULL CHANNEL OPEN)`**
을 찍는다 — 즉 *"결정 함수의 거짓양성 채널이 열렸다"* 는 **진단 결론**으로, 실제 원인인
*"평범 스트림 캡처가 25쌍 중 일부만 됐다"* 는 **도구 사실**과 다르다. 전면 실패면
`S(gp)=∅` ⇒ P0 ⇒ **`MEASUREMENT ABSENT`**(역시 `NOCAP`이 아니다).
게이트 #21의 라벨층 형태이며, **원인을 말해 줄 필드가 원자료에 이미 있는데 읽지 않는다.**

**실측 재현**
```python
raw = H.raw_from_world(H._healthy())
raw["legs"]["graph_plain"] = H._fake_leg([set(range(100))], True, 1, mode="graph",
                                         capture="partial", replay="partial")
H.score(raw)["verdict"]   # -> UNDETERMINED (DECISION QUANTITY'S NULL CHANNEL OPEN)

raw["legs"]["graph_plain"] = H._fake_leg([set()], True, 0, mode="graph",
                                         capture="fail", replay="fail")
H.score(raw)["verdict"]   # -> UNDETERMINED (MEASUREMENT ABSENT)
```

**닫는 법(GPU 0)**: 어댑터가 `graph_plain`의 두 상태도 읽어 **`Δ_null` 게이트보다 앞에서**
`NOCAP`/`NOREPLAY`로 라우팅한다(규칙 순서 `P3 → [gp CAP/REP] → Δ_null`).
최소 대안: §5.3 표에 *"평범 graph 레그의 캡처 실패는 `NULLCH`/`ABSENT`로 나타나며, 그
라벨을 인용할 때는 원자료의 `legs.graph_plain.capture_status`를 함께 읽어야 한다"* 를
**사전등록 문장으로** 넣는다. 어느 쪽이든 §10.2 계약표에 행을 추가한다.

---

## **G3 — ★★`NOCAP`이 세 개의 다른 사건을 뭉갠다: (i) 레그 부재 (ii) 평범 side 스트림의 warmup 실패 (iii) green 스트림 캡처 거부. 판정서 `.json`에 구분할 필드가 0개**

**위치**
* `p0a_graph_sm_confinement.py:514-519` (`_tool_status` — leg가 `dict`가 아니면 `"fail"`)
* `:189-191` — **warmup 예외**가 `ev["capture_exc"] = f"warmup: {exc!r}"` 로 기록된다.
  warmup은 `self.side`(**평범** 스트림) 위에서 **스크래치**로 도는 **eager** 실행이므로,
  그 실패는 *"green 스트림에서 그래프 캡처가 불가능하다"* 와 **아무 관계가 없다.**
* `:296-302` + `:386,391,405,409,412,415` — 증분 flush 설계상 **벽시계 kill 시 레그가 통째로
  빠진 원자료**가 디스크에 남는 것이 **정상 산물**이다. 그 원자료를 채점하면 `NOCAP`이 나온다.
* `:664-706` (`_diagnostics`) — `capture_events` / `capture_exc` / `replay_exc`를
  **소비하는 코드가 파일 전체에 0곳**(`grep -n "capture_exc\|capture_events"` 확인:
  기록 4곳, 소비 0곳). 판정서 `.json`에 실패 사유가 **들어가지 않는다**.

**무엇이 문제인가.** §5.3은 `NOCAP`을 *"green 스트림 위 그래프 캡처 가용성"* 이라는
**도구 사실**로 등록했다. (i)과 (ii)는 그 문장이 아니다. 실질 라벨을 만들지는 않으므로
합격 기준 (b)의 최악형은 아니지만, **판정서만 읽어서는 어느 사건인지 알 수 없다**
— 정본에 *"green 스트림에서 캡처가 안 됐다"* 로 등재될 위험이 그대로다.

**실측 재현**
```python
raw = H.raw_from_world(H._healthy()); del raw["legs"]["graph_green"]
H.score(raw)["verdict"]   # -> UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM)
#   adapter: capture_status='fail' replay_status='fail'
#   'absent'/'exc'/'event'를 담은 필드: NONE
# warmup 실패 픽스처(capture_exc="warmup: ...")도 같은 라벨, 판정서에 exc 미포함(False)
```

**닫는 법(GPU 0)**: (a) `_tool_status`가 **레그 부재**를 `"absent"`로 구분하고 규칙이
그것을 `ABSENT`로 라우팅 · (b) warmup 실패에 **자체 필드**(`warmup_ok`)를 주고
`capture_exc`에 섞지 않는다 · (c) `_diagnostics`에 graph 레그별
`{expected, captured, replayed, warmup_failed, first_capture_exc, first_replay_exc}`
요약을 넣어 **판정서 `.json` 하나로 §5.3의 라벨과 그 원인을 함께 읽게** 한다
(§11: *"판정서 전 필드가 `.json`에 완전히"*).

---

## **G4 — ★★변이 스위트에 `intact != mutated` 판별 assert가 없다 (4회차 E6가 규칙 파일에서 잡아 `T5′`로 고친 결함이 하네스로 이사)**

**위치**: `p0a_graph_sm_confinement.py:1096-1116` (`selftest_mutants`) — `got_intact == intact`
와 `got_mut == mutated`만 검사한다. 규칙 파일의 대응물은 있다:
`p0a_rule_totality.py:424-427` (`T5′ 위 witness는 반드시 판별해야 한다`).

**무엇이 문제인가.** 사전등록 §10.2가 스스로 *"이것이 이제 이 프로브의 위험 표면 전체다"*
라고 적은 계약표를 지키는 유일한 기계는 이 9개 변이다. 그 스위트가 **판별하지 못하는
witness를 통과**시키면, *"9/9 변이 통과이므로 계약이 옳다"* 는 문장(rev7이 금지어로 적어
둔 바로 그 문장)이 **기계적 근거 없이** 만들어진다. 교훈 #53/#70의 정확한 형태.

**실측 재현**(감사가 실행): `capture_fails_open`의 witness를
*"P1이 CAP보다 먼저 발화해 intact도 mutant도 `ABSENT`"* 인 퇴화 witness로 바꾸고
표를 `(RULE.ABSENT, RULE.ABSENT)`로 선언 →
```
[PASS] capture_fails_open: intact=UNDETERMINED (MEASUREMENT ABSENT)
[PASS] capture_fails_open: mutant=UNDETERMINED (MEASUREMENT ABSENT)
MUTANTS ALL PASS        (rc = 0)
```
스크립트 사본: `/tmp/.../scratchpad/harness_degenerate.py`(감사 스크래치, 대상 트리 무수정).

★**현재 9행은 전부 판별한다**(위 셀프테스트 로그로 확인) — **잠복 결함이지 활성 결함이 아니다.**

**닫는 법**: `selftest_mutants` 루프에 한 줄 —
`ck(f"{name}: 두 라벨이 다르다", intact != mutated)`.

---

## **G5 — ★어댑터 계약이 `run()`에 기계적으로 묶여 있지 않다 (A5는 원리적으로 이 드리프트를 못 잡는다)**

**위치**
* 어댑터/픽스처 공유 상수: `:113-124` (`DECISION_LEGS`, `WORLD_LEG`, `GREEN_STREAMS`,
  `PLAIN_CONTROL_STREAMS`) — `_fake_driver(:780-786)`와 `_attached(:522-532)`가 **같은 상수**를 쓴다.
* `run()`은 **문자열 리터럴**로 쓴다: `:381, :389, :408, :410, :413` (leg 키) ·
  `:400-404` (driver 스트림 키).
* A3d(`:1042-1043`)는 **픽스처 키 vs `DECISION_LEGS`** 만 비교한다 — `run()`은 보지 않는다.
* 스윕 수도 미결속: `meta["sweeps_per_green_leg"]=2`(`:354`)를 어댑터가 **읽지 않는다**.

**무엇이 문제인가.** 이것이 감사 지시 4번(*"A5가 항등식인가"*)의 **정확한 답**이다.
A5는 **항등식이 아니다**(B10 참조 — 스키마·포화·부착·규칙은 전부 생산자/규칙 것이다).
그러나 **레그 이름과 스트림 이름 축에서는 픽스처와 어댑터가 같은 상수를 공유**하므로,
`run()`이 그 축에서 어긋나도 **516,096 세계 전수 통과가 그대로 유지된다.**
어긋난 결과는 fail-closed(`ABSENT`/`NOATT`)이므로 **거짓 판정은 아니지만 예산 전액이
조용히 소실된다** — 규모만 작을 뿐 S-6 F2와 같은 형태다.

**감사가 확인한 것(오늘은 옳다)**: `run()`의 AST에서 뽑은 값이 상수와 **완전히 일치**한다.
```
run() legs   = {eager_plain, graph_plain, eager_green, eager_green_prefill, graph_green}
             = DECISION_LEGS = set(WORLD_LEG.values())            -> True
run() driver = {green_prefill, green_decode, plain_pre, plain_post}
             = GREEN_STREAMS | PLAIN_CONTROL_STREAMS              -> True
```
⇒ **지금 배선은 옳다. 지키는 것이 없을 뿐이다.**

**닫는 법(GPU 0, ~15줄)**: `selftest_adapter`에 A7 신설 — `ast`로 `run()`의
`legs[...]`/`cross[...]` 대입 키와 `CEN.driver_readout({...})` 인자 dict 키를 뽑아
`DECISION_LEGS`/`WORLD_LEG`/`CROSS_LEGS`/`GREEN_STREAMS ∪ PLAIN_CONTROL_STREAMS`와
**집합 동일**을 요구(감사가 쓴 스니펫 재사용 가능). 함께: 어댑터가
`len(_sweeps(leg)) == raw["sweeps_per_green_leg"]`(green 3레그)를 검사하고 불일치 시
fail-closed. (스윕 수가 1로 떨어지면 `escape_replicated`가 조용히 `False`가 되고
`Δ_split`이 `None`이 된다 — 지금은 아무도 안 본다.)

---

## **G6 — ★sbatch가 stage 3의 rc를 검사하지 않는다: 채점 실패가 exit 0으로 끝난다 (정본이 R0에서 이미 등재하고 미수정으로 남긴 결함의 복제)**

**위치**: `p0a_graph_sm_confinement.sbatch:102-103`
```bash
python3 "$OUT/p0a_graph_sm_confinement.py" --analyze "$OUT/p0a_raw_${JOBTAG}.json" \
        --tag "$JOBTAG" --outdir "$OUT" | tee "$OUT/p0a_verdict_${JOBTAG}.txt"
```
rc 검사 없음, `set -e` 없음(`:55`는 `set -uo pipefail`), 이후는 `echo`뿐(`:105-113`).
⇒ `analyze`가 예외를 던지면 **job은 exit 0**, `.txt`에는 traceback, **`p0a_verdict_<jobid>.json`은
존재하지 않는다.** stage 1(`:78-84`, exit 20)·stage 2(`:93-99`, exit 21)는 제대로 검사한다.

**정본 대조**: `PROJECT_STATUS.md:4375` R0 결과 감사 caveat **(c)** — *"`smid_l0_run.sbatch:86`이
stage 4 rc 미검사(스코어링 실패가 exit 0으로 끝날 수 있음) — engine-porter 이관 항목으로
등재(**미수정**)"*. ⇒ **알려진 미수정 결함이 새 스크립트로 상속됐다**(교훈 #71 형태).

**닫는 법**: `RC_AN=${PIPESTATUS[0]}` + `[ "$RC_AN" -ne 0 ] && { echo "SCORING FAILED"; exit 22; }`
+ `[ -s "$OUT/p0a_verdict_${JOBTAG}.json" ] || exit 22`. exit 22를 §5.3/§10-14 표에 등재.

---

## **G7 — ★§10 완료 상태표의 검증 방법이 재현되지 않는다 (`grep -in physical` = 0 이라 적었으나 4 hits)**

**위치**: 사전등록 `PREREG_P0A_2026-08-22.md:575` (§10 표 1행) — 검증란 *"`grep -in physical` = **0**"*.
실제:
```
p0a_graph_sm_confinement.py:17   ... established to be a physical SM index, and this file never calls it one.
p0a_graph_sm_confinement.py:629  "label, not an established physical SM index, ..."
p0a_graph_sm_confinement.sbatch:31  ... that it is a physical SM index. ...
p0a_graph_sm_confinement.sbatch:32  ... Nothing here may be written as "physical SM".
```
**내용은 옳다**(전부 금지 명제의 **부정**이다 — §6 라벨 상한을 지킨다). 문제는 **검증 방법이
거짓**이라는 것이고, 그것은 rev5가 **E6**으로 지적해 *"이력표에 적는 항목마다 검증 방법을
병기한다"*(`:83`)고 약속한 바로 그 규율의 위반이다(교훈 #67). 검증 방법이 재현되지 않으면
다음 세션은 표를 신뢰할 수 없다.
★같은 표의 다른 두 검증은 **재현된다**: `grep -n "TOL"` = **0** ✅ · §10-17 docstring 최상단
사전등록 정본 선언 ✅(`:4-6`). §12의 *"stage 1 ≈3분"* 도 **실측 193 s로 재현** ✅.

**닫는 법**: 1행의 검증란을 실제로 통과하는 술어로 교체(예: *"`physical`의 모든 출현이
부정문 안에 있다"* 를 검사하는 코드 검사, 또는 `grep -in "physical SM" | grep -v -e "not " -e "never"` = 0).

---

## **G8 — ★§8 "자유 모수 **전수** 목록"이 전수가 아니다 (그중 하나는 결정 게이트에 비대칭으로 작용한다)**

**미등재 목록**
1. ★**평범 레그의 census 횟수 = 1 sweep**. §8-22(`:541`)는 **green 레그만** 2회로 동결한다.
   `run()`은 `eager_plain`·`graph_plain`을 **1회**(`:381`, `:389-390`), green 3레그를
   **2회**(`:408`, `:410-411`, `:413-414`) 돌린다. ★이것이 무해하지 않은 이유:
   **`P4c`는 `S(eg) ∪ S(egp)`(각 2 sweep = 50 launch)와 `D = S(eager_plain)`(1 sweep =
   25 launch)의 *집합 동일*을 요구**하고, `Δ_null`은 두 1-sweep 평범 레그의 *집합 동일*을
   요구한다. 커버리지 비대칭이 곧바로 `NOPAIR`/`NULLCH`(= 예산 전액 소실)로 간다.
2. side 스트림의 **개수와 수명**(`:167` — shim당 1개 = **sweep당 1개**, 총 8개).
3. launch마다 `torch.cuda.CUDAGraph()` **생성·`del`**(= graph private pool 생성/해제 125회,
   `:194`, `:212`). 캡처 단위는 §8-12에 있으나 **그래프 객체 수명**은 없다.
4. `plain_post` 스트림(`:396`) — §8-19(`:538`)는 *"green 두 스트림 + **평범 스트림**"*(단수)이라
   적는데 하네스는 평범 **2개**를 판독한다(음성 대조가 더 강해지는 방향이라 무해).
5. `--sample`(`:1279-1281`) — stage 1을 조용히 약화시킬 수 있는 CLI 노브.
   sbatch(`:77`)는 쓰지 않는다 ✅. §8에 *"제출 런에서는 금지"* 로 등재할 것.

★**동시에, 감사가 발견한 좋은 소식 — 근거는 있는데 사전등록에 없다**: R0(job 889631)의
사다리는 **첫 grid 점(108 blocks)에서 이미 포화**한다 —
`idx0_*: [108,108,108,108,108] min_hits 155` · `idx1_prefill: [74×5] min_hits 210` ·
`idx1_decode: [34×5] min_hits 473`. ⇒ P2/P3/P5/Δ_null/P4b/**P4c**의 *집합 동일* 게이트는
**충분히 도달 가능**하며, 위 (1)의 비대칭이 실제로 물릴 확률은 낮다. **그러나 이 타당성
논증이 사전등록에 한 줄도 없다** — 없으면 `NOPAIR`가 나왔을 때 "설계가 원래 그랬다"와
"기판이 달라졌다"를 사후에 구분할 수 없다.

**닫는 법(GPU 0)**: §8에 위 5행 추가(전부 "조정 금지"), `meta`에 `sweeps_per_plain_leg` 기록,
그리고 §4 또는 §12에 **R0 사다리 실측을 근거로 한 도달가능성 한 문단**을 등재.

---

## **G9 — ★§9 자기무력화 조건이 절반만 강제된다 (§9-3·§9-4는 기록만 되고 참조값이 어디에도 없다)**

| 조건 | 하네스에서 | 판정 |
|---|---|---|
| §9-5 granularity/기판 불일치 | `run():328-332` **SystemExit** + 어댑터 `:574-580` **fail-closed**, 변이 `granularity_unchecked`가 load-bearing 실증 | ✅ **강제됨** |
| §9-3 `nvidia-smi` compute mode | `_compute_mode():436-444` → `meta:366` → 판정서 `run_nvidia_smi_compute_mode`. ★**대조 없음, 참조값 없음** | ❌ 기록만 |
| §9-4 생산자 미수정(sha256 기록) | `meta:360-364`에 3파일 기록. ★**기대값 없음** — R0 원자료의 `sha256_unmanifested`는 `pdmux_context.py`·`sgl_kernel/spatial.py` **2개뿐이고 `smid_l0_census.py`는 없다** | ❌ 기록만 |
| §9-1·§9-2 | 사람 조건 | — |

**감사가 지금 확정해 주는 참조값**(이대로 §9에 박아 두면 조건이 강제 가능해진다):
* R0 compute mode = **`Default`** (`smid_l0_census`의 job 889631, gpu40,
  `smidl0_889631.out:7` — *"NVIDIA A100-SXM4-80GB, 81920 MiB, Default"*).
* 현재 생산자 `smid_l0_census.py` sha256 =
  **`1cee21918f34e7f1f207aaef0fc13d03717ea38500e9b937c7de3c49ae96a325`**
  (mtime 2026-08-21 18:27 — job 889631 실행(2026-08-22 01:45) **이전**이므로 R0가 쓴 그 파일이다).

**닫는 법**: 위 두 값을 §9에 등재하고, 불일치 시 `analyze()`가 판정서 최상단에
`substrate_mismatch: {...}` 배너를 찍게 한다(§9-3의 *"판정서 최상단에 병기"* 이행).

---

# 비차단 (등재만, 제출을 막지 않음)

| # | 내용 | 위치 |
|---|---|---|
| **N1** | `capture="partial"` 축은 규칙 세계 공간에 없지만 **규칙은 `!= "ok"`로 잡는다** — 감사가 실측: `capture="partial"`→`NOCAP`, `replay="partial"`→`NOREPLAY`, 둘 다 `World.__init__`이 `S_gg`를 비운다. ⇒ **차단 아님**(감사 지시 6번의 답). 다만 그 축은 `"fail"`을 통해서만 변이된다 | `p0a_rule_totality.py:98-99,159-162`; 하네스 `:224-232` |
| **N2** | AST 가드는 **속성 호출을 따라가지 않는다** — `CEN.*`/`RULE.*`로 세탁한 target-layer 읽기는 안 보인다. 감사 변이 4종 실측: 직접 읽기 ✅잡음 · **신규 모듈 헬퍼** 세탁 ✅잡음 · 금지 driver 키 ✅잡음 · `CEN.*` 세탁 ❌미탐지. R0에서 상속된 한계이고 하네스는 새 `CEN` 읽기를 추가하지 않는다 | `smid_l0_census.py:1577-1579` |
| **N3** | `CROSS_LEGS`(`:116`)가 **죽은 상수** — 교차 레그 이름은 `:421-423`에 리터럴로 다시 적혀 있다(G5와 같은 드리프트 형태, 서술 전용이라 무해) | `:116, :421-423` |
| **N4** | `_leg_set`은 `isinstance(u, list)`를 요구하고 `_leg_sweep_sets`는 요구하지 않는다 ⇒ 오형 sweep에서 `S(gg)`와 `escape_replicated`가 **다른 관점**으로 계산될 수 있다(둘 다 더 싼 라벨 쪽으로 실패하므로 안전) | `:463-470` vs `:473-475` |
| **N5** | `_leg_set`은 `bool`을 `int`로 받아들이지만 `_leg_min_hits`는 명시적으로 배제한다(`:486`). 생산자는 bool을 내지 않는다 | `:467-469` |
| **N6** | `_SWEEP_CACHE`(`:742`)가 **같은 dict 객체**를 여러 레그에 별칭으로 넣는다. 현재 어떤 검사도 sweep을 in-place 변경하지 않고 변이 러너는 JSON 왕복(`:1112`)하므로 안전 — 다음 픽스처 작성자에게는 지뢰 | `:742-766` |
| **N7** | §11이 나열한 긍정형 필드명(`escape_empty`·`capture_succeeded`·`positivity_ok`)은 하네스가 내는 이름(`exact_set_match`·`capture_status`·`min_hits_by_leg`)과 다르다. **극성 결함은 없다**(게이트 #56 준수 실측: `green_ctx_attached` 긍정형 파생 + 생산자 원값 `green_ctx_is_null` 보존, A3b/A3c) | 사전등록 `:646-651` vs `:287-289, :676-706` |
| **N8** | `descriptive_legs_incomplete`는 `_leg_graph`가 **예외를 던질 때만** 참이 된다. 교차 레그가 25쌍 중 0쌍만 캡처해도(예외 없음) `False`로 남는다(상태 자체는 `cross_legs`에 보고됨) | `:421-430, :699-705` |
| **N9** | 비용 표기 불일치(doc-steward): `PROJECT_STATUS.md:4389` = 예상 **0.17 GPU-hr** ↔ 사전등록 §12 = 실사용 ≈**0.1**, 등록 상한 **0.33** | — |
| **N10** | **§12 검증**: stage 1 전 세계 공간 `--selftest` 벽시계 **193 s** 실측(로그인 노드) — *"≈3분"* 재현 ✅. 20분 상한에 여유 충분 | — |
| **N11** | (규칙층, 범위 밖) `SUBSTANTIVE`에 `UNSEP`가 들어 있어 규칙 자신의 T2가 이것을 실질 라벨로 취급하는데, §5는 같은 것을 *"중립 어휘"* 라 부른다. 하네스 결함 아님 | `p0a_rule_totality.py:60` vs 사전등록 `:412` |
| **N12** | (규칙층, 범위 밖) `RULE.GRANULARITY`는 **하드코딩된 `(8,0)`** 에서 온다(`p0a_rule_totality.py:75`). cc `(8,6)` 노드는 granularity 2로 **양쪽 검사를 통과**한다. 파티션이 A100을 고정하므로 실질 위험 없음 | — |

---

# ★ 깨뜨리려 했으나 버틴 것 (14건)

> 이 절이 없으면 *"차단만 9건이니 하네스가 부실하다"* 로 오독된다. **아래는 전부 감사가
> 능동적으로 공격해 실패한 것들이며, 그중 다수는 이전 회차들이 지목한 결함의 정확한 수리다.**

* **B1 ★ `run()` → `assemble_raw()` → `world_from_raw()` 필드 단위 대조 (감사 지시 1번)**:
  `run()`의 AST에서 뽑은 leg 키·driver 스트림 키가 어댑터 상수와 **완전히 일치**.
  ⇒ ★**"표/어댑터가 읽는데 `run()`이 쓰지 않는 필드"는 0건** — **S-6 F2 형태의 결정론적
  예산 전액 소실 경로는 없다.** (없는 것은 그것을 지키는 *가드*뿐 = G5.)
* **B2 생산자 스키마 일치**: `_census_target`이 내는 키는 정확히 `{ladder, union, hits,
  min_hits}`(`smid_l0_census.py:644-646`)이고, A1b가 그것을 **생산자 AST에서 읽어** 픽스처와
  대조한다(항등식 아님). A1c는 **생산자 자신의 `_saturated`** 를 픽스처 사다리에 양방향으로
  돌린다. 픽스처가 생산자 언어를 쓰지 않는 경로를 찾지 못했다.
* **B3 교차 레그 동기화 (감사 지시 2번)**: `_census_once`(`smid_l0_census.py:614-625`)는
  census 스트림만 sync하지만, shim이 `_launch` 안에서 `self.replay_stream.synchronize()`
  (`:208`)를 **직접** 호출하고 그 뒤 `.tolist()`가 다시 sync한다. 결정 레그는 캡처=replay
  스트림이라 동일. **동기화 결손 없음.**
* **B4 ★ warmup 스크래치(§8-24)가 실제로 오염을 막는다 (감사 지시 3번)**: census 커널
  (`smid_l0_census.py:253-279`)은 네 텐서를 **저장만** 하고 읽지 않으므로 `torch.empty_like`의
  미초기화 내용이 결과에 들어갈 수 없다. 스크래치는 side 스트림에서만 쓰이고
  `self.side.synchronize()`(`:188`)가 `del scratch`(`:192`)보다 **앞**이라 캐싱 할당자의
  cross-stream free 위험도 닫힌다. 생산자가 census 텐서를 `full(-1)`로 만들고 음수를
  건너뛰므로 실패한 launch는 기여 0. ⇒ ★**§8-24가 막으려던 "탈출 제조" 경로는 실제로 닫혀 있다.**
* **B5 거짓 `PRESERVED` 경로 탐색 실패**: 캡처는 실행하지 않고, replay만이 green 레그
  텐서의 유일한 기록자이며, warmup은 스크래치만 건드린다. eager green launch가 graph 레그
  텐서를 채우는 경로를 찾지 못했다.
* **B6 좁아진 graph 레그가 채점되는 경로 없음**: 부분 캡처(0<cap<25) → `"partial"` → `NOCAP`,
  부분 replay → `NOREPLAY`, 두 경우 모두 `World.__init__`이 `S_gg`를 비운다. 실측 확인.
* **B7 sticky CUDA 오류 전파**: shim 밖(`_census_once`의 launch/sync)에서 나는 오류는
  `_census_target` → `run()`을 뚫고 나가 **exit 21**이 된다. shim 안에서 삼켜져도 다음
  `_census_once`가 터진다. ⇒ **잘린 집합이 아니라 exit 21**로 끝난다(fail-closed).
* **B8 ★ AST 가드가 이 파일에서 load-bearing이다 (감사 지시 "AST 가드")**: 변이 실측 —
  `world_from_raw`에 직접 `total_sm_reported` 읽기 → **잡힘** · `_diagnostics`가 부르는
  **신규 모듈 헬퍼**로 세탁 → **잡힘** · `_attached`에 금지 driver 키 `rc` → **잡힘**.
  도달 함수 15개에 `world_from_raw` 포함(A2c 비공허). A2d는 알려진 누출 픽스처로 발화 실증.
* **B9 규칙 재구현 없음**: `grep -n "TOL"` = 0, 결정 호출은 `RULE.score(w)` 한 줄(`:620`).
  rev6의 항등식형 *"두 구현 일치 검사"* 는 실제로 사라졌다.
* **B10 ★ A5는 항등식이 아니다 (감사 지시 4번)**: 픽스처↔어댑터가 공유하는 것은 **레그·스트림
  이름 축뿐**(= G5)이고, (i) sweep 스키마(A1b가 생산자 AST에서 읽음) (ii) 포화 술어
  (양쪽 다 생산자 `_saturated` 호출) (iii) 부착 술어(생산자 `_greenctx_attached`/
  `_greenctx_detached` 호출) (iv) 규칙 자신 — 넷은 **공유하지 않는다**. 게다가 변이 9종이
  같은 `raw` 픽스처 위에서 라벨을 **뒤집는다** — 항등식이면 불가능하다.
* **B11 결정 게이트의 도달가능성**: R0 실측 사다리가 첫 grid 점에서 포화(min_hits 155/210/473)
  ⇒ *집합 동일* 게이트들이 knife-edge가 아니다(단, 사전등록 미기재 = G8).
* **B12 sbatch 프롤로그**: `set -uo pipefail` + `module load` 조합이 `smid_l0_run.sbatch:36-40`
  과 **바이트 동형**이고 그 스크립트는 job 889631로 완주했다. ⇒ **S-6 E4(`set -u` 즉사)
  형태는 해당 없음.** stage 1/2의 rc 검사도 정상(exit 20/21).
* **B13 §9-5 이중 강제**: `run()`과 어댑터 양쪽에 있고, 변이 `granularity_unchecked`가
  어댑터 쪽이 load-bearing임을 실증(intact `ABSENT` ↔ mutant `PRESERVED`).
* **B14 §5.1 필수 병기 3건이 판정서에 전부 있다**: `min_hits_by_leg` ·
  `delta_label_ranks`(Δ 각 라벨의 레그 내 히트 분포 순위) · `delta_size_vs_split_size`
  (|Δ| vs |Δ_split|). §2.3의 서술 목록(`graph_green_covers_D` · `escape_attributed` ·
  `D_size` + `run_total_sm_reported` · `green_pair_disjoint` · `exact_set_match` ·
  `null_channel` · `cross_legs`)도 전부 있다. **항등식 진단으로의 퇴행 없음.**

---

## 감사 지시 7항목에 대한 직접 답

| # | 질문 | 답 |
|---|---|---|
| 1 | §10.2 계약표가 전수인가 | **어댑터가 읽는데 `run()`이 안 쓰는 필드 = 0건**(B1, 위험한 방향은 깨끗). 반대로 **`run()`이 쓰는데 표가 안 읽는 필드**가 있다 → **G2**(`graph_plain`의 캡처/replay 상태) · **G3**(`capture_exc`/`capture_events`). 표를 `run()`에 묶는 기계 없음 → **G5** |
| 2 | launch-shim이 같은 것을 재는가 | **교차 레그 동기화 충분**(shim이 직접 replay 스트림 sync, B3) · **결정 레그는 동일 스트림** · **캡처 실패의 후속 오염은 exit 21로 끝난다**(B7). ✅ |
| 3 | warmup 스크래치가 오염을 막는가 | **막는다.** 커널이 입력을 읽지 않고, side sync가 free보다 앞이며, 미기록 칸은 `-1`로 건너뛴다(B4). `torch.empty_like`가 캡처 풀과 상호작용하는 경로 없음(할당은 캡처 **밖**, `__enter__`의 `empty_cache`는 살아 있는 텐서를 못 건드림) ✅ |
| 4 | A5가 항등식인가 | **아니다**(B10). 단 **레그·스트림 이름 축에서만** 픽스처와 어댑터가 상수를 공유하므로 그 축의 드리프트는 못 잡는다 → **G5** |
| 5 | 변이 9종이 계약 각 줄을 고정하는가 | **9행 전부 오늘은 판별한다**(실측 라벨 확인). 그러나 **판별을 강제하는 assert가 없다** → **G4**(퇴화 witness로 `ALL PASS` 실측) |
| 6 | `capture="partial"` 축 | **비차단**. 규칙이 `!= "ok"`로 잡아 `NOCAP`을 내고 `S_gg`를 비운다(실측, N1) |
| 7 | exit 코드 계약 | 결정 레그 실패 → **exit 21 ✅** · 서술 레그 실패 → **채점 허용 ✅** · **stage 3 실패 → exit 0 ❌** → **G6** |

---

## 이 감사가 **하지 않은** 것 / 이 판정서가 **licence하지 않는** 문장

* GPU에서 한 줄도 돌리지 않았다. **green 스트림 위 그래프 캡처가 가능한지는 여전히 미지수**다
  (rev7 헤더의 자백 유효). 그것이 안 되면 `NOCAP`이며 **도구 가용성 사실이지 한정 상실이 아니다**.
* 규칙층은 재감사하지 않았다(§2·§4·§5 불변). N11·N12는 분리 등재.
* ★**금지 문장 승계**: *"하네스가 감사를 통과했으므로 GO다"* · *"어댑터가 검증됐다"* ·
  *"9/9 변이 통과이므로 계약이 옳다"* · *"규칙층이 닫혔다"* · *"구멍 C가 닫혔다"* ·
  *"P0-A가 R4에 답했다"*. ★신설: *"하네스층 감사가 하네스를 옳다고 판정했다"* — 이 판정서는
  **"판정을 만들 수 있는 경로 9개를 찾았고 전부 GPU 0으로 닫힌다"** 고만 말한다.
* 성능 문장 **0건**. 정책 순위 **0건**. **HE0 불변.** gate #13/#16 · switch-cost
  *"닫았다"* 금지 전부 유지.

## 재현 방법 (전부 프로젝트 venv, GPU 불요)

```bash
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe
python p0a_graph_sm_confinement.py --selftest        # ALL PASS, 193 s
grep -in physical p0a_graph_sm_confinement.py p0a_graph_sm_confinement.sbatch   # G7: 4 hits
grep -n "capture_exc\|capture_events" p0a_graph_sm_confinement.py               # G3: 기록 4, 소비 0
sed -n '100,105p' p0a_graph_sm_confinement.sbatch                               # G6: rc 미검사
```
G1·G2·G3의 라벨 실측, G4의 퇴화 witness, G5의 AST 대조, B8의 가드 변이 4종은
본문에 코드 스니펫으로 그대로 실려 있다(전부 `import p0a_graph_sm_confinement as H` +
`import p0a_rule_totality as RULE` 후 수 줄).

---

**판정: `GO-with-conditions` — G1–G9를 닫고 `--selftest`가 `ALL PASS`면 제출.**
차단 9건 합계 예상: **GPU 0 · 코드 ~60줄 · 사전등록 ~10곳.**

---
---

# 후속 검증 (2026-08-23, claims-auditor, 같은 세션 2라운드)

> **대상**: 커밋 `0ba9394` *"fix(bcg_probe): close harness-layer audit blockers G1-G9 (prereg rev8)"*
> (`p0a_graph_sm_confinement.py` +410 / `.sbatch` +17 / `PREREG_P0A_2026-08-22.md` +91,
> ★`p0a_rule_totality.py` **0줄 변경** — 규칙층 불변 확인).
> **단일 판정 질문**: *"G1–G9 수리가 (i) 실제로 그 결함을 닫았고 (ii) **새 결함을 만들지
> 않았는가**? 지금 상태로 제출해도 되는가?"*
> **읽기 전용** — 대상 파일 수정 0건, 이 절만 추가.

## ★ 후속 판정: **`SUBMIT-OK-with-notes`**

**G1–G9 아홉 건 전부 닫혔고, 아홉 건 모두 감사가 독립적으로 재현했다.**
**새로 발견된 차단은 0건이다** — *"판정이 (a) 배선 (b) 측정실패 라벨링 (c) 항등식
(d) 사후 재해석으로 만들어지는 경로"* 를 다시 7갈래로 공격했고 **전부 실패했다**.
신규 지적 **7건(H1–H7)은 전부 비차단**이며, 그중 **H1만 제출 전에 닫기를 권한다**
(**문서 3줄 · 코드 변경 0 · GPU 0**).

★**이 트랙의 5회 연속 패턴("수리가 다음 최고 차단을 만든다")은 이번에 재발하지 않았다.**
수리가 만든 신규 표면은 있으나(H2·H3·H5) 전부 **판정을 만들 수 없는 방향**
(실질 라벨 → `ABSENT`, 또는 서술 레그 한정)이고, 그중 둘은 수리 문서가 스스로 예고했다.

**자기검사 실측**: `--selftest` → **`ALL PASS`, rc=0, 171 s**(§12 *"≈3분"* 재현),
A5 **516,096 세계**, 변이 **13행 × 3검사**, A7a–d, A8a–b, R1–R3 + 판독 변이 2.
`bash -n p0a_graph_sm_confinement.sbatch` → OK.

---

## 1. 차단 9건 — 닫혔는가 (전부 감사가 독립 재현)

| # | 닫혔나 | 감사의 독립 재현 |
|---|---|---|
| **G1** | ✅ | 잡음의 가장 싼 형태(per=[{40},{41}])가 이제 **`LOST (UNREPLICATED)`**. 변이 `replication_any_instead_of_same`(intact `LOST_UNREP` / mutant `LOST_PART`)이 교집합 술어가 load-bearing임을 실증. §5 표 3행과 §8-25가 **같은 술어**로 통일된 것을 diff로 확인 |
| **G2** | ✅ | `graph_plain` partial → **`ABSENT`**(전에는 `NULLCH`). 변이 `plain_graph_gate_removed`(intact `ABSENT` / mutant `NULLCH`) |
| **G3** | ✅ | (a) 레그 부재 → `ABSENT`(변이 `absent_leg_treated_as_fail`, mutant가 `NOCAP`) · (b) warmup 실패가 `warmup_ok`/`warmup_exc`로 분리되어 **`capture_exc`를 오염시키지 않는다** · (c) warmup 25/25 실패 픽스처를 채점 → 판정서에 `graph_leg_failures` **존재**(`warmup_failed`·`first_warmup_exc` 포함) 실측 |
| **G4** | ✅ | ★1라운드에서 `MUTANTS ALL PASS`를 냈던 **바로 그 퇴화 witness를 다시 주입** → 이번엔 **rc=1, `[FAIL] … the two declared labels differ … both UNDETERMINED (MEASUREMENT ABSENT)`**. 13행 전부 3번째 검사 통과 |
| **G5** | ✅ | ★**변이 3종 실측**: `run()`이 leg 키를 `graph_plain`→`graph_plane`으로 흘리면 **A7a FAIL**, driver 스트림 키를 흘리면 **A7b FAIL**(intact은 4/4 PASS). ⇒ A7은 항등식이 아니고 실제로 물린다 |
| **G6** | ✅ | `RC_AN=${PIPESTATUS[0]}` + `.json` 비어있음 검사 → **exit 22**. `bash -n` 통과. §5.3 표에 행 신설(`:499`) 확인. §10-14 행(`:638`)은 서술 레그 규칙이라 **여전히 정확** ⇒ **표 간 모순 없음**(감사 지시 6번의 답) |
| **G7** | ✅ | A8a/A8b PASS. `_banned_adjective_mentions()`가 4건을 찾고 **4건 모두 부정문**임을 요구 — *"0건이어야 한다"* 는 거짓 술어가 **통과하는 술어**로 교체됐다 |
| **G8** | ✅ | §8-28–32 신설(평범 sweep 수 · side 스트림 수명 · 그래프 객체 수명 · `plain_post` · `--sample` 금지) · `meta["sweeps_per_plain_leg"]` 기록 · §12에 **R0 사다리 근거 도달가능성 문단**(`idx0_* [108×5] min_hits 155` 등) 신설 |
| **G9** | ✅ | ★**참조값을 감사가 독립 재계산**: `R0_CENSUS_SHA256` 길이 **64** ∧ 디스크의 `smid_l0_census.py` sha256과 **일치**, `R0_COMPUTE_MODE == "Default"`. 배너 동작 실측 — 정상 런은 배너 **없음**, 드리프트 런은 **최상단 첫 키**가 `substrate_mismatch`이고 **`verdict`는 보존**(게이트가 아니라 배너, §9-3 요구 그대로) |

★**비차단 동반 수리 5건도 확인**: **N3**(`run()`이 `zip(CROSS_LEGS, …)`) · **N4**(`_leg_sweep_sets`가
`isinstance(list)` 요구) · **N7**(§11 이름 목록 정정) · **N8**(교차 레그 status가 `ok`가 아니면
`descriptive_legs_incomplete`) · **N12**(`run()`이 cc `(8,0)` 명시 요구).

---

## 2. ★ 감사가 지정받은 7개 공격 지점 — 결과

### 1. 새 pre-World 게이트 3종이 원래 라벨을 가리는가 → **실질 라벨을 만들지는 못한다. 단 H1·H2·H3.**

★**구조적 보증을 먼저 확정한다**: 하네스가 낼 수 있는 판정 문자열은
`p0a_graph_sm_confinement.py:774`(`RULE.ABSENT`)와 `:778`(`RULE.score(w)`) **두 곳뿐**이다
(grep 전수). ⇒ **하네스는 규칙이 소유하지 않은 라벨을 발명할 수 없다.**
그리고 세 게이트는 전부 `return None, rec` → `RULE.ABSENT`이므로 **단조 보수적**이다:
라벨을 **실질 → `ABSENT` 방향으로만** 움직일 수 있고 그 반대는 불가능하다.
⇒ 합격 기준 (a)(b)(c)(d) 어느 것도 열리지 않는다.

**그러나 "실질 라벨을 가로채는 세계"는 실재한다**(감사 실측, → **H3**):
```
graph_plain capture="partial" 이면서 S(gp)==S(ep)==108 (null channel 완벽히 깨끗)
  규칙이 냈을 라벨 : CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY
  현재 판정        : UNDETERMINED (MEASUREMENT ABSENT)
```
R0 실측이 **첫 grid 점에서 이미 108 포화**임을 감안하면 25쌍 중 5쌍만 캡처돼도 `S(gp)`는
108이다 — 즉 *"부분 캡처인데 대조는 완벽"* 은 실재 가능한 세계이고, 이제 런 전체가 버려진다.
**방향은 안전**(회수 불가한 손실이지 거짓 판정이 아님)하고 §5의 green 레그 규칙
(*"25쌍 전부 성공"*)과 **정합**하지만, 이것은 **§4에 없는 새 선행조건**이다.

**덜 정확한 `why`를 주는가** → 준다(→ **H2**). 계측 사망·미부착이 동시에 있어도 `why`는
평범 레그만 지목하고, `instrument_alive`·`attached`·`set_sizes`가 기록에서 **사라진다**.

### 2. A5가 새 게이트 때문에 약해졌는가 → **아니다. 교훈 #70에 해당하지 않는다.**

세 게이트는 A5의 516,096 세계에서 **한 번도 발화하지 않는다**(픽스처가 항상 등록값을 쓴다).
그러나 그것이 *"검사가 발화 불가"* 는 **아니다**:
* **발화 분기는 A5 밖에서 전수 검증된다** — 게이트마다 **전용 witness + 변이**가 있고
  (`_w_plain_graph_partial`·`_w_graph_green_absent`·`_w_wrong_sweep_count`),
  `intact=` 검사가 *"게이트가 실제로 발화한다"* 는 양성 대조이며 `mutant=` 검사가
  *"게이트를 지우면 라벨이 바뀐다"* 는 load-bearing 실증이다(3행 전부 PASS 실측).
* **게이트의 상수는 A5가 묶어 준다**(감사 실측): `GREEN_LEGS`에서 `eager_green_prefill`을
  빼는 변이본을 만들자 **A3a가 즉시 FAIL**(`clean run` → `ABSENT`)했다 — 픽스처는 sweep 수를
  리스트 길이로 직접 만들고 게이트는 `GREEN_LEGS`/`SWEEPS_*`에서 유도하므로, 두 경로가
  갈리면 A5/A3가 소리를 낸다.
⇒ **정당한 설계다.** (남는 것은 "그 게이트들이 규칙 정본 밖에 있다"는 별개 문제 = **H1**.)

### 3. G1 교집합 술어가 반대 방향 결함을 만드는가 → **만든다. 단 유계이고 가시적이다(H7).**

감사 구성 세계: 42 라벨 탈출(sweep A = {40..60}, sweep B = {61..81}, **완전 서로소**)
```
  판정: CONFINEMENT_LOST_THROUGH_GRAPH_REPLAY (UNREPLICATED)   ← 보고, 회부 없음
  adapter.escape_replication.per_sweep = [[40..60], [61..81]]  ← 판정서에 전부 실림
```
즉 **디바이스 전체 규모의 실 탈출도, 두 census가 라벨을 하나도 공유하지 않으면 회부되지
않는다**. 다만 (i) **절대로 `PRESERVED`가 되지 않는다**(양쪽 다 `LOST (*)`이며 정지·보고),
(ii) `per_sweep`·`replicated_any`·`Δ_split`이 판정서 `.json`에 **전부** 실려 사람이 회부를
판단할 수 있다, (iii) R0가 **첫 grid 점에서 포화**하므로 두 census가 큰 탈출을 라벨 하나도
공유하지 않을 물리적 개연성은 낮다. ⇒ **교집합 방향 자체는 옳다**(잡음의 가장 싼 형태가
최고가 판정을 사던 채널을 닫는 것이 더 값지다). **H7**은 문구 1줄 권고.

### 4. `absent` 4번째 값이 규칙과 충돌하는가 → **충돌하지 않는다.**

`RULE.score`/`World.__init__`이 `!= "ok"`만 보므로 `absent`도 `NOCAP`으로 갈 **수 있지만**,
어댑터의 두 게이트가 그 전에 전부 가로챈다: `graph_plain`은 `gp_cap != "ok"`(=`absent` 포함),
`graph_green`은 `"absent" in (cap, rep)`. **status를 읽는 레그가 그 둘뿐**이므로
`absent`가 `World`에 도달하는 경로는 없다. 이중 경로는 변이 `absent_leg_treated_as_fail`
(intact `ABSENT` / mutant **`NOCAP`**)이 지킨다 — 즉 *"가로채기를 지우면 정확히 그 사고가
난다"* 가 스위트 안에서 실증돼 있다. **안전.**

### 5. A7이 항등식인가 → **아니다(A7a·A7b는 강하다). 단 A7c·A7d는 약하다(H4).**

* **A7a·A7b**: 감사가 `run()`을 흘려 **각각 FAIL시켰다**(위 표 G5). 항등식이 아니다.
* **A7c·A7d**: *"`run()`이 상수를 이름으로 참조하는가"* 다. 감사 실측 — `run()`에
  `_ = CROSS_LEGS` 한 줄을 두고 교차 레그 이름을 **다시 리터럴로 적은** 변이본이
  **A7c를 통과**했다. **항등식은 아니지만**(참조를 지우면 FAIL한다) *"하려는 일"* 을
  하지 않고도 만족된다. 같은 이유로 `run()`이 `SWEEPS_GREEN` 대신 `2`를 박아도 A7d는
  통과한다(다른 호출부에 이름이 남아 있어서). ⇒ **H4**.
  ★다만 후자는 어댑터의 sweep 수 게이트가 하류에서 fail-closed로 잡는다.

### 6. exit 22가 §5.3/§10-14와 정합하는가 → **정합한다.**

§5.3에 행 신설(`:499`, *"채점(stage 3) 실패 → exit 22 + `.json` 부재"*). §10-14(`:638`)는
**서술 레그 실패가 채점을 막지 않는다**는 별개 명제라 수정 불요이고 모순도 없다.
`bash -n` 통과. `set -uo pipefail` 하에서 `${PIPESTATUS[0]}`은 파이프라인 직후에 읽히므로
안전하고, `tee`가 성공해도 `python3`의 rc가 잡힌다(주석이 그 이유를 정확히 적고 있다).

### 7. 새 필드가 AST 가드를 통과하는가 → **통과하고, 가드는 여전히 물린다.**

A2a/A2b PASS. 감사가 **신규 코드 3곳에 누출을 심어** 전부 잡히는 것을 확인했다:
새 `graph_plain` 게이트에 `total_sm_reported` → 잡힘 · `_escape_replicated`에 driver `rc`
→ 잡힘 · `_diagnostics`에 `nsmid_observed` → 잡힘. score 도달 함수 15개에 어댑터 관통 유지.
`graph_leg_failures`·`tool_status_by_leg`·`capture_summary`·`warmup_ok`는 금지 토큰도
드라이버 키도 들이지 않는다.

---

## 3. 신규 지적 — **전부 비차단** (H1만 제출 전 권고)

### **H1 — ★제출 전 권고(문서 3줄, 코드 0). §4가 이제 stale하고, 그 자기 선언이 거짓이 됐다.**

새 pre-World 게이트 3종(sweep 수 · `graph_plain` tool status · 레그 부재)은
**판정을 가르는 선행조건**이다(H3가 실측: would-be `PRESERVED` → `ABSENT`). 그런데
* **§4 표에 없다**(감사 grep: §4는 P0/P0g/P0p/P1/P2/P3/Δ_null/P4a/P4b/P4c/P5_* 만 적는다),
* **§4의 평가 순서 블록에 없다**(`P0(ep,gp,eg) → P1 → P2 → P3 → Δ_null → P4a → P4b_lo/hi → …`),
* **`p0a_rule_totality.py`에 없다** ⇒ **T1(전체성)·T2(비공허)·T3(도달가능)·T4/T4′(변이)·
  T5′(판별)의 밖**에 있다. 이 트랙이 5회에 걸쳐 만든 기계가 **결정 절차의 일부를 보지 못한다**.

그리고 §4는 스스로 *"★평가 순서는 §14의 코드가 정본이다"* 라고 적어 두었다 — **그 문장이
이제 거짓이다.** 이것은 5회차 **F2**(*"본문이 아직 교체 전 규칙을 지시한다 — 하네스
구현자가 본문을 읽으면 다른 규칙을 구현한다"*)의 **거울상**이다: 다음 세션이 §14 코드를
정본이라 믿으면 실제로 도는 게이트 3개를 못 본다.

**왜 그래도 차단이 아닌가**: (i) 하네스는 `RULE`이 소유하지 않은 라벨을 낼 수 없다
(`:774`·`:778` 전수) · (ii) 세 게이트는 **단조 보수적**(실질 → `ABSENT` 한 방향) ·
(iii) §5.3·§10.2에 **등재돼 있고** 각각 witness+변이가 있다 · (iv) 게이트를 규칙 파일로
옮기는 것은 **규칙층 개정**이고, 5회 감사가 수렴 선언한 층을 다시 여는 값이 이득보다 크다.

**권고(제출 전)**: §4에 문단 하나 — *"§4의 순서 앞에 하네스층 pre-World 게이트 3종이 있다.
이들은 `p0a_rule_totality.py` 밖에 있으므로 §14의 T1–T5′가 검증하지 않으며, 전부
`UNDETERMINED (MEASUREMENT ABSENT)`로만 떨어진다(실질 라벨을 만들 수 없다). 목록·라벨·
변이는 §5.3과 §10.2에 있다."* + *"평가 순서는 §14 코드가 정본"* 문장을 **"§4 이후의 평가
순서는"** 으로 한정. **GPU 0 · 코드 0.**

### **H2 — 새 게이트가 발화할 때 G3(c)의 진단이 사라진다 (G3의 축소 재발, 실측)**

`score():771-777` — `w is None`이면 `_diagnostics`를 **부르지 않는다**. 감사 실측:

| 사건 | 판정 | `graph_leg_failures` | `adapter.instrument_alive` | `adapter.tool_status_by_leg` |
|---|---|---|---|---|
| warmup 25/25 실패(구 게이트, `capture="fail"`) | `NOCAP` | ✅ | ✅ | ✅ |
| `graph_plain` partial(**신규 게이트**) | `ABSENT` | ❌ | ❌ | ✅ |
| `graph_green` 부재(**신규 게이트**) | `ABSENT` | ❌ | ❌ | ✅ |
| sweep 수 불일치(**신규 게이트**) | `ABSENT` | ❌ | ❌ | ❌ |

즉 **도구 실패가 원인일 때 도구 실패 계정이 판정서에서 빠진다** — G3(c)가 지적한 그 형태가,
더 작게, 새 경로로 돌아왔다. 원자료에는 전부 있고 `why`가 레그를 이름으로 지목하므로
**차단은 아니다**. **수리 3줄**: 이른 반환 전에
`rec["graph_leg_failures"] = {n: (legs.get(n) or {}).get("capture_summary") …}` 를 채운다.

### **H3 — G2 게이트가 감사 권고보다 엄격해져, 대조가 깨끗한데도 런을 버린다 (실측, §12 미기재)**

위 §2-1의 실측. 방향은 안전하지만 결과적으로 이 런은 **75회 캡처(평범 25 + green 50)가
전부 성공해야** 실질 판정에 도달한다. **§12의 비용 문단에 그 실패 프로파일이 없다**
(G8이 신설한 도달가능성 문단은 *커버리지*만 논증하고 *캡처 성공률*은 논증하지 않는다).
권고: §12에 한 줄 — *"실질 판정은 75회 캡처 전부 성공을 요구한다. 부분 실패는
`ABSENT`/`NOCAP`이며 재제출 사유다."*

### **H4 — A7c·A7d는 "이름을 언급했는가" 검사다 (항등식은 아니나 약하다, 실측)**

`_run_wiring():1070-1073`. 실측: `run()`에 `_ = CROSS_LEGS`만 남기고 교차 레그 이름을
리터럴로 되돌린 변이본이 **A7c 통과**. 마찬가지로 `SWEEPS_GREEN` 자리에 `2`를 박아도
A7d 통과(다른 호출부에 이름이 남아서).
★**반증 가능한 강한 형태**(항등식이 아니다 — 리터럴을 되돌리면 반드시 FAIL한다):
*"`run()`의 AST에 `CROSS_LEGS`의 어떤 원소와도 같은 문자열 상수가 존재하지 않는다"*.
이 세션이 A7c 초판을 항등식이라고 스스로 폐기한 판단은 옳았고, 그 **음의 형태**가 정확한 답이다.

### **H5 — N3 수리가 교차 레그의 이름과 스트림 쌍을 분리했다 (서술 한정, 무검사)**

`run():477-478` — `zip(CROSS_LEGS, ((g_decode, plain), (plain, g_decode)))`.
이전에는 이름과 스트림 쌍이 **같은 리터럴 튜플 안**에 붙어 있었다. 이제 `CROSS_LEGS`의
순서를 바꾸면 **두 서술 레그의 라벨이 조용히 뒤바뀐다** — 그리고 이 프로브에서 가장
흥미로운 서술 산출물(*"한정이 캡처 시점에 박히는가 replay 스트림에서 오는가"*)이 정확히
그 두 레그다. 결정량이 아니므로(§3-C) **판정을 못 바꾼다**. 수리: `zip` 대신 dict, 또는
이름↔스트림 대응을 A7에서 assert.

### **H6 — 소소한 것 3건**
* `analyze()`의 복사 목록(`:901-907`)에 **`sweeps_per_plain_leg`가 없다** — §11의
  *"판정서 전 필드가 `.json`에"* 에 대한 작은 결손(`sweeps_per_green_leg`만 실린다).
* `graph_plain` 게이트의 `why`가 *"did not complete"* 라고 적는데, 레그가 **아예 없을 때**
  (`absent`)도 같은 문장이 나온다 — 시작조차 안 한 레그에 대해 *"완료하지 못했다"* 는 부정확.
  (status 값 자체가 `absent`로 병기되므로 오독 위험은 작다.)
* `adapter` 기록에 `escape_replicated`(**결정**)와 `escape_replication.replicated_any`
  (**서술**)가 나란히 있다. §5·§8-25가 어느 쪽이 지배하는지 못박아 두었으므로 안전하나,
  인용 시 혼동 가능 — `replicated_any`에 `"(DESCRIPTIVE, NOT THE DECISION)"` 접미 권고.

### **H7 — G1의 잔여 거짓음성에 보고 항목 한 줄 (위 §2-3)**
§5의 `LOST (UNREPLICATED)` 행 필수 병기에 **`|per[0] ∪ per[1]|`** 을 명시할 것(문턱 아님).
현재도 `per_sweep`이 실리므로 값은 계산 가능하지만, *"작은 잡음"* 과 *"큰 서로소 탈출"* 을
판정서 한 줄에서 구분할 수 있게 하는 것이 §5.1 정신이다.

---

## 4. 다시 깨뜨리려 했으나 버틴 것 (수리 이후, 9건)

* **C1 라벨 발명 불가**: 하네스가 verdict를 쓰는 곳은 `:774`(`RULE.ABSENT`)·`:778`
  (`RULE.score(w)`) **두 곳뿐**(grep 전수). 규칙이 소유하지 않은 문자열은 나올 수 없다.
* **C2 신규 게이트의 단조성**: 세 게이트 전부 `return None, rec` → `ABSENT`. **실질 라벨을
  만드는 방향으로는 작동할 수 없다.**
* **C3 `absent`가 `NOCAP`에 도달하는 경로 없음**: status를 읽는 레그 2개 모두 앞단 게이트가
  덮는다. 변이가 그 가로채기를 지키고 있다.
* **C4 `set.intersection(*per)`의 arity 사고 없음**: sweep 수 게이트가 `len(per)==2`를
  강제한다. 실측 — graph_green 0 sweep → `ABSENT`, 6 sweep → `ABSENT`(예외 아님).
* **C5 AST 가드 여전히 load-bearing**: 신규 코드 3곳 누출 심기 전부 탐지.
* **C6 G9 참조값이 실제로 맞다**: sha256 64자 ∧ 디스크 파일과 일치(감사 재계산),
  compute mode `Default`. **오타 하나가 매 런에 거짓 배너를 찍었을 자리**인데 정확하다.
* **C7 배너가 판정을 오염시키지 않는다**: 드리프트 런에서 `verdict`가 **보존**되고 배너만
  최상단에 붙는다(§9-3이 요구한 *"병기"*, 게이트 아님).
* **C8 `zip` 페어링이 오늘 옳다**: `capture_green_replay_plain ↔ (g_decode, plain)`,
  `capture_plain_replay_green ↔ (plain, g_decode)` — 이름과 방향이 일치(H5는 *지키는 것이
  없다*는 지적이지 지금 틀렸다는 지적이 아니다).
* **C9 `_leg_sweep_sets`의 루프 변수 제약이 fail-closed**: 주석대로 `one`을 `sweep`으로
  "정리"하면 `sweep_A_only` 앵커가 2회 출현이 되어 **변이 스위트가 즉시 FAIL**(exit 20)한다.
  조용히 깨지지 않는다.

---

## 5. ★ G2 이탈에 대한 명시적 답 (수리 세션이 반증을 요청한 지점)

> *"`NOCAP`을 쓰지 않았다 — 그 라벨의 등록 문구가 `ON GREEN STREAM`이라 평범 레그에 쓰면
> 판정서가 거짓 문장을 담는다. 이 판단이 옳은지 반증해 달라."*

**반증 실패 — 그 판단이 감사의 원 권고보다 낫다.** 근거:
* `NOCAP`의 등록 문자열은 `p0a_rule_totality.py:58`
  `"UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM)"` 이다. 평범 레그 실패에 그것을
  쓰면 **인용 대상 아티팩트에 거짓 문장이 들어간다** — 이 프로젝트에서 그것은
  *"덜 구체적인 참"* 보다 훨씬 비싼 실패다(게이트 #21·#61의 정신).
* 감사의 원 권고(*"`NOCAP`으로 라우팅"*)를 그대로 따르려면 **새 라벨을 신설**해야 하고,
  그것은 `LABELS` 집합·T3 도달가능성·T4 변이를 건드리는 **규칙층 개정**이다 — 5회 감사가
  수렴 선언한 층을 여는 값이 이득보다 크다.
* 채택된 형태(`ABSENT` + 레그를 지목하는 `why` + `tool_status_by_leg`)는 **참이고**,
  구분에 필요한 정보를 **판정서 안에** 남긴다. `MEASUREMENT ABSENT`는 *"대조 측정이
  없다"* 는 뜻이므로 이 사건에 대해 정확하다.
⇒ ★**이 이탈은 등재해 둘 가치가 있다** — *"감사 권고가 라벨의 등록 문구와 충돌하면
권고가 아니라 문구가 이긴다"*.

---

## 6. 제출 권고

**`SUBMIT-OK-with-notes` — H1(문서 3줄)만 닫고 `sbatch`.**
H2·H3·H6은 같은 커밋에 태워도 좋고(합계 ~8줄), H4·H5·H7은 다음 개정으로 미뤄도 된다.
**신규 차단 0건 · GPU 지출 0 · 새 성능 판정 0건 · 정책 순위 변경 0건 · HE0 불변.**

★**제출 후에도 유지되는 금지 문장**: *"하네스층 감사가 하네스를 옳다고 판정했다"* ·
*"구멍 C가 닫혔다"* · *"P0-A가 R4에 답했다"* · *"규칙층이 닫혔다"* ·
*"9/9(13/13) 변이 통과이므로 계약이 옳다"*. ★**신설**: *"후속 검증이 SUBMIT-OK를 줬으므로
하네스가 검증됐다"* — 이 절이 말하는 것은 **"차단 9건이 닫혔고 새 차단을 못 찾았다"** 뿐이다.
그리고 **green 스트림 위 그래프 캡처가 가능한지는 여전히 미지수**다(불가능하면 `NOCAP`,
도구 가용성 사실이지 한정 상실이 아니다).

## 7. 후속 라운드 재현 방법
```bash
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd .../results/bcg_probe
python p0a_graph_sm_confinement.py --selftest     # ALL PASS, 171 s, 13 mutants x 3
bash -n p0a_graph_sm_confinement.sbatch
```
M1(A7 변이 3종)·M2(A7c 약점)·M3(게이트 가로채기)·M4(G1 역방향)·M5(진단 결손)·
M7(G4 퇴화 witness 재주입)·M8(가드 누출 3종)·M9(`GREEN_LEGS` 드리프트)·M10–M13은
전부 `import p0a_graph_sm_confinement as H` + 수 줄이며 본문에 결과가 실려 있다.
