# 판정서 — Stage 0‴ A0 **하네스층** 적대 감사 (2026-08-25, claims-auditor, 게이트 #34 **2단**)

대상: `stage0ppp_a0_probe.py` · `stage0ppp_a0_analyze.py` · `stage0ppp_a0.sbatch`
기준: `PREREG_STAGE0PPP_A0_2026-08-24.md`(rev4) · 규칙 정본 `RULE_REV=4`, sha `583c4ab0…`
**GPU 지출 0**(sbatch 제출 0건). 모든 재현은 로그인 노드 CPU에서 실제 실행.

> ★**메인 세션 재현 확인(2026-08-25)**: B1·B2를 판정서 수령 후 **독립 재현**했다.
> `_kernels()` → `ValueError: @jit functions should be defined in a Python file`,
> 그리고 `libtorch_cuda.so`에 `CUDA graphs must be captured on a non-default stream.`
> 문자열 실재. ⇒ **B1·B2 CONFIRMED.**

---

## ① 단일 판정

# `NO-GO` — 지금 제출하면 안 된다

**★死因은 없다.** 등록 질문도 규칙층도 살아 있고, 차단 15개는 **전부 편집으로 닫히는 구현층 결함**이다.
그러나 둘은 **런킬러**(제출하면 반드시 측정조건 라벨만 나온다), 셋은 **어댑터가 사전등록의 결정량을
표현할 수 없게** 만든다.

| 순서 | 무엇이 먼저 죽는가 | 나오는 라벨 |
|---|---|---|
| 1 | 프로브가 **triton 커널을 하나도 만들지 못한다**(B1) | `MEASUREMENT_ABSENT` (확률 ≈1) |
| 2 | B1을 고쳐도 **L2·L2′ 캡처가 default 스트림이라 실패**(B2) | `PROBE_INVALID` (확률 ≈1) |
| 3 | 둘 다 고쳐도 어댑터가 **Q1=아니오를 표현 못 한다**(B3·B4) | `{KSET_CONSTRUCTIBLE, TRACE_TRUNCATED}` **이진** |

★3차 감사가 남긴 예측 — *"남은 위험이 사는 곳은 하네스다(`l2_post`·`profile`·`ctx`·`stream`)"* — 는
**네 축 전부에서 적중했다.**

---

## ② 어댑터 계약 — World 축별 판정 (14축)

| # | 축 | 값이 어디서 오나 | 등록 일치 | 판정 |
|---|---|---|---|---|
| 1 | `run_ok` | 하네스가 고른다(상수 `True`) | 부분 | ★`B14` 무해하나 死코드 |
| 2 | `export` | 측정(부분) | ✗ | ★★`B10` **fail-open**(`nsys stats` 자체가 죽으면 `ok`) + green readout 실패를 여기로 오귀속 |
| 3 | `l1` | 측정 | ✓ | 통과 |
| 4 | `l2` | 측정이나 **오염**(L4=0이면 L2′ 행을 L2가 흡수) | ✗ | ★★`B3-b` |
| 5 | `l3` | 측정 | ✓ | 통과 |
| 6 | `l2_post` | ★★★**측정이 아니라 L4에서 파생** | ✗✗ | ★★★`B3` **최대 결함** |
| 7 | `capture` | 프로브 raw. `L2p_capture` 미독 | 부분 | ★★★`B2` + ★`B14` |
| 8 | `green` | 측정 — **프로브 자신의 드라이버 호출** | ✓ | 통과(비순환 확인). 실패 시 오라벨(★`B15`) |
| 9 | `profile` | 측정에서 **버킷팅** | ✗✗ | ★★★`B4` |
| 10 | `ctx` | 측정에서 **매핑** | ✗✗ | ★★★`B5` |
| 11 | `stream` | 측정 | ✓ | ★★`B12`(항등식은 **아님** — 위험은 반대편) |
| 12–13 | `join_target`·`join_rate` | 측정 | ✓ | 통과 — 3분기 전부 재현 |
| 14 | `l2_join` | ★★★**측정이 아니다** | ✗✗ | ★★★`B7` — R6 수리 무력화 |

**companion**: `companion()` **호출이 어디에도 없다**. `--sqlite-graph`는 필수 인자인데 **한 번도
읽히지 않는다**. `E1B_*` 8개 라벨 **전부 도달 불가** → `B6`.

### ★ `ctx` 정밀 판정
`"parent"` ⟺ **L4 행에 0/NULL 아닌 greenContextId가 2개 이상**. 실측:

| 주입 | 나온 `ctx` | 라벨 |
|---|---|---|
| `greenContextId = NULL`(=부모) | `null` | **`KSET_STREAM_ONLY_ATTRIBUTION`** |
| `greenContextId = 0`(=부모) | `zero` | **`KSET_STREAM_ONLY_ATTRIBUTION`** |
| 서로 다른 green id 2개 | `parent` | `ATTRIBUTION_DISCONFIRMED` |

**옳지 않다.** nsys 자신의 리포트가 `NULLIF(greenContext, 0) AS "GreenCtx"`(`cuda_gpu_trace.py:88`)로
**0을 "green 아님"의 인코딩으로 쓴다.** ⇒ 등록된 **적극적 반대 증거**가 `0`/`NULL`로 나타나는데
어댑터가 그것을 최상위 계열 실질 라벨로 보낸다. 규칙이 N4에 대해 스스로 적어둔 문장
(*"positive disconfirming evidence produced the top positive label"*, `rule.py:204-207`)이
**한 층 아래에서 그대로 재발**했다. ★어댑터는 `contextId`를 **SELECT해 담아놓고 한 번도 쓰지 않는다**.

### ★ `profile` 정밀 판정
어댑터 문턱 `0.995/0.905/0.885` vs 규칙 `PROFILE_FRAC`. 실측:
- **밴드 반전**: 측정 frac이 **정확히 `Q1_FRAC`(0.90)** 인데 규칙 `T9a`와 **반대로** 채점(어댑터 컷 0.905).
- **`tail_missing` 오정의**: `len(seq) < 20`뿐이고 `seq[-1]==0` 가지는 **死코드**.
- ★**프로브 그래프가 노드 1개** ⇒ `expected=1` ⇒ 결손이 있으면 **무조건 `tail_missing` → `TRACE_TRUNCATED`**:
  `19/20·18/20·10/20·1/20` 전부 동일 라벨 ⇒ **`PRIMARY_ESTIMAND_UNCONSTRUCTIBLE` 등 5개 도달 불가,
  `Q1_FRAC`은 한 번도 검사되지 않는다.**
- **`eager_only` 도달 불가**: `elif L4 and not any(r["node"] for r in L4)` — `L4`는 이미 `node` 필터를 통과.
- ★**자기참조**: `profile` 도달 집합이 *"nsys가 노드 행에 launch correlationId를 공유시키는가"* 에 의존 —
  **그것이 바로 Q2b, A0가 재려는 양이다.**

### ★ `stream` — 항등식은 아니다
`mismatch`가 나는 세계를 실제로 만들었다. 위험은 반대 방향: 그래프 내부 스트림 거동이
**`CONTRADICTORY_ATTRIBUTION`("우리 버그, 판정 안 함")** 으로 사전 라벨돼 판정이 소거된다(`B12`).

---

## ③ 다리 귀속 · sbatch · fail-closed

**3-1. NVTX 없는 다리 귀속 — 성립하지 않는다(현 상태)**: `exec`로 만든 함수는 `linecache`에 없어
`triton/runtime/jit.py:503-505`의 `inspect.getsourcelines`가 `OSError` → `ValueError`. **첫 커널부터 터진다.**
⇒ 커밋 메시지의 *"one triton jit function per leg … NVTX가 필요 없던 이유"* 는 **한 번도 실행된 적 없는 주장**.
**L4=0이면 `cut=None`이라 L2′가 0으로 잡힌다**(실측) ⇒ **등록된 결정적 음성이 측정조건으로 오분류**.
같은 경로가 green 생성 실패·eager fallback도 삼킨다. **`graph` 실행은 돌고 비용을 쓰지만 산출이 0.**

**3-2. sbatch — 대체로 이행, 빠진 것 3개**: 스위치 전수·`node` 먼저·`graph:host-only`·비exclusive·
`--comment`·`--time`·`TRITON_CACHE_DIR`·규칙 sha(**네 곳 전부 일치 확인**) ✓ / **권한 3종·컴퓨트 노드
버전·git HEAD가 `.json`에 없다**(sbatch 자신이 `.out`을 비인용으로 선언) · **프로브·어댑터 미핀**.

**3-3. fail-closed — 깨졌다(두 형태)**: `StringIds` 없음 / `demangledName` 없음 → **크래시**(라벨 아님) ·
`greenContextId` 컬럼 없음 → **`KSET_STREAM_ONLY_ATTRIBUTION`**(실질 라벨로 샘).

---

## ④ 차단 (요약)

**런킬러**: `B1`(triton jit 불가) · `B2`(default 스트림 캡처 금지).
**결정량 표현 불가**: `B3`(`l2_post` 파생) · `B4`(profile 4종) · `B5`(`ctx=parent` 오매핑) ·
`B6`(companion 미구현) · `B7`(`l2_join` 공허).
**라벨 채널**: `B8`(컬럼 부재→실질 라벨) · `B9`(크래시) · `B10`(export fail-open) · `B11`(§10 병기 누락) ·
`B12`(내부 스트림→"우리 버그").
**등록 이행**: `B13`(행 술어 미등록 변경) · `B14`(핀 부재) · `B15`(green readout 실패 오라벨).

---

## ⑤ 반증 실패

1. **`green` 축 비순환성** — nsys `isGreenContext`를 쓰지 않고 프로브 드라이버 호출. `primary_sm` 결함을
   실제로 고쳤다. **작성자 주장이 맞다.**
2. **`cuStreamGetGreenCtx` 실증 선례**(`smid_l0_raw_889631.json`, `rc=0`).
3. **파티션 목표값** `divide_sm(108,(8,0),2) → [(74,34),(54,54)]`, 등록값 일치.
4. **`SPIN_NS ≥ 100 µs` 보장** 성립(`iter_cap`보다 시간 조건이 먼저 문다).
5. **warmup 오염 공격 실패**(eager라 `graphNodeId==0`으로 배제 — 단 **등록된 이유로는 아님**).
6. **최상위 위양성 경로를 못 찾았다** — 배관 실패 15종 주입, 부당한 `KSET_CONSTRUCTIBLE`은 `B7` 하나뿐.
7. **sbatch §6 전수** — `nsys --help`로 재확인, 틀린 것 없음.
8. **`stream` 축 항등식 혐의 실패** — `mismatch` 세계 실재.

> ★ **"반증 실패 — 단 B1–B7을 닫기 전엔 `GO` 아님."**

---

## ⑥ 값어치 — **없다. 단 이유는 가격이 아니다**

가격은 싸다(상한 0.33 GPU-hr). **문제는 산출 분포**: 오늘 제출하면 `MEASUREMENT_ABSENT` 확률 ≈1(B1),
B1만 고치면 `PROBE_INVALID` ≈1(B2), 둘 다 고쳐도 답이 **이진**이고 정보 있는 쪽은 **그 사실을 진술하는
것이 금지된 라벨**로 나온다(B3·B4). 수리는 **전부 편집**이고 **대부분 GPU 0으로 검증된다**.
⇒ 권고는 *"사지 마라"* 가 아니라 **"지금 이 배선으로는 사지 마라"**. 3차 감사의 *"구매 저지 사유는
소멸했다"* 는 **규칙층에 대해** 유효하고 이 판정서는 그것을 뒤집지 않는다.

### 절차 (GPU 0 → 1회 제출)
①B1·B2·B3 ②B4·B5·B7 ③B6 결정(구현 또는 `graph` 삭제, **둘 중 하나를 사전등록에 등재**)
④B8–B11 ⑤자유 모수 재등록(#15·#19·#27) ⑥**변이 테스트 4종**(L4=0⇒`UNCONSTR` · gctx=0/NULL⇒`DISCONF` ·
frac=0.900⇒`OK` 쪽 · L2 조인 실패⇒`EXPUNVER`) — **뒤집히지 않으면 수리가 안 된 것**(교훈 #53)
⑦**배관 스모크를 job 안 leg 0으로**(교훈 #25) ⑧1회 제출.

---

## ⑦ 쓰면 안 되는 문장 (신설)

- ✗ ★★★*"어댑터는 fail-closed다"* — 반증됨(B8·B9).
- ✗ ★★★*"다리 귀속이 커널 이름으로 선다 / NVTX가 필요 없음이 확인됐다"* — **미실행**.
- ✗ ★★★*"A0가 Q1을 `Q1_FRAC=0.90` 기준으로 채점한다"* — 그 문턱은 **한 번도 평가되지 않는다**.
- ✗ ★★★*"`ATTRIBUTION_DISCONFIRMED`가 적극적 반대 증거를 잡는다"* — 등록 의미로는 **도달 불가**.
- ✗ ★★★*"E1-(b)를 같은 job에서 산다 / 추가 비용 0"* — companion은 **호출되지 않는다**.
- ✗ ★★*"`TRACE_TRUNCATED`가 나왔으니 트레이스가 잘렸다"* — L4 비었음·green 생성 실패·eager 실행을 전부 포함.
- ✗ ★★*"`l2_join`이 R6를 이행한다"* · ★★*"권한 3종이 아티팩트에 병기됐다"* · ★*"규칙 sha가 핀돼 있으므로
  채점이 보호된다"*(어댑터·프로브는 미핀).
- ✗ ★*"이 판정서가 3차 감사의 '지금 사라'를 뒤집었다"* — **뒤집지 않는다.**

### 참고 — 규칙층 관찰(하드 스톱 준수, 감사 아님)
`rule.py`에 `t18`·`t23`이 각각 **두 번 정의**(뒤가 앞을 그림자 처리), `t22`는 정의됐으나 미등록 死코드.
**의미 변화 없음**(`all_pass: true`, sha 네 곳 일치). 재감사 요구 없음.
