# 사전등록 — λ0 (캠페인 0단계, λ\* 측정) **rev5** (2026-09-14)

> **이력**: rev1–rev4 **4연속 `NO-GO`**(GPU 지출 0). rev4 死因 = **N1**(앵커 술어 조건 (b)가
> live에서 실효 항등식) · **N2**(그 수리가 등록 라벨을 뒤집는데 미등록).
> 판정서 `VERDICT_lambda0_rev4_2026-09-14.md` §6의 **F1–F6**이 rev5의 필수 조건이며 **전부 GPU 0**이다.
> **사용자 결정(2026-09-14)**: *"(a) rev5 한 번 더"* + ★**정지 조건** — rev5도 **검정력 사유로**
> `NO-GO`면 rev6으로 가지 않고 **(b) 생산자 하네스 수정 + 재실행** 경로로 승급한다.
> 이 문서는 **claims-auditor 규칙층 감사 대기**이며 `GO` 이전 sbatch 금지.

## 0. rev4 → rev5 변경표 (재감사 색인)

| 조건 | 구현/등록 | 상태 |
|---|---|---|
| **F1** 조건 (b)를 셀별 재계산 | `lambda0_lambda_inf.py` 전면 재작성(`:108-146` 상수, `:198-276` 재계산, `:279-361` `decide()`) | **완료**(§1) |
| **F2** 분지 전환 사실과 바뀌는 전부를 실행 전 등록 | 이 문서 **§2** | **이 문서가 이행** |
| **F3** 외부 사실 승계 5항 | 이 문서 **§3** (+ ⑤는 코드·문안 동시 수정) | **이 문서가 이행** |
| **F4** 도달가능성 assert 집계 축 | 이 문서 **§4** + `lambda0_reachability.py` `REGISTERED_MAP` | **부분**(§4에 한계 등재) |
| **F5** 하네스 도달범위(변이 라우팅·Z1/Z4/Z5/Z19·죽은 상수·FALLBACK 예산) | `lambda0_mutation_check.py:176-225` · `lambda0_plan.py:118-123,:546-557` | **완료**(§5) |
| **F6** 등록 무결성(결정경로 sha·배너·`--record`) | `lambda0.sbatch:104-140,:166-177` | **완료**(§5) |

## 1. F1 — 조건 (b)는 이제 **셀별**로 재계산된다

술어는 `<instr>/I_log_offsets.txt` + **같은 job의 `srv_warmup.log`**에서 셀별
`max #running-req`를 **스스로 재계산**한다(파일 2개·GPU 0). 등록 규약 3종:
① 구간 경계 = **start 배타 · end 포함**(newpair §5-I3 (D4)) ② 세는 줄 = **전 줄**(1차) / `Decode batch`
한정(변형) ③ 앞 셀 drain tail **미제거**(1차) / 제거(변형). **세 규약이 전부 48 미만이어야 실격이
확정**되고, **규약이 갈리면(일부 통과·일부 실패) 그 자체가 실격 사유**로 등록된다
(`… conventions DISAGREE`) — 이것이 "어느 규약을 쓸까"를 자유도가 아니게 만든다.

### 1-1. ★셀별 실측 (job 907959 `instrument/`, 쉬핑 코드)

| 셀 | 구간(1-based) | 전 줄 | decode-only | 앞셀 tail 제거 | F5(≥48) |
|---|---|---|---|---|---|
| I2 (256,512) conc 1 | 85..4226 | **1** | **1** | **1** | ✗(판정 대상 아님) |
| **I3a_shapeA** inf/64 | 4227..8434 | **48** | **48** | **48** | ✅ |
| **I3b_shapeB** inf/64 | 8435..18380 | **2** | **2** | **2** | ★**✗** |

관측 줄 수 I2 4128/4119/4128 · I3a 3904/3880/3904 · I3b 9632/9481/9632, 구간 길이 4142/4208/9946.

### 1-2. ★rev4 판정서의 `I3b = 11/11/2`를 **정정한다**(11은 `splitlines()` 산물)
`srv_warmup.log`에는 로더 진행표시줄이 만든 **bare `\r` 116개**(`\n`-줄 38–62 내부)가 있다.
생산자 `i_mark`는 **`wc -l`**(`r2_correctness.sbatch:502`)로 오프셋을 만들어 **18,382줄** 기준인데
Python `splitlines()`는 **18,498줄**을 낸다 ⇒ 모든 오프셋이 **116줄 앞당겨지고** I3b 창이 I3a의
drain tail(그 `#running-req`가 **11**)을 삼킨다. **생산자 자신의 줄 번호 체계에서 I3b는 2**이다.
⇒ rev4 판정서의 "11"은 인용 시 **이 정정과 함께**만 쓴다. **판정 방향은 불변**(11도 2도 48 미만).
줄 분할 규약은 `wc -l`로 못박고 변이(F1c)로 고정했다.

## 2. ★F2 — 분지 전환: **`FALLBACK`을 1차 등록으로 승격한다**

### 2-1. 수리가 분지를 바꾼다 — 두 출력의 대조 (쉬핑 코드, 같은 입력)
```
BEFORE (rev4 술어를 그대로 복원)        AFTER (rev5 쉬핑 코드)
LAMBDA0_MODE=ANCHORED                   LAMBDA0_MODE=FALLBACK
LAMBDA0_LAMBDA_INF_A=3.093892399983411  LAMBDA0_LAMBDA_INF_A=
LAMBDA0_LAMBDA_INF_B=0.6955869013158362 LAMBDA0_LAMBDA_INF_B=
LAMBDA0_DISQUALIFICATIONS=[]            LAMBDA0_DISQUALIFICATIONS=["I3b_shapeB: max #running-req
                                          over its OWN log interval [8435, 18380] is all_lines=2,
                                          decode_only=2, drop_prev_cell_tail=2 < 48 …"]
```
★**이 문서는 분지를 알고 고른다. 그 사실 자체를 등록한다** — 그것이 T2(분지를 알고 고르고
숨기는 것) 死因을 피하는 유일한 길이다(rev4 판정서 F2). **1차 등록 = `FALLBACK`**,
`ANCHORED`는 **"수리 전 반사실"로만** 보존하며 **어떤 결과 주장에도 쓰지 않는다**.

### 2-2. 분지가 바꾸는 **전부** (쉬핑 `lambda0_plan.py` 출력 전사)

| 항목 | `ANCHORED`(반사실) | ★**`FALLBACK`(1차 등록)** |
|---|---|---|
| λ_inf (A, B) | 3.0939 / 0.6956 | **2.1 / 0.675** |
| seed 쌍 | 2732 / 1138 (max\|Ēbar−1\| 0.0186 / 0.0189) | **4386 / 4162** (0.0146 / 0.0151) |
| A 사다리 | 0.773 · 1.55 · 2.63 · 4.02 · 6.19 | **1.1 · 1.8 · 3.0 · 4.9 · 8.0** |
| B 사다리 | 0.313 · 0.487 · 0.800 · 1.18 | **0.45 · 0.62 · 0.85 · 1.15** |
| A 창(conservative) | [0.7455, 5.548] = ratio [0.241, 1.793] | **[1.045, 7.191]** = ratio [0.4976, 3.424] |
| B 창(conservative) | [0.303, 1.062] = ratio [0.4356, 1.527] | **[0.4275, 1.034]** = ratio [0.6333, 1.532] |
| 예산 @λ\*=λ_inf | 1.241 GPU-h | **1.348 GPU-h** |
| 예산 0.25× 코너 | 2.592 GPU-h | **3.509 GPU-h**(상한 3.60 이내) |
| A 저부하 후보 | a_r0, a_r1 | a_r0, a_r1 |
| B 저부하 후보 | b_r0…b_r3 | b_r0…b_r3 |
| F4 반복 셀 | a_r4_s2 (1.0119) · b_r3_s2 (0.9879) | a_r4_s2 (1.0151) · b_r3_s2 (0.9948) |

### 2-3. 반전 격자 (등록 — 이 표가 "분지를 알고 골랐다"의 내용이다)
**쉬핑 `lambda0_reachability.py`의 168점 지도**(grid = 0.1 0.2 0.25 0.35 0.5 0.7 0.85 1 1.2 1.45 1.8 2.5;
`B`=KNEE_BRACKETED `N`=KNEE_NOT_BRACKETED `H`=LADDER_TOO_HIGH `L`=LADDER_TOO_LOW):

```
A HHNBBBBBBBBL   B HHHHNBBBBBLL   MEASURED job 907959 (I3a 3.0939 / I3b 0.6956)   [ANCHORED 반사실]
A HHHBBBBBBBBL   B HHHHNBBBBBLL   point estimate lower  (λ*(A)=2.1)
A HHHBBBBBBBBL   B HHHHNBBBBBLL   point estimate upper  (λ*(A)=2.5)
A HHHBBBBBBBBL   B HHHHNBBBBBLL   absolute lower bound  (A=1.08, B=0.35)
A HHBBBBBBBBBL   B HHHHNBBBBBLL   absolute upper bound  (A=7.15, B=1.20)
A HHNBBBBBBBBL   B HHHHNBBBBBLL   B=6 plateau variant   (A=3.91, B=1.00)
A HHHHNBBBBBBB   B HHHHHNBBBBLL   ★FALLBACK literal ladders   [1차 등록]
```
**같은 참 λ\*에서의 분지 반전**(rev4 판정서 §1 V1-a/V1-b, 쉬핑 코드 산출 — 그 문서 sha
`7fdc7ab0fa58a18176520e5ae4d1dcfc4a1ab83449ea34eec8f4e0a7917c4506`에서 전사):
shape **B는 13점 중 5점 반전** — 0.35 `KNEE_NOT_BRACKETED`→`LADDER_TOO_HIGH` · 0.40
`KNEE_BRACKETED(0.400)`→`LADDER_TOO_HIGH` · 0.45 `KNEE_BRACKETED(0.450)`→`LADDER_TOO_HIGH` ·
0.50 `KNEE_BRACKETED(0.500)`→`KNEE_NOT_BRACKETED` · 1.20 `KNEE_BRACKETED(1.0373)`→`LADDER_TOO_LOW`
(★앞 네 점은 등록 커버리지 밴드 [0.303, 1.062] **안**). 비반전 점: 0.30 둘 다 `LADDER_TOO_HIGH` ·
0.675 둘 다 `KNEE_BRACKETED 0.675` · 1.00 `KNEE_BRACKETED 0.9152` vs `0.8925`.
shape **A는 12점 중 3점 반전** — 0.80 `KNEE_NOT_BRACKETED`→`LADDER_TOO_HIGH` · 1.00
`KNEE_BRACKETED(1.000)`→`KNEE_NOT_BRACKETED` · 8.00 `LADDER_TOO_LOW`→`KNEE_BRACKETED(7.1389)`
(앞 두 점은 밴드 [0.7455, 5.548] 안).

## 3. F3 — 외부 사실 승계

① **§4.1의 "`I3_max_running_req.txt` = 48 ⇒ 엔진 상한 도달(F5 충족)" 행은 폐기**하고 §1-1의
   셀별 표로 대체한다. 그 파일의 새 역할은 **veto 전용 교차검사**(전 로그 재계수 = 48 = 파일값;
   **실격을 강제할 수는 있어도 앵커를 부여할 수는 없다** — 보수 방향 단조).
② **N-7 · N-8 · N-9 · N-11을 문자 그대로 필수 병기에 추가**한다. 특히
   **N-8: `λ_inf(B) = 0.6956 req/s`는 포화 처리율·상한 앵커로 인용할 수 없다** ·
   **N-7: I2(동시성 1)의 TTFT 43.0 ms·ITL 12.96 ms는 D44가 아니라 비분할 108 SM 값이다**.
③ **NPC-I의 "D44" 절 정정**: *"분할 혼합비 미측정 — 'D44에서 쟀다'는 쓸 수 없다"*.
④ **"ITL 상수 3중 확인" 삭제**: 세 값(13.06 · 13.07–13.48 · 12.96)은 전부 **B=1·비분할 108 SM**
   이며 **같은 양의 3회 측정**이다(R4C-6). κ 모형이 요구하는 **D44·부하 배치(B̄≈6–27)의 ITL은
   여전히 미측정**. ★`lambda0_reachability.py:28-32`의 "confirmed three ways" 문안도 함께 정정한다.
⑤ **"판정 전" 문안과 그 테스트를 함께 고친다** — 새 (모델,백엔드) 쌍의 correctness 게이트는
   **이미 판정됐다**(job 908179 `PASS`, 2026-09-14). `tests/test_lambda0_prereg.py`의
   `TestPreregDocument.setUp`이 아직 **rev4 파일**을 읽고
   `test_does_not_prejudge_the_correctness_gate_outcome`이 아직 `판정 전` 문구를 요구한다 ⇒
   **rev5 파일로 repoint + 문구 교체**가 이 회차의 제출 게이트에 포함된다.
   ★**정정 문안(등록)**: *"이 사전등록은 correctness 게이트의 결과를 예단하지 않는다. 2026-09-14
   기준 그 게이트는 (Nano-9B-v2-Base, flashinfer, ctx16384, D44)에서 `PASS`로 **판정됐고**,
   그 `PASS`가 인증하는 것은 S·O층 토큰 id 동일성과 cudagraph(decode 한정) 유지 **한 문장뿐**이며
   **P2 착수를 승인하지 않는다**(NP-8)."*

## 4. F4 — 도달가능성 집계 축 (부분 이행 + 한계 등재)

`lambda0_reachability.py`가 이제 **시나리오별 168점 지도 전체**를 `REGISTERED_MAP`으로 고정하고
7행 전부를 인쇄·대조한다(§2-3). ★**등재해야 하는 사실**: 4개 verdict의 "정의역 비어있지 않음"은
**7 시나리오를 pooled한 뒤에만 참**이다 —
★**실제로 도는 시나리오(FALLBACK)의 shape A 행 `HHHHNBBBBBBB`에는 `L`이 없다** ⇒
**`LADDER_TOO_LOW[A]`는 그 시나리오에서 ∅**이다(R4C-5). 코드가 이 부재를 테스트로 고정해
조용히 다시 pooled되지 않게 했다. **시나리오별 assert 자체(F4 본문)는 여전히 열려 있다** —
이 문서는 그것을 **닫혔다고 주장하지 않는다**.
pooled census(참고): A `B`56 `N`3 `H`19 `L`6 / B `B`34 `N`7 `H`29 `L`14.

## 5. F5·F6 — 구현 완료 (검증 전사)

**변이 7종 전부 차단**(`lambda0_mutation_check.py`, 54/54 blocked, escape 0, 72 selftest run):
`Z1`(MULT[B] 1.70→1.16, **reachability 경유**로만 검출: shape B ratio 1.45 `KNEE_BRACKETED`→`LADDER_TOO_LOW`) ·
`Z4`(analyze `DRAIN_MODEL_TOL` 2→3, **reachability 경유**: ratio 0.50 `KNEE_NOT_BRACKETED`→`KNEE_BRACKETED`) ·
`Z5`(F5 문턱 48→32) · `Z19`(배수 가드 경계 `<=`→`<`) ·
**`F1a`(술어를 전역 max로 되돌림) → 변이본이 rev4의 결정을 그대로 재현하고 새 live leg가 거부**
(`{'mode': 'ANCHORED', … 'disqualifications': []}`) · `F1b`(구간 start 포함) · `F1c`(`splitlines`).
⇒ **F5-1의 라우팅은 장식이 아니라 하중 부재다**(Z1·Z4는 그 경로로만 잡힌다).
**F6**: `$OUT/REGISTRATION_SHA256.txt`에 결정 경로 + 사전등록 md의 sha256을 기록하고
**10개 미만이면 abort**(프로브 실행 14개 기록). `lambda0.sbatch:113` 배너의 "rev3" 오인쇄와
`:118`의 존재하지 않는 `--record` 플래그 제거 완료. ★**`PREREG_MD` 기본값이 이 파일명
(`PREREG_LAMBDA0_REV5_2026-09-14.md`)이며 없으면 `ABORT_F6`(exit 3, GPU 0)** 이다.
CPU 회귀 **697 tests OK**.

## 6. F1이 만든 **새 자유 표면 5종** — 전부 코드로 닫고 여기서 등록한다
1. **줄 분할 규약** `wc -l`(생산자와 동일) vs `splitlines()` — I3b 2↔11. **`wc -l`로 고정**, 변이 F1c.
   ★**분지는 두 규약에서 같지만 표는 다르다** — 수치를 인용할 때 규약을 병기한다.
2. **구간 경계** start 배타/end 포함. ★**이 아티팩트에서는 판별 불가** — 경계 4줄(84·4226·8434·18380)이
   전부 `#running-req` 없는 HTTP `GET` 줄이라 ±1줄 이동이 값을 바꾸지 않고 `lines_in_interval`만
   바꾼다. 그래서 규약은 **같은 로그의 실제 슬라이스로 만든 leg**(7903줄이 48, 7904.. 최대 43)로
   고정했다. **"live 데이터가 경계 규약을 입증했다"고 쓸 수 없다.**
3. **F5를 적용할 셀** = `request_throughput`을 인용하는 두 셀(I3a·I3b). I2는 **보고만 하고 판정하지
   않는다**(동시성 1은 구성상 F5 대상이 아니다). 두 선택 모두 여기서는 `FALLBACK`.
4. **규약 만장일치 요구** — 일부 통과/일부 실패는 그 자체가 실격(`conventions DISAGREE`).
5. **`I3_max_running_req.txt`의 새 역할** = veto 전용(위 §3①). **보수 방향 단조.**

## 7. 예산·벽시계
`FALLBACK` 최악 코너 **3.509 GPU-h** vs `#SBATCH --time=04:30:00` ⇒ 여유 **약 59분**.
제출 전 CPU 게이트가 **약 10분**(변이 하네스 5분[8 워커 병렬; 순차 시 ~45분] + selftest ~3.5분,
그중 reachability 88초)을 쓴다 ⇒ **여유 59분 중 ~10분**. CPU 코어가 8 미만인 노드에서는
`LAMBDA0_MUTATION_JOBS`로 낮춘다.

## 8. 승계 — 인용 금지 전수 (표제 문장 전사 · 전문은 rev4 `78bb617b…` §9–§10)

★**게이트 #6은 닫히지 않는다 (가장 중요한 승계 문장, Q1)** — 이 캠페인 0단계는 **어떤 결과가
나오더라도** 방법론 게이트 **#6**("용량 먼저 측정")을 닫지 못한다. 게이트 #6이 요구하는 용량은
**지표 절벽 대비** 용량이고 이 단계가 재는 것은 throughput 포화다. 905835 원자료를 정본 술어
(TTFT≤3000 ms ∧ 요청내부 token-ITL p95≤60 ms)로 재채점하면 shape B(8192-in)의 SLO 절벽은
**0.59·λ\*_throughput 아래**에 있다(**0.59×에서 goodput 53.8% · 0.89×에서 5.8% · 1.27×에서 0.8%**)
⇒ **λ\*_SLO(B) < 0.59 × λ\*_throughput**(비 추정 2.3–3.4×). 따라서 *"λ\*를 측정했으므로 W3/W4의
부하 라벨이 참이 되었다"* 는 문장은 **쓸 수 없다**.

- **Q2** — λ\*(B1)의 측정은 (Nano-9B-v2, flashinfer) 쌍의 R2 correctness를 **어느 정도도 확인하지
  않는다**(이 단계와 job 905835는 legacy 루프 + fixed split만 발화시켰다). ★**rev5 갱신**: 그 쌍의
  correctness는 이제 **별도 게이트(job 908179)가 `PASS`로 판정**했으며 **이 단계가 그것을 자기
  공로로 인용하는 것을 금지한다**.
- **Q3** — 이 단계가 내놓는 두 λ\*는 **W4를 파라미터화하지 못한다 — 상호 배타적이다**. 어떤 단일
  스칼라도 두 phase를 동시에 0.80×로 만들 수 없다(λ\*=0.675 → prefill 0.79× ✓ / decode 0.21–0.25× ✗;
  λ\*=2.1 → decode 0.79× ✓ / prefill 2.47× ✗). ★**Q3 편집 주석(인용문 불변)**: 병행 워크스트림이
  워크로드 정의를 phase별 독립 λ\*로 고쳤으므로 Q3의 마지막 문장은 여전히 참이다 — 이 단계는
  **두 수**를 줄 뿐이며 **그 코드 변경을 자기 공로로 인용하는 것을 금지한다**.
- **Q4** — *"B1의 용량이 시스템의 용량"* 이 아니다. 워크로드 이름(W8 "near saturation", W9
  "overload")은 **B1에서만** 참인 서술이고 B4에서 같은 trace는 자기 용량의 0.2×일 수도 5×일 수도 있다.
- **Q5** — shape A·B 두 shape를 쟀다는 것이 W1·W5·W6·W7·W8·W9의 부하 라벨을 고치지 않는다.
  그 워크로드들의 shape는 미측정으로 남으며 λ\*는 shape 의존적이다
  (이 단계 자신이 두 shape에서 다른 값을 낸다).
- **L1** — `achieved/realized_offered`는 **포화도가 아니다**(§8-2의 항등식). 이 추정량으로 얻은
  "비포화" 라벨을 out ≳ 256 토큰 shape에 인용하는 것을 금지한다.
- **L2** — rev2의 *"청정 비포화 셀 3/3이 1.00±0.02로 수렴"* 은 **(8192 in, 96 out) 한정 사실**이다
  (같은 보정을 (256,512)에 적용하면 기준선이 0.93–0.97로 내려간다). shape를 명시하지 않고
  *"실현-분모 보정이 인공물을 제거했다"* 를 인용하는 것을 금지한다.
- **L3** — `d44_r1_o96`이 "포화 셀"인 것은 **배수 인공물**이다(측정 0.8988 vs 이상값 0.9295).
  M1 차단의 데이터 근거로 인용 금지(합성 쌍둥이 케이스 **3b**만 인용 가능).
- **R3C-1** — rev3 이하의 규칙으로는 λ\*(shape B = 8192-in)를 **측정할 수 없다**
  (`KNEE_BRACKETED[B]`의 정의역이 λ_inf(B) ≤ 1.70 req/s인 모든 경우에 ∅). rev4가 E1/E2로 해소.
- **R3C-2** — `LADDER_TOO_HIGH`(÷4 재설계)는 rev3에서 **한 번도 발화할 수 없다**(A·B 공통). rev4 해소.
- **R3C-3** — ★**rev4·rev5에도 유효한 caveat**: 브래킷 양변의 **측정창이 서로 다르다**(저측 600–700 s,
  고측 170 s). 보고되는 λ\*(A)를 "용량"이라 인용할 때는 **(창 길이, N) 튜플을 함께** 적어야 하고
  사다리 상단 바로 위 구간에서 과소평가가 발생한다(참 8.0 → 보고 7.155, **−10.6%**, 라벨은
  `KNEE_BRACKETED`; §2-3 지도에 −7…−25%로 등재).
- **R3C-4** — *"변이 N/N 차단, escape 0"* 은 **등록된 모듈·변이 목록 한정 사실**이다(rev5에서도
  같은 제한이 유효 — §9 λ5-6).
- **NPC-I** — **λ_inf는 legacy warm-up boot · D44 · cudagraph ON에서만 측정되며 B4(true-dual)의
  포화 상한은 측정되지 않았다.** ★**정정 승계(F3③)**: *"분할 혼합비 미측정 — 'D44에서 쟀다'는
  쓸 수 없다"*.
- **N-7 · N-8 · N-9 · N-11**(F3②, job 907959 결과 감사) — 특히 **N-8: `λ_inf(B) = 0.6956 req/s`는
  포화 처리율·상한 앵커로 인용할 수 없다** · **N-7: I2(동시성 1)의 TTFT 43.0 ms·ITL 12.96 ms는
  D44가 아니라 비분할 108 SM 값이다**.
- **R4C-1…R4C-6**(rev4 판정서 §7) 전부 유효.

### 8-1. 실행 후 필수 병기 (rev1 판정서 §5 + NPC-I, 문자 승계)
①λ\*는 **B1(legacy, fixed D44) 한정 throughput 포화율**이며 split을 바꾸면 5× 변한다 ②정본 술어
goodput은 이 단계에서 측정되지 않았다 ③상단 셀의 TTFT/ITL 백분위는 **인용 불가**(의도적 과포화)
④`switch_count`·split 체류분포는 판정에 쓰이지 않았다(fixed split) ⑤포화 셀의 achieved는 도착
실현에 무관하나 **비포화 셀의 `ach/off`는 도착 실현 계수**이며 실현 분모로 바꾼 뒤에도 비포화 셀의
영점은 1이 아니라 **κ**다(L1) ⑥부하 생성기가 다르다(이 단계 `sglang.bench_serving` Poisson vs
캠페인 `pdmux_eval.trace_loadgen` 고정 도착시각) ⇒ **λ\*는 다른 하네스에서 측정된 상수로 캠페인을
파라미터화한다** ⑦**NPC-I**(위).

### 8-2. 배수(drain) 항등식과 그 귀결 (L1의 근거, 문자 승계)
`achieved/realized_offered`는 `(N/(N−1)) · span / duration`이고 `duration = span + L`(L = 배수)이므로
곧 **`(N/(N−1)) · span / (span + L)`** 이다 — 모델링 선택이 아니라 **항등식**이며, 그 영점이
**κ**(코드·표에서는 `kappa`/`kappa_pred`)다. shape A의 배수는 **511 토큰 × ITL**이고 probe C가
**이 arm(D44)** 에서 실측한 ITL p50은 **19.92 / 20.29 / 20.38 / 23.53 ms**다(job 907959 I2의
ITL(B=1) **12.96 ms**는 **비분할 108 SM 값이라 이 자리에 쓸 수 없다** — N-7).

### 8-3. 인용 고정 (교훈 80 · 게이트 #110)
캠페인 쪽 인용은 커밋 `bddff6a`에 고정한다. W4 shape의 정본 앵커 =
**`benchmarks/pdmux_eval/workloads.py:159-160`**(`8192 if is_prefill else 256` /
`64 if is_prefill else 512`) ⇒ **W4 decode phase는 (256, 512)** 이며 이는 W3 전부와 정확히 같다
(근사가 아니다). ★**이 앵커는 그 이후 W4 구현(phase별 독립 λ\*)이 줄을 보존했음이 확인됐고
`line_citations.json`에 2026-09-14 등록됐다.**

신규 게이트 제안 `G-λ4-1…3`(rev4 §8)은 이 문서가 이행한다.

## 9. 신규 인용 금지 / 필수 병기
- **λ5-1**(인용금지) *"rev5가 λ_inf를 앵커로 확인했다"* — rev5의 1차 등록은 **`FALLBACK`**이며
  앵커는 **자격 미달로 실격**됐다(I3b = 2 < 48, 세 규약 전부).
- **λ5-2**(필수병기) 셀별 값을 인용할 때는 **줄 분할 규약**(`wc -l` 기준)과 rev4 판정서의 `11`이
  `splitlines()` 산물이라는 §1-2를 병기한다.
- **λ5-3**(인용금지) *"live 데이터가 구간 경계 규약을 입증했다"*(§6-2).
- **λ5-4**(필수병기) *"실제로 도는 시나리오(FALLBACK)에서 `LADDER_TOO_LOW[A]`는 ∅"*(§4).
- **λ5-5**(필수병기) *"이 문서는 분지가 `ANCHORED`→`FALLBACK`으로 바뀐다는 것을 **알고** 썼고,
  그 사실과 바뀌는 전부(§2-2·§2-3)를 실행 전에 등록했다."*
- **λ5-6**(인용금지) *"변이 47/47 또는 54/54 차단 = escape 0"*을 **등록 목록 밖**으로 일반화하는 것
  (R4C-4 승계).

## 10. 이 회차가 하지 못하는 것 · 제출 게이트
- **λ\*는 아직 측정되지 않았다**(게이트 #6). 이 회차는 그 측정의 **0단계 설계**다.
- **P2는 열리지 않는다** — 블로커(λ0 미측정 · W4 λ\* 실측 부재 · 게이트 #6) 불변, Claim D 등급 불변.
- **F4의 시나리오별 assert는 미완**(§4).
- ★**결정 입력 3종이 git 밖이다**(`.gitignore:29,:30` — `srv_warmup.log`·`I3*.jsonl`) ⇒
  **fresh clone은 분지를 재도출할 수 없고** F6의 digest가 유일한 provenance다. 술어 selftest는
  입력이 없으면 **fail-closed**(저장소 단위 unittest는 해당 leg를 skip).
- **제출 게이트**: ①이 문서가 규칙층 `GO` ②§3⑤의 테스트 repoint + 문구 교체 완료
  ③`presubmit.py` 차단 0(또는 범위 한정 OVERRIDE + 사용자 승인) ④CPU 회귀 전체 통과
  ⑤`#SBATCH --comment="field=efficientai;appl=pytorch"` 확인.
