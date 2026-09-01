# VERDICT — CP 규칙층 감사 **2회차** (2026-08-28)


> ⚠️ **날짜 정정**: 파일명/본문 날짜 2026-08-28은 오기 — 실제 작성/실행일 **2026-09-01**. 상세: `DATE_CORRECTION_NOTE.md`.

**판정: `NO-GO`** · claims-auditor(적대, read-only) · 방법론 게이트 #34 1단 재제출 ·
GPU 지출 **0** · 새 성능 판정 **0건** · 정본 변경 **0건** · 저장소 파일 수정 **0건**

**대상**: `cp_rule.py`(RULE_REV=2) + `DESIGN_CP_BASELINE_REV2_2026-08-28.md` + `cp_rule_selftest.py`
+ `served_population.json` + `spec/reach_*.json`.
**단일 판정 질문**: *"rev2를 사전등록으로 승격해 Stage CP-0을 제출해도 되는가?"*
**1회차**: `../audit_cp_rules_2026-08-28/VERDICT.md`(`NO-GO`, D1–D9 · L1–L12).

> **메인 세션 독립 재확인(4건, 전부 CONFIRMED)**
> **K1** 현행 `design_reachability.py`(TOOL_REV=2)로 `spec_C1_canonical.json` 재실행 →
> `FINDING RESTRICTIONS_INERT` + `PRIOR_UNREGISTERED`, **`VERDICT: RESTRICTIONS_INERT`**
> (*"The restrictions excluded nothing. This run could not have failed, so its DISCRIMINATING
> would have carried no evidence."*). `presubmit.py` 현재 **BLOCK 3건**(그중 CP 1건). ✔
> **K6** *"3% 미만 차이는 headline 아님"*은 **`/scratch/ehmoon/whlee/CLAUDE.md:74`의 게이트 #3**이고
> `PROJECT_STATUS.md`에는 **0건**. `cp_rule.py`의 1차 임계 주석이 그것을 PROJECT_STATUS로 적었다. ✔
> **L1** `reports/CONSENSUS.md`에서 `2.15–2.35×`는 이제 **1863행**(설계는 :1803). ✔
> **환경(K1·L1의 공통 원인)** 이 세션 작업 중 **다른 세션의 커밋이 저장소에 들어왔다** —
> `1c6bb59`(14:23:35, CONSENSUS +169줄 ⇒ 줄 좌표 이동) · `5d180a6`(14:42:56, 도달가능성 도구
> rev2 ⇒ 기존 인증 전부 낡음). 새 도구는 **CP만이 아니라 TC1 `reach_spec_rev3_A.json`도**
> `RESTRICTIONS_INERT`로 막는다. ✔

---

## 답 (단일 판정 질문)

아니다. rev2는 rev1보다 **실질적으로 나아졌다**(argmax 제거·단일 레버 C1·조건부 지출 제거·모집단
재측정·`ci_shape` 전역 함수화는 전부 진짜다). 그러나 승격 불가 — 세 층이다.
**(1) 저장소 자신의 규율 도구가 지금 이 설계를 막는다**(K1). **(2) §6.2가 "rev2 최대 개선"으로 내건
*"C1은 크기까지 산다"*가 증거가 아니라 무지의 산물이다** — `capacity`를 물리적으로 거의 확실한
`above`로 고정하면 C1은 C4와 같아지고 크기 라벨 7개가 전부 소멸한다. §8-3은 그것이 미정임을
인정하면서 §6.2는 단정한다(자기모순). **(3) 각 수리의 파급이 재도출되지 않았다** — `ci_shape`는
전역 함수가 됐으나 나머지 3축은 여전히 산문 술어이고, 외부에서 넣은 **변이 14개 중 9개가
자기검사를 통과**했다(`capacity` 게이트 반전 · rev1 D5-3 단일채널 정체 게이트 복원 ·
`ladder_shape` 엄격성 · 등록 상수 6개 전부). 그리고 L1(번호체계) 수리 자체가 **규칙 파일 안에
새 출처 허위**를 만들었다.
**이번 회차 死因의 약 71%가 rev2 수리의 그림자다.**

---

## 1회차 死因 해소 대조표 — **CLOSED 3 / PARTIAL 6**

| D | 판정 | 요지 |
|---|---|---|
| **D1** `ci_shape` 전역 함수 | **PARTIAL** | 지목 좌표는 CLOSED(전역 함수·분할 증명·독립 2차 구현·변이 5·`straddles` 확정·power-first 비삼킴 전부 이행). 그러나 병(*"데이터→축 사상이 산문이면 구멍은 그대로"*)이 **5축 중 2축만** 고쳐졌다. `label()`은 `ci_shape()`/`ladder_shape()`를 **한 번도 호출하지 않는다**(사전계산 문자열만 소비) |
| **D2** argmax 제거 | **CLOSED** | 코드 전수 확인, 선택 단계 0건. 단 K5 파생 |
| **D3** ladder CI화 | **PARTIAL** | 세 절 축자 이행. 그러나 **등록한 작동특성이 틀린 것** — D3의 병은 *"실신호→폐기"*(대안 하 삭제율)인데 등록된 것은 귀무 하 차단율. `LADDER_CONFLICT`가 1차 CI를 여전히 선점 |
| **D4** 단일 레버 estimand | **CLOSED (scoped)** | 이 설계의 **최대 실질 개선**, 반증 실패. 단 `scope_banner`는 죽은 필드(L7) |
| **D5** arm 정체 | **PARTIAL** | 규칙층은 통일. 그러나 **§4.2가 요청 채널 단독으로 cps2048/4096을 탈락**시킨다 — rev1 D5-3의 단일채널 규칙이 arm-집합 결정 층으로 자리만 옮겼다 |
| **D6** 모집단 | **PARTIAL** | 측정은 **완전 CLOSED**(재실행 값 동일). *"근거 포함"* 절 미이행(K3) |
| **D7** 인용·capacity | **PARTIAL** | 절·값은 옳다. **줄 좌표가 틀렸고**(L1) *"무-모수"*가 거짓(K2) |
| **D8** 임계 3종 | **PARTIAL** | 예산 sacct 재현 확인(n=34, mean 8.02분, 7.22–11.93) · CI 추정량 명시 ✔. **새 자유 모수 3건**(δ 확률변수·capacity 추정량·CP-0(c) n 미등록) |
| **D9** 무조건 지출 | **CLOSED** | 규칙 파일에 지출 조건 0건 |

합격 기준 (a)가 *"부분 해소는 미해소"* ⇒ **(a) FAIL**.

---

## 死因 (런킬러) — 7건

### K1. 1차 대조의 도달가능성 인증이 현행 도구에서 `RESTRICTIONS_INERT`이고, `presubmit.py`가 지금 차단한다 **[NEW — 저장소 층 수리의 그림자]**
저장된 `reach_*.json` 5개에 `findings`·`tool_rev` 키가 **없다**(= tool rev1 산출물). 현행 TOOL_REV=2로 재실행:
```
spec_C1_canonical    RESTRICTIONS_INERT (실질 11라벨 전부 2/3 동일비율 생존) + PRIOR_UNREGISTERED
spec_C4_parity       PRIOR_UNREGISTERED
spec_D_cps4096       RESTRICTIONS_INERT (1/3) + PRIOR_UNREGISTERED
spec_D_cps4096_slack / spec_R    NOTHING_PURCHASABLE
```
`presubmit.py`의 `BLOCKING_REACH ⊇ {RESTRICTIONS_INERT, PRIOR_UNREGISTERED}`이고 CP spec은 이미
`active` 등재 ⇒ **현재 체크리스트 rc=1**. C1의 유일한 제약(`req_split=yes`)은 `DEGENERATE_ARM`
판정에만 쓰여 **실질 라벨을 하나도 배제하지 않는다**. §6.2 표의 `DISCRIMINATING`은 현재 트리에서 거짓.
**해소**: 5 spec에 `priors` 등록 + C1에 실제로 배제하는 제약 등록 + TOOL_REV=2로 5 인증 재생성 +
나머지 4 spec 레지스트리 등재.

### K2. `capacity`가 "무-모수"가 아니고, *"C1은 크기까지 산다"*가 무지의 산물이다 **[SHADOW of D7]**
반사실 실행 — C1에 `capacity=above`를 추가하면:
```
DESIGN substantive: {CP_WINS_RANK_ONLY:6, LADDER_CONFLICT:14, CP_LOSES_RANK_ONLY:6, CP_EQUIV_RANK_ONLY:4}
unreachable: [CP_EQUIV_SIZED, CP_LOSES_GE_DELTA, CP_LOSES_SIZE_UNRESOLVED, CP_LOSES_SUBDELTA,
              CP_WINS_GE_DELTA, CP_WINS_SIZE_UNRESOLVED, CP_WINS_SUBDELTA]      (= C4와 동일)
```
그리고 `above`는 물리적으로 거의 확실하다 — 규칙 정의상 `above` = "두 arm 중 **하나라도** achieved <
offered"이고 정본 §1-32(`CONSENSUS.md:1863`)가 이 trace HI를 **2.15–2.35× 과부하**로 기록한다.
⇒ CP-1a가 크기 라벨을 낼 가능성은 희박한데 §6.2 reading 1은 *"크기를 사려고 rev1이 만든 조건부
단계가 애초에 필요 없었다"*고 단정하고, §8-3은 같은 문서에서 *"`capacity` 값은 미정"*이라 적는다.
**미등록 자유 모수**: achieved throughput 추정량 · CP-0(e)의 어느 3 rate · n=2 노이즈·CI 부재 ·
동률 규칙 · **LO/HI 2상인데 축은 대조당 스칼라 1개**(estimand↔축 불일치).
**해소**: `capacity`를 코드 함수로 고정 + 추정량·rate·n·동률 등록 · §6.2 reading 1 철회/조건화 ·
§7에 *"CP-0(e) 전에 C1이 크기를 산다"* 금지문.

### K3. `REQ_SPLIT_MIN_N = ceil(δ·N) = 6`은 유도가 아니고, 전제를 정본이 반박하며, seed에 취약 **[SHADOW of D6/D8]**
δ=0.03은 **goodput(req/s) 비율**, N=200은 **요청 개수** — 등치하려면 *"각 요청이 goodput에 등가
기여"*가 필요하다. 정본이 반대를 측정해 뒀다: **§1-24**(`CONSENSUS.md:1853`) rate 2 stall probe
**17개 중 16개**에서 stall 전 구간 prefill 중이던 요청이 그 probe의 **최장 프롬프트(2469–2776 tok)**;
**§3 항목89**(`:4092-4100`) monolithic 스톨이 만든 58–60ms 이봉. ⇒ 긴 프롬프트 **하나**를 쪼개면
동시 decode 중인 **여러** 요청의 ITL 다리가 뒤집힌다. 그리고 문턱이 탈락시키는 arm(cps2048, 4개)이
하필 **>2048 tok = 정본이 스톨 원인으로 지목한 p99 꼬리**다.
**seed 취약(감사자 측정)**: seed 1/2/3/7 → cps512 51/44/57/38 · cps1024 10/16/12/**6** ·
cps2048 4/4/4/**0** · p99 2776/3024/2961/1249. **cps1024가 seed 7에서 정확히 문턱값**이다.
그리고 **10 rep의 seed 정책이 미등록**(하네스가 `--seed`를 안 주므로 기본 1 ⇒ 10 rep이 **동일한
200요청**을 서빙 ⇒ paired t-CI는 **한 realization 조건부**이고 그 배너가 없다).

### K4. 자기검사가 판정 분기를 덮지 않는다 — 외부 변이 **14개 중 9개 생존** **[SHADOW of D1]**
```
M1  ci_shape sign nonstrict        DETECTED     M4  capacity 게이트 반전          SURVIVED
M2  ci_shape ge STRICT             DETECTED     M5  DEGENERATE_ARM AND->OR       SURVIVED
M3  ladder BEFORE power            DETECTED     M6  ladder_shape lo>0 -> lo>=0   SURVIVED
M13 incoherent -> always None      DETECTED     M7-M12 등록 상수 6개 전부         SURVIVED
M14 within_delta EQUIV -> WINS     DETECTED       (REQ_SPLIT_MIN_N/DELTA_REL/BATCH_MARGIN_REL/
                                                   N_REP/ALPHA/LADDER_POINTS)
```
**M4는 K2가 걸린 바로 그 분기**, **M5는 rev1 D5-3 수리를 되돌린 변이**(CONSENSUS §3 항목53 정면
위반), **M6은 D3 수리의 핵심 술어**. 변이 목록을 자기검사 자신이 저술하는 한 축·상수·미작성 사상은
원리적으로 밖이다. 합격 기준 (d) 둘째 절(*"분기를 전수 덮는가"*) → **아니다.**

### K5. δ가 데이터 의존 확률변수가 됐고 편향이 크기 주장 쪽이며, 자기검사가 검증한 δ는 1차 대조의 δ가 아니다 **[SHADOW of D2/D4]**
rev1의 δ는 정본 상수 d44=3.220에 묶여 0.0966 **고정**이었다. rev2는 참조 arm을 대조마다 바꿨고
C1의 참조는 **한 번도 측정된 적 없는 `fused_mono`**다 ⇒ `lo >= delta` 비교가 **같은 10 rep에서
추정한 임의 문턱**과의 비교이며 그 확률성이 미계상. 정본 §1-1이 pdmux>fused를 기록하므로
goodput(fused_mono) < goodput(d44) ⇒ δ 축소 ⇒ `GE_DELTA` 진입 용이 ⇒ **크기 주장 쪽 편향**.
그리고 `cp_rule_selftest.py:43`은 `DELTA = 0.0966`(d44의 δ)만 검증한다 — **사전등록될 임계가
검증된 적 없다.**

### K6. L1(번호체계) 수리가 **규칙 파일 안에 새 출처 허위**를 만들었다 — 최소 6건, 그중 1차 임계 **[SHADOW of L1]**
rev2 헤더는 *"번호 + 문서명 + 문구 첫머리를 함께 적는다"*고 선언했다. **문서명이 틀렸다.**
`PROJECT_STATUS.md:5818`이 *"`CLAUDE.md`의 기존 8개 게이트에 **추가로**, 이 정본에 등재한다"*라
적어 두 체계가 **서로소**임을 밝힌다.

| rev2 표기 | 인용 문구 | 실제 출처 | PS 방법론 게이트 #N의 실제 내용 |
|---|---|---|---|
| PS 게이트 **#3** | *"3% 미만 차이는 headline 아님"* | **CLAUDE.md:74** (PS에 **0건**) | keepalive 관련 |
| PS 게이트 **#4** | conjunctive goodput | CLAUDE.md #4 | "집계 단위를 먼저 정하고 …" |
| PS 게이트 **#5** | 보고 필수 항목 | CLAUDE.md #5 | `--max-running-requests` 관련 |
| PS 게이트 **#6** | metric cliff·용량 먼저 | CLAUDE.md #6 | "논리적으로 독립인가" |
| PS 게이트 **#7** | duration 합산 | CLAUDE.md #7 | "게이트 자신이 #6을 위반할 수 있다" |
| PS 게이트 **#2** | 변화 trace | CLAUDE.md #2 | "파티션 활성률을 사전등록 게이트로" |

`DELTA_REL = 0.03`은 **1차 임계**이고 그 옆 출처 주석이 거짓 — **교훈 #80**의 교과서적 재발이자
1회차 L1이 기록한 *"정정문이 같은 문장 안에서 다시 틀린다"*의 **연속 2회차**.
게다가 rev2는 자기 규약을 **6번** 어긴다(DESIGN의 맨 `#N` 6건, 세 체계에 흩어짐).
★**정확한 인용(반증 실패)**: PS #24·#26·#34·#37·#66·#70, PS "다음 실험 gate" #8·#10 — **전부 실재·정확.**

### K7. 1차 운영점이 정본이 문서화한 **ITL metric cliff 위**에 앉아 있고 §7이 절벽 증폭 크기 주장을 막지 않는다 **[NEW]**
**§3 항목89**(`:4092-4100`) 58–60ms 이봉, 60ms 임계가 그 모드 위(d24: 55ms 10.2%→65ms 99.3%) ·
**§1-32 한정(3)**(`:1863`) 임계 55→60ms에서 pass율 **+72~+75%p**. CP 처치의 기전이 **바로 그 스톨
폭을 바꾸는 것**이므로, 기전대로 작동할수록 60ms에서의 goodput 차이는 **절벽 증폭분과 분리 불가**.
규칙의 유일한 절벽 게이트 `capacity`는 **과부하 형태**만 다루고 ITL 임계 형태를 다루지 않으며,
사다리는 **보고**가 아니라 **차단 스크린**이다. §7은 `LADDER_CONFLICT` 오독만 막고
**`CP_WINS_GE_DELTA` 크기의 직접 인용은 막지 않는다.**

---

## 국소 결함 (L1–L15, 요지)

| # | 내용 | 계보 |
|---|---|---|
| L1 | ★**정본 줄 좌표 2건 오류**: `CONSENSUS.md:1803`→실제 **1863**, `:1472-1473`→실제 **1533**. 둘 다 `d11243a`에서는 옳았고 커밋 **`1c6bb59`(14:23:35, +169줄)**가 밀었다 — DESIGN 파일 mtime(14:26)보다 **앞선다**. `check_line_citations.py --check`는 `0 compared, 0 violation OK`(공허) | SHADOW of D7/L8 |
| L2 | `presubmit_registry.json`에 5 spec 중 **C1만** 등재. §9-2는 "통과 시 등재"라 적었는데 이미 등재됨(문서↔상태 불일치) | SHADOW of L8 |
| L3 | ★§3.3 *"E-D 항목은 이 워크로드로 구매 불가"*가 **채널을 구분하지 않는다**. E-D의 cps 2048/4096은 **배치 예산 레버**인데 요청 채널로만 탈락시켰다. 평균 341 tok × 6요청 ≈ 2048 ⇒ 배치 캡은 상시 물릴 개연성 | SHADOW of D5/D6 |
| L4 | ★`ladder_conflict_null_rate` 부수 주장(*"양의 상관 ⇒ conflict 더 드물다"*)이 **증명 없는 단정이고 일반적으로 거짓**. 반례: 3점씩 두 블록(블록 내 ρ=1, 부호 반대, 블록 간 ρ=−0.3) ⇒ 평균 쌍 상관 **+0.22 > 0**인데 P(conflict)≈1. 자기검사가 이를 **검사가 아니라 print**로 실어 검증된 것처럼 읽힌다. 검사 자체(`rate < ALPHA`)도 거의 항등 | SHADOW of D3 |
| L5 | ★`LADDER_CONFLICT`가 1차 CI를 선점하는데, 사다리 6점은 **같은 rep의 재채점**이고 CLAUDE.md 게이트 #8이 *"SLO를 바꿔 평가할 땐 재스코어 금지"*다. **재스코어 파생 스크린에 직접 측정에 대한 거부권**을 줬다. C1 실질 세계의 **28/60(47%)** | SHADOW of D3 |
| L6 | **Holm과 `ci_shape`가 미연결** — `ci_shape`가 소비하는 CI가 raw t-CI인지 Holm 보정 구간인지 미정. C4까지 같은 가족에 넣어 1차 검정력을 깎는 것도 미논증 | SHADOW of D2 |
| L7 | `CONTRASTS`의 `scope_banner`·`primary`·`family`를 **읽는 코드가 없다**(죽은 필드) | SHADOW of D4 |
| L8 | **CP-0 (c)의 n 미등록**인데 paired CI를 요구. (e)가 n=2면 t(1) 임계 12.71로 발화 불능 ⇒ `batch_budget`이 `slack`으로 굳는다 | SHADOW of D8 |
| L9 | ★**n=10에서 처음 무가정 옵션이 열렸는데 쓰지 않았다** — 정본(`:2204-2212`)은 n=5의 부호뒤집기 두측 p 하한 2/32=0.0625를 기록하고 §1-1(`:1308`)은 *"등가 판정 29건 전부 paired-t 정규 가정 위"*라 적는다. n=10이면 2/2¹⁰=0.00195 ⇒ **정확 순열 CI 가능** | NEW |
| L10 | **rep seed 정책 미등록**(K3) ⇒ CI가 한 워크로드 realization 조건부인데 배너 없음 | NEW |
| L11 | §5.0의 *"한 rep의 5 arm 같은 노드"*는 **실행 가능**(S9)하나 그러면 job당 5 부팅이 되어 §5.5 예산 basis(`sacct --name=sgptv`, 1 arm/1 job)와 job 형태가 달라진다. §8-1 하드와이어 목록에도 그 구조 변경이 없다 | SHADOW of D8/L12 |
| L12 | `measure_served_population.py`가 하네스 토크나이저 경로를 재현 안 함(`get_tokenizer` vs `AutoTokenizer`), seed 대비 토크나이저 생성 순서도 다름. `provenance.matches_harness`는 미검증 주장 | NEW |
| L13 | §4.2의 p99 교차검증은 **199번째 순서통계량 1개**의 일치. 결론은 살았으나(S2) 동일성 단정은 과함 | NEW |
| L14 | rev1 SUPERSEDED 배너 3종은 **내용 정확**하나 *"다음 서술은 거짓"*이라 적어 나머지가 참인 듯 읽힌다. 1회차가 거짓으로 지목한 것 중 **최소 4건 누락**(§4.3 일반명제화·"처방 #1 첫 이행"·"전 arm 상수 −1"·n=6/예산 3× 과대) | NEW |
| L15 | `incoherent()`는 옳으나 §6.2가 도구의 자기 고지(*"CATEGORICAL 격자만 본다"*)를 인용 안 함 | NEW |

---

## 반증 실패 — 살아남은 것

- **S1 `served_population.json` 완전 재현** — 감사자 재실행: p50 217 / p95 1029 / p99 2776 / max 3089 /
  mean 341.4, `{512:51, 1024:10, 2048:4, 4096:0}`, total 68276 **전부 일치**. 샘플러 기전도 코드 확인. **D6 측정은 실질 CLOSED.**
- **S2 "p99=2776 교차검증"은 공허하지 않다** — seed 2/3/7에서 p99가 3024/2961/**1249**로 크게 흔들리고
  **seed=1에서만** 정본 §1-16의 2776과 일치 ⇒ 우연 아님(단 L13 한정).
- **S3 ★§0/D5 정정은 참** — `server_args.py:781-784`가 `_handle_gpu_memory_settings`를 **무조건** 호출,
  `:1196` `elif gpu_mem < 90*1024:` → `:1199-1200` 8192. A100-80GB는 앞선 `<20/<35/<60 GB` 어디에도
  안 걸리고 이후 재할당은 DP-attention(`:2652`)뿐. `g2_run.sbatch:114` plain은 빈 문자열,
  `p1op_run.sbatch:53-56`은 agnostic 분기에만. ⇒ **정본 fused arm은 8192에서 돌았다.**
  ★**이 캠페인의 동기는 rev1보다 강해졌고 그 강화가 정당하다.**
- **S4 예산 basis 정확 재현** — `sacct -X --name=sgptv -S 2026-07-01`: n=34 COMPLETED, **mean 8.02분,
  7.22–11.93**. 게이트 #77(안 산 프로브)의 1회차 지적이 실제로 닫혔다.
- **S5 코드 좌표 전수 정확** — `server_args.py:6125/6129-6131/1196-1200` · `multiplexing_mixin.py:1169-1177` ·
  `pdmux_context.py:20` · `holb_probe.py:574`(에코 ⇒ L6 재라벨 옳음) · `sharegpt_vary_bench.sbatch`
  전 좌표. §8-1의 `:35-41`은 rev1의 `:36-42`보다 **개선**.
- **S6 §4.3의 L5 정정 산술 정확** — `65536//2048 = 32 < 54` ⇒ cps2048은 퇴화 안 함. **예측**으로 명시(게이트 #70 준수).
- **S7 D2·D9는 진짜 닫혔다** — 규칙 파일 전문 확인, 선택/지출 조건 0건. `IMPOSSIBLE_WORLD` 산술 정합.
- **S8 `ci_shape` 전역 함수는 실제로 전역·배타·전사** — 엄격성 변이 2종을 자기검사가 잡는다.
  `pos_indet` 셀 서술도 참. **1회차 D1의 지목 좌표에서 완전 수리.**
- **S9 §5.0의 same-node 요구는 실행 가능** — `p1op_run.sbatch:51,98,127-134`가 이미 **job 1개 안 다중 arm
  부팅 + 순서 무작위**를 구현. "실행 불가능한 요구" 의심은 **반증**(L11 예산 basis 불일치만 남음).
- **S10 `LADDER_CONFLICT`가 도달가능성 판정을 만든 것 아님** — 비실질 강등해도 C1 10라벨·C4 3라벨로 **판정 불변**.
- **S11 기각 판본 검사는 여전히 진짜** — `CLIFF_ILLPOSED 36`, `NOTHING_PURCHASABLE` 재현.
- **S12 인용정지 위반 0건** — `check_citation_stops.py` 4파일 400/294/254/101줄 **전부 0위반/16규칙 OK**.
  §1.2 HE0 표는 정본 §1-7과 **완전 일치**.
- **S13 §5.0 부수 발견 참** — 정본이 쓰는 `sharegpt_vary_bench.sbatch:88,92`는 현재 94/96.
- **S14 §5.1(f) Granite 재현성 한정이 정본(`:1438-1440`)과 일치**(correctness 결함 아님).
- **S15 §3.2 식별가능성을 정본 §1-7 追記로 깨려 했으나 실패** — 그 追記는 **mean-ITL** 스코어러에
  대한 것이고 rev2는 그것을 명시 거부하고 **p95**를 쓴다. §5.0의 스코어러 선택이 §3.2를 구했다.
- **S16 1회차 S7 규율 전부 유지** + §8이 부채를 **먼저** 적은 것은 1회차 권고의 정직한 이행.
- **S17 1회차 L7(상대경로) 닫힘** — `out`이 절대경로가 되어 stray 파일 0건.

---

## 합격 기준별 판정 — **(a)~(e) 전부 FAIL**

| 기준 | 판정 | 근거 |
|---|---|---|
| (a) D1–D9 해소 | **FAIL** | CLOSED 3 / PARTIAL 6 |
| (b) ★국소에 그치지 않음 | **FAIL** | D1이 2/5축 · D5가 규칙층만 · D7이 절·값만(줄 좌표 신규 오류) · L1 수리가 문서명 오귀속 · D2/D4가 δ 파급 미재도출 · 인접 트랙 도구 수리 미추종(K1) |
| (c) 새 항등식·자유 모수 없음 | **FAIL** | `IMPOSSIBLE_WORLD` ✔ / `capacity` ✘ / Holm↔CI ✘ / ladder ✘ / `REQ_SPLIT_MIN_N` ✘ / **신규 δ 확률변수** ✘ |
| (d) 자기검사 실패 가능 | **FAIL** | 실패 가능은 참이나 **분기 전수 커버 아님(9/14 생존)** |
| (e) 인용·§7 포괄 | **FAIL** | 게이트 문서명 6건 + 줄 좌표 2건 + 자기 규약 위반 6건. §7 미포괄 최소 5종 |

---

## "수리는 국소, 주장은 전역" 판정 — **死因의 약 71%(5/7), 국소 포함 75–80%가 직전 수리의 그림자**

| 이번 결함 | 그림자의 근원 | 기전 |
|---|---|---|
| K4 | D1 | 전역 함수화는 참이나 그 *이유*를 나머지 3축·상수·`label()` 분기에 재도출 안 함 |
| K2 | D7 | `capacity`를 대조 위로 올렸으나 *"그럼 C1의 값은 누가 정하나"* 미재도출 ⇒ 미측정을 자유로 읽어 헤드라인이 됨 |
| K3·L3 | D6/D5 | 개수 재측정은 참이나 *"같은 단위인가"*·*"배치 채널은"* 미재도출. 단일채널 규칙이 arm-집합 층에 잔존 |
| K5 | D2/D4 | 참조 arm 이동이 δ를 정본 상수 → 미측정량으로 옮긴 파급 미재도출. 자기검사는 옛 δ만 검증 |
| K6 | L1 | 규약 선언만 하고 매핑 대조를 안 함. **"정정문이 같은 문장에서 다시 틀린다"의 연속 2회차** |
| L1 | D7 / 1회차 판정서 | 감사 판정서의 좌표를 **재검증 없이 승계**했고 그 사이 정본이 밀렸다(교훈 #79·#80) |
| K1 | **저장소 층** | 같은 날 도구가 rev2로 오르며 `RESTRICTIONS_INERT` 신설, CP 인증은 rev1 산출물로 잔존. **"세계모형이 실험설계 성장을 못 따라간다"의 부수형 — 이번엔 인접 트랙의 수리를 못 따라감** |

**§8 부채 6건 재분류**: ①②④⑤ 부채 유지 · **③(CP arm 용량 미측정)은 부채가 아니라 死因**(K2) ·
**⑥(스냅샷 미등재)은 부채가 아니라 진행 중 오류의 은폐막**(L1).

---

## 신규 방법론 교훈 후보 (등재는 doc-steward 판단)

1. ★★★★ **미측정 축을 "자유"로 읽으면 도달가능성 증명서가 무지를 능력으로 번역한다.** 축을 제약하지
   않는 이유는 (i) 실제로 갈릴 수 있다 / (ii) **아직 안 재서 모른다** 둘인데 증명서는 구분 못 하고 둘 다
   "도달 가능"으로 센다. ⇒ 각 비제약 축에 (i)/(ii)를 명시하고 (ii)면 그 축이 결정하는 라벨을 인증에서
   **제외**하라. 1회차 교훈 #1의 **쌍대 형태**.
2. ★★★★ **규율 도구가 개정되면 그 도구의 모든 기존 산출물이 자동으로 미검증이 된다.** ⇒ 산출물에
   **도구 판본을 필수 필드**로 박고, 도구 개정 커밋은 **등재된 소비자 전부 재실행**을 같은 커밋의
   의무로. "수리는 국소, 주장은 전역"의 **도구 판**.
3. ★★★★ **번호체계 수리는 "문서명을 적어라"로 끝나지 않는다 — 매핑을 대조해야 한다.** 저장소 안에서
   **2회 연속**(TC1 rev1 → CP rev2). ⇒ 인용 문구를 해당 문서에 grep해 **0건이면 실패**로 처리하는 기계
   검사를 붙여라(비용: 명령 1줄).
4. ★★★ **자기검사의 변이 목록은 자기검사가 저술하는 한 자기 편향을 못 벗어난다.** ⇒ 변이를 **분기·축
   값·상수의 전수 열거로 생성**하고 각 변이에 "어느 검사가 잡는가"를 표로 등록. 미매칭 변이 = 미검사 분기.
5. ★★★ **문턱을 "유도"라 부르려면 좌우변 단위가 같아야 한다** — 차원 검사 + 그 등가 가정이 **자기
   캠페인이 사려는 기전과 모순되지 않는지** 확인. 부수: seed 하나의 표본에서 잰 문턱을 "결정론적 상수"라
   부르면 seed가 바뀌는 순간 arm 집합이 바뀐다.
6. ★★ **차단 스크린이 등록할 작동특성은 귀무 발화율이 아니라 대안 하 삭제율이다.** 그리고 도구 출력의
   설명문과 `[PASS]` 항목을 **시각적으로 분리**하라(설명문이 검증된 것처럼 읽힌다).

---

## rev3 최소 경로 (권고 순서, 전부 GPU 0)

1. **K1** — 5 spec에 `priors` 등록 + C1에 실제 배제하는 제약 + TOOL_REV=2로 인증 재생성 + 4 spec 등재
   (가장 싸고, 없으면 `presubmit`이 기계적으로 막는다)
2. **K2** — `capacity` 코드 함수화 · §6.2 reading 1 철회/조건화 · §7 금지문 신설
3. **K3/L3** — `REQ_SPLIT_MIN_N` 폐기 또는 기전 기반 재유도 · arm 집합 결정을 CP-0(c) 뒤로 ·
   §3.3 채널 구분 재서술 · rep seed 정책 등록
4. **K4** — 변이를 분기·축·상수 전수 열거로 재생성(M4·M5·M6·M7–M12가 반드시 FAIL해야 한다)
5. **K5** — δ를 정본 상수에 고정하거나 확률성을 CI에 계상 · 자기검사 δ 교체
6. **K6/L1** — 게이트 문서명 6건 + 맨 `#N` 6건 + 줄 좌표 2건 정정 후 **즉시** `--snapshot` 등재
7. **K7** — §7 절벽 증폭 금지문 + 사다리를 차단에서 **크기 보고**로 재배치 검토
8. **L4–L15** + rev1 배너 누락 4건 보강

> **질문은 여전히 산다.** chunked prefill은 실재하는 미등록 교락이고(S3, 코드 직접 확인),
> 단일 레버 C1은 rev1보다 확실히 낫다(D4 CLOSED).
> **사지 말아야 할 것은 이 판본의 결정 규칙과 그 인증이다.**
