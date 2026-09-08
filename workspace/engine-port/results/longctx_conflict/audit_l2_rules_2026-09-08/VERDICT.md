# 판정서 — `PREREG_L2_2026-09-08.md` rev1 규칙층 적대 감사 (1회차)

2026-09-08 · claims-auditor · **`NO-GO`** · 死因 8 · 차단 11 · 반증 실패 9
· GPU 0 · sbatch 0 · 파일 수정 0 · **트랙 계열 누적 11연속 `NO-GO`**

★**기록 경위**: 감사가 read-only 규율상 파일을 쓰지 않아 **메인 세션이 반환문을 기록**한다.
§9에 메인 세션이 **독립 재실행으로 확인한 항목**을 표시한다.

## 단일 판정 질문에 대한 답

> **Stage 0이 실제로 가르친 것은 "선언(declaration)을 실현(realization)으로 읽지 마라"였는데,
> 이 설계는 그 선언을 *컨트롤러의 목표*에서 *드라이버의 할당*으로 옮겨 앉혔을 뿐이고,
> 그 이동을 정당화하려고 §3.2가 잔류(residency)를 재는 유일한 채널을 판정에서 금지했으므로,
> "실현"이라는 단어가 뜻하던 것이 이번 판본에서 구조적으로 측정 불가가 되었다.**

자유 표면 이동: `자 → 추정량 → 격자 m` **→ `목표 → 할당`**.

## 死因 (fatal) 8건

**F1 ★★ §2.2 전체가 오귀속.** Stage 0의 `PIN_CHECK`(`stage0_xctrl/
stage0_pdmux_capture.sbatch:173-179`)는 `controller_decision.target_decode_sms`의
**히스토그램** = 정책의 **목표값**이다. `CONSENSUS §1-25(a)`의 항등식은
`e1_pin_check.py`의 **prefill 축 시간가중 pin 게이트**(다른 하네스·축·날짜·실패 모드).
★`CONSENSUS §1-22`가 **이 혼동을 선제 금지**해 뒀다 — *"Stage 0(§1-21)의 D108 앵커
실패와 같은 구조이나 **원인은 다르다**"*. ⇒ §2가 말한 "두 기구 고장"은 실은 **한 기구,
한 기전**(target≠realized)이며 §2.1과 §2.2는 같은 사건의 두 서술이다.

**F2 ★★ 틀린 게이트를 폐기하며 필요한 계측기를 함께 버렸다.** `e1_pin_check.py`에는 두
함수가 있다 — `compute_time_weighted_pin_gate(:226)`(§1-25 항등식, 폐기 옳음)과
**`compute_decode_realized(:411)`**(decode 축 잔류). 후자는 `CONSENSUS §1-26`이
*"`E1_DECODE_REALIZED≥0.90`이 **sticky에서는 항등식이 아닌 진짜 게이트**"*라 **지명**했다.
§3.2의 금지문 한 줄이 둘 다 덮는다 ⇒ **없던 병에 약을 쓰고, 있는 약을 버렸다.**

**F3 ★ green readout은 잔류가 아니라 능력(capability)을 잰다.** 모듈 자신이
*"a one-shot startup observation"*(`multiplexing_mixin.py:223`)이라 적고 호출은 요청 서빙
**이전**(`:125`). stream-group별 드라이버 smCount만 보고하며 **decode가 어느 그룹에서
돌았는지 구조적으로 모른다.** ⇒ §2.3이 위험이라 지목한 희석(4–19%)을 이 계측기는
**원리상 탐지 불가**. 라벨 이름은 `REALIZATION_FAILED`인데 재는 것은 ALLOCATION(게이트 #72).

**F4 ★★ §3.2 검증은 이 격자에서 실패할 수 없다(게이트 #9의 15번째 재발).**
A1 원자료가 드라이버 파라미터를 기록해 뒀다 — **A100 green-context 입도 2 SM·정렬 2**.
등록 격자 `{16,44,92}`는 **전부 짝수·전부 ≥2** ⇒ 요청↔readout 불일치는 홀수/2 미만에서만
가능하고 격자에 그런 셀이 없다. `REALIZATION_FAILED` 정지 규칙은 **죽은 가지**.

**F5 ★★ §2.3의 사실 주장이 이미 디스크에서 반증돼 있었고 방향이 반대다.**
(i) Stage 0의 부하는 decode-only가 **아니었다** — `stage0_client.py:13-19`가 이 기전에
2026-07-25에 이미 이름을 붙이고 대책을 구현했다(*"a pure decode-only phase is forced to
full 108 SM … The keepalive holds split_prefill active"*), 분할 arm은 `--keepalive-prefill 2`로 돌았다.
(ii) 잔류는 이미 측정됐다 — `PARTITION_RESIDENCY_STAGE0.md`: D16/D44/D92가 nominal 분할을
**60–97%** 실현(예측의 정반대).
(iii) `D16 ≡ D108`의 실제 원인은 **둘 다 16이었기 때문**이다(D108 arm이 82–97%에서 16 SM).
⇒ **사전등록이 자기 §2.1이 인용한 바로 그 C1 재집계를 거꾸로 읽었다**(항목79).
부수: §4 S0 스모크 #1은 **이미 지불된 프로브를 다시 사는 것**(게이트 #77의 반대 형태).

**F6 §0의 "바꾸지 않는 것"이 거짓.** Stage 0은 decode-only가 아니었으므로(F5),
"decode-only 유지"는 **선언되지 않은 네 번째 변경**이고 "세 가지만"이 거짓이다.
`longcontext_trace_plan.md` §6의 "decode-only"를 **미검증 상속 전제**로 베꼈다.

**F7 `BINDING` 문턱의 출처가 질문을 바꾼다.** 유일 후보 C2 `2.36–2.91×`는 **다른 regime의
효과 크기**이지 결정 문턱이 아니다. 문턱을 2.36으로 두면 `NON_BINDING`의 뜻이 "민감도 없음"이
아니라 **"8B short-ctx보다 작음"**이 되는데, §3.4·§1은 거기서 *"long-ctx 충돌 死·HE0 강화"*를
도출한다 — **그 도출이 타당하지 않다.** 게다가 Stage 0 H arm D16/D92가 **3.35–3.58×**로
문턱 위에 앉아 있어 도달가능성이 이미 기울었다(게이트 #93+#86).

**F8 1차 추정량 안에 처치와 정렬된 미통제 구조 차이.** `r = ITL(16)/ITL(92)`의 분자·분모
부팅이 **stream group 수가 다르다** — `pdmux_d16.yml: sm_group_num 3` vs
`pdmux_d92.yml: sm_group_num 4`(D=92는 guard-satisfier 행 필수). green context 수가 다르면
cudagraph 캡처 집합·메모리 분할이 다르다. `CONSENSUS §1-26`이 이미 등재했는데 사전등록은
**언급조차 하지 않는다** — §2.3이 배웠다는 *"confound INSIDE the estimand"*의 재발.

## 차단 (blocking) 11건 요약

B1 등록 도구 `unpaired_t_ci`가 **`main`에 없다**(브랜치에만) — 게이트 #97의 정확한 형태 ·
B2 `m=9`가 실제 검정 수와 불일치(비율 CI 9 + T-vs-M 3 + ctx 추세 + 단측 3), D=44가 family에
있는지 미등록 · B3 §5-(b)가 열려 있으면 `m`이 구속력 없음 · B4 `UNDERPOWERED`의 "크다"가
자유 모수 · B5 "ctx와 함께 증가"의 검정통계량 미등록(게이트 #66) · **B6 물리적으로 결정적인
4번째 결과(ctx-무관 binding)가 `INCONCLUSIVE`로 버려진다** ⇒ §1의 *"어느 쪽이든 정보가 있다"*가
이 라벨 집합에서 거짓 · B7 arm×노드 교락, Stage 0이 갖고 있던 **job-내 노드 페어링을 버린다** ·
B8 부하 파라미터(동시성·출력 토큰·프롬프트 수) 미등록 — decode batch가 답을 정한다 ·
B9 job 893663 인용 스코프 위반(원 문서가 *"green_sm이 D를 따라간다(일반형)"* 금지, D=92 미관측) ·
**B10 금지한 계측기의 값(`E1_DECODE_REALIZED`)을 채택한 계측기의 근거로 인용** ·
B11 §3.1의 ctx 미결은 grep 한 번이었다(`stage0_pdmux_capture.sbatch:103`
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` — **넘겨서 돌렸다**; 저장소에 등록된 입장도
있다: `lff_bench.sbatch:32` *"L=16000 > Zamba2 4096 (**timing valid**)"*) ·
B12 드라이버 입도 정지 규칙 리스크 미등록.

## 합격 기준 판정

| # | 기준 | 판정 | 근거 |
|---|---|---|---|
| 1 | 두 기구 고장 실재 + 설계가 둘 다 닫음 | **불성립** | F1(기구는 하나), F2 |
| 2 | 검증이 실패할 수 있음 | **불성립** | F4(격자 전부 짝수), F3 |
| 3 | 라벨 전수·상호배타·도달가능 | **부분 불성립** | 전수·배타는 성립. `REALIZATION_FAILED` 도달 불가(F4), 결정적 4번째 결과 미할당(B6) |
| 4 | 미결 3건이 정말 비어 있음 | **불성립 3/3** | §3.1 **표가 이미 `H=Zamba2-2.7B`와 `{16,44,92}`를 등록** ⇒ (b)(c) 도달가능성 0. 표 vs 산문 자기모순에서 사전등록은 **표가 이긴다**. (a)는 F7 (게이트 #86) |
| 5 | 금지문이 자신은 참 | **불성립** | (i) *"green readout이 확인하기 전엔"* → green readout은 실현을 **확인할 수 없다**(F3) (ii) §2.3 본문 자신이 금지된 소급 설명을 하고 있고 그것이 반증 대상(F5). 게이트 #90 |

## 반증 실패 9건 (살아 있음 — rev2가 승계 가능)

1. **`decode_sms=108`이 sticky 축의 점이 될 수 없다 — 코드가 정말 강제**
   (`multiplexing_mixin.py:331-338` `1 <= i <= real_sm_group_num-2`,
   `pdmux_context.py:124-127`이 `(0,108)`을 마지막 인덱스로 하드코딩). **축 제거는 옳은 판단.**
2. §2.3의 엔진 인용문 **정확**(문구·4-19%·81-96%·job 872077 전부 일치).
3. §2.3(b)의 기전은 코드에서 **정말 따라온다**(`adjust_stream_groups:922-924` / `:952-953`) —
   무너지는 것은 그 예측을 **Stage 0에 적용한 부분**뿐이다.
4. `real_sm_group_num >= 3`은 만족되며 애초에 구속하지 않는다(`pdmux_context.py:35-36`이
   config 단계에서 이미 거부). 3점 전부 sticky 타깃 도달 가능.
5. ★**decode-only + sticky가 Stage 0의 coupled 공변 confound를 실제로 닫는다 —
   설계의 중심 아이디어는 옳다.** 죽는 것은 검증기(F3/F4)·귀속(F1)·문턱(F7)·구조 차이(F8).
6. green readout이 *원리상* 항등식이라는 강한 주장은 실패 — **"이 격자에서 발화 불가"**로 좁혀야 함.
7. `NON_BINDING` 도달 불가 주장도 실패 — **문턱이 뜻을 바꾼다**(F7)로 좁힘.
8. **PRIORITY 문서의 L−2 우선 논증은 깨지지 않았다** — 겨냥한 대상은 옳다.
9. 정본 인용 정확성 대부분(§2.1 C1/§1-21, §3.1 ctx4096 블로커, 게이트 #83) — §1-25만 예외.

## rev2가 해야 할 것

**GPU 0 선행(재작성 전 필수)**: `compute_decode_realized`를 job 893663 텔레메트리에 재실행해
sticky 하 §1-25 항등식 붕괴를 독립 재현 · **Stage 0 텔레메트리(이미 디스크)에 같은 함수를
돌려 §2.3 예측의 사후 검정을 완료** — GPU를 사기 전에 결론이 난다.

**S0′(부팅 4회, ≈0.1 GPU-hr, 판정 없음)**: ① `GRID_HONORED` d16·d44·d92 + **양성대조 = 홀수
셀(d17)에서 readout이 반드시 불일치** ② `RESIDENCY` — **`E1_DECODE_REALIZED`를 판정 지표로**,
문턱 ≥0.95, sticky ON/OFF 각 1부팅 ③ `NULL_SD` 동일 셀 4부팅, **같은 노드/다른 노드 층화**.

**본 캠페인**: 전 셀 `sm_group_num` **균질화**(F8 제거) · 부하 파라미터 등록 + realized decode
batch size 동반 공표 · **job-내 D 루프로 노드 페어링 복구**(가능하면 paired t-CI로 상향) ·
문턱은 귀무 `r=1.0`·실무 바닥 3%(C2는 **문턱이 아니라 사후 맥락 참조**) ·
`NON_BINDING`은 "상한 CI < 1.03" · 라벨 4개(`BINDING_CTX_INCREASING`/`BINDING_CTX_FLAT`/
`NON_BINDING`/`INCONCLUSIVE`) · 도구는 **실행 브랜치에 병합한 뒤** 등록하고 `level` 명시.

★**순서를 뒤집어라: `E1_DECODE_REALIZED`가 게이트이고 green readout이 보조다.**

## §9. 메인 세션 독립 재검증

| 항목 | 결과 |
|---|---|
| **F1** | `stage0_pdmux_capture.sbatch:173-179` 직접 확인 — `target_decode_sms` **히스토그램**이 맞다. D=108은 *"reference (R2 off) → no target_decode_sms expected"* 로 **pin-check 자체를 하지 않는다**. **확인** |
| **F5** | `stage0_client.py:13-19` keepalive 독스트링 확인 · `PARTITION_RESIDENCY_STAGE0.md` 잔류표 확인(D16 60–94%, D44 74–96%, D92 78–97%, **D108이 16SM 80–97%**). **확인 — §2.3은 방향이 반대였다** |
| **F4** | A1 원자료 `a1smoke_…result.txt:12`: `minSmPartitionSize: 2, smCoscheduledAlignment: 2`. 격자 전부 짝수. **확인** |
| **F8** | `pdmux_d16.yml:5 sm_group_num: 3` vs `pdmux_d92.yml:16 sm_group_num: 4`. **확인** |
| **B11** | `stage0_pdmux_capture.sbatch:103 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. **확인 — grep 한 번이었다** |
| 미재검증 | F2·F3·F6·F7 및 B1–B10·B12는 감사 반환문 그대로이며 메인 세션이 재실행하지 않았다 |
