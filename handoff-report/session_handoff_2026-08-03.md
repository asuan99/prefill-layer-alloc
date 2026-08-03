# 세션 핸드오프 — 2026-08-02 ~ 2026-08-03

## 이번 세션 요약

2026-08-01 결과 4건을 claims-auditor에 회부하는 것으로 시작해, **감사 → 후속 측정 →
재감사**를 두 바퀴 돌았다. 1차 감사(2 REFUTED / 2 NOT-YET-SUPPORTED / 1 CONFIRMED)로
E1 본 스윕도 종결도 시기상조임이 확인돼 M1/M2/M4/M5를 GPU 0으로 처리하고 **M3
Transformer-control 대조**를 사전등록·제출했다(job 872077). M3는 깨끗이 돌았으나
pin 게이트가 19/64를 void시켰고, 그 원인을 쫓아 M3R rev1(872236)·rev2(872497)를
추가로 돌린 끝에 **2차 감사에서 pin 게이트 자체가 항등식임이 밝혀졌다**. GPU job 3건
제출(872077 / 872236 / 872497), 전부 COMPLETED. **이번 세션에 내 판단이 네 번
뒤집혔고**, 마지막 것은 자가발견이 아니라 외부 감사로 잡혔다.

---

## 결정·측정

### 1. 1차 claims-auditor 회부 (2026-08-01 결과 4건) — 미통과

| 주장 | 판정 |
|---|---|
| (i) "천장 = 설정 성질" 일반화 | NOT-YET-SUPPORTED |
| (ii) cap 96 ≈ 192 | **REFUTED** |
| (iii) mamba pool 항등식 | **CONFIRMED**(단 "1.0=포화"는 max 통계) |
| (iv) HEADLINE NONE = cap 아티팩트 | **REFUTED** |
| (v) Hs8 이상치가 knee 오염 | **REFUTED**(대상 오귀속) |

⇒ (A) E1 제출도 (B) "설계상 도달 불가" 종결도 시기상조. 종결 논거 3다리 중 둘이 무너짐:
(b) cap 왜곡은 rate 16 = 운영대역 5.7배 밖(운영대역 동시성 12–44 < cap 48이라 구속 불가),
(c) "제외규칙이 C2 레버를 지운다"는 C2 자신의 데이터로 기각(SM16→44가 log-range의 75–81%).

### 2. M1 — knee를 코드로, 룽 표를 공통 rate로 (GPU 0)

- `e1_analyze.py --knee-scan` 신설. 역설계한 정의가 **발표된 20셀을 정확히 재현** ⇒ 고정.
- **"네 arm 공통 knee 2.80" 철회** — knee의 치역이 probe 격자(~10점)뿐이라 일치가 부분
  강제. **살아남는 건 순서**(d92가 먼저: T8/M8/Hs8 9/9, Ha8 8/9).
- FRAGILE knee 3건. Hs8 d16의 취약성은 실재하나 **원인 귀속이 틀렸다**(그 이상치를 지워도 불변).
- `--common-rate` 신설. **arm×룽 표가 rate-confound**였음 확인(T8만 rate 12, 나머지 2 —
  §4.3.5(b) 명시 위반). 공통 rate 2 재계산 시 **T8 행만 바뀜**.

### 3. M2 — ITL estimand 진단 (GPU 0, 게이트 #8 준수)

오염은 실재하나 **국소적**(d92와 M8 seed1 = 이미 제외되는 셀). 주장된 기전 2개는 불성립.
⚠️ **내 첫 stall 검정이 38/40에서 발동한 건 임계값 아티팩트**(scale 없는 절대 임계) → 17/40.
**`HEADLINE-ELIGIBLE = NONE`은 공통rate·두 estimand·두 seed 전부에서 생존.**

### 4. M4 — stall 원인 종결 (GPU 0)

절대 방출시각 재구성: **17개 stall probe 중 16개**에서 최장 프롬프트의 monolithic prefill이
stall 전 구간을 덮고, 크기가 **D에 단조**(prefill SM = 108−D). **고칠 수 없음** —
`server_args.py:6130`이 pdmux일 때 `chunked_prefill_size == -1`을 하드 assert ⇒
venue positioning의 **(A) green-context 종속** 버킷.
⚠️ 핸드오프(2026-08-02)의 기전 추측이 **옳았고** 감사자 REFUTED가 **틀린 검정**을 썼다.

### 5. M5 — 폐기 (실험 → assertion)

운영점 cap TOST는 **공허**(구속 불가한 곳의 동등성 검정). cap 96안도 자가 기각 —
Ha8 SSM pool 6.78→13.5GB로 배가돼 이미 52%만 잡힌 attention KV pool 잠식.
`--max-mamba-cache-size`는 **공통 절대상수가 아니라 `= cap` 규칙**.

### 6. M3 (job 872077) — 실행은 깨끗, 판정은 NO VERDICT

64/64 probe, 에러 0, in-run 게이트 3종 전부 PASS, `n_keep=194`, 3h01m.
설계: T8(양성대조)+Ha8 × d16/d24/d44/d54 × 공통 rate 2 × **8 block × 1 seed**, `NP_MIN=200`.

★**설계 중 자가발견**: `g`가 두 부팅을 가로지르는 비율이라 **seed가 아니라 block이 반복
단위**다. 초안(4 seed in 2 boot-pair)은 n_indep=2(허용 sd 0.021 = 발화 불가)로,
batch-cap pseudo-replication을 그 수정본 안에서 재생산한 것이었다. 또 percentile
bootstrap이 n=4서 커버리지 79.8% → **t-구간으로 교체**.

### 7. ★★ 2차 claims-auditor 회부 — pin 게이트가 항등식이었다

**독립 재현 완료**(telemetry 120파일, prefill-active 스냅샷 **77,688개**):

```
prefill_sms != target  <=>  decode_running_batch_size == 0
  off-target & decode-empty : 50,328
  on-target  & decode-busy  : 27,360
  위반 (양방향)             :      0
```

`multiplexing_mixin.py:773,792-794`가 decode batch가 비면 **설계대로** 무분할로
떨어뜨린다(§1-22 "의도된 경로"). ⇒ 시간가중 pin 게이트는 파티션 제어가 아니라
**"prefill in-flight 중 decode가 안 비어 있던 시간 몫"**을 잰다 = **방법론 게이트 #6 위반**.

따름정리: **조건부 pin = 27,360/27,360 = 1.000 정확.** 872077 재채점 **64/64 PASS**,
`n_indep=8` 회복.

**폐기**: M3의 19/64 void, rev1·rev2의 존재 이유, `MIN_PA_SNAPSHOTS`(추정 대상의
여집합 — 핀이 잘 걸릴수록 미측정 판정. UNMEASURABLE 20/20이 pin 최상위 두 seed에 집중).
DESIGN.md의 "M3 표본 2–34"도 오기(rev1 수치, M3는 4–130).

### 8. ★★ 정작 게이트가 없던 축 — decode 실현 배분 (872077 전수, n=8/셀)

| arm | d16 | d24 | d44 | d54 |
|---|---|---|---|---|
| T8 | **0.038** | 0.047 | 0.082 | 0.093 |
| Ha8 | 0.104 | 0.110 | 0.148 | 0.187 |

decode 작업시간의 **4–19%만 라벨대로 실현**, 81–96%는 무분할 108 SM.
⇒ (i) E1 격자는 **C2와 다른 물리량**(C2는 prefill 16 고정·decode 연속 D) — 긴장 A를
이 격자의 비로 닫을 수 없다. (ii) 희석 계수가 **셀마다 다르다** ⇒
`g = A_free(d16)/A_free(d54)`는 SM 수준과 engagement 비율을 동시에 움직인다 =
**추정량 내부 교락**. Stage 0 D108의 decode 축 판본이며 게이트가 아예 없었다.

### 9. 사전등록 불일치 해소 (forward-only)

**RULE_BOUNDS 채택** — 각 arm을 자기 임계를 마주보는 CI 경계로 판정(control 하한 ≥ 1.5,
test 상한 ≤ 1.15). test 쪽은 **귀무가설 수용**이라 점추정만으로는 증거가 못 된다.
★ **872077에서 발화하지 않는 쪽**이라 정직하게 채택 가능, **이후 job에만 구속**.

**872077 최종**: 전 게이트 통과, `n_indep=8`, **T8 g=1.837 [1.666, 2.009]**,
**Ha8 g=1.068 [0.947, 1.190]**, **NO VERDICT — 사유는 정확히 하나**(RULE_POINT는
발화, RULE_BOUNDS는 미발화 ⇒ 규칙이 정작 중요한 지점에서 모호).

---

## 코드·문서 변경 (전부 커밋됨, **push 없음**, 작업 트리 clean)

| 커밋 | 내용 |
|---|---|
| `93a83c1` | 직전 세션이 남긴 정본 갱신 캐리오버(자기 기록과 달리 미커밋 상태였음) |
| `d004cfb` | M1(knee 코드화·`--common-rate`)·M2(`estimand_check.py`)·M4 종결·M5 폐기·M3 사전등록+하네스 |
| `175ce75` | 2026-08-01 기록 중 감사가 뒤집은 것에 supersession 박스 |
| `6cd7053` | CONSENSUS §1-24, CLAIM_EVIDENCE_MATRIX·EXPERIMENT_ROADMAP 정정 |
| `7c6e6f2` | pin 검증을 GPU 없이(→ **2차 감사에서 논거 기각**, `a865c99`가 정정) |
| `903194b` | pin 게이트를 분석기에 배선 + realizability 스윕 |
| `a865c99` | pin 게이트 표본 2–28 발견, trace-force 복원, rev2 하네스 |
| `f18eea9` | UNMEASURABLE을 PASS로 세던 것 수정 |
| `8e77ed0` | **pin 게이트 = 항등식**, 조건부/decode 게이트 신설, 872077 재채점, 정본 3종 반영 |

**신규 파일**: `estimand_check.py`, `m3_analyze.py`, `m3_pin_check.sh`,
`m3r_realizability.sbatch`, `m3r_analyze.sh`, `m3r2_pinforce.sbatch`,
`m3r2_analyze.sh`, `e1_m3_control.sbatch`, `FINDINGS_M1_M2_2026-08-02.md`,
`M1_M2_output_2026-08-02.txt`, `m3_pin_872077.txt`.
**정본**: `DESIGN.md` §4.3.7·§4.3.8(a)–(h), `CONSENSUS.md` §1-24·§1-25,
`PROJECT_STATUS.md` 방법론 게이트 **#7**, `CLAIM_EVIDENCE_MATRIX.md`,
`EXPERIMENT_ROADMAP.md`, `MEMORY.md` + 메모리 항목 9·10.

⚠️ `m3_pin_872077.txt`는 앞서 "결과 아티팩트라 미커밋"이라 했다가 **판단을 바꿔 커밋**했다
(분석기가 읽는 cite-blocking 게이트 기록이라 재현성에 필요).

---

## 열린 항목 / 다음 세션 시작점

1. **★최우선 — 872077을 RULE_BOUNDS 하에서 재실행할지 결정.** 열린 양은 **Ha8 상한이
   1.15 아래로 내려오는가**(현재 1.190). 필요한 건 block 수 증가(8 → 12–16)뿐이며 설계
   변경은 없다. **재실행 시 decode 실현표(§8)를 `g` 옆에 반드시 병기**할 것.
2. **`g`의 해석이 미완**이다 — decode 라벨이 4–19%만 실현되므로 `g`는 "decode SM 탄력도"가
   아니라 "동시성 구간에서의 효과 × engagement 비율"이다. **긴장 A(HE2 vs C2)를 이
   추정량으로 닫으려면 희석 보정이 선행**돼야 한다.
3. **미측정 — decode가 왜 비는가**(자연 도착 공백 vs decode 굶김). 감사자도 반증 실패로
   인정. d16에서 decode-empty가 큰 것은 두 기전 모두와 정합적. 기존 telemetry 필드로
   대부분 가능(waiting_queue 깊이 vs 완주 분해), ≈1 h.
4. **이월(미해소)**: `s8p_prefill`을 claims-auditor에, `s8p_analyze.py` 버그 2건.
5. **rev2(872497) 데이터는 보존하되 목적은 철회**됐다. 재사용하려면 새 질문이 필요하다.

---

## 미완·주의

- **오늘 신규 결론 중 claims-auditor를 통과한 것**: pin 게이트 항등식(2차 감사 + 내
  독립 재현), decode 실현 4–19%(동일). **나머지는 미감사** — 특히 §2·§3의 M1/M2 산출물,
  M4 기전, RULE_BOUNDS 채택 근거.
- **E1 본 스윕은 여전히 미제출.** 선행 게이트 M3가 NO VERDICT이므로 진행 불가.
- **정책 비교 수치 없음.** 이번 세션 측정은 전부 진단·게이트이고 goodput A/B는 없다.
  `g`는 정책 비교가 아니라 "ITL 축이 D에 반응하는가"의 선행 질문이다.
- **실행 중 job 없음**(872077/872236/872497 전부 COMPLETED).
- ★**메타 교훈(재사용 가치 최상, 메모리 항목 10)**: 이번 세션에 판단이 **네 번** 뒤집혔다
  — cap 96 / trace-force 불필요 / UNMEASURABLE 집계(3연속 재도입) / pin void 해석.
  마지막은 **틀린 두 가설 사이에서만 논쟁**한 유형이다: "표본 잡음" vs "진짜 언핀" 둘 다
  *"게이트가 재는 양이 옳다"*를 공유해서, 아무리 정교히 비교해도 답에 닿을 수 없었다.
  **회부서에 내 판단과 그 반대만 담으면 감사자도 같은 전제에 갇힌다.** 경합 가설이
  공유하는 전제를 명시하고 그것 자체를 검정 대상에 넣을 것.
- ★**방법론 게이트 #7 신설**: 게이트를 만들 때 그 게이트가 재는 양이 통과 조건과
  **논리적으로 독립인지 먼저 증명하라**. 정확성을 강제하려 만든 게이트가 #6을 위반했고,
  표본 부족을 막으려 만든 게이트가 **추정 대상의 여집합**을 셌다.
