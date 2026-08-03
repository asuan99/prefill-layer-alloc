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

---
---

# 세션 핸드오프 (속행 #2) — 2026-08-03 오후~야간

> 위 문서는 같은 날 **앞선 회차**(pin 게이트 항등식 발견까지)를 다룬다. 아래는 그 뒤
> **"872077을 블록 늘려 재실행할까"** 하나로 시작해 앵커 경로 폐기까지 간 속행분이다.

## 이번 세션 요약

시작 질문은 **"872077을 RULE_BOUNDS 하에서 block 8→12–16으로 재실행할까"** 하나였다.
답은 "하지 마라"로 나왔지만, **그 이유가 세 번 바뀌었다** — (1) 검정력 부족 → (2) 희석
attenuation 아티팩트 → (3) **estimand 미식별**(최종). GPU는 sticky smoke 9분과 취소된
D=54 run 17분만 썼고, 나머지는 전부 GPU 0 감사·재분석이다. **내 판단이 이번 회차에도
두 번 뒤집혔다**(희석 보정 가설, C2 앵커 경로). 둘 다 GPU를 크게 쓰기 **전에** 잡혔다.

---

## 결정·측정

### 1. 872077 재실행 판단 — 검정력 계산 (GPU 0)

`m3_analyze.py --job 872077` 재실행으로 per-block 확보:
Ha8 `[1.118, 1.214, 1.171, 1.168, 0.847, 1.158, 1.012, 0.86]`, mean **1.0685**, sd **0.1453**.
임계까지 gap 0.0815 ⇒ **gap < sd**.

| blocks | CI 상한 | 상한≤1.15 확률 | 추가 GPU |
|---|---|---|---|
| 12 | 1.161 | 43% | +3.0 h |
| 16 | 1.146 | **55%** | +3.0 h |
| 24 | 1.130 | 75% | +6.0 h |
| 32 | 1.121 | 87% | +9.1 h |
| 40 | 1.115 | 93% | +12.1 h |

⇒ 제안됐던 8→12–16은 **동전던지기**. (기판 수정 후 sd가 달라지므로 **이 표는 사전에 무효**.)

### 2. ★내 "희석 attenuation" 가설 — claims-auditor **REFUTED**

주장: 실현률 4–19%이므로 `A_free(dD)=w·A(D)+(1−w)·A(108)` 혼합이고 보정 시 g≈1.62–1.70.

반증 3줄(감사자 독립 재분석, 872077 telemetry 64 + bench 64):
1. **control-arm reductio** — 같은 식을 T8에 넣으면 corrected g **21–29×**(정본 C2
   2.36–2.91×를 더 좁은 구간에서 10배 위반). 모형은 T8 d16 split-조건부 ITL p95를
   352–360 ms로 요구하나 **실측 30.67 ms**.
2. **de-engagement 직접 실험** — engagement→0으로 보내도 `A_free`는 **1–11%만** 변한다
   (Ha8 d16 113.51→112.29, d54 107.45→95.74). **w=0에서 g = Ha8 1.173 / T8 1.796.**
3. **"A(108) 셀 무관" 가정 위반** — SPLIT-only에서 **Ha8 g = 0.920 [0.842, 0.998]**
   (CI가 1 배제, **부호 반대**). T8 헤드라인 1.837은 **UNSPLIT-only에서 1.795로 재현**
   (= decode SM 대비가 정의상 0인 모집단에서 효과 전부가 나온다).

유형 = confound #1(서빙 측정을 오프라인 모형으로 대체) + #6(`w`는 자유 모수가 아니라
**prefill duty cycle 그 자체**). **살아남은 것: engagement가 낮다는 전제 자체는 견고**
(세 계측기 교차확인 — snapshot 3.8–18.7%, event-driven 2.7–8%, 토큰 기준).

### 3. 872077의 진짜 사유 — **estimand 미식별** (결론 CONFIRMED)

`pdmux_context.py:initialize_stream_groups`가 `[(108,0)] + divisions + [(0,108)]`를
하드코딩하고 `multiplexing_mixin.py:773,792-794`가 prefill 부재 시 그 plain 그룹으로
되돌린다 ⇒ **"decode가 D SM에서 돌았다" ⟺ "prefill이 동시에 in-flight였다"가 같은 사건.**
어떤 통계도 decode-SM 레버와 prefill 간섭을 분리 못 한다. **n으로 해결 안 됨.**
⇒ `g`는 이 격자 한정 **은퇴**, 블록 증설 **선행 금지**.

부수: `A_free` 자체 결함(`PREFILL_BLOCK_TOK=1024`가 prefill 작업의 **74–77% 통과**,
꼬리 유발 요청 input 221–804 tok로 전부 임계 아래; 요청의 **27.5–29.5%가 outlen≤25**라
요청별 p95가 max로 퇴화) + arm 간 **decode batch 교락**(T8 4.5 vs Ha8 15.8)으로
"attributable to the arm" 문구 현재 사용 불가.

### 4. decode-empty 기전 분해 (result-analyst, `m3_decode_empty.py`, GPU 0) — **미감사**

★**내 진단 프레이밍이 틀렸다**: `E1_DECODE_REALIZED`는 decode-active 시간에 **조건부**라
decode 공백은 정의상 제외된다(실측 기여 T8 −0.0006±0.0046 / Ha8 −0.0001±0.0028 = 0).
**희석의 원인은 prefill 부재** — block-paired (d54−d16) 분해에서 gradient의 **100%가
prefill 점유율로 설명**(잔차 CI가 0 포함).

- 부하창 내 decode-empty는 **0.6–1.3%**뿐. 원자료 15–19%는 **측정창 아티팩트**
  (warmup→dataset 준비 갭 14.9–17.7 s + 종료 후 꼬리 5.2–6.2 s) ⇒ telemetry 분석은
  **클라이언트 `duration`으로 앵커한 부하창**에서 해야 한다(검증: 창−duration = 0.06–0.15 s).
- ★**realized 천장이 워크로드 성질이고 arm마다 3배 다르다**(ΣTTFT/decode-busy로
  T8 0.120–0.154 / Ha8 0.381–0.558) ⇒ **부하를 올려도 engagement는 안 오른다**
  (decode는 이미 ~99% busy). — 이 항목만 claims-auditor와 **독립 수렴**해 인용 가능.
- ★**죽은 telemetry 필드 4개**(이후 모든 분석이 재사용): `decode_ready_queue_depth`
  **항등 0**(`multiplexing_mixin.py:490-492` dual-worker 가드 안에서만 채워지는데 872077은
  legacy), `active_decode_sequences` **≡ `decode_running_batch_size`**,
  `decode_idle_ratio`/`prefill_idle_ratio` **항등 0.0**(`controller.py:55` 선언만),
  `running_batch_occupancy` ≡ min(1, drb/48), `prefill_admission_blocked` ≡ (pqd>0 ∧ pab==0).
  ⇒ 내가 프롬프트에 쓴 H-starve 예측 조건이 **정의상 발생 불가**였다.
- UNDETERMINED: gradient의 **크기**(스냅샷 dt p95 195–805 ms가 prefill span과 같은 자릿수
  ⇒ "prefill 에피소드 길이"는 **물리량 아님, 인용 금지**; t_pa 기반 +30.6 ms vs 클라이언트
  TTFT +7.5 ms **미해소 모순**). 해소법 = prefill batch start/end **이벤트 emit** 후 재측정.

### 5. 대체 추정량 이관 (`m3_conditional.py`, GPU 0)

`A_free` → **조건부 per-token 추정량**. 단위 = 개별 ITL 구간 1개(요청별 집계 제거),
`split_frac ≥0.90` SPLIT / `≤0.10` UNSPLIT / 사이 배제(AMBIG 실측 0.3–1.4%),
셀-블록당 SPLIT 1,390–7,045 토큰. **UNSPLIT control이 안전장치**(대비 정의상 0).
회피 근거: T8 d16 pooled per-token p95 **11.60** vs `A_free` **28.31**.

정렬 규율 확정: `phase=="benchmark"` 마커는 **warm-up 요청**에 발화하므로 probe 경계가
아니다 ⇒ **필터 금지**. `ALIGN_R_MIN=0.95`는 **flag-only**(LOO 실측에서 가장 약한 block을
빼면 T8 대비가 1.688→**1.822**로 오히려 커진다). ★감사자 자기 정정: ALIGN-WEAK는
1개가 아니라 **2개**(T8 d16 b1 r=0.882, b6 r=0.930) — 보고 수치는 8 block 전수라 불변.

★**임계 스윕[UNAUDITED]**: `PREFILL_BLOCK_TOK` 1024→0에 **무릎 없음**, 임계 0에서
**d16-vs-d54 대비 소멸**(Ha8 0.986, T8 1.005) ⇒ 임계는 자유 모수가 아니라 **답을 정하는
손잡이**. `A_free` 은퇴 확정.

### 6. `PDMUX_STICKY_PARTITION` 구현 + correctness gate (job **872800**, ~9분)

되돌림 경로 **5곳 전수** 처리(핵심 `adjust_stream_groups:904`), `event_loop_pdmux_coord`
조합은 init에서 `RuntimeError` 거부(반쪽 sticky 방지), decode-empty는 **의도적 index-0
release**(realized가 decode-active 가중이라 hold는 upside 0), OFF는 short-circuit으로
patch 전과 동등(**독립 재구현 pre-patch selector** 대비 3 config × decode_bs 6종 × prefill
2종 전 격자 테스트), **cudagraph 보존**(`capture()`가 모든 stream-group 인덱스 캡처).

correctness gate 전부 PASS: CPU 40 tests + sticky 12 tests + GPU smoke에서 OFF/ON greedy
출력 6개 **byte-identical**.

★**realized 관측(n=1, 성능 아님)**: `E1_DECODE_REALIZED` **OFF 0.0839 → ON 1.0000**
(사전등록 ≥0.90 초과, 튜닝 없음). 스냅샷 교차확인: ON은 decode-busy 139/139가
`(idx1, 92, 16)`, index 2 진입 0. ⇒ **이 게이트가 항등식이기를 멈추고 진짜 게이트가 됐다**
— 이 패치가 주장하는 전부. ⚠️ decode-active wall time이 arm 간 다름(72.8 vs 164.3 s):
**분모 미매칭, 성능 측정 아님.**

### 7. ★C2 → `G_LEVER` 앵커 도출 시도 → claims-auditor **경로 폐기 판정**

`c2_anchor.py`로 도출 시도. 감사 결과 **주장 1만 생존, 2·3·4·5 전부 반증/미지지**.

★**§0 결정적 발견(감사자 신규)** — 같은 arm·같은 서버 플래그(`s8_sweep.sbatch:141-146`
vs `e1_m3_control.sbatch:212-218` 직접 대조)·매칭 batch에서 **"decode 16 SM"의 per-token
ITL이 2.6× 다르다**:

| 출처 | 조건 | decode@16SM p50 |
|---|---|---|
| C2 865493 (SPLIT, batch bin 5) | prefill 16, decode 16 | **28.79 ms** |
| C2 `FINDINGS_8B` §2 batch=4 | prefill 16, decode 16 | 28.48 ms |
| **872077 (E1 격자)** d16, decode batch 4.5 | prefill 92, decode 16 | **11.09 ms** |

⇒ 둘 중 하나가 거짓: **(i)** 872077의 `decode_sms==16`이 실제 16-SM 실행이 아니다
(`DESIGN.md` §4.3.11이 **명시적으로 미검증으로 남긴 잔여 층** — green context 생성이
하드웨어 SM 부여를 보장하는지 재프로브 안 함), 또는 **(ii)** C2의 28–31 ms가 decode-SM
비용이 아니라 그 셀 배치(`[16,16,76-idle]` + 상시 동거 keepalive prefill)의 성질이다.
**어느 쪽이든 C2 비를 E1 격자로 이식 불가.** = confound #1의 실측 대 실측 재현.

기타 감사 판정:
- **주장 1(실현 검증)** 결론 CONFIRMED(비순환 재계산: 865493 측정창 @D **0.989–1.000**,
  @108 0.000–0.007; 865533은 측정창에서도 0.30–0.73 FAIL). **단 서술 2건 REFUTED** —
  (a) 108 시간은 warmup이 아니라 **창 밖 drain 전용**(warmup도 @D≈1.000), (b) ★정본의
  "활성률 0.66–0.93"은 `realized_pin_check.py`의 **스냅샷 개수 가중**이지 시간가중
  all-busy(0.803–0.958)가 아니다 ⇒ **방법론 게이트 #4 정면 위반**, 이 감사에서만 3회.
- ★**구조적 함의**: in-window residency와 UNSPLIT 표본은 **구성상 여집합** ⇒
  `E1_DECODE_REALIZED ≥ 0.90`을 통과하는 런에는 **음성대조가 정의상 존재할 수 없다**
  (865493 UNSPLIT n=0; sticky ON은 D108 0초). 사전등록에 명시 필요.
- **주장 2(primary p95→p50)** 관측 CONFIRMED / 기전 귀속 REFUTED / **처방 REFUTED**:
  ① 전환-근접 배제해도 SPLIT p95는 ≤2%만 이동(오염원이 estimand로 전이 안 됨)
  ② 오염 기전이 **파티션 전환 인접 구간**(사건의 83–87%가 전환 0.5 s 이내, `prefill_active>0`
  @108 = 0.0000이라 M4 stall 서명으로 설명 불가)이고 sticky ON에서 **소멸**
  ③ ★**872077 실측 `T8 sp_p50 = 0.996 [0.990,1.002]`** ⇒ p50으로 바꾸면 **양성대조조차
  1.00**이라 어떤 `G_LEVER>1`에서도 발화 불가 = 캠페인을 **구조적 NO VERDICT로 확정**.
  그리고 872077에서는 같은 p95 음성대조가 **반대 방향**으로 깨진다(T8 un_p95 0.880 [0.853,0.907]).
- **주장 3(편향 부호=하한)** NOT-YET-SUPPORTED: (ii) 가산/곱셈 **미식별**(Hs8은 비 일정,
  T8은 차도 비도 불일정; 두 job이 keepalive를 동시 변경 = confound #10), (iii) **증거가
  항을 0으로 만든다**(C2 분할 셀은 전부 **1410 MHz 고정**, 클럭이 떨어지는 건 **무분할 np뿐**
  — T8 median 1290 ⇒ **E1/sticky가 낼 throttling을 C2는 안 낸다**는 반대 방향 경고),
  (i) span 절단만 생존. 그리고 X의 하한이 Y의 임계를 정당화하지 못한다.
- **주장 4(`G_LEVER`=1.41)** REFUTED: "한 격자 스텝"은 사후 정당화(끝점 선택만으로
  [1.41, 2.40] 도달 가능), 2.02는 **게이트-FAIL job 4셀** 포함(내적 비일관), 그리고
  **1.41이 872077 T8 CI 하한 1.227을 가로지른다**(판정이 임계 소수점에서 갈리는 위치).
- **주장 5(`G_FLAT`=1.25)** 방법 PLAUSIBLE / 숫자 REFUTED: re-score 저촉 **아님**
  (§4.3.8이 이미 관측 sd로 blocks를 정한 선례). 그러나 LOO 8개 실측 시 실제 t95 반폭은
  **0.303 ⇒ 1.30**이고 **Ha8 점추정 1.340 > 1.30**이라 규칙이 자기 데이터에서 뒤집힌다.
  더 큰 문제: sticky가 `n_split`을 한 자릿수 이상 늘려 **사후 sd가 떨어지므로**
  사전-sticky sd 기반 `G_FLAT`은 **관대해지는 방향 = 귀무 오수용** 편향.
- **`n_indep` = 1**(셀당 부팅 1개, 865493↔865533은 keepalive 설정이 달라 replicate가
  아니라 **다른 조건**) ⇒ **세 번째 pseudo-replication**.
- ★**내 회부서의 regime 매칭 서술이 단위 오류로 뒤집힘**: 872077의 12.8은 **concurrency**,
  C2의 12.7은 **decode batch**. 같은 단위로 맞추면 T8이 **2.8× 어긋나고** Ha8이 잘 맞는다.

### 8. D=54 추가 측정 (jobs **872920/872921**) — 제출 17분 뒤 **취소**

설계는 지시대로 pseudo-replication을 피했다(d16+d54를 같은 캠페인, **4 block**, 셀 순서
block 패리티 교대, arm T8+Hs8). 그러나 두 가지로 무효:

1. **감사가 전제를 폐기**(위 §7).
2. ★**experiment-runner가 독립적으로 자기-무효화 버그 발견**: `s8_keepalive_prompt_224.txt`가
   **1793 토큰**인데 `CTXCAP = CTX+OUTTOK+256 = 1792` ⇒ 모든 keepalive가 `HTTP 400`.
   **865493은 byte-identical한 같은 파일로 `keepalive_errors=0`**이었고 두 srv.log 모두
   `CTXCAP=1792`를 찍는다 ⇒ **2026-07-27 이후 엔진 트리 churn으로 context-length 거부가
   엄격해졌다 = s8_scaleup 전체의 재현 불가 요인.** 자명한 수정은 `KEEPA_REPS≤223`(미적용).
   결과: co-residency가 ~90–100% → **31–34%** 붕괴, 측정된 전 셀 `REALIZED_PIN` FAIL
   (T8 blk1 d16 0.316 / d54 0.628, Hs8 d16 0.342 / d54 0.577).

★★**독립 수렴(가장 값어치 있는 산출)**: sticky OFF에서는 prefill in-flight일 때만 분할이
유지되므로 **keepalive 포화가 C2 물리의 하중 부재**였다 — 즉 **C2의 높은 residency는
decode 파티션 제어가 아니라 워크로드 장치의 산물**. 세 경로가 같은 결론에 독립 도달:
(A) 코드 읽기(`_init_sticky_partition` docstring, `multiplexing_mixin.py:206-231`),
(B) keepalive 사망 시 실측 붕괴, (C) 이번 run block-1 telemetry 교차표
(`prefill_active>0` @D = **0.975–0.996**, @108 = **0.000**, 4파일) — 감사자가 865493/865533에서
낸 0.943–0.996 / 0.0000과 같은 모양이나 **다른 대조로** 도달.
기록: `results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md`.
한계: 이 캠페인은 **np(무분할) 셀을 안 돌려** 감사자의 "np만 1290 MHz 하락"을 확증 못 함
(분할 셀은 1396–1410 MHz 평평).

---

## 코드·문서 변경

### 커밋됨 (4개, **push 없음** — origin/main 대비 4 ahead)

| 커밋 | 내용 |
|---|---|
| `f60a128` | `feat(engine)`: `PDMUX_STICKY_PARTITION`(+151/−17) + `tests/test_sticky_partition.py`(333줄) + `results/sticky_smoke/` |
| `3900720` | `test(s8_frontier)`: `m3_conditional.py` + `m3_decode_empty.py` |
| `fd518db` | `docs(status)`: 정본 5종(`PROJECT_STATUS.md`·`CONSENSUS.md` §1-26/§1-27·§3-15·`CLAIM_EVIDENCE_MATRIX.md`·`EXPERIMENT_ROADMAP.md`·`DESIGN.md` §4.3.9–4.3.12) + 이전 회차 핸드오프 |
| `bf6c74e` | `docs(handoff)`: `research_arc_2026-08-03.md`(561줄, DERIVED 서술 문서 — 정본 아님) |

⚠️ `stksmoke_..._result.txt`는 리포 전역 `.gitignore`(`**/results/**/*_result.txt`)에 걸려
제외됐다(리포 전체가 `*_result.txt` 0개 추적, 큐레이션된 `*_verdict_<date>.md`만 추적).
핵심 수치는 `f60a128` 커밋 메시지에 있다.

### 미커밋 (6개, 전부 미추적)

- `results/s8_frontier/c2_anchor.py` — C2 앵커 도출 분석기(**감사 결과 경로 폐기**, 스크립트
  자체는 §7 재현에 필요)
- `results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md` — keepalive 재현성 결함 + 독립 수렴 기록
- `results/s8_scaleup/{d54_block_ratio.py, pdmux_p16_d54.yml, s8_sweep_d54.sbatch,
  runtime_source_manifest_d54.sha256}` — D=54 하네스(취소됐으나 **설계는 유효**, 재사용 가능)

### 메모리

`MEMORY.md`가 183줄 → 10줄로 압축(내용은 토픽 파일로 이동, `slo-aware-scheduling-track.md`
660줄로 확장). `deconfound-measurement-lessons.md`에 항목 11("항등식에서 파생된 양을 자유
모수처럼 나누지 마라" + 재사용 가능한 검정 순서: **대조군 reductio → 직접 관측치 대조 →
핵심 가정 부분집합 분해") · 항목 12("**진단과 처방을 같은 턴에 하면 처방은 자기가 감사한
것이다**") 추가.

---

## 열린 항목 / 다음 세션 시작점

### ★최우선 — §0의 2.6× 모순을 어떻게 가를 것인가

`DESIGN.md` §4.3.11의 **미검증 잔여 층**("green context를 `(92,16)`으로 만들었다 →
하드웨어가 실제 16 SM을 부여했는가")이 이제 **872077 전체와 이번 세션 sticky 결과가 딛고
선 바닥**이 됐다. 세 갈래 중 (iii)이 참이면 **Stage 0급 정정**이다.

**미결 선택지 3개**(사용자 결정 대기):
1. **하드웨어 SM 부여 직접 검증부터**(engine-porter, GPU 소량) — 가장 근본·가장 쌈. **권고.**
2. **감사자 제안 S1 전체**(≈1 GPU-시간): T8 단일, d16+d54, **sticky ON**,
   `PDMUX_R2_POLICY=fixed`, 워크로드 2종을 **같은 부팅 계열**에서 — (a) C2 복제(폐루프
   conc16, 고정 1024/512, keepalive 8) vs (b) E1 복제(ShareGPT rate 2), 2 block.
   판정: C2 복제가 2.0–2.2 재현 + E1 복제 ~1.0 → **워크로드/batch 귀속** / 둘 다 ~1.0 →
   **C2 28–31 ms가 셀 배치 성질**(C2 스코프 문구 수정) / 둘 다 큼 → **872077 d16 라벨
   미실현**(Stage 0급). 필수 계측: 셀별 SM clock, realized decode batch, `prefill_active` 동거율.
   ⚠️ **하네스 확장 필요** — `s0dc_client.py`는 폐루프 합성 클라이언트, E1은 ShareGPT
   trace replay + 개루프 rate. 같은 서버(같은 포트)에 두 클라이언트를 순차 투입하도록
   `run_cell` 확장이 필요하고 E1 클라이언트 CLI 계약 미조사.
3. 1 → 2 순차.

### 임계 사전등록 — **UNDETERMINED 유지 (§4.3.12(d) 그대로)**

- **`G_LEVER`**: C2 앵커 경로 **폐기**. 감사자 대안 2개 — (α) sticky 파일럿의 T8 양성대조
  실측 분포 + **효과크기** 논증, (β) 절대 임계를 없애고 **arm 간 대비** `g_T8/g_Ha8`의
  block-paired CI가 1을 배제하는지(스케일 자유; 단 arm 교락 때문에 **batch-매칭 rate** 필요).
  ★**둘 다 감사자 발안이므로 독립 사전등록 필요** — 감사자가 자기 발안의 승인 주체가 될 수 없다.
- **`G_FLAT`**: 방법(사후 분산에서 최소 발화 간격)은 채택 가능, **숫자 폐기**. 사후-sticky
  파일럿에서 sd를 측정한 뒤 정하되, 순수 power 임계 대신 **TOST 동등성 마진**으로 바꾸고
  마진 근거를 "분해능"이 아니라 **실질 유의성**으로 논증할 것(감사자 발안 ⇒ 독립 사전등록).
- **primary는 `p95(SPLIT)` 유지**, p50은 secondary. 전환-근접 진단은 **게이트 아닌 진단**으로 병기.
- **음성대조가 sticky에서 정의 불가**임을 사전등록에 명시(§4.3.12(e)(i) 확장).

### 정본에 아직 안 들어간 것 (doc-steward 대기)

§7 감사 판정 전체(앵커 폐기·§0 모순·주장 1의 서술 정정 2건·음성대조 구조적 부재)와
§8(keepalive 재현성 결함·독립 수렴). **`G_LEVER`/`G_FLAT` UNDETERMINED 유지**도 명시 필요.
정본 정정 필수 항목: **"파티션 활성률 0.66–0.93"은 count-weighted**이고 측정창 시간가중은
0.99+, **108 시간은 warmup이 아니라 drain 전용**.

### 이월 (미해소, 이전 회차부터)

- `s8p_prefill` claims-auditor 미회부, `s8p_analyze.py` 버그 2건.
- decode가 **왜** 비는가(자연 도착 공백 vs 굶김) — 기존 telemetry로 ≈1 h. 단 §4에서
  H-starve 예측 필드가 **정의상 발생 불가**로 드러나 접근 재설계 필요(상한은 잡힘: ≤0.39%).
- realization gradient의 **크기** — prefill batch start/end 이벤트 emit 후 재측정 필요.
- rev2(872497) 데이터 보존, 목적 철회.

---

## 미완·주의

- **claims-auditor 통과분**: 주장 A REFUTED / estimand 미식별 / `A_free` 결함 / arm 교락 /
  C2 앵커 폐기 / 주장 1 결론. **미통과**: `m3_decode_empty.py` 산출 전체(단 "부하 올려도
  engagement 안 오름"만 독립 수렴으로 인용 가능), `c2_anchor.py`의 임계 스윕,
  `NOTES_D54_ANCHOR`의 keepalive 발견, **감사자 자신이 이번에 새로 생산한 표들**.
- **정책 비교 수치 없음.** 이번 회차 측정은 전부 진단·게이트·correctness다. goodput A/B 없음.
- **긴장 A(HE2 vs C2)는 전혀 닫히지 않았다.** "Ha8에 decode-SM 레버가 없다"도 **CONFIRMED
  아님** — 현재 데이터는 그 질문에 답하지 못한다.
- **실행 중 job 없음**(872800 COMPLETED, 872918/872919/872920/872921 CANCELLED).
- ★**메타 교훈 2건**:
  1. **진단과 처방을 같은 턴에 하면 처방은 자기가 감사한 것이다.** 이번 회차에 두 번
     실증됐다 — 감사자가 주장 A를 죽이고 대체 추정량도 냈고(임계 스윕이 [UNAUDITED]로 남음),
     result-analyst가 p95 음성대조 실패를 발견하고 primary 변경도 처방했다(**처방만 세 겹으로
     반증**). **진단자와 처방자를 분리하라.**
  2. **경합 가설이 공유하는 전제를 검정 대상에 넣는 규율이 두 번 작동했다.** 회부서에
     명시한 덕에 주장 A가 GPU 쓰기 전에 죽었고, 감사자가 "p95 오염 vs 정상"의 공유 전제
     ("SPLIT 집단이 decode-SM 대비를 담고 있다")를 찾아 그것 자체를 반증했다.
