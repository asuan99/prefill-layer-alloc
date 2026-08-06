# S2 (job 873015) 독립 재현 — claims-auditor CONFIRMED (scoped), 2026-08-05

> **정본 위계.** 이 문서는 `results/s2_sticky/`(job 873015) 원자료의 **claims-auditor
> 감사 통과 판정 기록**이다. 상위 정본은 `PROJECT_STATUS.md`(8B decode-SM 프론티어
> "2026-08-05" 소절) → `reports/CONSENSUS.md`(§1-31) → 이 문서 순. 이 문서와 상위
> 정본이 충돌하면 상위 정본이 이긴다(충돌 발생 시 doc-steward에게 플래그).
>
> **새 성능 판정 금지.** 아래는 §0(`DESIGN.md` §4.3.13, `CONSENSUS.md` §1-28~§1-30,
> `PROJECT_STATUS.md` "8B decode-SM 프론티어" §0)의 **behavioural closure** 기록이지
> throughput/latency/goodput/`g`/`G_LEVER`/`G_FLAT`/decode-SM elasticity 판정이
> **아니다**. 이 문서에 그런 판정은 0건이다.

## 0. 배경

`CONSENSUS.md` §1-28~§1-30, `PROJECT_STATUS.md` "8B decode-SM 프론티어" §0,
`results/s8_frontier/DESIGN.md` §4.3.13–§4.3.15가 §0을 열린 항목으로 남겼다: C2
(job 865493)와 872077(E1 격자)의 "decode 16 SM" per-token ITL이 **2.6× 다르다**
(28.79ms vs 11.09ms). 2026-08-03(4차 속행)에서 이분법 (i)/(ii)가 세 번째 후보
(iii)("`split_frac≥0.90` 라벨이 순수하지도 완전하지도 않다")로 확장됐으나 오프라인
분리는 불가했고, S2(sticky-partition GPU 런)가 인과 시험으로 지정됐다
(`PREREG_S2_STICKY_ITL_2026-08-03.md`). S2는 job **873015**로 제출·완료됐고,
result-analyst의 초기 분석([`S2_ANALYSIS_2026-08-04.md`](S2_ANALYSIS_2026-08-04.md))
을 claims-auditor가 **2026-08-05에 독립 재현·감사**해 **CONFIRMED (scoped)** 판정과
명시적 정본 기록 허가를 냈다. 이 문서는 그 감사 결과를 정본 위계로 전파하기 위한
1차 산출물이다.

**원자료**: `results/s2_sticky/`(job 873015, sticky ON, T8 d16/d54, ShareGPT rate 2,
n=8 블록, cudagraph ON, gpu37; server args·매니페스트는 §6 참조), 비교대상
`results/s8_frontier/`(872077, sticky OFF) · `results/s8_scaleup/`(865493, C2).
**분석 스크립트**: result-analyst 산출 [`s2_analyze.py`](s2_analyze.py) +
[`S2_ANALYSIS_2026-08-04.txt`](S2_ANALYSIS_2026-08-04.txt)(raw). claims-auditor의
독립 재현 스크립트(`s2_replicate.py`, `s2_report.py`, `cadence.py` 등)는 **현재
에이전트 scratchpad에만 존재**(세션 격리 디렉터리, 이 리포지토리 밖) —
`s2_report_out.txt`가 그 raw 출력이며 아래 수치는 이 출력과 `S2_ANALYSIS_2026-08-04.md`
양쪽에서 교차 확인됐다. **스크립트 자체를 `results/s2_sticky/`로 이관하는 작업은
별도로 남겨둔다**(이 커밋의 범위 밖) — 재현 필요 시 result-analyst/claims-auditor에게
재생성을 요청할 것.

---

## 1. 주 결과 — pooled per-token ITL p50

**S2 (job 873015, sticky ON, T8 d16/d54, ShareGPT rate 2, n=8 블록, cudagraph ON,
gpu37)**: pooled per-token ITL p50 = **28.92 ms**(d16; per-block 28.91±0.13, t95
[28.81, 29.02]) / **12.04 ms**(d54; 12.05±0.09). d16 값은 사전등록 `[28,34] ms`
안이며, **논쟁 대상인 `split_frac` 라벨을 전혀 쓰지 않고** 재현된다 — 317,342개
클라이언트 ITL 구간의 raw median이다. **`PREREG_S2_STICKY_ITL_2026-08-03.md` §4
row 1 발화.**

## 2. 선택기 실현 검증 — 하드웨어 SM 부여 검증 아님

선택기는 decode-active 시간의 사실상 전부에서 목표 division을 유지했다:
`E1_DECODE_REALIZED`(시간가중, decode-busy) = **0.9990±0.0017**(d16) /
**0.9994±0.0008**(d54), 16/16 cell-block ≥0.995, 동일 추정량이 pre-patch job
872077에서는 0.0380/0.0925. ON에서 decode-busy ∧ 108 SM 스냅샷 **0건**, d54
guard row (64,44) **미선택**.

★**이는 선택기 인덱스를 인증할 뿐 하드웨어 SM 부여를 인증하지 않는다.**
`decode_sms`는 `arbiter.sm_counts[stream_index]`의 재진술이다(`dual_worker.py:
608-623`). sticky ON에서 `stream_idx = _sticky_fixed_idx`는 코드 불변식이다
(`multiplexing_mixin.py:882-891`). "`create_greenctx_stream_by_value(92,16)`가
하드웨어에 실제로 16 SM을 부여하는가"를 직접 프로브하는 S3-class 계측은 **여전히
미실행**이다.

## 3. §0 이분법의 종결 — behavioural, scope 한정

**§1-28 §0의 이분법은 거짓으로 닫힌다(behavioural).**

- **(i)**("872077의 `decode_sms==16`이 실제 16-SM 하드웨어 실행이 아니다")는
  **하드웨어 형태로 REFUTED, 라벨 형태로 CONFIRMED** — 872077 d16이 실제로는
  decode-busy 시간의 96.2%를 D108(무분할)에서 보냈고 D16은 3.8%뿐이었다(§4
  참조). "hardware 16-SM이 실제로 실행됐는가"라는 강한 하드웨어 형태로는
  검증되지 않았으나(§2의 잔여 스코프), "872077의 셀 라벨이 실제 실현
  파티션을 대표하지 않는다"는 라벨 형태로는 확인된다.
- **(ii)**("C2의 28–31ms가 decode-SM 비용이 아니라 셀 배치 성질이다")는
  **DISFAVOURED** — 28–31ms 수준이 keepalive 없는 open-loop ShareGPT에서,
  decode batch가 C2보다 2.6배 작은 상태에서, C2의 0.93×(5.4% 빠름)로
  재현된다(§5). C2의 특이한 워크로드 장치(closed-loop 클라이언트, 상시
  keepalive prefill 동거, decode batch ≈12.7)가 없어도 같은 수준이 나온다는
  것은, 그 수준이 배치 성질이라기보다 partition 자체의 성질이라는 쪽을
  지지한다.
- 살아남는 답은 **(iii)** — `split_frac≥0.90`은 D-파티션 실행 토큰 클래스를
  격리하지도 완결하지도 못한다(872077에서는). sticky ON 하에서는 이 라벨
  문제 자체가 무의미해진다(SPLIT 인구가 사실상 전체 인구와 같아짐, §7
  참조 — `p50(SPLIT)`와 `p50(all)`이 16 cell-block 전부 |diff| ≤0.001%).

**Scope: behavioural·selector-level.** 부여 SM 수의 직접 프로브는 없다(S3 미실행).

## 4. (iii)의 기전 — 계측에서 독립 도출

`runtime_snapshot`은 개수-서브샘플링(`PDMUX_DUAL_WORKER_TRACE_EVERY=32`)이고
양 캠페인 모두 `PDMUX_TRACE_FORCE_PREFILL=0`이다. **decode-busy 조건부** 스냅샷
케이던스는 네 arm 전부 **정확히 16 decode step**(872077 d16 0.177s / d54 0.176s;
873015 d16 0.465s / d54 0.193s). 따라서 ITL 구간 하나는 스냅샷 **1/16개**를
걸치며, 872077 d16의 SPLIT 모집단(11,124 토큰) 전체가 8블록 합 **~120개
스냅샷**에서 번져 나온다 — **기대 순도 ≈6%**로, S0-R의 mode 분해가 다른
경로로 얻은 6.6–9%와 일치한다.

이 기전 도출은 §10의 정정(`S2_ANALYSIS_2026-08-04.md` "Instrument check" 문단
정정)과 정확히 짝을 이룬다 — 그 문단의 오류(decode-idle cadence를 decode-busy
cadence로 오인)를 바로잡으면 이 기전이 정확히 도출된다.

## 5. 배치/노드/바이너리 아티팩트 배제

**11.09 → 28.92 ms는 배치 아티팩트도 노드/바이너리 아티팩트도 아니다.**

- decode-batch 기여는 세 독립 추정에서 **2.3–4.7%**: bin matching 4.7%
  (873015 d16 pooled 2.635 vs batch-matched 2.558), within-run slope 2.7%
  (OFF 자신의 ITL-vs-batch 기울기 +0.0812ms/batch로 예측한 +0.49ms/17.94ms),
  블록간 회귀 2.3–3.2%.
- 전역 효과(노드 gpu36→gpu37, 1파일 바이너리 차 — `multiplexing_mixin.py`만
  매니페스트가 다름, 캠페인 날짜)는 **d54 companion이 ≤1.097×로 상한**을
  준다(같은 before/after에서 10.97→12.04).
- 독립적으로, ~29–34ms 수준은 **872077 자신 안에 gpu36·pre-patch 바이너리로
  존재**한다 — 그 job의 클라이언트측 slow mode가 셀을 따라간다(d16 34.0 /
  d24 22.0 / d44 15.0 / d54 14.2 ms, S0-R mode decomposition).

## 6. 워크로드 페어링

`input_lens`와 `output_lens`가 4개 job×cell × 8 블록 전부 sha256 동일; server
args 19키 일치(**양쪽 cudagraph ON**, `disable_cuda_graph=False`), `random_seed`
만 상이. 매니페스트는 `multiplexing_mixin.py` 1개 파일만 다르다(872077 =
pre-sticky-patch 바이너리, 873015 = sticky-patch 적용 바이너리) — 나머지 10개
런타임 파일은 해시 동일. **이는 872077이 "같은 빌드의 OFF arm"이 아니라
"pre-patch 바이너리"임을 뜻하며**, §4.3.11이 주장하는 "OFF는 short-circuit으로
byte-identical" 논거와 별개로 872077 자체는 그 patch 이전 빌드다(양성 관측이나,
"한 빌드 안의 A/B"로 서술하지 않는다).

## 7. 결과(outcome) 게이트 0개 — 방법론 게이트 #9의 네 번째 재발

**이 런에는 결과(outcome) 게이트가 0개다.** `AMBIG_FRAC`·`MIN_N_SPLIT`·
`PREREG_S2` §3.1 일치검사(`p50(SPLIT)`≈`p50(all)`)는 sticky ON 하에서
**항등식**이다 — `E1_DECODE_REALIZED ≥ 0.90`을 통과하는 순간 SPLIT과 UNSPLIT은
정의상 거의 전체 인구를 SPLIT으로 만들고(d16 100.000% SPLIT, d54 99.969%),
그 조건 아래서 `p50(SPLIT)=p50(all)`은 실패할 수 없다. `E1_DECODE_REALIZED`는
arm 간(OFF vs ON)에는 비항등식이나 **ON arm 안에서는 코드 불변식**이다
(`stream_idx = _sticky_fixed_idx`). `ALIGN_R`은 계측 flag(정확도 지시자)일 뿐
결과 게이트가 아니다.

⇒ **사전등록 게이트 중 어느 것도 "28.92가 나올지 11이 나올지"를 사전에
제약하지 않았다.** 이는 관측치 자체의 타당성을 훼손하지 않지만(§1의 값은
직접 측정치), "사전등록 게이트가 결과를 걸러냈다"는 서술은 성립하지 않는다.

**방법론 게이트 #9의 네 번째 재발**(`PROJECT_STATUS.md` "방법론 게이트" #9 —
"자기가 검증하려는 코드를 복사한 게이트는 항등식에 가깝다"의 계열; 세 번째
재발은 `S2_ANALYSIS_2026-08-04.md` §3이 이미 §3.1 일치검사 자체를 항등식으로
기록했다 — "third recurrence of methodology gate #9 in this campaign line").
이번(네 번째)은 §3.1 하나가 아니라 **런 전체의 결과 게이트가 구조적으로
0개**라는, 앞선 재발보다 넓은 형태다. `PROJECT_STATUS.md`·`CONSENSUS.md` §3에
등재.

## 8. d54 companion — 사전등록 구간 [13,16] 미달

**d54 companion은 사전등록 구간 [13,16]을 빗나갔다**(관측 12.03, per-block CI
[11.96,12.12] — 전체가 13 미만). **`PREREG_S2_STICKY_ITL_2026-08-03.md`에는
"primary 적중 + companion 미스"에 대한 규칙이 없다**(§4의 3-way readout이 d16
단독 함수이고, §5가 companion을 아예 언급하지 않는다). 사후 분석(§8 forensics,
`S2_ANALYSIS_2026-08-04.md` §8)은 이 미스를 [13,16] 구간의 **외삽 오류**로
귀속한다(anchor가 C2의 d44=14.68을 d54 대용으로 쓴 cross-cell 외삽이었고,
C2 자신의 3점 곡선으로 d54를 직접 외삽하면 12.79ms — 이미 구간 하한 13
미만) — sticky 교란은 배제(부호 반대: sticky는 d54를 위로 움직임), 잔차
~5% 신호는 오프라인으로 판정 불가. **이 사실은 §0 관련 어떤 인용에도 동반해야
한다** — "primary가 [28,34]에 적중했다"만 인용하고 companion 미스를 생략하지
않는다.

## 9. §4.3.12(f) 판별 예측 — 설계상 미판정, 모형 반증 아님

`DESIGN.md` §4.3.12(f)의 헤드라인 판별 예측(T8≈1.85, 프리필 주도 가설이 맞다면)은
빗나갔다 — 관측 block-paired ratio `sp_p50` **r = 2.402 ± 0.008, t95 [2.395,
2.408]**(예측 범위 밖). 그러나 **판별을 담당하는 arm(Ha8)은 이 격자에 제출되지
않았다** — T8만 실행됐고 Ha8(예측 ~0.92 또는 ~1.6으로 갈리는 arm)이 없어 §4.3.9의
"CI가 겹치지 않아 discriminate 가능"이라는 판별력 자체가 이 제출분에서
성립하지 않는다. ⇒ **§4.3.12(f)는 설계상 미판정**이며, 그 앵커(1.85/0.92/1.6)는
**은퇴한 `g`/`A_free` 통화**로 쓰였다(방법론 게이트 #7·§1-24/§1-26이 이미 기록한
attenuation 계열 문제 — decode-D 대비 prefill(108−D) 상보성이 T8 단독으로는
"prefill 주도" 대 "decode-SM 레버 유효" 두 모형을 분리하지 못한다). **모형
반증으로 읽을 수 없다** — 인용 시 이 문구를 동반한다.

## 10. `S2_ANALYSIS_2026-08-04.md`의 계측 오류 정정

`S2_ANALYSIS_2026-08-04.md:61-64`의 "Instrument check" 문단은 **사실 오류**다 —
"텔레메트리 케이던스 ~2.0ms"는 **decode-idle** 값이고 decode-busy 조건부로는
0.177–0.465s다(§4 참조). 따라서 "11ms 구간이 5.6 스냅샷을 걸친다"는 실제로는
**0.06개**이며 **90–260× 틀렸고 방향 주장(ON이 더 잘 resolve된다)도 반대**다.
해당 문서 해당 문단에 **정정 표시를 달았다**(원문 보존, 삭제하지 않음 —
`S2_ANALYSIS_2026-08-04.md:65-83` 참조). **정정 전 그 문단 인용 금지.**

## 11. §0 최상위 열린 항목 해제

`PROJECT_STATUS.md`·`CONSENSUS.md`·`MEMORY.md`의 §0("872077 vs C2 decode-16SM
ITL 2.6× 불일치")은 **"미해소 3지선다"에서 "CONFIRMED (scoped)로 종결, 단
하드웨어 층 미프로브"로 전환**한다. C2 자체(레버 존재, 2.36–2.91×, scoped)의
등급·수치는 이 종결로 **영향받지 않는다**(자기완결적 4-arm matched-batch
캠페인) — 바뀐 것은 "C2와 E1/sticky 격자가 같은 물리량을 재는가"라는 상위 질문
뿐이다: 이제 **같은 축이라는 쪽으로 behavioural하게 confirmed**됐고, C2를
sticky 격자로 이식·앵커하는 것은 (여전히 `G_LEVER`/`G_FLAT` 미결이라는 별도
이유로) 계속 금지되지만 그 금지의 근거였던 §0의 축-불일치 의심 자체는
해소됐다.

**단, 아래 "여전히 막힌 것"(§12)을 반드시 함께 인용한다.**

## 12. E1은 열리지 않는다 — 네 가지 독립 사유

§0의 종결이 E1(예산 제약 하 net-positive 판정 게이트, `[108−D,D]` 프론티어)을
다시 열지 않는다. 이유 네 가지, 전부 독립:

1. **`G_LEVER`/`G_FLAT` 여전히 UNDETERMINED**(`DESIGN.md` §4.3.12(d)). post-sticky
   블록 sd가 0.022로 붕괴(pre-sticky t95 half-width 0.303 대비 ~14×) ⇒
   pre-sticky 산포로 교정한 임계는 **null 채택 편향**을 만든다. 임계
   사전등록은 여전히 미결.
2. ★**sticky 기판이 질문을 바꾼다.** decode-busy면 (92,16)이 유지된다 =
   **prefill SM 92개가 벽시계 시간의 ~77% 유휴 상태**(§1 관측 값 기준,
   decode busy·prefill idle이 76.31%(OFF)/76.76%(ON) 상태에서 prefill이
   놀고 있음). E1의 프론티어 질문은 예산 제약 `[108−D, D]` 하의
   **co-located 배분**인데 sticky arm은 **단일-테넌트 decode 측정에
   가깝다** ⇒ **다른 estimand**.
3. **prefill 축 미통제** — d16에서 TTFT p50이 46.3 → 63.2ms로 움직였고 기전
   주장이 없다(prefill SM 할당 자체는 불변이지만 결과 TTFT가 변했다). E1은
   ITL(D)와 TTFT(108−D)를 동시에 요구하는데, 이 런은 후자를 통제하거나
   측정하지 않는다.
4. **음성대조 부재**(d16 UNSPLIT n=0/8블록, §9 — 구조적으로 불가능, "레버
   부재"의 증거가 아니라 "이 estimand 하에서 정의 불가"의 증거) + 게이트
   S1이 여전히 실행 불가("부분 실현" 분기 부재, `DESIGN.md` §4.3.15(e)) +
   **하드웨어 부여 층 미프로브**(§2, S3 미실행).

⇒ **긴장 A(HE2 vs C2)는 이번 회차로 전혀 닫히지 않았다.**

## 13. 다음 실험 gate

우선순위(claims-auditor):

- **(α) sticky-ON 고-D 대조 셀** — ~25–30 GPU-min, 4 block, decode≈92 SM
  고정 division. **사전등록 예측**: p50 → **12–13ms**(C2 d92 재계산 12.88ms
  기준). 29ms 근처면 **수준을 만든 것이 SM 수가 아니라 처치(sticky
  partition) 자체**이고 §0 판정 전체가 무너진다. **이것이 이 판정을
  반증할 수 있는 유일한 값싼 실험** — 다음 세션의 최우선 gate.
- **(β) OFF 1블록 `TRACE_FORCE_PREFILL=1`** — ~7 GPU-min, §4의 샘플링 법칙을
  직접 검정. 예측: D16 시간 share 3.8%→~10%, SPLIT 순도 ~6%→~1.0.
- **(γ) S3(하드웨어 층)** — per-decode-step green-context handle 직접 로깅.
  §2의 잔여 스코프를 닫는 유일한 실험.
- **(δ) 같은 바이너리 OFF arm** — sticky flag만 끈 단일 부팅, 1블록만.
  **등재 선행조건 아님**(§6이 이미 기록한 "872077은 pre-patch 바이너리"라는
  약점을 보강하는 확인용, 이 종결의 전제가 아님).

---

## 등재 금지 (사유 함께 기록)

이 문서·상위 정본에 다음 문장을 **등재하지 않는다**:

| 문장 | 사유 |
|---|---|
| "slow-mass 잔차 2.03×" / "(iii)는 양적 미종결" | **REFUTED** — count-share vs time-share 단위 불일치. 매칭 단위 비 **0.97** |
| "하드웨어가 16 SM을 부여했다" | 미검증. `decode_sms`는 선택기 재진술. S3 미실행 |
| "S2는 A/B다" / "sticky의 인과 효과" | before/after 비교다(§6, 매니페스트 1파일 차). 인과 읽기를 허가하는 것은 무작위화가 아니라 §5의 ≤1.097× 상한 논증 |
| "음성대조 통과" / "배타성 회복" | d16 UNSPLIT **n=0/8블록 = 검정 불능**. 클래스가 물리적으로 부재(§9) |
| `g`·`G_LEVER`·`G_FLAT`·decode-SM 탄력도·goodput·HE0·긴장 A 일체 | `PREREG_S2_STICKY_ITL_2026-08-03.md` §5.3, `DESIGN.md` §4.3.12(d). p95 비 2.222(≈2.227)는 **값+CI+"판정 없음, UNSPLIT control n=0"** 동반해서만 인용 |
| "28.92는 구간 중앙에서 견고" | 하단(28ms)에서 **3.3%**. 자기 측정 계통 오프셋(C2 대비 −5.4%)과 같은 크기 여유일 뿐 |

## 규율

- 절대날짜(2026-08-05). 원문 삭제 금지, 취소선/철회 표기로 이력 보존(§10 정정도
  동일 원칙).
- 정본 위계: `PROJECT_STATUS.md` > `reports/paper/` > `reports/CONSENSUS.md` >
  `RESUME.md` > 이 문서를 포함한 `results/` 산출물.
- 이 문서는 **claims-auditor의 2026-08-05 감사 허가분**을 doc-steward가
  영속화한 것이다 — 새로운 판정을 추가하지 않는다.

---

*기록: doc-steward, 2026-08-05. 원자료 `results/s2_sticky/`(job 873015),
`results/s8_frontier/`(872077), `results/s8_scaleup/`(865493). 분석
`S2_ANALYSIS_2026-08-04.md`(result-analyst) + claims-auditor 독립 재현(2026-08-05,
스크립트는 현재 에이전트 scratchpad에만 존재, 리포지토리 이관 필요 — 별도 후속
작업). 사전등록 `PREREG_S2_STICKY_ITL_2026-08-03.md`. 상위 정본 갱신:
`PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-05" 소절, `CONSENSUS.md`
§1-31·§3-20, `CLAIM_EVIDENCE_MATRIX.md` Claim A.*
