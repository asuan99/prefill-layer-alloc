# 감사 판정서 — `g16_analyze.py` **분석기층 재감사** (2026-08-23, claims-auditor)

**판정: `CONFIRMED with conditions` (분석기 무결 — 조건 C1–C4)** · GPU 지출 **0** ·
새 성능 판정 **0건** · 대상 트리 수정 **0건**.

> ★**작성 경위**: 감사자가 읽기 전용 규율로 대상 트리 안에 파일을 쓰지 않고 전문을 텍스트로
> 반환했고, 메인 세션이 그것을 이 파일에 보존했다. 수치·판정은 감사자 산출물이다.

**단일 판정 질문**: *정본 §1-33에 승격된 G16 결론이 분석기 **사후 패치**에 의해 만들어졌을
경로가 있는가?* → ★**없다.** 원자료를 **7개 분석기 빌드**(사전등록 원본 → 현재 디스크본)
전부로 재채점했고 **모든 결정량이 불변**이다.

---

## 1. ★★ 사후 패치 4건 재채점 — 전부 결정량 불변 (직접 실증)

| 빌드 | sha16 | 커밋 | 시각 | LO | HI | K9 | HI `D_ttft` | HI `gap_U` | LO `gap_U` |
|---|---|---|---|---|---|---|---|---|---|
| v0 사전등록 | `5f476167` | `4f05dce` | 08-16 22:23 | ITL_SATURATED | ITL_SATURATED | NO_MORE | d44 4/4 boot 1.000 | (gapC 0.2602) | (gapC 0.0316) |
| v1 F1–F4 | `c60bf168` | `8311d6c` | 08-16 22:41 | ITL_SATURATED | ITL_SATURATED | NO_MORE | 동일 | (0.2602) | (0.0316) |
| v2 K11 | `49e83d7f` | `53b8aad` | 08-17 00:39 | ITL_SATURATED | ITL_SATURATED | NO_MORE | 동일 | 0.6342 | 0.3040 |
| v3 동률가드 | `505b0e8d` | `502c5bf` | 08-17 15:21 | ITL_SATURATED | ITL_SATURATED | NO_MORE | 동일 | 0.6342 | 0.3040 |
| v4 E-2 | `f57310bf` | `bef48c1` | 08-17 15:59 | ITL_SATURATED | ITL_SATURATED | NO_MORE | 동일 | 0.6342 | 0.3040 |
| v5 `boot_s` | `2c9626f2` | `3ac0cb5` | 08-17 16:16 | ITL_SATURATED | ITL_SATURATED | NO_MORE | 동일 | 0.6342 | 0.3040 |
| **v6 디스크** | **`bac783be`** | `42b701a` | 08-17 19:00 | ITL_SATURATED | ITL_SATURATED | NO_MORE | 동일 | 0.6342 | 0.3040 |

**v0→v6 불변**: verdict · K9 액션 · HI `D_ttft`=d44 식별(4/4, boot 1.000, tied 0) ·
HI `D_itl` 미식별(rank 1/4, boot 0.5115) · LO `D_ttft` d24 3/4 boot 0.741 미식별 ·
LO `D_itl` d16 2/4 boot 0.8093 미식별 · 사다리 36 rung · exact band · operating point.

**패치별 판정**
1. **K11 교체**(`49e83d7f`, 08-17 00:39) — ★**데이터 이전임을 아티팩트로 독립 확인**:
   최초 blk1 산출물 mtime **08-17 04:53:19**, 스모크(884292/884320)는 **d44 단일 arm**이라
   상단 arm 구조를 볼 수 없었다(커밋 메시지 주장에 의존하지 않는 확인). 교체 *이전* 정의
   (`contenders`)로도 gap 0.0316(LO)/0.2602(HI) < δ=1.0 ⇒ `ITL_SATURATED`. 병기 대안
   (`top_half_by_sm`)은 7-arm 격자에서 **문자 그대로 동일 집합**.
2. **동률 가드**(`505b0e8d`, 데이터 5.8h 후 — 정본 자기고지) — **무효과 실증**:
   `n_blocks_tied=0`·`tied_fraction=0.0` 전 4셀, v2=v3. 완화 근거("빈 ITL 요청 0건")도
   **독립 재계산 확인**(56 부팅 합계 `empty_itl = 0`) ⇒ `+inf` 분기 미발화로 동률 원리상 불가.
3. **E-2 가산 패치**(`f57310bf`) — 결정 필드 v3=v4, 신규는 `citable`·`block_verdict_flags`·
   `residency_fraction_by_boot`·`placement_by_block`뿐.
4. **`boot_s` 유도**(`2c9626f2`) — v4=v5. 유도식 검산: blk1/d16 **37.881972 s** vs 나머지
   27 부팅 32.64–32.88 s ⇒ E-3 H18 이상 재현. **결정 경로 밖 side output**.
5. **주석 전용**(`bac783be`) — 비주석 diff **0줄**. 저장된 `G16_RESULTS_2026-08-17.json`을
   **키 추가/삭제 0건, 부동소수 last-ulp 12개 + `analyzer_path`만** 차이로 재현.

⇒ ★**"바뀌지 않았다"가 등재 가능한 음성으로 성립한다.**

---

## 2. ★ F1–F4는 게이트 #43의 **재발이 아니라 역방향 수리**

F1이 한 일 = 최초 구현이 §4의 bare argmin `D_itl`을 §6 K1 게이트-조건부로 **조용히 재정의**한
것을 **되돌린 것**. 시점 08-16 22:41 = 데이터 **6시간 이상 전**. v0는 `delta=None`, v1부터
`delta`(LO 0/HI 20)·`sign_verdict` 존재 — **판정 불변**이고 양 phase `delta_citable=False`,
정본 승격금지 목록이 이미 "`delta` 단독"을 막아 뒀다. **잔여 재정의 없음**(게이트본
`D_itl.arm=None`과 bare본 `D_itl_arm=d64`를 **둘 다** 방출). **K1–K10은 7 빌드 전부 불변**,
K11만 신설.

---

## 3. ★ S1 사다리 결손 — 해소 진짜, 다른 은폐 없음

`s_itl_exact_bands` + 경고가 양 phase 방출, **감사 독립 스크립트가 동일 breakpoint 재현**.
* HI `[58.533, 58.625)` **폭 0.0920 ms** → Δ_SLO **+20** · `[58.625, 60.461)` 1.836 → −10 ·
  `[60.461, 83.952)` 23.49 → −20.
* LO 유일 협대역 0.0316 ms가 **12.27 ms**에 있어 사다리 바닥 45 ms보다 33 ms 아래 ⇒ 은폐 불가.
⇒ **사다리 범위 안 sub-1ms 밴드는 HI 0.0920 하나뿐이고 정본이 이미 보고했다.**
⚠️**병기 필수**: 그 0.0920 ms는 HI d64 블록간 SD(0.66 ms)의 **~1/7** — 정본의 "무판정"이
몸 사리기가 아니라 **정확한 처리**라는 근거.

---

## 4. ★★ 양성대조 — 신규 결함 2건 (게이트 #9 계열, **결론 비영향**)

**(a) 추정량 채널에는 이빨이 있다(항등식 아님)**

| 변이 | 실패한 대조 |
|---|---|
| `itl_quantile` 0.95→0.90 | PC-A, PC-E, PC-D |
| `median`→`fmean`(M_ttft) | PC-E, PC-D |
| `duration`×3(게이트 #7 방향) | PC-A |

PC-C는 셋 다 통과 — **서수 전용**이라 약하나 항등식은 아니다.

**(b) ★★ 그러나 대조 묶음이 이번 감사 대상 4건 전부에 눈이 멀었다**

| 변이 | 대조 | 실제 영향 |
|---|---|---|
| K11 44→34 | **all_passed=True** | ★**LO 판정이 뒤집힌다** |
| 동률 가드 제거 | **all_passed=True** | 이 데이터엔 무효과 |
| tie-break 방향 반전 | **all_passed=True** | — |
| `S_itl` `<=`→`<` | **all_passed=True** | — |

⇒ ★**커밋 `53b8aad`/`502c5bf`/`bef48c1`/`3ac0cb5`가 근거로 든 "PC-A..PC-E 전후 동일"은
그 패치들이 무해하다는 증거가 아니다.** 무해성은 §1의 **직접 재채점**이 세운 것이다.

**(c) ★★ PC-B가 자기 주석이 약속한 일을 하지 않는다**(§3 항목64와 **별건·미해결**)
주석: *"`load_campaign_grid` records under this same tag …"*. 그러나 `main()`에서
`run_controls()`가 `run_campaign()`**보다 먼저** 돌고, PC-B는 자기가 만든 `probe0` 호출로
`signature_set("headline")`을 스냅샷한다. **변이 증명**: `_headline_estimands`가
`itl_quantile=0.90`을 쓰도록 바꿔 캠페인 경로만 갈라놓아도 **PC-B 통과**, 기록된 서명은 여전히
`0.95`. G16엔 무해(실제 전 인자 기본값)하나 **"대조가 캠페인 코드 경로를 탔다"의 근거로 인용 금지**.

---

## 5. ★★ 두 번째 눈 — **있다. 부채 노트가 시사하는 것보다 강하다**

`audit_g16_results_2026-08-17/`가 `g16_analyze.py`를 **import하지 않는** 독립 재구현이며
전부 재실행됨: `indep_perboot.py` → 보존본과 **완전 동일** 재생성(LO d16 12.3042 / HI d64
58.5330) · `sens_delta_slo.py` → `gap_upper` 0.3040/0.6342, exact breakpoint 전부,
**P(gap>δ)=0.2659**(정본 0.266) 재현 · `sens2.py` → 처리량 argmax **d64** 5.6096 vs d44
5.5361, 페어드 **+0.0735±0.0156**(**+1.33%**), `P(first-band Δ_SLO>0)=0.6322`,
분포 `{−10:0.2407, 0:0.127, +10:0.1196, +20:0.5127}`.

★★**결정적 사실**: `g16_analyze.py` 안에 **`delta_slo` 부트스트랩이 존재하지 않는다**(전수
확인). ⇒ 정본 §1-33의 **가장 하중이 큰 두 문장** — *"처리량 최적 REFUTED(argmax=d64)"* 와
*"payoff 구간 P(부호>0)=0.632 ⇒ 무판정"* — 은 **애초에 분석기 산출물이 아니라 독립 감사
재도출물**이며 **구조적으로 사후 패치 4건과 무관**하다.

**두 번째 눈의 한계(정직)**: 추정량 *정의*는 구성상 공유(percentile을
`pdmux_eval.analyze.percentile`에서 복사, duration 합산). 판정 상태기계·사다리 라벨은
재구현하지 않고 **그 입력**(블록별 argmin·부트스트랩 분포)만 재현 — 다만 그 입력이 판정을
결정하므로 실질 커버리지는 충분.

---

## 6. ★ `analyzer_sha256` — 정본 참조 오류 (실재, 결정 비영향)

* 디스크 `g16_analyze.py` = **`bac783be5ba40974b279a2c8f5ee486d62091b3a4f9000fee5c051e899e5b9ec`**
* `G16_RESULTS_2026-08-17.json:3` 자기기록 = **`bac783be…`** ✅
* **`reports/CONSENSUS.md`** — *"`analyzer_sha256=2c9626f2…`"* ⇒ **사실과 다름**
* **`PROJECT_STATUS.md`** — 동상, **폐기된 판본** 지목
* **`PREREG_G16_RULES_REV3_2026-08-16.md`** 자기모순: **1040행** `bac783be…`(옳음) vs
  **1041행** *"판본 이력: `2c9626f2…`(최종)"*
* `bac783be`는 **정본 문서 어디에도 없다**(아티팩트 파일에만)

⚠️★**이 오류가 감사 과업 지시문에도 그대로 전파됐다**(메인 세션이 `2c9626f2…`를 정본
참조값으로 제시) — §3 항목41/50(*"인용금지·참조는 정본 등재만으로 전파되지 않는다"*)의
**provenance층 변종**.

---

## 7. 신규 발견 — K11 민감도 (스코프 조건으로 등재 필요)

| K11 | LO verdict / K9 | LO gap | HI verdict / K9 | HI gap |
|---|---|---|---|---|
| 16 | UNIDENTIFIED / **ADD_2_BLOCKS** | 2.2217 | UNIDENTIFIED / ADD_2 | 25.4189 |
| 24 | UNIDENTIFIED / **ADD_2_BLOCKS** | 2.2217 | UNIDENTIFIED / **ADD_2** | 1.9282 |
| **34** | ★**UNIDENTIFIED / ADD_2_BLOCKS** | 1.9151 | ITL_SATURATED / NO_MORE | 0.6342 |
| **44(등록)** | ITL_SATURATED / NO_MORE | 0.3040 | ITL_SATURATED / NO_MORE | 0.6342 |
| 54 / 64 | ITL_SATURATED / NO_MORE | 0.3040 | ITL_SATURATED / NO_MORE | 0.6342 |

⇒ ★**LO의 `ITL_SATURATED`는 U-정의에서 한 칸 여유뿐이다.** §6이 "upper arms"를 정의 없이
뒀으므로 이 자유도는 **실재했다**. 완화 3건: (i) 선택이 **데이터 이전**이고 아티팩트로 확인 ·
(ii) 44는 7-arm 격자의 top-half와 **동일 집합** · (iii) addendum C 이전 테이블의 두 정의
(`contenders`·`top_half`) 둘 다 같은 판정. ★**34는 아무도 제안한 적 없는 정의다.**

---

## 8. 정본 영향 — 등재 결론 **0건 영향**, 참조 서술 **3곳**

| 위치(★앵커로 지목) | 문제 | 수리 |
|---|---|---|
| `reports/CONSENSUS.md` — `` `analyzer_sha256=2c9626f2…` `` | 사실 오류 | `bac783be…`로 정정 |
| `PROJECT_STATUS.md` — `` g16_analyze.py(`analyzer_sha256=2c9626f2…` `` | 폐기 판본 지목 | 동일 정정 |
| `PREREG_G16_RULES_REV3_2026-08-16.md:1041` | 1040행과 자기모순 | "(최종)" 라벨 이동 |

「다음 실험 gate」 **#17 (f)**(*"분석기 누적 미감사"*)는 ★**해소 가능** — 단 아래 조건 4건을
함께 등재할 때만.
⚠️**정본 두 문서가 dirty 상태**(동시 세션 편집 중) ⇒ ★**행 번호가 아니라 앵커 문자열로 수리**.

---

## 9. CONFIRMED 조건 4건

* **C1** `2c9626f2…` **인용 금지**. 위 3곳 정정 + `citation_stops.tsv` 등재.
* **C2** *"PC-A..PC-E 전후 동일"* 을 K11·동률가드·tie-break·`S_itl` 경계 무해성의 근거로
  **인용 금지** — 변이 테스트로 **대조가 그 4건 전부에 눈멀었음이 실증**됐다. 근거는 §1의
  **패치별 직접 재채점**을 인용하라.
* **C3** **K11 스코프 등재**: LO의 `ITL_SATURATED`·K9 `NO_MORE_BLOCKS`는 **U 정의가
  min SM ≥ 44일 때만** 성립(≥34이면 `UNIDENTIFIED`/`ADD_2_BLOCKS`). HI는 ≥34까지 성립.
* **C4** **PC-B**를 *"대조가 캠페인 코드 경로를 탔다"* 의 근거로 **인용 금지**(§3 항목64와
  **별건·미해결**).

## 10. 이를 닫을 실험 (E1–E4 전부 GPU 0)

* **E1(C2 해소)** — 분석기 대조에 **손잡이 음성대조 4종** 신설(K11→34 · 동률가드 제거 ·
  tie-break 반전 · `S_itl` strict `<`). 각 변이본에서 등록 검사 **최소 1건이 반드시 실패**
  해야 한다(§3 항목53). 실패시키지 못하는 손잡이는 **"대조 불가"로 보고**.
* **E2(C4 해소)** — PC-B 비교를 `run_campaign` **뒤로** 옮기거나 실캠페인 파일 서명과
  비교하도록 변경. 그 뒤 감사 변이본(`m8_pathdiverge.py`)에서 **PC-B가 실패해야** 수리 확인.
* **E3(C3)** · **E4(C1)** — 문서 등재·정정만.
* **E5(선택, GPU>0, 이 판정에 불요)** — K11=34가 탔을 K9 분기(`ADD_2_BLOCKS` → 6블록 +
  K12 rank ≥5/6)는 **평가 불가**(블록 5–6 부재, 게이트 #17/#19가 사후 추가 금지).
  LO 포화가 U-정의에 견고하다고 주장하려면 **별도 사전등록 6블록 캠페인(≈2 GPU-hr)** 필요.
  그 전엔 **C3 스코프 조건이 유효**하다.

## 11. 감사하지 **않은** 것 (범위 오독 방지)

하네스(`g16_grid.sbatch`·telemetry 배선·realized-probe) · `pdmux_eval.analyze` 자체(분석기와
두 번째 눈이 **같은 추정량 정의를 공유**) · `residency_scope_2026-08-17/`의 A-2 수치 ·
`verdict_reachability`/`calibrate_donor_rule`(9/9 통과 확인했으나 적대적 탐침 없음).

**반증 실패 — 단 C1–C4 등재 전엔 `CONFIRMED(분석기 무결)` 아니다.**
새 성능 판정 0건 · 정책 순위 변경 0건 · HE0 불변 · *"gate #16을 닫았다"* 금지 불변 · GPU 0.

**재현 아티팩트**: scratchpad `g16audit/{vers,out,mut,indep}/` + `fingerprints.json`
(7 빌드 · 변이본 m1–m8 · K11 스윕, 전부 CPU 재실행 가능).
