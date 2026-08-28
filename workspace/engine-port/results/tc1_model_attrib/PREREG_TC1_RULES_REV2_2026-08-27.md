# 사전등록 **TC1 rev2** — 모델 귀속 (rev1 `NO-GO` 반영)

> ★★**SUPERSEDED (2026-08-28)** — 규칙층 재감사 `NO-GO`(死因 F5–F7 · 차단 B19–B30), 그리고
> **설계층 도달가능성 검사에서 `NOTHING_PURCHASABLE`**(argmax=d44 시나리오). **설계 근거 재사용 금지.**
> 후속 `PREREG_TC1_RULES_REV3_2026-08-28.md` · 판정서 `audit_tc1_rules_rev2_2026-08-28/VERDICT.md`.

2026-08-27 · ★**규칙층 초안 — 미감사** · GPU 지출 **0** · 미제출 · 새 성능 판정 **0건**

선행: [`PREREG_TC1_RULES_2026-08-27.md`](PREREG_TC1_RULES_2026-08-27.md)(rev1, ★**SUPERSEDED**) +
[`audit_tc1_rules_2026-08-27/VERDICT.md`](audit_tc1_rules_2026-08-27/VERDICT.md)(`NO-GO` · 死因 F1–F4 · 차단 B1–B18)
규칙: [`tc1_rule_rev2.py`](tc1_rule_rev2.py) `RULE_REV=2` · **589,824 세계** · 자체검사 **13건 PASS** ·
`rule_sha256 = 0c56eab68a1c98c0…`

> ★금지: *"rev2가 규칙층을 통과했다"* · *"TC1의 死因이 닫혔다"*.
> ★★**rev2 단독으로 닫히지 않는 것 하나**: F2 확정에는 감사가 지정한 **0.3 GPU-hr 프로브**가 필요하다(§5.3).

---

## 0. 인용 규약 — rev1의 정정문이 **또 틀렸다**

rev1 §0은 `stage0ppp_a0_rule.py`의 `gate #66` 오인용을 보고하면서, 그 자리에
*"rules-as-code discipline은 CONSENSUS §3 **항목66**"* 이라 적고 `VERIFIED` 딱지를 붙였다.
**대조 결과 CONSENSUS §3 항목66은 "사다리 해상도"(G16, 2026-08-17)이고, 맞는 것은 항목81 = PROJECT_STATUS 게이트 #61**이다.
⇒ **항목80이 예측한 전파가 "다음 판본"이 아니라 같은 문장 안에서 재발했다.** rev2에서 정정한다.

그리고 rev1의 진단 두 개가 더 틀렸다:
- **체계는 둘이 아니라 넷**이다 — `CLAUDE.md` #1–8 / `PROJECT_STATUS.md` #1–80 / `CONSENSUS.md` §3 #1–100 / 메모리 topic #1–80.
  rev1이 "항목53/66/80"이라 쓴 것은 전부 **메모리 번호를 CONSENSUS 번호로 표기**한 것이다.
- **"오인용 3건"은 하계**다. 감사가 PREREG+rule 파일만 세어 15파일 ~50건을 확인했고,
  그중 `nsl_eb_rule.py:302`는 rev1이 ★★★로 **유일 사례**라 표시한 (3)과 **같은 부류**다.

**rev2 규약**: 모든 게이트 인용에 **이름을 병기**하고, 체계 A는 `CLAUDE.md 게이트 #N`으로 한정한다.
번호를 검증 못 한 규율은 **번호 없이 이름으로만** 쓴다. ★**저장소 전수 정정은 이 캠페인의 범위가 아니다** —
별건 등재(doc-steward)로 남긴다.

---

## 1. 발견 → 변경 대응표 (감사가 이 표로 대조하면 된다)

### 死因

| # | 감사 지적 | rev2 변경 |
|---|---|---|
| **F1** | `PDMUX_STICKY_PARTITION`+`PDMUX_SLO_SCHED`가 `multiplexing_mixin.py:298-303`에서 RuntimeError. 우회하면 realized 4–19% < 문턱 0.90 ⇒ 양쪽 다 `MEASUREMENT_ABSENT` | ★**sticky 요구를 철회**한다. 수입 자체가 **범주 오류**였다 — sticky는 M3의 *파티션 라벨로 조건화된 per-token ITL*을 구하려던 장치이고, **TC1의 estimand는 end-to-end goodput이라 파티션에 조건화하지 않는다**. 정본 HE0 캠페인도 sticky 없이 돌았으므로 TC1은 거기 맞춰야 한다. ★**M4R 감사가 이 방향을 독립적으로 지지**한다(`../m4r_confinement/audit_m4r_rules_2026-08-27/VERDICT.md` F1: 이 기판에서 파티션 조건화는 SM과 aliasing되고, `CONSENSUS.md` §1-26(B)가 *"decode가 D SM에서 돌았다 ⟺ prefill이 in-flight였다는 같은 사건"* 을 이미 등재). realized는 **문턱 없는 공변량**이 아니라 **비교가능성 축**(`compar`)이 된다 — 두 arm의 realized 체류가 `REALIZED_MAX_GAP=0.15`를 넘으면 `REALIZED_MISMATCH`로 차단 |
| **F2** | anchor를 static argmax에 못 박으면 컨트롤러가 LO·HI 양쪽에서 anchor를 안 떠나 `NO_FLIP_BOTH_LOSE`가 기전적으로 강제 | ★anchor = **정본 상수, 모델 무관**(`sharegpt_vary_bench.sbatch:40`). Stage 1은 **비교자만** 공급하고 anchor와 **분리**된다. 그리고 감사의 진짜 지적(*"반응적 컨트롤러가 애초에 이길 수 있는가"*)에는 **관측 가능한 축 `visits_argmax`** 로 답한다 — 컨트롤러가 자기 argmax 인덱스를 한 번도 방문하지 않으면 그 arm은 `CONTROLLER_DEGENERATE`. 산출은 telemetry `runtime_snapshot.stream_index`의 **시간가중 체류분포**(신규 계측 불요) |
| **F3** | 하네스가 `gpL`/`gpH`/`gpC` 셋을 내는데 어느 것인지 미등록 | ★**`gpC`(COMBINED, 라운드 duration 합산)** 를 1차로 등록(`CLAUDE.md 게이트 #7`). 정본 §1-7의 3.220이 그 값이다. `gpL`/`gpH`는 **필수 부수 보고**. §3(f)의 rate 정규화로 pooled 가중치가 arm마다 달라지는 문제는 **flip이 두 개의 within-model 부호 비교**이므로 오염되지 않는다 — 이 논증을 §4에 명시 등록한다 |
| **F4** | *"부호는 단조 재척도에 불변"* 이 대수적으로 거짓(임계가 재척도 **뒤에** 적용) + 판별력 축 부재 | ★그 주장을 **철회**한다. §9의 정직한 서술은 *"부호도 견디지 못한다"* 이다. **`discrim_H`/`discrim_T` 축 신설** — conjunctive 술어 통과율이 `(0.05, 0.95)` 밖이면 `NOT_DISCRIMINATING`. 근거는 정본 §1-1의 *"위반 0건 술어의 goodput은 throughput의 다른 이름 — 판별력 0"* |

### 차단

| # | rev2 변경 |
|---|---|
| **B1** | `compar` 축 신설(위 F1). realized는 산문이 아니라 **축**이다 |
| **B2** | `boot`를 **모델별 2축**(`boot_H`/`boot_T`)으로 분리 |
| **B3** | ★**격자 확장을 규칙에서 제거**한다. rev1은 "1회 확장 후 재실행"을 산문으로만 허용해 사후 자유를 남겼다. rev2는 **격자를 처음부터 `d16…d74` 7점으로 고정**(전부 저장소에 존재)하고, 그래도 끝점이면 `GRID_EDGE_UNRESOLVED`로 **차단**한다. `edge_lo`가 구조적으로 해소 불가(d08은 `results/cudagraph_probe/`에만 존재)라는 감사 지적을 **한계로 등재**한다 |
| **B4** | ★감사가 옳다 — **정본이 이미 답했다.** `CONSENSUS.md` §1-33(G16, jobs 884336/884410/884411/884412)이 **7-arm 확장 격자**에서 HI `M_ttft` argmin = **내부점 d44**(4/4 블록)를 측정했다. rev2는 이것을 **격자 내부성 근거로만 인용**하고 **비교자 크기로는 인용하지 않는다**(게이트 #41 telemetry 드리프트 — 비교자는 TC1 자신의 Stage 1에서 온다). `GOODPUT_BASE` 모호성도 이로써 해소: **δ는 TC1 자신의 Stage 1 값에서 계산**한다 |
| **B5** | ★`sign`을 4상태로 재정의 — `win` / `lose_sig`(Δ̂ ≤ −δ ∧ CI가 0 배제) / `equiv`(±δ TOST 통과) / `undecided`. `UNDERPOWERED_NULL`이 **도달 가능**해지고 `INCONCLUSIVE`와 **서로소**다(`T5` 검증) |
| **B6** | 같은 변경으로 해소 — branch-B(`NO_FLIP_BOTH_LOSE`)는 이제 **두 arm 모두 `lose_sig` 또는 `equiv`** 를 요구한다. *"이긴 증거가 없다"* 로는 못 간다(`T3` 검증) |
| **B7** | ★`MDE_SLACK=2.0` **폐기**. 대체: `power_adequate ⟺ P(win | Δ = 2δ) ≥ 0.80` — **결정 문턱에서 유도**된 기준이다. 이빨 확인: CV 6% arm은 거부된다(`T13`) |
| **B8** | ★arm은 **페어링되지 않는다**(별도 부팅)고 명시 등록. Welch 2-표본, **임계값을 실현 df에서 계산**(하드코딩 3.182 제거). Stage 1의 n=3 SD는 **설계 시점 sizing 전용**이고 판정 SD는 Stage 2 자신에서 온다 |
| **B9** | 계측 불가 술어 **삭제**. `visits_argmax`(F2) + `switch_count` + `stream_index` 시간가중 체류분포 — **셋 다 기존 telemetry에서 산출** |
| **B10** | ★음성대조 **재설계**. 2분할(arm당 n=2, t₀.₉₇₅,₁=12.71 ⇒ 공허 통과)을 폐기하고 **치환 검정**으로 교체 — TC1 **자신의 Stage 2 rep**에서 arm 라벨을 치환해 flip 통계량의 귀무 분포를 전수 산출(C(8,4)=70). 라벨 함수가 아니라 **부호 추정량**을 실행하므로 항등식이 아니고, 2026-07 데이터를 안 쓰므로 **게이트 #41 모순도 해소** |
| **B11** | ★양성대조 **재설계**. 주입을 `+4δ`로 키워 여유를 확보하고(rev1은 0.75 se), **실패 분기를 등록**한다. 그리고 무연산이던 변이 테스트를 **짝 구조**로 교체 — 같은 하네스에 `0` 주입 시 **발화하면 안 된다**. 두 방향이 다 통과해야 대조가 성립한다 |
| **B12** | `scoring`을 **모델별 2축**으로. `sign_*`이 **p95 채점**임을 규칙 파일 상수(`SIGN_SCORING`)로 명시 |
| **B13** | ★`band` 축 **삭제**(§3(f) 하에서 장식). `cliff`를 3상태로 확장해 `capacity_unmeasurable`을 흡수. 그리고 감사 지적을 수용해 **§6에 설계층 도달가능성 절**을 신설 — 실질 라벨마다 **실험이 실제로 만들 수 있는 세계**를 하나씩 적는다(`T2`는 격자 내부 성질이라 이걸 못 잡는다) |
| **B14** | ★**명명 mutant 9종 + `uncovered_mutants: []` 메타검사** 신설(A0 rev4 수준). 각 가드가 자기 mutant를 갖는지도 검사(`T8`) |
| **B15** | 과대 서술 **정정**. `T9`가 정확히 이렇게 검증한다 — 순서는 **차단의 이름만** 바꾸고 *"실질 판정에 도달하는가"* 는 **바꾸지 않는다**(2,304 라벨 이동, 도달 여부 불변) |
| **B16** | `tokenizer_skew`에 **문턱과 결정 역할** 부여 — 두 모델 median `input_lens` 비가 `1 ± 0.20` 밖이면 `TOKENIZER_SKEW`로 **차단**. 사후 설명 채널이 아니다 |
| **B17** | anchor 인덱스 **사상 등록** — 부팅 시 `pdmux_slo.yml`의 division 리스트를 덤프·해시해 아티팩트에 기록하고, `idx → D` 표를 판정서에 싣는다 |
| **B18** | 개수 불일치 정정 — rev2 자체검사는 **13건**이고 문서·코드·JSON이 같은 값을 쓴다 |

---

## 2. 질문 (rev1과 동일)

> 같은 green context · 같은 conjunctive-SLO goodput · 같은 ShareGPT 변화-trace에서,
> `sign(best dynamic − 자기 best static)`이 hybrid arm과 size-matched Transformer arm에서 다른가?

---

## 3. 항등식 사전 점검 rev2 (게이트 #40 — 결정량 자체가 항등식일 수 있다)

rev1의 (a)(b)(f)는 유효하고 (c)는 강화됐다. **신규·정정**:

- **(a′)** rev1은 "정본 argmax d44가 격자 끝"이라 했으나 그건 **4-arm 격자 한정**이다.
  정본 §1-33이 **7-arm 격자에서 내부점 d44**를 이미 측정했다 ⇒ 격자는 처음부터 7점으로 고정하고, Stage 1은 이 사실을 **재구매하지 않는다**.
- **(b′)** 단일 비교자 + 표본 분할 유지. 단 anchor는 **비교자와 분리**(F2).
- **(c′)** 동적 arm = `bind+GATE` 유지. **그러나 그 arm이 이길 수 있는지 자체가 검정 대상**이며,
  `visits_argmax`가 그것을 관측 가능하게 만든다. 방문조차 안 하면 **모델 차이가 아니라 설계 도달 실패**다.
- **(g′)** ★rev1의 grep이 **플래그 존재만 보고 저장소가 직접 쓴 예외 문자열을 안 읽었다**(F1의 직접 원인).
  M4R 감사도 같은 형태를 냈다 — grep이 **정본의 어휘가 아니라 저자의 어휘**로 돌았다.
  rev2 규약: 등록하는 모든 `PDMUX_*` 조합에 대해 **`raise`·`assert`·`undefined` 문자열을 함께 grep**하고 결과를 §7에 첨부한다.

---

## 4. Estimand (F3 · B12)

`sharegpt_vary_bench.sbatch:82-102`에서 축자. **1차 = `COMBINED` goodput**(`gpC`, 라운드 duration 합산).
**1차 채점 = 요청 내부 ITL p95**(`CLAUDE.md 게이트 #4`), **2차 = mean**(정본 HE0 비교가능성 다리).
두 채점이 어느 모델에서든 부호에서 어긋나면 `SCORING_DEPENDENT`로 **차단**.

`sign(M)`: `win` / `lose_sig` / `equiv`(TOST) / `undecided` — §1 B5 참조.
검정 = **Welch 2-표본**(비페어), 임계값은 실현 df에서. δ_M = `0.03 × goodput(argmax-static, M)`.

★**cross-model pooled 가중치가 달라도 flip은 오염되지 않는다**: flip은 두 개의 **within-model** 부호 비교이고,
모델 간에는 어떤 크기도 비교하지 않는다. 이 논증을 등록하며, 위반 문장을 §11에 금지로 넣는다.

---

## 5. 격자 · 단계

| 단계 | 무엇 | 격자 | rep |
|---|---|---|---|
| **0** | 모델별 off-cliff 용량 `C_M` | rate 스캔 | 2 seed |
| **1** | static argmax + SD (**비교자만**) | `d16…d74` **7점 고정** × 2 모델 | n=3 |
| **2** | 부호 판정 | {argmax-static, bind+GATE} × 2 모델 | n=4 (신규 rep) |

anchor = **정본 상수, 모델 무관** · rate = `0.48·C_M` / `1.90·C_M` · Stage 1 결과는 Stage 2 제출 전 **동결·SHA-256**.

### 5.3 ★rev2로 닫히지 않는 것 — 선행 프로브 2건
1. **F1 확정 (GPU 0)**: sticky+`SLO_SCHED` 부팅 실패를 CPU init으로 재현해 아티팩트에 기록.
2. **F2 확정 (≈0.3 GPU-hr)**: Zamba2 `bind+GATE`를 `PDMUX_SLO_ANCHOR_IDX`=d44 인덱스로 **1 rep**.
   `switch_count≈0` ∧ goodput ≈ d44-static − ε 이면 **F2가 실측 확정**되고, 그때 §1 F2의 수리가
   충분한지 **재감사**가 필요하다. ★**이 프로브 없이 본 캠페인을 제출하지 않는다.**

---

## 6. 결정 규칙 · 설계층 도달가능성 (B13)

[`tc1_rule_rev2.py`](tc1_rule_rev2.py) — 16축 **589,824 세계**, 자체검사 13건, 명명 mutant 9종.
가드 순서 등록: `boot → cliff → compar → discrim → grid → visits → scoring`.

★**설계층 도달가능성**(`T2`가 못 잡는 것 — 실험이 실제로 만들 수 있는 세계를 라벨마다 하나씩):

| 라벨 | 실험이 이걸 만들 수 있는가 |
|---|---|
| `FLIP` / `REVFLIP` | ✅ 두 모델이 서로 다른 부호를 내면. **단 `visits_argmax=yes`가 선행조건**이고 그것이 §5.3-2로 검정된다 |
| `NO_FLIP_BOTH_WIN` | ✅ 두 컨트롤러가 각자 argmax를 3% 초과해 이기면 |
| `NO_FLIP_BOTH_LOSE` | ✅ 단 이제 **증거 있는 음성**(`lose_sig`/`equiv`)이 필요하다 |
| `UNDERPOWERED_NULL` | ✅ Stage 1 SD가 커서 `P(win|2δ) < 0.80`이면 |
| `CONTROLLER_DEGENERATE` | ✅ §5.3-2가 실제로 이 세계를 겨눈다 |
| `NOT_DISCRIMINATING` | ✅ rate가 용량 대비 너무 낮거나 높으면 |
| `REALIZED_MISMATCH` | ✅ sticky를 뺐으므로 두 arm의 auto-revert 비율이 다를 수 있다 |

---

## 7. 대조 (게이트 #9 항등식 계열 · #44 빈 서명 · 변이 테스트)

- **음성 = 치환 검정**(B10): TC1 자신의 Stage 2 rep에서 arm 라벨을 치환, C(8,4)=70 전수. 귀무 발화율 보고.
- **양성 = `+4δ` 주입**(B11): 반드시 발화. **실패 분기 등록**.
- **짝 음성 서명 = `0` 주입**: 발화하면 **실패**. 두 방향이 다 통과해야 대조 성립.
- **변이**: 위 둘을 맞바꾼 변이본에서 **반드시 실패**해야 한다.
- **grep 첨부**(§3(g′)): 등록 `PDMUX_*` 조합의 `raise`/`assert`/`undefined` 검색 결과.

---

## 8. 하네스 요건 (2단계 대조표)

1. `MODEL`/`CTX` 인자화 — ★**TC0는 "부분 완료"**다(정본 vary 하네스에서 Qwen2.5-3B 미실행).
2. ★**sticky를 켜지 않는다**(F1). realized 체류는 telemetry `stream_index` 시간가중으로 **보고**한다.
3. `stream_index` 시간가중 체류분포에서 `visits_argmax` · dwell · `switch_count` 산출.
4. p95·mean 두 채점을 같은 jsonl에서. 사전등록 분석기가 **결정 규칙의 모든 항을 계산**(게이트 #20).
5. 모델별 `input_lens` median 비 산출(B16 문턱).
6. `--max-mamba-cache-size = cap`(Zamba2만). Qwen엔 없음 — **제거 대상이 아니라 보고 대상**.
7. 두 모델을 **같은 배치·같은 트리·같은 telemetry 설정**으로(게이트 #41).
8. 부팅 시 `pdmux_slo.yml` division 리스트 덤프·해시(B17).

---

## 9. 닫지 못하는 것 (F4 정정 반영)

1. ★**두 체크포인트의 비교다.** rev1이 제시한 탈출구(*"부호는 단조 재척도에 불변"*)는 **거짓**이다 —
   goodput은 임계 지시함수이고 임계는 재척도 **뒤에** 적용된다. 정직한 서술: **부호도 견디지 못한다.**
   그래서 `discrim` 축이 *"두 모델이 임계 대비 판별 구간에 있다"* 를 **선행 검정**한다. 그것으로도
   파라미터·층수·tokenizer 차이는 **남는다**. C2b를 폐기시킨 반론(E4)을 TC1이 피하지 **못한다**.
2. flip이 발화해도 **"왜"에는 답하지 않는다** — hybrid 특이성 후보가 최소 둘(층 조성 / admission이 mamba pool에 결합).
   *"층 타입 때문이다"* 는 금지.
3. **일반화 불가**: 1 hybrid × 1 Transformer × 1 워크로드 × 1 SLO × 1 기판.
4. ★**HE0를 되살리지도 죽이지도 않는다.** rev1 §1.1-4는 이를 *"TC1의 측정 실패로 라벨링"* 이라 적었는데
   규칙의 `BOTHWIN`은 *"중심 negative가 흔들린다"* 는 **실질 라벨**이었다(감사 지적, 사후 선택권).
   **rev2 확정**: `BOTHWIN`은 실질 라벨이며, 발생 시 **정본 재검토 사유**이지 측정 실패가 아니다. 산문을 코드에 맞춘다.
5. **Claim B(green-context 종속)를 닫지 않는다.**
6. ★**`edge_lo`는 구조적으로 해소 불가**(d08 자산이 정본 ladder에 없음, B3).

---

## 10. 손잡이 사전 등재 (게이트 #19)

`PDMUX_SLO_MODE=binding` + `PDMUX_SLO_FEAS_GATE=1` · **`PDMUX_SLO_ANCHOR_IDX` = 정본 상수(모델 무관)** ·
`PDMUX_TPOT_SLO_MS/TTFT` = 60/3000 · `FEAS_OCC/MARGIN/DWELL/EMA` = 엔진 기본(모델별 튜닝 **금지**) ·
★**`PDMUX_STICKY_PARTITION` 미사용** · `max_running_requests` = 48(이 캠페인의 손잡이 **아님**) ·
rate = `0.48·C_M`/`1.90·C_M` · `N_REPS`=4 · `DELTA_REL`=0.03 · `POWER_AT`=2.0 · `PAIRED`=False.

---

## 11. 금지 문장

rev1의 7건 유지 + 신설: *"rev2가 규칙층을 통과했다"* · *"sticky를 뺐으니 realized는 문제가 아니다"* ·
*"TC1이 C2b 반론을 피한다"*(rev1 §1.1-1 철회) · *"두 모델의 goodput 크기를 비교했다"*(within-model 부호만) ·
*"`BOTHWIN`은 측정 실패다"* · *"오인용은 3건이다"*.

---

## 12. 1단계 감사에 묻는 것 — 단 하나 + 합격 기준 (게이트 #34 · #37)

> ★**판정 질문**: *"F1–F4의 수리가 **지목된 좌표에서만 국소적으로** 이루어졌는가, 아니면 파급까지 재도출됐는가?"*
> (3회차 A1 감사가 명명한 형태 — *"수리는 국소, 주장은 전역"* — 이 이 판본에도 있는지)

**합격 기준**: ① F1–F4 각각이 **새 강제를 만들지 않았는가**(rev1의 §3(a) 처방이 F2를 만든 것이 선례).
② B5/B6의 4상태 `sign`이 **TOST 등록 없이 해석을 떠받치는가**. ③ §7의 세 대조(치환·주입·짝 음성 서명) 중
항등식·빈 서명이 있는가. ④ §6 설계층 도달가능성 표가 **정직한가**. ⑤ §0의 정정이 이번엔 맞는가.
