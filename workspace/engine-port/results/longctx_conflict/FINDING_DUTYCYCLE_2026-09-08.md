# 처치 듀티사이클 `f` — **정본 §1-26(B)의 재확인**, 그 대수적 형태, 그리고 열린 스코프

2026-09-08 · **GPU 0** · 발단 = [`audit_ratio_rules_2026-09-08/VERDICT.md`](audit_ratio_rules_2026-09-08/VERDICT.md) 死因 R1 ·
**새 성능 판정 0건 · 정책 순위 변경 0건** · **미감사**

> ## ⛔⛔ 최우선 정정 — **이것은 새 발견이 아니다**
>
> 초판은 이 문서를 *"발견"*이라 적었다. **거짓이다.** 기전은
> **`CONSENSUS §3 항목26(=§1-26(B))`(2026-08-03, claims-auditor)이 이미 등재**했고
> 문장까지 거의 같다:
>
> > *"코드 사실: … `multiplexing_mixin.py:773,792-794`가 prefill 비-in-flight 시 무조건
> > `(0,108)`로 되돌린다 — 즉 이 기판에서 **"decode가 D SM에서 돌았다"⟺"prefill이 동시에
> > in-flight였다"는 같은 사건**이다."*
>
> 같은 항목이 그 귀결까지 적었다 — `g`가 *"decode-SM 탄력도 라벨을 단 **prefill-SM 탄력도**"*
> 이며 **"n으로 해결되지 않는 설계 결함"**이고, 다음 게이트가
> **`PDMUX_STICKY_PARTITION` 구현 → sticky에서 `E1_DECODE_REALIZED≥0.90`이 항등식이 아닌
> 진짜 게이트**라는 것까지. ⇒ **§2의 "스위치" 역시 정본이 이미 처방한 것이다.**
> (줄번호는 그 뒤 이동했다: 현 트리에서 `:922-923`·`:952-953`.)
>
> ★**그래서 이 문서의 실질 내용은 셋뿐이다**: **(1)** 그 기전의 **대수적 형태**
> `f/C = R/(1−R)`와 그것이 *모양 비교 설계*에 대해 갖는 **불가능성**(§1) ·
> **(2)** ★**메인 세션이 등재된 정본(§1-26(B))을 적용하지 않고 `RATIO`를 설계했다는 규율
> 실패**(게이트 #18의 형태 — 저장소가 이미 낸 진단을 새 설계가 안 읽었다) ·
> **(3)** §1-26이 *"이 격자 한정"*으로 스코프를 묶었으므로 **정책 캠페인에서의 듀티사이클은
> 여전히 열린 질문**이라는 것(§5).

> ## ⛔⛔ **§1의 대수는 철회됐다** (감사 死因 R3, 메인 세션 재검증 완료)
>
> `f ≈ ρ_pf`라는 전제가 **같은 데이터에서 반증**됐다: g16 28파일 전수에서
> **`split ∧ prefill_active==0`이 1,643건 = split 구간의 35.8%**(arm별 71.1%→16.1%),
> 반대 방향(`nonsplit ∧ pab>0`)은 **0건**. 기전은 이벤트 루프의 **record-skew**
> (`split_prefill_batch = None` 직후 같은 iteration의 마지막 sync가 발화).
> ⇒ **아래 `f/C = R/(1−R)`와 "등록 좌표에서 0.0089"는 지지되지 않는다.**
> ⚠️단 **코드층 분기 술어**는 결정 시점에 참일 수 있다 — 반증된 것은 **스냅샷 기록 시점의
> 동치**다. ★이 구분은 정본 문장(`CLAIM_EVIDENCE_MATRIX.md:264` 등의 *"⇔ … 반례 0"*)에도
> 걸리며 **doc-steward 소관**이다(이 세션은 정본 미수정).
> ★**§2 sticky 처방은 영향 없다** — §1-26(B)가 이미 낸 것이고 이 대수에 기대지 않는다.

## 1. ⛔철회된 대수 — 원문 보존

**코드 사실 2개**(직접 열람):
- `multiplexing_mixin.py:922-923` — 분할은 `not running_batch.is_empty() and (split_prefill_batch or sticky_partition_enabled)`일 때만 적용된다. **sticky OFF면 `and split_prefill_batch`로 붕괴**한다(같은 줄 주석이 그렇게 명시).
- `:952-953` — prefill 부재 + decode 활성이면 `set_current_stream_idx(real_sm_group_num - 1)` = **무분할 `(0,108)`**.

⇒ **sticky OFF에서 처치 듀티사이클 `f`(=decode-active 시간 중 분할이 걸린 비율) ≈ prefill
in-flight 비율 = prefill 이용률 `ρ_pf`.**

`PREREG_RATIO` 표기로 `ρ_pf = W·R`, `C = W(1−R)`이므로

$$\boxed{\;\frac{f}{C} \;=\; \frac{W R}{W(1-R)} \;=\; \frac{R}{1-R}\;}$$

**`W`가 소거된다.** ⇒ ★**같은 `C`에서 `R`이 다른 두 셀은 `f`가 반드시 다르다.**
`f_A/f_B = [R_A/(1−R_A)] / [R_B/(1−R_B)]`, 등록 좌표에서 **0.0089**.

> ### ⇒ **sticky OFF에서는 "같은 decode 부하, 다른 워크로드 모양" 비교가 원리적으로 불가능하다.**
> 처치가 걸려 있는 시간 자체가 **prefill 부하**이고, prefill 부하는 모양(`R`)이 정하기 때문이다.

이것이 감사 R1이 말한 *"튜닝으로 못 넘는 구조적 교락"*의 정확한 형태이고,
`ratio_power.py`류 검정력 계산으로는 절대 드러나지 않는 종류다(잡음이 아니라 **정의**의 문제).

## 2. 스위치 `PDMUX_STICKY_PARTITION` — ★**§1-26(B) 처방의 재기술**(신규 아님)

같은 줄(`:918-923`)의 주석이 정확히 그 기능을 적는다:

> *"keeps its green-context division **while prefill is idle** instead of falling back to the
> plain unpartitioned group below. With the flag OFF `sticky_partition_enabled` is False and
> the disjunction collapses to the pre-patch `and self.split_prefill_batch`."*

⇒ **sticky ON이면 decode가 도는 한 분할이 유지되므로 `f ≈ 1`이 되고, `f`가 `R`에서 풀린다.**
`PDMUX_R2_POLICY=fixed`에서는 `_sticky_fixed_idx`가 상수라 **런 전체가 한 division에 앉는다**
(`:925-932`). 균질화 config(`sm_group_num: 5`)에서 target 16/44/92 → idx 1/2/3 **유일 매칭**
(`:474-486`, [메인 세션 확인](#4-메인-세션이-직접-확인한-것)).

## 3. ★그래서 무엇이 바뀌나 — **sticky는 계측 선택이 아니라 정책 정의 선택이다**

| | sticky OFF | sticky ON |
|---|---|---|
| decode가 도는데 prefill이 없을 때 | **무분할 108 SM으로 복귀** | **분할 유지**(decode가 D SM을 계속 점유) |
| `f` | `≈ W·R`(모양의 함수) | `≈ 1`(모양과 무관) |
| 이것이 뜻하는 정책 | **기회주의적 분할**(prefill이 있을 때만 나눔) | **고정 SM 분할**(정적 파티션) |

★★**규율 실패의 정확한 형태**: `CONSENSUS §1-26(B)`가 **2026-08-03에** 이 기전과
sticky 처방을 등재했는데, **2026-09-08의 `RATIO` 설계가 그것을 읽지 않고 sticky를 껐다.**
정본은 바뀌어 있었고(`g` 은퇴·인용 금지) 새 설계가 그 조항을 적용하지 않은 것이다.
⇒ 이것은 *"저장소가 같은 진단을 두 번 냈다"*(게이트 #18)가 아니라 **"정본이 이미 바뀌었는데
새 설계가 그 정본을 안 읽었다"** — 더 나쁜 쪽이다.

★**`PREREG_RATIO` §3.3이 sticky를 끈 근거는 rev3에서 승계한 문장**
(*"rev2가 sticky를 산 이유는 실현 게이트를 세우기 위해서였고 ITT는 그 게이트를 안 쓴다"*)
**이었다. 그 문장은 sticky를 계측 손잡이로만 다뤘고, 그것이 동시에 *어떤 정책을 재는지*를
정한다는 것을 다루지 않았다.** ⇒ 이 문서가 그 누락을 등재한다.

⚠️**2F9와의 관계 — 이 문서는 판정하지 않는다.** 직전 회차 감사 死因 **2F9**는
*"**decode-only** + prefill 유휴 + sticky는 PD-mux 운영점이 아니다"*였다. `RATIO` 설계의
부하는 **decode-only가 아니라 실제 서빙 부하**(prefill이 계속 도착)이므로 2F9의 전제가
성립하지 않는다 — **고 보이나 그것은 이 문서의 독해이고, 2F9는 등재된 감사 판정이므로
다음 판본이 감사에 명시적으로 올려야 한다.**

## 4. 메인 세션이 직접 확인한 것 (GPU 0)
| 항목 | 확인 |
|---|---|
| 분할 적용 조건·무분할 복귀 | `multiplexing_mixin.py:918-923`, `:952-953` 직접 열람 |
| sticky ON의 고정 인덱스 | `:925-932`(`_sticky_fixed_idx` 사용), `:279-345`(`_init_sticky_partition`, 경계 인덱스 제외 후 유일 division 선택) |
| 균질화 config에서 target 매칭 유일성 | `_build_r2_policy` `decode_states={16,44}` 가드 통과 · `_r2_decide_idx`(`:474-486`) 정확 일치 ⇒ 16/44/92 → idx 1/2/3 |
| `num_hidden_layers = 56` | Nemotron-Nano-9B-v2-Base `config.json`(pattern 길이 56, attn 4·mamba 27·mlp 25) |
| `f_A ≈ 0.0039` / `f_B ≈ 0.4375` | `forward_count = min(56, 65536//extend)` 산술 |

## 5. ★스코프 관찰 (판정 아님 — 다음 세션이 잃지 않도록 등재)

`PDMUX_STICKY_PARTITION`을 설정하는 스크립트를 전수 grep하면 **프로브·스모크 계열만** 나온다
(`s2_sticky/`·`sticky_smoke/`·`p1_gates/gate2/g2s*`·`kernel_mech/a1_smoke/`·`tc1_model_attrib/probes/`).
**정본 정책 캠페인 스크립트**(`slo_sched/sharegpt_vary_bench.sbatch`·`interactive_bench.sbatch`)에는
**0건**이다.

⇒ 정본의 static-split 결과(HE0·argmax d44 등)가 측정한 "static split"은 **기회주의적 분할**
(sticky OFF)이며, 그 캠페인들에서도 처치 듀티사이클은 prefill in-flight 비율이었다.

★**왜 이것이 §1-26으로 이미 닫히지 않는가**: §1-26(B)는 자기 판정을 **`g` 격자 한정**
(*"이 격자 한정 은퇴, sticky-partition 기판 수정 전 인용 금지"*)으로 묶었다. **정책 캠페인
(HE0·static sweep)에서의 듀티사이클이 얼마였는지는 그 항목이 다루지 않았다.**
⇒ 이 스코프는 **열려 있고**, 기존 telemetry로 **GPU 0에 측정 가능**하다(진행 중).

★★**이것은 관찰이고 판정이 아니다.** 이 사실이 정본의 어떤 결론을 바꾸는지는
**이 문서가 검토하지 않았다** — 정본 캠페인은 부하·워크로드가 달라 `f` 값도 다르고,
HE0는 **같은 기판 위 정책 간 비교**라 듀티사이클이 양쪽에 공통일 수 있다.
**쓰면 안 되는 문장**: *"정본 결과가 기회주의적 분할이라 무효다"* · *"HE0가 흔들린다"* ·
*"정본은 static split을 잰 적이 없다"* — 전부 **미검토**다. 검토하려면 별건 감사가 필요하다.

## 6. 쓰면 안 되는 문장
- *"sticky를 켜면 `RATIO` 설계가 산다"* — `f`는 풀리지만 2F9·B2–B12는 그대로다(§3 ⚠️).
- *"`f = ρ_pf`가 실측됐다"* — **코드 사실 + 산술**이다. 실측은 S0가 한다.
- *"iso-C ∧ iso-f 격자가 존재하지 않는다"* — **sticky OFF에서** 그렇다(§1). sticky ON에서는
  `f ≈ 1`로 상수라 조건이 자명해진다 — 즉 **불가능성은 sticky OFF에 한정**이다.
- §5의 스코프 관찰을 정본 결론에 대한 판정으로 인용하는 것.
