# R0b — graduated per-type layer-aware & decode-SM knee (Zamba2-2.7B)

> ⚠️ **정정(2026-07-07, 사용자 지적으로 발견)**: 아래 "decode knee≈84 / 잉여 없음 / agnostic over-provisioning 아님"
> 결론은 **방법론적 오류에서 나온 아티팩트**다. `PDMUX_LA_SM_MAP` 구현이 decode를 **독립 green-ctx**에 고정하는데,
> 이는 pdmux prefill 파티션(`stream_group[idx][0]`)과 **조율 안 됨 → 겹침 → 경합**(binary la와 같은 결함, four-pass C).
> **결정적 대조**: agnostic은 decode를 **coordinated** pdmux 쌍(decode 34–54 / prefill 상보 74–54)에서 돌려 **42ms**;
> **같은 54 SM을 미조율 green-ctx**에 두면 **121ms**. ⇒ "84 아래 붕괴"는 **decode SM 요구가 아니라 조율 오버헤드**.
> **잉여는 실재**하고 agnostic이 이미 회수(경decode 가벼울 때 prefill에 최대 74 SM). ⇒ 아래 "graduated도 짐"은
> **미조율 구현이 짐**을 재확인할 뿐, 개념(coordinated per-type)을 반증하지 못함. **coordinated per-type la = event_loop
> 수술 필요, 미구현·미검증**((D) granularity는 논증이지 이 측정 아님). 실전 권고 agnostic은 유지되나 "la 근본 열위"는 미증명.
> 상세: 대화 로그 + `pdmux_context.py:134`(상보 쌍)·`zamba2.py:343`(독립 green-ctx). 원 측정치(표)는 유효하나 해석이 위와 같이 바뀜.


작성: 2026-07-07 (22:00 KST 이후 지연작업). 발단: 사용자 지적 — 은퇴시킨 layer-aware가
실제 아이디어(**per-type graduated SM 할당**)가 아니라 **2단계 이진 근사**(민감층=full / 둔감층=floor 16)
이고, agnostic의 over-provisioning(GPU 저활용)을 보고서가 서술 못 함. → graduated 구현+측정.

## 구현
`models/zamba2.py` `_tgt`에 env `PDMUX_LA_SM_MAP="mamba:M,attn:A"` 추가 (각 layer TYPE을
자기 green-ctx(N)에 고정). green-ctx는 임의 SM 지원(`create_greenctx_stream_by_value`) →
binary는 메커니즘 제약 아님. 하위호환(unset=기존 binary). 하네스 `r0b_graduated_bench.sbatch`.
전부 clean async(`sglang.bench_serving`), in3600/out32, num-prompts 120, rates 1-6, triton, SLO TTFT≤3s·TPOT≤60ms.

## 측정 (goodput@SLO req/s, rep1; measured. goodput=derived good/dur)
| decode SM (mamba/attn) | job | r1 | r2 | r3 | r4 | r1 TPOT | r2 TPOT | 판정 |
|---|---|---|---|---|---|---|---|---|
| **agnostic ~96 dynamic** (R0a) | 834914 | 1.11 | **2.07** | 0.67 | 0.31 | 40 | 42 | 기준 |
| g_96_96 | 835069 | 1.18 | 2.04 | 0.52 | 0.29 | 43 | 44 | ≈ agnostic |
| **g_84_84 (knee)** | 835113 | 1.19 | 1.96 | 0.50 | 0.29 | 43 | 45 | **fine but ≤ agnostic** |
| g_72_72 | 835112 | 0.77 | 0.00 | 0.00 | 0.00 | 56 | 121 | starved |
| g_54_54 | 835044 | 0.77 | 0.00 | 0.00 | 0.00 | 56 | 122 | starved |
| g_54_84 (release mamba, protect attn) | 835208 | 0.86 | 0.00 | 0.00 | 0.00 | 52 | 119 | starved |
| g_40_84 (release mamba, protect attn) | 835209 | 0.80 | 0.00 | 0.00 | 0.00 | 54 | 119 | starved |
| g_54_16 / g_72_16 (release attn) | 835045/835070 | 0.57/0.62 | 0.00 | 0.00 | 0.00 | 60 | 155 | starved |
| la_bin (mamba16/attn~base, R0a) | 834916 | 0.81 | 0.00 | 0.00 | 0.00 | 55 | 129 | starved |

## 판정 (B5): **어떤 fixed/graduated/per-type 할당도 agnostic을 못 이김. 은퇴 강화. 에스컬레이션 불필요.**

핵심 발견:
1. **decode SM knee ≈ 84, 양 타입 공통**(서빙 부하 하). <84 → decode 굶주림(TPOT 119–155ms, goodput 0);
   ≥84 → 정상(~44ms). ⇒ **knee가 낮은 타입이 없음 = 환원할 잉여 없음.** mamba든 attn이든 84 아래로 내리면 스텝 붕괴.
   micro(batch=1) "mamba 둔감→환원가능"은 서빙 batch서 성립 안 함(P1.7 재확인·정량화).
2. **g_96_96 ≈ agnostic** → agnostic 실효 decode SM ≈ 96(≈full). 균일 fixed도 knee 위면 agnostic과 동등.
3. **g_84_84**(knee에 정확히, prefill에 24 SM = agnostic의 ~2배 배정)조차 **agnostic 못 이김**
   (r2 1.96 vs 2.07, r3 0.50 vs 0.67). ⇒ **정적 분할 < agnostic의 동적 per-step 배정.**
4. ⇒ **agnostic은 over-provisioning 아님.** 서빙 batch서 decode가 실제로 ~full SM 필요(knee~84);
   binary la가 진 건 strawman이라서가 아니라 **환원할 잉여가 없어서**(graduated도 동일 이유로 패배).

## 사용자 비판 정리
- (framing) "테스트된 la=이진 근사" → **타당**. graduated 구현·측정으로 보완.
- (substance) "agnostic이 SM 낭비/graduated가 회수 가능" → **측정으로 반증**(knee~84, 잉여 없음, 정적<동적).

## 보고서 반영
sm_policy_report.html §07에 이 표+결론 콜아웃 추가; "any implementation" 문구 완화(직접측정으로 대체);
green-ctx 임의-SM 각주. dev-tree 수정은 src/models/zamba2.py 미러 + env/dev_tree_edits.md §10.

## 미측정 (열린 항목, 결론 불변 예상)
- 진짜 **coordinated 재파티션**(decode 창 ∥ 상보 prefill 파티션, event_loop 수술)은 미구현 — §07(D) granularity상
  per-layer 잉여는 어차피 prefill이 못 잡음으로 논증. 단 knee~84라 애초에 환원할 잉여 자체가 없음(더 근본).
- 다른 모델(NemotronH/Granite)의 knee는 별도(regime 의존); Zamba2 in3600/out32 한정 결과.
