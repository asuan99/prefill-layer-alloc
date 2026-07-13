# cudagraph 환경 전환 재측정 — engine 기판을 사다리 위로

작성: 2026-07-13. 동기: [system_vs_engine_vs_sim.md](../../reports/system_vs_engine_vs_sim.md) §5의 "가장 결정적
후속" = **cudagraph 가능 환경서 prefill-side layer-aware가 goodput으로 전환되는지**. 기존 전 서빙 캠페인은
`--disable-cuda-graph`로 측정됐고(=fidelity-ladder의 "engine=lower bound·비관 기판"), 그 (B)축을 실제로 걷어낸다.

하네스 `probe1_plain.sbatch`·`probe2_pdmux.sbatch`·`probe3_cg.sbatch`. 서버 로그 `srv_*.log`. Zamba2-2.7B, A100-80GB, triton backend, clean async `bench_serving`.

---

## TL;DR

1. **"pdmux+mamba cudagraph 불가"는 틀렸다** — 근본 불가가 아니라 **하네스가 수동으로 끈 것**. py3.14/torch2.9 +
   hybrid(mamba)+triton + **pdmux green-ctx**서 캡처·replay 정상(Probe 1·2).
2. **decode wall 제거**: TPOT plain 62→13.5ms, **pdmux 41→12ms**(rate2). 전 정책 decode ~3–4× 가속.
3. **정책 랭킹은 유지·선명해짐**: prefill-bound=**tuned-d24 ≈ SLO-v7b ≫ agnostic**; decode-heavy=**SLO-v7b ≈ tuned-d44 > agnostic**.
   **SLO-aware = 여전히 무튜닝 generalist**(양 regime 1–2위). 절대 goodput은 no-cudagraph 대비 ~1.5–2×↑.
4. ★**layer-aware 전환 = 실패, 반증 강화**: lacoord의 prefill TTFT 우위는 실재(697ms ≈ d24 615)하나 **decode가
   SLO 붕괴**(gp rate4 **0.027**). 이유 = ★**cudagraph ⊥ sub-step layer-aware**(아래). fidelity-ladder의 "cudagraph가
   구제할지 모른다"는 **반대로 판명** — cudagraph가 격차를 넓힌다.

---

## Probe 1 — plain(pdmux無) cudagraph OFF vs ON: 캡처되나·벽 내려가나

캡처 성공: `Capture cuda graph begin … bs [1,2,4,8,12,16,24,32,40,48] … end. 3.97s`, 에러 0.

| in2000/o96 plain | cgOFF | **cgON** | 개선 |
|---|---|---|---|
| rate2 Median TPOT | 62.70 (SLO 초과) | **13.51** | 4.6× |
| rate2 P99 TPOT | 142.2 | 39.1 | 3.6× |
| rate2 Median ITL | 39.1 | 10.3 | 3.8× |
| rate4 Median TPOT | 82.41 | 54.04 | 1.5× |

→ 벽 실재·제거 가능 확인. (주: prior "170–205ms"는 NemotronH/구-client 아티팩트; clean Zamba2 plain cgOFF=62–82ms.)

## Probe 2 — pdmux(agnostic) cudagraph ON vs OFF: green-ctx와 호환되나

★캡처 성공(pdmux 하), SANITY 정확, **크래시·illegal-memory 0**. green-ctx SM 분할이 graph replay를 안 깬다.

| in2000/o96 pdmux | cgOFF | **cgON** | 개선 |
|---|---|---|---|
| rate2 Median TPOT | 40.97 | **12.00** | 3.4× |
| rate2 Median ITL | 37.40 | **11.82** | 3.2× |
| rate4 Median TPOT | 39.03 | **18.86** | 2.1× |

→ **핵심 관문 통과**: pdmux 표준 경로(`event_loop_pdmux`→`run_batch` decode)는 cudagraph를 탄다.

## Probe 3 — core 정책 재측정 (cudagraph ON, 양 regime)

모두 cgON. core(agn/d24/d44/slo)=`event_loop_pdmux`(run_batch=cudagraph). **lacoord=`event_loop_pdmux_coord`
(`forward_split_decode`=eager, cudagraph replay 불가) → CAVEAT: decode 미가속**. goodput@SLO(TTFT≤3s∧meanITL≤60ms), num-prompts 100.

### PREFILL-BOUND in3600/o32
| 정책 | gp r2 | gp r3 | gp r4 | TTFT50 r2 | TPOT50 r2/r3/r4 |
|---|---|---|---|---|---|
| **d24 (tuned)** | **2.276** | **1.839** | **0.857** | 615 | 20.7 / 31.6 / 37.4 |
| **SLO-v7b** | 2.130 | 1.148 | 0.727 | 774 | 16.9 / 28.6 / 43.1 |
| agnostic | 0.989 | 0.268 | 0.207 | **3025** | 17.8 / 20.6 / 20.6 |
| **lacoord**(caveat) | 1.313 | **0.162** | **0.027** | 697 | **47.3 / 98.6 / 99.5** |

- **tuned-d24 최적, SLO 근접 2위**(기존 결론 유지). **agnostic은 prefill 과소공급**으로 TTFT 3s 초과(붕괴) — decode엔 SM 후하나 prefill 굶음.
- ★**lacoord**: TTFT 697ms(=d24 615 수준, agn 3025의 1/4)로 **prefill 우위 실재**하나, **decode TPOT 47→99ms·ITL 78ms로 SLO 붕괴** → gp r4 0.027. prefill 이득이 decode 붕괴로 상쇄 이상.

### DECODE-HEAVY in2000/o96
| 정책 | gp r2 | gp r3 | gp r4 | gp r6 |
|---|---|---|---|---|
| **SLO-v7b** | 2.266 | **3.323** | **3.626** | 1.019 |
| **d44 (tuned)** | 2.263 | 3.311 | 3.200 | **1.383** |
| agnostic | 2.256 | 2.878 | 1.721 | 0.682 |

- **SLO-v7b가 r2–4 최고, d44 근접**; r6(과포화)선 d44>slo. **SLO는 무튜닝으로 best-static 매칭/상회**(decode-heavy). 전 TPOT<60ms(cudagraph)라 decode SLO는 고rate까지 여유, goodput은 TTFT가 지배.

---

## ★핵심 원리 — cudagraph ⊥ sub-step layer-aware

cuda graph 캡처는 **고정 스트림 위 고정 op 시퀀스**를 요구한다. layer-aware의 본질 = **sub-step green-ctx
재분할**(decode를 층타입 창 ~19개로 쪼개 창 사이 SM 재배분). 이건 **단일 그래프로 캡처 불가**:
- coord 경로는 창마다 `forward_split_decode`(eager)를 호출 → 캡처된 그래프를 **replay 못 함** → decode 영구 eager.
- agnostic/tuned/SLO는 **step당 split 1개** → decode 전체가 고정 시퀀스 → **깨끗한 그래프 → cudagraph 3–4× 획득**.

⇒ layer-aware는 (D) drain 비용에 지는 데 **더해**, 가장 큰 decode 최적화(cudagraph)를 **구조적으로 포기**한다.
Probe 3가 실증: lacoord decode ITL 78ms vs agn/SLO 20ms(=cudagraph 격차 그대로). **layer-aware의 정의적
메커니즘(sub-step type-switching)이 cudagraph를 foreclose한다.**

## fidelity-ladder 열린 항목 판정 — RESOLVED (negative)

[system_vs_engine_vs_sim.md](../../reports/system_vs_engine_vs_sim.md) §3 최상위 open("prefill-side layer-aware가
cudagraph 환경서 goodput 전환되나") **닫힘 = NO**. 세부:
- **(B) decode wall은 substrate CHOICE였다**(한계 아님): core 정책은 cudagraph로 벽 넘음 → engine을 사다리
  위로(Bullet 근접) 이동 성공. **fidelity-ladder 재분류: (B)를 "artifact"→"기존 측정의 flag 선택"으로 격하.**
- **단, layer-aware만은 그 벽을 못 넘는다** — 환경이 아니라 **자기 메커니즘 탓**(cudagraph 비양립). 따라서
  **"cudagraph가 layer-aware를 구제"는 반증**; 오히려 core가 벽을 넘고 layer-aware는 남아 **격차 확대**.
- **prefill TTFT 기전은 여전히 확증**(lacoord 697<<3025) — 죽는 건 decode side. prefill-only 이득을 살리려면
  decode를 cudagraph 표준 경로로 두고 **prefill에만** 층-예약을 하는 형태가 필요(= sub-step 아님; [[prefill-layer-alloc-status]]
  §14 "layer-type-aware 예약"). 그건 별도 설계이며 본 coord 구현과 다름.

## 함의 (실전)
1. **운영점은 cudagraph-ON**: 전 정책 decode ~3–4× 빠름·goodput ~1.5–2×↑. 기존 서빙 수치는 전부 하한.
   **정책 결론(SLO=generalist·tuned=per-regime 최적·agnostic=prefill 취약·layer-aware=死)은 랭킹 불변**(cudagraph서 재확인).
2. **layer-aware 최종**: sub-step 형태는 cudagraph 시대에 **더** 불리(死 강화). 유일 잔존 가능성=**prefill-only 층예약 +
   decode는 cudagraph 표준경로**(미구현·별 트랙).
3. **SLO-aware**: cudagraph 하에서도 무튜닝 cross-regime 우위 재확인. 단 임계값(TPOT_SLO/HI/LO)은 no-cudagraph
   TPOT(~40ms)에 맞춰 튜닝됐으므로 **cudagraph TPOT(~13–40ms)에 재튜닝하면 추가 이득 여지**.
