# 세션 핸드오프 — 2026-08-25 (2026-08-24 야간 연속)

## 이번 세션 요약

kernel_mech 트랙의 **Stage 0‴**을 규칙층 3회 + 하네스층 1회 감사를 거쳐 **실제로 완주**시켰고
(job 892556, `KSET_CONSTRUCTIBLE`), NSL 트랙에서 **①(GPU-0 구간 묶기)** 와 **E-A(cap↔mamba↔KV 배너 실측)**
를 완주했다. 그다음 **A1**(엔진 기판) 설계가 **처음으로 死因 2건**을 받고 멈췄다.
**세션 GPU 지출 0.124 GPU-hr · 새 성능 판정 0건 · HE0·정책 순위 불변 · 적대 감사 7회.**
★이 세션의 성격: **측정보다 감사가 많았고, 감사가 매번 실재하는 결함을 찾았다.**

---

## 결정·측정

### 1. Stage 0‴ A0 — **완주** (job 892554 → 892556)
- 규칙 rev1 `NO-GO`(X1–X13) → rev2 `NO-GO`(N1–N13) → rev3 `NO-GO`(R1–R10, ★**"구매 저지 사유 소멸"**,
  권고 **(a) 지금 사라**) → rev4 → 하네스층 `NO-GO`(B1–B15) → rev5 → **제출**.
- **결과**: `KSET_CONSTRUCTIBLE`, basis **`ctx+stream`**. L4 노드 행 **100/100**(`frac=1.000`),
  `greenContextId=4`, `streamId 149 == L3`, **조인율 1.000**(replay 100 / 캡처 0),
  `green: realized 34 = target`. 대조 4겹 전부 건강(**L2′=100**).
  → `workspace/engine-port/results/kernel_mech/stage0ppp/{RESULT_A0_892556_2026-08-25.md, a0_verdict_892556.json}`
- ★★**기판 한정 필수**: `substrate="synthetic_probe_graph"`(노드 5개, green 스트림 1개).
  **엔진 decode 그래프로 전이되지 않는다**(사전등록 §0-1). `frac=1.000`이라 **`Q1_FRAC` 문턱은 미관측**.
- ★★**첫 실행 892554가 더 중요**: 전 축이 동일하게 깨끗한데 `TRACE_TRUNCATED`가 나왔고 원인은
  **sbatch 순서 버그**(`nsys stats`가 기존 `.sqlite` 때문에 exit 1). **하네스가 배관 실패를 실질 판정으로
  바꾸지 않고 측정조건 라벨을 냈다 — 게이트 #21이 실제 실행에서 처음으로 지켜졌다.**

### 2. NSL ① — **구간 묶기**(GPU 0)
- 감사 B1이 지목한 참 술어 `running_bs + |can_run_list| ≥ cap`의 `can_run_list`이 **telemetry 44필드에 없다**
  ⇒ 계산 불가. 대신 **하계 L**(`running_bs ≥ cap`) ⊆ T ⊆ **상계 U**(`running_bs + queue_depth ≥ cap`)로 묶음.
- **`BRACKET_DECIDES`**: 4부팅 전부 L·U가 `X=0.05`의 같은 쪽, **T ∈ [0.080, 0.119]**, 관측 불가분 **≤3.9 pp**.
  ★**P0의 등록 항등식이 바로 그 하계였다** — T를 부분집합으로 대체하고 동치라 선언한 것.
- 추정기 민감도: 시간 가중 0.080–0.088 vs 개수 가중 0.0014–0.0015(**58–60배**), ★**상계도 같이 뒤집힘**.
  → `nsl_lever/{RESULT,PREREG}_NSL_P0_REANALYSIS_2026-08-24.md`

### 3. NSL E-A — **`ARITHMETIC_CONFIRMED`** (job 892561, 0.103 GPU-hr, 요청 0건)
| cell | cap | mamba | **KV 토큰** | 실현 cap |
|---|---|---|---|---|
| c24 | 24 | 24 | 337,039 | 24 |
| c48 | 48 | 48 | 327,601 | 48 |
| c96 | 96 | 96 | 308,727 | 96 |
| ★c48m96 | 48 + mamba 96 명시 | 96 | ★**308,727** | 48 |

- **P2 완전 선형 — mamba 슬롯당 정확히 393 attention-KV 토큰**(두 구간 일치).
- ★★**C1 대조가 결정적**: cap 48인데 KV 예산이 c96과 **오차 0으로 동일** ⇒ **KV 예산은 cap이 아니라
  mamba pool을 따른다.** 코드 산술만으로는 두 해석이 **둘 다 맞아 보였고**, 실측이 갈랐다.
- ⇒ ★**`--max-mamba-cache-size` 고정이 KV 예산을 cap에서 실제로 분리한다** = **③(손잡이 순화)의 처방이 작동함이 실증**.
- ★**B3는 닫히지 않는다 — 크기를 얻었다**: cap 48→24는 KV를 **+9,438 토큰(+2.9%)** 끌고 간다.
  → `nsl_lever/{RESULT_NSL_EA_892561_2026-08-25.md, nsl_ea_banner_892561.json}`

### 4. A1(엔진 기판) — ★★★ **`NO-GO`, 死因 2건**(처음)
- **A2**: E-L2/E-L4가 **같은 커널·같은 이름**이라 가르는 채널이 없다(A0의 고유 커널명에 대응물 없음).
  남은 선택지는 **순환**(Q2ae) 또는 **미정의 시계 다리**.
- **A1**: E-L3를 별도 부팅에 두면 nsys `streamId`가 **리포트 로컬**이라 `stream="match"` **원리적 불가**
  ⇒ `mismatch` 663,552 세계에서 `KSET_CONSTRUCTIBLE` **0건**, 최선 세계가 `CONTRADICTORY_ATTRIBUTION`.
- **런킬러 B1**: `decode_step_count`가 저장소 **전 telemetry에서 0**(재현 확인).
- ★**감사 재설계**: 死因 2건과 차단 절반이 **"3% 듀티로 95번 교차"** 한 뿌리에서 나오고,
  **`PDMUX_STICKY_PARTITION`**(구현 완료·correctness gate 전부 PASS·realized 0.0839→1.0000)이 그 뿌리를 없앤다.
  sticky ON이면 다리가 **시간창이 아니라 부팅으로** 갈린다.
  → `kernel_mech/audit_a1_rules_2026-08-25/VERDICT.md`

---

## ★ 내가 저지르고 정정한 것 (다음 세션이 승계할 것)

1. **NVTX 근거 무효** — rev5–rev8 + B6 + Stage 0″ **6개 문서**가 상속한 `grep -rn nvtx src/` = 0건이
   **틀린 트리**(PD-mux 오버레이). 실제 엔진엔 NVTX가 4개 파일에 있다. ★**결론은 유지되나 이유가 다르다**
   (모듈 forward hook이라 replay 미발화 + layerwise). ★★그리고 **저장소가 이미 답을 갖고 있었다** —
   `reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` V12/V14/V15가 file:line·기전·**올바른 트리 grep**
   까지 등재했고, **kernel_mech 최초 설계(2026-08-15)가 "사전등록 전 필수"라 적은 15분 프로브를 9일간 안 샀다.**
2. **"초판 T6b가 항등식" 오진 철회**(rev2 감사 X2) — `g_contra`가 깬다. 검사↔mutant **짝짓기 실패**였다.
3. **`T14`/`T15`가 `T4`에 함의**(rev2 감사 N2) — 한계 구속 **0세계**인데 "n=276,480 구속"으로 적었다. 두 줄 철회.
4. **하네스가 규칙의 답을 조용히 결정**(하네스층 감사) — `l2_post`가 L4에서 파생돼 **L4가 비면 답을 지운 뒤
   그 답을 말하는 걸 금지**했다. `profile`·`ctx`·`l2_join`도 같은 형태.
5. **기전 서술 19문장 정정**(적대 검증) — 특히 *"decode는 memory-bound"* 는 **정본 인용 금지 문장**이고,
   이 프로젝트는 **phase 간 대역폭 간섭을 측정한 적이 없다**. 측정된 기전은 **§1-24 monolithic prefill stall**과
   **§1-34 split-state residency**다. bank conflict도 **과잉 기각**이었다(HBM 채널 경합은 못 잰 것).
   *"세 판본에 논증이 없다"* 도 **틀렸다**(rev1·rev2·rev3 전부에 위치 논증 있음; 없는 건 **효력 크기 사전 정보**).
6. **A1에서 `N_min` 死코드 + 단위 오독(1 스냅샷=16 step) + `K1_HIGH_AT` 출처 허위** — ★**A0의 X1을
   그 결함을 감사한 뒤 쓴 규칙에서 글자 그대로 재발**시켰다. 거짓 문장 2건은 규칙 파일에서 정정 완료.

---

## 코드·문서 변경 (전부 커밋됨, 워킹트리 clean)

- `kernel_mech/stage0ppp/` — `stage0ppp_a0_rule.py`(`RULE_REV=4`, 1,327,104 세계·27 mutant·17 검사) ·
  `_probe.py` · `_analyze.py` · `.sbatch` · `PREREG_..._2026-08-24.md`(**rev5**) · 결과·자기검사 아티팩트
- `kernel_mech/a1/a1_q3k1_rule.py`(`RULE_REV=1`, Q3 864 / K1 28) + `DESIGN_A1_ENGINE_SUBSTRATE_2026-08-25.md`
- `kernel_mech/NVTX_EVIDENCE_CORRECTION_2026-08-24.md` + rev5·rev6·rev7·rev8·B6·Stage 0″ **6개 문서에 정정 배너**
- `nsl_lever/` — `PREREG/RESULT_NSL_P0_REANALYSIS` · `nsl_p0_reanalysis.py` · `MEMO_NSL_STEP2_DECISION` ·
  `PREREG/RESULT_NSL_EA_CAPBANNER` · `nsl_ea_capbanner.sbatch` · `nsl_ea_analyze.py`
- 판정서 6건: `audit_stage0pp` · `audit_stage0ppp_a0{,_rev2,_rev3}` · `audit_a0_harness` · `audit_a1_rules` ·
  `nsl_lever/audit_nsl_p0_capbind` · `nsl_lever/audit_mechanism_narrative`
- 정본: `PROJECT_STATUS.md` · `reports/CONSENSUS.md`(rev47 + 이번 세션 등재) — **표 5행의 이스케이프 안 된 `|`도 수리**

---

## 열린 항목 / 다음 세션 시작점

★**시작점 한 줄**: **A1을 sticky 기반으로 §2·§5 재작성**(감사 `audit_a1_rules_2026-08-25/VERDICT.md` §⑤ 절차 1–7).

1. **A1 재설계** — sticky ON으로 다리를 **부팅 분리** · `stream`·`green` 축의 **엔진 응답자 명시 등록**
   (`green`은 **서버 프로세스 안에서** `cuStreamGetGreenCtx`+`cuGreenCtxGetDevResource` 배선 필요) ·
   **Q3 채널 결정**(`decode_step_count` 모드 등록 vs `decode_iterations`) — **어느 쪽이든 비영 실증 먼저** ·
   Q3 규칙 rev2(`N_min` 실제 참조·`export=partial` 가드·창별 축·검사 **2개 이상**, 단 **서로 다른 절편**) ·
   K1 rev2(구간 4개·부팅쌍 축·ITL 필드) · D1 정정. → 규칙층 재감사 → 하네스 → 하네스층 감사 → 제출.
2. **NSL ③ 먼저, ② 나중** — E-A가 ③의 처방을 실증했으므로 순서가 뒤집혔다.
   ③ = `--max-mamba-cache-size` 전 셀 고정 + 배너 실측 기록. 그 위에서 ②(사이트별 카운터 엔진 패치).
   ★**②만 사는 것은 값이 없다**(MEMO §4).
3. **감사가 제안한 나머지 실험**: E-B(귀속, ②+③) · E-C(대역폭 — `kernel_mech` 규칙층 통과가 선행) ·
   E-D(`bsweep_regime` 사전등록 rev3 존폐 결정, 현 아티팩트는 B≤23으로 cap 근처 미관측).

---

## 미완·주의

- ★**A1 설계는 死因 2건 상태** — 재설계 전 어떤 문장도 인용 금지.
- ★**A0 결과는 기판 한정** — *"이 기판에서, **엔진 없이 만든 합성 spin-커널 그래프**에 대해"* 를 떼면 안 된다.
  `Q1_FRAC` 문턱은 **미관측**(`frac=1.000`).
- ★**E-A는 B3를 닫지 않는다** — confound에 **크기를 준 것**(+2.9%)뿐.
- ★**NSL 점 술어는 여전히 무효**(B1) — *"cap이 문다/안 문다"* 어느 쪽도 금지.
- **`kernel_mech` rev7·rev8 차단(B1–B5·B7 / C1–C10)은 전부 불변** — A0도 A1도 하나도 건드리지 않았다.
- **실행 중 job 없음.** 루트 `.out`/`.err` 미커밋 대상 아님(결과는 `results/<campaign>/`에만).
- ★**아직 claims-auditor에 안 건 주장**: 이 핸드오프의 요약 문장들(개별 근거는 전부 판정서·결과 문서에 있다).
