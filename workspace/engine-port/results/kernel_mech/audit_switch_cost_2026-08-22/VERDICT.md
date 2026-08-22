# 감사 판정서 — `DESIGN_SWITCH_COST_2026-08-22.md` (2026-08-22, claims-auditor, 게이트 #34 1단)

**판정: `NO-GO` — 차단 D1–D6 · caveat C1–C7.** 게이트 #55 보존본.
★**이 파일은 늦게 만들어졌다** — `STEP0_SWITCH_GAP_2026-08-22.md`가 이 경로를 인용하는데 파일이
없었다(2026-08-22 git-committer가 교차인용 끊김으로 발견). **게이트 #55를 만든 세션이 같은 날
스스로 어겼다** — 교훈 파일에 등재 대상.

## 차단 6건

- ★**D1 전제가 사실오류 + 무료 대안 미고려.** 설계 §1의 *"drain 자체를 잰 아티팩트는 0건"* 은
  **거짓**. `holb_probe.py:428-430`이 직전 decode end → 이번 decode start를 **device CUDA
  이벤트**로 이미 재고, `:203`이 `stream_key`를 함께 적어 **전환 라벨을 이미 들고 있다**.
  운영점 아티팩트가 `results/p1_gates/gate2/*.holb.jsonl`에 존재(cudagraph-ON·pdmux-ON).
  ⇒ 정확한 서술은 ★**"그 아티팩트를 이 질문으로 분석한 적이 0건"**(**교훈 #18**).
  **GPU 0 재분석을 본 캠페인의 선행 단계로 등재하라.**
- ★**D2 케이던스가 추정량을 만드는 경로 미차단.** `:1199`가 **매 반복** `decode_stream.synchronize()`
  를 하므로 전환의 marginal은 `prefill_stream.synchronize()` 하나 ⇒ `c`는 상수가 아니라
  **"전환 순간 in-flight prefill 청크의 잔여 시간"** 이고 그 분포는 **전환을 prefill span의 어디에
  꽂느냐**의 함수다. 실측: 상위 5%가 총합의 86–88%. OLS 가중에서 **`N=2` 하나가 기울기의 50%**.
  설계의 차단(큐·배치 등가)은 이 층에 **닿지 않는다**. 게다가 강제 교대는 같은 두 index를 왕복해
  **`c`를 아래로 편향**시킨다(정본 확증 방향).
- **D3 3점 `1/N` 선형성 점검이 공허참 논리곱**(잔차 자유도 1, 검정력 ≈0) ⇒ `SWITCH_COST_MEASURED`
  조건이 사실상 첫 항으로 붕괴. **게이트 #42 재발.**
- **D4 ⑤/⑥의 코드 전제 오류.** `update_decode_attn_backend`는 `model_runner.py:2624-2625`
  **속성 대입 1줄**, cudagraph는 `(idx,bs)` 사전 캡처 ⇒ 가를 것이 구조적으로 없고 "cold backend"
  전제도 코드와 어긋난다.
- **D5 `alternate` 범위 불변이 검사 불가 술어.** `SplitDecision.admission_limited`(`controller.py:65`)
  → `r2_admission_limited`(`:400`) → **prefill 수락 차단**(`:933`) ⇒ 정책 객체는 **부하를 바꾸는
  두 번째 출력 채널**을 갖는다. 필드 수준 불변 + 변이 테스트로 재작성 필요.
- **D6 label(k) 채널 불완전·편향 단방향.** `split_transition`은 (a) `_r2_decide_idx` 안에서만
  방출 (b) **결정 이벤트**라 전환 없이도 방출 (c) `adjust_stream_group` 경로와 `:911-912`의
  full-SM 강등은 **무기록** (d) `decode_step_count` 필드가 **존재하지 않는다**(조인 키 부재)
  (e) 텔레메트리가 큐 포화 시 **이벤트를 버린다**. (c)(e)의 오라벨은 전부 switch→no-switch 방향
  ⇒ **`c`를 작게 만든다**. ★수리 = **`stream_key`로 라벨링**하고 텔레메트리는 교차확인으로만.

## caveat 7건 (요지)

C1 `transient`가 경계 내 호스트 비용을 못 뺀다(실측 550 ms급 사건이 `other_stream_fw_ms=0`) ·
C2 "부팅 분산이 상쇄된다"와 "부트스트랩 1차 단위=부팅(n=4)"의 자기모순 · C3 블록 길이·경계 수
미기재인데 예산 선확정 · C4 3×2 교차의 대조균형·분석 모형 미기재 · ★C5 **§8-2의 상금 천장
논증이 표적을 잘못 겨눔**(`PDMUX_LA_COORD_OPT`는 **이미 구현**돼 있고 `event_loop_pdmux_coord`
(layer-type 死 트랙) 전용) · C6 ⑦의 20% 문턱이 `c` 상대값이라 순환 · ★C7 **정본층 overclaim 2건**
(`venue_positioning.md:178`·`CLAIM_EVIDENCE_MATRIX.md:512`의 *"switch_count≈0 … 이미 직접 계측"* —
실제 5/8/18/21이고 직접 계측된 건 CPU뿐) + §1-8 근거 4행 중 2행이 **폐기 벤치**(stationary r8,
방법론 게이트 #2) ⇒ doc-steward 이송.

## 권고 경로 (감사가 지정한 순서)

**Step 0(GPU 0·패치 0)** 기존 `*.holb.jsonl`을 `stream_key`로 갈라 재분석 → **Step 1** `alternate`
노브 1건만(필드 수준 불변 + 변이 테스트) → **Step 2** 본 캠페인 → **Step 3** 정본 진술.
★**"엔진 패치 2건에는 값하지 않고, 1건에는 값한다"** — NVTX 패치는 **불필요**(HOLB 채널로 충분).

## 후속 (2026-08-22)

Step 0이 집행됐고 그 결과도 **`REFUTED`** 됐다 → [`../audit_step0_2026-08-22/VERDICT.md`](../audit_step0_2026-08-22/VERDICT.md).
결론: **인덱스 변경 자체 `s ≤ 0.040 ms`**, 진짜 기전은 **prefill 생애주기 경계**,
**residency가 82배** ⇒ 감사 권고는 *"switch-cost 트랙을 GPU 0으로 닫고 예산을 쓰지 마라"*.
