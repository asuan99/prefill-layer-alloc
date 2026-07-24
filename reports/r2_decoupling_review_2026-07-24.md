# R2 "decoupling" 구현체 코드 리뷰 — 무엇을 실제로 분리하는가

작성: 2026-07-24. 읽기 전용 리뷰. 코드 변경 없음. 근거는 전부 file:line.
대상: `PDMUX_TRUE_DUAL_WORKER=1` R2 경로 + R2 policies(fixed/generic/hybrid) + admission 골격.

정본 맥락(PROJECT_STATUS.md §확정결과 3, MEMORY realtrace): 死因 얽힘은
`decode 굶김 → 잔류시간↑ → running batch 포화 → prefill admission 차단 → TTFT 폭발`.
진짜 headroom(+16%)은 별도 device pool(92 prefill SM + 24 decode SM = 116 > 108)에서만 열리고,
단일-GPU dual-worker 천장 = coupled ceiling(+2%).

파일 기준:
- `workspace/engine-port/src/multiplex/dual_worker.py`
- `workspace/engine-port/src/multiplex/multiplexing_mixin.py` (mixin)
- `workspace/engine-port/src/multiplex/controller.py`
- `workspace/engine-port/src/patches/pdmux_thread_local_role.patch`

---

## Q1. 무엇을 분리하는가 / 무엇을 분리하지 않는가

### 분리하는 것 (control plane) — CONFIRMED

- **두 개의 long-lived host issue OS 스레드.** role별 `RoleWorkerThread`
  (dual_worker.py:284, 383-384), 스레드 이름 `pdmux-prefill-worker` /
  `pdmux-decode-worker` (dual_worker.py:297-298), daemon.
- **role별 task queue.** 각 `RoleWorkerThread`가 자기 `queue.Queue`를 소유
  (dual_worker.py:294). condition-backed 작업 큐.
- **immutable ExecutionContext.** role/stream_index/prefill_sms/decode_sms/generation을
  담은 frozen dataclass (dual_worker.py:137-148), submit마다 새로 생성(dual_worker.py:475-483).
- **thread-local PD-mux role.** ContextVar 패치(pdmux_thread_local_role.patch:17-25)로
  module-global `_ENABLE_PDMUX_P_TP`를 ContextVar화. `_activate_role_context`가
  스레드 안에서 `set_pdmux_status(role is PREFILL)`를 걸어(mixin:348-354) 두 스레드가
  동시에 서로 다른 prefill/decode role을 유지하게 함. **이것이 true-dual의 핵심 enabler.**
- **CUDA stream 이슈 분리.** prefill_stream / decode_stream 각각에 submit(mixin:927, 996).
  단, 이 두 스트림은 **legacy event_loop_pdmux에도 이미 존재**하던 green-context stream
  group(mixin:785-786)이다. dual-worker가 새로 만든 건 스트림이 아니라 **그 위에 올리는
  host 이슈 스레드**다.

### 분리하지 *않는* 것 (data / resource plane) — CONFIRMED, 전부 공유

- **(a) running batch(`max_running_requests`) = 공유.** 두 role이 단일 scheduler 속성
  `self.running_batch` / `self.split_prefill_batch` 위에서 동작한다. decode future는
  `decode_batch = self.running_batch`를 클로저(mixin:920), prefill future는
  `self.split_prefill_batch`를 클로저(mixin:983). 완료 prefill은 **같은** running batch로
  in-place 병합: `self.running_batch.merge_batch(self.split_prefill_batch)`(mixin:1050)
  또는 `self.running_batch = self.split_prefill_batch`(mixin:1052). `max_running_requests`는
  단일 scheduler cap(mixin:221, 556). **role별 batch 없음.**
- **(b) KV cache pool = 공유.** `token_to_kv_pool_allocator`,
  `req_to_token_pool`(및 그 `mamba_pool`)는 scheduler-wide 단일 객체(mixin:271-274).
  role별 KV 파티션·dual allocator **없음.**
- **(c) SM(green-context 파티션) = 단일 공유 whole-phase 파티션.** `SharedGpuArbiter`는
  **하나의** `stream_index`와 generation만 추적(dual_worker.py:162-163). 두 worker가 같은
  index로 `select_partition(stream_idx)` 호출(mixin:919, 982). 코드 주석이 명시적:
  "Tracks one shared whole-phase GPU partition for both workers"(dual_worker.py:151).
  `(prefill_sms, decode_sms)` 쌍은 합 ≤ 108이고 **하나의 index로 joint 선택**된다 —
  SM은 두 role "사이에" 쪼개지지만 **독립 소유가 아니라 단일 coupled index**로 묶여 있다.
  이는 정확히 §1-20의 coupled 단일-GPU 천장이지 92+24=116의 별도-풀 regime이 **아니다.**

**판정:** R2 true-dual은 control-plane(스레드/큐/role ContextVar/스트림 이슈)만 분리하고
resource-plane(batch/KV/SM)은 전면 공유한다.

---

## Q2. §1-4 얽힘을 깨는가 — 부분(control-plane만), 死因 미접촉

死因 얽힘은 **공유 running-batch + 공유 KV**에 산다: decode 굶김 → 잔류↑ →
running_batch가 `max_running_requests` 도달 → prefill admission 차단 → TTFT 폭발.

R2 dual-worker는:
- running_batch를 여전히 공유(mixin:1050-1052),
- KV를 여전히 공유(mixin:271-274),
- `max_running_requests` 단일 cap 유지.

admission gate 자체도 동일한 공유-batch gate다: `update_split_prefill_batch` →
`get_new_batch_prefill()`(mixin:764)가 공유 running batch에 대해 admit한다.
`DualWorkerState`는 심지어 block 이유로 `"shared_batch_capacity"`를 명시적으로 관측
(dual_worker.py:591-598) — 즉 코드 스스로 병목이 공유 batch임을 인정한다.

얽힘이 사는 substrate가 **전혀 바뀌지 않는다.** 얽힘은 resource-coupling 현상
(batch 용량 + KV 점유)이지 control-plane 현상이 아니므로, control-plane만 분리하는
R2 dual-worker는 **구성상 이 얽힘을 깰 수 없다.**

부가: R2 admission latch(`r2_admission_limited`)는 decoupling이 아니라 **새 proactive
throttle**이다. overload_streak≥2일 때 prefill admission을 거부(controller.py:183,
mixin:758-759). 이는 backlog 압력을 줄이지만 **prefill을 조여서(TTFT 손해)** 그렇게 하는
것 — coupled substrate 위의 policy lever지 decoupling이 아니다.

**Claim D 판정:** 이 구현으로 Claim D(true dual-worker가 coupling 감소, H-Architecture)를
측정하면 **死因을 건드리지 못한다.** 관측되는 win/loss는 host-thread overlap(control-plane)에
귀속될 뿐 resource decoupling이 아니다. §1-20이 진짜 headroom은 별도 device pool에서만,
단일-GPU 천장은 coupled ceiling이라 했고, 이 구현은 batch/KV/SM 전부 공유하는 단일-GPU다 —
**구성상 coupled ceiling에 앉아 있다.** 따라서 Claim D를 "얽힘 깨기" 주장으로 유의미하게
측정할 수 없으며, 실험은 "host-thread overlap"과 "decoupling"을 confound하고 coupled
천장을 초과할 수 없다.

---

## Q3. 단일-GPU 두 스레드 vs 진짜 disaggregation — 단일-GPU 확정, state transfer 전무

- **단일 GPU, 두 host 스레드.** `_activate_role_context`가 두 role 모두
  `torch.cuda.device(self.gpu_id)`(mixin:352) — 동일 device. 두 번째 device도, device
  pool도 없다.
- **state-transfer 경로 전무.** src 전체 grep: `disagg` / `state_transfer` /
  `transfer_state` / `migrate` / `kv_transfer` / `send_kv` / `recv_kv` = **NONE**.
- prefill→decode 핸드오프는 **같은 device 위 in-place batch merge**(mixin:1050-1052)이지
  전이가 아니다.

**판정:** disaggregation 없음, state transfer 없음. 코드에 후자는 **전무**(설계 흔적조차 없음).

---

## Q4. Hybrid 상태 전이(Mamba conv + SSM recurrent) — 미구현

disaggregation 경로가 없으므로 KV 전이도, 따라서 mamba conv / ssm recurrent state 전이도
없다. multiplex 코드 grep: `conv_state` / `ssm_state` / `recurrent_state` /
`mamba_state` = **NONE in multiplex**. 코드가 mamba를 아는 건 **telemetry/policy 용도뿐**:
`req_pool.mamba_pool` 점유 읽기(mixin:274), `kv_mamba_occupancy` telemetry(mixin:420).
conv/ssm state 마이그레이션 처리는 0.

**판정:** hybrid disaggregation 미구현 (스캐폴딩조차 없음).

---

## Q5. Correctness 게이트 상태

1. **CPU unit 회귀 — PASS.** `python -m unittest discover -s .../tests` = 23 tests OK
   (fake batch/event 기반, GPU 없음). test_dual_worker는 concurrency 프리미티브만 검증
   (스레드 이름, 파티션 전환 거부, in-flight event block; test_dual_worker.py:138-208).
2. **GPU correctness(legacy vs 새 경로 디코딩 동치) — 미충족.** 해당 테스트 **없음**, 증거
   **없음**. gate #2 unmet.
3. **thread-local role 패치 fail-fast — CONFIRMED wired.** `pdmux_role_is_thread_local()`
   capability probe(패치 line 28-30) + mixin:80-84에서 미적용 시 RuntimeError. 미패치
   dev-tree(심볼 부재) → import/AttributeError → fail-fast. presence gate로 성립.
4. **admission starvation latch 버그 — CONFIRMED (stale True latch, clear 경로 부재).**
   `r2_admission_limited`는 `_r2_decide_idx` 안에서만 set(mixin:288). `_r2_decide_idx`는
   `self.split_prefill_batch`가 truthy일 때만 호출됨(gate mixin:844-849, 특히 847).
   그런데 latch가 True이면 `update_split_prefill_batch`가 새 split_prefill_batch 생성을
   거부(mixin:758-759). ⇒ latch가 한 번 True가 되고 현재 split batch가 None으로 배수되면
   `_r2_decide_idx`를 다시 돌릴 경로가 없어 **latch가 True로 고착 → prefill admission 영구
   차단.** False로 되돌리는 곳은 초기화(mixin:111)뿐. **clear 경로 없음.** 어떤 Claim D
   측정도 이 prefill 정체가 오염시킨다.
5. **server seed — R2 wiring에서 미처리.** multiplex 코드에 seed 고정 없음. R1 postmortem이
   "server seed가 달랐다"를 confound로 지목(PROJECT_STATUS R1 판정)했는데 R2에서도 동일 위험
   잔존 → 캠페인에서 arm 간 동일 seed 고정 필수.
6. **telemetry 대칭성 — 구조적으로 대칭, 경험적으로 미검증.** 모든 arm이 동일
   `AsyncJsonlTelemetry` 경로 사용(mixin:103, 423), architecture 필드로 legacy/r1_observer/
   true_dual 구분(mixin:404-408)하되 writer 경로는 동일. 그러나 observer effect <3% gate는
   paired on/off 실측이 필요 — **그 측정 없음**(아래 참조).

**R2가 GPU correctness gate를 통과한 이력? NO.**
- `results/r2_eval/` 디렉토리는 **존재하지 않음**(스크립트 `scripts/r2_eval/`만 존재:
  engine_bench_runner.sh, generate_campaign.sh, policy_adapter.sh, r2_eval.sbatch).
- `architecture=true_dual` telemetry가 **어디에도 없음**(전체 jsonl grep = NONE).
- ⇒ true dual-worker는 **GPU 실행 아티팩트를 한 번도 생산한 적 없다.** results/r2_eval 실제로 비어있음(미생성) 확인.

---

## Claim D를 유의미하게 측정하려면 무엇이 선결인가

1. **admission latch 버그 수정.** split batch 배수 시 latch clear 경로 추가 또는
   decode-only iteration에서도 policy 재평가(mixin:288/758/844-849). 안 고치면 prefill
   정체가 모든 측정을 confound.
2. **GPU correctness 동치 테스트 추가.** legacy vs true-dual 동일 입력 디코딩 출력 일치
   (gate #2). 현재 전무.
3. **Claim D의 주장 범위를 먼저 확정.**
   - "host-thread overlap이 control-plane coupling을 줄인다"라면: single-thread same-stream
     vs dual-thread를 명시적 control로 두고 측정하되 **"얽힘 깨기"로 팔 수 없다**(얽힘은
     공유 batch/KV에 산다). 천장은 coupled ceiling.
   - "얽힘 깨기 / +16% headroom"이라면: running-batch + KV + SM을 실제로 파티션(별도 device
     pool)해야 함 — **부재.** 이는 disaggregation + hybrid(mamba conv/ssm) state transfer를
     요구하며 **둘 다 미구현.**
4. **server seed를 arm 간 동일 고정** + **telemetry observer effect <3%를 paired on/off로
   실측**(현재는 구조적 대칭만, CI 0 포함 여부 미검증).
5. **cudagraph-ON(운영점) 호환 확인.** true-dual은 run_batch를 future로 submit하고 경계에서
   synchronize한다(mixin:1018, 875-876). cudagraph 캡처가 이 경로에서 여전히 성립하는지
   미검증 — 운영점 진입 여부가 결정적.

---

## 요약 판정표

| # | 항목 | 판정 |
|---|------|------|
| 1 | 분리: host 스레드 / role 큐 / role ContextVar / 스트림 이슈 | CONFIRMED (control-plane) |
| 1 | 미분리: running batch / KV pool / SM 파티션 | CONFIRMED 전면 공유 |
| 2 | 死因 얽힘을 깨는가 | 아니오 (control-plane만, substrate 불변) |
| 2 | Claim D가 死因을 건드리는 실험인가 | 아니오 (overlap과 decoupling confound, coupled 천장) |
| 3 | 진짜 disaggregation / state transfer | 전무 (단일 GPU, in-place merge) |
| 4 | Hybrid(mamba conv/ssm) 상태 전이 | 미구현 |
| 5 | CPU 회귀 | PASS |
| 5 | GPU correctness 동치 | 미충족 (테스트·증거 전무) |
| 5 | thread-local role fail-fast | CONFIRMED wired |
| 5 | admission latch stale-True 버그 | CONFIRMED (clear 경로 부재) |
| 5 | server seed 처리 | R2 wiring에 없음 |
| 5 | telemetry 대칭 | 구조적 대칭 / 경험적 미검증 |
| 5 | R2 GPU correctness gate 통과 이력 | 없음 (results/r2_eval 미생성, true_dual telemetry 전무) |
