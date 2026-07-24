# Hybrid LLM PD-mux dual-worker implementation

최종 갱신: 2026-07-23. 현재 판정은
[`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)를 따른다.

## 두 경로의 구분

| Mode | 구현 | 용도 |
|---|---|---|
| `PDMUX_DUAL_WORKER=1` | 기존 scheduler queue/batch view와 logical observer | R1 재현·observer-effect 측정 |
| `PDMUX_TRUE_DUAL_WORKER=1` | 두 long-lived host issue thread, role task queue, explicit execution context | R2 architecture 가설 검증 |

R1은 독립 worker가 아니며 `job_862512`를 architecture result로 사용하지 않는다.
R2 코드의 존재도 Claim D의 성능 증거가 아니다.

## R2 ownership

```text
Request stream
      │
central admission / KV / split coordinator
      │
  ┌───┴──────────────────┐
  │                      │
prefill task queue    decode task queue
prefill host thread   decode host thread
prefill CUDA stream   decode CUDA stream
  │      handoff event   │
  └──────── shared GPU ──┘
```

분리되는 상태:

- role task queue, condition/wake-up와 host issue thread
- prefill/decode CUDA stream activation
- role execution context, busy time, task count와 error state
- in-flight resource lease

공유되는 상태:

- process, CUDA device/context, model weights, HBM/L2
- central request/KV/admission lifetime
- one selected `(prefill SM, decode SM)` partition generation
- model runner와 graph/backend cache

현재 제거한 coupling은 host issue queue와 role wake-up이다. KV/admission/split
coordination 및 물리 GPU interference는 의도적으로 공유된다.

## Safety invariants

- 모든 role callback은 immutable `ExecutionContext`를 받는다.
- 두 role은 같은 partition generation lease만 동시에 가질 수 있다.
- active lease가 하나라도 있으면 split transition을 거부한다.
- prefill 완료 event 이후에만 coordinator가 request를 decode-ready로 전환한다.
- module-global PD-mux role은 `ContextVar` patch로 thread-local화한다.
- patch capability가 없으면 true dual startup을 실패시킨다.
- layer boundary에서는 split을 바꾸지 않는다.

SGLang model runner/KV lifecycle의 실제 thread safety는 GPU smoke 및 soak test가
통과하기 전까지 미확정이다. 기본 mode는 legacy다.

## Telemetry

`PDMUX_TELEMETRY_PATH`는 architecture flag와 독립적이다. 모든 baseline에서
동일한 bounded queue와 writer thread를 사용해 scheduler thread의 file I/O를
제거한다. phase marker, architecture, queue/batch, split, role task/busy time,
dropped-event count를 기록한다. KV와 CUDA replay/overlap counter는 engine hook
연결 후 추가 검증한다.

## Validation order

1. legacy telemetry off/on 및 R1 observer off/on의 paired overhead
2. one-request와 fixed D24/D44 token correctness
3. cancellation, OOM, KV alloc/free, shutdown
4. CUDA Graph replay와 concurrent issue
5. 30분 deadlock/race soak
6. legacy fixed 대 true dual fixed의 architecture experiment

상세 acceptance criterion은
[`paper/EXPERIMENT_ROADMAP.md`](paper/EXPERIMENT_ROADMAP.md)에 있다.
