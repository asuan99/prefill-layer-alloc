# Code implementation status and remaining validation

최종 갱신: 2026-07-23

| 위치 | 구현 내용 | 공개 interface | Telemetry/test | 남은 위험 |
|---|---|---|---|---|
| `src/multiplex/dual_worker.py` | 두 role thread/task queue, immutable execution context, thread-safe lease와 request phase | `ExecutionContext`, `TrueDualWorkerRuntime` | concurrency/safe-transition unit test | model runner 및 KV lifecycle의 실제 GPU thread safety |
| `src/multiplex/multiplexing_mixin.py` | R1 observer와 R2 true dual 분리, role-thread GPU issue, symmetric async trace, unsafe transition 차단 | `PDMUX_TRUE_DUAL_WORKER`, `PDMUX_TELEMETRY_PATH` | CPU compile; GPU smoke 필요 | SGLang v0.5.10 internal mutable state |
| `src/multiplex/profile.py` | versioned environment/profile, conservative interpolation와 floor | `HybridModelProfileV1`, `DecodeFloorEstimate` | round-trip, compatibility, margin tests | operational profile 미수집 |
| `src/multiplex/controller.py` | fixed/generic/Hybrid policy와 hysteresis/dwell/overload; safe-boundary engine hook | `RuntimeSnapshot`, `SplitDecision`, `MultiplexingPolicy` | controller unit tests | GPU calibration과 admission behavior validation |
| `src/multiplex/telemetry.py` | bounded async JSONL writer와 phase markers | `RuntimeEvent`, `AsyncJsonlTelemetry` | close/drop/write test | CUDA/KV counters 연결 |
| `benchmarks/pdmux_eval/` | W1–W9 trace, SSE replay, paired campaign, p95-ITL goodput/bootstrap | campaign/trace JSON v1 | deterministic/paired/stat tests | real trace adapters와 GPU loadgen validation |
| `scripts/bootstrap/sync_engine_tree.sh` | tracked source sync, role patch, hash manifest | `SGLANG_ENGINE_DEV` | dry-run/runtime import check 필요 | upstream source drift |

## True dual data flow

```text
central admission/KV/split coordinator
          │             │
 prefill task queue   decode task queue
          │             │
 prefill host thread  decode host thread
          │ CUDA event  │
          └── shared GPU/context/weights/KV ──┘
```

Prefill batch 완료 event와 request ownership을 coordinator가 decode-ready 상태로
전환한다. Decode worker만 running-batch 실행을 issue한다. 두 역할은 동일
partition generation의 lease를 동시에 가질 수 있지만 lease가 남아 있는 동안
split 변경은 거부된다.

## Runtime integration gate

True dual은 `parallel_state.py`의 PD-mux role이 thread-local임을 capability
function으로 확인한다. patch가 없으면 실행을 중단한다. 다음 GPU 검증 전에는
기본값이 off다.

1. one-request correctness와 shutdown
2. prefill-only/decode-only stream issue
3. concurrent fixed D24/D44 token equality
4. cancellation, OOM, KV alloc/free
5. CUDA Graph replay와 attention backend stability
6. 30분 soak test에서 deadlock/race 없음

실패 시 R1 observer/legacy에는 영향을 주지 않으며 Claim D를 측정하지 않는다.
