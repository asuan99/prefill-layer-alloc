# Hybrid LLM PD-mux dual-worker implementation

## Scope

This is the first implementation step after the existing PD-mux baseline. It
separates phase ownership inside one scheduler process and one CUDA context:

```text
PrefillWorker  ─┐
                ├─ SharedGpuArbiter ── one selected whole-phase SM split
DecodeWorker   ─┘
        │
PhaseCoordinator ─ request phase metadata only
```

The workers are cooperative scheduler objects, not Python threads. GPU streams,
model weights, allocators, and request lifetime remain shared with the existing
SGLang scheduler. The implementation intentionally does not add per-layer SM
switching or make KV-cache behavior a research metric.

## Existing baseline pinned before new experiments

The canonical baseline remains `reports/CONSENSUS.md`:

- change-trace goodput: `d44 3.220±0.013`, `d34 3.171±0.025`, and
  `bind+GATE 3.132±0.019`;
- tight SLO-rate-8 attainment: `d44 73.2%` versus `bind+GATE 44.3%`;
- `d16` reaches TTFT p50 `7.24s` despite assigning 92 SMs to prefill,
  while `d24` reaches `1.21s` with 84 prefill SMs.

These numbers are a reproduction gate, not a claim that dual-worker already
improves end-to-end performance. They preserve the existing conclusion that
TTFT/TPOT SLO attainment, rather than raw throughput alone, determines the
policy ranking. The known failure path remains:

```text
decode starvation → longer ITL → decode residency/running-batch growth
→ prefill admission delay → TTFT growth
```

The dynamic controller and layer-aware results are comparison controls. The
attention/Mamba layer profile is connected only after queue-level traces are
available; per-layer switching is not reactivated by this patch.

## What is implemented

`multiplex/dual_worker.py` (deployed as
`sglang/srt/multiplex/dual_worker.py`) provides:

- `PrefillWorker`: waiting queue view, active prefill batch, chunk progress,
  queue age, admission latency, and TTFT-side state.
- `DecodeWorker`: decode-ready queue, running batch, step count, and TPOT/ITL-
  side timing.
- `SharedGpuArbiter`: a logical lease over one `(prefill SM, decode SM)` stream
  group. Both workers may hold the same partition; a partition switch requires
  the previous leases to be drained.
- `PhaseCoordinator`: `PREFILL_WAITING → PREFILL_RUNNING → DECODE_READY →
  DECODE_RUNNING → COMPLETE` request transitions.

`SchedulerMultiplexMixin.event_loop_pdmux()` records these ownership and phase
boundaries when `PDMUX_DUAL_WORKER=1`. The legacy path remains the default, so
the established `fused`, static `d16/d24/d34/d44`, and dynamic-controller
baselines are not changed by this patch.

The current scheduler's allocator and admission checks are deliberately still
the safety authority. Consequently, this patch measures whether a request is
waiting because of scheduler/shared-batch capacity, but does not yet claim
that GPU contention has been removed. That distinction is the required first
experiment: queue-level blocking and SM-level interference must be reported
separately.

## Metrics exposed by the state objects

The primary experiment should collect:

- `prefill.snapshot.queue_depth`
- `prefill.snapshot.queue_age_ms`
- `prefill.snapshot.admission_latency_ms`
- `prefill.snapshot.admission_block_reason`
- `decode.snapshot.active_batch_size`
- `decode.snapshot.last_tpot_ms`
- `decode.snapshot.step_count`
- selected stream index and `(prefill_sms, decode_sms)` from the arbiter lease

TTFT, TPOT/ITL, goodput, SLO attainment, and stream overlap remain the end-to-
end evaluation metrics. Cache state is retained only for correctness and
memory-safety checks.

## Literature boundary

Bullet is useful as a reference for role-local engine/scheduler ownership and
shared coordination metadata. MuxWise is useful as a reference for keeping
prefill/decode execution independent inside one process while sharing the GPU
through a resource policy. Neither paper is treated as an identical runtime or
as a requirement to reproduce cache optimizations. The present implementation
chooses the common design point relevant to this study: independent logical
queues with shared same-GPU whole-phase resource arbitration.

## Run and next step

Enable the experimental state path with:

```bash
PDMUX_DUAL_WORKER=1 ...existing PD-mux launch...
```

First compare the legacy and dual-worker traces at the same static split under
decode-heavy saturation, prefill bursts, alternating phase load, low-decode-SM
splits, and overload. Only after queue coupling is characterized should the
attention/Mamba profiling be connected to whole-phase SM allocation. Per-layer
resource switching remains disabled unless measured sensitivity differential
exceeds its window and synchronization cost.
