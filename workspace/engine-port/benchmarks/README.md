# R2 benchmark package

`pdmux_eval` is the canonical workload/campaign/statistics layer for experiments
after R1. `scripts/r2_eval/engine_bench_runner.sh` starts the server and
`trace_loadgen.py` replays these immutable arrivals through SGLang's streaming
endpoint, retaining client-observed token timestamps.

```bash
python -m pdmux_eval.workloads --workload W3 --sustainable-rate 4 \
  --count 256 --seed 1 --output results/r2_eval/traces/W3.jsonl
python -m pdmux_eval.campaign --trace-dir results/r2_eval/traces \
  --workloads W3 --baselines B1 B4 B5 B6 --repetitions 5 \
  --output results/r2_eval/campaign.json
```

All policy arms in a pair receive the same trace hash, workload seed, server
seed, CUDA Graph setting, capacity and SLO. The run order is randomized.

`B2` and `B8` records are emitted with `requires_offline_oracle=true`. They are
not runnable until `select_static_baselines` or `trace_aware_oracle` has produced
the workload-specific split/schedule; the policy adapter deliberately rejects
unresolved oracle records.

Build or validate an operational profile with:

```bash
python -m pdmux_eval.profile_cli build --metadata model.json \
  --measurements decode_surface.csv --output profile.json
python -m pdmux_eval.profile_cli validate profile.json
```

The CSV columns are `decode_sms,batch_size,context_tokens,itl_p50_ms,
itl_p95_ms,itl_p99_ms,repeats,measured_steps,residual_p95_ms`. Component-level
attention/SSM curves may be stored as profile metadata, but serving decisions
use the full-model surface.
