# N3 6-group config boot probe (2026-10-05)

Workspace `pdmux-probe` (slug `wsp-7ix1cg4xl1jd`, A100 SXM, $1.48/h). Scope: PRE-SUBMISSION
functional + cost probe for `results/n3_capacity/configs/pdmux_n3_A_agn.yml` (6 stream groups).
NOT a measurement, NOT a campaign cell. No performance judgment.

Repo commit: 69fbbb0741ce22df5fda351ba3c585307891cef1 (uploaded as a git bundle to
`/io/code/69fbbb0.../`, cloned+checked-out inside the Workspace at `/opt/pdmux/prefill-layer-alloc`,
then `install_runtime.sh` applied the same recipe as image `sglang-runtime@1e0d335b`).
Workspace image at probe time: `sglang-runtime@c85fc198` (older digest; NOT updated for this probe --
see DEVIATION below).

Model: nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base, HF snapshot revision `dc0661c829b14e5b9246c05cfa89094a0875e052`.

## DEVIATION

1. Image digest mismatch: Workspace `pdmux-probe` was still on `sglang-runtime@c85fc198` (from
   2026-10-04 bring-up), not `@1e0d335b` (current `vessl.env` pin). Recreating the Workspace with a
   new image is a separate cost action requiring its own approval, so this probe reused the existing
   (already-approved, already-running-cost) Workspace instance and applied `install_runtime.sh`
   identically. The engine code (what actually matters for this probe -- config parsing, green-ctx
   boot, cudagraph capture) is installed fresh from the commit, independent of the base image digest.
2. First boot attempt (all 3 units) failed immediately at import time:
   `ImportError: ... libnuma.so.1: cannot open shared object file`. This reproduces the known
   DEVIATION recorded 2026-10-04 for this same Workspace/image (`libnuma1 수동 설치`,
   session_handoff_2026-10-04.md sec 6) -- the base image lacks `libnuma1`. Fixed with
   `apt-get install -y libnuma1` before re-running. No GPU time was consumed by the failed attempt
   (failure was at Python import, before any CUDA/model work); the 3 failed launches are kept in
   `boot_output.log` / `n3_probe_6group_2026-10-05_attempt1_libnuma_fail/` for the record.

## Boots performed

All boots used the SRV flags copied verbatim from `n3_capacity.sh` (lines 337-344): model/dtype/
backend/pdmux-config/--disable-overlap-schedule/--chunked-prefill-size -1/--disable-radix-cache/
mem-fraction-static 0.82/max-running-requests 48/context-length 16384/random-seed 1, with
`PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=44 PDMUX_GREEN_READOUT=1 PDMUX_PHASE_EVENTS=1`, sticky
UNSET, cudagraph ON (no `--disable-cuda-graph`). Port drawn from 20000-20799 (same window as the
runner; does not overlap this container's ephemeral range, see FACTS). Warm-up = 8 requests,
concurrency 1, random-input-len 256 / random-output-len 512 (same shape as the runner's warm-up).

| unit | config | groups | boot_wall_s | load_weight_s | capture_s | capture_mem_usage_gb | avail_mem_after_capture_gb | max_total_num_tokens | warmup_wall_s (8 req, c=1) | teardown_wall_s | green read-out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1_cold | pdmux_n3_A_agn.yml | 6 | 124.31 | 28.58 | 51.60 | 1.27 | 11.59 | 2,721,290 | 75.60 | 15.01 | GREEN OK, 6/6 realized==target, sticky_target_index=null |
| A2_warm | pdmux_n3_A_agn.yml | 6 | 59.11 | 12.38 | 14.12 | 1.27 | 11.59 | 2,721,290 | 76.54 | 15.01 | GREEN OK, 6/6 realized==target, sticky_target_index=null |
| H5 (comparison) | pdmux_homog5.yml | 5 | 60.11 | 11.66 | 12.67 | 1.07 | 11.79 | 2,722,025 | 75.99 | 15.01 | GREEN OK, 5/5 realized==target, sticky_target_index=null |

6-group vs 5-group (warm, A2_warm vs H5): +0.20 GB capture memory, +1.45 s capture time,
KV tokens -735 (-0.027%). This matches the KISTI-derived prediction in
`CONFIG_CANDIDATES_2026-10-04.md` line 169 ("+0.20 GB 캡처 메모리, +1.9 s 캡처, KV -0.027%") almost
exactly in sign and magnitude (capture-time delta measured slightly smaller: 1.45 s vs 1.9 s
predicted, within cold/warm run-to-run noise).

Reference from the registered budget model (`PREREG_N3_CAPACITY_2026-10-04.md` sec 8-3, "per-boot
부대비용 = λ0 VESSL 실측(E2 A13 §5-1): A 160.1 s · B 121.0 s, first boot cold +92.1 s"): that figure
is a per-UNIT overhead budget (boot + settle + admission bookkeeping, not boot-wall-clock alone), so
it is not a 1:1 comparison with `boot_wall_s` above; it is reported here only for context, not as a
validated match.

Functional confirmation: 6-group config boots, captures cudagraph, and serves successfully; green
read-out shows all 6 realized (prefill_sm, decode_sm) pairs exactly equal to the config's targets in
both the cold and warm boot, sticky stays unset (`sticky_target_index: null`, `is_sticky_target:
false` on every group) as required by the N3 arm definition.

## Container facts (FACTS)

- `ip_local_port_range` = `32768 60999`; `ip_local_reserved_ports` = empty. Listen window 20000-20799
  does not overlap (same check the runner itself performs, O1).
- `/sys/fs/cgroup/cpu.max` = `1100000 100000` -> CPU quota 11 cores; `nproc` reports 96 (visible !=
  quota, memory item 268).
- GPU: 1x NVIDIA A100-SXM4-80GB, driver 580.105.08, UUID `GPU-941e43e5-57a2-e715-ad67-f0dc78b8bd9c`,
  81920 MiB total.
- `ss -Htan` after all 3 teardowns: 0 TIME_WAIT sockets on any of the 3 ports used (20741, 20180,
  20559) or elsewhere -- no recurrence of the E2 port-reuse failure signature (O1-O4) in this run.

## Not done

No N3 measurement grid, no result interpretation/verdict, no edits to any sealed/registered file
(config, prereg, plan, analyze). Workspace paused after this probe.
