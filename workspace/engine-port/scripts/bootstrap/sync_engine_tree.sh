#!/bin/bash
# Install tracked PD-mux sources into an editable SGLang v0.5.10 tree.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
track_root="$(cd "${script_dir}/../.." && pwd)"
# PDMUX_ROOT = the execution host's work root (holds prefill-layer-alloc/,
# sglang_engine_dev/, sglang_engine_venv/ as siblings; see CLAUDE.md
# "환경 / 실행"). Only used to build the SGLANG_ENGINE_DEV *default* --
# SGLANG_ENGINE_DEV still wins outright when set, exactly as before this
# change (the `${SGLANG_ENGINE_DEV:-...}` expansion is untouched). When
# PDMUX_ROOT is also unset we derive it from this script's own location
# (three levels above track_root = workspace/engine-port) instead of a
# hardcoded path, because the old KISTI Neuron path is now a read-only
# archive, not a place to default onto.
pdmux_root="${PDMUX_ROOT:-$(cd "${track_root}/../../.." && pwd)}"
runtime_python="${SGLANG_ENGINE_DEV:-${pdmux_root}/sglang_engine_dev/python}"
manifest_path="${1:-${track_root}/results/runtime_source_manifest.sha256}"

if [[ ! -f "${runtime_python}/sglang/srt/distributed/parallel_state.py" ]]; then
  echo "ERROR: SGLang runtime not found under ${runtime_python}" >&2
  exit 2
fi

# Serialize concurrent syncs. SLURM array tasks share one dev tree, and two of
# them installing at once race inside `install` (it unlinks then re-creates the
# target, so the loser hits EEXIST) and in `patch`; that surfaced as
# `install: cannot create regular file ...: File exists` and killed job
# 864230_0 (2026-07-26). Reproduced 7-8/10 concurrent runs without this lock,
# 0/10 with it. /scratch is Lustre mounted with `flock` (not `localflock`), so
# the lock is coherent across nodes.
sync_lock="${PDMUX_SYNC_LOCK:-$(dirname "${runtime_python}")/.pdmux_sync.lock}"
sync_lock_wait="${PDMUX_SYNC_LOCK_WAIT:-900}"
if [[ -z "${PDMUX_SYNC_LOCK_HELD:-}" ]]; then
  export PDMUX_SYNC_LOCK_HELD=1
  # `status=$?` inside `if ! cmd` would read the negation (always 0) and exit
  # 0 on a lock timeout, letting a job run against an unsynced tree.
  status=0
  flock --timeout "${sync_lock_wait}" "${sync_lock}" \
    "${BASH_SOURCE[0]}" "$@" || status=$?
  if (( status != 0 )); then
    echo "ERROR: sync failed or lock not acquired within ${sync_lock_wait}s" \
      "(lock ${sync_lock}, status ${status})" >&2
    exit "${status}"
  fi
  exit 0
fi

patch_file="${track_root}/src/patches/pdmux_thread_local_role.patch"
if ! grep -q "def pdmux_role_is_thread_local" \
  "${runtime_python}/sglang/srt/distributed/parallel_state.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${patch_file}"
fi

for source in \
  "${track_root}/src/multiplex/dual_worker.py" \
  "${track_root}/src/multiplex/multiplexing_mixin.py" \
  "${track_root}/src/multiplex/profile.py" \
  "${track_root}/src/multiplex/controller.py" \
  "${track_root}/src/multiplex/telemetry.py" \
  "${track_root}/src/multiplex/holb_probe.py" \
  "${track_root}/src/multiplex/chunk_probe.py" \
  "${track_root}/src/multiplex/green_readout.py" \
  "${track_root}/src/multiplex/multigpu_guard.py" \
  "${track_root}/src/multiplex/admission_probe.py" \
  "${track_root}/src/multiplex/decode_probe.py"; do
  target="${runtime_python}/sglang/srt/multiplex/$(basename "${source}")"
  install -D -m 0644 "${source}" "${target}"
done

# Head-of-line-blocking probe hooks in Scheduler.run_batch (2026-08-06).
# Arm-agnostic instrumentation, DEFAULT OFF: without PDMUX_HOLB_PATH the three
# hook sites collapse to `is not None` checks and no probe object is built.
# Lives in scheduler.py (not the pdmux mixin) precisely so that the fused arms
# -- event_loop_overlap and event_loop_normal -- are instrumented by the same
# code as event_loop_pdmux; a pdmux-only hook would reproduce the Gate-1 defect
# where the telemetry existed only on the --enable-pdmux arm.
# See results/p1_gates/gate2/DIRECT_BLOCKING_DESIGN.md.
holb_patch="${track_root}/src/patches/holb_probe_scheduler_hooks.patch"
if ! grep -q "maybe_create_holb_probe" \
  "${runtime_python}/sglang/srt/managers/scheduler.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${holb_patch}"
fi

# Chunked-prefill probe construction hook in Scheduler.__init__ (2026-08-28).
# CP-0 prerequisite P1 (PREREG_CP0_2026-08-28.md section 2): before this the tree
# had no way to observe chunked prefill outside the pdmux mixin. Arm-agnostic and
# DEFAULT OFF: without PDMUX_CHUNK_PROBE_PATH `maybe_install_chunk_probe` returns
# None and installs no wrapper at all, so the admission path and `run_batch` keep
# running the pristine engine functions with no added branch. Counters live in the
# scheduler layer on purpose -- `--chunked-prefill-size -1` empties the piecewise
# CUDA graph capture list (server_args.py:1254-1259, :1397-1415 ->
# model_runner.py:2486-2490), so an instrument inside a graph-captured prefill
# path would let the negative control pass for the wrong reason (P1-g).
# MUST run after the holb patch: it shares context lines with it.
# See sglang/srt/multiplex/chunk_probe.py.
chunk_patch="${track_root}/src/patches/chunk_probe_scheduler_hook.patch"
if ! grep -q "maybe_install_chunk_probe" \
  "${runtime_python}/sglang/srt/managers/scheduler.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${chunk_patch}"
fi

# I-5 admission-reason probe construction hook in run_scheduler_process (2026-10-19,
# SPEC_common_measurement_v0 sec 9; wired 2026-10-20 with decision 15).  Arm-agnostic and
# DEFAULT OFF: without PDMUX_ADMISSION_PATH `maybe_install_admission_probe` returns None and
# installs no wrapper.  The hunk sits after the Scheduler class, so no fingerprinted
# scheduler.py citation moves.  MUST run after the holb and chunk patches: its offsets are
# computed on the tree they produce.  See sglang/srt/multiplex/admission_probe.py.
admission_patch="${track_root}/src/patches/admission_probe_scheduler_hook.patch"
if ! grep -q "maybe_install_admission_probe" \
  "${runtime_python}/sglang/srt/managers/scheduler.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${admission_patch}"
fi

# I-4 decode forward GPU time + I-6 cudagraph padding probe hook in run_scheduler_process
# (2026-10-20, decision 16 O-2 -- same ENGINE_DIGEST change as I-5 + conv int64).  Arm-agnostic and
# DEFAULT OFF: without PDMUX_DECODE_PROBE_PATH `maybe_install_decode_probe` returns None and the
# target ModelRunner.forward stays pristine.  When ON it shadows forward() on that ModelRunner
# INSTANCE, which event_loop_normal / event_loop_overlap / event_loop_pdmux all reach through
# TpModelWorker.forward_batch_generation (vanilla fused, Llama and PD-mux recorded alike).
# CUDA events are recorded on the forward's own stream and only polled with query() -- no
# synchronize, no cross-stream wait (lockstep untouched; see decode_probe.py docstring).
# MUST run after the admission patch: its context is the admission hook's last line.
decode_patch="${track_root}/src/patches/decode_probe_scheduler_hook.patch"
if ! grep -q "maybe_install_decode_probe" \
  "${runtime_python}/sglang/srt/managers/scheduler.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${decode_patch}"
fi

# Triton causal_conv1d prefill kernel: 64-bit token offsets (2026-10-20, decision 15).
# Two same-line casts in _causal_conv1d_fwd_kernel (chunk_offset, sequence_start_index);
# without them NemotronH's non-contiguous xBC view (stride 22,656) wraps int32 offsets from
# ~94.8k tokens in one forward (chunk -1).  Upstream file, previously unhashed -> manifest
# entry 29.  Idempotent: grep guard.  See the patch header.
conv_int64_patch="${track_root}/src/patches/triton_causal_conv1d_int64.patch"
if ! grep -q "int64: token offsets" \
  "${runtime_python}/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${conv_int64_patch}"
fi

# Zamba2 (Zyphra/Zamba2-*): NEW config + model files (dev_tree_edits.md items 1-2).
# Brought under sync + manifest on 2026-08-04 with the per-layer-type timing
# rework, so the instrumentation that produces ZBLT2 lines is reproducible from
# the tracked source instead of a manual copy.
install -D -m 0644 "${track_root}/src/configs/zamba2.py" \
  "${runtime_python}/sglang/srt/configs/zamba2.py"
install -D -m 0644 "${track_root}/src/models/zamba2.py" \
  "${runtime_python}/sglang/srt/models/zamba2.py"

# The other three hybrids (dev_tree_edits.md items 6, 8, 9) were MANUAL copies
# until 2026-09-13 and their model files were absent from the hash manifest, so
# a campaign served by NemotronH / Falcon-H1 / Granite-4 recorded no provenance
# for the model implementation it actually ran -- including
# `forward_split_prefill`, i.e. the method that makes PD-mux SPLIT_PREFILL
# possible on those models at all. Verified byte-identical to the dev tree at
# the time of this change (sha256 of src/models/{nemotron_h,falcon_h1,
# granitemoehybrid}.py == the installed copies), so installing them is a no-op
# on the current tree and a repair on any rebuilt one.
for hybrid_model in nemotron_h falcon_h1 granitemoehybrid; do
  install -D -m 0644 "${track_root}/src/models/${hybrid_model}.py" \
    "${runtime_python}/sglang/srt/models/${hybrid_model}.py"
done

# Pure Mamba2 (state-spaces/mamba2-*) Stage 0 negative-control arm: install the
# NEW config + model files, then apply the tracked arch-registration patch
# (configs/__init__, hf_transformers_utils registry, server_args dispatch,
# kv_cache_mixin cell_size==0 guard). Idempotent: grep-guard before patching.
install -D -m 0644 "${track_root}/src/configs/mamba2.py" \
  "${runtime_python}/sglang/srt/configs/mamba2.py"
install -D -m 0644 "${track_root}/src/models/mamba2.py" \
  "${runtime_python}/sglang/srt/models/mamba2.py"
mamba2_patch="${track_root}/src/patches/mamba2_pure_ssm_arch.patch"
if ! grep -q 'model_arch in \["Mamba2ForCausalLM"\]' \
  "${runtime_python}/sglang/srt/server_args.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${mamba2_patch}"
fi

# PD-mux capability/device assert (multi-GPU M2, 2026-10-12; design sec 2-5,
# audit R-4).  One same-line assert inside the divide_sm (auto-grid) branch of
# pdmux_context.initialize_stream_groups -- the only place current_device() is
# read.  manual_divisions configs never execute it.  See the patch header for
# why it fires for gpu_id != 0 under TP/DP > 1.  Idempotent: grep guard.
device_assert_patch="${track_root}/src/patches/pdmux_context_device_assert.patch"
if ! grep -q "PD-mux auto grid (divide_sm)" \
  "${runtime_python}/sglang/srt/multiplex/pdmux_context.py"; then
  patch --forward --batch -p1 -d "${runtime_python}" < "${device_assert_patch}"
fi

mkdir -p "$(dirname "${manifest_path}")"
# MANIFEST (2026-09-13: 17 -> 24 -> 25 entries; 2026-10-12: -> 27; 2026-10-20: -> 29 -> 30).  Two kinds of line:
#   (a) installed from tracked source  -- multiplex/*, configs/{mamba2,zamba2},
#       models/{mamba2,zamba2,nemotron_h,falcon_h1,granitemoehybrid}; the sync
#       above rewrites these, so a mismatch means the tracked source changed.
#   (b) upstream-but-load-bearing      -- parallel_state / scheduler /
#       server_args / model_runner_kv_cache_mixin / memory_pool (patched in
#       place) and configs/{nemotron_h,falcon_h1,granitemoehybrid,mamba_utils}
#       plus layers/attention/flashinfer_backend.py (pristine upstream, NOT
#       installed here).  These are hashed because they decide the served
#       geometry: configs/nemotron_h.py maps hybrid_override_pattern ->
#       layers_block_type (which layers are attention vs mamba) and
#       mamba_utils.py turns that into mamba_cache_per_req, i.e. how much state
#       one request holds -- the capacity a lambda* label depends on.
#       flashinfer_backend.py was added 2026-09-13 (entry 25) because switching
#       NemotronH off triton made it load bearing for a CONCLUSION, not just for
#       speed: its decode plan is called with fixed_split_size=None and
#       disable_split_kv=False outside --enable-deterministic-inference, so the
#       adaptive split-KV schedule it picks is what the R2 correctness gate's
#       O-tier equality rests on (newpair_prereg/PREREG_NEWPAIR_2026-09-13.md
#       sec 3.3).  NOTE this hashes SGLang's backend wrapper, NOT the installed
#       `flashinfer` wheel -- that version is recorded separately in the
#       r2_correctness provenance dump.  Recording them cannot prevent drift,
#       but it makes drift visible instead of silent.
# 2026-10-12 (multi-GPU M2): entries 26-27 APPENDED -- multiplex/pdmux_context.py
# (now patched: device assert above; before this it was pristine upstream and
# unhashed although it decides the stream groups) and multiplex/multigpu_guard.py
# (new tracked source: TP>1 fail-fast predicate + per-rank output naming).
# ★A registration that compares the WHOLE entry set (n2_plan.engine_check,
# n3_plan, e2_vessl_plan) will report these two as installed-but-unregistered:
# that is the intended fail-closed outcome for a tree built from this change,
# not a defect -- those campaigns pin the pre-M2 engine.
# 2026-10-20 (decision 15, ONE ENGINE_DIGEST change): entries 28-29 APPENDED --
# multiplex/admission_probe.py (new tracked source, I-5; its scheduler.py hook is
# already covered by entry 10) and layers/attention/mamba/causal_conv1d_triton.py
# (pristine upstream until now, unhashed although the long-prefill limit depends on it;
# now patched: int64 offsets above).  Every plan sealed on the 27-entry digest
# (82a5fc5b..., M3 rev12 / PREFLIGHT_PASS / G5a) fails closed on a tree built from this.
# 2026-10-20 (decision 16 O-2, SAME digest change as 28-29 -- the 29-entry digest 2b1ab054... was
# never deployed): entry 30 APPENDED -- multiplex/decode_probe.py (new tracked source, I-4 + I-6; its
# scheduler.py hook is covered by entry 10).  Predicted digest and fail-closed list: dev_tree_edits.md
# items 34-36.
# The first 17 lines and their order are UNCHANGED, so `sha256sum -c` of a
# pre-2026-09-13 manifest (job_907100 / job_907456) still verifies exactly the
# same 17 files; entries 18-24 and now 25 are APPENDED, never inserted, so every
# earlier manifest stays a prefix of every later one.
# Write via temp+rename so a concurrent reader never sees a truncated manifest.
manifest_tmp="$(mktemp "${manifest_path}.XXXXXX")"
trap 'rm -f "${manifest_tmp}"' EXIT
sha256sum \
  "${runtime_python}/sglang/srt/distributed/parallel_state.py" \
  "${runtime_python}/sglang/srt/multiplex/dual_worker.py" \
  "${runtime_python}/sglang/srt/multiplex/multiplexing_mixin.py" \
  "${runtime_python}/sglang/srt/multiplex/profile.py" \
  "${runtime_python}/sglang/srt/multiplex/controller.py" \
  "${runtime_python}/sglang/srt/multiplex/telemetry.py" \
  "${runtime_python}/sglang/srt/multiplex/holb_probe.py" \
  "${runtime_python}/sglang/srt/multiplex/chunk_probe.py" \
  "${runtime_python}/sglang/srt/multiplex/green_readout.py" \
  "${runtime_python}/sglang/srt/managers/scheduler.py" \
  "${runtime_python}/sglang/srt/configs/mamba2.py" \
  "${runtime_python}/sglang/srt/models/mamba2.py" \
  "${runtime_python}/sglang/srt/configs/zamba2.py" \
  "${runtime_python}/sglang/srt/models/zamba2.py" \
  "${runtime_python}/sglang/srt/server_args.py" \
  "${runtime_python}/sglang/srt/model_executor/model_runner_kv_cache_mixin.py" \
  "${runtime_python}/sglang/srt/mem_cache/memory_pool.py" \
  "${runtime_python}/sglang/srt/models/nemotron_h.py" \
  "${runtime_python}/sglang/srt/models/falcon_h1.py" \
  "${runtime_python}/sglang/srt/models/granitemoehybrid.py" \
  "${runtime_python}/sglang/srt/configs/nemotron_h.py" \
  "${runtime_python}/sglang/srt/configs/falcon_h1.py" \
  "${runtime_python}/sglang/srt/configs/granitemoehybrid.py" \
  "${runtime_python}/sglang/srt/configs/mamba_utils.py" \
  "${runtime_python}/sglang/srt/layers/attention/flashinfer_backend.py" \
  "${runtime_python}/sglang/srt/multiplex/pdmux_context.py" \
  "${runtime_python}/sglang/srt/multiplex/multigpu_guard.py" \
  "${runtime_python}/sglang/srt/multiplex/admission_probe.py" \
  "${runtime_python}/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py" \
  "${runtime_python}/sglang/srt/multiplex/decode_probe.py" \
  > "${manifest_tmp}"
mv -f "${manifest_tmp}" "${manifest_path}"
trap - EXIT

echo "Synced PD-mux runtime: ${runtime_python}"
echo "Hash manifest: ${manifest_path}"
