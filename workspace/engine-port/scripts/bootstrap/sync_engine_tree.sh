#!/bin/bash
# Install tracked PD-mux sources into an editable SGLang v0.5.10 tree.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
track_root="$(cd "${script_dir}/../.." && pwd)"
runtime_python="${SGLANG_ENGINE_DEV:-/scratch/ehmoon/whlee/sglang_engine_dev/python}"
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
  "${track_root}/src/multiplex/telemetry.py"; do
  target="${runtime_python}/sglang/srt/multiplex/$(basename "${source}")"
  install -D -m 0644 "${source}" "${target}"
done

# Zamba2 (Zyphra/Zamba2-*): NEW config + model files (dev_tree_edits.md items 1-2).
# Brought under sync + manifest on 2026-08-04 with the per-layer-type timing
# rework, so the instrumentation that produces ZBLT2 lines is reproducible from
# the tracked source instead of a manual copy. The remaining hybrids
# (nemotron_h / falcon_h1 / granitemoehybrid) are still manual copies -- see
# dev_tree_edits.md items 6, 8, 9.
install -D -m 0644 "${track_root}/src/configs/zamba2.py" \
  "${runtime_python}/sglang/srt/configs/zamba2.py"
install -D -m 0644 "${track_root}/src/models/zamba2.py" \
  "${runtime_python}/sglang/srt/models/zamba2.py"

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

mkdir -p "$(dirname "${manifest_path}")"
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
  "${runtime_python}/sglang/srt/configs/mamba2.py" \
  "${runtime_python}/sglang/srt/models/mamba2.py" \
  "${runtime_python}/sglang/srt/configs/zamba2.py" \
  "${runtime_python}/sglang/srt/models/zamba2.py" \
  "${runtime_python}/sglang/srt/server_args.py" \
  "${runtime_python}/sglang/srt/model_executor/model_runner_kv_cache_mixin.py" \
  "${runtime_python}/sglang/srt/mem_cache/memory_pool.py" \
  > "${manifest_tmp}"
mv -f "${manifest_tmp}" "${manifest_path}"
trap - EXIT

echo "Synced PD-mux runtime: ${runtime_python}"
echo "Hash manifest: ${manifest_path}"
