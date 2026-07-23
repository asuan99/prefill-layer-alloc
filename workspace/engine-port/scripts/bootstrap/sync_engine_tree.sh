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

mkdir -p "$(dirname "${manifest_path}")"
sha256sum \
  "${runtime_python}/sglang/srt/distributed/parallel_state.py" \
  "${runtime_python}/sglang/srt/multiplex/dual_worker.py" \
  "${runtime_python}/sglang/srt/multiplex/multiplexing_mixin.py" \
  "${runtime_python}/sglang/srt/multiplex/profile.py" \
  "${runtime_python}/sglang/srt/multiplex/controller.py" \
  "${runtime_python}/sglang/srt/multiplex/telemetry.py" \
  > "${manifest_path}"

echo "Synced PD-mux runtime: ${runtime_python}"
echo "Hash manifest: ${manifest_path}"
