#!/bin/bash
# Reproduce the PDMUX_PHASE_DIAG CPU tests on a machine WITHOUT a usable sgl_kernel (e.g. the local sm_120 PC).
#   bash workspace/engine-port/tests/run_diag_tests_with_stub.sh [SOURCE_ENGINE_PYTHON] [WORKDIR]
# 1. copies SOURCE_ENGINE_PYTHON (default: <work root>/sglang_engine_dev/python, the pristine-v0.5.10 + earlier
#    syncs tree; never modified) to WORKDIR/tree/python,
# 2. installs THIS checkout into the copy exactly as a VESSL Job does (scripts/vessl/install_runtime.sh:
#    sync -> devtree_manual_edits.patch -> sync) and prints the pathless ENGINE_DIGEST,
# 3. runs the diag suites with tests/cpu_stub (fake sgl_kernel) + the copy first on PYTHONPATH.
# Nothing outside WORKDIR is written.  The stub makes these suites importable; it is not a GPU substitute.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/.." && pwd)"
project="$(cd "${eng}/../.." && pwd)"
src="${1:-$(cd "${project}/.." && pwd)/sglang_engine_dev/python}"
work="${2:-$(mktemp -d "${TMPDIR:-/tmp}/pdmux_diag_tests.XXXXXX")}"
[[ -f "${src}/sglang/srt/distributed/parallel_state.py" ]] || { echo "ERROR: no SGLang tree at ${src}" >&2; exit 2; }
mkdir -p "${work}/tree"
rm -rf "${work}/tree/python"
cp -a "${src}" "${work}/tree/python"
PDMUX_PROJECT_ROOT="${project}" SGLANG_ENGINE_DEV="${work}/tree/python" PDMUX_SYNC_LOCK="${work}/sync.lock" \
  bash "${eng}/scripts/vessl/install_runtime.sh" "${work}/manifest.sha256" > "${work}/install.log" 2>&1
echo "ENGINE_DIGEST $(python3 "${eng}/scripts/bootstrap/engine_digest.py" digest --manifest "${work}/manifest.sha256")"
echo "workdir ${work}"
cd "${here}"
SGLANG_ENGINE_DEV="${work}/tree/python" PYTHONPATH="${here}/cpu_stub:${work}/tree/python" \
  python3 -m unittest -v test_phase_diag test_engine_gate_diag test_loadgen_abs_mono test_decode_probe \
  test_phase_events test_multigpu_guard 2>&1 | tee "${work}/tests.log" | grep -E "^(Ran|OK|FAILED)|\.\.\. (FAIL|ERROR)"
