#!/bin/bash
# cpu_probe.sh -- minimal CPU-container probe (2026-10-16): can the Job image (no GPU) import sgl_kernel and the
# CL-3 import-gap test modules?  Target of launch.sh --spec resourcespec-a100cpu (job_entry.sh CPU mode puts the
# CUDA forward-compat libcuda on LD_LIBRARY_PATH).  Correctness probe only; exits 0 iff the three imports pass.
set -uo pipefail
PROJECT="${PDMUX_PROJECT_ROOT:?run through job_entry.sh}"; JOB="${PDMUX_JOB_NAME:?}"
OUT="${PROJECT}/workspace/engine-port/results/cpu_probe/${JOB}"; mkdir -p "${OUT}"
PY="${PDMUX_PF_PY:-python}"; export PYTHONPATH="${SGLANG_ENGINE_DEV:-}"; rc=0
{ echo "which python: $(command -v "${PY}")  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"; ls /usr/local/cuda-13.0/compat 2>&1 | head -3
  nvidia-smi -L 2>&1 | head -1; nproc; cat /sys/fs/cgroup/cpu.max 2>&1; } > "${OUT}/env.txt" 2>&1
"${PY}" -W ignore -c 'import torch, sgl_kernel; print("torch+sgl_kernel ok", sgl_kernel.__file__)' > "${OUT}/import_sgl_kernel.txt" 2>&1 || rc=1
"${PY}" -W ignore -c 'from sgl_kernel import spatial; print("spatial import ok")' > "${OUT}/import_spatial.txt" 2>&1 || echo "(spatial import failed: informational)" >> "${OUT}/import_spatial.txt"
"${PY}" -W ignore -c 'import torch, sgl_kernel; import sglang.srt.multiplex.multiplexing_mixin; print("full ok")' > "${OUT}/import_full.txt" 2>&1 || rc=1
cd "${PROJECT}/workspace/engine-port/tests" || exit 2
timeout 900 "${PY}" -m unittest -v test_chunk_probe test_mem_telemetry_symmetry test_phase_events test_r2_admission_latch \
  test_r2_admission_persistence test_sticky_partition test_trace_force_prefill test_true_dual_prefill_ownership \
  test_worker_grad_guard test_holb_probe > "${OUT}/unittest_modules.log" 2>&1
echo "unittest_rc=$?" >> "${OUT}/unittest_modules.log"
cat "${OUT}"/env.txt "${OUT}"/import_*.txt; tail -8 "${OUT}/unittest_modules.log"
echo "probe_rc=${rc}"; exit "${rc}"
