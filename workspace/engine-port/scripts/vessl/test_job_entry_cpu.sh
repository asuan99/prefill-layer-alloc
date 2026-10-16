#!/bin/bash
# Self-check for job_entry.sh CPU-only mode (2026-10-16).  Local, CPU; uses HEAD (must contain this file).
#   test_job_entry_cpu.sh                 run the checks
#   (internal) with PDMUX_JE_SELFTEST_INNER=1 it is the trivial Job target -- the guard stops self-recursion
set -uo pipefail
if [[ "${1:-}" == "--as-target" || "${PDMUX_JE_SELFTEST_INNER:-}" == 1 ]]; then echo "target ran; LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"; exit 0; fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo="$(git -C "${here}" rev-parse --show-toplevel)"
T="${TMPDIR:-/tmp}/jecpu.$$"; mkdir -p "${T}"; trap 'rm -rf "${T}"' EXIT
commit="$(git -C "${repo}" rev-parse HEAD)"; fails=0
chk() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails+1)); fi; }
mkdir -p "${T}/io/code/${commit}"
git -C "${repo}" bundle create "${T}/io/code/${commit}/repo.bundle" HEAD >/dev/null 2>&1
(cd "${T}/io/code/${commit}" && sha256sum repo.bundle > repo.bundle.sha256)
run() {  # name spec extra-env...
  local n="$1" spec="$2"; shift 2
  env -i PATH="${PATH}" HOME="${T}" PDMUX_COMMIT="${commit}" PDMUX_JOB_NAME="$n" PDMUX_CAMPAIGN=t PDMUX_IMAGE=img@sha256:0 \
    PDMUX_SPEC="${spec}" PDMUX_TARGET=workspace/engine-port/scripts/vessl/test_job_entry_cpu.sh PDMUX_TEST_SKIP_SYNC=1 \
    PDMUX_IO="${T}/io" PDMUX_DATA="${T}/data" PDMUX_OPT="${T}/opt/$n" PDMUX_HEARTBEAT_S=0 PDMUX_JE_SELFTEST_INNER=1 "$@" \
    bash "${here}/job_entry.sh" > "${T}/$n.out" 2>&1
}
mkdir -p "${T}/data/hf" "${T}/opt/c1" "${T}/opt/c2" "${T}/opt/g"
# --- CPU by env flag and by spec slug
run c1 resourcespec-x PDMUX_CPU_ONLY=1; run c2 resourcespec-a100cpu
for n in c1 c2; do R="${T}/data/runs/$n/meta"
  chk "$n DONE exit_code=0" "grep -q '^exit_code=0' ${T}/io/results/$n/DONE"
  chk "$n CPU_ONLY.txt recorded" "grep -q '^cpu_only=1' ${R}/CPU_ONLY.txt"
  chk "$n substrate.json cpu_only true, gpu null" "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(not (r[\"cpu_only\"] is True and r[\"gpu\"] is None))' ${R}/substrate.json"
  chk "$n no clk.csv / energy.csv / substrate_history" "[[ ! -e ${R}/clk.csv && ! -e ${R}/energy.csv && ! -e ${T}/data/substrate_history.tsv ]]"
  chk "$n [PHASE] CPU-only line in stdout" "grep -q 'PHASE .*CPU-only mode' ${T}/$n.out"
  chk "$n SHA256SUMS verifies" "(cd ${T}/io/results/$n && sha256sum -c --quiet SHA256SUMS)"
done
# --- GPU path untouched: only when this machine has nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  mkdir -p "${T}/opt/g"; run g resourcespec-a100x1; R="${T}/data/runs/g/meta"
  chk "gpu path: no CPU_ONLY.txt, substrate.json has gpu, clk.csv present" "[[ ! -e ${R}/CPU_ONLY.txt && -e ${R}/clk.csv ]] && python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))[\"gpu\"] is None)' ${R}/substrate.json"
else echo "skip gpu-path run (no GPU here)"; fi
# --- GPU lines of job_entry.sh are byte-identical to the base: the CPU change may only INSERT lines
base="${PDMUX_JE_BASE:-e43749c91b350a702fbdc7487767c84d6835d793}"
dels="$(git -C "${repo}" diff "${base}" HEAD -- workspace/engine-port/scripts/vessl/job_entry.sh | grep -c '^-[^-]')"
chk "job_entry.sh vs ${base:0:8}: 0 deleted lines" "[[ ${dels} -eq 0 ]]"
echo "failures=${fails}"; exit $((fails > 0))
