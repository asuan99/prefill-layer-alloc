#!/bin/bash
# ONE Job, two sequential stages on the same final engine tree and the same node (first GPU step, NOT SUBMITTED):
#   stage 1  scripts/vessl/engine_gate_diag.sh  (diag GPU gate D0-D9, 5 Nano boots, own hard cap 120 min)
#   stage 2  ph_smoke.py                        (policy Q-3 smoke: non-sticky fixed applies D-dependent splits; bind+GATE boots; 3 boots, cap 75 min)
#   launch.sh --campaign policy_gate_smoke --target workspace/engine-port/results/cap_char/policy_harness/ph_gate_smoke.sh
# Boots cannot be shared between the stages (different env/arms); the saving is one image pull + install + substrate stamp, and
# both stages see the same node (a gate/smoke disagreement is not a node effect).  rc = max of the two stage results; stage 2
# runs even if stage 1 failed (both facts are wanted).  Nothing here prints the environment.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../../.." && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
digest="$(tr -d ' \n' < "${here}/ENGINE_DIGEST.txt")"
if [[ "${digest}" == "PENDING_FINAL_TREE" || ${#digest} != 64 ]]; then echo "ABORT_ENGINE_DIGEST_PENDING: ${here}/ENGINE_DIGEST.txt" >&2; exit 13; fi
export PDMUX_DIAG_GATE_DIGEST="${digest}"
summary="${here}/../policy_runs/${name}"; mkdir -p "${summary}"

bash "${eng}/scripts/vessl/engine_gate_diag.sh"; rc1=$?
echo "stage1 engine_gate_diag rc=${rc1}" | tee "${summary}/stages.txt"

sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
tok="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/tokenizer.json"
python3 "${here}/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${tok}" --out "${summary}/w1_lengths.json" || { echo "ABORT_DATASET: W1 build failed" >&2; echo "stage2 not run" >> "${summary}/stages.txt"; exit $(( rc1 > 13 ? rc1 : 13 )); }
timeout -k 60 4500 python3 "${here}/ph_smoke.py" --out "${summary}/smoke" --w1-lengths "${summary}/w1_lengths.json"
rc2=$?
echo "stage2 ph_smoke rc=${rc2}" | tee -a "${summary}/stages.txt"
exit $(( rc1 > rc2 ? rc1 : rc2 ))
