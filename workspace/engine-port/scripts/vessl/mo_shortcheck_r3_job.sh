#!/bin/bash
# VESSL Job target: fourth short confirmation Job of the motivation track (NON-MEASUREMENT).  Same preflight as mo_shortcheck_job.sh (one GPU, engine digest).
# Boots only: a31 on vLLM as-is (control) and with layer_types = num_hidden_layers x "full_attention" (vLLM is_hybrid False).
# The third Job's a31 layer_types boot died at config parse (31 entries for 56 layers); its S4 SG records passed and are not repeated.
set -euo pipefail
export MO_SHORTCHECK_ARGS="--only S3 --s3-cands as_is,layer_types --budget-min 30"
exec bash "$(dirname "${BASH_SOURCE[0]}")/mo_shortcheck_job.sh"
