#!/bin/bash
# VESSL Job target: third short confirmation Job of the motivation track (NON-MEASUREMENT).  Same preflight as mo_shortcheck_job.sh (one GPU, engine digest).
# Boots only: a31 on vLLM as-is (control) and with layer_types all attention; observation records ON/OFF on the fused patched tree with piecewise OFF.
# The earlier Job's PD-mux records already passed and are not repeated.
set -euo pipefail
export MO_SHORTCHECK_ARGS="--only S3,S4 --s3-cands as_is,layer_types --s4-cfgs SG --budget-min 60"
exec bash "$(dirname "${BASH_SOURCE[0]}")/mo_shortcheck_job.sh"
