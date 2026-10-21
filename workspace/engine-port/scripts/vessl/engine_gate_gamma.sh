#!/bin/bash
# G-gamma job entry; family via --env PDMUX_GATE_FAMILY=A (agnostic) | F (fused_sym).  Hard cap 270 min (self-enforced).
# PDMUX_GATE_GAMMA_PHASE=ext runs the registered one-time extension pairs 11-16 (a separate Job, pooled by `pool-gamma`).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${here}/engine_gate_common.sh"
gate_main gamma 270
exit $?
