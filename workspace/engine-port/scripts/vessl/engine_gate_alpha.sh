#!/bin/bash
# G-alpha job entry (design GATE_SPLIT_DESIGN_2026-10-20.md sec 2).  Hard cap 120 min (self-enforced, PDMUX_GATE_HARDCAP_MIN).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${here}/engine_gate_common.sh"
gate_main alpha 120
exit $?
