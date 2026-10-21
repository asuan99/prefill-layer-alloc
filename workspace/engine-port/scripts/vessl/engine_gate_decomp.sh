#!/bin/bash
# G-gamma observer-effect DECOMPOSITION job entry (decision 21; diagnostic, not a gate).  Levels {OFF, I5, I6, I46, TEL} x 4 blocks,
# one Job, Williams order.  Hard cap 250 min (self-enforced).  Raw data stays on the cluster volume /data/runs/<job>/raw;
# only summaries are exported (job_entry exports results/ only).  Design: engine_gate_decomp.py docstring.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${here}/engine_gate_common.sh"
gate_main decomp 250
exit $?
