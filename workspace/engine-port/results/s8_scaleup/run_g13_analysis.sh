#!/bin/bash
# Chained: wait for the scorer, then run the pre-registered analyzer unblinded.
# Flags are exactly the ones the audit's unblinding procedure fixed -- in
# particular --c2r-json is NOT overridden (it is a side channel while blinded).
cd "$(dirname "$0")"
while pgrep -f s8_c2r_score >/dev/null; do sleep 60; done
[ -s G13_SCORE_2026-08-21.json ] || { echo "SCORER PRODUCED NOTHING"; exit 1; }
python3 g13_analyze.py --score-json G13_SCORE_2026-08-21.json \
        --out G13_ANALYSIS_2026-08-21.json --unblind
