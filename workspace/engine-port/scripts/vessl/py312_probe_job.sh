#!/bin/bash
# VESSL Job target for the Python 3.12 boot probe. Needs the image in vessl.env PDMUX_VESSL_IMAGE_SG312 (see py312_probe.py).
set -euo pipefail
: "${PDMUX_PROJECT_ROOT:?}"
exec python3 "${PDMUX_PROJECT_ROOT}/workspace/engine-port/scripts/vessl/py312_probe.py" \
  --out-root "${PDMUX_PROJECT_ROOT}/workspace/engine-port/results/py312_probe" --budget-min 150
