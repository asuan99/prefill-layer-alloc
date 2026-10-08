#!/usr/bin/env bash
# Container-path power-cap probe (VESSL Workspace/Job, ~1 min on a GPU spec).  NOT executed yet —
# submission needs the operating-model approval (launch dry-run -> user approval).  Read-only except
# for one *no-op* set: `-pl <current limit>` (same value) so nothing changes even if it succeeds.
# Records what the 2026-10-04 substrate probe did not: effective capabilities and the exact NVML error
# for power-limit / clock / persistence operations.
set -u
OUT="${1:-/tmp/power_cap_probe}"; mkdir -p "$OUT"
{
  echo "# $(date -u +%FT%TZ) uid=$(id -u) user=$(id -un)"
  grep -E '^Cap(Inh|Prm|Eff|Bnd|Amb)' /proc/self/status
  command -v capsh >/dev/null && capsh --decode="$(grep CapEff /proc/self/status | awk '{print $2}')"
  ls -l /dev/nvidiactl /dev/nvidia0 2>&1
  cat /proc/driver/nvidia/params 2>/dev/null | grep -iE 'ModifyDeviceFiles|DeviceFileMode' || true
  nvidia-smi --query-gpu=name,driver_version,power.limit,power.default_limit,power.min_limit,power.max_limit,persistence_mode,clocks.sm,clocks.max.sm --format=csv
  CUR=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
  echo "## nvidia-smi -pl $CUR (no-op: same as current)"; nvidia-smi -pl "$CUR"; echo "rc=$?"
  echo "## nvidia-smi -lgc 1410,1410 (expected: permission denied)"; nvidia-smi -lgc 1410,1410; echo "rc=$?"
  echo "## nvidia-smi -rgc"; nvidia-smi -rgc; echo "rc=$?"
  echo "## nvidia-smi -pm 1"; nvidia-smi -pm 1; echo "rc=$?"
  echo "## pynvml nvmlDeviceSetPowerManagementLimit (same value)"
  python3 - <<'PY'
try:
    import pynvml as n
    n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(0)
    cur = n.nvmlDeviceGetPowerManagementLimit(h)
    try:
        n.nvmlDeviceSetPowerManagementLimit(h, cur); print("set-same-value: OK", cur)
    except n.NVMLError as e:
        print("set-same-value: NVMLError", e)
    print("constraints", n.nvmlDeviceGetPowerManagementLimitConstraints(h))
except Exception as e:
    print("pynvml unavailable:", e)
PY
  echo "## after"; nvidia-smi --query-gpu=power.limit,persistence_mode,clocks.sm --format=csv,noheader
} 2>&1 | tee "$OUT/probe.txt"
