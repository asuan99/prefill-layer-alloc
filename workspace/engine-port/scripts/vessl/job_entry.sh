#!/bin/bash
# In-container entry point for every VESSL Cloud measurement Job.
#
# WHY THIS EXISTS.  A VESSL Job is a one-shot container: anything written outside a
# mounted volume disappears when it exits (succeeded or not), there is no automatic
# retry, the node is shared, clocks cannot be pinned, and the driver is patched by the
# platform.  This script turns that into the contract the project already relies on:
# exact commit, recorded substrate, per-run caches, results on a persistent volume,
# and a copy + SHA256SUMS on the object volume EVEN WHEN THE TARGET FAILS (trap).
# Plan: handoff-report/vessl_operating_model_2026-10-02.md.
#
# The image is PUBLIC and holds only pristine SGLang; this commit's engine code is installed at
# start by scripts/vessl/install_runtime.sh (step 2).
#
# It is launched by scripts/vessl/launch.sh, which uploads it next to the repo bundle,
# so the Job command is just `bash /io/code/<commit>/job_entry.sh`.
#
# REQUIRED ENV (set by launch.sh with `vesslctl job create -e`)
#   PDMUX_COMMIT     full commit hash to check out from the uploaded bundle
#   PDMUX_JOB_NAME   unique job name (also the run directory name)
#   PDMUX_CAMPAIGN   campaign label (recorded only; result paths come from the target)
#   PDMUX_TARGET     repo-relative script to run (e.g. workspace/engine-port/scripts/...)
#   PDMUX_IMAGE      image reference incl. @sha256 digest (the container cannot see it)
#   PDMUX_SPEC       resource-spec slug (recorded only)
# OPTIONAL
#   PDMUX_ARRAY_SPEC run the target through scripts/run/array_runner.sh with this spec
#   PDMUX_IO / PDMUX_DATA  mount points (default /io = object volume, /data = cluster volume)
#
# NEVER dump the whole environment: VESSLCTL_ACCESS_TOKEN and secrets (HF_TOKEN) are in it.
set -uo pipefail

: "${PDMUX_COMMIT:?}" "${PDMUX_JOB_NAME:?}" "${PDMUX_CAMPAIGN:?}" "${PDMUX_TARGET:?}" \
  "${PDMUX_IMAGE:?}" "${PDMUX_SPEC:?}"
IO="${PDMUX_IO:-/io}"
DATA="${PDMUX_DATA:-/data}"
OPT="${PDMUX_OPT:-/opt/pdmux}"                    # image layout root (override only for local tests)
PROJECT="${OPT}/prefill-layer-alloc"              # same path the image already uses
RUN="${DATA}/runs/${PDMUX_JOB_NAME}"              # persistent while the job runs
OUT="${IO}/results/${PDMUX_JOB_NAME}"             # what launch/fetch.sh downloads
CODE="${IO}/code/${PDMUX_COMMIT}"
# CPU-only mode (2026-10-16): CPU resource spec or PDMUX_CPU_ONLY=1.  Skips the GPU substrate stamp, clock and
# energy samplers (step 3) and records that in meta/CPU_ONLY.txt.  GPU Jobs (neither condition) run the original path.
cpu_only=0
if [[ "${PDMUX_CPU_ONLY:-}" == 1 || "${PDMUX_SPEC}" == resourcespec-a100cpu ]]; then cpu_only=1; fi

if [[ -e "${OUT}/DONE" ]]; then
  echo "ERROR: ${OUT}/DONE already exists; job names must be unique" >&2
  exit 2
fi
mkdir -p "${RUN}"/{meta,logs,cache} "${OUT}"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# ---- progress visible in `vesslctl job logs <slug>` (container stdout) ------------------------------
#   [PHASE] lines mark each job_entry stage; [T] lines mirror the target's stdout/stderr (the file
#   logs/target.log stays the complete record; known-noise lines are not mirrored); [HB] heartbeat every
#   PDMUX_HEARTBEAT_S (default 300 s, 0 = off) names the newest result file this run wrote and its last line.
#   PDMUX_LOG_MIRROR=0 restores the old file-only behaviour.  Mirroring never changes the target's exit code.
phase() { echo "[PHASE $(utc)] $*"; }
hb_pid=""
echo "start_utc=$(utc)" > "${RUN}/meta/timing.txt"
clk_pid=""
energy_pid=""

finish() {
  local rc=$?
  [[ -n "${hb_pid}" ]] && kill "${hb_pid}" 2>/dev/null
  phase "finish rc=${rc} (exporting results; DONE written last)"
  [[ -n "${clk_pid}" ]] && kill "${clk_pid}" 2>/dev/null
  [[ -n "${energy_pid}" ]] && kill "${energy_pid}" 2>/dev/null
  echo "end_utc=$(utc)" >> "${RUN}/meta/timing.txt"
  { echo "# at end $(utc)"; cat /sys/fs/cgroup/cpu.stat 2>/dev/null; } >> "${RUN}/meta/cgroup_cpu.txt"
  echo "${rc}" > "${RUN}/meta/exit_code"
  python3 - "${RUN}/meta" <<'PY' || true
import csv, json, pathlib, sys
meta = pathlib.Path(sys.argv[1]); p = meta / "clk.csv"
rows = list(csv.reader(p.open())) if p.exists() else []
body = [r for r in rows[1:] if len(r) >= 5]
def _throttled(v):  # bit 0x1 = GpuIdle (not a throttle); any other active bit counts
    try:
        return (int(v.strip(), 16) & ~0x1) != 0
    except ValueError:
        return False
active = [r for r in body if _throttled(r[4])]
summary = {"samples": len(body), "throttle_active_samples": len(active),
           "throttle_flag": bool(active)}
# Multi-GPU (2026-10-12): clk.csv has the GPU `index` as column 6 (appended, so columns 1-5 keep
# their positions).  The three keys above stay the all-GPU totals (identical to before for 1 GPU);
# per_gpu adds the same counts per index so a throttled GPU on an x8 node is visible.
if body and all(len(r) >= 6 for r in body):
    per = {}
    for r in body:
        d = per.setdefault(r[5].strip(), {"samples": 0, "throttle_active_samples": 0})
        d["samples"] += 1
        d["throttle_active_samples"] += int(_throttled(r[4]))
    for d in per.values():
        d["throttle_flag"] = d["throttle_active_samples"] > 0
    summary["per_gpu"] = dict(sorted(per.items(), key=lambda kv: (len(kv[0]), kv[0])))
(meta / "clk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
  # Export: run metadata + logs, and only the result files this run created/changed.
  mkdir -p "${OUT}/meta" "${OUT}/logs" "${OUT}/results"
  cp -a "${RUN}/meta/." "${OUT}/meta/"
  cp -a "${RUN}/logs/." "${OUT}/logs/"
  if [[ -d "${RUN}/results" && -f "${RUN}/meta/start.marker" ]]; then
    (cd "${RUN}/results" && find . -type f -newer "${RUN}/meta/start.marker" \
       ! -path '*/.triton_cache/*' -print0) \
      | rsync -a --from0 --files-from=- "${RUN}/results/" "${OUT}/results/"
  fi
  (cd "${OUT}" && find . -type f ! -name SHA256SUMS ! -name DONE -print0 | sort -z \
     | xargs -0 -r sha256sum > SHA256SUMS)
  echo "exit_code=${rc} $(utc)" > "${OUT}/DONE"           # written last
  exit "${rc}"
}
trap finish EXIT

# ---- 1. exact commit from the uploaded bundle --------------------------------------
(cd "${CODE}" && sha256sum -c repo.bundle.sha256) || { echo "ERROR: bundle checksum" >&2; exit 3; }
rm -rf "${PROJECT}.new"
git clone -q --no-checkout "${CODE}/repo.bundle" "${PROJECT}.new" \
  && git -C "${PROJECT}.new" checkout -q "${PDMUX_COMMIT}" \
  || { echo "ERROR: checkout ${PDMUX_COMMIT}" >&2; exit 3; }
[[ "$(git -C "${PROJECT}.new" rev-parse HEAD)" == "${PDMUX_COMMIT}" ]] || exit 3
rm -rf "${PROJECT}"                                   # empty placeholder in the public image
mv "${PROJECT}.new" "${PROJECT}"
git config --global --add safe.directory '*'

# results/ -> persistent volume, keeping tracked files (preregs, campaign scripts live there)
mkdir -p "${RUN}/results"
cp -a "${PROJECT}/workspace/engine-port/results/." "${RUN}/results/"
rm -rf "${PROJECT}/workspace/engine-port/results"
ln -s "${RUN}/results" "${PROJECT}/workspace/engine-port/results"
ln -sfn "${DATA}/hf" "${PROJECT}/hf_cache"                # legacy $ROOT/hf_cache convention
touch "${RUN}/meta/start.marker"
phase "checkout ${PDMUX_COMMIT:0:7} ok; results -> ${RUN}/results"

# ---- 2. install this commit's engine code into the PUBLIC (pristine-SGLang) image ---
export PDMUX_ROOT="${OPT}" PDMUX_PROJECT_ROOT="${PROJECT}"
if [[ -n "${PDMUX_TEST_SKIP_SYNC:-}" ]]; then        # local mock test only -- flagged in meta
  echo "install skipped (PDMUX_TEST_SKIP_SYNC) -- NOT A VALID MEASUREMENT RUN" > "${RUN}/meta/SYNC_SKIPPED_TEST"
else
  phase "install_runtime start"
  bash "${PROJECT}/workspace/engine-port/scripts/vessl/install_runtime.sh" \
    "${RUN}/meta/runtime_source_manifest.sha256" > "${RUN}/logs/install_runtime.log" 2>&1 \
    || { echo "ERROR: install_runtime.sh (see logs/install_runtime.log)" >&2; tail -20 "${RUN}/logs/install_runtime.log"; exit 4; }
  phase "install_runtime done"
fi

# ---- 3. substrate stamp (B0) + clock/throttle trace ---------------------------------
if [[ "${cpu_only}" == 1 ]]; then
  # No GPU in this container: libcuda.so.1 is absent from the loader path, so `import sgl_kernel` fails
  # (probe 2026-10-16, pdmux-build).  The image ships the CUDA forward-compat libcuda; put it on the path
  # only if it exists, and record what was done.
  compat=/usr/local/cuda-13.0/compat
  if [[ -e "${compat}/libcuda.so.1" ]]; then
    export LD_LIBRARY_PATH="${compat}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; compat_note="added ${compat}"
  else compat_note="NOT added (${compat}/libcuda.so.1 missing)"; fi
  { echo "cpu_only=1 reason=$([[ "${PDMUX_CPU_ONLY:-}" == 1 ]] && echo PDMUX_CPU_ONLY || echo cpu_spec) spec=${PDMUX_SPEC}"
    echo "skipped: nvidia-smi substrate stamp, nvcc, topo, substrate_history.tsv row, clk sampler, energy sampler"
    echo "LD_LIBRARY_PATH compat: ${compat_note}"; } > "${RUN}/meta/CPU_ONLY.txt"
  phase "CPU-only mode: GPU substrate stamp and clock/energy samplers skipped; LD_LIBRARY_PATH compat ${compat_note}"
  # Delimiter CPU_PY, not PY (2026-10-17): tests/test_multigpu_launcher._heredoc("substrate.json") takes the FIRST
  # PY-delimited heredoc naming substrate.json; it must resolve to the GPU stamp below, not to this CPU record.
  python3 - "${RUN}/meta" <<'CPU_PY'
import json, os, pathlib, platform, sys
rec = {"cpu_only": True, "gpu": None, "hostname": platform.node(), "commit": os.environ["PDMUX_COMMIT"],
       "image": os.environ["PDMUX_IMAGE"], "spec": os.environ["PDMUX_SPEC"], "job_name": os.environ["PDMUX_JOB_NAME"],
       "campaign": os.environ["PDMUX_CAMPAIGN"], "target": os.environ["PDMUX_TARGET"],
       "array_spec": os.environ.get("PDMUX_ARRAY_SPEC", "")}
(pathlib.Path(sys.argv[1]) / "substrate.json").write_text(json.dumps(rec, indent=2) + "\n")
CPU_PY
  { echo "# at start $(utc)"; echo "nproc=$(nproc)"; echo "cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)";
    cat /sys/fs/cgroup/cpu.stat 2>/dev/null; } > "${RUN}/meta/cgroup_cpu.txt"
else
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current \
  --format=csv,noheader > "${RUN}/meta/gpu.csv" 2>&1
nvcc --version > "${RUN}/meta/nvcc.txt" 2>&1
# Multi-GPU (2026-10-12): device list + interconnect topology of THIS container's GPUs, raw
# (the topo parser lives in multigpu_probe_helper.py; raw text is the record).
nvidia-smi -L > "${RUN}/meta/gpu_list.txt" 2>&1
nvidia-smi topo -m > "${RUN}/meta/topo_m.txt" 2>&1
python3 - "${RUN}/meta" <<'PY'
import json, os, pathlib, platform, sys
meta = pathlib.Path(sys.argv[1])
keys = ["name", "uuid", "driver_version", "compute_cap", "memory_total", "clocks_max_sm",
        "power_limit", "mig_mode"]
# One row per visible GPU.  rec["gpu"] stays the FIRST row (every 1-GPU consumer reads that);
# gpus / gpu_count / gpu_uuids are the all-GPU view (the old code recorded GPU 0 only).
rows = [[c.strip() for c in ln.split(",")]
        for ln in (meta / "gpu.csv").read_text().splitlines() if ln.strip()]
gpu = rows[0]
gpus = [dict(zip(keys, r), index=i) for i, r in enumerate(rows)]
rec = {"gpu": dict(zip(keys, gpu)), "gpu_count": len(gpus), "gpus": gpus,
       "gpu_uuids": [g["uuid"] for g in gpus], "hostname": platform.node(),
       "commit": os.environ["PDMUX_COMMIT"], "image": os.environ["PDMUX_IMAGE"],
       "spec": os.environ["PDMUX_SPEC"], "job_name": os.environ["PDMUX_JOB_NAME"],
       "campaign": os.environ["PDMUX_CAMPAIGN"], "target": os.environ["PDMUX_TARGET"],
       "array_spec": os.environ.get("PDMUX_ARRAY_SPEC", "")}
import hashlib
opt = pathlib.Path(os.environ.get("PDMUX_OPT", "/opt/pdmux"))
for f in ("pip_freeze.txt", "python_torch.txt"):
    p = opt / f
    rec["image_" + f + "_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
(meta / "substrate.json").write_text(json.dumps(rec, indent=2) + "\n")
PY
hist="${DATA}/substrate_history.tsv"
read -r uuid drv < <(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(",".join(r.get("gpu_uuids") or [r["gpu"]["uuid"]]), r["gpu"]["driver_version"])' "${RUN}/meta/substrate.json")
if [[ -f "${hist}" ]]; then
  prev_drv="$(awk -F'\t' -v c="${PDMUX_CAMPAIGN}" '$2==c{d=$5} END{print d}' "${hist}")"
  if [[ -n "${prev_drv}" && "${prev_drv}" != "${drv}" ]]; then
    echo "WARNING: driver ${prev_drv} -> ${drv} within campaign ${PDMUX_CAMPAIGN} (substrate mix)" \
      | tee "${RUN}/meta/SUBSTRATE_WARNING"
  fi
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(utc)" "${PDMUX_CAMPAIGN}" "${PDMUX_JOB_NAME}" "${uuid}" "${drv}" \
  "$(hostname)" >> "${hist}"
# CPU-side substrate (λ0 VESSL result audit C8, 2026-10-04): the cgroup CFS quota (cpu.max) and the
# visible core count differ from KISTI's cpuset; record quota + throttle counters (cpu.stat
# nr_throttled/throttled_usec) at start and end so CPU throttling is visible, not inferred.
{ echo "# at start $(utc)"; echo "nproc=$(nproc)"; echo "cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)";
  cat /sys/fs/cgroup/cpu.stat 2>/dev/null; } > "${RUN}/meta/cgroup_cpu.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active,index \
  --format=csv -lms 100 > "${RUN}/meta/clk.csv" 2>/dev/null &
clk_pid=$!
# GPU power/energy (NVML, read-only, no extra privilege).  Three files:
#  meta/energy.csv        : board energy counter (mJ) polled every 100 ms (counter updates ~100 ms but
#                           integrates continuously -> interval totals exact; probe job-absefkt9e88x).
#                           Columns epoch_s, mono_s, energy_mJ: mono_s = time.monotonic() of the same read, i.e.
#                           CLOCK_MONOTONIC, the clock of engine telemetry (perf_counter) -- an exact bridge from
#                           power/energy (epoch) to telemetry (monotonic) without srv.log pairing.
#  meta/power_samples.csv : NVML buffered power samples (nvmlDeviceGetSamples NVML_TOTAL_POWER_SAMPLES,
#                           driver timestamps, ~20 ms apart, step response 12-34 ms; probe job-tkgxusf12n0v)
#                           drained every 200 ms -- the only sub-100 ms power signal on this substrate
#  meta/sm_clk_samples.csv: NVML buffered SM clock samples (NVML_PROCESSOR_CLK_SAMPLES, MHz) drained the same
#                           way, so power-cap clock drops can be aligned with power at the same resolution
# Best-effort: a missing pynvml / unsupported call writes a note and never fails the job.
# Cost (local RTX 5060 Ti, 2026-10-08): ~8 % of one CPU core, ~34 MB RSS.  Opt out with PDMUX_ENERGY_SAMPLER=0.
if [[ "${PDMUX_ENERGY_SAMPLER:-1}" != "0" ]]; then   # opt-out only (unset = on); 0 disables the whole block
python3 - "${RUN}/meta/energy.csv" "${RUN}/meta/power_samples.csv" "${RUN}/meta/sm_clk_samples.csv" > "${RUN}/meta/energy_sampler.log" 2>&1 <<'PY' &
import sys, time
out, out_s, out_c = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import pynvml as n
    n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(0)
    n.nvmlDeviceGetTotalEnergyConsumption(h)
except Exception as e:  # noqa: BLE001
    print("ENERGY_SAMPLER_UNAVAILABLE", repr(e), flush=True); raise SystemExit(0)
have_s, last = True, 0
try:
    n.nvmlDeviceGetSamples(h, n.NVML_TOTAL_POWER_SAMPLES, 0)
except n.NVMLError_NotFound:
    pass                                     # supported, buffer momentarily empty
except Exception as e:  # noqa: BLE001
    have_s = False; print("POWER_SAMPLES_UNAVAILABLE", repr(e), flush=True)
fs = open(out_s, "w", buffering=1) if have_s else None
if fs:
    fs.write("t_us,power_mW\n")
have_c, last_c = True, 0
try:
    n.nvmlDeviceGetSamples(h, n.NVML_PROCESSOR_CLK_SAMPLES, 0)
except n.NVMLError_NotFound:
    pass
except Exception as e:  # noqa: BLE001
    have_c = False; print("CLK_SAMPLES_UNAVAILABLE", repr(e), flush=True)
fc = open(out_c, "w", buffering=1) if have_c else None
if fc:
    fc.write("t_us,sm_mhz\n")
with open(out, "w", buffering=1) as f:
    f.write("epoch_s,mono_s,energy_mJ\n")
    k = 0; nerr = 0
    while True:
        try:
            e_ = n.nvmlDeviceGetTotalEnergyConsumption(h)
            f.write(f"{time.time():.6f},{time.monotonic():.6f},{e_}\n")
        except Exception as e:  # noqa: BLE001
            nerr += 1
            if nerr <= 20:                    # cap the log: a persistent failure must not flood the run dir
                print("ENERGY_READ_ERROR", repr(e), flush=True)
        if fs and k % 2 == 0:
            try:
                _, smp = n.nvmlDeviceGetSamples(h, n.NVML_TOTAL_POWER_SAMPLES, last)
                for x in smp:
                    if x.timeStamp > last:
                        fs.write(f"{x.timeStamp},{x.sampleValue.uiVal}\n"); last = x.timeStamp
            except n.NVMLError_NotFound:      # no new sample since `last` -- normal, not an error
                pass
            except Exception as e:  # noqa: BLE001
                nerr += 1
                if nerr <= 20:
                    print("POWER_SAMPLES_READ_ERROR", repr(e), flush=True)
        if fc and k % 2 == 0:
            try:
                _, smp = n.nvmlDeviceGetSamples(h, n.NVML_PROCESSOR_CLK_SAMPLES, last_c)
                for x in smp:
                    if x.timeStamp > last_c:
                        fc.write(f"{x.timeStamp},{x.sampleValue.uiVal}\n"); last_c = x.timeStamp
            except n.NVMLError_NotFound:
                pass
            except Exception as e:  # noqa: BLE001
                nerr += 1
                if nerr <= 20:
                    print("CLK_SAMPLES_READ_ERROR", repr(e), flush=True)
        k += 1
        time.sleep(0.1)
PY
energy_pid=$!
else
  echo "energy sampler disabled (PDMUX_ENERGY_SAMPLER=0)" > "${RUN}/meta/energy_sampler.log"
fi
fi   # cpu_only

# ---- 4. run the target ---------------------------------------------------------------
export HF_HOME="${DATA}/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${RUN}/cache/triton" XDG_CACHE_HOME="${RUN}/cache/xdg"
export FLASHINFER_WORKSPACE_BASE="${RUN}/cache/flashinfer"   # per-run JIT caches (race precedent)
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}"
cd "${PROJECT}"
target="${PROJECT}/${PDMUX_TARGET}"
[[ -f "${target}" ]] || { echo "ERROR: target not found: ${PDMUX_TARGET}" >&2; exit 2; }
run_target() {
  if [[ -n "${PDMUX_ARRAY_SPEC:-}" ]]; then
    PDMUX_RUN_LOG_DIR="${RUN}/logs/array" \
      bash "${PROJECT}/workspace/engine-port/scripts/run/array_runner.sh" "${target}" "${PDMUX_ARRAY_SPEC}"
  else
    bash "${target}"
  fi
}
# Logging knobs are job_entry-only: read them here, record them (they add co-resident processes during the
# measurement -> substrate record), and remove them from the environment so a campaign script's knob allowlist
# (n2/n3_capacity.sh: ABORT_ENV unregistered knob) never sees them (review 2026-10-12 D3, D7).
log_mirror="${PDMUX_LOG_MIRROR:-1}"; hb_s="${PDMUX_HEARTBEAT_S:-300}"
unset PDMUX_LOG_MIRROR PDMUX_HEARTBEAT_S
echo "log_mirror=${log_mirror} heartbeat_s=${hb_s}" > "${RUN}/meta/log_mirror.txt"
heartbeat() {   # newest file this run wrote under results/ (cells, marks, ledgers) + its last line
  local t0=$SECONDS newest line n listing
  while sleep "${hb_s}"; do
    listing=$(find "${RUN}/results" -type f -newer "${RUN}/meta/start.marker" ! -path '*/.triton_cache/*' \
                -printf '%T@ %p\n' 2>/dev/null | sort -n)            # one scan per beat
    n=$(printf '%s' "${listing}" | grep -c . || true)
    newest=$(printf '%s\n' "${listing}" | tail -1 | cut -d' ' -f2-)
    line=""; [[ -n "${newest}" ]] && line=$(tail -c 400 "${newest}" 2>/dev/null | tail -1 | tr -d '\r' | cut -c1-240)
    echo "[HB $(utc) +$((SECONDS - t0))s] new_files=${n} newest=${newest#"${RUN}/results/"} :: ${line}"
  done
}
phase "target start: ${PDMUX_TARGET}"
if [[ "${log_mirror}" == 0 ]]; then
  run_target > "${RUN}/logs/target.log" 2>&1
  target_rc=$?
else
  [[ "${hb_s}" != 0 ]] && { heartbeat & hb_pid=$!; }
  # tee -p (--output-error=warn-nopipe): if the mirror (grep/sed) dies, tee keeps writing target.log and the
  # target never gets SIGPIPE (review 2026-10-12 D2).  Fallback to plain tee only if this tee lacks -p.
  tee_opt=(-p); tee -p </dev/null >/dev/null 2>&1 || tee_opt=()
  run_target 2>&1 | tee "${tee_opt[@]}" "${RUN}/logs/target.log" \
    | { grep --line-buffered -v -e 'pynvml package is deprecated' -e 'import pynvml  # type: ignore' \
        | sed -u 's/^/[T] /'; true; }
  target_rc=${PIPESTATUS[0]}
fi
phase "target end rc=${target_rc}"
exit "${target_rc}"
