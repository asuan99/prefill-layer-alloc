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

if [[ -e "${OUT}/DONE" ]]; then
  echo "ERROR: ${OUT}/DONE already exists; job names must be unique" >&2
  exit 2
fi
mkdir -p "${RUN}"/{meta,logs,cache} "${OUT}"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "start_utc=$(utc)" > "${RUN}/meta/timing.txt"
clk_pid=""

finish() {
  local rc=$?
  [[ -n "${clk_pid}" ]] && kill "${clk_pid}" 2>/dev/null
  echo "end_utc=$(utc)" >> "${RUN}/meta/timing.txt"
  echo "${rc}" > "${RUN}/meta/exit_code"
  python3 - "${RUN}/meta" <<'PY' || true
import csv, json, pathlib, sys
meta = pathlib.Path(sys.argv[1]); p = meta / "clk.csv"
rows = list(csv.reader(p.open())) if p.exists() else []
body = [r for r in rows[1:] if len(r) >= 5]
active = [r for r in body if r[4].strip() not in ("0x0000000000000000", "0x0", "Not Active")]
(meta / "clk_summary.json").write_text(json.dumps(
    {"samples": len(body), "throttle_active_samples": len(active),
     "throttle_flag": bool(active)}, indent=2) + "\n")
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
rm -rf "${OPT}/_image_project" && mv "${PROJECT}" "${OPT}/_image_project"
mv "${PROJECT}.new" "${PROJECT}"
git config --global --add safe.directory '*'

# results/ -> persistent volume, keeping tracked files (preregs, campaign scripts live there)
mkdir -p "${RUN}/results"
cp -a "${PROJECT}/workspace/engine-port/results/." "${RUN}/results/"
rm -rf "${PROJECT}/workspace/engine-port/results"
ln -s "${RUN}/results" "${PROJECT}/workspace/engine-port/results"
ln -sfn "${DATA}/hf" "${PROJECT}/hf_cache"                # legacy $ROOT/hf_cache convention
touch "${RUN}/meta/start.marker"

# ---- 2. engine tree = this commit's src (image was built from an earlier commit) ---
export PDMUX_ROOT="${OPT}" PDMUX_PROJECT_ROOT="${PROJECT}"
if [[ -n "${PDMUX_TEST_SKIP_SYNC:-}" ]]; then        # local mock test only -- flagged in meta
  echo "sync skipped (PDMUX_TEST_SKIP_SYNC) -- NOT A VALID MEASUREMENT RUN" > "${RUN}/meta/SYNC_SKIPPED_TEST"
else
  bash "${PROJECT}/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
    "${RUN}/meta/runtime_source_manifest.sha256" > "${RUN}/logs/sync_engine_tree.log" 2>&1 \
    || { echo "ERROR: sync_engine_tree.sh (see logs)" >&2; exit 4; }
fi
cp "${OPT}/runtime_source_manifest.sha256" "${RUN}/meta/image_runtime_source_manifest.sha256" 2>/dev/null
if cmp -s "${PROJECT}/workspace/engine-port/env/devtree_manual_edits.patch" \
          "${OPT}"/_image_project/workspace/engine-port/env/devtree_manual_edits.patch; then
  echo same > "${RUN}/meta/manual_edits_patch_vs_image"
else
  echo DIFFERENT > "${RUN}/meta/manual_edits_patch_vs_image"   # image must be rebuilt
  echo "ERROR: devtree_manual_edits.patch changed since the image was built" >&2; exit 4
fi

# ---- 3. substrate stamp (B0) + clock/throttle trace ---------------------------------
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current \
  --format=csv,noheader > "${RUN}/meta/gpu.csv" 2>&1
nvcc --version > "${RUN}/meta/nvcc.txt" 2>&1
python3 - "${RUN}/meta" <<'PY'
import json, os, pathlib, platform, sys
meta = pathlib.Path(sys.argv[1])
gpu = [c.strip() for c in (meta / "gpu.csv").read_text().splitlines()[0].split(",")]
keys = ["name", "uuid", "driver_version", "compute_cap", "memory_total", "clocks_max_sm",
        "power_limit", "mig_mode"]
rec = {"gpu": dict(zip(keys, gpu)), "hostname": platform.node(),
       "commit": os.environ["PDMUX_COMMIT"], "image": os.environ["PDMUX_IMAGE"],
       "spec": os.environ["PDMUX_SPEC"], "job_name": os.environ["PDMUX_JOB_NAME"],
       "campaign": os.environ["PDMUX_CAMPAIGN"], "target": os.environ["PDMUX_TARGET"],
       "array_spec": os.environ.get("PDMUX_ARRAY_SPEC", "")}
import hashlib
opt = pathlib.Path(os.environ.get("PDMUX_OPT", "/opt/pdmux"))
for f in ("pip_freeze.txt", "python_torch.txt", "runtime_source_manifest.sha256"):
    p = opt / f
    rec["image_" + f + "_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
(meta / "substrate.json").write_text(json.dumps(rec, indent=2) + "\n")
PY
hist="${DATA}/substrate_history.tsv"
read -r uuid drv < <(python3 -c 'import json,sys; g=json.load(open(sys.argv[1]))["gpu"]; print(g["uuid"], g["driver_version"])' "${RUN}/meta/substrate.json")
if [[ -f "${hist}" ]]; then
  prev_drv="$(awk -F'\t' -v c="${PDMUX_CAMPAIGN}" '$2==c{d=$5} END{print d}' "${hist}")"
  if [[ -n "${prev_drv}" && "${prev_drv}" != "${drv}" ]]; then
    echo "WARNING: driver ${prev_drv} -> ${drv} within campaign ${PDMUX_CAMPAIGN} (substrate mix)" \
      | tee "${RUN}/meta/SUBSTRATE_WARNING"
  fi
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(utc)" "${PDMUX_CAMPAIGN}" "${PDMUX_JOB_NAME}" "${uuid}" "${drv}" \
  "$(hostname)" >> "${hist}"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 100 > "${RUN}/meta/clk.csv" 2>/dev/null &
clk_pid=$!

# ---- 4. run the target ---------------------------------------------------------------
export HF_HOME="${DATA}/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${RUN}/cache/triton" XDG_CACHE_HOME="${RUN}/cache/xdg"
export FLASHINFER_WORKSPACE_BASE="${RUN}/cache/flashinfer"   # per-run JIT caches (race precedent)
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}"
cd "${PROJECT}"
target="${PROJECT}/${PDMUX_TARGET}"
[[ -f "${target}" ]] || { echo "ERROR: target not found: ${PDMUX_TARGET}" >&2; exit 2; }
if [[ -n "${PDMUX_ARRAY_SPEC:-}" ]]; then
  PDMUX_RUN_LOG_DIR="${RUN}/logs/array" \
    bash "${PROJECT}/workspace/engine-port/scripts/run/array_runner.sh" "${target}" "${PDMUX_ARRAY_SPEC}" \
    > "${RUN}/logs/target.log" 2>&1
else
  bash "${target}" > "${RUN}/logs/target.log" 2>&1
fi
