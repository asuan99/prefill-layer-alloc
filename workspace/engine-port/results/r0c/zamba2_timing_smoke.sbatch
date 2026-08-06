#!/bin/bash
#SBATCH --job-name=zb2smoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
# ============================================================================
# GPU CORRECTNESS GATE for the 2026-08-04 Zamba2 instrumentation rework.
# PREPARED, NOT SUBMITTED -- submission is experiment-runner's call.
#
# This proves the three GPU-side gates; it measures NOTHING and must not be
# quoted for performance.
#
#   GATE 3a  OFF-path output equivalence, cudagraph ON (the operating point):
#            pre-patch zamba2.py vs patched zamba2.py, greedy, byte-identical.
#   GATE 3b  same at cudagraph OFF (eager, the P5 harness condition).
#   GATE 3c  instrumentation ON must not change the output either
#            (patched/eager/timing-OFF vs patched/eager/timing-ON).
#   GATE C   with timing ON, every emitted block must satisfy
#            closure = sum(buckets)/loop >= SGLANG_ZAMBA_CLOSURE_MIN, and zero
#            ZBLT_CLOSURE_FAIL lines. Also checks the block accumulator resets
#            (blk increments while blk_n stays == SGLANG_ZAMBA_TIMING_EVERY).
#
# Arms boot 5 servers sequentially. The pre-patch file is extracted from a PINNED
# commit and its sha256 is verified, so "before" is not whatever happens to be in
# the tree. The dev tree is shared, so each arm HOLDS the same flock that
# sync_engine_tree.sh uses for its whole duration (this blocks other jobs' sync
# for the length of the job -- schedule accordingly), the installed file's sha256
# is re-checked after each arm, and a trap restores the patched file on any exit.
#
# Run:  sbatch zamba2_timing_smoke.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

REPO=/scratch/ehmoon/whlee/prefill-layer-alloc
DEV=/scratch/ehmoon/whlee/sglang_engine_dev/python
OUT=$REPO/workspace/engine-port/results/r0c
HERE=$REPO/workspace/engine-port/triage
export HF_HOME=$REPO/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$HERE/.triton_cache"
MODEL=Zyphra/Zamba2-2.7B

PREPATCH_COMMIT=6c06da8bec42c9b14a4880695be4a0426a48da37
# "before" must be a specific artifact, so the pre-patch file identity is PINNED.
# The post-patch hash is only RECORDED (it moves with every later edit; pinning it
# would turn ordinary maintenance into a spurious abort).
PREPATCH_SHA=6ba79046499f3954b73d6ed8cb3a76add2bdd3022beff7ccca8f9fd3ecd5f21b

JOB=${SLURM_JOB_ID:-local}
WORK=$OUT/.zb2smoke_$JOB; mkdir -p "$WORK"
RESULT=$OUT/zb2smoke_result_${JOB}.txt; : > "$RESULT"
TARGET=$DEV/sglang/srt/models/zamba2.py
PATCHED=$WORK/zamba2_patched.py
LEGACY=$WORK/zamba2_prepatch.py
LOCK=${PDMUX_SYNC_LOCK:-/scratch/ehmoon/whlee/sglang_engine_dev/.pdmux_sync.lock}

cp "$REPO/workspace/engine-port/src/models/zamba2.py" "$PATCHED"
( cd "$REPO" && git show "$PREPATCH_COMMIT:workspace/engine-port/src/models/zamba2.py" ) > "$LEGACY"

restore() { cp "$PATCHED" "$TARGET" 2>/dev/null; }
trap restore EXIT

log() { echo "$@" | tee -a "$RESULT"; }

got_pre=$(sha256sum "$LEGACY" | cut -d' ' -f1)
got_post=$(sha256sum "$PATCHED" | cut -d' ' -f1)
log "HEADER job=$JOB model=$MODEL"
log "HEADER prepatch_commit=$PREPATCH_COMMIT prepatch_sha256=$got_pre expect=$PREPATCH_SHA"
log "HEADER patched_sha256=$got_post (recorded, not pinned)"
if [ "$got_pre" != "$PREPATCH_SHA" ]; then
  log "ABORT: pre-patch file identity check failed (expected $PREPATCH_SHA)"
  exit 2
fi
if [ "$got_pre" = "$got_post" ]; then
  log "ABORT: patched file == pre-patch file; nothing to compare"
  exit 2
fi

# 8 fixed prompts, greedy (temperature=0), sent ONE AT A TIME so batch
# composition -- and therefore the output -- is deterministic across boots.
cat > "$WORK/prompts.json" <<'PY'
["The capital of France is",
 "In 1969 the first humans landed on",
 "Write the first five prime numbers:",
 "A hybrid language model combines attention with",
 "def fibonacci(n):",
 "The three primary colors are",
 "Once upon a time in a distant galaxy",
 "State-space models process sequences by"]
PY

run_arm () {   # $1=arm name  $2=model file  $3=extra server flags  $4=timing(0|1)
  local arm=$1 mfile=$2 flags=$3 timing=$4
  local port=$((35100 + RANDOM % 400))
  local srvlog="$OUT/zb2smoke_srv_${arm}_${JOB}.log"
  local outfile="$WORK/gen_${arm}.txt"

  # Hold the sync lock for the WHOLE arm, not just the copy: another job running
  # sync_engine_tree.sh between the copy and the model import would silently
  # reinstall the patched file, turning the legacy arm into a second patched arm
  # and making the comparison vacuously PASS. The post-arm sha re-check below is
  # the backstop if the lock is ever bypassed.
  exec 9>"$LOCK"
  flock 9 || { log "ARM $arm FAIL (could not take $LOCK)"; return 1; }
  cp "$mfile" "$TARGET"
  local sha_before; sha_before=$(sha256sum "$TARGET" | cut -d' ' -f1)
  log "ARM $arm file_sha=$sha_before timing=$timing flags='$flags'"

  if [ "$timing" = 1 ]; then
    export SGLANG_ZAMBA_TIMING=1 SGLANG_ZAMBA_TIMING_EVERY=8 SGLANG_ZAMBA_CLOSURE_MIN=0.95
    export PDMUX_FIXED_DECODE_SM_FILE="$WORK/.smoke_sm"
    echo "full@smoke" > "$WORK/.smoke_sm"
  else
    unset SGLANG_ZAMBA_TIMING SGLANG_ZAMBA_TIMING_EVERY SGLANG_ZAMBA_CLOSURE_MIN
    unset PDMUX_FIXED_DECODE_SM_FILE
  fi

  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend triton --disable-piecewise-cuda-graph \
    --disable-radix-cache --mem-fraction-static 0.80 --context-length 4096 \
    $flags --host 127.0.0.1 --port $port > "$srvlog" 2>&1 &
  local srv=$!
  local ok=0
  for i in $(seq 1 220); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $srv 2>/dev/null || break; sleep 5
  done
  log "ARM $arm boot_ok=$ok port=$port"
  if [ "$ok" = 1 ]; then
    python3 - "$port" "$WORK/prompts.json" "$outfile" <<'PY'
import json, sys, urllib.request
port, pfile, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.load(open(pfile))
with open(out, "w") as f:
    for p in prompts:
        body = json.dumps({"text": p, "sampling_params": {
            "max_new_tokens": 48, "temperature": 0.0, "top_p": 1.0,
            "top_k": 1, "ignore_eos": True}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", body,
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        f.write(json.dumps({"prompt": p, "text": r["text"]}, ensure_ascii=False) + "\n")
PY
    log "ARM $arm gen_lines=$(wc -l < "$outfile") gen_sha=$(sha256sum "$outfile" | cut -d' ' -f1)"
  fi
  kill $srv 2>/dev/null; pkill -9 -f "launch_server.*--port $port" 2>/dev/null; sleep 3
  local sha_after; sha_after=$(sha256sum "$TARGET" | cut -d' ' -f1)
  flock -u 9; exec 9>&-
  if [ "$sha_before" != "$sha_after" ]; then
    log "ARM $arm INVALID: model file changed mid-arm ($sha_before -> $sha_after)"
    rm -f "$outfile"
  fi
}

# ---- arms -----------------------------------------------------------------
run_arm legacy_cg  "$LEGACY"  ""                      0
run_arm patched_cg "$PATCHED" ""                      0
run_arm legacy_eg  "$LEGACY"  "--disable-cuda-graph"  0
run_arm patched_eg "$PATCHED" "--disable-cuda-graph"  0
run_arm patched_tm "$PATCHED" "--disable-cuda-graph"  1
restore

# ---- gates ----------------------------------------------------------------
gate_diff () {  # $1=gate id  $2=arm a  $3=arm b
  local a="$WORK/gen_$2.txt" b="$WORK/gen_$3.txt"
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then log "GATE $1: FAIL (missing output: $2/$3)"; return 1; fi
  if diff -q "$a" "$b" >/dev/null; then log "GATE $1: PASS ($2 == $3, byte-identical)"; return 0
  else log "GATE $1: FAIL ($2 != $3)"; diff "$a" "$b" | head -20 | tee -a "$RESULT"; return 1; fi
}

rc=0
gate_diff 3a_offpath_cudagraph_on  legacy_cg  patched_cg || rc=1
gate_diff 3b_offpath_eager         legacy_eg  patched_eg || rc=1
gate_diff 3c_timing_on_no_effect   patched_eg patched_tm || rc=1

TMLOG="$OUT/zb2smoke_srv_patched_tm_${JOB}.log"
python3 - "$TMLOG" 0.95 8 <<'PY' | tee -a "$RESULT"
import re, sys
log, cmin, every = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
lines = [l for l in open(log, errors="replace") if "ZBLT2 mode=" in l]
fails = [l for l in open(log, errors="replace") if "ZBLT_CLOSURE_FAIL" in l]
pat = re.compile(r"blk=(\d+) blk_n=(\d+) dirty=(\d+) shape=(\d+)x(\d+) "
                 r"closure=([\d.naif-]+) closure_min=([\d.naif-]+)")
if not lines:
    print("GATE C: FAIL (no ZBLT2 emit lines -- instrumentation did not run)")
    sys.exit(0)
bad, blks, ns = [], [], []
for l in lines:
    m = pat.search(l)
    if not m:
        bad.append("unparsable: " + l.strip()[-120:]); continue
    blk, bn, dirty, _bs, _nt, clo, _clomin = m.groups()
    blks.append(int(blk)); ns.append(int(bn))
    try:
        c = float(clo)
    except ValueError:
        bad.append("non-numeric closure: " + clo); continue
    if c < cmin:
        bad.append(f"closure={c:.4f} < {cmin}")
import statistics
clos = []
for l in lines:
    m = pat.search(l)
    if m:
        try:
            clos.append(float(m.group(6)))
        except ValueError:
            pass
print(f"GATE C: emits={len(lines)} closure_fail_lines={len(fails)} "
      f"blk_max={max(blks)} blk_n_set={sorted(set(ns))}")
if clos:
    # CALIBRATION INPUT: closure < 1 also absorbs inter-span launch gaps in eager
    # mode. Read this distribution before treating 0.95 as a defect detector.
    print(f"GATE C(dist): closure min={min(clos):.4f} p50={statistics.median(clos):.4f} "
          f"max={max(clos):.4f} n={len(clos)}")
# block accumulator must RESET at emit: blk indexes advance, blk_n stays == every
reset_ok = set(ns) == {every} and max(blks) >= 1
print("GATE C(reset): " + ("PASS" if reset_ok else
      f"FAIL (blk_n should always be {every}; block never reset?)"))
print("GATE C(closure): " + ("PASS" if not bad and not fails else "FAIL"))
for b in bad[:10]:
    print("   " + b)
PY

log "SMOKE RESULT: $( [ $rc = 0 ] && echo 'output-equivalence gates PASS' || echo 'output-equivalence gates FAIL' )"
log "NOTE: this job is a correctness gate only. No timing number from it is a measurement."
exit $rc
