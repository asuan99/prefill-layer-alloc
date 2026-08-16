#!/bin/bash
#SBATCH --job-name=g16grid
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched/%x_%j.err
# ============================================================================
# G16 grid harness -- executes PREREG_G16_RULES_REV3_2026-08-16.md sec 5/8/9
# PLUS the doc's trailing addendum A (2026-08-16 stage-2/harness audit
# NO-GO fix). Do not change the design here -- if the RULES look wrong,
# REPORT it, don't patch it in this file.
#
# Separate from `sharegpt_vary_bench.sbatch` on purpose (that script is the
# R2-campaign reproduction path and is not touched by this campaign, sec5:
# "원 격자 전부 재실행... 2026-07-15~18 데이터와 섞지 않는다"). This script
# borrows sharegpt_vary_bench.sbatch's server flags and bench_serving
# invocation verbatim (sec5: "워크로드 원본 그대로... 하나라도 바꾸면 PC-A가
# 깨진다") but restructures the job as "1 job = 1 BLOCK = 7 arms x one boot
# each, sequential" instead of "1 job = 1 arm".
#
# H1-H18 crosswalk + verification results + addendum-A resolution table:
# G16_HARNESS_NOTES_2026-08-16.md (same directory). This job produces
# ARTIFACTS + ASSERTS ONLY -- no estimand/decision-rule computation
# (PREREG sec4/sec6 -- that is the next stage's `pdmux_eval`-based analyzer).
#
# NOT a claim job. Nothing here goes to canon without claims-auditor first.
#
# ---- addendum A summary (2026-08-16, stage-2 NO-GO -> fix) ----
#   A-1 (H3->H3'): realized_pin_check.py's frac-based PASS/FAIL is REMOVED
#     from this harness entirely (K10: it gates nothing, at any threshold --
#     it is monotone in decode SM and the old PIN_GATE=0.80 disabled all 7
#     arms). Replaced by TWO separate tools: H3'-a
#     (smsplit_realized_probe.probe_one, gates on exact== True) and H3'-b
#     (pdmux_eval.analyze.controller_summary, time-weighted, RECORD ONLY).
#   A-2: arm labels are conditional on the decode-active/co-resident
#     window; H3'-b's per-arm residency_fraction quantifies that.
#   A-3: block-level wall-clock/curvature drift caveat -- this harness
#     records t_boot0 per boot (H13) so the next-stage analyzer can run a
#     drift-corrected estimator alongside the position-balanced one.
#   A-4: g16_arm_order.py's --verify is an algebraic identity, not
#     independent design-quality evidence -- documented in that file.
#   A-5: H4 implementation judged sufficient, 2 conditions: (i) commit the 3
#     harness files before block-1 submission [process condition, cannot be
#     enforced by this script -- see G16_A5_COMMIT_CONDITION_WARNING below],
#     (ii) sha256 the 3 harness files into the per-block manifest [done,
#     H9 section below].
#   A-6: H13 per-boot JSON sidecar (new), H2'/H6' (assertion strengthening,
#     wired to the failure path), H14 env hygiene (expanded unset list),
#     H15 (NP/ROUNDS hard assert in FULL mode), H16 (0-byte artifact
#     cleanup), H17 (smoke tag separated to smoke1), H18 (cold-cache fix at
#     block position 0 -- job-local warm-up boot chosen; see notes doc for
#     why the shared-cache alternative was NOT chosen).
#   A-7: smoke pass conditions 8/9/10 added; smoke arms extended to 3
#     (d44, d64, d74) so both (44,64) and (34,74) -- both outside the E-3
#     probe's directly-verified set -- get an exact-match check.
#
# ---- addendum B-5(a) summary (2026-08-16, pre-block-1 caveats, THIS FIX) ----
#   (1) untracked/dirty detection for the 3 harness files (git ls-files
#     --error-unmatch + git status --porcelain), separate from the weaker
#     rev-parse-failure check -- see G16_HARNESS_UNCOMMITTED below. FULL mode:
#     loud, non-fatal. SMOKE mode: plain note (B-1(ii): smoke is valid without
#     a commit). Detection only -- the commit itself is a human/main-session
#     action, out of scope for this script.
#   (2) N1 fix: write_sidecar's call moved to AFTER the ARTIFACT_OK judgment
#     (was: unconditional "completed" BEFORE the judgment); new status value
#     `artifact_invalid` for a boot whose H2'/H6' checks failed.
#   (3) N2 fix: warm-up bench/server-log artifacts moved into
#     `$WARMUP_DIR` ($HERE/g16_warmup_artifacts/) AND the bench-serving
#     output file gets a `.IGNORE` suffix -- either alone would satisfy
#     "not matched by a `*_LO.jsonl`/`*_HI.jsonl` glob", both applied.
#   (4) N3 fix: H14's curated per-name unset list replaced by a blanket
#     `unset "${!PDMUX_@}" "${!SGLANG_ZAMBA_@}"` immediately followed by
#     re-exporting only the 3 vars this campaign needs
#     (PDMUX_TELEMETRY_PATH/PDMUX_RUN_ID/PDMUX_WORKLOAD_ID).
#   (5) N7: doc-only scope note (G16_HARNESS_NOTES's N7 section) -- no code
#     change in this file. The 2026-07-15/18 sgptv grid ran WITHOUT telemetry,
#     so this campaign's d16/d24 "drift anchor" comparison to it is build
#     drift + instrumentation overhead combined, not build drift alone.
#
# Usage:
#   sbatch g16_grid.sbatch <BLOCK_IDX 1-4>              # full block, 7 arms
#   sbatch --export=ALL,G16_SMOKE=1 g16_grid.sbatch      # smoke (sec9+A-7):
#                                                         # 3 arms (d44, d64,
#                                                         # d74), NP=40,
#                                                         # ROUNDS=1, 1 block
#   sbatch --test-only g16_grid.sbatch 1                 # scheduler dry-run
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/slo_sched
SMSPLIT_PROBE_DIR=$ROOT/workspace/engine-port/results/smsplit_realized
PDMUX_EVAL_DIR=$ROOT/workspace/engine-port/benchmarks
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
JOBTAG="${SLURM_JOB_ID:-local}"
# H18 (addendum A-6/A-7): kept job-unique on purpose. The alternative
# addendum A-1 offered -- reverting to the shared warm TRITON_CACHE_DIR used
# by `sharegpt_vary_bench.sbatch:30` -- reintroduces the exact "parallel job
# cache race" hazard this project's own tooling already warns about
# (sync_engine_tree.sh's note; s8_sweep/batchcap precedent), because >=2 of
# the 4 G16 blocks can plausibly be scheduled onto different nodes hitting
# the same filesystem path concurrently. The chosen fix is a job-local
# throwaway warm-up boot (see "H18 warm-up" below) that equalizes block
# position 0 with positions 1-6 while keeping the cache job-unique. Reasoning
# recorded in full in G16_HARNESS_NOTES_2026-08-16.md.
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_g16_${JOBTAG}"

MODEL=Zyphra/Zamba2-2.7B; CTX=4096
SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
RLO=3; RHI=12

SMOKE="${G16_SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  # sec9 smoke spec + addendum A-7 item 8: 3 arms (d44, d64, d74) -- d64 was
  # added because BOTH (44,64) and (34,74) are outside E-3's directly-
  # verified set, so d74 alone was not enough to exercise H3'-a on both.
  # NP=40, ROUNDS=1, 1 block.
  BLOCK_IDX=1
  # H17 (addendum A-6): smoke tag SEPARATED from the real block-1 tag. The
  # pre-fix version reused BLOCK_TAG=blk1, so a NP=40/ROUNDS=1 smoke
  # artifact and a real campaign block-1 boot were byte-for-byte
  # indistinguishable by filename/workload_id.
  BLOCK_TAG="smoke1"
  ARM_ORDER="d44 d64 d74"
  N_EXPECT_ARMS=3
  NP="${NP:-40}"
  ROUNDS="${ROUNDS:-1}"
  MIN_SNAPSHOTS=5
  MODE_TAG="smoke"
  echo "G16_MODE=SMOKE block_idx=$BLOCK_IDX block_tag=$BLOCK_TAG arms=[$ARM_ORDER] NP=$NP ROUNDS=$ROUNDS"
else
  BLOCK_IDX="${1:?BLOCK_IDX (1-4) required, e.g.: sbatch g16_grid.sbatch 1}"
  case "$BLOCK_IDX" in
    1|2|3|4) ;;
    *) echo "BAD_BLOCK_IDX=$BLOCK_IDX (must be 1-4)"; exit 1 ;;
  esac
  BLOCK_TAG="blk${BLOCK_IDX}"
  ARM_ORDER="$(python3 "$HERE/g16_arm_order.py" "$BLOCK_IDX")" \
    || { echo "ARM_ORDER_GEN_FAILED block_idx=$BLOCK_IDX"; exit 1; }
  N_EXPECT_ARMS=7
  NP="${NP:-200}"
  ROUNDS="${ROUNDS:-3}"
  MIN_SNAPSHOTS=20
  MODE_TAG="full"
  # H15 (addendum A-6): hard assert, not just a default -- an env override
  # can silently break K6/PC-A ("워크로드 원본 그대로... 하나라도 바꾸면
  # PC-A가 깨진다") without this, since `${NP:-200}` only applies the
  # default when NP is UNSET, not when it is set to something wrong.
  if [ "$NP" != "200" ] || [ "$ROUNDS" != "3" ]; then
    echo "H15_VIOLATION FULL mode requires NP=200 ROUNDS=3 (K6/PC-A) but got" \
      "NP=$NP ROUNDS=$ROUNDS -- refusing to run"
    exit 1
  fi
  echo "G16_MODE=FULL block_idx=$BLOCK_IDX arms=[$ARM_ORDER] NP=$NP ROUNDS=$ROUNDS"
fi
N_GOT_ARMS=$(echo "$ARM_ORDER" | wc -w)
if [ "$N_GOT_ARMS" != "$N_EXPECT_ARMS" ]; then
  echo "ARM_ORDER_COUNT_MISMATCH got=$N_GOT_ARMS want=$N_EXPECT_ARMS order=[$ARM_ORDER]"
  exit 1
fi
echo "G16_ARM_ORDER block=$BLOCK_TAG order=[$ARM_ORDER]"

# --- H9: engine manifest sha + git commit, once per block/job -------------
MANIFEST="$HERE/g16_manifest_${BLOCK_TAG}_${JOBTAG}.sha256"
"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" "$MANIFEST" \
  || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }
GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

# --- A-5(ii): sha256 the 3 harness files into the SAME per-block manifest
# (C2-R 11->15 drift precedent -- H9's own logic, extended to the harness
# itself, not just the engine tree). Appended in sha256sum-compatible format
# ("<hash>  <file>") so `sha256sum -c` still works against the combined file.
HARNESS_SHA_JSON="{"
_hfirst=1
for _hf in "$HERE/g16_grid.sbatch" "$HERE/g16_assert.py" "$HERE/g16_arm_order.py"; do
  _hsha="$(sha256sum "$_hf" | awk '{print $1}')"
  printf '%s  %s\n' "$_hsha" "$_hf" >> "$MANIFEST"
  [ "$_hfirst" = 1 ] || HARNESS_SHA_JSON="${HARNESS_SHA_JSON},"
  HARNESS_SHA_JSON="${HARNESS_SHA_JSON}\"$(basename "$_hf")\":\"$_hsha\""
  _hfirst=0
  echo "G16_HARNESS_SHA256 file=$(basename "$_hf") sha256=$_hsha"
done
HARNESS_SHA_JSON="${HARNESS_SHA_JSON}}"
MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
echo "G16_MANIFEST path=$MANIFEST sha256=$MANIFEST_SHA"
echo "G16_GIT_COMMIT=$GIT_COMMIT"
echo "G16_HARNESS_SHA_JSON $HARNESS_SHA_JSON"
# A-5(i) is a PROCESS condition this script cannot enforce (it would need to
# know what a human did before invoking sbatch; the commit itself is out of
# scope for this script -- task: "1번은 메인 세션이 사용자 승인 후 한다"). This
# only DETECTS and reports the state. The check below is the weakest possible
# signal (git repo not found / rev-parse failed entirely) -- kept as a
# fallback, but see G16_HARNESS_UNCOMMITTED right after it for the actual
# per-file check (B-5(a)#1 fix: the old rev-parse-only check was blind to "is
# a git checkout, but these 3 files are untracked/dirty", which is this
# job's actual live state as of 2026-08-16 -- verified below).
if [ "$GIT_COMMIT" = "unknown" ]; then
  echo "G16_A5_COMMIT_CONDITION_WARNING harness tree is not a git checkout or" \
    "rev-parse failed -- A-5(i) (commit before block-1 submission) cannot be" \
    "satisfied (weakest signal only -- see G16_HARNESS_UNCOMMITTED below)"
fi

# --- B-5(a)#1 fix: untracked/dirty detection for the 3 harness files -------
# Two independent per-file checks, either one marks a file "uncommitted":
#   (1) `git ls-files --error-unmatch <f>` fails (nonzero) -> file is not
#       tracked at all (this is this job's actual current state -- all 3
#       files are untracked as of 2026-08-16, which the pre-fix rev-parse-only
#       check above could NOT see, since $ROOT IS a valid git checkout).
#   (2) `git status --porcelain -- <f>` is non-empty -> file IS tracked but
#       has an uncommitted change (modified/staged/etc).
G16_HARNESS_UNCOMMITTED=0
G16_UNCOMMITTED_FILES=""
for _hf in "$HERE/g16_grid.sbatch" "$HERE/g16_assert.py" "$HERE/g16_arm_order.py"; do
  if ! git -C "$ROOT" ls-files --error-unmatch "$_hf" >/dev/null 2>&1; then
    G16_HARNESS_UNCOMMITTED=1
    G16_UNCOMMITTED_FILES="$G16_UNCOMMITTED_FILES $(basename "$_hf"):untracked"
    continue
  fi
  _porcelain="$(git -C "$ROOT" status --porcelain -- "$_hf" 2>/dev/null | head -1)"
  if [ -n "$_porcelain" ]; then
    G16_HARNESS_UNCOMMITTED=1
    G16_UNCOMMITTED_FILES="$G16_UNCOMMITTED_FILES $(basename "$_hf"):dirty[$_porcelain]"
  fi
done
echo "G16_HARNESS_UNCOMMITTED=$G16_HARNESS_UNCOMMITTED files=[$G16_UNCOMMITTED_FILES]"
# FULL mode: loud warning, NON-FATAL (no `exit`). rev3 addendum B-1(ii)
# re-audited this harness GO for smoke submission WITHOUT a commit
# precondition -- ground: every boot's sidecar already carries
# `harness_sha256` (H9/A-5(ii), a content hash of these same 3 files,
# computed a few lines above), which content-addresses exactly which harness
# text produced that boot's data regardless of git state. A missing commit is
# a reproduction-CONVENIENCE gap (no single `git show <sha>:g16_grid.sbatch`
# one-liner), not a data-validity gap -- killing a FULL-mode block over it
# would contradict that re-audit judgment (task: "스모크는 커밋 없이도
# 유효하다는 것이 재감사 판정이다"). SMOKE mode gets the same detection but a
# plain (non-banner) note, since B-1(ii) already covers it explicitly.
if [ "$G16_HARNESS_UNCOMMITTED" = 1 ]; then
  if [ "$MODE_TAG" = "full" ]; then
    echo "########################################################################"
    echo "# G16_HARNESS_UNCOMMITTED_WARNING (FULL mode, block=$BLOCK_TAG)"
    echo "# harness files not tracked-and-clean:$G16_UNCOMMITTED_FILES"
    echo "# A-5(i) (commit 3 harness files before block-1 submission) NOT satisfied."
    echo "# NOT blocking this run (harness_sha256 sidecar field content-addresses"
    echo "# the harness regardless of git state -- rev3 addendum B-5(a)#1)."
    echo "# Record this in the block-1 result document; commit before block 2-4"
    echo "# if at all possible."
    echo "########################################################################"
  else
    echo "G16_HARNESS_UNCOMMITTED_NOTE mode=$MODE_TAG (smoke does not require a" \
      "commit precondition per rev3 addendum B-1(ii) -- informational only)"
  fi
fi

GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)"
NODE_NAME="$(hostname -s 2>/dev/null || echo unknown)"

# --- H10: per-boot co-tenancy snapshot --------------------------------------
cotenancy_snapshot() {
  local LABEL="$1"
  local F="$HERE/g16_${BLOCK_TAG}_${LABEL}_cotenancy_${JOBTAG}.txt"
  {
    echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ) LABEL=$LABEL NODE=$NODE_NAME JOB=$JOBTAG"
    echo "--- squeue on this node (partition amd_a100nv_8, all users/states) ---"
    squeue --all -p amd_a100nv_8 -w "$NODE_NAME" -o "%.10i %.9P %.20j %.8T %.10M %N %.8u" 2>&1
    echo "--- nvidia-smi compute-apps on this node ---"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
  } > "$F"
  echo "G16_COTENANCY label=$LABEL node=$NODE_NAME file=$F"
}

# --- H16: rename (not delete) 0-byte artifacts from a failed boot so the
# analyzer's glob doesn't mistake "boot never wrote data" for "boot wrote an
# empty round" -- evidence preserved, just relabeled so it sorts out of the
# way. -------------------------------------------------------------------
mark_failed_artifacts() {
  local runid="$1"
  local f
  for suffix in LO.jsonl HI.jsonl telemetry.jsonl; do
    f="$HERE/${runid}_${suffix}"
    if [ -f "$f" ] && [ ! -s "$f" ]; then
      mv "$f" "${f}.FAILED"
      echo "G16_MARK_FAILED_ARTIFACT file=$f -> ${f}.FAILED"
    fi
  done
}

# --- H13 (addendum A-6): per-boot JSON sidecar. The single artifact that
# closes LO/HI attribution (per-round phase/t_start/t_end), the A-3 drift
# covariate (t_boot0), smoke-vs-campaign discrimination (mode), and
# failed-boot identification (status + telem_rc/fields_rc). Reads scalars
# via env (heredoc is single-quoted -> no bash substitution inside; all
# values cross the boundary through os.environ / argv only). -------------
write_sidecar() {
  local runid="$1" status="$2"
  local sidecar="$HERE/${runid}_sidecar.json"
  G16_SIDECAR_STATUS="$status" \
  G16_ARM="${ARM:-}" \
  G16_BLOCK_TAG="$BLOCK_TAG" \
  G16_BOOT_IDX="${BOOT_IDX:-}" \
  G16_T_BOOT0="${T_BOOT0:-}" \
  G16_T_BOOT1="${T_BOOT1:-}" \
  G16_T_HEALTHY="${T_HEALTHY:-}" \
  G16_ROUNDS_JSONL="${ROUNDS_JSONL:-}" \
  G16_REALPROBE_JSON="${REALPROBE_JSON:-}" \
  G16_NP="$NP" \
  G16_ROUNDS="$ROUNDS" \
  G16_NODE="$NODE_NAME" \
  G16_GPU_UUID="$GPU_UUID" \
  G16_MANIFEST_SHA="$MANIFEST_SHA" \
  G16_GIT_COMMIT_ENV="$GIT_COMMIT" \
  G16_HARNESS_SHA_JSON="$HARNESS_SHA_JSON" \
  G16_MODE_TAG="$MODE_TAG" \
  G16_TELEM_RC="${TELEM_RC:-}" \
  G16_FIELDS_LO_RC="${FIELDS_LO_RC:-}" \
  G16_FIELDS_HI_RC="${FIELDS_HI_RC:-}" \
  python3 - "$sidecar" <<'PYEOF'
import json, os, sys

out_path = sys.argv[1]


def _f(name, cast=None, default=None):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if cast is None:
        return v
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


rounds = []
rounds_path = os.environ.get("G16_ROUNDS_JSONL", "")
if rounds_path and os.path.exists(rounds_path):
    with open(rounds_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rounds.append(json.loads(line))
            except json.JSONDecodeError:
                pass

realized_a = realized_b = exact = None
rp_path = os.environ.get("G16_REALPROBE_JSON", "")
if rp_path and os.path.exists(rp_path):
    try:
        with open(rp_path) as fh:
            rp = json.load(fh)
        realized_a = rp.get("realized_a")
        realized_b = rp.get("realized_b")
        exact = rp.get("exact")
    except (json.JSONDecodeError, OSError):
        pass

harness_sha = {}
hs = os.environ.get("G16_HARNESS_SHA_JSON", "")
if hs:
    try:
        harness_sha = json.loads(hs)
    except json.JSONDecodeError:
        pass

record = {
    "status": os.environ.get("G16_SIDECAR_STATUS", "unknown"),
    "arm": os.environ.get("G16_ARM", ""),
    "block": os.environ.get("G16_BLOCK_TAG", ""),
    "boot": _f("G16_BOOT_IDX", int),
    "t_boot0": _f("G16_T_BOOT0", float),
    "t_boot1": _f("G16_T_BOOT1", float),
    "t_healthy": _f("G16_T_HEALTHY", float),
    "rounds": rounds,
    "NP": _f("G16_NP", int),
    "ROUNDS": _f("G16_ROUNDS", int),
    "node": os.environ.get("G16_NODE", ""),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "gpu_uuid": os.environ.get("G16_GPU_UUID", ""),
    "manifest_sha": os.environ.get("G16_MANIFEST_SHA", ""),
    "git_commit": os.environ.get("G16_GIT_COMMIT_ENV", ""),
    "harness_sha256": harness_sha,
    "mode": os.environ.get("G16_MODE_TAG", ""),
    "realized_a": realized_a,
    "realized_b": realized_b,
    "exact": exact,
    "telem_rc": _f("G16_TELEM_RC", int),
    "fields_rc": {
        "lo": _f("G16_FIELDS_LO_RC", int),
        "hi": _f("G16_FIELDS_HI_RC", int),
    },
}
with open(out_path, "w") as fh:
    json.dump(record, fh, indent=2, sort_keys=True)
print("G16_SIDECAR " + json.dumps(record, sort_keys=True))
PYEOF
}

BOOTS_FAILED=0
FAILED_ARMS=""
PIN_FAILED_ARMS=""
D74_WALLTIME_S=""; D74_BOOT_S=""; D74_BENCH_S=""
SMOKE1=""; SMOKE2=""; SMOKE2B=""; SMOKE3=""; SMOKE4=""

# --- H18 warm-up (addendum A-3-3/A-6/A-7): throwaway warm-up boot, ONCE per
# block/job, using this block's FIRST arm's config. Not counted as a boot for
# H1 purposes (writes no blk<N>_<arm>_boot<idx>-named artifact). Purpose is
# ONLY to prime the (job-unique, see note above) Triton JIT cache before the
# per-arm loop starts, so block position 0 pays the same cache-warm cost as
# positions 1-6 (all 7 yml configs share sm_group_num/split_forward_token_
# budget/decode_bs_divisor -- only manual_divisions differs -- so warming
# with ANY one arm's config should prime the kernels every arm needs).
# --- N2 fix (B-5(a)#3): warm-up artifacts moved OUT of the campaign
# namespace, on two independent axes (either alone would satisfy the rule
# text's "하위 디렉터리로 빼거나 .IGNORE 접미"; both are applied for
# defense-in-depth, matching this file's existing N1-fix style of "move AND
# relabel"): (1) a dedicated subdirectory `$WARMUP_DIR`, not `$HERE` itself,
# so a non-recursive `glob.glob("$HERE/*_LO.jsonl")`/`*_HI.jsonl` (or any
# other `$HERE/g16_*` pattern a future analyzer uses) never descends into it;
# (2) the bench-serving output file itself carries a `.IGNORE` suffix on top
# of that, so even a recursive/misdirected glob rooted one level up still
# can't match a literal `*_LO.jsonl`/`*_HI.jsonl` suffix. The warm-up bench
# call writes the SAME schema (ttfts/itls/duration) as a real campaign round
# (sec 5's own `sharegpt.bench_serving` invocation, just NP=2 instead of
# NP=200) with no sidecar/assert of its own -- undistinguishable from real
# data by content alone, hence both namespace fixes instead of relying on
# either one.
WARMUP_DIR="$HERE/g16_warmup_artifacts"
mkdir -p "$WARMUP_DIR"
WARMUP_ARM="${ARM_ORDER%% *}"
WARMUP_CFG="$HERE/pdmux_${WARMUP_ARM}.yml"
if [ -f "$WARMUP_CFG" ]; then
  echo "G16_WARMUP_START block=$BLOCK_TAG cfg=$WARMUP_CFG dir=$WARMUP_DIR"
  WPORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  WSRVLOG="$WARMUP_DIR/g16_${BLOCK_TAG}_warmup_${JOBTAG}_srv.log"
  python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
    --attention-backend triton --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
    --enable-pdmux --pdmux-config-path "$WARMUP_CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
    --host 127.0.0.1 --port $WPORT > "$WSRVLOG" 2>&1 &
  WSRV=$!
  wok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$WPORT/health 2>/dev/null || true)" = 200 ] && { wok=1; break; }
    kill -0 $WSRV 2>/dev/null || { echo "WARMUP_SERVER_DIED block=$BLOCK_TAG"; break; }
    sleep 5
  done
  if [ "$wok" = 1 ]; then
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $WPORT \
      --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len 4000 \
      --num-prompts 2 --request-rate 3 \
      --output-file "$WARMUP_DIR/g16_${BLOCK_TAG}_warmup_${JOBTAG}_bench.jsonl.IGNORE" \
      >> "$WSRVLOG" 2>&1 || echo "WARMUP_BENCH_NONZERO block=$BLOCK_TAG (non-fatal, cache priming only)"
  else
    echo "WARMUP_BOOT_FAILED block=$BLOCK_TAG (non-fatal -- position-0 arm may still see cold-cache cost)"
  fi
  kill $WSRV 2>/dev/null; pkill -9 -f "launch_server.*--port $WPORT" 2>/dev/null; sleep 5
  echo "G16_WARMUP_DONE block=$BLOCK_TAG ok=$wok"
else
  echo "G16_WARMUP_SKIPPED block=$BLOCK_TAG reason=missing_cfg cfg=$WARMUP_CFG"
fi

for ARM in $ARM_ORDER; do
  # H7: explicit case covering d64|d74, which the reference
  # sharegpt_vary_bench.sbatch:36 does NOT (it only has d16|d24|d34|d44|d54).
  case "$ARM" in
    d16|d24|d34|d44|d54|d64|d74) DSM="${ARM#d}"; CFG="$HERE/pdmux_${ARM}.yml" ;;
    *) echo "BAD_ARM=$ARM"; BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"; continue ;;
  esac
  if [ ! -f "$CFG" ]; then
    echo "MISSING_CONFIG arm=$ARM cfg=$CFG"
    BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"
    continue
  fi

  # H1: one boot = one measurement, no in-block retries/reboots for the same
  # arm (n_indep comes from the 4 separate block JOBS, not from within-job
  # repetition -- s8_sweep.sbatch:130-186's defect was reps inside one boot).
  BOOT_IDX=1
  RUNID="g16_${BLOCK_TAG}_${ARM}_boot${BOOT_IDX}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  JLO="$HERE/${RUNID}_LO.jsonl"; JHI="$HERE/${RUNID}_HI.jsonl"
  : > "$JLO"; : > "$JHI"
  # G16_N3_BLOCK_START (regression-test extraction marker -- keep this and
  # its closing counterpart below in place; a dry-run test sources the text
  # between them verbatim to confirm the bulk-unset does not also erase the
  # 3 exports it is immediately followed by.)
  # --- N3 fix (B-5(a)#4): blanket env hygiene, applied BEFORE the 3 needed
  # exports (order matters: bulk-unset first, or it would immediately erase
  # the exports below too). The prior curated H14 unset list (addendum A-6,
  # 9 names) already missed 7 PHYSICAL-config vars on its first pass
  # (SGLANG_ZAMBA_TIMING/_PREFILL_KNEE -- per-layer CUDA-event instrumentation
  # = execution overhead -- + PDMUX_FIXED_PREFILL_SM_FILE, zamba2.py:519-526
  # prefill-layer pin -- + PDMUX_HOLB_PATH/PDMUX_AGN2_SM/PDMUX_LA_SM_MAP/
  # PDMUX_LA_FLOOR_SM), which demonstrates a curated list is structurally
  # incapable of staying complete: the engine tree defines 55 PDMUX_* + 4
  # SGLANG_ZAMBA_* env vars total as of 2026-08-16 (grepped; see
  # G16_HARNESS_NOTES's N3 section for the full list), most of which are
  # per-model-family levers this campaign never touches on purpose. Rather
  # than extend the curated list to 16 names (or whatever the next audit
  # finds missing), unset EVERY currently-set PDMUX_*/SGLANG_ZAMBA_* var --
  # whatever SLURM's default --export=ALL leaked from the submitting shell --
  # THEN re-export only the 3 this campaign actually reads on its code path
  # (H2: PDMUX_TELEMETRY_PATH/PDMUX_RUN_ID/PDMUX_WORKLOAD_ID -- no R2 policy,
  # no dual-worker, no SLO scheduler, no LA/AGN coordination; the arm's
  # manual_divisions comes from the yml via --pdmux-config-path, a CLI flag,
  # not an env var). This closes the leak surface at its root instead of
  # chasing individual names one audit at a time. `${!PREFIX@}` safely
  # expands to zero words (no error under `set -u`) when nothing matches, so
  # this is a no-op on a clean submitting shell.
  for _pv in "${!PDMUX_@}" "${!SGLANG_ZAMBA_@}"; do
    unset "$_pv"
  done
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"
  export PDMUX_WORKLOAD_ID="g16_${ARM}"
  # G16_N3_BLOCK_END

  ROUNDS_JSONL="$HERE/${RUNID}_rounds.jsonl.tmp"; : > "$ROUNDS_JSONL"
  REALPROBE_JSON="$HERE/${RUNID}_realizedprobe.json"
  CTRLSUMMARY_JSON="$HERE/${RUNID}_controller_summary.json"
  TELEM_RC=""; FIELDS_LO_RC=""; FIELDS_HI_RC=""; PIN_RC=""; CTRLSUMMARY_RC=""
  T_HEALTHY=""; T_BOOT1=""

  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")

  cotenancy_snapshot "${ARM}_boot${BOOT_IDX}_pre"

  echo "=== block=$BLOCK_TAG arm=$ARM decode_sm=$DSM boot=$BOOT_IDX port=$PORT run_id=$RUNID ==="
  T_BOOT0=$(python3 -c "import time; print(time.monotonic())")
  # Server flags verbatim from sharegpt_vary_bench.sbatch (cudagraph ON,
  # triton, radix off, mem-frac 0.82, max-running 48, chunked-prefill -1,
  # overlap-schedule disabled): sec5 "하나라도 바꾸면 PC-A가 깨진다".
  python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
    --attention-backend triton --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
    --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null || true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED arm=$ARM boot=$BOOT_IDX"; break; }
    sleep 5
  done
  # A-7 item 10 / H13: boot-vs-bench decomposition anchor.
  T_HEALTHY=$(python3 -c "import time; print(time.monotonic())")
  echo "boot_ok=$ok arm=$ARM block=$BLOCK_TAG boot=$BOOT_IDX"

  if [ "$ok" != 1 ]; then
    echo "BOOT_FAILED arm=$ARM block=$BLOCK_TAG boot=$BOOT_IDX"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -30
    BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"
    kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
    mark_failed_artifacts "$RUNID"
    write_sidecar "$RUNID" boot_failed
    continue
  fi

  BENCH_OK=1
  for r in $(seq 1 $ROUNDS); do
    echo "== block=$BLOCK_TAG arm=$ARM round=$r LO (rate $RLO) =="
    T_LO0=$(python3 -c "import time; print(time.monotonic())")
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len 4000 \
      --num-prompts $NP --request-rate $RLO --output-details --output-file "$JLO" \
      2>>"$SRVLOG" | grep -E "Median TTFT|Median TPOT" || BENCH_OK=0
    T_LO1=$(python3 -c "import time; print(time.monotonic())")
    printf '{"phase":"LO","round":%d,"t_start":%s,"t_end":%s}\n' "$r" "$T_LO0" "$T_LO1" >> "$ROUNDS_JSONL"

    echo "== block=$BLOCK_TAG arm=$ARM round=$r HI (rate $RHI) =="
    T_HI0=$(python3 -c "import time; print(time.monotonic())")
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len 4000 \
      --num-prompts $NP --request-rate $RHI --output-details --output-file "$JHI" \
      2>>"$SRVLOG" | grep -E "Median TTFT|Median TPOT" || BENCH_OK=0
    T_HI1=$(python3 -c "import time; print(time.monotonic())")
    printf '{"phase":"HI","round":%d,"t_start":%s,"t_end":%s}\n' "$r" "$T_HI0" "$T_HI1" >> "$ROUNDS_JSONL"
  done
  T_BOOT1=$(python3 -c "import time; print(time.monotonic())")
  WALL_S=$(python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$T_BOOT0" "$T_BOOT1")
  # A-7 item 10: decompose into boot-until-healthy vs bench-only.
  BOOT_S=$(python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$T_BOOT0" "$T_HEALTHY")
  BENCH_S=$(python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$T_HEALTHY" "$T_BOOT1")
  echo "G16_BOOT_WALLTIME arm=$ARM block=$BLOCK_TAG boot=$BOOT_IDX seconds=$WALL_S boot_s=$BOOT_S bench_s=$BENCH_S"
  if [ "$ARM" = "d74" ]; then
    D74_WALLTIME_S="$WALL_S"; D74_BOOT_S="$BOOT_S"; D74_BENCH_S="$BENCH_S"
  fi

  cotenancy_snapshot "${ARM}_boot${BOOT_IDX}_post"

  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5

  if [ "$BENCH_OK" != 1 ]; then
    echo "BENCH_FAILED arm=$ARM block=$BLOCK_TAG boot=$BOOT_IDX (bench_serving nonzero in >=1 round)"
    BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"
    mark_failed_artifacts "$RUNID"
    write_sidecar "$RUNID" bench_failed
    continue
  fi

  # --- H2' assert (addendum A-6): telemetry must contain >= MIN_SNAPSHOTS
  # events with event=="runtime_snapshot" AND phase=="benchmark" AND
  # "decode_sms" in event -- NOT merely "file is non-empty" (the pre-fix
  # check, equivalent to "did you export the env var").
  python3 "$HERE/g16_assert.py" runtime_snapshots "$PDMUX_TELEMETRY_PATH" "$MIN_SNAPSHOTS"
  TELEM_RC=$?

  # --- H6' assert (addendum A-6): non-empty ttfts, itls length match,
  # positive duration, no nulls, round count = ROUNDS, LO/HI separate files.
  python3 "$HERE/g16_assert.py" fields "$JLO" "$ROUNDS"; FIELDS_LO_RC=$?
  python3 "$HERE/g16_assert.py" fields "$JHI" "$ROUNDS"; FIELDS_HI_RC=$?

  # --- H3'-a assert (addendum A-1): driver-realized coordinate probe, ONCE
  # per boot, run standalone (server already killed above -- E-3-style
  # isolation, no green-context contention with a live server). THE ONLY
  # thing that disables an arm (K10) -- see g16_assert.py's
  # assert_realized_probe docstring for why co-residency/pin fraction, at
  # ANY threshold, must not gate.
  python3 - "$DSM" "$REALPROBE_JSON" <<PYEOF
import json, sys
sys.path.insert(0, "$SMSPLIT_PROBE_DIR")
dsm = int(sys.argv[1])
out_path = sys.argv[2]
rec = {"requested_a": 108 - dsm, "requested_b": dsm, "arm_decode_sm": dsm}
try:
    import torch
    from sgl_kernel import spatial as _sgl_spatial  # noqa: F401 -- registers create_greenctx_stream_by_value
    from smsplit_realized_probe import probe_one
    torch.cuda.init()
    torch.cuda.set_device(0)
    rec.update(probe_one(108 - dsm, dsm, 0))
except Exception as exc:  # noqa: BLE001 -- fail-safe: record the error, don't crash the boot
    rec["error"] = repr(exc)
with open(out_path, "w") as fh:
    json.dump(rec, fh, indent=2, sort_keys=True)
print("G16_REALIZED_PROBE " + json.dumps(rec, sort_keys=True))
PYEOF
  cat "$REALPROBE_JSON" 2>/dev/null || echo "REALPROBE_JSON_MISSING path=$REALPROBE_JSON"
  python3 "$HERE/g16_assert.py" realized_probe "$REALPROBE_JSON"
  PIN_RC=$?
  if [ "$PIN_RC" != 0 ]; then
    echo "ARM_DISABLED_REALIZED_MISMATCH arm=$ARM block=$BLOCK_TAG boot=$BOOT_IDX" \
      "(H3'-a: smsplit_realized_probe.probe_one(108-$DSM,$DSM,0) exact != true, or" \
      "probe errored; excluded -- downstream analyzer must redefine S_max/S_min over" \
      "surviving arms only)"
    PIN_FAILED_ARMS="$PIN_FAILED_ARMS $ARM"
  fi

  # --- H3'-b (addendum A-1): time-weighted dwell diagnostic. RECORD ONLY --
  # NEVER gates arm disablement, at any value (K10). controller_summary is
  # time-weighted (avoids count-subsample bias, gate #15) and its
  # residency_fraction (keyed by the ACTUALLY-realized decode_sms value,
  # including the 0/108 unpartitioned states) is what addendum A-2's
  # "arm labels are conditional on the co-resident window" caveat needs
  # quantified per arm.
  python3 - "$PDMUX_TELEMETRY_PATH" "$CTRLSUMMARY_JSON" <<PYEOF
import json, sys
sys.path.insert(0, "$PDMUX_EVAL_DIR")
from pdmux_eval.analyze import controller_summary
telem_path, out_path = sys.argv[1], sys.argv[2]
events = []
with open(telem_path) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
summary = controller_summary(events)
with open(out_path, "w") as fh:
    json.dump(summary, fh, indent=2, sort_keys=True)
print("G16_CONTROLLER_SUMMARY " + json.dumps(summary, sort_keys=True))
PYEOF
  CTRLSUMMARY_RC=$?
  cat "$CTRLSUMMARY_JSON" 2>/dev/null || echo "CONTROLLER_SUMMARY_JSON_MISSING path=$CTRLSUMMARY_JSON"

  # G16_N1_BLOCK_START (regression-test extraction marker -- keep this and
  # its closing counterpart below in place; g16_n1_regression_test.py sources
  # the text between them verbatim. Do not remove/rename without updating
  # that test.)
  # --- N1 fix (B-5(a)#2): ARTIFACT_OK MUST be decided BEFORE the sidecar is
  # written, and the sidecar's status field must be DERIVED from that
  # decision, not a hardcoded literal. Pre-fix, `write_sidecar "$RUNID"
  # completed` ran here unconditionally, and only THEN did the ARTIFACT_OK
  # check (reading the very same TELEM_RC/FIELDS_*_RC already computed above)
  # discover the boot's artifacts were unusable -- producing a sidecar that
  # claimed status="completed" for a boot the harness itself was about to
  # reject (reproduced synthetically: status="completed" co-occurring with
  # fields_rc.hi=1 -- see g16_n1_regression_test.py). Fix applied here: (a)
  # moved -- ARTIFACT_OK is computed first, write_sidecar is called exactly
  # once, after the decision; (b) new status value -- a rejected boot's
  # sidecar now says status="artifact_invalid", not "completed" and not
  # silently reused from one of the pre-existing boot_failed/bench_failed
  # labels (those are earlier-stage failures -- server never came up / bench
  # process exited nonzero -- this is a THIRD failure class: the server ran
  # and bench_serving exited 0, but the artifacts it wrote don't satisfy
  # H2'/H6'). H2'/H6' wired to the failure path (addendum A-6, unchanged by
  # this fix): TELEM_RC/FIELDS_*_RC used to be computed and printed but never
  # ACTED on, so an all-requests-failed round silently PASSED. This does NOT
  # touch PIN_RC -- H3 exclusion stays a per-arm exclusion, not a boot
  # failure (rev3 sec9's own "값싼 신중함" framing; see notes doc
  # compromise #3).
  ARTIFACT_OK=1
  [ "$TELEM_RC" -eq 0 ] || ARTIFACT_OK=0
  [ "$FIELDS_LO_RC" -eq 0 ] || ARTIFACT_OK=0
  [ "$FIELDS_HI_RC" -eq 0 ] || ARTIFACT_OK=0

  echo "G16_ARM_SUMMARY block=$BLOCK_TAG arm=$ARM boot=$BOOT_IDX" \
    "telemetry_rc=$TELEM_RC fields_lo_rc=$FIELDS_LO_RC fields_hi_rc=$FIELDS_HI_RC" \
    "pin_rc=$PIN_RC ctrlsummary_rc=$CTRLSUMMARY_RC artifact_ok=$ARTIFACT_OK" \
    "lo=$JLO hi=$JHI telemetry=$PDMUX_TELEMETRY_PATH run_id=$RUNID"

  if [ "$ARTIFACT_OK" != 1 ]; then
    echo "ARTIFACT_INVALID arm=$ARM block=$BLOCK_TAG boot=$BOOT_IDX telemetry_rc=$TELEM_RC" \
      "fields_lo_rc=$FIELDS_LO_RC fields_hi_rc=$FIELDS_HI_RC (H2'/H6': boot produced" \
      "unusable artifacts; wired to the failure path per addendum A-6)"
    BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"
    mark_failed_artifacts "$RUNID"
    # --- H13/N1: sidecar written AFTER the ARTIFACT_OK judgment, with a
    # status that reflects it (not "completed"). ---------------------------
    write_sidecar "$RUNID" artifact_invalid
  else
    # --- H13/N1: sidecar written AFTER the ARTIFACT_OK judgment. ----------
    write_sidecar "$RUNID" completed
  fi
  # G16_N1_BLOCK_END

  if [ "$SMOKE" = "1" ] && [ "$ARM" = "d64" ]; then
    # A-7 item 8 (second half): (44,64) is also outside E-3's directly-
    # verified set, so d74 alone is not enough -- check d64 too.
    [ "$PIN_RC" -eq 0 ] && SMOKE2B=PASS || SMOKE2B=FAIL
    echo "G16_SMOKE_CHECK 8b_REALIZED_PROBE_EXACT_D64=$SMOKE2B file=$REALPROBE_JSON"
  fi

  if [ "$SMOKE" = "1" ] && [ "$ARM" = "d74" ]; then
    # sec9 pass conditions 1-4 + 7, plus addendum A-7 items 8/9/10, printed
    # explicitly (no eyeballing logs). 5 (BOOTS_FAILED) and 6
    # (manifest/commit) are block-level, printed after the loop.
    [ "$TELEM_RC" -eq 0 ] && SMOKE1=PASS || SMOKE1=FAIL
    [ "$PIN_RC" -eq 0 ] && SMOKE2=PASS || SMOKE2=FAIL
    if [ "$FIELDS_LO_RC" -eq 0 ] && [ "$FIELDS_HI_RC" -eq 0 ]; then SMOKE3=PASS; else SMOKE3=FAIL; fi
    case "$RUNID" in *smoke1_d74_boot1*) SMOKE4=PASS ;; *) SMOKE4=FAIL ;; esac
    echo "G16_SMOKE_CHECK 1_TELEMETRY_RUNTIME_SNAPSHOTS=$SMOKE1 min=$MIN_SNAPSHOTS"
    echo "G16_SMOKE_CHECK 2_REALIZED_PROBE_EXACT_D74=$SMOKE2 file=$REALPROBE_JSON"
    echo "G16_SMOKE_CHECK 3_FIELD_ASSERT=$SMOKE3"
    echo "G16_SMOKE_CHECK 4_FILENAME_CONVENTION=$SMOKE4 run_id=$RUNID"
    echo "G16_SMOKE_CHECK 7_D74_WALLTIME=PASS seconds=$D74_WALLTIME_S"
    # A-7 item 9: report the H2' count + the H3'-b time-weighted residency
    # actually observed, so the campaign's dwell expectations are on record
    # (informational -- non-gating, this is a baseline measurement, not a
    # threshold check).
    SMOKE9_COUNT=$(python3 - "$PDMUX_TELEMETRY_PATH" <<'PYEOF'
import json, sys
n = 0
with open(sys.argv[1]) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "runtime_snapshot" and e.get("phase") == "benchmark" and "decode_sms" in e:
            n += 1
print(n)
PYEOF
)
    SMOKE9_RESIDENCY=$(python3 - "$CTRLSUMMARY_JSON" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as fh:
        d = json.load(fh)
    print(json.dumps(d.get("residency_fraction", {})))
except Exception:
    print("{}")
PYEOF
)
    echo "G16_SMOKE_CHECK 9_H2_COUNT_AND_RESIDENCY=PASS" \
      "runtime_snapshot_benchmark_count=$SMOKE9_COUNT residency_fraction=$SMOKE9_RESIDENCY"
    echo "G16_SMOKE_CHECK 10_D74_BOOT_BENCH_SPLIT=PASS boot_s=$D74_BOOT_S bench_s=$D74_BENCH_S"
  fi
done

echo "G16_BLOCK_DONE block=$BLOCK_TAG arms_total=$N_GOT_ARMS BOOTS_FAILED=$BOOTS_FAILED" \
  "failed_arms=[$FAILED_ARMS] pin_failed_arms=[$PIN_FAILED_ARMS]"

if [ "$SMOKE" = "1" ]; then
  [ "$BOOTS_FAILED" -eq 0 ] && SMOKE5=PASS || SMOKE5=FAIL
  if [ -s "$MANIFEST" ] && [ "$GIT_COMMIT" != "unknown" ]; then SMOKE6=PASS; else SMOKE6=FAIL; fi
  echo "G16_SMOKE_CHECK 5_BOOTS_FAILED=$SMOKE5 value=$BOOTS_FAILED"
  echo "G16_SMOKE_CHECK 6_MANIFEST_COMMIT=$SMOKE6 manifest=$MANIFEST commit=$GIT_COMMIT"
  if [ "$SMOKE1" = PASS ] && [ "$SMOKE2" = PASS ] && [ "$SMOKE2B" = PASS ] \
     && [ "$SMOKE3" = PASS ] && [ "$SMOKE4" = PASS ] && [ "$SMOKE5" = PASS ] \
     && [ "$SMOKE6" = PASS ] && [ -n "$D74_WALLTIME_S" ] && [ -n "$D74_BOOT_S" ] \
     && [ -n "$D74_BENCH_S" ]; then
    echo "G16_SMOKE_OVERALL=PASS (sec9 conditions 1-7 + addendum A-7 items 8/9/10 satisfied)"
  else
    echo "G16_SMOKE_OVERALL=FAIL (>=1 condition not satisfied -- do NOT submit the full campaign)"
  fi
fi

rm -rf "$TRITON_CACHE_DIR"
echo "G16_DONE block=$BLOCK_TAG job=$JOBTAG"
