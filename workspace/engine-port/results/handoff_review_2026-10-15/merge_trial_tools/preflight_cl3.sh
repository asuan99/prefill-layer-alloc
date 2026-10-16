#!/bin/bash
# preflight_cl3.sh -- merge preflight for the CL-3 unverified tests (DRAFT 2026-10-15, never run in a container).
#
# WHAT.  VERDICT_spec_v0_1_and_closure_round2 sec 6 cond. 1: before the first GPU Job on the merge tree (G0 or M3),
# run the CL-3 tests that cannot run locally (sgl_kernel import, venv path) on the PRE tree (main 7b15214) and the
# POST tree (PDMUX_PF_POST, required; the 2026-10-15 draft targeted the merge dc06c870) IN THE SAME CONTAINER, pair them by
# (test id, signature), and label:
#   PREFLIGHT_PASS | PREFLIGHT_FAIL_NEW | PREFLIGHT_UNVERIFIABLE      (rules: preflight_judge.py header)
# This is a CORRECTNESS probe, not a measurement: no number from it is a performance result.
# Design + decisions: handoff_review_2026-10-15/PREFLIGHT_merge_cl3_design_2026-10-15.md.
#
# MODES
#   Job (default; target of scripts/vessl/launch.sh, runs after job_entry.sh installed the engine):
#       launch.sh --campaign preflight_cl3 --target workspace/engine-port/results/handoff_review_2026-10-15/merge_trial_tools/preflight_cl3.sh
#       ★ the script, judge, lists must be COMMITTED before --submit (launch.sh bundles HEAD).
#   --dry-run          local, CPU: bash/py syntax, id lists vs the local pins, git ancestry + src diff mapping, judge
#                      selftest, plan printout.  Creates nothing under the repo.
#   --dry-run-trees    --dry-run plus: build both arm trees from LOCAL checkouts (git worktree in a scratch dir, copy of
#                      $PDMUX_PF_SRC_DEV) and run the tree checks.  No test suite is run.  Needs $PDMUX_PF_SRC_DEV.
#   --judge-only PRE_LOG[+LOG] POST_LOG[+LOG]   re-judge fetched logs (one repetition each)
# ENV (all PDMUX_*, so launch.sh --env accepts them)
#   PDMUX_PF_PRE                   full sha (default below)          PDMUX_PF_REPS   repetitions per arm (default 2, ABBA)
#   PDMUX_PF_POST                  REQUIRED full sha of the final POST (no default since M3 rev12 harness audit r1 R-H2, 2026-10-16)
#   PDMUX_PF_MODULES               space list -> run only those test modules (smoke; judge result then NOT a preflight)
#   PDMUX_PF_WORK                  scratch root (default $PDMUX_DATA/runs/$JOB/pf_work)   PDMUX_PF_SUITE_TIMEOUT (s, 3000)
#   PDMUX_VENV                     venv path the sbatch-driven tests should source (see venv probe)
# Exit code = judge: 0 PASS, 1 FAIL_NEW, 3 UNVERIFIABLE (Job shows failed on non-PASS; DONE+SHA256SUMS are written anyway).
# Job mode also writes OUT/PREFLIGHT_PASS.json (+ .sha256) with preflight_judge.py `record` (R-H2): the exact bytes the M3 gate
# (mg_plan.preflight_gate (i)) reads once committed as results/multigpu/PREFLIGHT_PASS.json(+.sha256) -- copied, never retyped.
# NEVER dumps the environment: only PDMUX_* names+values are recorded.
# r2 (2026-10-17, judge revision; design doc sec 10): suites run through pf_unittest_runner.py (same -v text + LOG.status.json from
# the TestResult callbacks; the judge gets --require-structured); the CPU-spec decision (PDMUX_CPU_ONLY=1 or PDMUX_SPEC=
# resourcespec-a100cpu, AND no GPU visible) is written to meta/spec_decision.txt and passes --cpu-spec, which is the ONLY way the
# pre-declared CPU-timing ids (preflight_cl3_cpu_timing_declared.txt) leave the judgement; a CPU spec with a visible GPU is INVALID.
set -uo pipefail

PRE_DEFAULT=7b15214b2d554ec7f8394485892980af8271734c
PRISTINE_PDMUX_CONTEXT_SHA=b4e6ff1aa10a19f9cdd285088424b9388574fd291edfd466e155cbaa8e876b9e  # local main dev tree (unpatched), 2026-10-15
PIN_PATCH=workspace/engine-port/src/patches/pdmux_context_device_assert.patch

MODE=job; JUDGE_PRE=""; JUDGE_POST=""
case "${1:-}" in
  --dry-run) MODE=dry ;;
  --dry-run-trees) MODE=dry_trees ;;
  --judge-only) MODE=judge; JUDGE_PRE="${2:?}"; JUDGE_POST="${3:?}" ;;
  -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRE="${PDMUX_PF_PRE:-$PRE_DEFAULT}"; POST="${PDMUX_PF_POST:-}"
REPS="${PDMUX_PF_REPS:-2}"; MODS="${PDMUX_PF_MODULES:-}"; T_SUITE="${PDMUX_PF_SUITE_TIMEOUT:-3000}"
PY="${PDMUX_PF_PY:-python}"; [[ "${MODE}" == job ]] || PY="${PDMUX_PF_PY:-python3}"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[PF $(utc)] $*"; }

if [[ "${MODE}" == judge ]]; then
  exec "${PY}" "${TOOLS}/preflight_judge.py" --pre "${JUDGE_PRE}" --post "${JUDGE_POST}"
fi
# R-H2: the POST under test is always named explicitly (the old default dc06c870 is not the rev12 POST)
[[ "${POST}" =~ ^[0-9a-f]{40}$ ]] || { echo "PDMUX_PF_POST must be the full 40-hex sha of the final POST (got '${POST}')" >&2; exit 2; }

if [[ "${MODE}" == job ]]; then
  PROJECT="${PDMUX_PROJECT_ROOT:?run through job_entry.sh (PDMUX_PROJECT_ROOT)}"
  JOB="${PDMUX_JOB_NAME:?}"; DATA="${PDMUX_DATA:-/data}"
  OUT="${PROJECT}/workspace/engine-port/results/preflight_cl3/${JOB}"
  WORK="${PDMUX_PF_WORK:-${DATA}/runs/${JOB}/pf_work}"
  SRC_DEV="${SGLANG_ENGINE_DEV:?image must set SGLANG_ENGINE_DEV}"
else
  PROJECT="$(git -C "${TOOLS}" rev-parse --show-toplevel)"
  JOB="dry_$(date -u +%Y%m%dT%H%M%SZ)"; DATA="${PDMUX_DATA:-/nonexistent}"
  SCR="${TMPDIR:-/tmp}/preflight_cl3_${JOB}"; OUT="${SCR}/out"; WORK="${SCR}/work"
  SRC_DEV="${PDMUX_PF_SRC_DEV:-}"
fi
mkdir -p "${OUT}/logs" "${OUT}/meta" "${WORK}" || { echo "cannot create ${OUT}" >&2; exit 2; }
INVALID=(); SOFT=(); WTS=()
cleanup() {
  for wt in "${WTS[@]+"${WTS[@]}"}"; do git -C "${PROJECT}" worktree remove --force "${wt}" >/dev/null 2>&1; done
  git -C "${PROJECT}" worktree prune >/dev/null 2>&1
  [[ "${MODE}" == job ]] && rm -rf "${WORK}" 2>/dev/null
  return 0
}
trap cleanup EXIT
say "mode=${MODE} job=${JOB} out=${OUT}"

# ---- 0. what ran (provenance) ---------------------------------------------------------------------------------
{ echo "script_sha256=$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
  for f in preflight_judge.py sigcmp.py pf_unittest_runner.py preflight_cl3_l2_ids.txt preflight_cl3_targets.txt \
           preflight_cl3_cpu_timing_declared.txt; do
    echo "${f}_sha256=$(sha256sum "${TOOLS}/${f}" | cut -d' ' -f1)"; done
  echo "pre=${PRE}"; echo "post=${POST}"; echo "reps=${REPS}"; echo "modules=${MODS:-<all>}"
  echo "job_head=$(git -C "${PROJECT}" rev-parse HEAD 2>/dev/null)"
  "${PY}" - <<'PYE' 2>&1
import sys, platform
print("python", sys.version.split()[0], "host", platform.node())
try:
    import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
except Exception as e: print("torch import failed", repr(e))
PYE
  nvidia-smi -L 2>&1 | head -3
  env | grep '^PDMUX_' | grep -v -i 'token\|secret' | sort | sed 's/^/env /'
} > "${OUT}/meta/provenance.txt" 2>&1

# ---- 0b. spec decision (r2): the CPU-timing declaration applies only on a CPU spec with no GPU in sight ----------------------
GPU_SEEN=0
nvidia-smi -L 2>/dev/null | grep -q '^GPU ' && GPU_SEEN=1
"${PY}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1 && GPU_SEEN=1
"${PY}" "${TOOLS}/preflight_judge.py" spec --gpu-seen "${GPU_SEEN}" > "${OUT}/meta/spec_decision.txt" 2>&1; SPEC_RC=$?
CPU_SPEC=0; [[ ${SPEC_RC} -eq 0 ]] && CPU_SPEC=1

# ---- 1. static checks (all modes) --------------------------------------------------------------------------------
bash -n "${BASH_SOURCE[0]}" || INVALID+=("script syntax")
"${PY}" -I - "${TOOLS}" <<'PYE' > "${OUT}/meta/lists_check.txt" 2>&1 || INVALID+=("id lists inconsistent with the local pins")
import json, sys, os
t = sys.argv[1]
pin_pre = json.load(open(os.path.join(t, 'pin_pre.json'))); pin_post = json.load(open(os.path.join(t, 'pin_post.json')))
l2 = [l.strip() for l in open(os.path.join(t, 'preflight_cl3_l2_ids.txt')) if l.strip() and not l.startswith('#')]
tg = [l.rstrip('\n').split('\t') for l in open(os.path.join(t, 'preflight_cl3_targets.txt')) if l.strip() and not l.startswith('#')]
bad = []
if sorted(l2) != sorted(set(pin_post) - set(pin_pre)): bad.append('L2 list != pin_post - pin_pre')
ids = [v for s, k, v in tg if k == 'id']
mods = [v for s, k, v in tg if k == 'module']
if len([x for x in tg if x[0] == 'A_import9']) != 9: bad.append('A_import9 != 9 modules')
if len([x for x in tg if x[0] == 'B_holb4']) != 4: bad.append('B_holb4 != 4 ids')
if len([x for x in tg if x[0] == 'C_venv17']) != 17: bad.append('C_venv17 != 17 ids')
for i in ids:
    if i not in pin_post: bad.append('target id not a known local failure: ' + i)
for m in [v for s, k, v in tg if s == 'A_import9']:
    if 'unittest.loader._FailedTest.' + m not in pin_post: bad.append('module not a known local import failure: ' + m)
dec = [l.strip() for l in open(os.path.join(t, 'preflight_cl3_cpu_timing_declared.txt')) if l.strip() and not l.startswith('#')]
if len(dec) != 5 or len(set(dec)) != 5: bad.append('CPU-timing declaration != 5 distinct ids')
if [d for d in dec if d.startswith('test_n3_capacity.') and d not in l2]: bad.append('declared n3 id not an L2 id')
if [d for d in dec if d in ids or d.split('.', 1)[0] in mods]: bad.append('declared id overlaps a CL-3 target')
print('l2', len(l2), 'targets', len(tg), 'declared', len(dec), 'bad', bad); sys.exit(1 if bad else 0)
PYE
cat "${OUT}/meta/lists_check.txt"
"${PY}" "${TOOLS}/preflight_judge_selftest.py" > "${OUT}/meta/judge_selftest.txt" 2>&1 \
  || INVALID+=("judge selftest failed (see meta/judge_selftest.txt)")
tail -1 "${OUT}/meta/judge_selftest.txt"

git -C "${PROJECT}" cat-file -e "${PRE}^{commit}" 2>/dev/null || INVALID+=("PRE ${PRE} not in the repo")
git -C "${PROJECT}" cat-file -e "${POST}^{commit}" 2>/dev/null || INVALID+=("POST ${POST} not in the repo")
if [[ ${#INVALID[@]} -eq 0 ]]; then
  git -C "${PROJECT}" merge-base --is-ancestor "${PRE}" "${POST}" || INVALID+=("PRE is not an ancestor of POST")
  # which installed files may legitimately differ between the two arm trees (git diff of src/, mapped to the sync targets)
  git -C "${PROJECT}" diff --name-only "${PRE}" "${POST}" -- workspace/engine-port/src > "${OUT}/meta/src_diff_pre_post.txt"
  { echo "sglang/srt/multiplex/pdmux_context.py"      # patched in place by the device-assert patch (not a copied file)
    sed -n 's|^workspace/engine-port/src/multiplex/\(.*\)$|sglang/srt/multiplex/\1|p;
            s|^workspace/engine-port/src/models/\(.*\)$|sglang/srt/models/\1|p;
            s|^workspace/engine-port/src/configs/\(.*\)$|sglang/srt/configs/\1|p' "${OUT}/meta/src_diff_pre_post.txt"
  } | sort -u > "${OUT}/meta/allowed_tree_diff.txt"
  # patches changed other than the device-assert one would alter the arm trees in ways this script does not model
  grep 'src/patches/' "${OUT}/meta/src_diff_pre_post.txt" | grep -v 'pdmux_context_device_assert.patch' \
    > "${OUT}/meta/unmodelled_patch_changes.txt" && INVALID+=("src/patches changed beyond the device-assert patch")
  # the Job HEAD engine bytes vs POST: the arms are rebuilt from POST, so this is a note, not a failure
  # (R-H2) the same 7 gate paths as mg_const.PREFLIGHT_GATE_PATHS / mg_plan.preflight_gate (iii)
  git -C "${PROJECT}" diff --stat "${POST}" HEAD -- workspace/engine-port/src workspace/engine-port/tests \
    workspace/engine-port/scripts/bootstrap workspace/engine-port/scripts/r2_eval workspace/engine-port/env \
    workspace/engine-port/benchmarks/pdmux_eval workspace/engine-port/scripts/multigpu > "${OUT}/meta/post_vs_job_head.txt" 2>&1
fi

say "static checks done; invalid so far: ${#INVALID[@]}"
if [[ "${MODE}" == dry ]]; then
  cat <<PLAN
--- PLAN (dry-run; nothing was built) ---
 pre=${PRE:0:8} post=${POST:0:8}  reps=${REPS}  order: $( [[ ${REPS} -ge 2 ]] && echo "pre,post,post,pre (ABBA)" || echo "pre,post")
 per arm: git worktree of the commit under ${WORK}/<arm>/prefill-layer-alloc; dev tree = copy of ${SGLANG_ENGINE_DEV:-<image SGLANG_ENGINE_DEV>};
          pre arm: reverse device-assert patch + remove multigpu_guard.py, then that tree's install_runtime.sh; post arm: install_runtime.sh
 tests  : pf_unittest_runner.py --status-json LOG.status.json discover -s workspace/engine-port/tests  (= python -m unittest discover -v,
          plus per-id JSON; cwd = arm worktree; PDMUX_ROOT = arm root; PDMUX_PROJECT_ROOT unset)
 judge  : preflight_judge.py --require-structured --declared-cpu-timing (5 ids) $( [[ ${CPU_SPEC} == 1 ]] && echo --cpu-spec || echo "(not a CPU spec: declaration not applied)" )
          allowed tree diff: $(wc -l < "${OUT}/meta/allowed_tree_diff.txt" 2>/dev/null) file(s)
 invalid: ${INVALID[*]:-none}
PLAN
  [[ ${#INVALID[@]} -eq 0 ]] && exit 0 || exit 2
fi

# ---- 2. arm trees ------------------------------------------------------------------------------------------------
[[ -n "${SRC_DEV}" && -f "${SRC_DEV}/sglang/srt/distributed/parallel_state.py" ]] || INVALID+=("no SGLang tree at SRC_DEV='${SRC_DEV}'")
SRC_DEV_ROOT="$(dirname "${SRC_DEV}")"
declare -A WT DEV AROOT
build_arm() {   # arm sha
  local arm="$1" sha="$2" root="${WORK}/$1"
  local wt="${root}/prefill-layer-alloc" dev="${root}/sglang_engine_dev/python" m="${OUT}/meta/arm_${arm}"
  mkdir -p "${root}" "${m}"
  git -C "${PROJECT}" worktree add --detach "${wt}" "${sha}" > "${m}/worktree.log" 2>&1 || { INVALID+=("${arm}: worktree add failed"); return 1; }
  WTS+=("${wt}")
  [[ "$(git -C "${wt}" rev-parse HEAD)" == "${sha}" ]] || { INVALID+=("${arm}: worktree HEAD != ${sha}"); return 1; }
  [[ -d "${DATA}/hf" ]] && ln -sfn "${DATA}/hf" "${wt}/hf_cache"
  cp -a "${SRC_DEV_ROOT}" "${root}/sglang_engine_dev" || { INVALID+=("${arm}: dev tree copy failed"); return 1; }
  if [[ "${arm}" == pre ]]; then
    # The base tree was synced from the Job commit (post-like).  Take the merge-only effects back out of the copy.
    local pp="${WT[post]}/${PIN_PATCH}"
    if grep -q "PD-mux auto grid (divide_sm)" "${dev}/sglang/srt/multiplex/pdmux_context.py"; then
      patch --reverse --batch -p1 -d "${dev}" < "${pp}" > "${m}/reverse_patch.log" 2>&1 \
        || { INVALID+=("pre: cannot reverse the device-assert patch"); return 1; }
    fi
    rm -f "${dev}/sglang/srt/multiplex/multigpu_guard.py"
    grep -q "PD-mux auto grid (divide_sm)" "${dev}/sglang/srt/multiplex/pdmux_context.py" && { INVALID+=("pre: device-assert still present"); return 1; }
    [[ "$(sha256sum "${dev}/sglang/srt/multiplex/pdmux_context.py" | cut -d' ' -f1)" == "${PRISTINE_PDMUX_CONTEXT_SHA}" ]] \
      || INVALID+=("pre: pdmux_context.py after reversal != pristine sha (image base differs from the local tree?)")
    patch --dry-run --forward --batch -p1 -d "${dev}" < "${pp}" > "${m}/forward_check.log" 2>&1 \
      || INVALID+=("pre: reversed tree does not take the patch forward again (not the exact pre-image)")
  fi
  PDMUX_PROJECT_ROOT="${wt}" SGLANG_ENGINE_DEV="${dev}" \
    bash "${wt}/workspace/engine-port/scripts/vessl/install_runtime.sh" "${m}/manifest.sha256" > "${m}/install_runtime.log" 2>&1 \
    || { INVALID+=("${arm}: install_runtime.sh failed (see meta/arm_${arm}/install_runtime.log)"); return 1; }
  sha256sum --quiet -c "${m}/manifest.sha256" > "${m}/manifest_check.log" 2>&1 || { INVALID+=("${arm}: manifest check failed"); return 1; }
  WT[$arm]="${wt}"; DEV[$arm]="${dev}"; AROOT[$arm]="${root}"
  # which sglang does an interpreter with this arm's env import?  must be the arm copy, not the image's editable install
  PYTHONPATH="${dev}" "${PY}" -c 'import sglang, sys; print(sglang.__file__)' > "${m}/sglang_path.txt" 2> "${m}/sglang_import.err"
  grep -q "^${dev}/" "${m}/sglang_path.txt" || INVALID+=("${arm}: sglang imports from outside the arm tree ($(cat "${m}/sglang_path.txt" 2>/dev/null))")
  PYTHONPATH="${dev}" "${PY}" -c 'import sgl_kernel; print("sgl_kernel ok", sgl_kernel.__file__)' > "${m}/sgl_kernel.txt" 2>&1 \
    || SOFT+=("${arm}: import sgl_kernel failed (${m}/sgl_kernel.txt)")   # soft: finish building/checking both trees first
  return 0
}
if [[ ${#INVALID[@]} -eq 0 ]]; then
  # post first: pre needs the post worktree for the patch file; both from the same base copy.
  build_arm post "${POST}"; [[ ${#INVALID[@]} -eq 0 ]] && build_arm pre "${PRE}"
fi
if [[ ${#INVALID[@]} -eq 0 ]]; then
  # tree-difference fence: the two arm trees may differ only in files the git diff of src/ explains
  diff -rq --exclude=__pycache__ --exclude='*.pyc' --exclude='.pdmux_sync.lock' "${DEV[pre]}" "${DEV[post]}" > "${OUT}/meta/arm_tree_diff.txt" 2>&1
  : > "${OUT}/meta/arm_tree_diff_unexplained.txt"
  while IFS= read -r l; do
    f="$(sed -n "s|^Files [^ ]*/sglang_engine_dev/python/\([^ ]*\) and [^ ]* differ\$|\1|p; s|^Only in [^ ]*/sglang_engine_dev/python/\([^:]*\): \(.*\)\$|\1/\2|p" <<<"$l")"
    grep -qxF "${f}" "${OUT}/meta/allowed_tree_diff.txt" || echo "${l}" >> "${OUT}/meta/arm_tree_diff_unexplained.txt"
  done < "${OUT}/meta/arm_tree_diff.txt"
  [[ -s "${OUT}/meta/arm_tree_diff_unexplained.txt" ]] && INVALID+=("arm trees differ in files the src diff does not explain")
  for a in pre post; do sha256sum "${DEV[$a]}"/sglang/srt/multiplex/*.py | sed "s|${DEV[$a]}/||" > "${OUT}/meta/arm_${a}_multiplex_sha.txt"; done
fi
INVALID+=("${SOFT[@]+"${SOFT[@]}"}")
say "arm trees: invalid=${#INVALID[@]}"
if [[ "${MODE}" == dry_trees ]]; then
  echo "invalid: ${INVALID[*]:-none}"; cat "${OUT}/meta/arm_tree_diff.txt" 2>/dev/null | sed 's/^/diff /'
  [[ ${#INVALID[@]} -eq 0 ]] && exit 0 || exit 2
fi

# ---- 3. venv reachability (the 17 sbatch-driven tests source $PDMUX_VENV/bin/activate) -----------------------------
VENV_MODE=none; VENV=""
probe_venv() { bash -c 'set -euo pipefail; source "$1/bin/activate"; python -c "import sys, torch; print(sys.prefix)"' _ "$1" >> "${OUT}/meta/venv_probe.txt" 2>&1; }
if [[ ${#INVALID[@]} -eq 0 ]]; then
  : > "${OUT}/meta/venv_probe.txt"
  for cand in "${PDMUX_VENV:-}" "$("${PY}" -c 'import sys; print(sys.base_prefix)')"; do
    [[ -n "${cand}" && -f "${cand}/bin/activate" ]] || continue
    echo "== candidate ${cand}" >> "${OUT}/meta/venv_probe.txt"
    if probe_venv "${cand}"; then VENV="${cand}"; [[ "${cand}" == "${PDMUX_VENV:-}" ]] && VENV_MODE=env || VENV_MODE=image_python_prefix; break; fi
  done
  if [[ -z "${VENV}" ]]; then   # last resort: a real venv that sees the image's packages (provisions an env; changes no test and no code)
    cand="${WORK}/venv_syssite"; echo "== candidate ${cand} (created, --system-site-packages)" >> "${OUT}/meta/venv_probe.txt"
    { "${PY}" -m venv --system-site-packages "${cand}" || "${PY}" -m venv --without-pip --system-site-packages "${cand}"; } >> "${OUT}/meta/venv_probe.txt" 2>&1
    if [[ -f "${cand}/bin/activate" ]] && probe_venv "${cand}"; then VENV="${cand}"; VENV_MODE=created_system_site; fi
  fi
  echo "venv_mode=${VENV_MODE} venv=${VENV:-<none>}" | tee "${OUT}/meta/venv_mode.txt"
fi

# ---- 4. suites ------------------------------------------------------------------------------------------------------
declare -a PRE_LOGS=() POST_LOGS=()
tree_hash() { sha256sum "${DEV[$1]}"/sglang/srt/multiplex/*.py "${DEV[$1]}"/sglang/srt/managers/scheduler.py 2>/dev/null | cut -d' ' -f1 | sha256sum | cut -d' ' -f1; }
run_suite() {   # arm rep
  local arm="$1" rep="$2" log="${OUT}/logs/$1.rep$2.log" before after rc
  local -a unset_args=() cmd
  for v in $(compgen -e | grep '^PDMUX_'); do unset_args+=(-u "${v}"); done
  before="$(tree_hash "${arm}")"
  # r2 F3: the structured runner prints the same -v text and writes ${log}.status.json (per-id status from the TestResult callbacks)
  if [[ -n "${MODS}" ]]; then cmd=("${PY}" "${TOOLS}/pf_unittest_runner.py" --status-json "${log}.status.json" names ${MODS})
    cd "${WT[$arm]}/workspace/engine-port/tests" || return 1
  else cmd=("${PY}" "${TOOLS}/pf_unittest_runner.py" --status-json "${log}.status.json" discover -s workspace/engine-port/tests)
    cd "${WT[$arm]}" || return 1; fi
  say "suite ${arm} rep${rep} start"
  env ${unset_args[@]+"${unset_args[@]}"} PDMUX_ROOT="${AROOT[$arm]}" SGLANG_ENGINE_DEV="${DEV[$arm]}" PYTHONPATH="${DEV[$arm]}" \
      ${VENV:+PDMUX_VENV="${VENV}"} timeout -k 30 "${T_SUITE}" "${cmd[@]}" > "${log}" 2>&1
  rc=$?; echo "${rc}" > "${OUT}/logs/${arm}.rep${rep}.rc"
  after="$(tree_hash "${arm}")"
  [[ "${before}" == "${after}" ]] || INVALID+=("${arm} rep${rep}: dev tree changed during the suite (an sbatch-driven test re-synced it)")
  [[ ${rc} -eq 124 || ${rc} -eq 137 ]] && INVALID+=("${arm} rep${rep}: suite timed out (rc ${rc})")
  say "suite ${arm} rep${rep} rc=${rc} $(tail -3 "${log}" | tr '\n' ' ')"
  cd "${PROJECT}" || true
}
if [[ ${#INVALID[@]} -eq 0 ]]; then
  for ((i = 1; i <= REPS; i++)); do
    if (( i % 2 == 1 )); then order=(pre post); else order=(post pre); fi
    for arm in "${order[@]}"; do
      run_suite "${arm}" "${i}"
      if [[ "${arm}" == pre ]]; then PRE_LOGS+=("${OUT}/logs/pre.rep${i}.log"); else POST_LOGS+=("${OUT}/logs/post.rep${i}.log"); fi
    done
  done
fi

# ---- 5. judge --------------------------------------------------------------------------------------------------------
inv_args=(); for x in "${INVALID[@]+"${INVALID[@]}"}"; do inv_args+=(--invalid "${x}"); done
[[ -n "${MODS}" ]] && inv_args+=(--invalid "PDMUX_PF_MODULES smoke subset: not a full preflight")
[[ ${SPEC_RC} -eq 0 || ${SPEC_RC} -eq 1 ]] || inv_args+=(--invalid "spec decision rc ${SPEC_RC}: $(tr '\n' ' ' < "${OUT}/meta/spec_decision.txt")")
DECL="${TOOLS}/preflight_cl3_cpu_timing_declared.txt"
judge_r2=(--require-structured --declared-cpu-timing "${DECL}"); [[ ${CPU_SPEC} == 1 ]] && judge_r2+=(--cpu-spec)
if [[ ${#PRE_LOGS[@]} -eq 0 || ${#POST_LOGS[@]} -eq 0 ]]; then
  echo "PREFLIGHT_UNVERIFIABLE" > "${OUT}/PREFLIGHT_LABEL"; printf 'invalid: %s\n' "${INVALID[@]}" >> "${OUT}/PREFLIGHT_LABEL"; jrc=3
else
  "${PY}" "${TOOLS}/preflight_judge.py" --pre "${PRE_LOGS[@]}" --post "${POST_LOGS[@]}" "${inv_args[@]+"${inv_args[@]}"}" \
    "${judge_r2[@]}" --json "${OUT}/judge.json" > "${OUT}/PREFLIGHT_LABEL" 2>&1
  jrc=$?
fi
cat "${OUT}/PREFLIGHT_LABEL" | head -30

# ---- 6. gate artifact (M3 rev12 §19-6 (i), harness audit r1 R-H2): PREFLIGHT_PASS.json written HERE from judge.json -------------
#      verdict = the judge label (PASS only if the label is PREFLIGHT_PASS and every field is present); image = PDMUX_IMAGE;
#      merge_tree_engine_digest = pathless-v1 of the POST arm's install manifest (the ENGINE_DIGEST.txt procedure, POST tree's tool).
if [[ "${MODE}" == job ]]; then
  "${PY}" "${TOOLS}/preflight_judge.py" record --judge-json "${OUT}/judge.json" --post-commit "${POST}" \
    --merge-manifest "${OUT}/meta/arm_post/manifest.sha256" \
    --engine-digest-py "${WT[post]:-/nonexistent}/workspace/engine-port/scripts/bootstrap/engine_digest.py" \
    --image "${PDMUX_IMAGE:-}" --job-name "${JOB}" --declared-cpu-timing "${DECL}" \
    --out "${OUT}/PREFLIGHT_PASS.json" > "${OUT}/meta/pass_record.txt" 2>&1
  prc=$?; cat "${OUT}/meta/pass_record.txt"
  [[ ${prc} -ne 0 && ${jrc} -eq 0 ]] && jrc=3              # a PASS label without a complete gate record is not a PASS
fi
echo "elapsed_s=${SECONDS}" > "${OUT}/meta/elapsed.txt"
say "label: $(head -1 "${OUT}/PREFLIGHT_LABEL") (rc ${jrc})"
exit "${jrc}"
