#!/bin/bash
# Self-check for job_entry.sh CPU-only mode (2026-10-16) and the pristine/patched SGLang tree selection
# (install_runtime.sh step 0 + engine_tree.py boot, 2026-11-01).  Local, CPU.
# Tests HEAD plus the WORKING-TREE state of scripts/vessl/ and env/sglang_v0.5.10_python_tree.* (snapshot commit in a
# throwaway bare --shared clone; the real repo's index, refs and HEAD are not touched).
# The tree cases run the REAL install_runtime.sh inside job_entry.sh on `git archive v0.5.10 python` of the local
# SGLang checkout ($SGLANG_SRC, default <repo>/../sglang_engine_dev); they are skipped when that tag is unavailable.
#   test_job_entry_cpu.sh                 run the checks
#   (internal) with PDMUX_JE_SELFTEST_INNER=1 it is the trivial Job target -- the guard stops self-recursion
set -uo pipefail
if [[ "${1:-}" == "--as-target" || "${PDMUX_JE_SELFTEST_INNER:-}" == 1 ]]; then
  echo "target ran; LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  out="workspace/engine-port/results/jetest"; mkdir -p "${out}"
  env | LC_ALL=C sort > "${out}/target_env.txt"
  if [[ "${PDMUX_JE_SELFTEST_BOOT:-}" == 1 ]]; then        # exercise engine_tree.py boot inside the Job
    et=(python3 -I -S workspace/engine-port/scripts/vessl/engine_tree.py boot)
    allow=(--allow-env PDMUX_JE_SELFTEST_INNER --allow-env PDMUX_JE_SELFTEST_BOOT)
    # the "server": where `import sglang` would come from + what PD-mux env it sees (no sglang import: no deps here)
    probe='import importlib.util, json, os; print(json.dumps({"file": importlib.util.find_spec("sglang").origin, "pdmux": sorted(k for k in os.environ if k.startswith("PDMUX_")), "pythonpath": os.environ.get("PYTHONPATH", "")}))'
    b() { local n="$1"; shift; "$@" > "${out}/$n.out" 2> "${out}/$n.err"; echo "rc_$n=$?" >> "${out}/rc.txt"; }
    b patched  "${et[@]}" --stamp "${out}/patched.json" -- python3 -c "${probe}"
    b pristine env PDMUX_ENGINE_TREE=pristine "${et[@]}" --stamp "${out}/pristine.json" "${allow[@]}" -- python3 -c "${probe}"
    b knob     env PDMUX_ENGINE_TREE=pristine PDMUX_R2_POLICY=fixed "${et[@]}" --stamp "${out}/knob.json" "${allow[@]}" -- python3 -c "${probe}"
    b arg      env PDMUX_ENGINE_TREE=pristine "${et[@]}" --stamp "${out}/arg.json" "${allow[@]}" -- python3 -c "${probe}" --enable-pdmux
    b noallow  env PDMUX_ENGINE_TREE=pristine "${et[@]}" --stamp "${out}/noallow.json" -- python3 -c "${probe}"
    b bogus    env PDMUX_ENGINE_TREE=bogus "${et[@]}" --stamp "${out}/bogus.json" -- python3 -c "${probe}"
  fi
  exit 0
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo="$(git -C "${here}" rev-parse --show-toplevel)"
T="${TMPDIR:-/tmp}/jecpu.$$"; mkdir -p "${T}"; trap 'rm -rf "${T}"' EXIT
fails=0
chk() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails+1)); fi; }
# --- snapshot: HEAD + working tree of the paths under test, committed in a bare --shared clone
snap="${T}/snap.git"; git clone -q --bare --shared "${repo}" "${snap}"
sg() { GIT_DIR="${snap}" GIT_INDEX_FILE="${T}/snap.idx" GIT_WORK_TREE="${repo}" git "$@"; }
sg read-tree HEAD
(cd "${repo}" && sg add -A -- workspace/engine-port/scripts/vessl $(ls workspace/engine-port/env/sglang_v0.5.10_python_tree.* 2>/dev/null))   # pathspec is cwd-relative: run from the repo root
commit="$(GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t \
  sg commit-tree "$(sg write-tree)" -p "$(git -C "${repo}" rev-parse HEAD)" -m "test_job_entry_cpu snapshot")"
GIT_DIR="${snap}" git update-ref refs/heads/jetest "${commit}"; GIT_DIR="${snap}" git symbolic-ref HEAD refs/heads/jetest
mkdir -p "${T}/io/code/${commit}"
GIT_DIR="${snap}" git bundle create "${T}/io/code/${commit}/repo.bundle" HEAD jetest >/dev/null 2>&1
(cd "${T}/io/code/${commit}" && sha256sum repo.bundle > repo.bundle.sha256)
run() {  # name spec extra-env...
  local n="$1" spec="$2"; shift 2
  env -i PATH="${PATH}" HOME="${T}" PDMUX_COMMIT="${commit}" PDMUX_JOB_NAME="$n" PDMUX_CAMPAIGN=t PDMUX_IMAGE=img@sha256:0 \
    PDMUX_SPEC="${spec}" PDMUX_TARGET=workspace/engine-port/scripts/vessl/test_job_entry_cpu.sh \
    PDMUX_IO="${T}/io" PDMUX_DATA="${T}/data" PDMUX_OPT="${T}/opt/$n" PDMUX_HEARTBEAT_S=0 PDMUX_JE_SELFTEST_INNER=1 "$@" \
    bash "${here}/job_entry.sh" > "${T}/$n.out" 2>&1
}
mkdir -p "${T}/data/hf" "${T}/opt/c1" "${T}/opt/c2" "${T}/opt/g"
# --- CPU by env flag and by spec slug
run c1 resourcespec-x PDMUX_TEST_SKIP_SYNC=1 PDMUX_CPU_ONLY=1; run c2 resourcespec-a100cpu PDMUX_TEST_SKIP_SYNC=1
for n in c1 c2; do R="${T}/data/runs/$n/meta"
  chk "$n DONE exit_code=0" "grep -q '^exit_code=0' ${T}/io/results/$n/DONE"
  chk "$n CPU_ONLY.txt recorded" "grep -q '^cpu_only=1' ${R}/CPU_ONLY.txt"
  chk "$n substrate.json cpu_only true, gpu null" "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(not (r[\"cpu_only\"] is True and r[\"gpu\"] is None))' ${R}/substrate.json"
  chk "$n no clk.csv / energy.csv / substrate_history" "[[ ! -e ${R}/clk.csv && ! -e ${R}/energy.csv && ! -e ${T}/data/substrate_history.tsv ]]"
  chk "$n [PHASE] CPU-only line in stdout" "grep -q 'PHASE .*CPU-only mode' ${T}/$n.out"
  chk "$n SHA256SUMS verifies" "(cd ${T}/io/results/$n && sha256sum -c --quiet SHA256SUMS)"
done
# --- GPU path untouched: only when this machine has nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  mkdir -p "${T}/opt/g"; run g resourcespec-a100x1 PDMUX_TEST_SKIP_SYNC=1; R="${T}/data/runs/g/meta"
  chk "gpu path: no CPU_ONLY.txt, substrate.json has gpu, clk.csv present" "[[ ! -e ${R}/CPU_ONLY.txt && -e ${R}/clk.csv ]] && python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))[\"gpu\"] is None)' ${R}/substrate.json"
else echo "skip gpu-path run (no GPU here)"; fi
# --- GPU lines of job_entry.sh are byte-identical to the base: changes may only INSERT lines (tested state = snapshot).
#     The pristine-tree work does not touch job_entry.sh at all (its bytes are sealed by the cap_char harness digests).
base="${PDMUX_JE_BASE:-e43749c91b350a702fbdc7487767c84d6835d793}"
dels="$(GIT_DIR="${snap}" git diff "${base}" "${commit}" -- workspace/engine-port/scripts/vessl/job_entry.sh | grep -c '^-[^-]')"
chk "job_entry.sh vs ${base:0:8}: 0 deleted lines" "[[ ${dels} -eq 0 ]]"

# --- pristine/patched tree selection (2026-11-01) on the real v0.5.10 tree ----------------------------------------
sgsrc="${SGLANG_SRC:-$(dirname "${repo}")/sglang_engine_dev}"
if git -C "${sgsrc}" rev-parse -q --verify 'v0.5.10^{commit}' >/dev/null 2>&1 && command -v patch >/dev/null && command -v flock >/dev/null; then
  mkimg() {  # <opt>: image layout = git archive v0.5.10 + what `pip install -e` leaves in python/
    mkdir -p "$1/sglang_engine_dev"; git -C "${sgsrc}" archive v0.5.10 python | tar -x -C "$1/sglang_engine_dev"
    printf "__version__ = version = '0.0.0.dev0'\n" > "$1/sglang_engine_dev/python/sglang/_version.py"
    mkdir -p "$1/sglang_engine_dev/python/sglang.egg-info"; printf 'Name: sglang\n' > "$1/sglang_engine_dev/python/sglang.egg-info/PKG-INFO"
  }
  et=(python3 -I -S "${here}/engine_tree.py")
  ref="${here}/../../env/sglang_v0.5.10_python_tree.sha256"; ref_sha="$(sha256sum "${ref}" | cut -d' ' -f1)"
  # p1: real install inside job_entry (no SKIP_SYNC), then boot both trees from the target
  mkimg "${T}/opt/p1"; P1="${T}/opt/p1/sglang_engine_dev/python"; PR="${T}/opt/p1/sglang_pristine/python"
  run p1 resourcespec-a100cpu SGLANG_ENGINE_DEV="${P1}" PYTHONPATH="${P1}" PDMUX_JE_SELFTEST_BOOT=1
  R="${T}/data/runs/p1"; J="${R}/results/jetest"
  chk "p1 DONE exit_code=0 (real install_runtime ran)" "grep -q '^exit_code=0' ${T}/io/results/p1/DONE && grep -q '^manual_edits_patch=applied' ${R}/meta/runtime_source_manifest.sha256.install.txt"
  chk "p1 preserve OK against the COMMITTED v0.5.10 reference, not overridden" "grep -q '^status=OK' ${R}/meta/engine_tree_pristine_status.txt && grep -q '^ref_overridden=0' ${R}/meta/engine_tree_pristine_status.txt && grep -q \"^ref_sha256=${ref_sha}\$\" ${R}/meta/engine_tree_pristine_status.txt"
  chk "p1 preserved manifest == reference bytes; _version.py the only extra" "cmp -s ${ref} ${R}/meta/engine_tree_pristine_tree_manifest.sha256 && grep -q '^extras=sglang/_version.py$' ${R}/meta/engine_tree_pristine_status.txt"
  chk "p1 marker guard: preserved copy has no project marker, installed tree has all three" "! grep -rqsE 'pdmux_role_is_thread_local|maybe_create_holb_probe|maybe_install_chunk_probe' ${PR} && for m in pdmux_role_is_thread_local maybe_create_holb_probe maybe_install_chunk_probe; do grep -rqs \$m ${P1}/sglang/srt || exit 1; done"
  chk "p1 boot return codes (patched 0, pristine 0, refusals 6)" "[[ \"\$(tr '\n' ' ' < ${J}/rc.txt)\" == 'rc_patched=0 rc_pristine=0 rc_knob=6 rc_arg=6 rc_noallow=6 rc_bogus=6 ' ]]"
  chk "p1 patched boot: sglang from the installed image tree, env passed through unchanged" "python3 - ${J}/patched.out ${P1} <<'PY'
import json, sys; r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
sys.exit(not (r['file'] == sys.argv[2] + '/sglang/__init__.py' and 'PDMUX_COMMIT' in r['pdmux'] and r['pythonpath'] == sys.argv[2]))
PY"
  chk "p1 pristine boot: sglang from the copy, zero PDMUX_* in the server env, image tree dropped from PYTHONPATH" "python3 - ${J}/pristine.out ${PR} <<'PY'
import json, sys; r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
sys.exit(not (r['file'] == sys.argv[2] + '/sglang/__init__.py' and r['pdmux'] == [] and r['pythonpath'] == sys.argv[2]))
PY"
  chk "p1 stamps: engine_tree, manifest sha (pristine == committed ref sha), markers" "python3 - ${J} ${ref_sha} <<'PY'
import json, sys
d, ref = sys.argv[1], sys.argv[2]
p = json.load(open(d + '/pristine.json')); q = json.load(open(d + '/patched.json'))
ok = (p['engine_tree'] == 'pristine' and p['engine_tree_manifest_sha256'] == ref == p['engine_tree_ref_sha256']
      and p['engine_tree_ref_overridden'] is False and p['engine_tree_markers_found'] == [] and p['status'] == 'EXEC'
      and q['engine_tree'] == 'patched' and q['engine_tree_manifest_sha256'] != ref and q['engine_tree_ref_sha256'] is None
      and {'pdmux_role_is_thread_local', 'maybe_create_holb_probe', 'maybe_install_chunk_probe'} <= set(q['engine_tree_markers_found']))
sys.exit(not ok)
PY"
  chk "p1 refusals are explained" "grep -q 'PDMUX_R2_POLICY' ${J}/knob.err && grep -q 'enable-pdmux' ${J}/arg.err && grep -q 'PDMUX_JE_SELFTEST' ${J}/noallow.err && grep -q 'expected one of' ${J}/bogus.err && [[ -e ${J}/knob.json.refused.json && ! -e ${J}/knob.json ]]"
  # p2: tree already patched (Workspace re-run) + PDMUX_REQUIRE_PRISTINE=1 -> install_runtime exit 4 -> job exit 4
  mkimg "${T}/opt/p2"; printf '\ndef pdmux_role_is_thread_local(): pass\n' >> "${T}/opt/p2/sglang_engine_dev/python/sglang/srt/distributed/parallel_state.py"
  run p2 resourcespec-a100cpu SGLANG_ENGINE_DEV="${T}/opt/p2/sglang_engine_dev/python" PDMUX_REQUIRE_PRISTINE=1
  chk "p2 patched source + REQUIRE -> DONE exit_code=4, reason recorded, nothing preserved" "grep -q '^exit_code=4' ${T}/io/results/p2/DONE && grep -q 'not pristine' ${T}/data/runs/p2/meta/engine_tree_pristine_status.txt && [[ ! -e ${T}/opt/p2/sglang_pristine/python ]]"
  # p3: same patched source WITHOUT REQUIRE -> preservation FAIL recorded, install and job proceed (exit 0)
  mkimg "${T}/opt/p3"; printf '\ndef pdmux_role_is_thread_local(): pass\n' >> "${T}/opt/p3/sglang_engine_dev/python/sglang/srt/distributed/parallel_state.py"
  run p3 resourcespec-a100cpu SGLANG_ENGINE_DEV="${T}/opt/p3/sglang_engine_dev/python"
  chk "p3 patched source, no REQUIRE -> FAIL recorded, job exit 0" "grep -q '^exit_code=0' ${T}/io/results/p3/DONE && grep -q '^status=FAIL' ${T}/data/runs/p3/meta/engine_tree_pristine_status.txt"
  # default unchanged: HEAD install_runtime.sh vs this one on identical trees -> identical installed tree + manifests
  git -C "${repo}" show HEAD:workspace/engine-port/scripts/vessl/install_runtime.sh > "${T}/install_runtime_head.sh"
  mkimg "${T}/d_head"; mkimg "${T}/d_new"
  for v in head new; do
    s="${here}/install_runtime.sh"; [[ $v == head ]] && s="${T}/install_runtime_head.sh"
    env -i PATH="${PATH}" HOME="${T}" PDMUX_ROOT="${T}/d_$v" PDMUX_PROJECT_ROOT="${repo}" \
      SGLANG_ENGINE_DEV="${T}/d_$v/sglang_engine_dev/python" bash "$s" "${T}/d_$v/meta/m.sha256" > "${T}/d_$v.log" 2>&1
    echo "rc=$?" >> "${T}/d_$v.log"
  done
  normd() { sed -e "s|${T}/d_head|D|g; s|${T}/d_new|D|g" "$1"; }
  chk "default: HEAD and new install_runtime both rc=0" "grep -q '^rc=0' ${T}/d_head.log && grep -q '^rc=0' ${T}/d_new.log"
  chk "default: installed (patched) tree byte-identical to HEAD install_runtime's" "diff <(${et[*]} manifest ${T}/d_head/sglang_engine_dev/python) <(${et[*]} manifest ${T}/d_new/sglang_engine_dev/python) >/dev/null"
  chk "default: sync manifest + install.txt identical (paths normalised; sync_manifest_sha256 hashes absolute paths, so it is compared through the normalised manifest)" "diff <(normd ${T}/d_head/meta/m.sha256) <(normd ${T}/d_new/meta/m.sha256) >/dev/null && diff <(normd ${T}/d_head/meta/m.sha256.install.txt | grep -v '^sync_manifest_sha256=') <(normd ${T}/d_new/meta/m.sha256.install.txt | grep -v '^sync_manifest_sha256=') >/dev/null"
  chk "default: new install_runtime only ADDS files (pristine records) next to the manifest" "diff <(cd ${T}/d_head/meta && ls) <(cd ${T}/d_new/meta && ls | grep -v '^engine_tree_pristine_') >/dev/null && [[ -d ${T}/d_new/sglang_pristine/python && ! -e ${T}/d_head/sglang_pristine ]]"
else
  echo "skip pristine/patched tree cases (no ${sgsrc} with tag v0.5.10, or patch/flock missing)"
fi
echo "failures=${fails}"; exit $((fails > 0))
