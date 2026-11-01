#!/bin/bash
# Install the project's PD-mux engine code into the PUBLIC sglang-runtime image at container start.
#
#   install_runtime.sh <manifest_out>
#
# WHY.  The image is public (VESSL cannot pull from a private registry), so it carries only pristine
# SGLang v0.5.10.  Everything project-specific comes from the commit checked out at
# $PDMUX_PROJECT_ROOT (cloned from the org-private bundle by job_entry.sh, or by hand in a Workspace):
#   0. scripts/vessl/engine_tree.py preserve      -- keep a verified pristine copy (sglang_pristine/) first
#   1. scripts/bootstrap/sync_engine_tree.sh  -- src/ + tracked patches, writes the hash manifest
#   2. env/devtree_manual_edits.patch         -- the 5 manual-edit files (same order as the old image
#                                                build and env/cpu_regression_baseline_2026-09-18.md sec 1)
# Safe to re-run in a Workspace: sync is grep-guarded, and step 2 is skipped when already applied.
# Writes <manifest_out> (sync manifest) and <manifest_out>.install.txt (what was applied).
set -euo pipefail

manifest="${1:?usage: install_runtime.sh <manifest_out>}"
project="${PDMUX_PROJECT_ROOT:?set PDMUX_PROJECT_ROOT to the checked-out prefill-layer-alloc}"
runtime="${SGLANG_ENGINE_DEV:?image must set SGLANG_ENGINE_DEV}"
engine="${project}/workspace/engine-port"
manual="${engine}/env/devtree_manual_edits.patch"
mkdir -p "$(dirname "${manifest}")"

# 0. Pristine copy FIRST (2026-11-01).  Steps 1-2 patch ${runtime} IN PLACE, so before them keep a verified copy of
#    the image's tree at <runtime>/../../sglang_pristine/python (image: /opt/pdmux/sglang_pristine/python; cp -a,
#    checked against the committed env/sglang_v0.5.10_python_tree.sha256 = `git archive v0.5.10 python`;
#    status + manifests next to <manifest_out>).
#    Server boots choose a tree with PDMUX_ENGINE_TREE=pristine|patched via scripts/vessl/engine_tree.py boot (default
#    patched).  A failed preservation (e.g. a Workspace tree patched by an earlier run) is recorded and NOT fatal --
#    the patched install below is unchanged either way -- unless PDMUX_REQUIRE_PRISTINE=1 (exit 4).
#    PDMUX_PRISTINE_REF (local tests) swaps the reference; the status then says ref_overridden=1.
#    Lives here, not in job_entry.sh: job_entry.sh bytes are sealed by the cap_char harness digests.
pristine_ovr=(); [[ -n "${PDMUX_PRISTINE_REF:-}" ]] && pristine_ovr=(--ref-overridden)
if python3 -I -S "${engine}/scripts/vessl/engine_tree.py" preserve --src "${runtime}" \
     --dst "$(dirname "$(dirname "${runtime}")")/sglang_pristine" \
     --ref "${PDMUX_PRISTINE_REF:-${engine}/env/sglang_v0.5.10_python_tree.sha256}" \
     "${pristine_ovr[@]+"${pristine_ovr[@]}"}" --meta "$(dirname "${manifest}")"; then
  echo "pristine tree preserved"
elif [[ "${PDMUX_REQUIRE_PRISTINE:-0}" == 1 ]]; then
  echo "ERROR: PDMUX_REQUIRE_PRISTINE=1 and the pristine SGLang tree could not be preserved" >&2
  exit 4
else
  echo "WARNING: pristine SGLang tree NOT preserved (PDMUX_ENGINE_TREE=pristine boots will refuse)" >&2
fi

bash "${engine}/scripts/bootstrap/sync_engine_tree.sh" "${manifest}"

# Forward check FIRST.  `--batch` makes patch silently flip direction when a patch "looks
# reversed", so a bare `--reverse --dry-run` succeeds on an UNAPPLIED tree (caught 2026-10-02 by the
# byte-equivalence test against the old image).  `--force` disables that auto-flip.
if patch --dry-run --forward --batch -p1 -d "${runtime}" < "${manual}" > /dev/null 2>&1; then
  patch --forward --batch -p1 -d "${runtime}" < "${manual}"
  manual_state="applied"
elif patch --dry-run --reverse --force -p1 -d "${runtime}" < "${manual}" > /dev/null 2>&1; then
  manual_state="already-applied"
else
  echo "ERROR: devtree_manual_edits.patch neither applies nor is already applied" >&2
  exit 4
fi
# Re-hash AFTER the manual edits (they touch scheduler.py, a manifest entry), so the manifest
# describes the tree that actually runs -- the same post-edit state the KISTI live tree had when
# every historical job ran sync.  Sync is idempotent (grep-guarded), so this only rewrites hashes.
bash "${engine}/scripts/bootstrap/sync_engine_tree.sh" "${manifest}" > /dev/null

{
  echo "commit=$(git -C "${project}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "runtime=${runtime}"
  echo "manual_edits_patch=${manual_state} sha256=$(sha256sum "${manual}" | cut -d' ' -f1)"
  echo "sync_manifest_sha256=$(sha256sum "${manifest}" | cut -d' ' -f1)"
} > "${manifest}.install.txt"
cat "${manifest}.install.txt"
