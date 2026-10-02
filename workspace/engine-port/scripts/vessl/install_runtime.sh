#!/bin/bash
# Install the project's PD-mux engine code into the PUBLIC sglang-runtime image at container start.
#
#   install_runtime.sh <manifest_out>
#
# WHY.  The image is public (VESSL cannot pull from a private registry), so it carries only pristine
# SGLang v0.5.10.  Everything project-specific comes from the commit checked out at
# $PDMUX_PROJECT_ROOT (cloned from the org-private bundle by job_entry.sh, or by hand in a Workspace):
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
