#!/bin/bash
# Stage a clean build context and build the PUBLIC sglang-runtime image locally (no GPU needed).
#   - SGLang source = `git archive v0.5.10` of the dev tree (pristine upstream; NOT its working copy,
#     which carries the PD-mux sync + manual edits -- those must never enter the public image).
#   - recipe = this directory (Dockerfile, pinned requirements) + env/sitecustomize.py, committed state
#     only, so the image label can name the commit that produced it.
#   - NO engine-port src/patches: they are installed at container start (scripts/vessl/install_runtime.sh).
# Usage: build_image.sh [tag]   (default tag sglang-runtime:<short-commit>)
# Env:   PDMUX_ROOT (default: parent of the repo), SGLANG_SRC (default $PDMUX_ROOT/sglang_engine_dev)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${script_dir}" rev-parse --show-toplevel)"
pdmux_root="${PDMUX_ROOT:-$(dirname "${repo}")}"
sglang_src="${SGLANG_SRC:-${pdmux_root}/sglang_engine_dev}"
commit="$(git -C "${repo}" rev-parse HEAD)"
tag="${1:-sglang-runtime:${commit:0:7}}"

guard=(workspace/engine-port/env/docker workspace/engine-port/env/sitecustomize.py)
if [[ -n "$(git -C "${repo}" status --porcelain -- "${guard[@]}")" ]]; then
  echo "ERROR: uncommitted changes in the image recipe; commit first" >&2
  git -C "${repo}" status --short -- "${guard[@]}" >&2
  exit 2
fi

ctx="$(mktemp -d "${TMPDIR:-/tmp}/sglang-runtime-ctx.XXXXXX")"
trap 'rm -rf "${ctx}"' EXIT
mkdir -p "${ctx}/sglang_engine_dev"
git -C "${sglang_src}" archive v0.5.10 | tar -x -C "${ctx}/sglang_engine_dev"
git -C "${repo}" show "${commit}:workspace/engine-port/env/docker/Dockerfile.sglang-runtime" > "${ctx}/Dockerfile.sglang-runtime"
git -C "${repo}" show "${commit}:workspace/engine-port/env/docker/requirements.pdmux-sglang.txt" > "${ctx}/requirements.pdmux-sglang.txt"
git -C "${repo}" show "${commit}:workspace/engine-port/env/sitecustomize.py" > "${ctx}/sitecustomize.py"
# Guard: no project patch marker may be in the context (upstream itself ships multiplexing_mixin.py,
# so check markers that only our patches add).
if grep -rqs "pdmux_role_is_thread_local\|maybe_create_holb_probe\|maybe_install_chunk_probe" "${ctx}/sglang_engine_dev"; then
  echo "ERROR: build context SGLang tree is not pristine upstream" >&2; exit 2
fi

docker build --progress=plain -f "${ctx}/Dockerfile.sglang-runtime" \
  --build-arg REPO_COMMIT="${commit}" -t "${tag}" "${ctx}"
echo "built ${tag} (repo ${commit}, sglang $(git -C "${sglang_src}" rev-parse v0.5.10^{commit}))"
