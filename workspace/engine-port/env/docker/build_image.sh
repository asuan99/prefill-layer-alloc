#!/bin/bash
# Stage a clean build context and build the pdmux-sglang image locally (no GPU needed).
#   - SGLang source = `git archive v0.5.10` of the dev tree (NOT its working copy, which carries
#     the sync + manual edits already; the image must reproduce them from tracked sources).
#   - engine-port subset = `git archive <commit>` (committed state only; uncommitted edits are
#     refused so the image label can name the commit that produced it).
# Usage: build_image.sh [tag]   (default tag pdmux-sglang:<short-commit>)
# Env:   PDMUX_ROOT (default: parent of the repo), SGLANG_SRC (default $PDMUX_ROOT/sglang_engine_dev)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${script_dir}" rev-parse --show-toplevel)"
pdmux_root="${PDMUX_ROOT:-$(dirname "${repo}")}"
sglang_src="${SGLANG_SRC:-${pdmux_root}/sglang_engine_dev}"
commit="$(git -C "${repo}" rev-parse HEAD)"
tag="${1:-pdmux-sglang:${commit:0:7}}"

# env/docker (this directory) is excluded: it is build tooling, copied from the working tree.
guard=(workspace/engine-port/src workspace/engine-port/scripts/bootstrap workspace/engine-port/env
       ':!workspace/engine-port/env/docker')
if [[ -n "$(git -C "${repo}" status --porcelain -- "${guard[@]}")" ]]; then
  echo "ERROR: uncommitted changes under engine-port/{src,scripts/bootstrap,env}; commit first" >&2
  git -C "${repo}" status --short -- "${guard[@]}" >&2
  exit 2
fi

ctx="$(mktemp -d "${TMPDIR:-/tmp}/pdmux-imgctx.XXXXXX")"
trap 'rm -rf "${ctx}"' EXIT
mkdir -p "${ctx}/sglang_engine_dev" "${ctx}/engine-port"
git -C "${sglang_src}" archive v0.5.10 | tar -x -C "${ctx}/sglang_engine_dev"
git -C "${repo}" archive "${commit}" workspace/engine-port/src workspace/engine-port/scripts/bootstrap workspace/engine-port/env \
  | tar -x --strip-components=2 -C "${ctx}/engine-port"
cp "${script_dir}/Dockerfile.pdmux-sglang" "${script_dir}/requirements.pdmux-sglang.txt" "${ctx}/"

docker build --progress=plain -f "${ctx}/Dockerfile.pdmux-sglang" \
  --build-arg REPO_COMMIT="${commit}" -t "${tag}" "${ctx}"
echo "built ${tag} (repo ${commit}, sglang $(git -C "${sglang_src}" rev-parse v0.5.10^{commit}))"
