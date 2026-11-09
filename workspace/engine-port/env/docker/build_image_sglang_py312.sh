#!/bin/bash
# Build the PUBLIC image sglang-py312-runtime (sglang-vllm-runtime base byte-unchanged + /opt/sg312-venv; no project code).
# Usage: build_image_sglang_py312.sh [tag]   (default tag sglang-py312-runtime:0.5.10-<short commit>)
# Push is NOT done here: docker tag/push to ghcr.io/asuan99/sglang-py312-runtime, then record the digest in
# scripts/vessl/vessl.env (+ .example) as PDMUX_VESSL_IMAGE_SG312. The ghcr package must be made PUBLIC.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${script_dir}" rev-parse --show-toplevel)"
commit="$(git -C "${repo}" rev-parse HEAD)"
tag="${1:-sglang-py312-runtime:0.5.10-${commit:0:8}}"
if [[ -n "$(git -C "${repo}" status --porcelain -- workspace/engine-port/env/docker)" ]]; then
  echo "ERROR: uncommitted changes in the image recipe; commit first" >&2; exit 2
fi
ctx="$(mktemp -d "${TMPDIR:-/tmp}/sglang-py312-runtime-ctx.XXXXXX")"
trap 'rm -rf "${ctx}"' EXIT
git -C "${repo}" show "${commit}:workspace/engine-port/env/docker/Dockerfile.sglang-py312-runtime" > "${ctx}/Dockerfile.sglang-py312-runtime"
docker build --progress=plain -f "${ctx}/Dockerfile.sglang-py312-runtime" \
  --build-arg REPO_COMMIT="${commit}" -t "${tag}" "${ctx}"
echo "built ${tag} (recipe ${commit})"
