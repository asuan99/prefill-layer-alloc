#!/bin/bash
# stage_llama.sh -- one-time staging of the contrast model meta-llama/Llama-3.1-8B-Instruct (gated) onto the
# cluster volume, revision-pinned, from the CPU Workspace `pdmux-build`.  Decision 12' (DECISION_headline_G_
# shapeC_2026-10-19.md): the HF token travels ONLY over SSH stdin.
#
# TOKEN HANDLING (the rules this script enforces)
#   * The token is read from stdin by one python process (huggingface_hub.login(token=...), the library call
#     behind `hf auth login`; the CLI itself would need a tty or put the token in argv).  It is never in argv,
#     never in an exported variable, never written to the log, never echoed.  No `set -x` anywhere.
#   * The credential file lives in a private temp dir on the container's LOCAL disk (HF_TOKEN_PATH), not on
#     /data, so it cannot persist on the cluster volume.  `hf auth logout` + removal of that dir run from an
#     EXIT/INT/TERM trap, i.e. also when the download fails.  A final check says whether any token file is left.
#   * The python error output is masked (hf_[A-Za-z0-9]+ -> hf_***) before it is printed.
#
# LOCAL USAGE (from the local PC; the token value is never shown, it goes file -> ssh stdin):
#   scp workspace/engine-port/scripts/vessl/stage_llama.sh <ws>:/tmp/stage_llama.sh        # or run from /opt/pdmux
#   ssh <ws> 'bash /tmp/stage_llama.sh' < ~/.cache/huggingface/token
#   Output: STAGE_LLAMA lines only (no token).  `--check` (no stdin needed) lists what is already staged.
#
# WHAT IT DOES (order fixed by the request):
#   1. hf auth login (token from stdin)           2. download rev-pinned safetensors + config + tokenizer only
#   3. hf auth logout (always, via trap)          4. sha256 list of the downloaded files -> $HF_HOME/STAGED_LLAMA_SHA256SUMS
#   refs/main of the cache entry is set to the pinned revision so an offline `--model-path meta-llama/...`
#   resolves (stage_data.sh convention).
# ENV  HF_HOME (default /data/hf).   Needs network (this is the only script that does; Jobs/probes are offline).
set -uo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
REV="0e9e39f249a16976918f6564b8830bc894c89659"
export HF_HOME="${HF_HOME:-/data/hf}"
export PATH="/opt/conda/bin:${PATH}"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
key="models--${MODEL//\//--}"
snap="${HF_HOME}/hub/${key}/snapshots/${REV}"

if [[ "${1:-}" == "--check" ]]; then
  echo "STAGE_LLAMA check: refs/main=$(cat "${HF_HOME}/hub/${key}/refs/main" 2>/dev/null || echo MISSING) pinned=${REV}"
  ls "${snap}" 2>/dev/null | head -30
  exit 0
fi

# stdin must be the token (not a terminal): refuse to prompt, refuse an empty stdin
if [[ -t 0 ]]; then echo "STAGE_LLAMA ERROR: stdin is a terminal; pipe the token in (see header)" >&2; exit 2; fi

CRED_DIR="$(mktemp -d /tmp/stage_llama.XXXXXX)" || exit 2      # local disk, mode 700
chmod 700 "${CRED_DIR}"
export HF_TOKEN_PATH="${CRED_DIR}/token"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN     # never let an inherited env token bypass the login step
logged_in=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ ${logged_in} == 1 || -e "${HF_TOKEN_PATH}" ]]; then
    hf auth logout > "${CRED_DIR}/logout.log" 2>&1 || true
    echo "STAGE_LLAMA logout: done (rc of hf auth logout ignored; credential dir removed next)"
  fi
  rm -rf "${CRED_DIR}"
  if [[ -e "${HF_TOKEN_PATH}" || -e "${HF_HOME}/token" ]]; then
    echo "STAGE_LLAMA WARNING: a token file exists: ${HF_TOKEN_PATH} / ${HF_HOME}/token -- inspect (do not print it)"
  else
    echo "STAGE_LLAMA no token file left (checked HF_TOKEN_PATH and ${HF_HOME}/token)"
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

mkdir -p "${HF_HOME}"
# ---- 1. login: the whole of stdin is the token; python reads it, bash never holds it
if python3 -c '
import sys
tok = sys.stdin.read().strip()
if not tok:
    sys.exit("empty token on stdin")
from huggingface_hub import login
login(token=tok, add_to_git_credential=False)
' 2> "${CRED_DIR}/login.err"; then
  logged_in=1; echo "STAGE_LLAMA login: ok"
else
  sed -E 's/hf_[A-Za-z0-9]+/hf_***/g' "${CRED_DIR}/login.err" | tail -n 3 >&2
  echo "STAGE_LLAMA ERROR: login failed" >&2; exit 3
fi

# ---- 2. download: safetensors + config + tokenizer only, revision pinned
export HF_HUB_OFFLINE=0
if hf download "${MODEL}" --revision "${REV}" \
     --include "*.safetensors" "model.safetensors.index.json" "config.json" "generation_config.json" \
               "tokenizer*" "special_tokens_map.json" --exclude "original/*" \
     > "${CRED_DIR}/download.log" 2>&1; then
  echo "STAGE_LLAMA download: ok"
else
  sed -E 's/hf_[A-Za-z0-9]+/hf_***/g' "${CRED_DIR}/download.log" | tail -n 5 >&2
  echo "STAGE_LLAMA ERROR: download failed" >&2; exit 4
fi
refdir="${HF_HOME}/hub/${key}/refs"; mkdir -p "${refdir}"; printf '%s' "${REV}" > "${refdir}/main"
printf '%s\tmodel\t%s\t%s\tok\n' "$(utc)" "${MODEL}" "${REV}" >> "${HF_HOME}/STAGING_LOG.tsv" 2>/dev/null || true

# ---- 3. logout is done by the trap (runs now, on exit); do it explicitly first so the sha step is token-free
hf auth logout > "${CRED_DIR}/logout.log" 2>&1 || true
logged_in=0; rm -f "${HF_TOKEN_PATH}"
echo "STAGE_LLAMA logout: done"

# ---- 4. sha256 of what was downloaded (follows the blob symlinks)
if [[ -d "${snap}" ]]; then
  (cd "${snap}" && find . -type f -o -type l | sort | xargs -r sha256sum -- ) > "${HF_HOME}/STAGED_LLAMA_SHA256SUMS"
  echo "STAGE_LLAMA files: $(wc -l < "${HF_HOME}/STAGED_LLAMA_SHA256SUMS") -> ${HF_HOME}/STAGED_LLAMA_SHA256SUMS"
  cut -c1-16,65- "${HF_HOME}/STAGED_LLAMA_SHA256SUMS"
  du -shL "${snap}"
else
  echo "STAGE_LLAMA ERROR: snapshot ${snap} missing after download" >&2; exit 5
fi
exit 0
