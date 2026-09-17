#!/usr/bin/bash
# Pack everything that lives ONLY on this machine (not in git, not re-downloadable)
# into a small number of archives, so it can be copied to temporary storage when
# leaving the KISTI Neuron workspace (2026-09-17 migration).
#
# Usage (call with /usr/bin/bash — see memory login-shell-bash-function):
#   /usr/bin/bash tools/migration/pack_migration_bundle.sh results [DEST]  # untracked result artifacts
#   /usr/bin/bash tools/migration/pack_migration_bundle.sh final   [DEST]  # git bundle, Claude memory/tools, env, misc
#   /usr/bin/bash tools/migration/pack_migration_bundle.sh extra   [DEST]  # hand-made HF config dirs, editor settings
#   /usr/bin/bash tools/migration/pack_migration_bundle.sh hf_models DEST [REPO ...]  # optional: HF cache repos as .tar
#   /usr/bin/bash tools/migration/pack_migration_bundle.sh sums    [DEST]  # MANIFEST.tsv + SHA256SUMS
#   /usr/bin/bash tools/migration/pack_migration_bundle.sh verify  [DEST]  # sha256sum -c (run again after copying)
#
# Run `final` only after the handoff commit, so the git bundle contains it.
# Not packed by default (regenerable or secret): hf_cache/hub model weights (opt-in `hf_models`),
# venvs, pip/triton caches, external git clones (pinned in the handoff doc),
# ~/.claude/.credentials.json, .git/config (its remote URL embeds a token).
set -euo pipefail

stage="${1:?usage: pack_migration_bundle.sh results|final|sums|verify [DEST]}"
dest="${2:-/scratch/ehmoon/whlee/_migration_2026-09-17}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace="$(dirname "${repo}")"
claude_home="${HOME}/.claude"
claude_proj="${claude_home}/projects/-scratch-ehmoon-whlee"
zstd_opts=(-T4 -3 -q)

tmp=""
make_tmp() {  # only packing stages write; verify must work on a read-only copy
  mkdir -p "${dest}"
  tmp="$(mktemp -d "${dest}/.lists.XXXXXX")"
  trap 'rm -rf "${tmp}"' EXIT
}

# NUL-separated list of untracked+ignored files under a pathspec, minus caches.
untracked_files() {
  git -C "${repo}" status --ignored --porcelain=v1 --untracked-files=all -z -- "$@" \
    | tr '\0' '\n' | cut -c4- \
    | grep -vE '__pycache__/|\.pytest_cache/|(^|/)[^/]*triton[^/]*cache[^/]*/|(^|/)[^/]*cache[^/]*triton[^/]*/' \
    || true
}

pack_list() {  # pack_list <list-file (newline paths, relative to $2)> <root> <out.tar.zst>
  local list="$1" root="$2" out="$3"
  [[ -s "${list}" ]] || { echo "skip (empty): ${out}"; return 0; }
  tr '\n' '\0' < "${list}" \
    | tar -C "${root}" --null --no-recursion -T - -cf - \
    | zstd "${zstd_opts[@]}" -o "${out}.part"
  mv -f "${out}.part" "${out}"
  echo "packed $(wc -l < "${list}") files -> ${out} ($(du -h "${out}" | cut -f1))"
}

stage_results() {
  local out="${dest}/C_results_untracked"
  mkdir -p "${out}/telemetry_jsonl"
  untracked_files workspace/engine-port/results > "${tmp}/results.txt"
  grep -vE '\.jsonl(\.tmp)?$' "${tmp}/results.txt" > "${tmp}/nontelemetry.txt" || true
  pack_list "${tmp}/nontelemetry.txt" "${repo}" "${out}/results_nontelemetry.tar.zst"
  # One archive per campaign so a partial copy can prioritise; re-runs skip done ones.
  grep -E '\.jsonl(\.tmp)?$' "${tmp}/results.txt" \
    | awk -F/ -v d="${tmp}" '{c = (NF >= 5) ? $4 : "_toplevel"; print > (d "/jsonl_" c ".txt")}'
  local list campaign target
  for list in "${tmp}"/jsonl_*.txt; do
    [[ -e "${list}" ]] || continue
    campaign="$(basename "${list}" .txt)"; campaign="${campaign#jsonl_}"
    target="${out}/telemetry_jsonl/${campaign}.tar.zst"
    if [[ -f "${target}" ]]; then echo "exists, skip: ${target}"; continue; fi
    pack_list "${list}" "${repo}" "${target}"
  done
}

stage_final() {
  # A. git: every ref, including local-only branches. No .git/config (token).
  mkdir -p "${dest}/A_git"
  git -C "${repo}" bundle create "${dest}/A_git/prefill-layer-alloc.bundle" --all
  git -C "${repo}" bundle verify "${dest}/A_git/prefill-layer-alloc.bundle"
  git -C "${repo}" for-each-ref --format='%(refname) %(objectname)' \
    > "${dest}/A_git/refs.txt"

  # B. Claude Code: memory + project agents/skills + user settings (no credentials).
  mkdir -p "${dest}/B_claude"
  {
    echo "${claude_proj#/}/memory"
    echo "${workspace#/}/CLAUDE.md"
    echo "${workspace#/}/.claude"
    for f in settings.json plans skills; do
      [[ -e "${claude_home}/${f}" ]] && echo "${claude_home#/}/${f}"
    done
  } > "${tmp}/claude.txt"
  tr '\n' '\0' < "${tmp}/claude.txt" \
    | tar -C / --null -T - -czf "${dest}/B_claude/claude_memory_tools.tar.gz"
  # Session transcripts (optional, ~350 MB): everything in the project dir but memory.
  find "${claude_proj}" -mindepth 1 -maxdepth 1 ! -name memory -printf '%P\n' \
    | sed "s#^#${claude_proj#/}/#" > "${tmp}/transcripts.txt"
  # The live session keeps appending to its transcript; GNU tar exits 1 on
  # "file changed as we read it", which is acceptable here (anything >1 is not).
  local st
  set +e
  tr '\n' '\0' < "${tmp}/transcripts.txt" \
    | tar -C / --null --warning=no-file-changed -T - -cf - \
    | zstd "${zstd_opts[@]}" -o "${dest}/B_claude/claude_transcripts.tar.zst"
  st=("${PIPESTATUS[@]}")
  set -e
  (( st[1] <= 1 && st[2] == 0 )) || { echo "transcript archive failed: ${st[*]}" >&2; exit 1; }

  # D. Untracked, non-result files inside the repo (logs, figures, stray SLURM output,
  #    muxwise-zenodo which is not a git clone), minus venvs/caches/clones/hf_cache.
  mkdir -p "${dest}/D_misc"
  untracked_files . \
    | grep -vE '^workspace/engine-port/results/|^hf_cache/|^(bin|lib|include)/|^lib64$|^pyvenv\.cfg$|/\.venv/|^workspace/engine-port/external/(sglang-latest|muxwise|bullet)/' \
    > "${tmp}/misc.txt"
  pack_list "${tmp}/misc.txt" "${repo}" "${dest}/D_misc/repo_untracked_misc.tar.zst"
  # Derived workload traces (ShareGPT/LongBench filtered copies) used as bench inputs.
  (cd "${repo}" && find hf_cache/raw -type f) > "${tmp}/traces.txt"
  pack_list "${tmp}/traces.txt" "${repo}" "${dest}/D_misc/hf_cache_raw_traces.tar.zst"
  # Model snapshot pins (weights themselves are re-downloaded from the Hub).
  for d in "${repo}"/hf_cache/hub/models--* "${repo}"/hf_cache/hub/datasets--*; do
    # the scratch purge renamed some refs/main to refs/ToBeDelete_main
    printf '%s\t%s\n' "$(basename "${d}")" \
      "$(cat "${d}/refs/main" 2>/dev/null || cat "${d}/refs/ToBeDelete_main" 2>/dev/null || echo '?')"
  done > "${dest}/D_misc/hf_hub_revisions.tsv"
  if [[ -d "${workspace}/vllm_bench" ]]; then
    tar -C "${workspace}" -czf "${dest}/D_misc/vllm_bench.tar.gz" vllm_bench
  fi

  # E. Exact engine runtime source (also reproducible: v0.5.10 + sync + manual-edits patch).
  mkdir -p "${dest}/E_env"
  tar -C "${workspace}" --exclude=__pycache__ -cf - sglang_engine_dev \
    | zstd "${zstd_opts[@]}" -o "${dest}/E_env/sglang_engine_dev_src.tar.zst"
  cp -f "${workspace}/sglang_engine_venv/pyvenv.cfg" "${dest}/E_env/sglang_engine_venv.pyvenv.cfg"
  for x in bullet muxwise sglang-latest; do
    printf '%s\t%s\t%s\n' "${x}" \
      "$(git -C "${repo}/workspace/engine-port/external/${x}" rev-parse HEAD)" \
      "$(git -C "${repo}/workspace/engine-port/external/${x}" remote get-url origin | sed -E 's#//[^@/]*@#//#')"
  done > "${dest}/E_env/external_clone_pins.tsv"
}

stage_extra() {
  # Hand-made model dirs (config/tokenizer edits; weights are symlinks into hub/,
  # stored as links) and workspace editor settings.
  mkdir -p "${dest}/D_misc"
  tar -C "${repo}/hf_cache" -czf "${dest}/D_misc/hf_converted_model_dirs.tar.gz" \
    mamba2-2.7b-sglang mamba-codestral-7b-sglang
  tar -C "${workspace}" -czf "${dest}/D_misc/workspace_dotfiles.tar.gz" .vscode
}

stage_hf_models() {  # optional, large: one uncompressed tar per HF cache repo
  local hub="${repo}/hf_cache/hub" r purged
  if (( $# == 0 )); then
    echo "usage: pack_migration_bundle.sh hf_models DEST REPO... (e.g. models--Zyphra--Zamba2-2.7B)"
    for r in "${hub}"/models--* "${hub}"/datasets--*; do
      purged="$(find "${r}" -name 'ToBeDelete_*' | wc -l)"
      printf '  %-55s %8s  purged=%s\n' "$(basename "${r}")" "$(du -sh "${r}" | cut -f1)" "${purged}"
    done
    return 0
  fi
  mkdir -p "${dest}/F_hf_models"
  for r in "$@"; do
    [[ -d "${hub}/${r}" ]] || { echo "no such repo: ${r}" >&2; exit 2; }
    purged="$(find "${hub}/${r}" -name 'ToBeDelete_*' | wc -l)"
    if (( purged > 0 )); then
      echo "skip ${r}: ${purged} files already renamed by the scratch purge - re-download instead" >&2
      continue
    fi
    # snapshots/ hold relative symlinks into blobs/, which tar keeps as links.
    tar -C "${repo}/hf_cache" -cf "${dest}/F_hf_models/${r}.tar.part" "hub/${r}"
    mv -f "${dest}/F_hf_models/${r}.tar.part" "${dest}/F_hf_models/${r}.tar"
    echo "packed ${r} ($(du -sh "${dest}/F_hf_models/${r}.tar" | cut -f1))"
  done
}

stage_sums() {
  (cd "${dest}" && find . -type f ! -name SHA256SUMS ! -name MANIFEST.tsv ! -path './.lists.*' \
     -printf '%s\t%P\n' | sort -k2 > MANIFEST.tsv \
   && cut -f2 MANIFEST.tsv | tr '\n' '\0' | xargs -0 sha256sum > SHA256SUMS)
  echo "wrote ${dest}/MANIFEST.tsv and SHA256SUMS ($(wc -l < "${dest}/SHA256SUMS") files, $(du -sh "${dest}" | cut -f1))"
}

case "${stage}" in
  results) make_tmp; stage_results ;;
  final)   make_tmp; stage_final ;;
  extra)   stage_extra ;;
  hf_models) shift 2 || shift $#; stage_hf_models "$@" ;;
  sums)    stage_sums ;;
  verify)  (cd "${dest}" && sha256sum -c --quiet SHA256SUMS && echo "OK: all checksums match") ;;
  *) echo "unknown stage: ${stage}" >&2; exit 2 ;;
esac
