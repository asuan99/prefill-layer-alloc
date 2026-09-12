#!/bin/bash
# Keep the Claude Code tool definitions (agents, skills, workspace CLAUDE.md)
# under version control even though they must LIVE outside this repository.
#
# Why this exists (gate #157)
#   The agent/skill files encode the audit and discipline rules that canonical
#   documents cite by SHA-256 (e.g. PROJECT_STATUS.md anchors
#   .claude/agents/claims-auditor.md).  Those files live at
#   /scratch/ehmoon/whlee, which is NOT a git repository (its .git is an empty
#   directory), so until now their content had no history: a rule could change
#   with no diff, and the only provenance was a hash pasted into prose by hand.
#   Claude Code discovers agents/skills from the workspace root it is started
#   in, so the live copies cannot simply be moved in here.  Instead this repo
#   keeps the authoritative copies and this script moves bytes between the two
#   locations and proves they agree.
#
# Direction of truth
#   tracked copy (this directory) = authoritative, reviewable, has history.
#   live copy (workspace root)    = what Claude Code actually reads.
#   --import  live  -> tracked   (capture edits made in place, then commit)
#   --install tracked -> live    (restore on a fresh clone / another machine)
#   --check   compare both       (default; exit 3 on any drift)
#
# Honest limits
#   * `--check` run immediately after `--import` is an identity and carries no
#     information; it is informative only about drift that happened LATER.
#   * The check compares FILE SETS as well as hashes, because a deleted or
#     newly added tool file is exactly the kind of change a hash list of known
#     paths cannot see (de-confound lesson 89).
#   * Nothing here makes the live files immutable; it makes changes visible.
#
# usage:
#   sync_claude_tools.sh [--check | --import | --install | --manifest]
#   CLAUDE_TOOLS_LIVE=<dir> CLAUDE_TOOLS_TRACK=<dir> ...   (tests override both)
set -uo pipefail

track_root="${CLAUDE_TOOLS_TRACK:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
live_root="${CLAUDE_TOOLS_LIVE:-/scratch/ehmoon/whlee}"
manifest="${track_root}/claude_tools.manifest.sha256"
mode="${1:---check}"

# live path <-> tracked path.  One entry per tracked file; the workspace
# CLAUDE.md is stored under a different name so that it is not picked up as a
# second set of project instructions for this subdirectory.
pairs () {
  local f n
  printf '%s\t%s\n' "${live_root}/CLAUDE.md" "${track_root}/workspace/CLAUDE.workspace.md"
  for f in "${live_root}"/.claude/agents/*.md; do
    [[ -e "$f" ]] || continue
    printf '%s\t%s\n' "$f" "${track_root}/agents/$(basename "$f")"
  done
  for f in "${live_root}"/.claude/skills/*/SKILL.md; do
    [[ -e "$f" ]] || continue
    n="$(basename "$(dirname "$f")")"
    printf '%s\t%s\n' "$f" "${track_root}/skills/${n}/SKILL.md"
  done
  # Tracked files whose live counterpart is gone would be invisible to the loop
  # above (it walks the live side), so walk the tracked side too.
  for f in "${track_root}"/agents/*.md; do
    [[ -e "$f" ]] || continue
    printf '%s\t%s\n' "${live_root}/.claude/agents/$(basename "$f")" "$f"
  done
  for f in "${track_root}"/skills/*/SKILL.md; do
    [[ -e "$f" ]] || continue
    n="$(basename "$(dirname "$f")")"
    printf '%s\t%s\n' "${live_root}/.claude/skills/${n}/SKILL.md" "$f"
  done
}

write_manifest () {
  local tmp f
  tmp="$(mktemp "${manifest}.XXXXXX")" || return 2
  {
    echo "# sha256 of the TRACKED copies, $(date -Is)"
    while IFS=$'\t' read -r _ tf; do
      [[ -f "$tf" ]] && echo "$tf"
    done < <(pairs) | sort -u | xargs -r sha256sum | sed "s#${track_root}/##"
  } > "$tmp" && mv -f "$tmp" "$manifest" || { rm -f "$tmp"; return 2; }
  echo "manifest: ${manifest}"
}

case "$mode" in
  --check)
    rc=0
    while IFS=$'\t' read -r lf tf; do
      rel="${tf#"${track_root}/"}"
      if [[ ! -f "$lf" && ! -f "$tf" ]]; then continue; fi
      if [[ ! -f "$lf" ]]; then echo "MISSING_LIVE     ${rel}"; rc=3; continue; fi
      if [[ ! -f "$tf" ]]; then echo "UNTRACKED_LIVE   ${rel}  (${lf})"; rc=3; continue; fi
      if ! cmp -s "$lf" "$tf"; then echo "DRIFT            ${rel}"; rc=3; fi
    done < <(pairs | sort -u)
    if (( rc == 0 )); then
      echo "OK  $(pairs | sort -u | wc -l) tool files identical (live == tracked)"
      echo "    live=${live_root}  tracked=${track_root}"
    else
      echo "--- drift found: run --import (live is right) or --install (tracked is right)"
    fi
    exit "$rc" ;;
  --import|--install)
    n=0
    while IFS=$'\t' read -r lf tf; do
      if [[ "$mode" == --import ]]; then src="$lf"; dst="$tf"; else src="$tf"; dst="$lf"; fi
      [[ -f "$src" ]] || { echo "skip (no source): $src"; continue; }
      install -D -m 0644 "$src" "$dst" && n=$((n + 1))
    done < <(pairs | sort -u)
    echo "$mode: $n files"
    write_manifest || exit 2 ;;
  --manifest)
    write_manifest || exit 2 ;;
  *)
    sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
