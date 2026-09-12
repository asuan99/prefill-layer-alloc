#!/bin/bash
# Mutation tests for sync_claude_tools.sh --check.
#
# A drift checker that cannot fail is worthless (de-confound lesson 53: a test
# of a repair must fail on a variant that undoes the repair).  Each case below
# breaks the live/tracked agreement in one specific way and asserts that
# --check exits 3 AND names the right file with the right marker.  Runs on
# throwaway copies under a temp dir; never touches the real workspace.
#
# usage: test_sync_claude_tools.sh        (exit 0 = all cases behaved)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
live_src="${CLAUDE_TOOLS_LIVE:-/scratch/ehmoon/whlee}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fails=0

reset () {
  rm -rf "$tmp/live" "$tmp/track"
  mkdir -p "$tmp/live/.claude" "$tmp/track"
  cp -r "$here"/agents "$here"/skills "$here"/workspace "$tmp/track/"
  cp -f "$live_src/CLAUDE.md" "$tmp/live/"
  cp -rf "$live_src/.claude/agents" "$live_src/.claude/skills" "$tmp/live/.claude/"
}

check () { CLAUDE_TOOLS_LIVE="$tmp/live" CLAUDE_TOOLS_TRACK="$tmp/track" \
  /usr/bin/bash "$here/sync_claude_tools.sh" --check > "$tmp/out" 2>&1; echo $?; }

case_ () {  # $1 name, $2 expected rc, $3 expected grep pattern ("" = none)
  local name="$1" want="$2" pat="$3" rc
  rc="$(check)"
  if [[ "$rc" != "$want" ]]; then
    echo "FAIL ${name}: exit ${rc}, expected ${want}"; sed 's/^/     /' "$tmp/out"; fails=$((fails+1)); return
  fi
  if [[ -n "$pat" ]] && ! grep -qE "$pat" "$tmp/out"; then
    echo "FAIL ${name}: exit ok but output lacks /${pat}/"; sed 's/^/     /' "$tmp/out"; fails=$((fails+1)); return
  fi
  echo "ok   ${name}"
}

reset; case_ "M0 pristine copies agree" 0 "^OK  11 tool files identical"
reset; printf '\n# drift\n' >> "$tmp/live/.claude/agents/doc-steward.md"
case_ "M1 live agent edited" 3 "^DRIFT +agents/doc-steward\.md"
reset; printf '\n# drift\n' >> "$tmp/live/CLAUDE.md"
case_ "M2 live workspace CLAUDE.md edited" 3 "^DRIFT +workspace/CLAUDE\.workspace\.md"
reset; rm -f "$tmp/live/.claude/agents/result-analyst.md"
case_ "M3 live agent deleted" 3 "^MISSING_LIVE +agents/result-analyst\.md"
reset; rm -f "$tmp/live/.claude/skills/handoff/SKILL.md"
case_ "M4 live skill deleted" 3 "^MISSING_LIVE +skills/handoff/SKILL\.md"
reset; echo x > "$tmp/live/.claude/agents/brand-new.md"
case_ "M5 new live agent untracked" 3 "^UNTRACKED_LIVE +agents/brand-new\.md"
reset; mkdir -p "$tmp/live/.claude/skills/newskill"; echo x > "$tmp/live/.claude/skills/newskill/SKILL.md"
case_ "M6 new live skill untracked" 3 "^UNTRACKED_LIVE +skills/newskill/SKILL\.md"
reset; rm -f "$tmp/track/agents/venue-strategist.md"
case_ "M7 tracked copy deleted" 3 "^UNTRACKED_LIVE +agents/venue-strategist\.md"
reset; rm -f "$tmp/live/.claude/agents/engine-porter.md"
CLAUDE_TOOLS_LIVE="$tmp/live" CLAUDE_TOOLS_TRACK="$tmp/track" \
  /usr/bin/bash "$here/sync_claude_tools.sh" --install > /dev/null 2>&1
case_ "M8 --install restores a deleted live file" 0 "^OK  11 tool files identical"
reset; printf '\n# live is right\n' >> "$tmp/live/.claude/agents/claims-auditor.md"
CLAUDE_TOOLS_LIVE="$tmp/live" CLAUDE_TOOLS_TRACK="$tmp/track" \
  /usr/bin/bash "$here/sync_claude_tools.sh" --import > /dev/null 2>&1
case_ "M9 --import captures a live edit" 0 "^OK  11 tool files identical"

echo
if (( fails )); then echo "$fails case(s) FAILED"; exit 1; fi
echo "all cases behaved (drift is detected, repair modes work)"
