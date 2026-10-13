#!/usr/bin/bash
# INBOX 뷰: 각 트랙 파일(`<worktree>/tracks/<트랙>.md`)의 "사용자 결정 대기" 절만 모아 출력한다.
# 파일을 만들지 않는다 → 병행 세션이 같은 파일에 쓰는 일이 없다. 트랙 파일은 자기 worktree(branch)에 있으므로
# 병합해도 경로가 겹치지 않는다. 같은 트랙 파일이 여러 worktree에 있으면(병합 뒤) 그 트랙 worktree 쪽이 최신이다.
ROOT=${KISTI_ROOT:-$HOME/Experiments/KISTI}
declare -A seen
for w in "$ROOT"/prefill-layer-alloc-* "$ROOT"/prefill-layer-alloc; do
  for f in "$w"/tracks/*.md; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .md); [ -n "${seen[$name]}" ] && continue; seen[$name]=1
    sec=$(awk '/^## 사용자 결정 대기/{on=1;next} /^## /{on=0} on && NF' "$f")
    [ -n "$sec" ] && printf '### %s  (%s)\n%s\n\n' "$name" "${f#$ROOT/}" "$sec"
  done
done
