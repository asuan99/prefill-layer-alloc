#!/usr/bin/bash
# INBOX 뷰: 각 트랙 파일(`<worktree>/tracks/<트랙>.md`)의 "사용자 결정 대기" 절만 모아 출력한다.
# 파일을 만들지 않는다 → 병행 세션이 같은 파일에 쓰는 일이 없다. 트랙 파일은 자기 worktree(branch)에 있으므로
# 병합해도 경로가 겹치지 않는다. 같은 트랙 파일이 여러 worktree에 있으면(병합 뒤) 그 트랙 worktree 쪽이 최신이다.
#
# 2026-10-19 지연 경보 추가(읽기 전용): 트랙 파일의 마지막 커밋보다 그 트랙 결과 경로의 마지막 커밋이 새로우면
# "⚠ 지연" 한 줄을 낸다. 결정·감사 원문이 커밋됐는데 트랙 파일이 아직 옛 "결정 대기"를 보이는 경우를 잡기 위해서다
# (2026-10-15 Q1 사례). 커밋되지 않은 변경 수도 함께 보인다. 경보는 판정이 아니라 "트랙 파일을 다시 볼 것" 신호다.
ROOT=${KISTI_ROOT:-$HOME/Experiments/KISTI}
declare -A seen

# 트랙별 결과 경로(pathspec). 지정이 없으면 그 worktree 전체에서 tracks/를 뺀 것.
result_paths() {
  case "$1" in
    main)      echo "workspace/engine-port/results :(exclude)workspace/engine-port/results/power_cap" ;;
    power-cap) echo "workspace/engine-port/results/power_cap" ;;
    *)         echo ". :(exclude)tracks" ;;
  esac
}

stale_line() {   # $1=worktree $2=트랙 이름
  local w=$1 name=$2 t_track t_work work_info dirty
  t_track=$(git -C "$w" log -1 --format=%ct -- "tracks/$name.md" 2>/dev/null)
  [ -n "$t_track" ] || return 0
  # shellcheck disable=SC2046
  work_info=$(git -C "$w" log -1 --format='%ct %h %s' -- $(result_paths "$name") 2>/dev/null)
  t_work=${work_info%% *}
  # shellcheck disable=SC2046
  dirty=$(git -C "$w" status --porcelain -- $(result_paths "$name") 2>/dev/null | wc -l)
  if [ -n "$t_work" ] && [ "$t_work" -gt "$t_track" ]; then
    printf '⚠ 지연: 트랙 파일 %s < 결과 커밋 %s (%s)\n' \
      "$(date -d "@$t_track" '+%F %H:%M')" "$(date -d "@$t_work" '+%F %H:%M')" \
      "$(echo "$work_info" | cut -d' ' -f2- | cut -c1-90)"
  fi
  [ "$dirty" -gt 0 ] && printf '  (결과 경로 미커밋 변경 %s건)\n' "$dirty"
  return 0
}

# 트랙의 집 worktree. 2026-10-19 수정: 예전에는 glob 순서(-amd, -multigpu, -opchar, main)대로 처음 본 파일을 썼기 때문에
# multigpu 브랜치에 남은 옛 main.md·power-cap.md 사본이 main worktree의 최신본을 가렸다. 집 worktree를 먼저 읽는다.
home_of() {
  case "$1" in
    main|power-cap) echo "$ROOT/prefill-layer-alloc" ;;
    *)              echo "$ROOT/prefill-layer-alloc-$1" ;;
  esac
}
files=()
for f in "$ROOT"/prefill-layer-alloc/tracks/*.md "$ROOT"/prefill-layer-alloc-*/tracks/*.md; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .md); h=$(home_of "$name")
  [ "${f%/tracks/*}" = "$h" ] && files=("$f" "${files[@]}") || files+=("$f")
done
for f in "${files[@]}"; do
    w=${f%/tracks/*}
    name=$(basename "$f" .md); [ -n "${seen[$name]}" ] && continue; seen[$name]=1
    # 2026-10-21: 폐기된 트랙(파일 앞 10줄에 `<!-- track: retired -->`)은 이력 파일이라 건너뛴다.
    head -n 10 "$f" | grep -q '<!-- track: retired -->' && continue
    sec=$(awk '/^## 사용자 결정 대기/{on=1;next} /^## /{on=0} on && NF' "$f")
    warn=$(stale_line "$w" "$name")
    [ -n "$sec$warn" ] && printf '### %s  (%s)\n%s%s\n\n' "$name" "${f#$ROOT/}" "${warn:+$warn
}" "$sec"
done
