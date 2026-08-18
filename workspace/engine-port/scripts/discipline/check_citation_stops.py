#!/usr/bin/env python3
"""커밋에 **새로 추가되는 줄**이 정본 인용금지를 다시 쓰는지 검사한다.

왜 이 형태인가 (설계 이력을 남긴다):
  1차 설계는 저장소 전체를 스캔했다 -> 40파일에 30건이 떴고 **거의 전부 정당**했다.
  사전등록은 `NO_TAX`를 정의하면서 금지 문구를 써야 하고, 승격금지 목록은 금지 수치를
  열거해야 한다.  그 정도 소음이면 도구를 끄게 되므로 **없느니만 못하다.**
  ⇒ 실패 모드에 맞춘다: 2026-08-18의 실제 사고는 "**새 문서가 금지 수치를 교차검증으로
  인용**"이었다.  따라서 **스테이징된 diff의 추가 줄(+)만** 검사한다.

사용:
    python3 check_citation_stops.py              # git diff --cached 의 추가 줄
    python3 check_citation_stops.py --range A..B # 임의 리비전 범위
    python3 check_citation_stops.py --file F     # 파일 전체(수동 점검용)
종료: 0 = 통과, 1 = 위반, 2 = 사용법/레지스트리 오류.

한계(정직하게):
  - 추가 줄만 보므로 **기존 줄에 남은 위반은 못 잡는다.** 그건 감사의 몫이다.
  - 철회 문맥은 `[CS-OK]` 마커로만 면제한다(문맥 추론은 1차 설계에서 실패했다).
  - 레지스트리가 비면 통과시키지 않고 **에러로 죽는다**(빈 서명 구멍 방지).
"""
import re, sys, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "citation_stops.tsv")
OK_MARK = "[CS-OK]"          # 이 마커가 줄에 있으면 면제(의식적 인정 강제)

def load():
    rules = []
    for line in open(REG, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) != 3:
            print(f"REGISTRY_MALFORMED: {line!r}", file=sys.stderr); sys.exit(2)
        rules.append((re.compile(p[0]), p[1], p[2]))
    if not rules:
        print("REGISTRY_EMPTY -- refusing to pass", file=sys.stderr); sys.exit(2)
    return rules

def added_lines(rng=None):
    cmd = ["git", "diff", "-U0"] + ([rng] if rng else ["--cached"])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    path, ln = None, 0
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]; continue
        m = re.match(r"@@ -\S+ \+(\d+)", line)
        if m:
            ln = int(m.group(1)); continue
        if line.startswith("+") and not line.startswith("+++"):
            yield path, ln, line[1:]; ln += 1

def main():
    rules = load()
    args = sys.argv[1:]
    if args and args[0] == "--file":
        src = ((args[1], i, l) for i, l in
               enumerate(open(args[1], encoding="utf-8", errors="replace"), 1))
    elif args and args[0] == "--range":
        src = added_lines(args[1])
    elif args:
        print(__doc__); sys.exit(2)
    else:
        src = added_lines()
    hits = n = 0
    for path, i, line in src:
        n += 1
        if OK_MARK in line:
            continue
        for rx, s, why in rules:
            if rx.search(line):
                hits += 1
                print(f"{path}:{i}: CITATION_STOP /{rx.pattern}/ [{s}] {why}")
                print(f"    {line.strip()[:110]}")
    print(f"--- {n} added line(s), {hits} violation(s), {len(rules)} rules"
          + ("" if hits else "  OK"))
    if hits:
        print(f"  정당한 언급이면 그 줄에 {OK_MARK} 를 붙여 의식적으로 면제하라.")
    sys.exit(1 if hits else 0)

if __name__ == "__main__":
    main()
