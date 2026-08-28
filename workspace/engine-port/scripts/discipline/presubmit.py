#!/usr/bin/env python3
"""제출 전 체크리스트 — 규율 도구들의 **실행 지점**.

배경: 이 저장소는 규율 도구를 **네 개** 갖고 있는데 **셋 다 등록된 실행 지점이 없었다**
(2026-08-26 감사 B11: `check_line_citations.py`·`check_doc_facts.py`가 "사전등록 제출 전
체크리스트에 아직 등재 안 됨"; 2026-08-28 두 재감사가 `design_reachability.py`에 대해 같은
지적을 반복). 도구가 있는데 아무도 안 부르면 도구가 없는 것과 같다.

이 파일이 그 실행 지점이다.  `--registry`가 지정한 항목만 검사한다 —
디렉터리를 쓸어 담지 않는다.  같은 저장소에서 다른 세션이 동시에 작업할 수 있고,
그쪽의 진행 중 파일을 이 검사가 실패시키면 안 되기 때문이다.

  python3 presubmit.py --registry presubmit_registry.json
  echo $?    # 0 = 제출 가능, 1 = 차단

차단 조건 (하나라도 걸리면 제출 금지):
  * line-citation 드리프트
  * doc-fact 위반
  * 도달가능성이 NOTHING_PURCHASABLE 또는 SINGLE_LABEL_FORCED
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

# design_reachability.py는 2026-08-28에 저장소 루트 `scripts/discipline/`에 만들어졌고
# 다른 세션이 이미 그 경로로 부르고 있다.  경로 정본화는 doc-steward 소관이므로 여기서는
# 옮기지 않고 **두 위치 다 찾는다**.
REACH_CANDIDATES = [
    os.path.join(HERE, "design_reachability.py"),
    os.path.join(ROOT, "scripts", "discipline", "design_reachability.py"),
]

BLOCKING_REACH = {"NOTHING_PURCHASABLE", "SINGLE_LABEL_FORCED"}


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    reg = json.load(open(a.registry))
    blocks, notes = [], []

    # 1) line citations -----------------------------------------------------
    if reg.get("check_line_citations", True):
        # --all = manifest에 이미 있는 모든 문서. 문서를 안 주면 도구가
        # "no documents given"으로 끝나며 이는 통과가 아니라 **미실행**이다
        # (게이트 #21: 측정 실패를 게이트 실패로 라벨링 마라 — 그 거울상).
        rc, out = run([sys.executable, os.path.join(HERE, "check_line_citations.py"),
                       "--check", "--all"])
        (notes if rc == 0 else blocks).append(("line_citations", rc, out.strip().splitlines()[-1:] or [""]))

    # 2) doc facts ----------------------------------------------------------
    if reg.get("check_doc_facts", True):
        rc, out = run([sys.executable, os.path.join(HERE, "check_doc_facts.py")])
        (notes if rc == 0 else blocks).append(("doc_facts", rc, out.strip().splitlines()[-1:] or [""]))

    # 3) citation stops -- staged added lines only (그 도구 자신의 설계 결정)
    if reg.get("check_citation_stops", True):
        rc, out = run([sys.executable, os.path.join(HERE, "check_citation_stops.py")], cwd=ROOT)
        (notes if rc == 0 else blocks).append(
            ("citation_stops", rc, out.strip().splitlines()[-1:] or [""]))

    # 4) design-layer reachability, per REGISTERED spec ---------------------
    reach = next((p for p in REACH_CANDIDATES if os.path.exists(p)), None)
    for spec in reg.get("reachability_specs", []):
        sp = spec if os.path.isabs(spec) else os.path.join(ROOT, spec)
        if reach is None or not os.path.exists(sp):
            blocks.append(("reachability", 2, [f"missing tool or spec: {spec}"]))
            continue
        rc, out = run([sys.executable, reach, sp])
        verdict = "?"
        for ln in out.splitlines():
            if ln.strip().startswith("VERDICT:"):
                verdict = ln.split("VERDICT:")[1].strip()
        item = ("reachability", rc, [f"{os.path.basename(sp)} -> {verdict}"])
        (blocks if verdict in BLOCKING_REACH or rc != 0 else notes).append(item)

    print("=== presubmit ===")
    for name, rc, lines in notes:
        print(f"  OK     {name:14} {' '.join(lines)}")
    for name, rc, lines in blocks:
        print(f"  BLOCK  {name:14} {' '.join(lines)}")
    if blocks:
        print(f"\n제출 금지 — 차단 {len(blocks)}건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.")
        return 1
    print("\n제출 가능 — 규율 검사 통과. ★다른 死因은 이 검사가 보지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
