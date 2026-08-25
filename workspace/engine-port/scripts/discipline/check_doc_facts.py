#!/usr/bin/env python3
"""문서가 **자기 아티팩트에 대해 주장하는 수치**가 실제와 맞는지 검사한다.

왜 이 도구가 있는가
-------------------
A1 규칙층 감사 3회차가 이 사슬의 실패 형태에 이름을 붙였다:

    ★"수리는 국소, 주장은 전역" (repair-local truth vs document-global claims)
    — 각 수리는 감사자가 지목한 정확히 그 좌표에서 실재하고 검증 가능하다.
      그런데 그 수리가 자기 판본의 **다른 문장·다른 표·다른 수치**에 만든 파급을
      재도출하지 않기 때문에, 다음 회차 결함 목록의 대부분은 **직전 회차 수리의
      그림자**다.

같은 가족이 **세 번** 재발했다(1회차 B4 → 2회차 P3/P13–P20 → 3회차 R2):
문서가 "정합 세계 152,361"이라 적는데 코드는 166,194를 내고, "검사 3개"라 적는데
4개이고, "회귀 155"라 적는데 166이다. 매번 산문으로 *"수리 커밋마다 재검증하라"* 를
등재했고 매번 실패했다. **산문 규율이 세 번 실패했으면 그것은 기계로 옮길 일이다** —
`check_line_citations.py`가 인용에 대해 한 것과 같은 이동.

무엇을 하는가 / 못 하는가
-------------------------
등록된 사실마다 (a) **진리원**(아티팩트에서 읽는다)과 (b) **문서 안의 패턴**을 둔다.
패턴이 잡은 값이 진리원과 다르면 위반이다.  ★한 문서 안에서 같은 사실이 여러 번
나오면 **전부** 검사한다 — R2의 실질은 "한 곳은 고치고 다른 곳은 안 고쳤다"였다.

★**정직한 한계**: 이 도구는 **등록된 사실만** 본다.  등록되지 않은 주장은 검사되지
않으며, 레지스트리를 안 늘리면 도구가 커버리지를 **과장**하게 된다.  그래서
`--list`가 무엇이 등록됐는지 항상 출력하고, 위반 0일 때도 **등록 건수를 함께** 적는다.

사용:
    python3 check_doc_facts.py            # 전 사실 검사
    python3 check_doc_facts.py --list     # 무엇이 등록됐는지
종료: 0 통과, 1 위반, 2 진리원 접근 실패.
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.abspath(os.path.join(HERE, "..", ".."))


def _json(rel):
    with open(os.path.join(TRACK, rel)) as fh:
        return json.load(fh)


def _primary(key):
    return lambda: _json("results/kernel_mech/a1/selftest_a1_primary_2026-08-25.json")[key]


def _q3(key):
    return lambda: _json("results/kernel_mech/a1/selftest_a1_q3k1_2026-08-25.json")[key]


def _count_json_key(rel, key):
    return lambda: len(_json(rel)[key])


def _module_len(rel, name):
    """len() of a module-level list/dict, read by import."""
    def go():
        import importlib.util
        path = os.path.join(TRACK, rel)
        spec = importlib.util.spec_from_file_location("_m", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_m"] = m
        spec.loader.exec_module(m)
        return len(getattr(m, name))
    return go


def _manifest_entries():
    def go():
        man = _json("scripts/discipline/line_citations.json")
        return sum(len(v) for v in man.values())
    return go


def _regression_tests():
    """The CPU regression count, counted STATICALLY from the test files.

    ★The first version ran `unittest discover` as a subprocess and reported 91
    where the venv reports 166: several suites import the installed sglang tree
    and are skipped when it is absent.  A source of truth that changes with the
    caller's environment is not a source of truth -- it would have made this
    tool report a violation against a correct document.  Counting `def test_`
    is environment-independent and is what the document's number means.
    """
    def go():
        n = 0
        tdir = os.path.join(TRACK, "tests")
        for fn in sorted(os.listdir(tdir)):
            if not (fn.startswith("test_") and fn.endswith(".py")):
                continue
            with open(os.path.join(tdir, fn), errors="replace") as fh:
                n += len(re.findall(r"^\s+def test_", fh.read(), re.M))
        return n
    return go


DESIGN = "results/kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md"

# name -> (source-of-truth, document, regex with ONE capturing group)
# ★The regex must capture the number as the document writes it (commas kept).
FACTS = {
    "primary worlds enumerated": (
        _primary("worlds_enumerated"), DESIGN, r"세계 \*\*([\d,]+)\*\* 열거"),
    "primary worlds consistent": (
        _primary("worlds_consistent"), DESIGN, r"정합 \*\*([\d,]+)\*\*"),
    "primary labels": (
        _count_json_key("results/kernel_mech/a1/selftest_a1_primary_2026-08-25.json",
                        "labels"), DESIGN, r"라벨 \*\*(\d+)개\*\* · mutant"),
    "primary mutants": (
        _module_len("results/kernel_mech/a1/a1_primary_rule.py", "MUTANTS"),
        DESIGN, r"mutant \*\*(\d+)종\*\* · 검사 \*\*6개\*\*"),
    "q3 worlds": (_q3("q3_worlds"), DESIGN, r"Q3 \*\*([\d,]+) 세계\*\*"),
    "q3 mutants": (
        _module_len("results/kernel_mech/a1/a1_q3k1_rule.py", "Q3_MUTANTS"),
        DESIGN, r"mutant \*\*(\d+)종\*\* 전부 load-bearing"),
    "regression tests": (_regression_tests(), DESIGN, r"회귀 \*\*(\d+)\*\* PASS"),
    "citation manifest entries": (
        _manifest_entries(), DESIGN, r"매니페스트에 등재된 (\d+)건"),
}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for n, (_src, doc, pat) in sorted(FACTS.items()):
            print(f"  {n:32s} {os.path.basename(doc):48s} /{pat}/")
        print(f"--- {len(FACTS)} fact(s) registered.  ★Unregistered claims are "
              f"NOT checked; growing this list is the only way coverage grows.")
        return 0

    violations, checked = [], 0
    for name, (src, doc, pat) in sorted(FACTS.items()):
        try:
            truth = src()
        except Exception as exc:                      # noqa: BLE001
            print(f"  [ERR ] {name}: source of truth unreadable: {exc}")
            return 2
        path = os.path.join(TRACK, doc)
        with open(path, errors="replace") as fh:
            text = fh.read()
        hits = re.findall(pat, text)
        if not hits:
            violations.append(f"{name}: pattern never matches in "
                              f"{os.path.basename(doc)} -- the claim moved or "
                              f"was reworded; re-register it")
            continue
        for h in hits:                                # ★every occurrence
            checked += 1
            if int(h.replace(",", "")) != int(truth):
                violations.append(
                    f"{name}: document says {h}, artefact says {truth:,} "
                    f"({os.path.basename(doc)})")
    for v in violations:
        print(f"  [FAIL] {v}")
    print(f"--- {len(FACTS)} fact(s), {checked} occurrence(s) compared, "
          f"{len(violations)} violation(s)  {'OK' if not violations else 'FAIL'}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
