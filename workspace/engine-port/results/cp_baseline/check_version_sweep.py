#!/usr/bin/env python3
"""CP track version-sweep gate.  Zero GPU.

The 4th audit's central finding was not about rules: **11 of 24 defects were pure
propagation failures** -- a repair was made in one place and the other places that
cite it were not swept.  Its words: "이 11건 중 어느 것도 새로운 사고를 요구하지
않는다 ... 그것은 능력 부족이 아니라 체크리스트 부재다."  And one of them (§8's
"수용 시험 3개" left behind when §2.2 grew to 7) disabled the campaign's ONLY stop
rule -- so propagation failure is a run-killer, not a formatting issue.

This file is that checklist, as code.  Rules are derived from the exact instances
the audit found; each one fails loudly on the state that produced it.

Root cause of two of them, recorded because it is mechanical and repeatable: the
rev3 edits were applied with unasserted `str.replace()`.  Two anchors did not match,
the replacements silently did nothing, and the session reported them as applied.
Rule S5 exists so that a silent no-op cannot survive again.

Usage:  python3 check_version_sweep.py     # rc=0 clean, rc=1 blocked
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREG_CP0_2026-08-28.md")
VIOL = []


def viol(rule, msg):
    VIOL.append((rule, msg))


def read(p):
    return open(p, encoding="utf-8").read()


text = read(PREREG)

# S1 -- the stop rule must bind every registered acceptance test.
#      (audit G8/F5: §2.2 grew 3 -> 7 and §8 still said "3개")
n_tests = len(re.findall(r"^\| \*{0,2}★?\*{0,2}P1-[a-g]", text, re.M))
for m in re.finditer(r"P1 수용 시험 (\d+)개", text):
    if int(m.group(1)) != n_tests:
        viol("S1", f"문서가 'P1 수용 시험 {m.group(1)}개'라 적었으나 §2.2에 등록된 시험은 "
                   f"{n_tests}개다 — 중단 규칙이 신설 시험을 묶지 않는다")

# S2 -- every probe rate named in a spec's priors must be a registered probe rate.
#      (audit G3: grid moved {3,6,12} -> {2,4,8}, spec priors still argued from the old one)
import importlib.util as _il
_s = _il.spec_from_file_location("pred", os.path.join(HERE, "cp0_predicates.py"))
_p = _il.module_from_spec(_s); _s.loader.exec_module(_p)
rates = set(_p.PROBE_RATES)
spec = os.path.join(HERE, "spec_cp0_capacity.json")
if os.path.exists(spec):
    blob = json.dumps(json.load(open(spec, encoding="utf-8")), ensure_ascii=False)
    for r in re.findall(r"(?:rate|under|above)\s+(\d+)\s*(?:req/s)?", blob):
        if int(r) not in rates and int(r) not in (1,):
            viol("S2", f"{os.path.basename(spec)}의 priors가 rate {r}을 논거로 쓰는데 등록 "
                       f"PROBE_RATES는 {sorted(rates)}다 — 폐기된 격자의 논거가 남아 있다")

# S3 -- a canon line citation must be identical everywhere it appears in this track.
cites = {}
for fn in sorted(os.listdir(HERE)):
    if not fn.endswith((".py", ".md")) or fn == os.path.basename(__file__):
        continue
    for m in re.finditer(r"PROJECT_STATUS\.md:(\d+)", read(os.path.join(HERE, fn))):
        cites.setdefault(m.group(1), set()).add(fn)
if len(cites) > 1:
    viol("S3", f"같은 정본 문장에 대한 좌표가 트랙 안에서 갈린다: "
               + " / ".join(f"{k} -> {sorted(v)}" for k, v in sorted(cites.items())))

# S4 -- the revision banner must not declare something 미반영 that the body claims to fix.
banner = text[:text.index("## 0.")] if "## 0." in text else text[:4000]
unapplied = re.search(r"미반영[^\n]*(?:\n>[^\n]*)*", banner)
if unapplied:
    for tag in re.findall(r"\*\*?(F\d|L\d+)\*\*?", unapplied.group(0)):
        if re.search(rf"{tag}[^\n]{{0,40}}(반영|해소|수리|정정)", text[len(banner):]):
            viol("S4", f"이력 배너가 {tag}를 '미반영'으로 선언하는데 본문은 반영했다고 쓴다")

# S5 -- a flag the pre-registration argues for must appear in the flag list it argues about.
for flag, why in (("--disable-piecewise-cuda-graph",
                   "F3(단일 레버) 수리가 §6 부팅 플래그에 실제로 등재됐는가"),):
    if flag not in text:
        viol("S5", f"`{flag}`가 사전등록 어디에도 없다 — {why}. "
                   f"(rev3에서 이 치환이 앵커 불일치로 조용히 무효화됐다)")

# S6 -- audit-round counters must not lag the audits actually filed.
rounds = len([d for d in os.listdir(HERE)
              if d.startswith("audit_") and os.path.isdir(os.path.join(HERE, d))])
# Only FORWARD-LOOKING counters are stale-able.  A backward reference ("2회차 감사
# K6") is a citation and must not move.  The first version of this rule flagged
# both and produced 12 false positives -- recorded because a gate that cries wolf
# gets switched off, which is how the repository lost the last one.
for m in re.finditer(r"(\d)회차 감사(?=\s*(?:대기|를 받|을 받|를 걸|을 걸))", text):
    if int(m.group(1)) <= rounds:
        viol("S6", f"문서가 '{m.group(1)}회차 감사'를 앞으로 받을 것처럼 적었으나 이미 "
                   f"{rounds}건의 판정서가 있다 — 다음은 {rounds+1}회차다")

# S7 -- stage-dependent boot flags must match the stage they are registered for.
#      P1 (instrument validation) must NOT force piecewise off, because P1-g exists
#      to OBSERVE whether cps drives the capture list.  The CP-0 body MUST force it
#      off, because that is what makes cps the only lever.  Forking the P1 script
#      into the body script is exactly how the second one loses the flag
#      (PROJECT_STATUS.md "방법론 게이트" #66).
FLAG = "--disable-piecewise-cuda-graph"
for sb, must_have, why in (("p1_accept.sbatch", False,
                            "P1-g는 arm 간 piecewise 배너 차이를 관측해야 한다"),):
    path = os.path.join(HERE, sb)
    if not os.path.exists(path):
        continue
    has = FLAG in read(path)
    if has != must_have:
        viol("S7", f"{sb}가 `{FLAG}`를 "
                   f"{'넘긴다' if has else '안 넘긴다'} — 등록된 단계 규약과 다르다 ({why})")
    if not must_have and "P1 부팅은 이 플래그를 쓰지 않는다" not in text:
        viol("S7", f"{sb}가 `{FLAG}`를 생략하는 것이 사전등록에 명시돼 있지 않다 — "
                   f"의도된 차이와 누락을 구별할 수 없다")

# S8 -- two measurement definitions that job 899768 proved were unregistered, and
#      whose absence silently decided what P1-e measured.  Registering them is what
#      keeps the next run from failing for the same reason.
for needle, why in (
    ("forward_mode == EXTEND",
     "채널2의 prefill 판정 (extend_num_tokens>0은 DECODE의 stale 값을 통과시킨다: 116건 중 101건)"),
    ("t_start`로 클램프",
     "P1-e phase window가 서로소라는 등록 (초안의 +1.0초가 S를 B에 침범시켰다)"),
):
    if needle not in text:
        viol("S8", f"사전등록에 `{needle}`가 없다 — {why}")

print("=== CP version sweep ===")
for r, m in VIOL:
    print(f"  [{r}] {m}")
print(f"--- {len(VIOL)} violation(s)  " + ("BLOCKED" if VIOL else "OK"))
sys.exit(1 if VIOL else 0)
