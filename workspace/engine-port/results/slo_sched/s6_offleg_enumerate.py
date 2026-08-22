#!/usr/bin/env python3
"""S-6: MECHANICAL enumeration of every site that can reject a telemetry-OFF boot.

WHY THIS EXISTS
---------------
The S-6 preregistration claimed an "exhaustive" (전수) list of such sites three
times and was wrong three times (audit rounds 1 and 2).  The round-2 verdict
handed the next audit this question:

    "§4.1이 기계 산출인가 네 번째 손 열거인가 -- 손 열거면 그 자체로 NO-GO."

So the list is no longer written by hand.  This script derives it from the
harness source, is re-runnable, and carries a self-test (including mutation
tests) proving the derivation is load-bearing rather than vacuous.

WHAT IT IS NOT
--------------
* Not a proof of completeness.  It is complete *with respect to the registered
  token set* (TOKENS below), which is itself a free parameter and is declared
  as one in the preregistration.  A site that rejects an OFF boot without
  mentioning any registered token is invisible here.  That limit is the reason
  `--selftest` includes a planted-site check: it demonstrates the scanner sees
  a NEW gating site, but it cannot demonstrate the token set is exhaustive.
* Not a judgment.  It classifies and reports; it does not decide whether a site
  must be forked, branched, or left alone.

Usage:
    python3 s6_offleg_enumerate.py --selftest      # no repo scan needed
    python3 s6_offleg_enumerate.py --emit          # JSON + markdown to stdout
    python3 s6_offleg_enumerate.py --emit --out DIR
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --- REGISTERED INPUTS (free parameter #1: which files are scanned) ---------
TARGETS = ["g16_grid.sbatch", "g16_analyze.py", "g16_assert.py",
           "g16_n3_dryrun_check.py"]

# --- REGISTERED TOKEN SET (free parameter #2) -------------------------------
# A site is a candidate iff its line contains one of these.
TOKENS = [
    "TELEM_RC", "telem_rc", "PDMUX_TELEMETRY_PATH", "telemetry_rc",
    "MIN_SNAPSHOTS", "runtime_snapshot", "telemetry", "ARTIFACT_OK",
    "BOOTS_FAILED", "assert_runtime_snapshots",
]

# --- REGISTERED EXTRA TRIGGERS (free parameter #3) --------------------------
# A line with NO telemetry token can still block the campaign: `G16_SMOKE_OVERALL`
# is a conjunction over EVERY smoke item, so an item that is unreachable for a
# non-telemetry reason (e.g. pinned to an arm the campaign does not run) blocks
# submission just as hard.  Round-2 audit finding D5 is exactly this class, and
# a token-only scan is blind to it.
EXTRA_TRIGGERS = [
    re.compile(r"\bSMOKE\d+\w*\s*="),      # any smoke-item assignment
    re.compile(r"SMOKE_OVERALL"),           # the conjunction itself
    re.compile(r'\bcase\s+"\$RUNID"'),      # run-id literal gating
]

# --- CLASSIFIERS (registered; order matters, first match wins) --------------
CLASSIFIERS = [
    ("SMOKE_OVERALL", re.compile(r"SMOKE_OVERALL|SMOKE\d+\w*\"?\s*=\s*PASS\s*\]\s*&&")),
    ("SMOKE_ITEM",    re.compile(r"\bSMOKE\d+\w*=")),
    ("GATE_REJECT",   re.compile(r"ARTIFACT_OK=0|BOOTS_FAILED=|BOOTS_FAILED\+|mark_failed_artifacts")),
    ("ADOPT",         re.compile(r"adopted\s*=|telem_rc\"?\)\s*==|rejected\.append")),
    ("EXPORT",        re.compile(r"^\s*export\s|G16_TELEM_RC=|^\s*TELEM_RC=")),
    ("ASSERT_FN",     re.compile(r"def assert_runtime_snapshots|assert_runtime_snapshots\(")),
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ===========================================================================
# Shell guard-chain tracking
# ===========================================================================
_IF = re.compile(r"^(\s*)(?:el)?if\s+(.*?);\s*then\s*$")
_FI = re.compile(r"^(\s*)fi\b")
_CASE = re.compile(r"^(\s*)case\s+(.*?)\s+in\s*$")
_ESAC = re.compile(r"^(\s*)esac\b")
_ARM_LIT = re.compile(r'"\$ARM"\s*=\s*"(d\d+)"')
_RUNID_LIT = re.compile(r'case\s+"\$RUNID"\s+in\s+([^)]*)\)')


def guard_chains(lines):
    """-> list parallel to `lines`, each = list of enclosing conditions."""
    stack, out = [], []
    for ln in lines:
        m = _FI.match(ln) or _ESAC.match(ln)
        if m:
            ind = len(m.group(1))
            while stack and stack[-1][0] >= ind:
                stack.pop()
            out.append([c for _, c in stack])
            continue
        out.append([c for _, c in stack])
        m = _IF.match(ln)
        if m:
            stack.append((len(m.group(1)), m.group(2).strip()))
            continue
        m = _CASE.match(ln)
        if m:
            stack.append((len(m.group(1)), f"case {m.group(2).strip()}"))
    return out


def arm_pins(chain, line):
    """Arm literals this site is conditioned on (D5's question), mechanically."""
    pins = set()
    for c in chain:
        pins.update(_ARM_LIT.findall(c))
    pins.update(_ARM_LIT.findall(line))
    m = _RUNID_LIT.search(line)
    if m:
        pins.update(re.findall(r"d\d+", m.group(1)))
    return sorted(pins)


def classify(line):
    for name, rx in CLASSIFIERS:
        if rx.search(line):
            return name
    return "REF"


def scan_text(name, text):
    lines = text.splitlines()
    chains = guard_chains(lines) if name.endswith(".sbatch") else [[]] * len(lines)
    hits = []
    for i, ln in enumerate(lines, 1):
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not (any(t in ln for t in TOKENS)
                or any(rx.search(ln) for rx in EXTRA_TRIGGERS)):
            continue
        hits.append({
            "file": name, "line": i, "kind": classify(ln),
            "arm_pinned_to": arm_pins(chains[i - 1], ln),
            "guards": chains[i - 1][-3:],
            "text": stripped[:150],
        })
    return hits


def scan(files):
    out = []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(HERE, f)
        if not os.path.exists(p):
            out.append({"file": os.path.basename(f), "line": 0, "kind": "MISSING",
                        "arm_pinned_to": [], "guards": [], "text": "FILE NOT FOUND"})
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            out.extend(scan_text(os.path.basename(f), fh.read()))
    return out


# ===========================================================================
# SMOKE reachability -- the D5 question, answered mechanically
# ===========================================================================
def smoke_reachability(hits, arms_in_campaign):
    """Which SMOKE items can never be set PASS when the campaign runs `arms`."""
    items = {}
    for h in hits:
        if h["kind"] != "SMOKE_ITEM":
            continue
        for var in re.findall(r"\b(SMOKE\d+\w*)=", h["text"]):
            rec = items.setdefault(var, {"sites": [], "arm_pins": set(),
                                         "texts": []})
            rec["sites"].append(f'{h["file"]}:{h["line"]}')
            rec["arm_pins"].update(h["arm_pinned_to"])
            rec["texts"].append(h["text"])
    # direct reachability: is any assignment site on an arm the campaign runs?
    direct = {}
    for var, rec in items.items():
        pins = sorted(rec["arm_pins"])
        direct[var] = (not pins) or bool(set(pins) & set(arms_in_campaign))

    # ★ TRANSITIVE pass. `SMOKE9=PASS` is guarded by `SMOKE9A`/`SMOKE9B`, which
    #   are themselves d74-pinned: SMOKE9's assignment site runs, but it can only
    #   ever take the FAIL branch. A per-site arm check alone reports that item
    #   as "reachable" and understates the damage -- the round-2 D5 finding is
    #   exactly of this shape, so the tool must not reproduce the blind spot.
    for var, rec in items.items():
        deps = set()
        for h in rec["texts"]:
            # `echo "... 1=$SMOKE1 2=$SMOKE2 ..."` mentions every item but
            # creates no dependency. Excluding pure echo lines keeps the
            # dependency graph from being uselessly complete. What remains is
            # still an OVER-approximation (any SMOKE token on a non-echo line
            # counts as a dependency) -- deliberately so: over-approximating
            # dependencies can only mark MORE items unsatisfiable, which is the
            # conservative direction for a gate whose output is "the smoke
            # block must be rewritten".
            if h.lstrip().startswith("echo "):
                continue
            deps.update(v for v in re.findall(r"\b(SMOKE\d+\w*)\b", h)
                        if v != var)
        rec["deps"] = sorted(deps)
    changed = True
    satisfiable = dict(direct)
    while changed:
        changed = False
        for var, rec in items.items():
            if satisfiable[var] and any(not satisfiable.get(d, True)
                                        for d in rec["deps"]):
                satisfiable[var] = False
                changed = True

    rows = []
    for var, rec in sorted(items.items()):
        rows.append({"smoke_item": var, "arm_pins": sorted(rec["arm_pins"]),
                     "sites": rec["sites"], "depends_on": rec["deps"],
                     "site_runs": direct[var],
                     "can_ever_pass": satisfiable[var]})
    return rows


def build(files, arms):
    hits = scan(files)
    return {
        "kind": "s6_offleg_enumeration",
        "targets": [os.path.basename(f) for f in files],
        "sha256": {os.path.basename(f): (_sha256(os.path.join(HERE, f))
                                         if os.path.exists(os.path.join(HERE, f)) else None)
                   for f in files},
        "registered_tokens": TOKENS,
        "registered_extra_triggers": [rx.pattern for rx in EXTRA_TRIGGERS],
        "registered_classifiers": [c[0] for c in CLASSIFIERS],
        "campaign_arms": arms,
        "n_sites": len(hits),
        "sites": hits,
        "smoke_reachability": smoke_reachability(hits, arms),
        "limits": ("Complete only w.r.t. `registered_tokens`. A rejection site "
                   "that mentions none of them is invisible. This is a scanner, "
                   "not a proof."),
    }


def markdown(rep):
    L = [f'| # | file:line | kind | arm-pinned | text |', "|---|---|---|---|---|"]
    for i, h in enumerate(rep["sites"], 1):
        L.append(f'| {i} | `{h["file"]}:{h["line"]}` | {h["kind"]} | '
                 f'{",".join(h["arm_pinned_to"]) or "-"} | `{h["text"][:90]}` |')
    L += ["", f'SMOKE satisfiability with arms={rep["campaign_arms"]}:', "",
          "| item | pinned to | depends on | site runs | can ever PASS |",
          "|---|---|---|---|---|"]
    for r in rep["smoke_reachability"]:
        L.append(f'| `{r["smoke_item"]}` | {",".join(r["arm_pins"]) or "-"} | '
                 f'{",".join(r["depends_on"]) or "-"} | '
                 f'{"yes" if r["site_runs"] else "no"} | '
                 f'{"YES" if r["can_ever_pass"] else "**NO**"} |')
    return "\n".join(L)


# ===========================================================================
# Self-test -- including mutation tests (methodology lesson #53)
# ===========================================================================
SYNTH = '''\
ARM="d44"
if [ "$SMOKE" = "1" ] && [ "$ARM" = "d74" ]; then
  [ "$TELEM_RC" -eq 0 ] && SMOKE1=PASS || SMOKE1=FAIL
fi
if [ "$SMOKE" = "1" ]; then
  [ "$PIN_RC" -eq 0 ] && SMOKE2=PASS || SMOKE2=FAIL
fi
ARTIFACT_OK=1
[ "$TELEM_RC" -eq 0 ] || ARTIFACT_OK=0
if [ "$ARTIFACT_OK" != 1 ]; then
  BOOTS_FAILED=$((BOOTS_FAILED + 1))
fi
'''


def selftest():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        ok = ok and cond

    print("-- scanner on a synthetic harness")
    hits = scan_text("synth.sbatch", SYNTH)
    kinds = [h["kind"] for h in hits]
    chk("SMOKE_ITEM" in kinds, "finds SMOKE item assignments")
    chk("GATE_REJECT" in kinds, "finds the ARTIFACT_OK/BOOTS_FAILED rejection path")
    s1 = [h for h in hits if "SMOKE1=" in h["text"]][0]
    chk(s1["arm_pinned_to"] == ["d74"], "recovers the enclosing ARM pin (d74) for SMOKE1")
    s2 = [h for h in hits if "SMOKE2=" in h["text"]]
    chk(s2 and s2[0]["arm_pinned_to"] == [], "SMOKE2 is correctly NOT arm-pinned")

    print("-- SMOKE reachability logic")
    rows = {r["smoke_item"]: r for r in smoke_reachability(hits, ["d44"])}
    chk(rows["SMOKE1"]["can_ever_pass"] is False, "SMOKE1 unreachable when only d44 runs")
    chk(rows["SMOKE2"]["can_ever_pass"] is True, "SMOKE2 reachable when only d44 runs")
    rows74 = {r["smoke_item"]: r for r in smoke_reachability(hits, ["d44", "d74"])}
    chk(rows74["SMOKE1"]["can_ever_pass"] is True, "SMOKE1 reachable once d74 is in the arm set")

    print("-- TRANSITIVE satisfiability (the tool must not repeat D5's blind spot)")
    trans = SYNTH + '\nif [ "$SMOKE1" = PASS ]; then SMOKE99=PASS; else SMOKE99=FAIL; fi\n'
    th = scan_text("synth.sbatch", trans)
    tr = {r["smoke_item"]: r for r in smoke_reachability(th, ["d44"])}
    chk(tr["SMOKE99"]["site_runs"] is True,
        "SMOKE99's assignment site DOES run on d44 (not arm-pinned)")
    chk(tr["SMOKE99"]["can_ever_pass"] is False,
        "...but it can never PASS, because it depends on d74-pinned SMOKE1")

    print("-- guard-chain stack pops correctly")
    chains = guard_chains(SYNTH.splitlines())
    last = chains[-1]
    chk(all("d74" not in c for c in last), "d74 guard does not leak past its `fi`")

    print("-- PLANTED SITE (scanner sees a NEW gating site, not just known ones)")
    planted = SYNTH + '\n[ "$telemetry_rc" -eq 0 ] || ARTIFACT_OK=0\n'
    ph = scan_text("synth.sbatch", planted)
    chk(len(ph) == len(hits) + 1, "planted rejection site is found (+1 site)")

    print("-- MUTATION: the token set is load-bearing")
    # A line whose ONLY trigger is the token under test, so the mutation is
    # decisive rather than masked by EXTRA_TRIGGERS.
    # NB: "telemetry_rc" would NOT work here -- it contains the registered
    # token "telemetry" as a substring, so removing it leaves the line matched.
    # That near-miss is itself why this mutation is worth having.
    sole = SYNTH + '\n[ "$MIN_SNAPSHOTS" -gt 0 ] || echo sole_trigger_line\n'
    base_n = len(scan_text("synth.sbatch", sole))
    saved = list(TOKENS)
    try:
        TOKENS.remove("MIN_SNAPSHOTS")
        mut = scan_text("synth.sbatch", sole)
        chk(len(mut) == base_n - 1,
            "dropping a registered token LOSES exactly the line it alone "
            "triggered (list is not vacuous)")
    finally:
        TOKENS[:] = saved
    chk(len(scan_text("synth.sbatch", sole)) == base_n, "token set restored")

    print("-- MUTATION: EXTRA_TRIGGERS are load-bearing (D5 blindness check)")
    saved_x = list(EXTRA_TRIGGERS)
    try:
        EXTRA_TRIGGERS[:] = []
        blind = scan_text("synth.sbatch", SYNTH)
        chk(not any("SMOKE2=" in h["text"] for h in blind),
            "without EXTRA_TRIGGERS the non-telemetry SMOKE2 becomes INVISIBLE "
            "(this is the round-2 D5 blind spot, reproduced mechanically)")
    finally:
        EXTRA_TRIGGERS[:] = saved_x
    chk(any("SMOKE2=" in h["text"] for h in scan_text("synth.sbatch", SYNTH)),
        "EXTRA_TRIGGERS restored")

    print("-- MUTATION: the arm-pin extractor is load-bearing")
    unpinned = SYNTH.replace(' && [ "$ARM" = "d74" ]', "")
    up = scan_text("synth.sbatch", unpinned)
    u1 = [h for h in up if "SMOKE1=" in h["text"]][0]
    chk(u1["arm_pinned_to"] == [],
        "removing the ARM guard removes the pin (extractor reads the source, "
        "not a hardcoded table)")

    print("ALL PASS" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--arms", default="d44")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.emit:
        rep = build(TARGETS, a.arms.split(","))
        md = markdown(rep)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "S6_OFFLEG_SITES.json"), "w") as f:
                json.dump(rep, f, indent=2)
            with open(os.path.join(a.out, "S6_OFFLEG_SITES.md"), "w") as f:
                f.write(md + "\n")
            print(f"[emit] -> {a.out}")
        print(md)
        print(f"\n[n_sites={rep['n_sites']}]")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
