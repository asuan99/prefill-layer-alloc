#!/usr/bin/env python3
"""AF-1 self-test.  GPU 0.  `--freeze` regenerates ONLY the frozen oracle section.

Each check names the failure it exists for.  Checks marked (1st) were added after
`audit_af1_rules_2026-09-04/VERDICT.md` found the rev1 version of this file
passing mutations it claimed to stop.

  C1   constants partition into LABEL / DESIGN / INFRA, nothing unclassified
  C2a  every LABEL constant is load-bearing under VALUE SUBSTITUTION
       -> 6th CP-2 audit F6: `perturb()` truncated tuples, proving only that a
          ladder had 36 points and never that it had THOSE points.
  C2b  every LABEL constant is READ by some registered fold's source
       -> gate 87.  Independent of which substitution values were registered,
          which C2a cannot be (5th-audit L15: the mutation multiplier is itself
          an unregistered free parameter).
  C3   every DESIGN constant moves nothing, in BOTH directions (6th-audit L3)
  C4   every registered fold exists, is on PREDICATE_FOLDS, and passes its vectors
       -> 6th-audit F8-2 / gate 87.
  C5   the enumeration matches the frozen oracle, and the hand rows match the rule
  C6   meta pins the hand section's exact KEY SET, the substantive list and count,
       the pre-registration's anchor lines EXACTLY (not >=), and the document's own
       counts -- prohibitions, the three ledgers' rows, the limits
       -> (1st) A6/D10 deleted a label and a duplicate guard row with a count
          decrement and passed; a count cannot see a substitution.
  C7   every hand row's anchor is PAIRED -- it begins with that row's own world
       key and label -- and appears verbatim in the pre-registration; `why` names
       its own label; every registered order-independence reason is in the document
       -> (1st) E7/E8 swapped two outcomes with a two-file edit and never touched
          the document, because anchors were checked for set membership only.
  C8   every stop-rule branch is a hand world and gets the label the prose claims
       -> 6th-audit F3.
  C9   removing any incoherence clause moves at least one label
  C10  transposing any adjacent guard pair moves a label, OR the pair is
       registered as order-independent with a reason the document states
  C10b every guard moves a label when DELETED (so C10's exemption cannot hide an
       inert guard)
  C11  OUTCOME's key set EQUALS the screen's exact image, both directions; every
       substantive label is reachable; FORK is total, exhaustive, non-constant,
       drawn from the registered branch set, and its CONTENT is pinned row by row
       -> (1st) D2 registered a pair the screen cannot produce and the
          reachability check enumerated the declared axis product -- an identity;
          D3 changed a FORK value and nothing failed.
  C12  both screen strata are among the reported strata
  C13  no coherent world reaches an unregistered outcome
  C14  cross-file provenance: the ladder equals CP-2 rev2's tuples as parsed from
       its file, and every candidate coordinate appears in the survey document
       -> (1st) D6: shifting all 36 ladder cells by 1 ms passed, and prose cannot
          catch that.
  C15  the chain consequence is disclosed: if the candidates are totally ordered
       then `unique` can only ever be the maximum, and the document must say so
       -> (1st) D2: rev1 presented the survivor's identity as a finding.
  C16  PLANNED_BOOTS exceeds MIN_BOOTS, so a lost boot does not trip the stop rule
  C17  the memoised image is never stale under any registered substitution
  C18  no duplicate top-level definition in either module
       -> (3rd) J2: the previous revision's implementation was still in the file
          above the new one, one deletion from reverting the repair.
  C19  the ledgers' enforcement column uses a closed vocabulary, every check ID it
       cites EXISTS, and no inert check is cited as enforcement
       -> (3rd) J7, (4th) K4: rev4's version was a prefix test plus a hard-coded
          grep, so an inert citation and a ghost citation both passed.
  C22  each module's change history names its own current revision
       -> (6th) 등재 권고 1: a header stayed at rev2 for four revisions while a
          ledger row said otherwise.
  C21  the rule's predicate-constant list IS the predicate module's list, not a
       re-typed copy -> (4th) K7 + mutation MPREDCONSTDROP.
  C20b the token scan is non-vacuous for lowercase identifiers, so the widening
       cannot be silently reverted -> (5th) mutation MC20NARROW.
  C20c no unused entry sits in the registered external-identifier escape hatch
       -> (5th) mutation MEXTIDGHOST: an unused slot launders any ghost.
  C20  every label-shaped token in the prohibitions and the limits names something
       that exists -> (4th) K6: counting items protected a prohibition that
       covered a label the revision had already deleted.
       -> (2nd) L3: the cache key is a hand-maintained dependency list, and
          dropping a constant from it passed every check.

What this self-test CANNOT do
-----------------------------
A consistent edit across af1_rule.py, af1_predicates.py, af1_expected_labels.json
AND the pre-registration still passes.  C7's paired anchors force the fourth file
into that set for any change of DIRECTION; they do not force it for a change of
prose.  And no test can grep for a condition that was never registered -- 6th-audit
F4 was a REGRESSION found by grepping rev1 for clauses missing from rev2.  The
pre-registration's sec 0.1 ledger is that check, and it is a human one.
"""
import argparse, collections, inspect, itertools, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ORACLE = HERE / "af1_expected_labels.json"
CP2R2 = HERE / "cp2r2_predicates.py"
SURVEY = HERE / ".." / ".." / ".." / ".." / "reports" / "serving_slo_survey.md"

import af1_predicates as P
import af1_rule as R

FAIL = []


def ck(ok, name, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name
          + (("  -- " + detail) if (detail and not ok) else ""))
    if not ok:
        FAIL.append(name)
    return ok


def worlds():
    for c in itertools.product(*[R.AXES[k] for k in R.AXIS_ORDER]):
        yield dict(zip(R.AXIS_ORDER, c))


def key(w):
    return "|".join(w[k] for k in R.AXIS_ORDER)


def one_label(w, **kw):
    """`OUTCOME_GAP` is a label here: a coherent world the rule refuses to name."""
    try:
        return R.label(w, **kw)
    except R.OutcomeGap:
        return "OUTCOME_GAP"


class substituted:
    """Hold predicate constants at substituted VALUES for a whole check.

    ★The rev1 draft restored the constant inside `label_map` and then called
    `run_folds` afterwards, so the fold half of C2a always ran on the ORIGINAL
    constants and could not fail -- the "gate that is an identity" shape
    (`CONSENSUS.md` sec 3 item 9) inside the checker itself.
    """

    def __init__(self, **over):
        self.over = over

    def __enter__(self):
        self.old = {k: getattr(P, k) for k in self.over}
        for k, v in self.over.items():
            setattr(P, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(P, k, v)
        return False


def label_map(**over):
    with substituted(**over):
        return {key(w): one_label(w) for w in worlds()}


def _fkey(x):
    """JSON object keys are strings; stratum tables are keyed by float."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            try:
                k2 = float(k)
            except (TypeError, ValueError):
                k2 = k
            out[k2] = _fkey(v)
        return out
    if isinstance(x, list):
        return [_fkey(v) for v in x]
    return x


def _norm(x):
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return sorted((_norm(v) for v in x), key=repr)
    return x


def run_folds(oracle):
    ok = bad = 0
    first = ""
    for fname, vectors in oracle["hand_authored"]["folds"].items():
        fn = getattr(P, fname, None)
        if fn is None:
            bad += len(vectors)
            first = first or ("missing fold %s" % fname)
            continue
        for args, expect in vectors:
            try:
                got = fn(*_fkey(args))
            except Exception as e:                      # a fold must not raise
                bad += 1
                first = first or f"{fname} raised {type(e).__name__}: {e}"
                continue
            if isinstance(expect, dict) and "__truthy__" in expect:
                good = bool(got) == expect["__truthy__"]
            elif isinstance(got, float) and isinstance(expect, (int, float)):
                import math as _m
                good = (got == float(expect)) or (
                    _m.isfinite(got) and _m.isfinite(float(expect))
                    and abs(got - float(expect)) < 1e-9)
            else:
                good = _norm(got) == _norm(expect)
            if good:
                ok += 1
            else:
                bad += 1
                if not first:
                    first = f"{fname}{str(args)[:60]} -> {got!r} != {expect!r}"
    return ok, bad, first


# Registered value substitutions: REPLACEMENT VALUES of the same type, never
# truncations, with both directions for ordered constants.
SUBST = {
    "SURVEY_POINTS": [
        tuple(x for x in P.SURVEY_POINTS if x[0] != "batch_async"),
        tuple((n, t * 10.0, i * 10.0) for n, t, i in P.SURVEY_POINTS),
        # ★Cell-level, chain-preserving: the two mutations 2nd-audit E4 survived.
        # They move no label -- C14 is what must catch them, and C2a's criterion
        # is "a label OR a registered check moves".
        tuple((n, t, 50.0 if n == "voice" else i) for n, t, i in P.SURVEY_POINTS),
        tuple((n, 200.0 if n == "chat" else t, i) for n, t, i in P.SURVEY_POINTS),
    ],
    "LADDER_TTFT_MS": [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                       (50.0, 60.0, 70.0, 80.0, 90.0, 100.0)],
    "LADDER_ITL_MS": [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                      (300.0, 310.0, 320.0, 330.0, 340.0, 350.0)],
    # ★+1 ms on every cell: 1st-audit D6's mutation, which rev1 survived.
    "LADDER_TTFT_MS_SHIFT": [tuple(v + 1.0 for v in P.LADDER_TTFT_MS)],
    "SCREEN_STRATUM_Q": [0.99, 0.10],
    "SCREEN_STRATUM_Q_ALT": [0.90, 0.10],
    "UNCONTENDED_STAT_Q": [0.50, 1.00],
    "BAND_SD_MULT": [0.0, 20.0],
    "MARGIN_MIN_BAND_MULT": [1.0, 5.0],
    "STRATUM_TOKENS": [{0.10: 1.0, 0.50: 2.0, 0.90: 3.0, 0.99: 4.0},
                       {0.10: 3000.0, 0.50: 4000.0, 0.90: 5000.0, 0.99: 9000.0}],
    "PARAM_COUNT": [1.0e9, 7.0e10],
    "PEAK_BF16_TFLOPS": [156.0, 989.0],
    "HBM_BW_GBPS": [1555.0, 3350.0],
    "SM_TOTAL": [80, 132],
    "MIN_BOOTS": [3, 9],
    "ARMS": [("plain",), ("plain", "cp2048", "d44", "cp512")],
    "AGG_WITHIN_BOOT": ["mean", "max"],
    "AGG_ACROSS_BOOTS": ["median", "max"],
}
DESIGN_SUBST = {
    "REPORT_STRATUM_QS": [(0.10, 0.50, 0.90, 0.99, 0.995), (0.50, 0.90, 0.99)],
    "REPEATS_PER_STRATUM": [1, 99],
    "WARMUP_REQUESTS_PER_STRATUM": [0, 9],
    "PLANNED_BOOTS": [4, 12],
    "ARM_ORDER_BY_JOB": [(("plain", "cp2048", "d44"),) * 6,
                         tuple(reversed(P.ARM_ORDER_BY_JOB))],
    "MAX_NEW_TOKENS": [32, 256],
    # ★2nd-audit L1: rev2 registered the SAME value as the substitution, so C3 was
    # an identity for these two.
    "TEMPERATURE": [0.7, 1.0],
    # ★3rd-audit J7/N5: rev3's second value equalled the registered one, so one
    # direction of C3 was an identity.  A boolean has one other value, so it gets
    # one substitution and C3b below proves no registered substitution is a no-op.
    "IGNORE_EOS": [False],
    "BOOT_SEEDS": [(1, 2, 3, 4, 5), (2, 3, 5, 7, 11, 13)],
}


def _section(raw, header):
    """The text of one pre-registration section, header to next same-or-higher.

    Returns "" when the section is absent, so a DELETED section becomes a named
    check failure instead of a ValueError traceback (3rd-audit N7: blocking is
    not the same as reporting).
    """
    if header not in raw:
        return ""
    i = raw.index(header)
    depth = len(header.split()[0])
    # ★search from the END of this header's own line: starting at i+1 let the
    # section's own "## " match at offset 0 and returned a one-character slice.
    start = raw.index("\n", i) + 1
    j = len(raw)
    for m in re.finditer(r"(?m)^#{1,6} ", raw[start:]):
        if len(m.group(0).split()[0]) <= depth:
            j = start + m.start()
            break
    return raw[i:j]


def _table_rows(raw, header):
    """Markdown table body rows inside a section (header and ruler excluded)."""
    body = _section(raw, header)
    return sum(1 for ln in body.splitlines()
               if ln.startswith("| ") and not ln.startswith("|---")
               and not re.match(r"^\|\s*(조건|死因|#)\s*\|", ln))


def _read_names(fn):
    """Identifiers a function's CODE reads -- docstrings and comments excluded."""
    import ast as _ast, textwrap as _tw
    tree = _ast.parse(_tw.dedent(inspect.getsource(fn)))
    return {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}


def provenance_violations():
    """How many registered constants fail their CROSS-FILE provenance check.

    Used by C2a so that a constant which is load-bearing for PROVENANCE rather
    than for a label still counts as load-bearing -- the cell-level survey
    mutations move no label by construction (2nd-audit E4).
    """
    n = 0
    if CP2R2.exists():
        src = CP2R2.read_text()
        for nm in ("LADDER_TTFT_MS", "LADDER_ITL_MS"):
            m = re.search(nm + r"\s*=\s*\(([^)]*)\)", src)
            theirs = tuple(float(x) for x in m.group(1).split(",") if x.strip()) if m else ()
            if theirs != getattr(P, nm):
                n += 1
    sp = SURVEY.resolve()
    if sp.exists():
        pairs = set()
        for ln in sp.read_text().replace(",", "").splitlines():
            cells = [c.strip() for c in ln.split("|")]
            if len(cells) < 5:
                continue
            m1 = re.fullmatch(r"\**(\d+)ms\**", cells[2])
            m2 = re.fullmatch(r"\**(\d+)ms\**", cells[3])
            if m1 and m2:
                pairs.add((float(m1.group(1)), float(m2.group(1))))
        n += len({(t, i) for _n, t, i in P.SURVEY_POINTS} - pairs)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()

    oracle = json.load(open(ORACLE), object_pairs_hook=collections.OrderedDict)
    base = label_map()

    if a.freeze:
        before = json.dumps([oracle["hand_authored"], oracle["meta"],
                             oracle["_policy"]], sort_keys=True)
        oracle["frozen"] = collections.OrderedDict(sorted(base.items()))
        after = json.dumps([oracle["hand_authored"], oracle["meta"],
                            oracle["_policy"]], sort_keys=True)
        assert before == after, "freeze must not touch hand_authored, meta or _policy"
        json.dump(oracle, open(ORACLE, "w"), indent=2, ensure_ascii=False)
        print(f"frozen {len(oracle['frozen'])} worlds; hand section untouched")
        return 0

    print("=== AF-1 SELFTEST (RULE_REV=%d) ===" % R.RULE_REV)
    rows = oracle["hand_authored"]["rows"]

    # C1 -----------------------------------------------------------------
    declared = (list(P.LABEL_CONSTANTS) + list(P.DESIGN_CONSTANTS)
                + list(P.INFRA_CONSTANTS) + list(P.INTERNAL_CONSTANTS))
    CLASSES = ("LABEL_CONSTANTS", "DESIGN_CONSTANTS", "INFRA_CONSTANTS",
               "INTERNAL_CONSTANTS")
    # ★Only the three class tuples themselves are exempt.  rev2 exempted every
    # name ending in `_CONSTANTS`, so adding one was a way out of classification
    # entirely (2nd-audit L4).
    # ★3rd-audit N6: rev3 filtered on `n.isupper()`, so a constant named
    # `Sneak_Band_Mult` escaped classification entirely.  Everything module-level
    # that is not private, not callable and not an import is a constant.
    import types as _t
    # ★4th-audit K7: rev4 exempted every `_`-prefixed name, so a private
    # `_BAND = 2.0` could take over the label path with both gate-87 certificates
    # still green.  Private globals are classified too, in their own class.
    actual = [n for n in dir(P)
              if n not in CLASSES and not callable(getattr(P, n))
              and not isinstance(getattr(P, n), _t.ModuleType)
              and not n.startswith("__")]
    ck(len(declared) == len(set(declared)), "C1 no constant in two classes")
    ck(sorted(declared) == sorted(actual), "C1 partition covers every constant",
       "undeclared=%s ghost=%s" % (sorted(set(actual) - set(declared)),
                                   sorted(set(declared) - set(actual))))

    # C4 -----------------------------------------------------------------
    fok, fbad, fdetail = run_folds(oracle)
    ck(fbad == 0, "C4 all %d fold vectors pass" % (fok + fbad), fdetail)
    have = set(oracle["hand_authored"]["folds"])
    ck(have == set(R.PREDICATE_FOLDS), "C4 vectored folds == PREDICATE_FOLDS",
       "extra=%s missing=%s" % (sorted(have - set(R.PREDICATE_FOLDS)),
                                sorted(set(R.PREDICATE_FOLDS) - have)))
    ck(oracle["meta"]["fold_vector_count"] == fok + fbad,
       "C4 meta pins the fold-vector count")

    # C2 / C3 ------------------------------------------------------------
    prov0 = provenance_violations()
    ck(prov0 == 0, "C2 provenance baseline is clean", str(prov0))
    for name in P.LABEL_CONSTANTS:
        subs = list(SUBST.get(name, []))
        if name == "LADDER_TTFT_MS":
            subs += SUBST["LADDER_TTFT_MS_SHIFT"]
        moved = []
        for val in subs:
            with substituted(**{name: val}):
                lm = {key(w): one_label(w) for w in worlds()}
                f2 = run_folds(oracle)[1]
                prov = provenance_violations()
            moved.append(lm != base or f2 > fbad or prov > prov0)
        ck(subs and all(moved), "C2a LABEL constant is load-bearing by VALUE: %s" % name,
           "unmoved: %s" % ([v for v, m in zip(subs, moved) if not m][:1],))
        # ★4th-audit K7: rev4 used a substring test, which a docstring or comment
        # mention satisfies -- `screen_bound` could read a private `_BAND` while
        # its docstring still named `BAND_SD_MULT`.  Names are now taken from the
        # PARSED body, so only real reads count.
        readers = [f for f in R.PREDICATE_FOLDS if name in _read_names(getattr(P, f))]
        ck(bool(readers), "C2b LABEL constant is READ by the label path: %s" % name,
           "appears in no PREDICATE_FOLD's parsed body -- gate 87's hole")
    ident = [n for n in P.DESIGN_CONSTANTS
             for v in DESIGN_SUBST[n] if v == getattr(P, n)]
    ck(not ident, "C3b no registered DESIGN substitution equals its own value",
       str(sorted(set(ident))))
    identL = [n for n in P.LABEL_CONSTANTS
              for v in SUBST.get(n, []) if v == getattr(P, n)]
    ck(not identL, "C3b no registered LABEL substitution equals its own value",
       str(sorted(set(identL))))
    for name in P.DESIGN_CONSTANTS:
        still = []
        for val in DESIGN_SUBST[name]:
            with substituted(**{name: val}):
                lm = {key(w): one_label(w) for w in worlds()}
                still.append(lm == base and run_folds(oracle)[1] == fbad
                             and provenance_violations() == prov0)
        ck(all(still), "C3 DESIGN constant moves no label, both directions: %s" % name)

    # C5 -----------------------------------------------------------------
    ck(oracle["frozen"] == dict(base), "C5 enumeration matches frozen oracle",
       "%d frozen vs %d enumerated" % (len(oracle["frozen"]), len(base)))
    bad = [r["world"] for r in rows if base.get(r["world"]) != r["label"]]
    ck(not bad, "C5 every hand row matches the rule", str(bad[:3]))

    # C6 -----------------------------------------------------------------
    ck([r["world"] for r in rows] == oracle["meta"]["hand_world_keys"],
       "C6 meta pins the hand section's exact key set (order included)")
    ck(len(rows) == oracle["meta"]["hand_row_count"], "C6 meta pins hand row count")
    ck(len(set(r["world"] for r in rows)) == len(rows), "C6 no duplicated hand world")
    ck(oracle["hand_authored"]["substantive"] == R.SUBSTANTIVE,
       "C6 hand-authored substantive list equals the rule's")
    ck(len(R.SUBSTANTIVE) == oracle["meta"]["substantive_count"],
       "C6 meta pins the substantive count")
    covered = set(r["label"] for r in rows)
    ck(set(R.SUBSTANTIVE) <= covered, "C6 hand rows cover every substantive label",
       str(sorted(set(R.SUBSTANTIVE) - covered)))

    # C7 -----------------------------------------------------------------
    pre = HERE / oracle["meta"]["prereg_file"]
    text = ""
    if ck(pre.exists(), "C7 pre-registration file present", str(pre)):
        text = " ".join(pre.read_text().split())
    norm = lambda s: " ".join(s.split())
    # ★E1: the anchor carries world + label + BRANCH.  rev2 stopped at the label,
    # so flipping every prescription passed with a two-file edit while the
    # document still said otherwise.
    unpaired = [r["world"] for r in rows
                if not r["prose_anchor"].startswith(
                    "`%s` → `%s` → `%s`: " % (r["world"], r["label"], r["fork"]))]
    ck(not unpaired,
       "C7 every anchor is PAIRED with its own world key, label AND branch",
       str(unpaired[:2]))
    # ★2nd-audit L8: `label in why` is a substring test, so a `why` saying the
    # opposite of its label passed.  The anchor is built from the `why` tail, so
    # requiring them to agree exactly ties the two fields together.
    mism = []
    for r in rows:
        head = "`%s` → `%s` → `%s`: " % (r["world"], r["label"], r["fork"])
        tail = r["prose_anchor"][len(head):]
        if r["why"] != "%s — %s" % (r["label"], tail):
            mism.append(r["world"])
    ck(not mism, "C7 `why` is exactly its label plus the anchor's own text",
       str(mism[:2]))
    if text:
        missing = [r["world"] for r in rows if norm(r["prose_anchor"]) not in text]
        ck(not missing, "C7 every paired anchor appears verbatim in the prereg",
           str(missing[:2]))
        rmiss = [x["reason"] for x in oracle["meta"]["guard_order_independent_pairs"]
                 if norm(x["reason"]) not in text]
        ck(not rmiss, "C7 every order-independence reason appears in the prereg",
           str(rmiss[:1]))
        # ★2nd-audit E5: the only reverse check rev2 had was an INEQUALITY, so the
        # document could advertise rows the oracle did not have.  Exact now, and
        # the prose counts the document owns are pinned too.
        # the tables' own header row uses the literal placeholders, so exclude it
        nrows = len([m for m in re.findall(r"`[a-z_|]+` → `[A-Z_]+` → `[a-z0-9/_]+`: ",
                                           text) if "`world`" not in m])
        ck(nrows == len(rows),
           "C6 the prereg carries EXACTLY one anchor line per hand row",
           "%d anchors in the document vs %d hand rows" % (nrows, len(rows)))
        raw = pre.read_text()
        counts = {
            "prohibitions": raw.count("- ❌"),
            "ledger_0_1_rows": _table_rows(raw, "### 0.1"),
            "ledger_0_2_rows": _table_rows(raw, "### 0.2"),
            "ledger_0_3_rows": _table_rows(raw, "### 0.3"),
            # ★4th-audit K6: `ledger_0_4_rows` was the one ledger NOT counted --
            # the deletion ledger, made to satisfy gate 88, was itself deletable.
            "ledger_0_4_rows": _table_rows(raw, "### 0.4"),
            "limits": len(re.findall(r"(?m)^\d+\. ",
                                     _section(raw, "## 9."))),
        }
        # ★mutation MLIMITSWAP (4th audit, still alive after rev5's C20): the
        # count is unchanged when a limit's TEXT is swapped, and C20 only looks at
        # label-shaped tokens.  Each limit carries a verbatim anchor, the same
        # device the outcome table already uses.
        lim = _section(raw, "## 9.")
        miss = [a for a in oracle["meta"]["limit_anchors"]
                if " ".join(a.split()) not in " ".join(lim.split())]
        ck(not miss, "C6 every registered limit anchor is still in the document",
           str(miss[:2]))
        ck(len(oracle["meta"]["limit_anchors"]) == counts["limits"],
           "C6 one registered anchor per limit",
           "%d anchors vs %d limits" % (len(oracle["meta"]["limit_anchors"]),
                                        counts["limits"]))
        for k, v in counts.items():
            ck(oracle["meta"]["prereg_counts"].get(k) == v,
               "C6 meta pins the prereg's own count: %s" % k,
               "meta=%s document=%s" % (oracle["meta"]["prereg_counts"].get(k), v))

    # C19 ----------------------------------------------------------------
    if text:
        # ★3rd-audit J7: rev3's ledger cited `C3` -- a check that proves a constant
        # moves NOTHING -- as the thing that ENFORCES a registered condition.  The
        # column is now a closed vocabulary, and "moves nothing" is its own value.
        # closed vocabulary, three kinds: a check ENFORCES it / no machine check /
        # a check proves the constant moves NOTHING (an inertness certificate --
        # never enforcement).
        VOCAB = ("검사 ", "C", "기계 검사 없음", "부재 논증", "없음",
                 "의도된 부재", "승계", "이 표 자신", "무력 증명서")
        raw = pre.read_text()
        bad_cells = []
        for hdr in ("### 0.1", "### 0.2"):
            for ln in _section(raw, hdr).splitlines():
                if not ln.startswith("| ") or ln.startswith("|---"):
                    continue
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if len(cells) < 4 or cells[0] in ("조건", "死因"):
                    continue
                cell = cells[-1].replace("*", "").replace("`", "").strip()
                if not any(cell.startswith(v) for v in VOCAB):
                    bad_cells.append(cell[:40])
        ck(not bad_cells,
           "C19 every ledger enforcement cell uses the registered vocabulary",
           str(bad_cells[:3]))
        # ★4th-audit K4: rev4's second assertion was a hard-coded grep for one
        # string rev3 happened to write, and the first only looked at a prefix --
        # so citing an inert check, or a check that does not exist, both passed.
        # Check IDs are now PARSED and looked up.
        INERT = {"C3"}
        have = set(re.findall(r'"(C\d+[a-z]?) ', pathlib.Path(__file__).read_text()))
        cited, ghost, inert = set(), [], []
        for hdr in ("### 0.1", "### 0.2"):
            for ln in _section(raw, hdr).splitlines():
                if not ln.startswith("| ") or ln.startswith("|---"):
                    continue
                cell = ln.strip().strip("|").split("|")[-1]
                if "무력 증명서" in cell:          # declared inert, not enforcement
                    continue
                for cid in re.findall(r"`(C\d+[a-z]?)`", cell):
                    cited.add(cid)
                    if cid not in have:
                        ghost.append(cid)
                    if cid in INERT:
                        inert.append(cid)
        ck(not ghost, "C19 every cited check ID exists in this file", str(sorted(set(ghost))))
        ck(not inert, "C19 no inert check is cited as ENFORCEMENT", str(sorted(set(inert))))
        ck(bool(cited), "C19 the ledgers cite at least one check by ID")

    # C20 ----------------------------------------------------------------
    if text:
        # ★4th-audit K6: counting items does not stop an item from naming a label
        # that does not exist -- rev3 had a prohibition covering a label rev3 had
        # already removed, and the count protected it.
        # ★5th-audit Q1(c): rev5's token regex was `[A-Z][A-Z0-9_]{2,}` and every
        # BRANCH name in this design is lowercase, so `FORK_BRANCHES` sat in
        # `known` with no chance of ever being compared -- the check was true on
        # the empty set.  It now reads identifiers of either case, and keeps the
        # ones that look like identifiers rather than English prose (contain "_"
        # or start uppercase).
        known = (set(R.SUBSTANTIVE) | set(R.NON_SUBSTANTIVE) | set(R.FORK_BRANCHES)
                 | {v for vals in R.AXES.values() for v in vals} | set(R.AXES)
                 | set(dir(P)) | set(R.PREDICATE_FOLDS)
                 | {n for n, _t, _i in P.SURVEY_POINTS} | set(P.ARMS)
                 | {"AF", "CP", "SLO", "GPU", "ITL", "TTFT", "SD", "MFU", "SM",
                    "HBM", "A100", "BF16", "W1", "HE0", "PD", "C2",
                    "S1", "S2", "S3", "S4", "S5"})
        raw2 = pre.read_text()
        toks = set()
        for sect in ("## 8.", "## 9."):
            for m in re.finditer(r"`([A-Za-z][A-Za-z0-9_]{2,})`",
                                 _section(raw2, sect)):
                t = m.group(1)
                if "_" in t or t[0].isupper():
                    toks.add(t)
        ghosts = sorted(t for t in toks if t not in known)
        ck(not ghosts,
           "C20 every label-shaped token in the prohibitions and limits exists",
           str(ghosts[:4]))
        # ★mutation MC20NARROW: narrowing the regex back to UPPERCASE-only leaves a
        # clean document passing, so the widening that made this check see BRANCH
        # names at all was one edit from reverting.  The scan must be non-vacuous
        # in the direction it was widened for.
        ck(any(t[0].islower() for t in toks),
           "C20b the token scan actually reaches lowercase identifiers",
           "only %s -- the regex has been narrowed back" % sorted(toks)[:4])
        # ★6th-audit 등재 권고 2 (mutation MEXTIDLAUNDER): rev6 had an escape
        # hatch `external_identifiers` whose entries were accepted as known.  Its
        # own membership test asked whether the name appeared ANYWHERE in the
        # document, so parking a retired branch name there revived 5th-audit Q1
        # with rc=0.  The hatch is now GONE, and it turned out to be provably
        # unnecessary: every token this check scans is already covered by the
        # code's own registries, so `scanned - known` was empty without it.
        # C20c keeps it gone -- there is no list to launder through.
        ck("external_identifiers" not in oracle["meta"],
           "C20c no escape hatch exists for the token check",
           "a `known` override list is back; it is a laundering surface")

    # C8 -----------------------------------------------------------------
    hand = {r["world"]: r["label"] for r in rows}
    srw = oracle["meta"]["stop_rule_worlds"]
    ck(all(w in hand for w in srw), "C8 every stop-rule branch is a hand world",
       str([w for w in srw if w not in hand]))
    # ★2nd-audit L7: rev2 hard-coded the expected labels positionally, so swapping
    # a stop-rule world for a different world with the same label passed.  The
    # requirement is now structural: the stop-rule worlds must hit EVERY guard,
    # one each, in guard order.
    got = [base.get(w) for w in srw]
    want = [lab for _n, _p, lab in R.RULES]
    ck(got == want, "C8 stop-rule worlds hit every guard exactly once, in order",
       "%s != %s" % (got, want))

    # C9 / C10 -----------------------------------------------------------
    for i, (nm, _) in enumerate(R.INCOHERENCE):
        red = tuple(c for j, c in enumerate(R.INCOHERENCE) if j != i)
        ck({key(w): one_label(w, clauses=red) for w in worlds()} != base,
           "C9 incoherence clause is load-bearing: %s" % nm)
    indep = {tuple(x["pair"]) for x in oracle["meta"]["guard_order_independent_pairs"]}
    adj = [(R.RULES[i][0], R.RULES[i + 1][0]) for i in range(len(R.RULES) - 1)]
    for i, pair in enumerate(adj):
        sw = list(R.RULES)
        sw[i], sw[i + 1] = sw[i + 1], sw[i]
        if {key(w): one_label(w, rules=sw) for w in worlds()} != base:
            ck(True, "C10 guard order is load-bearing at %s<->%s" % pair)
        else:
            ck(pair in indep,
               "C10 order-independent pair %s<->%s is REGISTERED" % pair)
    ck(indep <= set(adj), "C10 no stale entry in the order-independent list")
    # ★2nd-audit L6: rev2 accepted any registered pair with a reason string, so a
    # FALSE claim of order-independence passed.  The claim is now verified.
    gpred = {n: pr for n, pr, _l in R.RULES}
    for pair in indep:
        both = [key(w) for w in worlds()
                if not R.incoherent(w) and gpred[pair[0]](w) and gpred[pair[1]](w)]
        ck(not both,
           "C10c registered order-independence is TRUE (guards are exclusive): "
           "%s<->%s" % pair, "both fire in %s" % (both[:1],))
    for i, (nm, _, _) in enumerate(R.RULES):
        red = [g for j, g in enumerate(R.RULES) if j != i]
        ck({key(w): one_label(w, rules=red) for w in worlds()} != base,
           "C10b guard is load-bearing when deleted: %s" % nm)

    # C11 ----------------------------------------------------------------
    img = P.reachable_pairs()
    ck(set(R.OUTCOME) <= img, "C11 no OUTCOME pair outside the screen's image",
       str(sorted(set(R.OUTCOME) - img)))
    ck(img <= set(R.OUTCOME), "C11 no reachable pair missing from OUTCOME",
       str(sorted(img - set(R.OUTCOME))))
    ck(set(R.SUBSTANTIVE) <= set(base.values()),
       "C11 every substantive label is reachable",
       str(sorted(set(R.SUBSTANTIVE) - set(base.values()))))
    trips = P.reachable_triples()
    outside = [w for w, l in base.items() if l in R.SUBSTANTIVE
               and (lambda c: (c[1], c[3], c[4]))(w.split("|")) not in trips
               and "unmeasured" not in w]
    ck(not outside,
       "C11 no substantive label sits on a world outside the TRIPLE image",
       str(outside[:3]))
    ck(all(R.fork_branch(l) in R.FORK_BRANCHES for l in set(base.values())),
       "C11 every label the rule can emit has a registered branch",
       str([l for l in set(base.values()) if R.fork_branch(l) not in R.FORK_BRANCHES]))
    ck(set(R.FORK) == set(R.SUBSTANTIVE), "C11 fork mapping is total and exhaustive",
       "extra=%s missing=%s" % (sorted(set(R.FORK) - set(R.SUBSTANTIVE)),
                                sorted(set(R.SUBSTANTIVE) - set(R.FORK))))
    ck(len(set(R.FORK.values())) >= 2, "C11 fork mapping is NOT constant")
    ck(set(R.FORK.values()) <= set(R.FORK_BRANCHES),
       "C11 fork branches come from the registered set")
    # ★4th-audit K6/P5: one direction let a ghost branch be added silently.
    ck(set(R.FORK_BRANCHES) - {"n/a"} <= set(R.FORK.values()),
       "C11 no registered branch is unused by the fork mapping",
       str(sorted(set(R.FORK_BRANCHES) - {"n/a"} - set(R.FORK.values()))))
    forkbad = [r["world"] for r in rows
               if r["fork"] != R.FORK.get(r["label"], "n/a")]
    ck(not forkbad, "C11 FORK content is pinned row by row by the hand oracle",
       str(forkbad[:3]))
    ck(len({r["fork"] for r in rows if r["fork"] != "n/a"}) >= 2,
       "C11 the hand oracle itself records more than one branch")

    # C17 ----------------------------------------------------------------
    # ★2nd-audit L3: `_IMAGE_CACHE`'s key is a hand-maintained dependency list and
    # nothing checked it, so dropping a constant from the key passed.  A stale
    # cache is exactly what that would produce, so compare cached against fresh.
    stale = []
    for name in P.LABEL_CONSTANTS:
        for val in SUBST.get(name, []):
            with substituted(**{name: val}):
                cached = P.reachable_pairs()
                keep = dict(P._IMAGE_CACHE)
                P._IMAGE_CACHE.clear()
                fresh = P.reachable_pairs()
                P._IMAGE_CACHE.clear()
                P._IMAGE_CACHE.update(keep)
            if cached != fresh:
                stale.append(name)
    ck(not stale, "C17 the image cache is never stale under any substitution",
       str(sorted(set(stale))))

    # C12 / C13 / C16 ----------------------------------------------------
    # ★4th-audit K7 / mutation MPREDCONSTDROP: rev5 made `PREDICATE_CONSTANTS`
    # derived, which stops it DRIFTING but not someone re-typing it as a literal.
    ck(tuple(R.PREDICATE_CONSTANTS) == tuple(P.LABEL_CONSTANTS),
       "C21 the rule's predicate-constant list IS the predicate module's",
       "%s != %s" % (tuple(R.PREDICATE_CONSTANTS), tuple(P.LABEL_CONSTANTS)))
    ck(P.SCREEN_STRATUM_Q in P.REPORT_STRATUM_QS
       and P.SCREEN_STRATUM_Q_ALT in P.REPORT_STRATUM_QS,
       "C12 both screen strata are reported strata")
    gaps = [w for w, l in base.items() if l == "OUTCOME_GAP"]
    ck(not gaps, "C13 no coherent world reaches an outcome gap", str(gaps[:3]))
    ck(P.PLANNED_BOOTS > P.MIN_BOOTS, "C16 planned boots exceed MIN_BOOTS (slack)")
    # ★4th-audit P4: sec 3 registers both properties and nothing checked either.
    ck(len(P.BOOT_SEEDS) == P.PLANNED_BOOTS,
       "C16 one registered seed per planned boot",
       "%d seeds vs %d boots" % (len(P.BOOT_SEEDS), P.PLANNED_BOOTS))
    ck(len(set(P.BOOT_SEEDS)) == len(P.BOOT_SEEDS), "C16 registered seeds are distinct")
    pos = [collections.Counter(o[i] for o in P.ARM_ORDER_BY_JOB)
           for i in range(len(P.ARMS))]
    ck(all(len(set(c.values())) == 1 and set(c) == set(P.ARMS) for c in pos),
       "C16 the arm rotation is BALANCED (every arm in every position equally often)",
       str([dict(c) for c in pos]))
    ck(len(P.ARM_ORDER_BY_JOB) == P.PLANNED_BOOTS,
       "C16 one registered arm order per planned boot")

    # C18 ----------------------------------------------------------------
    # ★3rd-audit J2: rev3 left the rev2 `stratum_axis` in the file above the new
    # one, so the whole repair was one deletion away from reverting and no check
    # saw it.  Gate 88's mirror image: here the danger is what did NOT go away.
    import ast as _ast
    for mod in (P, R):
        src = inspect.getsource(mod)
        names = [n.name for n in _ast.parse(src).body
                 if isinstance(n, (_ast.FunctionDef, _ast.ClassDef))]
        dup = sorted({n for n in names if names.count(n) > 1})
        ck(not dup, "C18 no duplicate top-level definition in %s"
           % mod.__name__, str(dup))

    # C22 ----------------------------------------------------------------
    # ★6th-audit 등재 권고 1 + mutation MHEADERREVERT: a change-history header
    # stayed at rev2 through four revisions while a ledger row claimed it had
    # been updated (gates #67/#70).  A header that does not name its own current
    # revision cannot be recording that revision.
    for mod, cname in ((P, "PRED_REV"), (R, "RULE_REV")):
        doc = mod.__doc__ or ""
        m = re.search(cname + r" = (\d+)", doc) if cname == "PRED_REV" else None
        rev = int(m.group(1)) if m else getattr(R, "RULE_REV")
        ck(("rev%d" % rev) in doc,
           "C22 %s's change history names its own current revision (rev%d)"
           % (mod.__name__, rev),
           "the header stops before the revision it is shipped with")

    # C14 ----------------------------------------------------------------
    if ck(CP2R2.exists(), "C14 CP-2 rev2 predicate file present"):
        src = CP2R2.read_text()
        for nm in ("LADDER_TTFT_MS", "LADDER_ITL_MS"):
            m = re.search(nm + r"\s*=\s*\(([^)]*)\)", src)
            theirs = tuple(float(x) for x in m.group(1).split(",") if x.strip()) if m else ()
            ck(theirs == getattr(P, nm),
               "C14 %s is CP-2 rev2's tuple verbatim" % nm,
               "%s vs %s" % (theirs, getattr(P, nm)))
    sp = SURVEY.resolve()
    if ck(sp.exists(), "C14 survey document present", str(sp)):
        # ★2nd-audit E4: rev2 asked whether the two NUMBERS appeared together on
        # some line, so `(150, 50)` matched "**150ms** | 30ms" and `(300, 50)` ->
        # `(200, 50)` matched a prose line quoting a DIFFERENT source.  The table
        # rows are parsed and the COORDINATE PAIRS compared as a set.
        pairs = set()
        for ln in sp.read_text().replace(",", "").splitlines():
            cells = [c.strip() for c in ln.split("|")]
            if len(cells) < 5:
                continue
            m1 = re.fullmatch(r"\**(\d+)ms\**", cells[2])
            m2 = re.fullmatch(r"\**(\d+)ms\**", cells[3])
            if m1 and m2:
                pairs.add((float(m1.group(1)), float(m2.group(1))))
        ours = {(t, i) for _n, t, i in P.SURVEY_POINTS}
        ck(bool(pairs), "C14 the survey table parsed at least one row",
           "regex matched nothing -- the document's format changed")
        ck(ours <= pairs,
           "C14 every candidate coordinate is a PARSED survey table row",
           "not in the table: %s" % sorted(ours - pairs))

    # C15 ----------------------------------------------------------------
    if text:
        ck(P.survey_is_chain(), "C15 the candidate set is a total order")
        ck(norm("`anchor = unique`가 뜻할 수 있는 것은 언제나 사슬의 최대원소") in text,
           "C15 the chain consequence is disclosed in the prereg")

    print()
    if FAIL:
        print("SELFTEST FAILED (%d): %s" % (len(FAIL), FAIL[:6]))
        return 1
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
