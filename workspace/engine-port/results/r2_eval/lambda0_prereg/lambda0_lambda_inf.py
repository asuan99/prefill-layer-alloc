#!/usr/bin/env python3
"""D18 -- the objective predicate that selects the anchored plan or the fallback.

rev2 branched on an operator environment variable (`PDMUX_LAMBDA0_FALLBACK=1`,
`lambda0.sbatch:104`).  The rev2 audit showed that discretion FLIPS a label: at
lambda*(B) = 1.10 the anchored window [0.5157, 1.835] brackets and the fallback
window [0.4302, 1.026] does not.  A branch that changes a verdict may not be a
human choice made after the fact, so the decision is moved into a predicate over
FILES that the adjacent pre-registration
(`results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` sec 5-I3)
either produced or did not.

ANCHORED iff ALL of:
  (a) `<instr>/I3a_shapeA.jsonl` and `<instr>/I3b_shapeB.jsonl` both exist and
      parse, and each reports a positive `request_throughput` over a
      `--request-rate inf` run (`request_rate` is `inf`/null in that mode);
  (b) `<instr>/I3_max_running_req.txt` exists and EVERY value it lists is >= 48.
      This is newpair F5: "if 48 is not reached, that cell's achieved rate may
      not be cited as a saturation throughput."  Not reaching it is recorded as
      cause-undetermined there, so here it simply disqualifies the anchor.
Otherwise FALLBACK.

lambda_inf(A) = I3a `request_throughput`, lambda_inf(B) = I3b `request_throughput`.

★lambda_inf is an UPPER-BOUND probe and is used ONLY to place the ladder.  It is
never the reported lambda*, and nothing in this stage assumes it bounds lambda*
from below (newpair (D7)/NP-3': "I3 fixes only the UPPER end of the ladder").

usage:
  lambda0_lambda_inf.py <instrument_dir>     # prints shell assignments, rc 0
  lambda0_lambda_inf.py --selftest
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

MIN_RUNNING_REQ = 48        # newpair F5
CELL_FILES = {"A": "I3a_shapeA.jsonl", "B": "I3b_shapeB.jsonl"}
RUNNING_FILE = "I3_max_running_req.txt"


def _sha256(p: Path):
    """E6: the anchor's provenance.  `PDMUX_LAMBDA0_INSTR` says WHERE to look,
    and the rev3 audit showed that choice can flip a shape-A label, so the
    resolved ABSOLUTE path and the digest of every artifact the decision reads
    go into the run record."""
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


def _last_int_per_line(text):
    vals = []
    for line in text.splitlines():
        found = re.findall(r"-?\d+", line)
        if found:
            vals.append(int(found[-1]))
    return vals


def decide(instr_dir):
    """Returns a dict: mode ANCHORED|FALLBACK, lambda_inf, and every reason."""
    d = Path(instr_dir)
    reasons, lam = [], {}
    for shape, fn in CELL_FILES.items():
        p = d / fn
        if not p.exists():
            reasons.append(f"missing {fn}")
            continue
        try:
            rec = json.loads(p.read_text().strip().split("\n")[0])
        except Exception as exc:
            reasons.append(f"unparseable {fn}: {type(exc).__name__}")
            continue
        thr = rec.get("request_throughput")
        if not isinstance(thr, (int, float)) or not thr > 0:
            reasons.append(f"{fn}: request_throughput not positive ({thr!r})")
            continue
        rate = rec.get("request_rate")
        if rate is not None and rate == rate and rate != float("inf"):
            reasons.append(f"{fn}: request_rate is {rate!r}, expected inf")
            continue
        lam[shape] = float(thr)

    rp = d / RUNNING_FILE
    if not rp.exists():
        reasons.append(f"missing {RUNNING_FILE}")
    else:
        vals = _last_int_per_line(rp.read_text())
        if not vals:
            reasons.append(f"{RUNNING_FILE}: no integer found")
        elif min(vals) < MIN_RUNNING_REQ:
            reasons.append(f"{RUNNING_FILE}: min(#running-req)={min(vals)} < "
                           f"{MIN_RUNNING_REQ} (newpair F5 -- the cell's achieved "
                           f"rate may not be cited as a saturation throughput)")

    anchored = (len(lam) == 2 and not reasons)
    inputs = {}
    for fn in list(CELL_FILES.values()) + [RUNNING_FILE]:
        fp = (d / fn).resolve()
        inputs[fn] = {"abspath": str(fp), "exists": fp.exists(),
                      "sha256": _sha256(fp),
                      "size": fp.stat().st_size if fp.exists() else None}
    return {"mode": "ANCHORED" if anchored else "FALLBACK",
            "lambda_inf": lam if anchored else {},
            "min_running_req": MIN_RUNNING_REQ,
            "disqualifications": reasons,
            "instrument_dir": str(d.resolve()),       # E6: ABSOLUTE, resolved
            "inputs": inputs}


def selftest():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # nothing present -> FALLBACK
        assert decide(d)["mode"] == "FALLBACK"

        def write(thr_a=2.4, thr_b=0.8, running="48", rate=None, drop=()):
            for shape, fn in CELL_FILES.items():
                if fn in drop:
                    (d / fn).unlink(missing_ok=True)
                    continue
                (d / fn).write_text(json.dumps(
                    {"request_throughput": thr_a if shape == "A" else thr_b,
                     "request_rate": rate}))
            if running is None:
                (d / RUNNING_FILE).unlink(missing_ok=True)
            else:
                (d / RUNNING_FILE).write_text(running)

        write()
        r = decide(d)
        assert r["mode"] == "ANCHORED", r
        assert r["lambda_inf"] == {"A": 2.4, "B": 0.8}, r
        # ★E6: the resolved ABSOLUTE dir and a digest per input must be
        # recorded, so a later reader can tell WHICH artifacts produced the
        # anchor (the rev3 audit showed the directory choice flips a label).
        assert Path(r["instrument_dir"]).is_absolute(), r
        assert set(r["inputs"]) == set(CELL_FILES.values()) | {RUNNING_FILE}, r
        for fn, meta in r["inputs"].items():
            assert meta["exists"] and meta["sha256"] and len(meta["sha256"]) == 64, (fn, meta)
            assert Path(meta["abspath"]).is_absolute(), (fn, meta)
        # a digest must actually track content
        before = r["inputs"][RUNNING_FILE]["sha256"]
        write(running="52")
        assert decide(d)["inputs"][RUNNING_FILE]["sha256"] != before
        write()

        # ★each disqualification independently forces FALLBACK -- otherwise the
        # predicate would be decorative and the branch would still be free.
        write(drop=("I3b_shapeB.jsonl",))
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        # ★the REASON must be recorded, not only the mode: E6 writes these into
        # the run record, and "why did this job take the fallback?" is exactly
        # the question the provenance line has to answer months later.
        assert any("I3b_shapeB.jsonl" in x for x in r["disqualifications"]), r
        write(running="31")
        assert decide(d)["mode"] == "FALLBACK", decide(d)
        write(running="I3a_shapeA 48\nI3b_shapeB 31")
        assert decide(d)["mode"] == "FALLBACK", decide(d)
        write(running="I3a_shapeA 48\nI3b_shapeB 52")
        assert decide(d)["mode"] == "ANCHORED", decide(d)
        write(running=None)
        assert decide(d)["mode"] == "FALLBACK", decide(d)
        write(thr_a=0.0)
        assert decide(d)["mode"] == "FALLBACK", decide(d)
        write(rate=4.0)            # a finite rate is not the I3 protocol
        assert decide(d)["mode"] == "FALLBACK", decide(d)
    print("LAMBDA_INF PREDICATE SELFTEST OK (anchored only when both I3 cells "
          "exist, ran at rate inf with positive throughput, and every reported "
          "#running-req >= %d; each condition shown to force FALLBACK alone; "
          "the resolved absolute dir and a content digest per input are "
          "recorded [E6])" % MIN_RUNNING_REQ)


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
        return 0
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    r = decide(sys.argv[1])
    print(f"LAMBDA0_MODE={r['mode']}")
    print(f"LAMBDA0_LAMBDA_INF_A={r['lambda_inf'].get('A', '')}")
    print(f"LAMBDA0_LAMBDA_INF_B={r['lambda_inf'].get('B', '')}")
    print("LAMBDA0_DISQUALIFICATIONS=" + json.dumps(r["disqualifications"]))
    print("LAMBDA0_INSTR_ABSPATH=" + r["instrument_dir"])
    for fn, meta in sorted(r["inputs"].items()):
        print("LAMBDA0_INPUT_SHA256_%s=%s" % (fn.replace(".", "_"),
                                              meta["sha256"] or "ABSENT"))
    # The machine-readable twin travels with the cells so the label JSON can
    # embed it (E6).
    return 0


if __name__ == "__main__":
    sys.exit(main())
