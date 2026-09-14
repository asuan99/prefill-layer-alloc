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

★rev5/F1 -- THE rev4 KILL CAUSE (audit N1), AND WHAT CHANGED
------------------------------------------------------------
rev4 read condition (b) out of `<instr>/I3_max_running_req.txt`.  The PRODUCER is
the block in `results/r2_correctness/r2_correctness.sbatch` that writes that file,
quoted here BY ITS CODE TEXT rather than by a line number (lam5A-11):

    grep -oE "#running-req: [0-9]+" "$OUT/srv_warmup.log" | awk '{print $2}' \
      | sort -n | tail -1 > "$IOUT/I3_max_running_req.txt"

i.e. ONE line holding the GLOBAL maximum over the whole warm-up server log.  So
rev4's `min(vals) >= 48` was identical to "the global max >= 48" = "SOME cell
touched 48", while the registered wording (newpair F5) is "EACH of the two cells
reached 48".  The disqualification path was an effective IDENTITY: the rev4 audit
ran the shipped predicate on job 907959 and got `ANCHORED` with zero
disqualifications, byte-for-byte equal to a mutant that honestly wrote
`vals = [max(vals)]`.

★WHY THE PRODUCER IS CITED BY CODE TEXT AND NOT BY LINE (lam5A-11).  That script
belongs to ANOTHER track, which is editing it: the block above sits at `:496-497`
at HEAD 8507cee and at `:580-581` in the uncommitted working tree of 2026-09-14
(D-none `R2C_GUARD` + `DN` boot grew it 573 -> 684 lines), and the rev4 verdict's
`:429-430` matches NEITHER.  A bare line number here would be true of exactly one
snapshot nobody can name later -- lesson 80, false provenance is most dangerous
inside a rules-canon file.  The quoted code text is stable, greppable and unique
in that file (verified: one occurrence at HEAD and one in the working tree), so
it survives the other track's edits.  The working-tree bytes it was checked
against are sha256 `ab55c07cc096bd4cdfe6d484fc9a983302561e03232f641c5994298bdc183bd3`.

The repair recomputes condition (b) PER CELL from the two artifacts the producer
really writes, both of which live in the same job directory:

    <instr>/I_log_offsets.txt   `wc -l` of srv_warmup.log before/after each
                                read-out.  Producer = that same script's
                                `i_mark ()` helper, which appends
                                `<NAME> <wc -l of srv_warmup.log>` and is called
                                as `i_mark "${I_NAME}_start"` / `..._end`
                                (code text, not a line number -- lam5A-11)
    <job>/srv_warmup.log        the engine's own `#running-req:` lines

and `I3_max_running_req.txt` is demoted to PROVENANCE + a cross-check: it can
still VETO the anchor (if our whole-log recount disagrees with it, the two
artifacts are not from the same run) but it can never GRANT one.

THE THREE REGISTERED CONVENTIONS (prereg rev5 sec 4.4; gate #113)
-----------------------------------------------------------------
"Recompute per cell" is itself a choice, so all three candidate readings are
computed, all three are printed, and the anchor requires ALL THREE to clear the
threshold.  Nothing is hidden and no single reading is privileged:

  all_lines            every line in the cell's interval that reports
                       `#running-req: N`                            [PRIMARY]
  decode_only          only `Decode batch` lines
  drop_prev_cell_tail  drop the interval's leading lines up to the cell's own
                       first `Prefill batch` (the previous read-out's drain)

Two further conventions are NOT free, because the producer fixes them:

  * INTERVAL = start EXCLUSIVE, end INCLUSIVE.  `i_mark` records `wc -l` BEFORE
    the bench command and again AFTER it, so the cell owns 1-based lines
    start+1 .. end (newpair sec 5-I3 (D4)).
  * LINE INDEXING = split on b"\\n" and nothing else, because the offsets were
    produced by `wc -l`, which counts newline bytes.  ★Python's `splitlines()`
    ALSO splits \\r, \\v, \\f, \\x1c-\\x1e, \\x85, U+2028/9; job 907959's log
    carries 116 bare \\r (the loader's progress bar), so `splitlines()` yields
    18498 lines against `wc -l`'s 18382 and every offset lands 116 lines early
    -- which makes cell I3b read 11 instead of 2.  Both readings fail F5, but
    only one of them is the producer's line numbering.

WHAT THE REPAIR DOES TO THE BRANCH: on job 907959 the per-cell recount is
I3a = 48 (F5 met) and I3b = 2 (F5 NOT met), so the predicate now answers
FALLBACK where rev4 answered ANCHORED.  That flip, and everything downstream of
it, is registered in prereg rev5 (F2) BEFORE the run.

ANCHORED iff ALL of:
  (a) `<instr>/I3a_shapeA.jsonl` and `<instr>/I3b_shapeB.jsonl` both exist and
      parse, and each reports a positive `request_throughput` over a
      `--request-rate inf` run (`request_rate` is `inf`/null in that mode);
  (b) for EACH of those two cells, the max `#running-req` the engine itself
      logged inside THAT CELL's interval of `srv_warmup.log` is >= 48 under ALL
      THREE registered conventions.  This is newpair F5: "if 48 is not reached,
      that cell's achieved rate may not be cited as a saturation throughput."
      Not reaching it is recorded as cause-undetermined there, so here it simply
      disqualifies the anchor.
  (c) `<instr>/I3_max_running_req.txt` exists and agrees with our recount of the
      whole log (same regex, same file) -- an integrity check that the offsets,
      the log and the summary line come from one run.
Otherwise FALLBACK.

lambda_inf(A) = I3a `request_throughput`, lambda_inf(B) = I3b `request_throughput`.

★lambda_inf is an UPPER-BOUND probe and is used ONLY to place the ladder.  It is
never the reported lambda*, and nothing in this stage assumes it bounds lambda*
from below (newpair (D7)/NP-3': "I3 fixes only the UPPER end of the ladder").

usage:
  lambda0_lambda_inf.py <instrument_dir>     # prints shell assignments, rc 0
  lambda0_lambda_inf.py --recount <instr>    # the per-cell table, no decision
  lambda0_lambda_inf.py --selftest
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

MIN_RUNNING_REQ = 48        # newpair F5
CELL_FILES = {"A": "I3a_shapeA.jsonl", "B": "I3b_shapeB.jsonl"}
RUNNING_FILE = "I3_max_running_req.txt"   # rev4's input; now provenance + veto
OFFSETS_FILE = "I_log_offsets.txt"        # F1 input 1, in <instr>/
SERVER_LOG = "srv_warmup.log"             # F1 input 2, in <instr>/../

RUNNING_RE = re.compile(rb"#running-req:\s*(\d+)")
DECODE_MARK = b"Decode batch"
PREFILL_MARK = b"Prefill batch"

# ★the three registered readings of "this cell's max #running-req".  The anchor
# needs ALL of them, so picking one is not a degree of freedom (gate #113).
CONVENTIONS = ("all_lines", "decode_only", "drop_prev_cell_tail")

# ★the live artifact the default branch reads.  The selftest is run against
# THESE BYTES, not against a hand-made fixture: rev4 died because its selftest
# exercised a two-line format the producer never writes (gate G-lam4-1).
LIVE_JOB_DEFAULT = HERE.parent.parent / "r2_correctness" / "job_907959"

# Registered recount of job 907959 (2026-09-13).  Same table as prereg rev5
# sec 4.4; asserted against the live bytes below so document and code cannot
# drift.  cell -> (interval_1based, all, decode_only, drop_tail,
#                  n_all, n_decode_only, n_drop_tail, lines_in_interval)
LIVE_RECOUNT = {
    "I2":         ((85, 4226),      1,  1,  1, 4128, 4119, 4128, 4142),
    "I3a_shapeA": ((4227, 8434),   48, 48, 48, 3904, 3880, 3904, 4208),
    "I3b_shapeB": ((8435, 18380),   2,  2,  2, 9632, 9481, 9632, 9946),
}
LIVE_LOG_LINES = 18382          # `wc -l < srv_warmup.log`
LIVE_GLOBAL_MAX = 48            # what the producer's `sort -n | tail -1` wrote

# Real slices of that same log, used as the fixtures for the legs the live
# artifact cannot discriminate (all values verified against the bytes):
#   (start_exclusive, end_inclusive) -> (all, decode_only, drop_tail)
W_OK_A = (4444, 5000)           # (48, 48, 48)  clears F5
W_OK_B = (7000, 7903)           # (48, 48, 48)  clears F5
W_LOW = (5661, 5691)            # (35, 35, 35)  32 <= max < 48
W_BOUND = (7903, 8000)          # (43, 43, 43); line 7903 itself reports 48, so
                                # a start-INCLUSIVE reading would say (48,48,43)


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


def live_job_dir() -> Path:
    """The job whose artifacts the predicate's LIVE selftest leg recounts.

    `LAMBDA0_I3_JOB_DIR` exists for the mutation harness, which runs a copy of
    this file from a temp dir -- and, since lam5A-2 (rev5 addendum sec B-1), for
    `lambda0.sbatch`, which BINDS it to the dir the branch is decided from
    (`export LAMBDA0_I3_JOB_DIR="$(dirname "$INSTR")"`).  Before that binding the
    two inputs were independent, so a substituted `PDMUX_LAMBDA0_INSTR` flipped
    the branch to ANCHORED while this selftest still certified job 907959 (audit
    W4').  ★It is therefore NOT a free knob: whatever it names must satisfy the
    registered LIVE_RECOUNT table below, so pointing it anywhere else FAILS.
    """
    return Path(os.environ.get("LAMBDA0_I3_JOB_DIR", str(LIVE_JOB_DEFAULT)))


def log_lines(raw: bytes):
    """Split EXACTLY as `wc -l` counts -- on b"\\n", nothing else.

    The offsets this predicate indexes with were produced by `wc -l` (the
    producer's `i_mark ()` helper; cited by code text, not by line -- lam5A-11).
    `bytes.splitlines()` would also break on
    \\r/\\v/\\f/\\x1c-\\x1e and shift every offset (116 lines on job 907959).
    """
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()                      # trailing newline is not a line
    return lines


def read_offsets(text: str):
    """`NAME <wc -l>` per line, the producer's own format (`i_mark`)."""
    off = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            off[parts[0]] = int(parts[1])
    return off


def cell_running_req(lines, start: int, end: int) -> dict:
    """Max `#running-req` inside ONE cell's interval, under all 3 conventions.

    INTERVAL: start EXCLUSIVE, end INCLUSIVE -- the producer marks `wc -l`
    before and after the read-out, so the cell owns 1-based lines start+1..end
    (newpair sec 5-I3 (D4)).
    """
    seg = lines[start:end]
    first_prefill = next((i for i, l in enumerate(seg) if PREFILL_MARK in l), 0)
    per = {"interval_1based": [start + 1, end], "lines_in_interval": len(seg),
           "drop_prev_cell_tail_lines": first_prefill}
    for conv in CONVENTIONS:
        if conv == "all_lines":
            sub = seg
        elif conv == "decode_only":
            sub = [l for l in seg if DECODE_MARK in l]
        else:
            sub = seg[first_prefill:]
        vals = [int(m.group(1)) for l in sub
                for m in (RUNNING_RE.search(l),) if m]
        per[conv] = max(vals) if vals else None
        per["n_" + conv] = len(vals)
    return per


def recount(instr_dir) -> dict:
    """★F1: the per-cell max `#running-req`, recomputed from the producer's own
    log.  Judged cells = exactly the cells whose `request_throughput` is cited
    as lambda_inf; every other cell in the offsets file is REPORTED, never
    judged (I2 runs at concurrency 1, so requiring 48 of it would be nonsense).
    """
    d = Path(instr_dir)
    job = d.parent
    op, lp = d / OFFSETS_FILE, job / SERVER_LOG
    out = {"conventions": list(CONVENTIONS), "convention_primary": CONVENTIONS[0],
           "interval_convention": "start_exclusive_end_inclusive",
           "line_convention": "wc_l_newline_only",
           "judged_cells": [Path(CELL_FILES[s]).stem for s in sorted(CELL_FILES)],
           "offsets_path": str(op.resolve()), "server_log_path": str(lp.resolve()),
           "cells": {}, "errors": []}
    if not op.exists():
        out["errors"].append(f"missing {OFFSETS_FILE} (F1 slices the log per cell "
                             f"with it, so without it no cell can be judged)")
    if not lp.exists():
        out["errors"].append(f"missing {SERVER_LOG} beside {d.name}/ (F1 reads "
                             f"the engine's own #running-req lines from it)")
    if out["errors"]:
        return out
    lines = log_lines(lp.read_bytes())
    off = read_offsets(op.read_text(errors="replace"))
    out["server_log_lines"] = len(lines)
    out["offsets"] = off
    # Whole-log max under the producer's own regex.  NOT a decision input: it is
    # the quantity `I3_max_running_req.txt` holds, recomputed so the two can be
    # compared (if they disagree, the log and the instrument dir are not one run).
    whole = [int(m.group(1)) for l in lines for m in (RUNNING_RE.search(l),) if m]
    out["server_log_global_max"] = max(whole) if whole else None
    stems = sorted({k[:-6] for k in off if k.endswith("_start")}
                   & {k[:-4] for k in off if k.endswith("_end")})
    for stem in stems:
        s, e = off[stem + "_start"], off[stem + "_end"]
        if s < 0 or e < s:
            out["cells"][stem] = {"error": f"malformed interval ({s}, {e}]"}
        elif e > len(lines):
            out["cells"][stem] = {"error": f"interval ({s}, {e}] runs past the "
                                           f"end of the log ({len(lines)} lines)"}
        else:
            out["cells"][stem] = cell_running_req(lines, s, e)
    return out


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

    # ---- condition (b), PER CELL (F1).  rev4 read one global-max line here.
    rc = recount(d)
    reasons.extend(rc["errors"])
    if not rc["errors"]:
        for stem in rc["judged_cells"]:
            m = rc["cells"].get(stem)
            if m is None:
                reasons.append(f"{OFFSETS_FILE}: no interval recorded for cell "
                               f"{stem}, so its #running-req cannot be checked")
                continue
            if m.get("error"):
                reasons.append(f"{stem}: {m['error']}")
                continue
            vals = [m[c] for c in CONVENTIONS]      # ★the F1 decision source
            shown = ", ".join(f"{c}={m[c]}" for c in CONVENTIONS)
            if any(v is None for v in vals):
                reasons.append(f"{stem}: no '#running-req' line inside its own "
                               f"log interval {m['interval_1based']} ({shown})")
            elif min(vals) < MIN_RUNNING_REQ:
                split = " -- and the three registered conventions DISAGREE, " \
                        "which disqualifies on its own" \
                        if max(vals) >= MIN_RUNNING_REQ else ""
                reasons.append(
                    f"{stem}: max #running-req over its OWN log interval "
                    f"{m['interval_1based']} is {shown} < {MIN_RUNNING_REQ}"
                    f"{split} (newpair F5 -- that cell's achieved rate may not "
                    f"be cited as a saturation throughput)")

    # ---- condition (c): the producer's summary line must describe this log.
    rp = d / RUNNING_FILE
    legacy = None
    if not rp.exists():
        reasons.append(f"missing {RUNNING_FILE}")
    else:
        vals = _last_int_per_line(rp.read_text())
        if not vals:
            reasons.append(f"{RUNNING_FILE}: no integer found")
        else:
            legacy = max(vals)
            got = rc.get("server_log_global_max")
            if got is not None and legacy != got:
                reasons.append(
                    f"{RUNNING_FILE} says {legacy} but recounting the whole "
                    f"{SERVER_LOG} with the producer's own regex gives {got}: "
                    f"the offsets, the log and the summary are not one run")

    anchored = (len(lam) == 2 and not reasons)
    inputs = {}
    for fn, base in ([(f, d) for f in list(CELL_FILES.values())
                      + [RUNNING_FILE, OFFSETS_FILE]] + [(SERVER_LOG, d.parent)]):
        fp = (base / fn).resolve()
        inputs[fn] = {"abspath": str(fp), "exists": fp.exists(),
                      "sha256": _sha256(fp),
                      "size": fp.stat().st_size if fp.exists() else None}
    return {"mode": "ANCHORED" if anchored else "FALLBACK",
            "lambda_inf": lam if anchored else {},
            "min_running_req": MIN_RUNNING_REQ,
            "disqualifications": reasons,
            "instrument_dir": str(d.resolve()),       # E6: ABSOLUTE, resolved
            "running_req_recount": rc,               # F1: every cell, every rule
            "running_req_file_value": legacy,        # provenance, not a decision
            "inputs": inputs}


# --------------------------------------------------------------- selftest legs
def _live_paths():
    job = live_job_dir()
    instr = job / "instrument"
    for p in (job / SERVER_LOG, instr / OFFSETS_FILE, instr / RUNNING_FILE):
        assert p.exists(), (
            "F1 recount artifact missing: %s -- this selftest is run against the "
            "PRODUCER's own bytes, not a fixture, because rev4 died of a "
            "selftest whose fixture the producer never writes (audit N1 / gate "
            "G-lam4-1).  LAMBDA0_I3_JOB_DIR is set by the mutation harness and "
            "-- since lam5A-2 -- by lambda0.sbatch, which binds it to the dir "
            "the branch is decided from; it is not a free knob."
            % p)
    return job, instr


def _live_leg():
    """★(A) the live artifact: the registered per-cell table, and the branch."""
    job, instr = _live_paths()
    rc = recount(instr)
    assert not rc["errors"], rc["errors"]
    assert rc["server_log_lines"] == LIVE_LOG_LINES, (
        "`wc -l` line count of the producer's log changed", rc["server_log_lines"])
    assert rc["server_log_global_max"] == LIVE_GLOBAL_MAX, rc["server_log_global_max"]
    assert rc["judged_cells"] == ["I3a_shapeA", "I3b_shapeB"], rc["judged_cells"]
    assert set(rc["cells"]) == set(LIVE_RECOUNT), sorted(rc["cells"])
    for stem, want in LIVE_RECOUNT.items():
        g = rc["cells"][stem]
        got = (tuple(g["interval_1based"]),
               g["all_lines"], g["decode_only"], g["drop_prev_cell_tail"],
               g["n_all_lines"], g["n_decode_only"], g["n_drop_prev_cell_tail"],
               g["lines_in_interval"])
        assert got == want, (stem, "registered", want, "got", got)
    # ★the flip rev5/F2 registers: rev4 printed ANCHORED here.
    r = decide(instr)
    assert r["mode"] == "FALLBACK", r
    assert r["lambda_inf"] == {}, r
    assert any("I3b_shapeB" in x and "F5" in x for x in r["disqualifications"]), r
    # I3a MET F5 and I2 is not judged at all, so neither may appear as a cause.
    assert not any("I3a_shapeA:" in x for x in r["disqualifications"]), r
    assert not any(x.startswith("I2") for x in r["disqualifications"]), r
    # ...and the file rev4 decided from says 48, i.e. the decision no longer
    # comes from it.
    assert r["running_req_file_value"] == LIVE_GLOBAL_MAX, r


def _fixture(td, cells, running="48", thr_a=2.4, thr_b=0.8, rate=None, drop=(),
             offsets=True, log=True):
    """A job directory whose `srv_warmup.log` IS the producer's real log
    (symlinked, byte for byte) and whose offsets file is written in the
    producer's own `NAME <wc -l>` format.  `cells` maps a cell stem to
    (start_exclusive, end_inclusive)."""
    job, _ = _live_paths()
    root = Path(td)
    instr = root / "instrument"
    instr.mkdir(parents=True, exist_ok=True)
    tgt = root / SERVER_LOG
    if tgt.is_symlink() or tgt.exists():
        tgt.unlink()
    if log:
        tgt.symlink_to(job / SERVER_LOG)
    for shape, fn in CELL_FILES.items():
        p = instr / fn
        if fn in drop:
            p.unlink(missing_ok=True)
            continue
        p.write_text(json.dumps({"request_throughput": thr_a if shape == "A"
                                 else thr_b, "request_rate": rate}))
    rf = instr / RUNNING_FILE
    if running is None:
        rf.unlink(missing_ok=True)
    else:
        rf.write_text(running)
    of = instr / OFFSETS_FILE
    if offsets:
        of.write_text("".join(f"{stem}_start {s}\n{stem}_end {e}\n"
                              for stem, (s, e) in cells.items()))
    else:
        of.unlink(missing_ok=True)
    return instr


def _vals(instr, stem):
    m = recount(instr)["cells"][stem]
    return tuple(m[c] for c in CONVENTIONS)


def selftest() -> None:
    ok = {"I3a_shapeA": W_OK_A, "I3b_shapeB": W_OK_B}
    _live_leg()
    with tempfile.TemporaryDirectory() as td:
        # nothing present -> FALLBACK
        assert decide(Path(td))["mode"] == "FALLBACK"

        d = _fixture(td, ok)
        r = decide(d)
        assert r["mode"] == "ANCHORED", r
        assert r["lambda_inf"] == {"A": 2.4, "B": 0.8}, r
        assert _vals(d, "I3a_shapeA") == (48, 48, 48), _vals(d, "I3a_shapeA")
        assert _vals(d, "I3b_shapeB") == (48, 48, 48), _vals(d, "I3b_shapeB")
        # ★E6: the resolved ABSOLUTE dir and a digest per input must be
        # recorded, so a later reader can tell WHICH artifacts produced the
        # anchor (the rev3 audit showed the directory choice flips a label).
        assert Path(r["instrument_dir"]).is_absolute(), r
        assert set(r["inputs"]) == (set(CELL_FILES.values())
                                   | {RUNNING_FILE, OFFSETS_FILE, SERVER_LOG}), r
        for fn, meta in r["inputs"].items():
            assert meta["exists"] and meta["sha256"] and len(meta["sha256"]) == 64, (fn, meta)
            assert Path(meta["abspath"]).is_absolute(), (fn, meta)
        # a digest must actually track content
        before = r["inputs"][OFFSETS_FILE]["sha256"]
        _fixture(td, {"I3a_shapeA": W_OK_A, "I3b_shapeB": W_OK_A})
        assert decide(d)["inputs"][OFFSETS_FILE]["sha256"] != before

        # ★F1 (b): the judged quantity is THIS CELL's interval, not the global
        # max.  Here the summary file still says 48 -- rev4 would ANCHOR -- and
        # cell B's own interval peaks at 43.
        d = _fixture(td, {"I3a_shapeA": W_OK_A, "I3b_shapeB": W_BOUND})
        assert _vals(d, "I3b_shapeB") == (43, 43, 43), _vals(d, "I3b_shapeB")
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        assert any("I3b_shapeB" in x and "43" in x and "F5" in x
                   for x in r["disqualifications"]), r
        # ★the INTERVAL convention is load bearing: line 7903 (the last line of
        # the preceding read-out) itself reports 48, so a start-INCLUSIVE
        # reading of the same offsets would hand cell B its neighbour's batch.
        d2 = _fixture(td, {"I3a_shapeA": W_OK_A,
                           "I3b_shapeB": (W_BOUND[0] - 1, W_BOUND[1])})
        assert _vals(d2, "I3b_shapeB") == (48, 48, 43), _vals(d2, "I3b_shapeB")
        assert decide(d2)["mode"] == "FALLBACK", decide(d2)   # conventions split
        assert any("DISAGREE" in x for x in decide(d2)["disqualifications"])

        # ★the threshold is 48, not "some number": a cell whose own interval
        # peaks at 35 is disqualified (auditor mutation Z5 lowers it to 32).
        d = _fixture(td, {"I3a_shapeA": W_OK_A, "I3b_shapeB": W_LOW})
        assert _vals(d, "I3b_shapeB") == (35, 35, 35), _vals(d, "I3b_shapeB")
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        assert any("35" in x and "F5" in x for x in r["disqualifications"]), r

        # ★each disqualification independently forces FALLBACK -- otherwise the
        # predicate would be decorative and the branch would still be free.
        d = _fixture(td, ok, drop=("I3b_shapeB.jsonl",))
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        # ★the REASON must be recorded, not only the mode: E6 writes these into
        # the run record, and "why did this job take the fallback?" is exactly
        # the question the provenance line has to answer months later.
        assert any("I3b_shapeB.jsonl" in x for x in r["disqualifications"]), r
        d = _fixture(td, ok, running=None)
        assert decide(d)["mode"] == "FALLBACK", decide(d)
        d = _fixture(td, ok, running="31")          # disagrees with the log
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        assert any(RUNNING_FILE in x and "not one run" in x
                   for x in r["disqualifications"]), r
        d = _fixture(td, ok, offsets=False)
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        assert any(OFFSETS_FILE in x for x in r["disqualifications"]), r
        d = _fixture(td, ok, log=False)
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        assert any(SERVER_LOG in x for x in r["disqualifications"]), r
        d = _fixture(td, {"I3a_shapeA": W_OK_A,
                          "I3b_shapeB": (W_OK_B[0], LIVE_LOG_LINES + 10)})
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        assert any("past the end of the log" in x for x in r["disqualifications"]), r
        d = _fixture(td, {"I3a_shapeA": W_OK_A})    # cell B has no interval
        r = decide(d)
        assert r["mode"] == "FALLBACK", r
        assert any("no interval recorded" in x for x in r["disqualifications"]), r
        d = _fixture(td, ok, thr_a=0.0)
        assert decide(d)["mode"] == "FALLBACK", decide(d)
        d = _fixture(td, ok, rate=4.0)       # a finite rate is not the I3 protocol
        assert decide(d)["mode"] == "FALLBACK", decide(d)

    print("LAMBDA_INF PREDICATE SELFTEST OK (F1: condition (b) recomputed PER "
          "CELL from the producer's own I_log_offsets.txt + srv_warmup.log under "
          "all %d registered conventions, checked against the LIVE job "
          "%s -- I3a=%d, I3b=%d, so the live branch is FALLBACK; the interval is "
          "start-exclusive/end-inclusive and shown load bearing on a real "
          "boundary line; lines are counted as `wc -l` does; each condition "
          "shown to force FALLBACK alone; the resolved absolute dir and a "
          "content digest per input are recorded [E6])"
          % (len(CONVENTIONS), live_job_dir().name,
             LIVE_RECOUNT["I3a_shapeA"][1], LIVE_RECOUNT["I3b_shapeB"][1]))


def _print_recount(instr_dir) -> int:
    rc = recount(instr_dir)
    print(f"# per-cell #running-req recount (F1) -- {rc['offsets_path']}")
    print(f"# log: {rc['server_log_path']} "
          f"({rc.get('server_log_lines')} lines, `wc -l` convention), "
          f"whole-log max {rc.get('server_log_global_max')}")
    print(f"# interval: {rc['interval_convention']}   judged: "
          f"{','.join(rc['judged_cells'])}   threshold {MIN_RUNNING_REQ}")
    for err in rc["errors"]:
        print(f"# ERROR {err}")
    hdr = ("cell", "interval(1-based)", *CONVENTIONS, "judged")
    print("%-14s %-18s %11s %11s %19s  %s" % hdr)
    for stem in sorted(rc["cells"]):
        m = rc["cells"][stem]
        if m.get("error"):
            print("%-14s %s" % (stem, m["error"]))
            continue
        print("%-14s %-18s %11s %11s %19s  %s"
              % (stem, "%d..%d" % tuple(m["interval_1based"]),
                 m["all_lines"], m["decode_only"], m["drop_prev_cell_tail"],
                 "YES" if stem in rc["judged_cells"] else "reported only"))
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--recount":
        return _print_recount(sys.argv[2])
    if len(sys.argv) != 2:
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
    # ★F1: the per-cell numbers the decision was actually made on, so the run
    # record carries them even if the log is later rotated away.
    rc = r["running_req_recount"]
    for stem in sorted(rc.get("cells", {})):
        m = rc["cells"][stem]
        judged = "judged" if stem in rc["judged_cells"] else "reported"
        if m.get("error"):
            print("LAMBDA0_RUNNING_REQ_%s=ERROR %s (%s)" % (stem, m["error"], judged))
        else:
            print("LAMBDA0_RUNNING_REQ_%s=%s lines=%d..%d interval_lines=%d (%s)"
                  % (stem, "/".join(str(m[c]) for c in CONVENTIONS),
                     m["interval_1based"][0], m["interval_1based"][1],
                     m["lines_in_interval"], judged))
    print("LAMBDA0_RUNNING_REQ_FILE_VALUE=%s (provenance only, never the "
          "decision)" % r["running_req_file_value"])
    # The machine-readable twin travels with the cells so the label JSON can
    # embed it (E6).
    return 0


if __name__ == "__main__":
    sys.exit(main())
