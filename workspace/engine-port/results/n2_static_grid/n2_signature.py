#!/usr/bin/env python3
"""N2(3-a) co-load non-joining signature S1-S3 and the Job label COLOAD_* (PREREG_N2_3A rev16 §7-2 (f); §15
`n2_signature.py`). CPU only. HARNESS-SEAL file -- Job-EA ONLY (Job-PROFILE has no co-load; nothing here runs in a
Job-PROFILE analysis). Implemented because the rule is fully registered and pure; the Job-EA end-to-end path that
feeds it (cell client with co-load, rho_off assembly H15-3) is NOT implemented yet.

S1 (R3-5): for every co span, the first decode_iteration launched after its prefill_span_end: bs > running_bs of
    that span_end => JOINED evidence; bs < running_bs => the b requests' own finish (NOT joined; B_FINISHED is §9-1's);
    bs == running_bs => no evidence.
S2: number of decode_iteration with bs > b inside W_co and the first 128 iterations of A2 == 0.
S3 (positive control, report only): for co spans k >= 2 the prefill_span_start.running_bs is in {b, b + n_done(k-1)};
    majority b + n_done(k-1) => "merge then filter observed"; always b => reading-error signal (report only).
Replicate signature: any S1 JOINED evidence or S2 > 0 => JOINED, else NOT_JOINED.
Job label (D-4 (ii)): population M = E-alpha replicates past the BENCH/PAYLOAD/RETRACT/RECORDER/CG stages with >= 1
co span in W_co, regardless of signature / predicate; k = #JOINED, n = |M|: n = 0 -> COLOAD_UNRESOLVED; k = 0 ->
COLOAD_NOT_JOINED; else COLOAD_JOINED_<k>_of_<n>.
"""
from __future__ import annotations

CO_INPUT = 8192


def co_spans(records, t_co):
    starts = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_start"}
    ends = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_end"}
    out = [(starts[k], ends[k]) for k in starts if k in ends
           and starts[k]["timestamp_monotonic_s"] >= t_co and starts[k].get("n_new", 0) >= 1
           and starts[k].get("extend_tokens") == starts[k]["n_new"] * CO_INPUT]
    return sorted(out, key=lambda se: se[0]["timestamp_monotonic_s"])


def signature(records, t_co, b, w_co, a2):
    iters = sorted((e for e in records if e.get("event") == "decode_iteration"), key=lambda e: e["t_launch"])
    spans = co_spans(records, t_co)
    s1 = []
    for st, en in spans:
        nxt = next((e for e in iters if e["t_launch"] > en["timestamp_monotonic_s"]), None)
        if nxt is None:
            s1.append("NO_ITERATION")
        elif nxt["bs"] > en["running_bs"]:
            s1.append("JOINED_EVIDENCE")
        elif nxt["bs"] < en["running_bs"]:
            s1.append("B_FINISH")
        else:
            s1.append("NO_EVIDENCE")
    s2 = sum(1 for e in list(w_co) + list(a2[:128]) if e["bs"] > b)
    s3 = []
    for k in range(1, len(spans)):
        prev_done = spans[k - 1][1].get("n_done", 0)
        rb = spans[k][0].get("running_bs")
        s3.append("MERGED" if rb == b + prev_done else ("B" if rb == b else "OTHER"))
    merged = sum(1 for x in s3 if x == "MERGED")
    s3_verdict = ("NA" if not s3 else "MERGE_THEN_FILTER_OBSERVED" if merged * 2 > len(s3)
                  else "ALWAYS_B_READING_ERROR_SIGNAL" if all(x == "B" for x in s3) else "MIXED")
    joined = any(x == "JOINED_EVIDENCE" for x in s1) or s2 > 0
    return dict(signature="JOINED" if joined else "NOT_JOINED", s1=s1, s2_bs_gt_b=s2, s3=s3, s3_verdict=s3_verdict,
                n_co_spans=len(spans))


def job_label(population):
    """population: signature dicts of the measurement-established E-alpha replicates (M, D-4 (ii))."""
    n = len(population)
    k = sum(1 for s in population if s["signature"] == "JOINED")
    if n == 0:
        return "COLOAD_UNRESOLVED", k, n
    if k == 0:
        return "COLOAD_NOT_JOINED", k, n
    return f"COLOAD_JOINED_{k}_of_{n}", k, n
