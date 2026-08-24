#!/usr/bin/env python3
"""NSL P0 step (1): register the estimator, bracket the cap predicate.  GPU 0.

Implements PREREG_NSL_P0_REANALYSIS_2026-08-24.md exactly.  This produces NO
verdict about whether the cap binds -- audit B1 invalidated that predicate and
only an engine patch repairs it.  What it produces is (a) the registered
time-weighted estimator, (b) a reproducible artefact, and (c) the width of the
bracket L <= T <= U, which decides whether the B1 engine patch is REQUIRED.
"""
import glob, hashlib, json, os, re, statistics, sys

X = 0.05                      # the threshold P0 registered
PAT = "../slo_sched/g16_blk*_d44_boot1_*_telemetry.jsonl"


def realized_cap(tel_path):
    """Audit B7 / memory lesson: read what the ENGINE settled on, not the CLI."""
    srv = tel_path.replace("_telemetry.jsonl", "_srv.log")
    if os.path.exists(srv):
        with open(srv, errors="ignore") as f:
            m = re.findall(r"max_running_requests=(\d+)", f.read())
        if m:
            return int(m[-1]), "server_banner"
    return None, "absent"


def analyse(path):
    snaps = []
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") != "runtime_snapshot":
            continue
        t = d.get("timestamp_monotonic_s")
        if t is None:
            continue
        snaps.append((t, d.get("decode_running_batch_size"), d.get("prefill_queue_depth"),
                      d.get("sample_index")))
    snaps.sort(key=lambda r: r[0])
    if len(snaps) < 2:
        return None

    cap, cap_src = realized_cap(path)
    obs_max = max((r[1] or 0) for r in snaps)
    if cap is None:
        cap, cap_src = obs_max, "observed_max"

    # time weight = gap to the NEXT snapshot; last snapshot weight 0 (prereg sec2)
    gaps = [snaps[i + 1][0] - snaps[i][0] for i in range(len(snaps) - 1)] + [0.0]
    total_t = sum(gaps)

    def frac(pred):
        tw = sum(g for (t, bs, qd, si), g in zip(snaps, gaps) if pred(bs, qd))
        cw = sum(1 for (t, bs, qd, si) in snaps if pred(bs, qd))
        return tw / total_t if total_t else float("nan"), cw / len(snaps)

    L_tw, L_cw = frac(lambda bs, qd: (bs or 0) >= cap)
    U_tw, U_cw = frac(lambda bs, qd: (bs or 0) + (qd or 0) >= cap)

    pos = [g for g in gaps[:-1] if g > 0]
    strides = sorted({snaps[i + 1][3] - snaps[i][3]
                      for i in range(len(snaps) - 1)
                      if snaps[i][3] is not None and snaps[i + 1][3] is not None})
    return {
        "file": os.path.basename(path),
        "n_snapshots": len(snaps), "total_time_s": round(total_t, 3),
        "cap": cap, "cap_source": cap_src, "observed_max_running_bs": obs_max,
        "sample_index_strides": strides[:6],
        "gap_ms": {"p50": round(statistics.median(pos) * 1e3, 3),
                   "p99": round(sorted(pos)[int(len(pos) * .99)] * 1e3, 3),
                   "max": round(max(pos) * 1e3, 3)} if pos else None,
        "L_time_weighted": round(L_tw, 5), "U_time_weighted": round(U_tw, 5),
        "L_count_weighted": round(L_cw, 5), "U_count_weighted": round(U_cw, 5),
        "bracket_width_tw": round(U_tw - L_tw, 5),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, PAT)))
    boots = [r for r in (analyse(f) for f in files) if r]
    if not boots:
        print("no data"); return 1

    # the decision rule, registered in prereg sec3 BEFORE these numbers existed
    straddle = [b["file"] for b in boots
                if b["L_time_weighted"] < X <= b["U_time_weighted"]]
    label = "BRACKET_STRADDLES" if straddle else "BRACKET_DECIDES"
    # estimator sensitivity (prereg sec3, last paragraph)
    def side(v):
        return "below" if v < X else "at_or_above"
    flip = [b["file"] for b in boots
            if side(b["L_time_weighted"]) != side(b["L_count_weighted"])
            or side(b["U_time_weighted"]) != side(b["U_count_weighted"])]

    out = {"date": "2026-08-24", "gpu_hr": 0.0, "X": X,
           "prereg": "PREREG_NSL_P0_REANALYSIS_2026-08-24.md",
           "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "label": label, "straddling_boots": straddle,
           "estimator_flips_side": flip, "boots": boots,
           "NOT_A_VERDICT": ("audit B1 invalidated the point predicate; these numbers "
                             "are NOT an answer to whether the cap binds")}
    p = os.path.join(here, "nsl_p0_reanalysis_2026-08-24.json")
    json.dump(out, open(p, "w"), indent=1)
    for b in boots:
        print(f"  {b['file'][:26]}  cap={b['cap']}({b['cap_source']})  n={b['n_snapshots']:>6}  "
              f"L_tw={b['L_time_weighted']:.5f}  U_tw={b['U_time_weighted']:.5f}  "
              f"| L_cw={b['L_count_weighted']:.5f}  U_cw={b['U_count_weighted']:.5f}")
    print(f"\n  X={X}  ->  {label}   straddling={straddle or 'none'}")
    print(f"  estimator flips a side on: {flip or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
