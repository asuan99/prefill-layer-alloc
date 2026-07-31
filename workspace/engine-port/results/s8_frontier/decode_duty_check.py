#!/usr/bin/env python3
"""Decode-side realized-partition duty-cycle gate (s8_frontier / E1).

★NEW 2026-07-31, from claims-auditor's audit of job 867231.

WHY THIS EXISTS
---------------
`e1_pin_check.compute_time_weighted_pin_gate` answers "did PREFILL work run
on the cell's target `prefill_sms`?" -- the prefill side only. The auditor
showed the decode side has an independent, larger problem that the prefill
gate is structurally blind to.

Time-weighting `decode_sms` over decode-active time in 867231's telemetry,
the fraction of decode-active time actually spent at the cell's LABELLED
decode SM count is:

    d16 0.110 | d24 0.112 | d44 0.166 | d54 0.201 | d92 0.518

The remainder runs unpartitioned at 108 SM (`CONSENSUS.md` §1-22's
auto-revert: green-context splitting only holds while split-prefill is
co-resident; when prefill drains, the runtime reverts to no split).

The consequence is not a reporting nicety. **The D axis moves two variables
at once**: how many SMs decode is limited to, AND what fraction of the time
that limit is in force -- and the second covaries monotonically with the
first (0.110 -> 0.518 across the grid). A D-comparison therefore cannot be
read as "the effect of giving decode more SMs" without this number beside
it. `D=16 (P92) pin_frac=0.950` looks reassuring but is a statement about
prefill sitting at 92 SM; it says nothing about decode sitting at 16.

This is the same label-vs-realized structure as Stage 0's D108 anchor
failure (`PROJECT_STATUS.md`, C1 CONFIRMED) -- there the cause was a config
bug, here it is the policy behaving as designed -- and the same
aggregation-unit discipline (methodology gate #4) applies: a gate ("did it
run in that partition?") is a TIME-weighted question, never a count.

ESTIMATOR
---------
Deliberately identical in structure to the prefill gate, so the two are
directly comparable and neither can be accused of a bespoke estimator:

  - population: every `runtime_snapshot` row (phase == "benchmark") with
    `decode_running_batch_size > 0`, weighted by the interval to the next
    row (same weight-to-next-row convention as
    `compute_concurrency_diagnostic`).
  - episodes: maximal contiguous runs of `decode_running_batch_size > 0`,
    regardless of which `decode_sms` they show (a mid-episode switch is an
    anomaly to surface, not to assume away).
  - CI: episode-level (cluster) bootstrap of the pooled ratio
    `sum(target_time) / sum(total_time)`, NOT Clopper-Pearson -- consecutive
    snapshots inside one episode are not iid Bernoulli trials.
  - `n_episodes` is the effective sample size; `n_episodes == 1` cannot be
    resampled and is reported as UNRELIABLE rather than given a
    confident-looking bound.

★KNOWN LIMITATION, stated up front: this reads the same `runtime_snapshot`
stream the prefill gate reads, so it inherits that stream's sampling
behaviour (`trace_every`, default 32). The auditor's 11-52% figures are a
first-order approximation for exactly this reason and the numbers themselves
are a re-measurement target -- an engine-side ACCUMULATED decode-time-per-
partition counter would settle them. What is NOT sensitive to the sampling
rate is the ORDERING and its monotone covariation with D, which is what
makes this a confound rather than a rounding error.

BATCH-CAP DIAGNOSTIC
--------------------
Also computes the decode batch-size histogram and the `kv_occupancy` at
which the batch truncates. This is the readout for the batch-cap experiment
(`batchcap.sbatch`): the auditor found `decode_running_batch_size` truncating
at exactly 48 in four of five cells -- the value of `--max-running-requests`,
a constant that appears NOWHERE in `DESIGN.md` -- while `kv_occupancy` at
that point was 0.024, i.e. not a resource limit at all. If the per-cell ITL
"ceilings" {d16 50.6, d24 37.9, d44 26.2, d54 23.9, d92 19.1 ms} move with
the cap, they are a property of the harness configuration, not of the model,
and the scope of "60 ms is out of reach for T8" changes accordingly.

Usage:
  decode_duty_check.py <telemetry.jsonl> <expect_d> [min_lower95]
                       [--t0 T] [--t1 T] [--json]
"""
import json
import sys

VALID_D = (16, 24, 44, 54, 92)


def compute_decode_duty_gate(telemetry_path, expect_d, t0=None, t1=None,
                             n_boot=10000, seed=20260731):
    """Mirror of e1_pin_check.compute_time_weighted_pin_gate on the decode
    axis. Returns the time-weighted fraction of decode-active time spent at
    `decode_sms == expect_d`, with an episode-cluster bootstrap lower bound,
    plus the realized-partition histogram and batch-cap diagnostics."""
    rows = []
    for line in open(telemetry_path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts = e.get("timestamp_monotonic_s")
        if ts is None:
            continue
        if t0 is not None and ts < t0:
            continue
        if t1 is not None and ts > t1:
            continue
        rows.append((ts, e.get("decode_sms"), e.get("decode_running_batch_size", 0),
                     e.get("kv_occupancy", 0.0), e.get("kv_total_occupancy", 0.0)))
    rows.sort(key=lambda r: r[0])

    episodes = []            # (t_total, t_target)
    realized_hist = {}       # {decode_sms: seconds}
    batch_hist = {}          # {decode_running_batch_size: seconds}
    kv_at_max_batch = []     # kv_occupancy samples at the observed max batch
    max_batch = 0
    cur_total = cur_target = 0.0
    in_episode = False

    for i, (ts, dsm, drb, kv, kvt) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else t1
        if nxt is None:
            continue
        dt = max(0.0, nxt - ts)
        if drb > 0:
            in_episode = True
            cur_total += dt
            if dsm == expect_d:
                cur_target += dt
            realized_hist[dsm] = realized_hist.get(dsm, 0.0) + dt
            batch_hist[drb] = batch_hist.get(drb, 0.0) + dt
            if drb > max_batch:
                max_batch = drb
                kv_at_max_batch = [(kv, kvt)]
            elif drb == max_batch:
                kv_at_max_batch.append((kv, kvt))
        else:
            if in_episode:
                episodes.append((cur_total, cur_target))
                cur_total = cur_target = 0.0
                in_episode = False
    if in_episode:
        episodes.append((cur_total, cur_target))

    n_episodes = len(episodes)
    t_total = sum(e[0] for e in episodes)
    t_target = sum(e[1] for e in episodes)
    duty_frac = (t_target / t_total) if t_total else float("nan")

    if n_episodes >= 2:
        import random as _random
        rng = _random.Random(seed)
        boots = []
        for _ in range(n_boot):
            sample = [episodes[rng.randrange(n_episodes)] for _ in range(n_episodes)]
            st_, sg_ = sum(e[0] for e in sample), sum(e[1] for e in sample)
            if st_ > 0:
                boots.append(sg_ / st_)
        boots.sort()
        lower95 = boots[int(0.025 * len(boots))] if boots else 0.0
        reliable = True
    elif n_episodes == 1:
        lower95 = duty_frac if duty_frac == duty_frac else 0.0
        reliable = False
    else:
        lower95 = 0.0
        reliable = False

    kv_mean = (sum(k for k, _ in kv_at_max_batch) / len(kv_at_max_batch)
               if kv_at_max_batch else float("nan"))
    kvt_mean = (sum(k for _, k in kv_at_max_batch) / len(kv_at_max_batch)
                if kv_at_max_batch else float("nan"))

    return dict(
        expect_d=expect_d, n_episodes=n_episodes,
        t_decode_active_total_s=t_total, t_target_s=t_target,
        duty_frac=duty_frac, duty_frac_lower95=lower95,
        bootstrap_reliable=reliable,
        realized_hist=realized_hist, batch_hist=batch_hist,
        max_batch=max_batch,
        kv_occupancy_at_max_batch=kv_mean,
        kv_total_occupancy_at_max_batch=kvt_mean,
        n_samples_at_max_batch=len(kv_at_max_batch),
    )


def decode_duty_verdict(g, min_lower95=0.80):
    """Deliberately NOT wired as cite-blocking yet. DESIGN.md pre-registered
    exactly two gates (realized prefill pin >= 0.80, partition activity
    >= 0.60); adding a third cite-blocking threshold after seeing 867231's
    data would be choosing a gate from the data -- the same violation the
    SLO-siting rules exist to prevent. This returns a verdict string so the
    number is reported and arguable, and the threshold question is escalated
    to the human, not silently enacted.

    On 867231's numbers a 0.80 threshold would fail every cell including
    d92 (max 0.518), which is itself the finding: no cell in that scan ran
    predominantly at its labelled decode partition."""
    if g["n_episodes"] == 0 or g["t_decode_active_total_s"] <= 0:
        return ("NO-DATA", "no decode-active time in this window (n_episodes=0)")
    note = "" if g["bootstrap_reliable"] else \
        f"  [UNRELIABLE: n_episodes={g['n_episodes']}, bootstrap CI not meaningful]"
    status = "ABOVE" if g["duty_frac_lower95"] >= min_lower95 else "BELOW"
    return (status,
            f"duty_frac={g['duty_frac']:.3f} lower95={g['duty_frac_lower95']:.3f} "
            f"vs reference {min_lower95} (n_episodes={g['n_episodes']})"
            f"  -- REPORTED, NOT CITE-BLOCKING (see decode_duty_verdict docstring)"
            + note)


def main():
    argv = list(sys.argv[1:])
    as_json = "--json" in argv
    if as_json:
        argv.remove("--json")
    t0 = t1 = None
    if "--t0" in argv:
        i = argv.index("--t0"); t0 = float(argv[i + 1]); del argv[i:i + 2]
    if "--t1" in argv:
        i = argv.index("--t1"); t1 = float(argv[i + 1]); del argv[i:i + 2]
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-3])
        return 2
    path, expect_d = argv[0], int(argv[1])
    min_lower95 = float(argv[2]) if len(argv) > 2 else 0.80
    if expect_d not in VALID_D:
        print(f"unknown D={expect_d}, not in {list(VALID_D)}")
        return 2

    g = compute_decode_duty_gate(path, expect_d, t0=t0, t1=t1)
    status, reason = decode_duty_verdict(g, min_lower95)
    if as_json:
        print(json.dumps({**g, "status": status, "reason": reason}, sort_keys=True))
        return 0

    print(f"DECODE_DUTY expect_d={expect_d}  {status}")
    print(f"  {reason}")
    print(f"  decode-active time {g['t_decode_active_total_s']:.3f}s, "
          f"at target {g['t_target_s']:.3f}s")
    tot = g["t_decode_active_total_s"] or 1.0
    print("  realized decode_sms (time-weighted):")
    for sm, sec in sorted(g["realized_hist"].items(),
                          key=lambda kv: -kv[1]):
        print(f"      decode_sms={sm}: {sec:8.3f}s ({sec / tot:6.1%})"
              + ("   <-- TARGET" if sm == expect_d else
                 "   <-- UNPARTITIONED (auto-revert, CONSENSUS 1-22)" if sm == 108 else ""))
    print(f"  max decode_running_batch_size = {g['max_batch']}"
          f"   (kv_occupancy there = {g['kv_occupancy_at_max_batch']:.4f}, "
          f"kv_total = {g['kv_total_occupancy_at_max_batch']:.4f}, "
          f"n={g['n_samples_at_max_batch']})")
    print("      -> if max_batch equals --max-running-requests while kv_occupancy "
          "is far below 1.0, the batch is CAPPED BY CONFIG, not by memory; the "
          "per-cell ITL ceiling is then a harness property (batchcap.sbatch tests this)")
    top = sorted(g["batch_hist"].items(), key=lambda kv: -kv[1])[:8]
    print("  decode batch-size occupancy (top 8, time-weighted):")
    for b, sec in top:
        print(f"      batch={b:4d}: {sec:8.3f}s ({sec / tot:6.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
