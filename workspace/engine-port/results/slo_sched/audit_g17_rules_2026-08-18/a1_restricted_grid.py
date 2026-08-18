#!/usr/bin/env python3
"""A1 -- does the G17 S2 grid (U = {d44,d54,d64,d74}) re-create a FORCED cell?

Read-only.  Inputs: ../audit_g16_results_2026-08-17/indep_perboot.json
(the audit's INDEPENDENT per-boot re-derivation of sec4's M_ttft / M_itl;
it reproduces campaign.arm_means to 4 dp, so it is a legitimate stand-in).

Question: G17 S2 proposes to run only U = {d44,d54,d64,d74} x sticky{ON,OFF}.
On that grid D_ttft = 44 = S_min(U).  The G16 prereg sec3 table pre-declares
that cell as TRUNCATED_LOW: "D_ttft = S_min  =>  Delta >= 0 FORCED
(positive inflation)".  This script executes that check on real data.
"""
import json, os, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "audit_g16_results_2026-08-17", "indep_perboot.json")
D = json.load(open(SRC))
BLOCKS = ["blk1", "blk2", "blk3", "blk4"]
SM = lambda a: int(a[1:])


def arm_means(phase, arms, blocks=BLOCKS):
    out = {}
    for a in arms:
        out[a] = {
            "M_itl": statistics.fmean(D[phase][a][b]["M_itl"] for b in blocks),
            "M_ttft": statistics.fmean(D[phase][a][b]["M_ttft"] for b in blocks),
        }
    return out


def decide(mu, arms):
    d_ttft = min(arms, key=lambda a: mu[a]["M_ttft"])
    d_itl = min(arms, key=lambda a: mu[a]["M_itl"])
    return d_ttft, d_itl


def exact_bands(mu, arms):
    """S_itl(slo) = min decode SM whose mean M_itl <= slo.  Returns the exact
    breakpoint list [(lo, hi, S_itl)] with hi=None for the open top band."""
    order = sorted(arms, key=SM)          # ascending SM
    pts = sorted({mu[a]["M_itl"] for a in arms})
    bands, prev_s = [], None
    for i, thr in enumerate(pts):
        s = None
        for a in order:                    # smallest SM that qualifies
            if mu[a]["M_itl"] <= thr + 1e-12:
                s = SM(a)
                break
        hi = pts[i + 1] if i + 1 < len(pts) else None
        if s != prev_s:
            bands.append([thr, hi, s])
        else:
            bands[-1][1] = hi
        prev_s = s
    return bands


def report(phase, arms, tag):
    mu = arm_means(phase, arms)
    d_ttft, d_itl = decide(mu, arms)
    smin, smax = min(SM(a) for a in arms), max(SM(a) for a in arms)
    forced = ("TRUNCATED_LOW (Delta>=0 FORCED)" if SM(d_ttft) == smin else
              "TRUNCATED (Delta<=0 side)" if SM(d_ttft) == smax else "interior")
    bands = exact_bands(mu, arms)
    print(f"\n=== {tag} | phase {phase} | arms {[a for a in sorted(arms,key=SM)]}")
    print(f"  mean M_itl : " + "  ".join(f"{a}={mu[a]['M_itl']:.3f}" for a in sorted(arms, key=SM)))
    print(f"  mean M_ttft: " + "  ".join(f"{a}={mu[a]['M_ttft']:.1f}" for a in sorted(arms, key=SM)))
    print(f"  D_ttft={d_ttft} (S_min={smin}, S_max={smax})  ->  forced-cell: {forced}")
    print(f"  D_itl(bare argmin)={d_itl}   Delta = {SM(d_itl)-SM(d_ttft):+d}")
    print("  exact S_itl bands (slo ms -> S_itl, Delta_SLO):")
    for lo, hi, s in bands:
        hs = f"{hi:.3f}" if hi is not None else "inf"
        print(f"    [{lo:.3f}, {hs})  S_itl={s:3d}  Delta_SLO={s-SM(d_ttft):+d}"
              f"   width={'inf' if hi is None else f'{hi-lo:.3f}'} ms")
    # operating point 60 ms
    s60 = min((SM(a) for a in arms if mu[a]["M_itl"] <= 60.0), default=None)
    print(f"  60 ms operating point: S_itl={s60}  Delta_SLO="
          f"{(s60-SM(d_ttft)) if s60 is not None else None}")
    return mu, d_ttft


def bootstrap_sign(phase, arms, n=20000, seed=1):
    """Block bootstrap of the reaching-edge band's Delta_SLO ( = D_itl - D_ttft )."""
    rnd = random.Random(seed)
    cnt, pos, zero, neg = {}, 0, 0, 0
    for _ in range(n):
        bs = [rnd.choice(BLOCKS) for _ in BLOCKS]
        mu = arm_means(phase, arms, bs)
        dt, di = decide(mu, arms)
        d = SM(di) - SM(dt)
        cnt[di] = cnt.get(di, 0) + 1
        pos += d > 0; zero += d == 0; neg += d < 0
    dist = {k: v / n for k, v in sorted(cnt.items(), key=lambda kv: SM(kv[0]))}
    return dist, pos / n, zero / n, neg / n


FULL = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
U = ["d44", "d54", "d64", "d74"]
U_PLUS_D34 = ["d34"] + U

for phase in ("HI", "LO"):
    for arms, tag in ((FULL, "G16 full 7-arm grid"), (U, "G17 S2 grid (U only)"),
                      (U_PLUS_D34, "U + d34 (repair option)")):
        report(phase, arms, tag)
        dist, p, z, nn = bootstrap_sign(phase, arms)
        print(f"  block-bootstrap D_itl distribution: "
              + " ".join(f"{k}={v:.4f}" for k, v in dist.items()))
        print(f"  P(Delta_SLO>0)={p:.4f}  P(=0)={z:.4f}  P(<0)={nn:.4f}")
