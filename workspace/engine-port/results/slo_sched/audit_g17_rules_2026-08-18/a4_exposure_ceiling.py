#!/usr/bin/env python3
"""A4 -- what is the STRUCTURAL ceiling on the G17 P1 gate quantity?

P1 (design sec4/S1): "sticky ON must reach W_tim_all >= 0.60 (OFF measured 0.157)".
Under sticky ON the (0,108) unpartitioned time is converted to time at the
nominal split, but the decode-EMPTY time (stream idx 0, code comment at
multiplexing_mixin.py:913-923 says sticky deliberately does NOT hold there)
stays in the denominator of W_tim_all.

ceiling(W_tim_all) = (t_at_D + t_at_108) / t_total = 1 - t_decode_empty/t_total

Read-only input: ../residency_scope_2026-08-17/residency_scope_2026-08-17.json
(the artifact G16_RESULTS sec2.3 itself cites).
"""
import json, os, statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "..", "residency_scope_2026-08-17",
                                "residency_scope_2026-08-17.json")))
boots = [b for b in J["boots"] if b.get("mode") == "full"]
print(f"boots(full)={len(boots)}")

by = defaultdict(list)
for b in boots:
    D = b["decode_sm"]
    for scope in ("ALL", "HI", "LO"):
        if scope == "ALL":
            st, tot = b["states_time_s"], b["sec_total"]
            da = b["sec_decode_active"]
            w_all, w_da = b["W_tim_all"], b["W_tim_da"]
            cor = b["co_resident_frac"]
        else:
            p = b["per_phase"][scope]
            st, tot, da = p["states_time_s"], p["sec_total"], p["sec_decode_active"]
            w_all, w_da = p["W_tim_all"], p["W_tim_da"]
            cor = None
        tD = st.get(str(D), 0.0)
        t108 = st.get("108", 0.0)
        t0 = st.get("0", 0.0)
        by[(b["arm"], scope)].append(dict(
            w_all=w_all, w_da=w_da, ceil_all=(tD + t108) / tot,
            idle_frac=t0 / tot, tD=tD, t108=t108, tot=tot,
            amp=(tD + t108) / tD if tD > 0 else float("inf"), cor=cor))

hdr = f"{'arm':>5} {'scope':>4} {'w_all':>7} {'w_da':>7} {'ceilingON(all)':>14} {'idle':>7} {'max amp':>8} {'co_res':>7}"
print(hdr)
for scope in ("ALL", "HI", "LO"):
    for arm in ("d16", "d24", "d34", "d44", "d54", "d64", "d74"):
        v = by[(arm, scope)]
        if not v:
            continue
        m = lambda k: statistics.fmean(x[k] for x in v)
        cor = [x["cor"] for x in v if x["cor"] is not None]
        print(f"{arm:>5} {scope:>4} {m('w_all'):7.4f} {m('w_da'):7.4f} "
              f"{m('ceil_all'):14.4f} {m('idle_frac'):7.4f} {m('amp'):8.2f} "
              f"{(statistics.fmean(cor) if cor else float('nan')):7.4f}")
    print()

print("READINGS")
u = ["d44", "d54", "d64", "d74"]
for scope in ("ALL", "HI", "LO"):
    ce = statistics.fmean(statistics.fmean(x["ceil_all"] for x in by[(a, scope)]) for a in u)
    wo = statistics.fmean(statistics.fmean(x["w_all"] for x in by[(a, scope)]) for a in u)
    wda = statistics.fmean(statistics.fmean(x["w_da"] for x in by[(a, scope)]) for a in u)
    print(f"  U mean, scope={scope}: w_all_OFF={wo:.4f}  ceiling_ON(all)={ce:.4f}  "
          f"max amplification (all-denominator) = {ce/wo:.2f}x ; "
          f"w_da_OFF={wda:.4f} -> ceiling 1.0 => amp {1/wda:.2f}x")
