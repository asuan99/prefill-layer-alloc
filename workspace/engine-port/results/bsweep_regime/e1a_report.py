#!/usr/bin/env python3
"""Pretty-print e1a_preanalysis_2026-08-14.json (T1-T4)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "e1a_preanalysis_2026-08-14.json")))


def f(x, w=7, p=4):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


for pair in ("44->92", "16->44", "16->92"):
    print(f"\n===== T1  eps and S(B), SM pair {pair}  (rep-level, realized SM & B matched) =====")
    print(f"{'B':>3} {'dw2':>6} {'padB':>4} | {'eps_Ha8':>8} {'nrep':>7} | {'eps_T8':>8} {'nrep':>7} |"
          f" {'S':>8} {'S_se':>7} {'df':>6} {'ci95_lo':>8} {'ci95_hi':>8} | {'S_pooled':>8} {'S_p95':>7}")
    for r in d["T1_S_of_B"][pair]:
        H, T = r.get("Ha8"), r.get("T8")
        if not H or not T:
            miss = "Ha8" if not H else "T8"
            print(f"{r['B']:>3} {r['dw_kappa2']:6.2f} {r['B_padded']:>4} | "
                  f"UNSCOREABLE (no {miss} cell at one/both SM)")
            continue
        ci = r.get("S_ci95") or [None, None]
        print(f"{r['B']:>3} {r['dw_kappa2']:6.2f} {r['B_padded']:>4} |"
              f" {f(H['eps'],8)} {H['n_rep_lo']:>3}/{H['n_rep_hi']:<3} |"
              f" {f(T['eps'],8)} {T['n_rep_lo']:>3}/{T['n_rep_hi']:<3} |"
              f" {f(r['S'],8)} {f(r.get('S_se'))} {f(r.get('S_df'),6,2)}"
              f" {f(ci[0],8)} {f(ci[1],8)} | {f(r.get('S_pooled_auditor_method'),8)}"
              f" {f(r.get('S_p95'))}"
              + ("" if r.get("S_scoreable") else "   <- UNSCOREABLE(no SE)"))

print("\n===== T2  regression S = k + b*dw =====")
for key, v in d["T2_regression"].items():
    print(f"\n--- {key}")
    if "status" in v:
        print("   ", v["status"], v.get("B_scoreable", ""))
        continue
    print("    B points:", v["B_points"])
    print("    x(dw)   :", [round(x, 2) for x in v["x"]])
    print("    S       :", [round(y, 4) for y in v["S"]])
    print("    S_se    :", [round(s, 4) for s in v["S_se"]])
    for name in ("OLS_unweighted", "WLS_by_measurement_se"):
        o = v[name]
        print(f"    [{name}] n={o['n_points']} k={o['k']:+.5f} b={o['b']:+.6f}")
        if "k_ci95_resid" in o:
            print(f"        residual-scale t-CI(df={o['df_resid']}): "
                  f"k in [{o['k_ci95_resid'][0]:+.5f},{o['k_ci95_resid'][1]:+.5f}] (t={o['t_k']:+.2f}, p={o['p_k']:.4f}) ; "
                  f"b in [{o['b_ci95_resid'][0]:+.6f},{o['b_ci95_resid'][1]:+.6f}] (t={o['t_b']:+.2f}, p={o['p_b']:.4f})")
        if "k_ci95_known" in o:
            print(f"        known-variance CI: k in [{o['k_ci95_known'][0]:+.5f},{o['k_ci95_known'][1]:+.5f}] (z={o['z_k']:+.2f}) ; "
                  f"b in [{o['b_ci95_known'][0]:+.6f},{o['b_ci95_known'][1]:+.6f}] (z={o['z_b']:+.2f})")

print("\n===== T2b threshold feasibility (prereg D1 R>=1.5) =====")
tf = d["T2_threshold_feasibility"]
print("  dw(B), kappa=2:", {k: round(v, 2) for k, v in tf["dw_kappa2"].items()})
print("  dw(B), kappa=1:", {k: round(v, 2) for k, v in tf["dw_kappa1"].items()})
for k, v in tf["R_max_if_k_is_0"].items():
    print(f"  R_max(k=0) {k:<45} = {v:.3f}" + ("   <- clears 1.5" if v >= 1.5 else "   <- CANNOT clear 1.5"))
print("  prereg recheck:", {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in tf["prereg_dw_numbers_recheck"].items()})

print("\n===== T3  rep-level sd(log ITL p50)  [cells with n_rep>=3, B>=4] =====")
for pair, s in d["T3_sd_eps"].items():
    if not s:
        print(f"\n--- {pair}: UNSCOREABLE (no cell with n_rep>=3 on both SM levels)")
        continue
    print(f"\n--- {pair}")
    for arm in ("Ha8", "T8"):
        if arm not in s:
            print(f"    {arm}: UNSCOREABLE"); continue
        a = s[arm]
        for sm, vv in a["per_sm"].items():
            print(f"    {arm} SM{sm}: cells(B,n_rep,sd_log)={vv['cells']}  "
                  f"pooled sd_log={vv['pooled_sd_log_p50']:.4f} "
                  f"(within-job {vv['pooled_sd_log_p50_within_job'] if vv['pooled_sd_log_p50_within_job'] is None else round(vv['pooled_sd_log_p50_within_job'],4)})")
        print(f"    {arm} => sd_rep(eps) = {a['sd_rep_eps_all']:.4f}"
              f"  (within-job {a['sd_rep_eps_within_job'] if a['sd_rep_eps_within_job'] is None else round(a['sd_rep_eps_within_job'],4)})")
    if "sd_rep_S_all" in s:
        print(f"    => sd_rep(S) = {s['sd_rep_S_all']:.4f}"
              f"  (within-job {s.get('sd_rep_S_within_job') if s.get('sd_rep_S_within_job') is None else round(s['sd_rep_S_within_job'],4)})")

print("\n===== T3b  MDE for b (power 0.80, two-sided 0.05) =====")
for pair, m in d["T3_MDE"].items():
    print(f"\n--- {pair}")
    for design, mm in m.items():
        if mm is None:
            continue
        print(f"  design {design}: B={mm['B_levels']} dw={[round(x,2) for x in mm['dw']]} "
              f"span={mm['dw_span']:.2f} sd_rep(S)={mm['sd_rep_S_used']:.4f}")
        for r in mm["rows"]:
            print(f"      n={r['n_rep_per_cell']}  {r['df_model']:<38} se(b)={r['se_b']:.5f}"
                  f"  MDE(b)={r['mde_b']:.5f}  == detectable S change over dw span: {r['mde_S_change_over_dw_span']:.4f}")

print("\n===== T4  reachability =====")
t4 = d["T4_reachability"]
print("  max B in matched intervals      :", t4["max_B_anywhere_in_matched_intervals"])
print("  max B in decode-active snapshots:", t4["max_B_in_decode_active_snapshots"])
print("  engine caps:", t4["engine_caps"])
print("  B>=24 / B>=40 presence:", t4["B24_or_B40_present"])
print("\n  realized-SM-conditioned B histogram (Ha8/T8, SM 16/44/92):")
for r in t4["realized_sm_conditioned_B_hist"]:
    print(f"    {r['arm']:>4} cell={r['cell']:<4} job={r['job']} realized_sm={r['realized_sm']:>3}"
          f" n_snap={r['n_snap']:>5} maxB={r['max_B']:>3} medB={r['median_B']:>3}"
          f" n(B>=12)={r['n_B_ge_12']:>5} ({100*r['frac_B_ge_12']:.1f}%)")
print("\n  realized-pin audit (decode-active snapshots by realized SM):")
for r in t4["realized_pin_sm_hist"]:
    tot = r["n_snap"]
    frac = {k: f"{100*v/tot:.1f}%" for k, v in r["sm_hist"].items()}
    print(f"    {r['arm']:>4} cell={r['cell']:<4} job={r['job']} n={tot:>5} -> {frac}")

print("\n===== SC1 producer identity =====")
print("  ", d["SC1_pooled_reproduces_s8_batch_matched"])
