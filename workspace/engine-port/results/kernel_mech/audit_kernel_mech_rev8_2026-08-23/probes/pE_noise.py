"""Probe E -- (1) are the three sec1.4 diagnostics distinguishable from their rev7
values, or is the 'rev7 -> rev8' column pure Monte-Carlo drift?
(2) how stable is the 4-boot CV estimate that sec3.5-18 branches on?"""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

print("== (1a) invalid_window_rate_worst_truth: shipped rev7=0.0875 rev8=0.0500 ==")
v = []
for s in range(200):
    rng = np.random.default_rng(3000+s)
    v.append(R.invalid_rate(R._cells_for(rng, R.TRUTHS["kernel_does_not_shrink"], 0.05)["hi"]))
v = np.array(v)
print(f"   over 200 independent draws of the SAME single cell (n=80 windows):")
print(f"   mean={v.mean():.4f} sd={v.std(ddof=1):.4f}  p05={np.quantile(v,.05):.4f} p95={np.quantile(v,.95):.4f}")
print(f"   => both 0.0875 and 0.0500 are ordinary draws; the design reports a single n=80 draw with no CI.")

print("\n== (1b) effective_noise_ratio: shipped rev7=2.599 rev8=2.781 ==")
r = []
for s in range(200):
    rng = np.random.default_rng(4000+s)
    a = R._cells_for(rng, R.TRUTHS["mixed"], 0.025, derive_g=True)["hi"]["Gi"]
    b = R._cells_for(rng, R.TRUTHS["mixed"], 0.025, derive_g=False)["hi"]["Gi"]
    r.append((a.std()/a.mean())/(b.std()/b.mean()))
r = np.array(r)
print(f"   mean={r.mean():.3f} sd={r.std(ddof=1):.3f} p05={np.quantile(r,.05):.3f} p95={np.quantile(r,.95):.3f}")

print("\n== (2) stability of the 4-boot CV that sec3.5-18 branches on ==")
print("   truth                cv_boot | P(CV<=0.025) P(0.025<CV<=0.05) P(CV>0.05)  <- registered 3-way branch")
for name, truth in R.TRUTHS.items():
    for cv in (0.01, 0.025):
        vals = []
        for rep in range(400):
            rng = np.random.default_rng(7000+rep)
            c = R._cells_for(rng, truth, cv)["hi"]
            pb = c["Gi"].sum(axis=1)/c["T"].sum(axis=1)
            vals.append(pb.std(ddof=1)/pb.mean())
        vals = np.array(vals)
        print(f"   {name:22s} {cv:5.3f} |  {np.mean(vals<=.025):.3f}        {np.mean((vals>.025)&(vals<=.05)):.3f}            {np.mean(vals>.05):.3f}")
