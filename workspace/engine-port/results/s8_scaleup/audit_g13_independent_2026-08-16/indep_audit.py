# independent reimplementation -- does NOT import design_g13_stats.py
import json, math, re, statistics as st
H="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_scaleup/"
d=json.load(open(H+"C2R_RESULTS_2026-08-16.json"))
for arm in ("M8","Ha8"):
    c16=d["cells"][f"{arm}/d16/SM16/b16"]["per_boot_median"]
    c92=d["cells"][f"{arm}/d92/SM92/b16"]["per_boot_median"]
    g=lambda k:int(re.search(r"blk(\d+)$",k).group(1))
    a={g(k):v for k,v in c16.items()}; b={g(k):v for k,v in c92.items()}
    bl=sorted(set(a)&set(b))
    v16=[a[x] for x in bl]; v92=[b[x] for x in bl]
    rp=[a[x]/b[x] for x in bl]
    rel=lambda xs: st.stdev(xs)/st.fmean(xs)*100
    r_unp=st.fmean(v16)/st.fmean(v92)
    sd_unp=math.hypot(rel(v16),rel(v92))
    sd_paired=rel(rp)
    # correlation between legs' per-boot relative deviations
    m1,m2=st.fmean(v16),st.fmean(v92)
    x=[(u/m1-1) for u in v16]; y=[(u/m2-1) for u in v92]
    cov=sum(p*q for p,q in zip(x,y))/(len(x)-1)
    rho=cov/(st.stdev(x)*st.stdev(y))
    print(f"{arm}: blocks={bl}")
    print(f"   leg16 relSD={rel(v16):.4f}%  leg92 relSD={rel(v92):.4f}%")
    print(f"   r_unpaired={r_unp:.6f}  r_paired_mean={st.fmean(rp):.6f}  pooled={d['ratios'][arm]['r']:.6f}")
    print(f"   sd_unpaired(hypot)={sd_unp:.4f}%   sd_PAIRED(actual r per boot-pair)={sd_paired:.4f}%   ratio={sd_paired/sd_unp:.3f}")
    print(f"   corr(leg16,leg92) across blocks = {rho:+.3f}")
    print(f"   per-block r = {[round(z,4) for z in rp]}")
