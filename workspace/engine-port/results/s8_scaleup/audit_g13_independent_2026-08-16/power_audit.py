import sys, math, random
sys.path.insert(0,'/tmp/claude-100018302/-scratch-ehmoon-whlee/92e9abf0-6135-4d93-9a31-3568e6f769ee/scratchpad')
from qdist import chi2q, fq

SJ = 0.5688257172317233          # doc prior sigma_job (%)
GAP = {"M8":5.118,"Ha8":9.196}
SB_DOC   = {"M8":0.06070258749223707,"Ha8":1.1924495168511018}   # doc: hypot of leg rel SDs
SB_PAIRED= {"M8":0.0371,"Ha8":1.5684}                            # actual per-boot-pair SD of r

def ub95_doc(sj,sb,k,m):
    var=(2.0/m**2)*((m*sj**2+sb**2)**2/(k-1)+sb**4/(k*(m-1)))
    df=max(1.0, 2*sj**4/var)
    return sj*math.sqrt(df/chi2q(0.05,df)), df

print("=== A. reproduce doc Table 4.2 (per-arm gap, doc sigma_boot) ===")
for arm,k,m in [("M8",3,3),("M8",4,3),("M8",6,3),("M8",8,3),
                ("Ha8",4,3),("Ha8",6,6),("Ha8",8,12),("Ha8",12,6)]:
    ub,df=ub95_doc(SJ,SB_DOC[arm],k,m)
    print(f"  {arm} k={k} m={m}: df_eff={df:.3f} UB95={ub:.3f}%  gap/UB={GAP[arm]/ub:.2f}  3sig_pass={GAP[arm]>=3*ub}")

print("\n=== B. same, but with the ACTUAL per-boot-pair SD of r (paired) ===")
for arm,k,m in [("M8",4,3),("M8",6,3),("Ha8",4,3),("Ha8",6,6),("Ha8",8,12),("Ha8",12,6)]:
    ub,df=ub95_doc(SJ,SB_PAIRED[arm],k,m)
    print(f"  {arm} k={k} m={m}: df_eff={df:.3f} UB95={ub:.3f}%  gap/UB={GAP[arm]/ub:.2f}  3sig_pass={GAP[arm]>=3*ub}")

print("\n=== C. doc section 4.3 window-scaling rows (sigma_boot assumed 1/sqrt(T)) ===")
for sb,k,m,lab in [(1.192,4,3,"60s"),(0.596,4,3,"240s"),(0.596,6,3,"240s k6"),(0.4216,4,3,"480s")]:
    ub,df=ub95_doc(SJ,sb,k,m)
    print(f"  Ha8 {lab}: sb={sb} df_eff={df:.3f} UB95={ub:.3f}% gap/UB={GAP['Ha8']/ub:.2f}")
print("   [same rows with paired-scaled sigma_boot 1.5684/sqrt(T/60)]")
for f,k,m,lab in [(1,4,3,"60s"),(2,4,3,"240s"),(2,6,3,"240s k6"),(math.sqrt(8),4,3,"480s")]:
    sb=1.5684/f
    ub,df=ub95_doc(SJ,sb,k,m)
    print(f"  Ha8 {lab}: sb={sb:.3f} df_eff={df:.3f} UB95={ub:.3f}% gap/UB={GAP['Ha8']/ub:.2f}")
