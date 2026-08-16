import sys, math, random, bisect
sys.path.insert(0,'/tmp/claude-100018302/-scratch-ehmoon-whlee/92e9abf0-6135-4d93-9a31-3568e6f769ee/scratchpad')
from qdist import chi2q, fq
random.seed(7)
SJ=0.5688
def df_of(sj,sb,k,m):
    var=(2.0/m**2)*((m*sj**2+sb**2)**2/(k-1)+sb**4/(k*(m-1)))
    return max(1.0,2*sj**4/var)
def cover(k,m,sb,sj=SJ,N=40000):
    dfB,dfW=k-1,k*(m-1)
    dfd=df_of(sj,sb,k,m); mult=math.sqrt(dfd/chi2q(0.05,dfd))
    Fq=fq(0.05,dfB,dfW); chiW=chi2q(0.05,dfW)
    c1=c3=0; z=0
    for _ in range(N):
        MSB=(m*sj**2+sb**2)*random.gammavariate(dfB/2,2)/dfB
        MSW=sb**2*random.gammavariate(dfW/2,2)/dfW
        v=(MSB-MSW)/m
        if v<=0: v=0.0; z+=1
        s=math.sqrt(v)
        if s*mult>=sj: c1+=1
        th=((MSB/MSW)/Fq-1.0)/m
        if math.sqrt(max(0.0,th)*MSW*dfW/chiW)>=sj: c3+=1
    return c1/N,c3/N,z/N
print("nominal 95pct upper bound on sigma_job; TRUE sigma_job = {:.4f} pct".format(SJ))
for lab,k,m,sb in [("M8 k4m3",4,3,0.0607),("M8 k6m3",6,3,0.0607),
                   ("Ha8 k4m3",4,3,1.1924),("Ha8 k12m6",12,6,1.1924),
                   ("Ha8 k8m12",8,12,1.1924),("Ha8 240s k6m3",6,3,0.596)]:
    c1,c3,z=cover(k,m,sb)
    print(f"  {lab:14s} sb={sb:.4f}: coverage(doc Satterthwaite-UB)={c1:.3f}   coverage(exact-F+Bonf)={c3:.3f}   P(sigma_hat=0)={z:.3f}")
