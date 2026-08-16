import sys, math, random, bisect
sys.path.insert(0,'/tmp/claude-100018302/-scratch-ehmoon-whlee/92e9abf0-6135-4d93-9a31-3568e6f769ee/scratchpad')
from qdist import chi2q, fq
random.seed(20260816)
SJ=0.5688; GAP={"M8":5.118,"Ha8":9.196}
SB_DOC={"M8":0.0607,"Ha8":1.1924}
# lookup table for chi2_{0.05,df}, df in [1,40]
GRID=[1.0+0.05*i for i in range(0,781)]
TAB=[chi2q(0.05,d) for d in GRID]
def chi2_lu(df):
    df=min(max(df,1.0),40.0); i=bisect.bisect_left(GRID,df)
    i=min(max(i,1),len(GRID)-1)
    x0,x1=GRID[i-1],GRID[i]; y0,y1=TAB[i-1],TAB[i]
    return y0+(y1-y0)*(df-x0)/(x1-x0)
def df_of(sj,sb,k,m):
    var=(2.0/m**2)*((m*sj**2+sb**2)**2/(k-1)+sb**4/(k*(m-1)))
    return max(1.0,2*sj**4/var)
def sim(arm,k,m,sb,sj=SJ,N=20000):
    dfB,dfW=k-1,k*(m-1)
    dfd=df_of(sj,sb,k,m); mult=math.sqrt(dfd/chi2q(0.05,dfd))
    Fq=fq(0.05,dfB,dfW); chiW=chi2q(0.05,dfW)
    ok1=ok2=ok3=neg=0; ubs=[]
    for _ in range(N):
        MSB=(m*sj**2+sb**2)*random.gammavariate(dfB/2,2)/dfB
        MSW=sb**2*random.gammavariate(dfW/2,2)/dfW
        v=(MSB-MSW)/m
        if v<=0: neg+=1; v=0.0
        s=math.sqrt(v); ub1=s*mult; ubs.append(ub1)
        if GAP[arm]>=3*ub1: ok1+=1
        d2=df_of(max(s,1e-9),sb,k,m); ub2=s*math.sqrt(d2/chi2_lu(d2))
        if GAP[arm]>=3*ub2: ok2+=1
        th=((MSB/MSW)/Fq-1.0)/m
        ub3=math.sqrt(max(0.0,th)*MSW*dfW/chiW)
        if GAP[arm]>=3*ub3: ok3+=1
    ubs.sort()
    return neg/N,ok1/N,ok2/N,ok3/N,ubs[N//2],ubs[int(.9*N)]
print("truth sigma_job = %.4f%% (doc prior);  rule: PASS iff gap >= 3*UB95"%SJ)
for arm,k,m,sb,lab in [("M8",4,3,SB_DOC["M8"],"doc"),("M8",6,3,SB_DOC["M8"],"doc"),("M8",8,3,SB_DOC["M8"],"doc"),
                       ("Ha8",4,3,SB_DOC["Ha8"],"doc"),("Ha8",12,6,SB_DOC["Ha8"],"doc"),("Ha8",8,12,SB_DOC["Ha8"],"doc"),
                       ("Ha8",6,3,0.596,"240s doc-scaled"),("Ha8",6,3,0.784,"240s paired-scaled")]:
    neg,p1,p2,p3,md,p90=sim(arm,k,m,sb)
    print(f"{arm:4s} k={k:2d} m={m:2d} sb={sb:.4f} [{lab:16s}] P(neg-var)={neg:.3f}  P(pass:design-df)={p1:.3f}  P(pass:plug-in df)={p2:.3f}  P(pass:exact-F)={p3:.3f}  medUB={md:.2f}%  p90UB={p90:.2f}%")
