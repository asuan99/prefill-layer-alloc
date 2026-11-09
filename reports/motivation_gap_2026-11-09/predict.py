import sys, json, copy, statistics as st
sys.path.insert(0,'scripts/costmodel')
import dstar_model as dm, analytic_terms as at
out={}
# A. M3 prior from Phase B ladders
d=json.load(open('results/stock_baseline/runs/ADDENDUM_after_J1b.json'))
rows=[]
src={'PM-ref':d['ladder']['PM-ref']['points'],'VL-stock':d['ladder']['VL-stock']['points'],
     'SG-stock':d['ladder_down']['SG-stock']['points'] if 'SG-stock' in d['ladder_down'] else None}
for arm,pts in src.items():
    if not pts: continue
    by={}
    for r,a in pts: by.setdefault(round(r,3),[]).append(a)
    for r,v in sorted(by.items()):
        if r>12.8: continue
        rows.append({'arm':arm,'rate':r,'mid':round(st.fmean(v),3),'lo':round(min(v),3),'hi':round(max(v),3)})
out['m3_prior']=rows
# B. M4a: full vs no-state lambda_cap
summ=json.load(open('../../reports/cost_model/dstar/workload_summary.json'))
co=at.decode_coefficients(at.nano9b_v2(), dm.HBM_BW); b0=co['b0_s_per_req']*1e3
hw=dm.hardware_layer(); u={(r['D'] if r['D']!=108 else '108u'):r['u'] for r in hw['rows']}
orig=copy.deepcopy(dm.T1); ns={}
for m,(form,r1,r2,beta) in orig.items():
    if form=='F3': ns[m]=(form,r1,r2,beta); continue
    dd=b0/u[m]; r1=list(r1); r2=list(r2); r1[1]=max(r1[1]-dd,0); r2[1]=max(r2[1]-dd,0); ns[m]=(form,r1,r2,beta)
def caps(T,wl):
    dm.T1=T; res={}
    for D in dm.D_GRID:
        mom=dm.moments(wl['Ls'],D); r=dm.lam_cap(D,wl,1.16,mom); res[D]=(round(r['lam_cap'],3),r['binding'])
    return res
m4=[]
for name in ('W1','W3'):
    wl=dm.workload(summ,name)
    f=caps(orig,wl); n=caps(ns,wl)
    for D in dm.D_GRID:
        m4.append({'wl':name,'D':D,'variant':'hybrid 항 포함','lam':f[D][0],'bind':f[D][1]})
        m4.append({'wl':name,'D':D,'variant':'state 항 제외','lam':n[D][0],'bind':n[D][1]})
out['m4']=m4
# grid divergence search: synthetic workloads
grid=[]
for L in (256,1024,4096,8192,16000):
    for R in (64,256,1024):
        wl={'ER':R,'ctx_decode':L+R/2,'slo':{'ttft_ms':3000.0,'itl_ms':60.0},'Ls':[L]*50}
        f=caps(orig,wl); n=caps(ns,wl)
        df=max(dm.D_GRID,key=lambda D:(f[D][0],-D)); dn=max(dm.D_GRID,key=lambda D:(n[D][0],-D))
        grid.append({'L':L,'R':R,'Dstar_full':df,'Dstar_nostate':dn,'lam_full':f[df][0],'lam_nostate':n[dn][0],'differs':df!=dn})
dm.T1=orig
out['m4a_grid']=grid
# C. D3 cap256 required mem fraction vs mean ctx  [derived lower bound]
a=at.nano9b_v2(); s=at.state_bytes_per_request(a); k=at.kv_bytes_per_token(a)
GiB=2**30; avail=78.74*GiB; w=16.58*GiB
out['d3']=[{'ctx':c,'mem':round((w+256*s+256*c*k)/avail,3)} for c in (0,500,1000,1500,2000,2500,3000,4000,6000,8000)]
out['d3_consts']={'s':s,'k':k}
# D. analytic decode step lower bound vs ctx, b=48: Nano vs Llama-3.1-8B
llama=at.Arch(d=4096,n_q=32,n_kv=8,d_h=128,n_h=128,p=80,G=8,N=128,K=4,Q=128,d_ff=14336,V=128256,n_A=32,n_M=0,n_F=32,mlp_gated=True)
m5=[]
for name,A in (('Nemotron-Nano-9B-v2',a),('Llama-3.1-8B',llama)):
    c0=at.decode_coefficients(A,dm.HBM_BW)
    for ctx in (256,1000,2000,4000,8000,16000,32000):
        t=(c0['a0_s']+c0['b0_s_per_req']*48+c0['c0_s_per_req_token']*48*ctx)*1e3
        m5.append({'model':name,'ctx':ctx,'ms':round(t,2)})
    out.setdefault('m5_coef',{})[name]={k2:v for k2,v in c0.items()}
json.dump(out,open(sys.argv[1],'w'),indent=1,ensure_ascii=False)
print(json.dumps({k:(v if k!='m3_prior' else len(v)) for k,v in out.items()},ensure_ascii=False)[:3500])
