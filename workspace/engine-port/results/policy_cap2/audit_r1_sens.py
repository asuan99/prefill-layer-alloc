import sys, json, collections, statistics
from pathlib import Path
R=Path('/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results')
sys.path.insert(0,str(R/'cap_char/policy_harness'))
import ph_spec as P, ph_verdict as V, ph_analyze as A
raw=R/'cap_char/policy_runs/pdmux-policy-cap2-e90e233-20261029023336/raw'
runs=A.load_runs(raw)
fl=json.load(open(raw/'floor.json'))['ttft_ms']
cands={'median':statistics.median(fl),'mean':statistics.mean(fl),'min':min(fl),'max':max(fl),
 'med_drop_short':statistics.median([x for x in fl if x>300]),'med_excl_first':statistics.median(fl[1:])}
def table(T,thr=0.9):
    c=collections.defaultdict(lambda:[0,0])
    for r in runs:
        D,rate=r['arm']['D'],r['spec']['rate']
        for s in r['segments']:
            for q in s['requests']:
                c[(D,rate)][0]+=1; c[(D,rate)][1]+=V.req_good(q,T,P.ITL_MS)
    out={}
    for D in sorted({k[0] for k in c}):
        rates=sorted(k[1] for k in c if k[0]==D)
        cap=None
        for rt in rates:
            n,g=c[(D,rt)]
            if g/n>=thr: cap=rt
            else: break
        out[D]=cap
    return out,c
for name,f in cands.items():
    for k in (2.5,3,3.5):
        o,c=table(k*f); best=max(o.items(),key=lambda x:(x[1] or 0))
        print(f'{name:16s} floor={f:7.1f} k={k} SLO={k*f:7.0f} caps={o} lam*={best[1]} bestD={best[0]}')
o,c=table(3*cands['median'])
for thr in (0.85,0.875,0.9,0.92,0.95):
    o,_=table(3*cands['median'],thr); print('thr',thr,o)
for D in (16,24): 
  for rt in sorted(k[1] for k in c if k[0]==D): print(D,rt,c[(D,rt)])
# per-seed D16 0.3375
for r in runs:
  if r['arm']['D'] in (16,24) and r['spec']['rate'] in (0.3375,0.225,0.50625):
    reqs=[q for s in r['segments'] for q in s['requests']]
    T=3*cands['median']
    print(r['run_id'], sum(V.req_good(q,T,P.ITL_MS) for q in reqs), len(reqs), 'ttft_fail',sum((q.get('ttft_ms') or 1e9)>T for q in reqs))
