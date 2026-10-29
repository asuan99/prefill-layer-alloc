import sys, json
from pathlib import Path
R=Path('/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results')
sys.path.insert(0,str(R/'cap_char/policy_harness'))
import ph_spec as P, ph_verdict as V
print([ (k,getattr(P,k)) for k in dir(P) if 'TTFT' in k or 'ITL' in k][:10])
d=R/'cap_char/policy_runs/pdmux-policy-cap2-e90e233-20261029023336/sustain/seg'
for f in sorted(d.glob('*.jsonl')):
    q=[json.loads(l) for l in open(f)]
    q.sort(key=lambda x:x['scheduled_arrival_s'])
    T=P.W1_TTFT_MS if hasattr(P,'W1_TTFT_MS') else 3000
    out=[]
    for n in (60,120,240,480):
        s=q[:n]; out.append((n, round(sum(V.req_good(x,T,P.ITL_MS) for x in s)/n,3), round(sorted(x['ttft_ms'] or 1e9 for x in s)[n//2]/1000,2)))
    # throughput: completion
    ends=[x['sent_s']+x['completion_s'] for x in q if x.get('completion_s')]
    print(f.name, out, 'last_end', round(max(ends),1), 'reqs/s overall', round(len(q)/max(ends),2), 'arrival_end',round(q[-1]['scheduled_arrival_s'],1))
