import sys, json, glob, os, numpy as np
sys.path.insert(0,'/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/multigpu')
import mg_label as L
ROOT=sys.argv[1]
fl=json.load(open(os.path.join(ROOT,'LABELS.json')))['floor']
for d in sorted(glob.glob(os.path.join(ROOT,'cells','*LO*'))):
    reqs=L.load_requests(os.path.join(d,'requests.jsonl')); n=len(reqs)
    out=[]
    for k in (3,5,10):
        good=0
        for r in reqs:
            if L.is_error(r): continue
            T=L.floor_T(fl, r['input_tokens'], k)
            itl=r.get('token_itl_ms') or []
            p95=np.percentile(itl,95) if itl else 0.0
            if T is not None and r['ttft_ms']/1000<=T and p95<=50: good+=1
        out.append(good/n)
    ttft_ok=sum(1 for r in reqs if not L.is_error(r) and (np.percentile(r['token_itl_ms'],95) if r.get('token_itl_ms') else 0)<=50)/n
    print(os.path.basename(d)[6:], 'attain k3/k5/k10 = %.2f/%.2f/%.2f  ITLp95<=50 frac %.2f'%(*out,ttft_ok))
