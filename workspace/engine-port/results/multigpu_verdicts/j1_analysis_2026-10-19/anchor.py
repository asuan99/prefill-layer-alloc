import sys, glob, os, numpy as np, statistics as st
sys.path.insert(0,'/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/multigpu')
import mg_label as L
base='/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/multigpu/manual_retrieval'
jobs={'J1':('pdmux-multigpu-2106f42-20261018074950','boot1__'),'J2':('pdmux-multigpu-e559c64-20261018114257','anchor__'),'J4':('pdmux-multigpu-213df1b-20261018114306','anchor__'),'J8':('pdmux-multigpu-3167f74-20261018114314','anchor__')}
for J,(j,tag) in jobs.items():
    for arm in ('fused_sym','agnostic'):
        for sh in ('A','X'):
            ds=sorted(glob.glob(f'{base}/{j}/results/multigpu/m3_{j}/cells/m3_N1_{arm}_{tag}{sh}_HI_s*'))
            if not ds: print(J,arm,sh,'none'); continue
            recs=[L.load_cell(d) for d in ds]
            a=[r['achieved'] for r in recs]; g=[r['g'] for r in recs]
            print(J,arm,sh,'n=%d med %.4f mean %.4f sd %.4f'%(len(a),st.median(a),np.mean(a),np.std(a,ddof=1)),'gmin %.3f'%min(g))
