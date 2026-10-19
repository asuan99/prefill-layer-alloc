import sys, json, glob, os, collections
sys.path.insert(0,'/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/benchmarks')
from pdmux_eval.analyze import controller_summary
R=sys.argv[1]
for b in sorted(glob.glob(R+'/boots/m3_N1_*/tel.jsonl')):
    ev=[json.loads(l) for l in open(b)]
    phases=collections.Counter(e.get('phase') for e in ev if e.get('event')=='runtime_snapshot')
    # treat all snapshots outside startup as benchmark
    ev2=[dict(e,phase='benchmark') for e in ev if e.get('event')=='runtime_snapshot' and e.get('phase')!='startup']
    s=controller_summary(ev2)
    import statistics as st
    d=s['dwell_s']
    print(os.path.basename(os.path.dirname(b)), dict(phases), 'trans',s['split_transitions'], {k:round(v,3) for k,v in s.get('residency_fraction',{}).items()}, 'dwell p50 %.2f'%(st.median(d) if d else -1))
