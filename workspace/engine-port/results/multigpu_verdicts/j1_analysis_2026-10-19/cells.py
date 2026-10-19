import sys, json, glob, os, statistics as st, csv, datetime as dt
import numpy as np
ROOT=sys.argv[1]
sys.path.insert(0, '/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/multigpu')
import mg_label as L
clk=json.load(open(os.path.join(ROOT,'CLOCK_BY_CELL.json')))['cells']
rows=[]
for d in sorted(glob.glob(os.path.join(ROOT,'cells','*'))):
    c=L.load_cell(d); m=c['meta']; reqs=L.load_requests(os.path.join(d,'requests.jsonl'))
    ok=[r for r in reqs if not L.is_error(r)]
    ttft=[r['ttft_ms'] for r in ok]
    rq_itl95=[np.percentile(r['token_itl_ms'],95) for r in ok if r.get('token_itl_ms')]
    allitl=np.concatenate([r['token_itl_ms'] for r in ok if r.get('token_itl_ms')])
    ck=clk.get(c['cell_id'],{})
    rows.append(dict(cell=c['cell_id'], arm=m['arm'], shape=m['shape'], rung=m['rung'], ext=m.get('ext'), seed=m['seed'], n=c['n_sent'],
      offered=m['offered'], T=c['T'], achieved=c['achieved'], g=c['g'], srv=c['n_server_errors'], cli=c['n_client_errors'],
      wall=m['t_end_epoch']-m['t_start_epoch'], t0=m['t_start_epoch'],
      ttft=[float(np.percentile(ttft,q)) for q in (50,95,99)], itl=[float(np.percentile(allitl,q)) for q in (50,95,99)],
      rq_itl95_p50=float(np.median(rq_itl95)), thr=ck.get('throttle_frac'), mhz=ck.get('sm_clock_mhz_median'), w=ck.get('power_w_median')))
json.dump(rows, open(sys.argv[2],'w'), indent=0)
for r in rows:
    print(f"{r['cell'][6:]:34s} n={r['n']:4d} off={r['offered']:.3f} ach={r['achieved']:.4f} g={r['g']:.3f} err={r['srv']}/{r['cli']} wall={r['wall']:.0f} T={r['T']:.0f} TTFT50/95/99={r['ttft'][0]/1e3:.2f}/{r['ttft'][1]/1e3:.2f}/{r['ttft'][2]/1e3:.2f}s ITL={r['itl'][0]:.1f}/{r['itl'][1]:.1f}/{r['itl'][2]:.0f}ms thr={r['thr']:.2f} mhz={r['mhz']} W={r['w']}")
