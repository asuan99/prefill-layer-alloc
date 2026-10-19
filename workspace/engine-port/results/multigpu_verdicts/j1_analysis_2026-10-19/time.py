import json, glob, os, sys, collections
ROOT=sys.argv[1]
rows=json.load(open(sys.argv[2]))
drv=[json.loads(l) for l in open(os.path.join(ROOT,'DRIVER_LOG.jsonl'))]
boots=[d for d in drv if 'boot' in d]
by=collections.defaultdict(list)
for r in rows:
    b=r['cell'].split('__')[0]; by[b].append(r)
tot_cell=0
for d in boots:
    cs=by[d['boot']]; w=sum(c['wall'] for c in cs); tot_cell+=w
    st=json.load(open(os.path.join(ROOT,'boots',d['boot'],'steps.json'))) if os.path.exists(os.path.join(ROOT,'boots',d['boot'],'steps.json')) else None
    print(f"{d['boot']:28s} boot_s={d['seconds']:.0f} cells={len(cs)} cell_wall={w:.0f} overhead={d['seconds']-w:.0f}")
print('sum boots', sum(d['seconds'] for d in boots), 'sum cell wall', tot_cell)
