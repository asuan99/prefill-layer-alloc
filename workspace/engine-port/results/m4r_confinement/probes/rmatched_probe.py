#!/usr/bin/env python3
"""M4R prerequisite probe 2 -- does an SM-matched contrast survive drain exclusion?

GPU 0.  Registered by audit_m4r_rules_2026-08-27 ("R_matched를 결정량으로 승격,
드레인 창 제외 등록").  Probe 1 (alias_verdict.json) showed P(decode_sms==D|confined)
= 1.0000 in 28/28 cells, so the rev1 estimand is barred; this asks whether the
SM-matched replacement is even measurable.

R_matched = rate(free & decode_sms==D & not drain) / rate(confined & decode_sms==D)
Drain = the first DRAIN_MS after every confined -> free transition (adjust_stream_groups
does 2x synchronize at the prefill boundary).
Registered viability floor: MIN_CLEAN_S of clean free@D time per cell.
"""
import json, glob, os, re
from collections import defaultdict

MAX_GAP_S, MIN_CLEAN_S = 1.0, 20.0
DRAIN_MS = [0, 25, 50, 100, 200]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rmatched_verdict.json")
F = {k: re.compile(r'"%s":\s*(-?[\d.]+)' % k) for k in
     ("timestamp_monotonic_s","prefill_active_batch_size","decode_running_batch_size",
      "decode_sms","decode_iterations")}

def scan(path, D):
    rows=[]
    for ln in open(path):
        if '"runtime_snapshot"' not in ln or '"benchmark"' not in ln: continue
        v={}
        for k,rx in F.items():
            m=rx.search(ln); v[k]=float(m.group(1)) if m else None
        if v["timestamp_monotonic_s"] is None: continue
        rows.append((v["timestamp_monotonic_s"], v["prefill_active_batch_size"] or 0,
                     v["decode_running_batch_size"] or 0,
                     int(v["decode_sms"]) if v["decode_sms"] is not None else None,
                     v["decode_iterations"]))
    rows.sort()
    out={}
    for dm in DRAIN_MS:
        Tc=Nc=Tf=Nf=0.0; last_conf_end=-1e9
        for (t0,p0,d0,sm0,i0),(t1,_,_,_,i1) in zip(rows,rows[1:]):
            dt=t1-t0
            if dt<=0 or dt>MAX_GAP_S or d0<=0: continue
            di=(i1-i0) if (i0 is not None and i1 is not None and i1>=i0) else 0
            if p0>0:
                last_conf_end=t1
                if sm0==D: Tc+=dt; Nc+=di
            else:
                if sm0==D and (t0-last_conf_end)*1000.0 >= dm:
                    Tf+=dt; Nf+=di
        out[dm]={"T_conf_D":Tc,"T_free_D_clean":Tf,
                 "R_matched":(Nf/Tf)/(Nc/Tc) if (Tf>0 and Tc>0 and Nc>0) else None}
    return out

cells=defaultdict(list)
for path in sorted(glob.glob("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/"
                             "engine-port/results/slo_sched/g16_blk*_d*_boot1_*_telemetry.jsonl")):
    D=int(re.search(r'_d(\d+)_',os.path.basename(path)).group(1))
    cells[D].append(scan(path,D))

res={"probe":"M4R-Rmatched","gpu_hours":0.0,"drain_ms_grid":DRAIN_MS,
     "min_clean_s":MIN_CLEAN_S,"cells":{}}
print(f"{'D':>4} {'drain':>6} {'T_conf@D':>9} {'T_free@D':>9} {'R_matched':>10}  viable")
for D in sorted(cells):
    res["cells"][D]={}
    for dm in DRAIN_MS:
        Tc=sum(b[dm]["T_conf_D"] for b in cells[D])
        Tf=sum(b[dm]["T_free_D_clean"] for b in cells[D])
        rs=[b[dm]["R_matched"] for b in cells[D] if b[dm]["R_matched"] is not None]
        R=sum(rs)/len(rs) if rs else None
        viable = Tf>=MIN_CLEAN_S and Tc>=MIN_CLEAN_S
        res["cells"][D][dm]={"T_conf_D":round(Tc,2),"T_free_D_clean":round(Tf,2),
                             "R_matched_mean":round(R,4) if R else None,"viable":viable,
                             "n_blocks":len(rs)}
        print(f"d{D:<3} {dm:6} {Tc:9.1f} {Tf:9.1f} "
              f"{(f'{R:10.4f}' if R else '        NA')}  {'yes' if viable else 'NO'}")
via={dm:sum(1 for D in res["cells"] if res["cells"][D][dm]["viable"]) for dm in DRAIN_MS}
res["viable_cells_by_drain"]=via
res["verdict"]=("RMATCHED_VIABLE" if via.get(50,0)>=2 else "RMATCHED_NOT_VIABLE")
json.dump(res,open(OUT,"w"),indent=2,ensure_ascii=False,sort_keys=True)
print(f"\nviable cells by drain_ms: {via}\nVERDICT: {res['verdict']}")
