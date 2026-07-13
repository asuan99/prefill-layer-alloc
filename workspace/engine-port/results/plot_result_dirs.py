#!/usr/bin/env python
"""engine-port results/ 각 디렉토리를 개별 시각화. 각 dir의 CSV(19-col 표준 스키마)를
goodput/TTFT/TPOT × regime으로. prefill_knee는 per-layer 민감도 전용 figure.
결과: results/<dir>/<dir>_viz.png"""
import os, re, csv, glob, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size":8,"axes.grid":True,"grid.alpha":0.3,"axes.axisbelow":True})

# 표준 스키마 인덱스
IX = dict(label=1, map=2, in_len=5, out_len=6, rate=7, ttft50=11, tpot50=13, good=15, gp=16, dur=17)

def num(x, d=float("nan")):
    try: return float(x)
    except: return d

def series_key(label, mapv):
    lab = re.sub(r'_?in\d+o\d+.*$','',label); lab = re.sub(r'_in\d+$','',lab)
    if mapv and not mapv.replace('.','').isdigit() and mapv not in ('measured','slo',''):
        return mapv
    return lab or mapv or label

def read_dir(d):
    rows=[]
    for f in glob.glob(os.path.join(d,"*.csv")):
        for r in csv.reader(open(f)):
            if len(r) < 18: continue
            if r[0].strip()=="model" or r[IX['rate']].strip() in ("rate",""): continue
            try:
                rows.append(dict(
                    skey=series_key(r[IX['label']].strip(), r[IX['map']].strip()),
                    in_len=int(num(r[IX['in_len']],0)), out_len=int(num(r[IX['out_len']],0)),
                    rate=num(r[IX['rate']]), ttft50=num(r[IX['ttft50']]),
                    tpot50=num(r[IX['tpot50']]), gp=num(r[IX['gp']])))
            except Exception: continue
    return rows

def regime_name(inl, outl):
    if inl>=3000: return f"PREFILL-bound in{inl}/o{outl}"
    return f"DECODE-heavy in{inl}/o{outl}"

def plot_csv_dir(d, name):
    rows = read_dir(d)
    if not rows: return False
    regimes = sorted(set((r['in_len'],r['out_len']) for r in rows if r['in_len']),
                     key=lambda t:-t[0])
    regimes = [rg for rg in regimes if rg[0]>0]
    if not regimes: return False
    skeys = sorted(set(r['skey'] for r in rows))
    cmap = cm.get_cmap('tab20' if len(skeys)>10 else 'tab10')
    color = {s:cmap(i%cmap.N) for i,s in enumerate(skeys)}
    metrics=[("gp","goodput@SLO (req/s)",False),("ttft50","TTFT p50 (ms)",True),("tpot50","TPOT p50 (ms)",True)]
    nR=len(regimes)
    fig,axes = plt.subplots(nR,3, figsize=(14,3.6*nR), squeeze=False)
    for ri,rg in enumerate(regimes):
        for ci,(mk,ml,logy) in enumerate(metrics):
            ax=axes[ri][ci]
            for s in skeys:
                pts=sorted([(r['rate'],r[mk]) for r in rows
                            if r['skey']==s and (r['in_len'],r['out_len'])==rg and r[mk]==r[mk]])
                if not pts: continue
                # average duplicate rates (reps)
                agg={}
                for x,y in pts: agg.setdefault(x,[]).append(y)
                xs=sorted(agg); ys=[sum(agg[x])/len(agg[x]) for x in xs]
                ax.plot(xs,ys,marker='o',ms=4,lw=1.4,color=color[s],
                        label=s if (ri==0 and ci==0) else None)
            if mk=="tpot50": ax.axhline(60,color='k',ls='--',lw=0.8)
            if mk=="ttft50": ax.axhline(3000,color='k',ls='--',lw=0.8)
            if logy: ax.set_yscale('log')
            ax.set_xlabel("request rate"); ax.set_title(f"{regime_name(*rg)}\n{ml}",fontsize=8)
    # shared legend
    handles,labels = axes[0][0].get_legend_handles_labels()
    ncol = min(6, max(2,len(labels)))
    fig.legend(handles,labels,loc="lower center",ncol=ncol,fontsize=7,
               bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(f"[{name}]  {DIR_DESC.get(name,'')}",fontsize=11,y=1.0)
    fig.tight_layout(rect=[0,0.04,1,0.97])
    out=os.path.join(d,f"{name}_viz.png"); fig.savefig(out,dpi=125,bbox_inches="tight"); plt.close(fig)
    print("wrote",out,f"({len(rows)} rows, {len(skeys)} series, {nR} regimes)")
    return True

def plot_prefill_knee(d):
    SM=[108,44,24,16,8]
    pk_attn=[10.97,25.18,45.78,68.52,135.4]; pk_mamba=[9.03,17.51,31.98,47.62,94.8]
    dk_attn=[3.14,6.27,11.77,17.51,34.0]; dk_mamba=[0.36,0.27,0.40,0.53,0.97]
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].plot(SM,pk_attn,'o-',color='#d62728',label='attn-prefill')
    ax[0].plot(SM,pk_mamba,'s-',color='#1f77b4',label='mamba-prefill')
    ax[0].set_title("(1) PREFILL per-layer ms vs SM\nboth SM-sensitive (only slope differs)")
    ax[0].set_xlabel("prefill SM (green-ctx pin)"); ax[0].set_ylabel("ms / layer"); ax[0].legend()
    ax[0].invert_xaxis()
    ax[1].plot(SM,dk_attn,'o-',color='#d62728',label='attn-decode')
    ax[1].plot(SM,dk_mamba,'s-',color='#1f77b4',label='mamba-decode (flat)')
    ax[1].set_title("(2) DECODE per-layer ms vs SM\nmamba flat -> free-lunch exists")
    ax[1].set_xlabel("decode SM"); ax[1].set_ylabel("ms / layer"); ax[1].legend(); ax[1].invert_xaxis()
    ax[2].plot(SM,[a/m for a,m in zip(pk_attn,pk_mamba)],'^-',color='#7f7f7f',label='PREFILL attn/mamba')
    ax[2].plot(SM,[a/m for a,m in zip(dk_attn,dk_mamba)],'v-',color='#000000',label='DECODE attn/mamba')
    ax[2].set_title("(3) cost-ratio attn/mamba\nprefill 1.2-1.4x (weak) vs decode 9-35x (strong)")
    ax[2].set_xlabel("SM"); ax[2].set_ylabel("attn/mamba cost-ratio"); ax[2].legend(); ax[2].invert_xaxis()
    ax[2].set_yscale('log')
    fig.suptitle("[prefill_knee]  does a prefill-side layer-aware lever exist? -> NO (differential too small)",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.94])
    out=os.path.join(d,"prefill_knee_viz.png"); fig.savefig(out,dpi=125,bbox_inches="tight"); plt.close(fig)
    print("wrote",out)

DIR_DESC = {
 "r0a":"policy comparison (fused/agnostic/layer_aware) - first clean-async serving",
 "r0b":"graduated per-type layer-aware & decode-SM knee (green-ctx SM pairs)",
 "r0c":"coordinated tuned-uniform decode-SM sweep (fixed prefill:decode split)",
 "r0d_coord_la":"coordinated per-type layer-aware - BUILT/CORRECT/REFUTED (TPOT blow-up)",
 "a_substrate":"(a) substrate isolation (LA_COORD_OPT) - 124->85ms but sign holds",
 "v4_multiwin":"v4 faithful multi-window overlap - BUILT/CORRECT/REFUTED",
 "pf_boundary":"prefill-type boundary layer-aware (PF) - worst",
 "pf_boundary_fix1":"PF fix#1",
 "pf_boundary_fix2":"PF fix#2 (chunk cap)",
 "slo_sched":"SLO-aware controller (v6/v7/v7b, isolation diag, mixed B)",
}

if __name__=="__main__":
    dirs=[x for x in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT,x))]
    for name in dirs:
        d=os.path.join(ROOT,name)
        if name=="cudagraph_probe": continue  # 별도 figure 존재
        if name=="prefill_knee": plot_prefill_knee(d); continue
        try: plot_csv_dir(d,name)
        except Exception as e: print("SKIP",name,repr(e))
