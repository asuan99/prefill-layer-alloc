#!/usr/bin/env python3
"""AUDIT B-1: are there really TWO channels (decode-SM->ITL, capacity->backlog->TTFT)?

Read-only. G16 4 blocks x 7 arms, HI phase, canonical loader/predicate.
"""
import json, math, sys, itertools
from pathlib import Path
import numpy as np

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16","d24","d34","d44","d54","d64","d74"]
BLOCKS = [("blk1","884336"),("blk2","884410"),("blk3","884411"),("blk4","884412")]
TT, IT = 3000.0, 60.0

def cell(phase, blk, job, arm):
    reqs, dur = load_bench_serving_rounds(SLO / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl")
    n = len(reqs)
    itl95 = [percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf for r in reqs]
    all_itl = [v for r in reqs for v in r.token_itl_ms]
    ttft_ok = [r.ttft_ms <= TT for r in reqs]
    itl_ok = [v <= IT for v in itl95]
    joint = [a and b for a,b in zip(ttft_ok,itl_ok)]
    ntok = sum(len(r.token_itl_ms) for r in reqs)
    return dict(n=n, dur=dur, thr=n/dur, gp=sum(joint)/dur,
                ttft_pass=100*sum(ttft_ok)/n, itl_pass=100*sum(itl_ok)/n,
                joint_pass=100*sum(joint)/n,
                tok_thr=ntok/dur,
                itl_mean=float(np.mean(all_itl)), itl_p50=float(np.median(all_itl)),
                req_itl_p50=float(np.median([v for v in itl95 if math.isfinite(v)])),
                ttft_p50=float(np.median([r.ttft_ms for r in reqs])))

data = {}
for phase in ("HI",):
    for arm in ARMS:
        for blk, job in BLOCKS:
            data[(phase,arm,blk)] = cell(phase, blk, job, arm)

# pooled-over-blocks (what the doc reports: 4 blocks concatenated)
def pooled(phase, arm):
    reqs_all, dur_all = [], 0.0
    for blk, job in BLOCKS:
        r,d = load_bench_serving_rounds(SLO / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl")
        reqs_all += r; dur_all += d
    n=len(reqs_all)
    itl95=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs_all]
    ttft_ok=[r.ttft_ms<=TT for r in reqs_all]; itl_ok=[v<=IT for v in itl95]
    joint=[a and b for a,b in zip(ttft_ok,itl_ok)]
    ntok=sum(len(r.token_itl_ms) for r in reqs_all)
    return dict(n=n,dur=dur_all,thr=n/dur_all,
                ttft_pass=100*sum(ttft_ok)/n, itl_pass=100*sum(itl_ok)/n,
                joint_pass=100*sum(joint)/n, tok_thr=ntok/dur_all,
                itl_mean=float(np.mean([v for r in reqs_all for v in r.token_itl_ms])))

print("### G16 HI, pooled over 4 blocks (reproduces the doc's table)")
print(f"{'arm':5}{'thr':>8}{'ttft%':>8}{'itl%':>8}{'joint%':>8}{'tok/s':>9}{'itl_mean':>10}")
P={}
for a in ARMS:
    p=pooled("HI",a); P[a]=p
    print(f"{a:5}{p['thr']:8.3f}{p['ttft_pass']:8.2f}{p['itl_pass']:8.2f}{p['joint_pass']:8.2f}{p['tok_thr']:9.1f}{p['itl_mean']:10.3f}")

thr=np.array([P[a]['thr'] for a in ARMS]); tp=np.array([P[a]['ttft_pass'] for a in ARMS])
ip=np.array([P[a]['itl_pass'] for a in ARMS]); tok=np.array([P[a]['tok_thr'] for a in ARMS])
im=np.array([P[a]['itl_mean'] for a in ARMS])
def r(x,y): return float(np.corrcoef(x,y)[0,1])
print(f"\n  r(thr, ttft_pass) = {r(thr,tp):.4f}   [doc says 0.976]")
print(f"  full-range thr spread = {(thr.max()-thr.min())/thr.min()*100:.2f}%  [doc 10.3%]")
pl=[ARMS.index(a) for a in ('d34','d44','d54','d64','d74')]
print(f"  plateau(d34+) spread  = {(thr[pl].max()-thr[pl].min())/thr[pl].min()*100:.2f}%  [doc 2.3%]"
      f"  r_plateau = {r(thr[pl],tp[pl]):.4f}  [doc 0.777]")

print("\n### IDENTITY CHECK 1: is TTFT-pass an algebraic function of achieved throughput?")
print("  deterministic-backlog model: arrivals at lambda, service at mu=thr;")
print("  TTFT_i = i*(1/mu - 1/lambda)  ->  pass_frac = min(1, S*mu*lam/(N*(lam-mu)))")
LAM = 12.0; S = 3.0
N = P['d16']['n']/4.0   # requests per block
for a in ARMS:
    mu=P[a]['thr']
    pred = min(1.0, S/((1/mu - 1/LAM)*N))*100 if mu < LAM else 100.0
    print(f"  {a}: mu={mu:.3f} predicted_ttft_pass={pred:6.2f}%  observed={P[a]['ttft_pass']:6.2f}%")
pred=np.array([min(1.0, S/((1/P[a]['thr']-1/LAM)*N))*100 for a in ARMS])
print(f"  r(predicted, observed) = {r(pred,tp):.4f}   -> TTFT-pass carries ~no information beyond mu")

print("\n### IDENTITY CHECK 2: is achieved throughput itself just the decode token rate?")
print(f"  r(thr, tok_thr)   = {r(thr,tok):.4f}")
print(f"  r(thr, 1/itl_mean)= {r(thr,1.0/im):.4f}")
print(f"  r(ttft_pass, 1/itl_mean) = {r(tp,1.0/im):.4f}")
print("  -> if these are ~1, 'capacity' is NOT a second channel; it is the SAME")
print("     decode-speed scalar read out on a second axis.")

print("\n### PAIRED n=4 BLOCK CI on the in-plateau throughput spread (doc: 'buried in CI')")
def tcrit(df): return {1:12.706,2:4.303,3:3.182}.get(df,1.96)
base='d34'
for a in ('d44','d54','d64','d74'):
    dif=np.array([data[('HI',a,b)]['thr']-data[('HI',base,b)]['thr'] for b,_ in BLOCKS])
    se=dif.std(ddof=1)/2
    print(f"  thr[{a}] - thr[{base}]: mean={dif.mean():+.4f} req/s  t(3)-CI=[{dif.mean()-tcrit(3)*se:+.4f},{dif.mean()+tcrit(3)*se:+.4f}]"
          f"  {'EXCLUDES 0' if (dif.mean()-tcrit(3)*se)*(dif.mean()+tcrit(3)*se)>0 else 'includes 0'}")
for a in ('d34','d44','d54','d74'):
    dif=np.array([data[('HI','d64',b)]['thr']-data[('HI',a,b)]['thr'] for b,_ in BLOCKS])
    se=dif.std(ddof=1)/2
    print(f"  thr[d64] - thr[{a}]: mean={dif.mean():+.4f} req/s  t(3)-CI=[{dif.mean()-tcrit(3)*se:+.4f},{dif.mean()+tcrit(3)*se:+.4f}]"
          f"  {'EXCLUDES 0' if (dif.mean()-tcrit(3)*se)*(dif.mean()+tcrit(3)*se)>0 else 'includes 0'}")

print("\n### does the joint goodput actually move inside the plateau? (paired n=4 blocks)")
for a in ('d44','d54','d64','d74'):
    dif=np.array([data[('HI',a,b)]['joint_pass']-data[('HI','d34',b)]['joint_pass'] for b,_ in BLOCKS])
    se=dif.std(ddof=1)/2
    print(f"  joint%[{a}] - joint%[d34]: mean={dif.mean():+.3f} pp  t(3)-CI=[{dif.mean()-tcrit(3)*se:+.3f},{dif.mean()+tcrit(3)*se:+.3f}]")
