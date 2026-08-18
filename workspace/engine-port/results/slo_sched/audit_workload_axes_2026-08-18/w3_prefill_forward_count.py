#!/usr/bin/env python3
"""AUDIT (read-only) of WORKLOAD_AXES_2026-08-18 W-1/W-2/W-3.

Claim under test: "prefill forward ~= 200 (341 tokens < budget 65536 => 1 chunk
per REQUEST)" and the layers/forward table indexed by per-request L.

Engine ground truth (schedule_batch.py:1570):
    extend_num_tokens = sum(len(ids) for ids in input_ids)   # BATCH SUM
so multiplexing_mixin.py:1128-1133 computes
    forward_count = max(1, 65536 // extend_num_tokens)       # layers per forward
    forwards_for_this_batch = ceil(n_layers / forward_count)
with extend_num_tokens = the batch's total extend tokens, NOT one request's L.

srv.log 'Prefill batch, #new-seq: N, #new-token: T' gives (N, T) per admitted
prefill batch => we can count real prefill forwards exactly.
"""
import glob, math, re, sys, json, collections

N_LAYERS = 54          # Zamba2-2.7B, confirmed by prefill_chunk_progress==54
BUDGET = 65536

pat = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")

def analyze(path):
    seqs = toks = batches = forwards = 0
    multi_seq = multi_fwd = 0
    newtok = []
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        n, t, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        batches += 1; seqs += n; toks += t
        lpf = max(1, BUDGET // t) if t > 0 else N_LAYERS
        f = math.ceil(N_LAYERS / lpf)
        forwards += f
        newtok.append(t)
        if n > 1: multi_seq += 1
        if f > 1: multi_fwd += 1
    return dict(batches=batches, seqs=seqs, toks=toks, forwards=forwards,
                multi_seq_batches=multi_seq, multi_forward_batches=multi_fwd,
                newtok=newtok)

if __name__ == "__main__":
    tot = collections.Counter()
    for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "g16_blk*_d44_boot1_*_srv.log")):
        r = analyze(p)
        nt = r.pop("newtok")
        print(f"{p}")
        print(f"   prefill BATCHES={r['batches']}  seqs={r['seqs']}  extend_tokens={r['toks']}")
        print(f"   prefill FORWARDS (layer-split, ceil(54/(65536//E)))={r['forwards']}")
        print(f"   batches with >1 seq: {r['multi_seq_batches']}  "
              f"batches needing >1 forward: {r['multi_forward_batches']}")
        print(f"   #new-token: max={max(nt)} p95={sorted(nt)[int(.95*len(nt))]} "
              f"mean={sum(nt)/len(nt):.1f}  frac(E>1213 => >1 fwd)="
              f"{sum(1 for x in nt if x > BUDGET//N_LAYERS)/len(nt):.3f}")
        for k, v in r.items():
            tot[k] += v
    print("\nTOTAL over files:", dict(tot))
    print("  seqs/batch =", round(tot['seqs']/tot['batches'], 3),
          " forwards/batch =", round(tot['forwards']/tot['batches'], 3),
          " forwards/seq =", round(tot['forwards']/tot['seqs'], 3))
