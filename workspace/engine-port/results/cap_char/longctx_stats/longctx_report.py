"""Aggregate longctx_lengths.tsv -> stats text. usage: python3 -I longctx_report.py lengths.tsv"""
import sys, csv, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
for r in rows:
    r["nano"] = int(r["nano"]); r["llama"] = int(r["llama"])
    r["grp"] = r["dataset"] + ("/E" if r["dataset"] == "lb1" and r["task"].endswith("_e") else "")
def q(xs, p): xs = sorted(xs); return xs[min(len(xs)-1, int(p*len(xs)))]
K = 1024
BINS = [("<=16k", 0, 16*K), ("16-32k", 16*K+1, 32*K), ("32k+-10%", int(.9*32*K), int(1.1*32*K)),
        ("64k+-10%", int(.9*64*K), int(1.1*64*K)), ("96k+-10%", int(.9*96*K), int(1.1*96*K)),
        ("<=128k", 0, 128*K), (">128k", 128*K+1, 10**12)]
def binrow(rs, key):
    return "  ".join(f"{n}:{sum(1 for r in rs if lo <= r[key] <= hi):>4}" for n, lo, hi in BINS)
def ratio(rs): return st.median([r["llama"]/r["nano"] for r in rs if r["nano"]])
def block(title, rs):
    print(f"\n[{title}] n={len(rs)} unique_ctx={len({r['ctx_sha16'] for r in rs})}")
    for key in ("nano", "llama"):
        v = [r[key] for r in rs]
        print(f"  {key:5s} p10 {q(v,.1):>7} p50 {q(v,.5):>7} p90 {q(v,.9):>7} p99 {q(v,.99):>7} max {max(v):>7} mean {st.mean(v):8.0f}")
    print(f"  bins(nano ): {binrow(rs,'nano')}")
    print(f"  bins(llama): {binrow(rs,'llama')}")
    print(f"  llama/nano median ratio {ratio(rs):.4f}")
print(f"BINS (k=1024): " + ", ".join(f"{n}=[{lo},{hi}]" for n, lo, hi in BINS) + "  (bins overlap by design: 32k band lies inside 16-32k? no: 32k band [29491,36045] straddles 32768)")
print("total rows", len(rows))
grps = defaultdict(list)
for r in rows: grps[r["grp"]].append(r)
print("\n===== per dataset =====")
for g in ("lb1", "lb1/E", "lb2", "ib"): block(g, grps[g])
print("\n===== per task =====")
bt = defaultdict(list)
for r in rows: bt[(r["grp"], r["task"])].append(r)
for k in sorted(bt): block("/".join(k), bt[k])
# unique-document counts
print("\n===== unique documents per band (distinct ctx hash with >=1 instance in band; primary = lb1 non-E + lb2 + ib) =====")
prim = [r for r in rows if r["grp"] != "lb1/E"]
def uniq(rs, key, lo, hi): return len({r["ctx_sha16"] for r in rs if lo <= r[key] <= hi})
for key in ("nano", "llama"):
    print(f" tokenizer {key}")
    for g in ("lb1", "lb2", "ib", "ALL"):
        rs = prim if g == "ALL" else grps[g]
        print(f"   {g:4s} inst:", "  ".join(f"{n}:{sum(1 for r in rs if lo<=r[key]<=hi):>4}" for n,lo,hi in BINS))
        print(f"   {g:4s} uniq:", "  ".join(f"{n}:{uniq(rs,key,lo,hi):>4}" for n,lo,hi in BINS))
print("\n  union lb1+lb2 only (no IB), unique:")
lbo = [r for r in prim if r["dataset"] != "ib"]
for key in ("nano", "llama"):
    print(f"   {key}:", "  ".join(f"{n}:{uniq(lbo,key,lo,hi):>4}" for n,lo,hi in BINS))
print("\n  IB excluding synthetic tasks (passkey, number_string, kv_retrieval, math_*, code_run), unique:")
SYN = {"passkey", "number_string", "kv_retrieval", "math_find", "math_calc", "code_run"}
nat = [r for r in prim if r["dataset"] != "ib" or r["task"] not in SYN]
for key in ("nano", "llama"):
    print(f"   {key}:", "  ".join(f"{n}:{uniq(nat,key,lo,hi):>4}" for n,lo,hi in BINS))
# duplicates
print("\n===== duplication (ctx hash) =====")
ch = defaultdict(set); cnt = defaultdict(int)
for r in rows: ch[r["ctx_sha16"]].add((r["grp"], r["task"])); cnt[r["ctx_sha16"]] += 1
for g in ("lb1", "lb1/E", "lb2", "ib"):
    rs = grps[g]; print(f" {g}: inst {len(rs)} uniq {len({r['ctx_sha16'] for r in rs})}")
multi_task = {h for h, s in ch.items() if len({t for _, t in s}) > 1}
print(" hashes appearing in >1 (group,task) across all:", sum(1 for h, s in ch.items() if len(s) > 1))
print(" hashes shared across >1 distinct task name (ignoring _e twins):", sum(1 for h, s in ch.items() if len({t.removesuffix('_e') for _, t in s}) > 1))
cross = [h for h, s in ch.items() if len({g for g, _ in s} - {"lb1/E"}) > 1]
print(" hashes shared across datasets (excluding lb1/E twin):", len(cross))
bt2 = defaultdict(lambda: [0, set()])
for r in prim:
    bt2[(r["grp"], r["task"])][0] += 1; bt2[(r["grp"], r["task"])][1].add(r["ctx_sha16"])
print(" tasks where inst != uniq ctx:")
for k, (n, s) in sorted(bt2.items()):
    if n != len(s): print(f"   {k[0]}/{k[1]}: inst {n} uniq {len(s)}")
# E twin containment
pe = {r["ctx_sha16"] for r in grps["lb1/E"]}; pn = {r["ctx_sha16"] for r in grps["lb1"]}
print(f" LongBench-E ctx hashes {len(pe)}; of which present in non-E: {len(pe & pn)}")
# ---- supplementary: per-task unique docs in the three long-ctx bands; E-inclusive pools ----
print("\n===== unique docs per long-ctx band, per task (primary + lb1/E as separate rows) =====")
B3 = [b for b in BINS if b[0] in ("32k+-10%", "64k+-10%", "96k+-10%")]
ZH = {"dureader", "lsht", "vcsum", "multifieldqa_zh", "passage_retrieval_zh"}
for key in ("nano", "llama"):
    print(f" tokenizer {key}   (task: 32k / 64k / 96k unique docs; instances in parens)")
    for (g, t), rs in sorted(bt.items()):
        u = [uniq(rs, key, lo, hi) for _, lo, hi in B3]; n = [sum(1 for r in rs if lo <= r[key] <= hi) for _, lo, hi in B3]
        if sum(n): print(f"   {g}/{t:34s}{'[zh]' if t.removesuffix('_e') in ZH else '    '} " + " / ".join(f"{a}({b})" for a, b in zip(u, n)))
def pool(rs, key): return [uniq(rs, key, lo, hi) for _, lo, hi in B3]
allr = rows
en_lb = [r for r in rows if r["dataset"] != "ib" and r["task"].removesuffix("_e") not in ZH]
en_nat = [r for r in rows if (r["dataset"] != "ib" or r["task"] not in SYN) and r["task"].removesuffix("_e") not in ZH]
print("\n pools (unique docs in 32k / 64k / 96k bands):")
for name, rs in (("lb1+lb1E+lb2 (all langs)", [r for r in rows if r["dataset"] != "ib"]),
                 ("lb1+lb1E+lb2 (en only; zh tasks dropped)", en_lb),
                 ("lb1+lb1E+lb2+IB natural (en only)", en_nat),
                 ("IB natural only (longbook_*, longdialogue, code_debug)", [r for r in rows if r["dataset"] == "ib" and r["task"] not in SYN])):
    for key in ("nano", "llama"): print(f"   {name:58s} {key:5s}", pool(rs, key))
# LB2 by its own length label
print("\n LB2 'length' label vs nano tokens (sanity):")
lab = defaultdict(list)
for r in grps["lb2"]: lab[r["meta"].split("/")[0]].append(r["nano"])
for k, v in lab.items(): print(f"   {k}: n={len(v)} min {min(v)} p50 {q(v,.5)} max {max(v)}")
# disjoint need
print("\n need for without-replacement fill: docs = seeds(4) x rates(R) x requests-per-cell(N) [per band]; pooled-across-rates variant = 4 x N")
for R in (3, 5):
    for N in (16, 32, 64): print(f"   R={R} N={N}: disjoint {4*R*N}   per-seed-shared-across-rates {4*N}   fully shared {N}")
