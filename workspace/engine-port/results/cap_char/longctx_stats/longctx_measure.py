"""Measure raw input token lengths (context+question, no chat template) for
LongBench v1/v2 and InfiniteBench with two tokenizers. Reads raw json/jsonl only.
usage: python3 -I longctx_measure.py <data_root> <out.tsv>
data_root contains lb1/data.zip lb2/data.json ib/*.jsonl tok_nano/tokenizer.json tok_llama/tokenizer.json
"""
import sys, os, json, zipfile, glob, hashlib
from tokenizers import Tokenizer
R, OUT = sys.argv[1], sys.argv[2]
def items():
    z = zipfile.ZipFile(os.path.join(R, "lb1", "data.zip"))
    for n in sorted(z.namelist()):
        if not n.endswith(".jsonl"): continue
        task = os.path.basename(n)[:-6]
        for i, l in enumerate(z.open(n)):
            r = json.loads(l)
            yield "lb1", task, i, r["context"], r["input"], r.get("language", "")
    for i, r in enumerate(json.load(open(os.path.join(R, "lb2", "data.json")))):
        yield "lb2", r["domain"], i, r["context"], r["question"], r["length"]+"/"+r["difficulty"]
    for f in sorted(glob.glob(os.path.join(R, "ib", "*.jsonl"))):
        task = os.path.basename(f)[:-6]
        for i, l in enumerate(open(f)):
            r = json.loads(l)
            ctx = r["context"]
            if isinstance(ctx, list): ctx = "\n".join(map(str, ctx))
            yield "ib", task, i, ctx, r.get("input") or "", ""
toks = {k: Tokenizer.from_file(os.path.join(R, d, "tokenizer.json")) for k, d in (("nano", "tok_nano"), ("llama", "tok_llama"))}
rows, buf = [], []
def flush():
    if not buf: return
    texts = [c + "\n" + q for (_, _, _, c, q, _) in buf]
    L = {k: [len(e.ids) for e in t.encode_batch(texts, add_special_tokens=False)] for k, t in toks.items()}
    for j, (ds, task, i, c, q, meta) in enumerate(buf):
        rows.append((ds, task, i, hashlib.sha256(c.strip().encode()).hexdigest()[:16], len(c), L["nano"][j], L["llama"][j], meta))
    buf.clear()
cur = 0
for it in items():
    buf.append(it); cur += len(it[3])
    if len(buf) >= 16 or cur > 20_000_000: flush(); cur = 0
flush()
with open(OUT, "w") as f:
    f.write("dataset\ttask\tidx\tctx_sha16\tchars\tnano\tllama\tmeta\n")
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
print("rows", len(rows))
