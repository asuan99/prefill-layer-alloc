#!/usr/bin/env python3
"""Motivation workload pool builders (CPU only).  Run: python3 -I build_pools.py <cmd> ...  Data paths are arguments.

  sharegpt-first  --sharegpt J --tokenizer T [--out-dir D]                         ph_trace.build_sharegpt_lengths (unchanged logic)
  openchat        --openchat J --tokenizer T --tokens F [--check OC_ROUNDS.json]   rounds with cumulative input, total <= 8192
  arxiv           --tokens F [--cap2-segments DIR]                                 existing arXiv-Sum 16k file: verify, extract lengths
  loogle          --jsonl J --tokenizer T --tokens F                               longdep_qa, output = answer length
  manifest                                                                         merge manifest fragments into manifest.json

Every builder refuses a tokenizer whose sha256 differs from the registered one and writes lengths/<name>.json (small, committed),
manifest_parts/<name>.json and, where listed, a token file (raw, not committed).  Same inputs -> byte-identical outputs.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mw_lib as L  # noqa: E402

MANI_DIR = HERE / "manifest_parts"


def tokenizer(path):
    sha = L.sha256_file(path)
    if sha != L.TOKENIZER_SHA256:
        raise SystemExit(f"ABORT_DATASET tokenizer sha {sha[:16]} != registered {L.TOKENIZER_SHA256[:16]}")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def write_part(name, part):
    MANI_DIR.mkdir(exist_ok=True)
    part = dict(part, workload=name, tokenizer_sha256=L.TOKENIZER_SHA256)
    (MANI_DIR / f"{name}.json").write_text(json.dumps(part, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in part.items() if k not in ("input", "output")}, sort_keys=True))
    print(name, "input", json.dumps(part["input"]))
    print(name, "output", json.dumps(part["output"]))


def write_lengths(name, rows):
    p = HERE / "lengths" / f"{name}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(rows, separators=(",", ":")))
    return p


def cmd_sharegpt(a):
    import ph_trace as PH
    tok = tokenizer(a.tokenizer)
    src_sha = L.sha256_file(a.sharegpt)
    if src_sha != L.SHAREGPT_V3_SHA256:
        raise SystemExit(f"ABORT_DATASET sharegpt sha {src_sha[:16]}")
    pool = PH.build_sharegpt_lengths(a.sharegpt, lambda t: len(tok.encode(t, add_special_tokens=False).ids))
    p = Path(a.out_dir) / "sharegpt_first.json" if a.out_dir else HERE / "lengths" / "sharegpt_first.json"
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(json.dumps(pool))                         # same bytes as ph_trace build-w1 (w1_lengths.json)
    sha = L.sha256_file(p)
    if sha != L.SHAREGPT_POOL_SHA256:
        raise SystemExit(f"ABORT_DATASET pool sha {sha[:16]} != registered {L.SHAREGPT_POOL_SHA256[:16]}")
    write_part("sharegpt_first", {
        "source": {"dataset": "ShareGPT_V3_unfiltered_cleaned_split.json", "sha256": src_sha,
                   "volume_path": "/data/hf/raw/ShareGPT_V3_unfiltered_cleaned_split.json"},
        "rule": "first human turn -> first gpt reply; both >= 4 tokens; total <= 4000 (ph_trace.build_sharegpt_lengths)",
        "output_mode": "ignore_eos=True, output_tokens = reply length",
        "prompt": "harness convention: [1]*n_input (length only, no text content)",
        "n_records": len(pool), "pool_sha256": sha, "length_table": "lengths/sharegpt_first.json",
        "input": L.dist([x for x, _ in pool]), "output": L.dist([y for _, y in pool])})
    return 0


def cmd_openchat(a):
    tok = tokenizer(a.tokenizer)
    convs = json.load(open(a.openchat, encoding="utf-8"))
    payload, rows, records, pairs = array("i"), [], [], []
    for ci, conv in enumerate(convs):
        ids_turns, h, kept = [], 0, []
        for it in conv["items"]:
            ids = tok.encode(it["value"], add_special_tokens=False).ids
            t = len(ids)
            if it["from"] in ("gpt", "assistant") and h > 0 and h + t <= L.OPENCHAT_MAX_TOTAL:
                kept.append((h, t))
            h += t
            ids_turns.append(ids)
        if not kept:
            continue
        off = len(payload)
        stream = [x for ids in ids_turns for x in ids]
        need = max(hh for hh, _ in kept)                   # only prefixes are referenced; store up to the longest one
        payload.extend(int(x) for x in stream[:need])
        for ri, (hh, tt) in enumerate(kept):
            rid = len(rows)
            rows.append([rid, ci, ri, hh, tt])
            records.append([rid, [[off, hh]], hh, tt])
            pairs.append([hh, tt])
    if a.check and json.load(open(a.check)) != pairs:
        raise SystemExit("ABORT_DATASET openchat rounds differ from the earlier check file")
    I = [p[0] for p in pairs]
    O = [p[1] for p in pairs]
    if L.q(I, .5) != 1597:
        raise SystemExit(f"ABORT_DATASET openchat p50 {L.q(I, .5)} != 1597 (earlier check)")
    src_sha = L.sha256_file(a.openchat)
    h = L.write_token_file(a.tokens, payload, records, {"workload": "openchat_rounds", "tokenizer_sha256": L.TOKENIZER_SHA256,
                                                        "source_sha256": src_sha, "note": "concatenated turn tokens, no chat template"})
    write_lengths("openchat_rounds", rows)
    write_part("openchat_rounds", {
        "source": {"dataset": "openchat/openchat_sharegpt4_dataset sharegpt_gpt4.json", "sha256": src_sha},
        "rule": f"gpt-turn rounds with cumulative input > 0 and input + reply <= {L.OPENCHAT_MAX_TOTAL}; turn tokens concatenated, no chat template, no special tokens",
        "output_mode": "ignore_eos=True, output_tokens = reply length",
        "prompt": "real text, conversation prefix token ids", "n_records": len(rows), "length_table": "lengths/openchat_rounds.json",
        "lengths_sha256": h["lengths_sha256"],
        "token_file": {"name": Path(a.tokens).name, "sha256": L.sha256_file(a.tokens), "bytes": Path(a.tokens).stat().st_size, "n_payload_tokens": len(payload)},
        "input": L.dist(I), "output": L.dist(O)})
    return 0


def cmd_arxiv(a):
    import mg_x_pretokenize as X
    rep = X.verify(a.tokens, want_sha=L.ARXSUM_TOKENS_SHA256)
    if rep["problems"]:
        raise SystemExit(f"ABORT_DATASET {rep['problems']}")
    h = X.read_header(a.tokens)
    rows = [[int(d), int(i), int(o)] for d, _off, i, o in h["docs"]]
    write_lengths("arxiv_summ", rows)
    I = [r[1] for r in rows]
    O = [r[2] for r in rows]
    extra = {}
    if a.cap2_segments:
        obs = []
        for f in sorted(glob.glob(str(Path(a.cap2_segments) / "*" / "*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                if not r.get("error") and "observed_output_tokens" in r:
                    obs.append(int(r["observed_output_tokens"]))
        extra = {"observed_natural_output_cap2": {**L.dist(obs), "frac_at_cap_1024": round(sum(x >= 1024 for x in obs) / len(obs), 4),
                                                  "source": "policy_runs/pdmux-policy-cap2-e90e233-20261029023336/raw/segments (ignore_eos=False, cap 1024)"}}
    write_part("arxiv_summ", {
        "source": {"dataset": "ccdv/arxiv-summarization test parquet (via mg_x_pretokenize.py)", "token_file_sha256": L.ARXSUM_TOKENS_SHA256,
                   "volume_path": "/data/datasets/arxsum_x/arxsum_x_tokens_v1.bin (already on the cluster volume)"},
        "rule": "existing arxsum_x_tokens_v1.bin: 0 < input <= 16384, abstract >= 2; n_output = abstract token count",
        "output_mode": "ignore_eos=False, max_new_tokens = 1024 (policy harness convention, decision 47); output length is run-determined",
        "prompt": "article token ids only, no summarization instruction",
        "n_records": len(rows), "length_table": "lengths/arxiv_summ.json (doc_id, input, abstract length)",
        "lengths_sha256": L.lengths_sha256([(r[1], r[2]) for r in rows]),
        "input": L.dist(I), "output": L.dist(O), "output_is": "abstract token length (reference); not the served natural output", **extra})
    return 0


def cmd_loogle(a):
    tok = tokenizer(a.tokenizer)
    payload = array("i")
    docs, records, rows, excl = {}, [], [], 0
    src_sha = L.sha256_file(a.jsonl)
    for line in open(a.jsonl, encoding="utf-8"):
        r = json.loads(line)
        if r["doc_id"] not in docs:
            ids = tok.encode(r["context"], add_special_tokens=False).ids
            docs[r["doc_id"]] = (len(docs), len(payload), len(ids))
            payload.extend(int(x) for x in ids)
        doc_idx, doff, nctx = docs[r["doc_id"]]
        ans = r["answer"]
        ans = " ".join(map(str, ans)) if isinstance(ans, list) else str(ans)
        qids = tok.encode(r["question"], add_special_tokens=False).ids if r["question"] else []
        nout = max(1, len(tok.encode(ans, add_special_tokens=False).ids))
        nin = nctx + len(qids)
        if nin > L.LOOGLE_MAX_INPUT or nin + nout > L.LOOGLE_CTX:
            excl += 1
            continue
        qoff = len(payload)
        payload.extend(int(x) for x in qids)
        rid = len(rows)
        rows.append([rid, doc_idx, nin, nout])
        records.append([rid, [[doff, nctx]] + ([[qoff, len(qids)]] if qids else []), nin, nout])
    h = L.write_token_file(a.tokens, payload, records, {"workload": "loogle_longdep", "tokenizer_sha256": L.TOKENIZER_SHA256,
                                                        "source_sha256": src_sha, "note": "context + question tokens, no instruction, documents stored once"})
    write_lengths("loogle_longdep", rows)
    write_part("loogle_longdep", {
        "source": {"dataset": "LooGLE longdep_qa.jsonl", "sha256": src_sha},
        "rule": f"longdep_qa; input = context + question tokens, input <= {L.LOOGLE_MAX_INPUT}, input + output <= {L.LOOGLE_CTX}; output = max(1, answer tokens); {excl} excluded",
        "output_mode": "ignore_eos=True, output_tokens = answer length",
        "prompt": "real text, context + question token ids, no instruction",
        "n_records": len(rows), "n_docs": len(docs), "n_excluded": excl, "length_table": "lengths/loogle_longdep.json",
        "lengths_sha256": h["lengths_sha256"],
        "token_file": {"name": Path(a.tokens).name, "sha256": L.sha256_file(a.tokens), "bytes": Path(a.tokens).stat().st_size, "n_payload_tokens": len(payload)},
        "input": L.dist([r[2] for r in rows]), "output": L.dist([r[3] for r in rows])})
    return 0


def cmd_manifest(a):
    parts = {p.stem: json.load(open(p)) for p in sorted(MANI_DIR.glob("*.json"))}
    missing = [w for w in L.WORKLOADS if w not in parts]
    if missing:
        raise SystemExit(f"missing manifest parts: {missing}")
    out = {"schema": "pdmux-motivation-workload-manifest/v1", "tokenizer_sha256": L.TOKENIZER_SHA256,
           "builder_sha256": {f: L.sha256_file(HERE / f) for f in ("mw_lib.py", "build_pools.py", "gen_traces.py") if (HERE / f).exists()},
           "quantile_rule": "nearest rank (mw_lib.q)", "workloads": parts}
    (HERE / "manifest.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("manifest.json", L.sha256_file(HERE / "manifest.json"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sharegpt-first")
    s.add_argument("--sharegpt", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out-dir")
    s = sub.add_parser("openchat")
    s.add_argument("--openchat", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--tokens", required=True)
    s.add_argument("--check")
    s = sub.add_parser("arxiv")
    s.add_argument("--tokens", required=True)
    s.add_argument("--cap2-segments")
    s = sub.add_parser("loogle")
    s.add_argument("--jsonl", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--tokens", required=True)
    sub.add_parser("manifest")
    a = ap.parse_args(argv)
    return {"sharegpt-first": cmd_sharegpt, "openchat": cmd_openchat, "arxiv": cmd_arxiv, "loogle": cmd_loogle, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
