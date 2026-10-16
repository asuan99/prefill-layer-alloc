#!/usr/bin/env python3
"""shape X (arXiv-Sum 16k) pre-tokenized token file -- PREREG_MULTIGPU rev12 §19-2 (R-8, F-7), §19-4 (floor document).  CPU only.

  build   --parquet P --tokenizer T --out F.bin [--report R.json]   write the token file (deterministic bytes)
  verify  F.bin [--sha256 H]                                           re-read, re-derive every registered statistic, print them
  lengths F.bin                                                         print the sha256 of the retained (input, output) length array

INPUTS (registered; a mismatch refuses to build -- ABORT_DATASET semantics, nothing is written):
  parquet   ccdv/arxiv-summarization document/test-00000-of-00001.parquet, sha256 X_PARQUET_SHA256 (columns article, abstract)
  tokenizer nvidia/NVIDIA-Nemotron-Nano-9B-v2 tokenizer.json, sha256 X_TOKENIZER_SHA256
TOKENIZATION: tokenizers.Tokenizer.from_file(T).encode_batch(<column>, add_special_tokens=False) -- the same call as the
  ARXSUM_STATS script (arxsum_stats.py), applied to `article` (input) and `abstract` (output length only).
RETAIN RULE: 0 < input <= X_MAX_INPUT (16384) and output >= X_MIN_OUTPUT (2).  doc_id = parquet row index.
FLOOR DOCUMENT (§19-4 F-7, literal): candidates = parquet rows whose `article` token count >= 16384, row index ascending;
  rng = numpy.random.default_rng(20261012); index = int(rng.integers(0, len(candidates))) called ONCE; floor doc = candidates[index].
  Its first 16384 tokens are stored (every floor point L uses the first L of them).
LENGTH ARRAY (F-7): retained docs in doc_id ascending order, int64 little-endian, interleaved (input_0, output_0, input_1, ...);
  its sha256 is `lengths_sha256` in the header and must be registered.

FILE FORMAT v1 (all little-endian):
  MAGIC (25 bytes) | uint64 header_len | header JSON (utf-8, sort_keys, compact separators) | int32 token payload
  header.docs = [[doc_id, token_offset, n_input, n_output], ...] (retained, doc_id ascending); header.floor = {...}; offsets count
  int32 tokens from the start of the payload.  Readers: read_header / doc_input_ids / floor_input_ids (stdlib only, used by
  scripts/multigpu/mg_loadgen.py and mg_boot.py inside the Job).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from array import array
from pathlib import Path

MAGIC = b"PDMUX-ARXSUM-X-TOKENS v1\n"
FORMAT = "pdmux-arxsum-x-tokens/v1"
X_PARQUET_SHA256 = "b34529dfeab4aabd4754cecc1603336a3449f66efee87f9b37192aa4c9dd91a1"
X_TOKENIZER_SHA256 = "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f"
X_MAX_INPUT = 16384
X_MIN_OUTPUT = 2
FLOOR_MIN_TOKENS = 16384
FLOOR_STORE_TOKENS = 16384
FLOOR_RNG_SEED = 20261012
FLOOR_RNG_CALL = "numpy.random.default_rng(20261012).integers(0, len(candidates)) (one call)"
ENCODE_CALL = "tokenizers.Tokenizer.from_file(tokenizer.json).encode_batch(column, add_special_tokens=False)"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def q(xs, p):
    """ARXSUM_STATS quantile: sorted(xs)[min(n-1, int(p*n))]."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def lengths_sha256(pairs):
    a = array("q")
    for i, o in pairs:
        a.append(int(i))
        a.append(int(o))
    if sys.byteorder != "little":
        a.byteswap()
    return hashlib.sha256(a.tobytes()).hexdigest()


def stats_of(pairs):
    I = [i for i, _ in pairs]
    O = [o for _, o in pairs]
    return {"n": len(pairs), "input_mean": round(sum(I) / len(I)), "input_p50": q(I, .5), "input_p99": q(I, .99),
            "output_mean": round(sum(O) / len(O)), "output_p50": q(O, .5), "output_p99": q(O, .99), "output_max": max(O),
            "input_max": max(I), "max_input_plus_output": max(i + o for i, o in pairs)}


def build(parquet, tokenizer, out, report=None):
    ps, ts = sha256_file(parquet), sha256_file(tokenizer)
    if ps != X_PARQUET_SHA256 or ts != X_TOKENIZER_SHA256:
        print(f"ABORT_DATASET parquet sha {ps[:16]} (want {X_PARQUET_SHA256[:16]}) tokenizer sha {ts[:16]} (want {X_TOKENIZER_SHA256[:16]})",
              file=sys.stderr)
        return 2
    import numpy as np
    import pyarrow.parquet as pq
    import tokenizers
    from tokenizers import Tokenizer
    t = pq.read_table(parquet, columns=["article", "abstract"]).to_pydict()
    tk = Tokenizer.from_file(str(tokenizer))
    arts = [e.ids for e in tk.encode_batch(t["article"], add_special_tokens=False)]
    outs = [len(e.ids) for e in tk.encode_batch(t["abstract"], add_special_tokens=False)]
    retained = [d for d in range(len(arts)) if 0 < len(arts[d]) <= X_MAX_INPUT and outs[d] >= X_MIN_OUTPUT]
    cands = [d for d in range(len(arts)) if len(arts[d]) >= FLOOR_MIN_TOKENS]
    rng = np.random.default_rng(FLOOR_RNG_SEED)
    idx = int(rng.integers(0, len(cands)))
    fdoc = cands[idx]
    floor = {"doc_id": fdoc, "n_tokens_total": len(arts[fdoc]), "candidates": len(cands),
             "candidate_rule": f"article tokens >= {FLOOR_MIN_TOKENS}, parquet row ascending", "rng": FLOOR_RNG_CALL, "index": idx}
    extra = {"source_parquet_sha256": ps, "tokenizer_json_sha256": ts, "tokenizers_version": tokenizers.__version__,
             "encode_call": ENCODE_CALL, "n_rows": len(arts),
             "n_excluded_output_lt2": sum(1 for d in range(len(arts)) if 0 < len(arts[d]) <= X_MAX_INPUT and outs[d] < X_MIN_OUTPUT)}
    header = write_file(out, {d: (arts[d], outs[d]) for d in retained}, floor, arts[fdoc][:FLOOR_STORE_TOKENS], extra)
    out = Path(out)
    rep = {k: v for k, v in header.items() if k != "docs"}
    rep["file_sha256"] = sha256_file(out)
    rep["file_bytes"] = out.stat().st_size
    rep["builder_sha256"] = sha256_file(__file__)
    if report:
        Path(report).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=1, sort_keys=True))
    return 0


def write_file(out, docs, floor, floor_tokens, extra=None):
    """Write a v1 token file.  docs: {doc_id: (input_ids, n_output)} (written doc_id ascending); floor: dict (doc_id, ...);
    floor_tokens: the stored floor prefix.  Returns the header.  (build() and the CPU tests use this one writer.)"""
    payload = array("i")
    rows = []
    for d in sorted(docs):
        ids, n_out = docs[d]
        rows.append([int(d), len(payload), len(ids), int(n_out)])
        payload.extend(int(x) for x in ids)
    floor = dict(floor, token_offset=len(payload), n_stored=len(floor_tokens))
    payload.extend(int(x) for x in floor_tokens)
    pairs = [(r[2], r[3]) for r in rows]
    header = dict(extra or {}, format=FORMAT, retain_rule=f"0 < input <= {X_MAX_INPUT} and output >= {X_MIN_OUTPUT}",
                  n_retained=len(rows), lengths_layout="retained docs, doc_id ascending, int64 LE interleaved (input, output)",
                  lengths_sha256=lengths_sha256(pairs), stats=stats_of(pairs), docs=rows, floor=floor, n_payload_tokens=len(payload))
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    if sys.byteorder != "little":
        payload.byteswap()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(payload.tobytes())
    return header


# ----------------------------------------------------------------------------- readers (stdlib only; used in the Job)
def read_header(path):
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a {FORMAT} file")
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    if h.get("format") != FORMAT:
        raise ValueError(f"{path}: format {h.get('format')!r}")
    h["_payload_start"] = len(MAGIC) + 8 + n
    return h


def _read_tokens(path, header, offset, n):
    a = array("i")
    with open(path, "rb") as f:
        f.seek(header["_payload_start"] + 4 * int(offset))
        a.frombytes(f.read(4 * int(n)))
    if sys.byteorder != "little":
        a.byteswap()
    if len(a) != int(n):
        raise ValueError(f"{path}: short read at offset {offset}")
    return a.tolist()


def doc_index(header):
    """doc_id -> (offset, n_input, n_output)."""
    return {int(d): (int(o), int(i), int(out)) for d, o, i, out in header["docs"]}


def doc_input_ids(path, header, doc_ids, index=None):
    index = index or doc_index(header)
    return {int(d): _read_tokens(path, header, index[int(d)][0], index[int(d)][1]) for d in doc_ids}


def floor_input_ids(path, header, n):
    fl = header["floor"]
    if int(n) > int(fl["n_stored"]):
        raise ValueError(f"floor length {n} > stored {fl['n_stored']}")
    return _read_tokens(path, header, fl["token_offset"], n)


def verify(path, want_sha=None):
    h = read_header(path)
    pairs = [(int(i), int(o)) for _d, _o, i, o in h["docs"]]
    ids = [int(d) for d, *_ in h["docs"]]
    problems = []
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        problems.append("docs not doc_id ascending / unique")
    if lengths_sha256(pairs) != h["lengths_sha256"]:
        problems.append("lengths_sha256 does not match the docs table")
    if stats_of(pairs) != h["stats"]:
        problems.append("stats do not match the docs table")
    off = 0
    for d, o, i, _out in h["docs"]:
        if int(o) != off:
            problems.append(f"doc {d}: offset {o} != {off}")
            break
        off += int(i)
    if int(h["floor"]["token_offset"]) != off or off + int(h["floor"]["n_stored"]) != int(h["n_payload_tokens"]):
        problems.append("floor offset / payload length inconsistent")
    size = Path(path).stat().st_size
    if size != h["_payload_start"] + 4 * int(h["n_payload_tokens"]):
        problems.append(f"file size {size} != header + payload")
    sha = sha256_file(path)
    if want_sha and sha != want_sha:
        problems.append(f"file sha256 {sha[:16]} != registered {want_sha[:16]}")
    return {"file_sha256": sha, "n_retained": h["n_retained"], "lengths_sha256": h["lengths_sha256"], "stats": h["stats"],
            "floor": h["floor"], "tokenizers_version": h["tokenizers_version"], "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parquet", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--report", default=None)
    v = sub.add_parser("verify")
    v.add_argument("file")
    v.add_argument("--sha256", default=None)
    ln = sub.add_parser("lengths")
    ln.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return build(a.parquet, a.tokenizer, a.out, a.report)
    if a.cmd == "verify":
        r = verify(a.file, a.sha256)
        print(json.dumps(r, indent=1, sort_keys=True))
        return 0 if not r["problems"] else 2
    print(read_header(a.file)["lengths_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
