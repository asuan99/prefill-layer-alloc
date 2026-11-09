#!/usr/bin/env python3
"""Motivation workloads: shared constants and small helpers (CPU only, stdlib only).

Workload ids (names, never measurement numbers; W1/W3 in workloads.py are a different namespace):
  sharegpt_first   user-first-turn ShareGPT (ph_trace.build_sharegpt_lengths; n_in, n_out; ignore_eos=True fixed output)
  openchat_rounds  openchat_sharegpt4 rounds, cumulative input, n_in + n_out <= 8192 (ignore_eos=True fixed output)
  arxiv_summ       arXiv summarization 16k, natural output (ignore_eos=False, max_new_tokens cap 1024)
  loogle_longdep   LooGLE long-dependency QA, output = answer length (ignore_eos=True), long stream of the interference cell

Token file format "pdmux-wl-tokens/v1" (little-endian): MAGIC | uint64 header_len | header JSON | int32 token payload.
Each record = [record_id, [[offset, n], ...], n_input, n_output]; the input ids are the concatenation of the segments, so
shared content (LooGLE document, openchat conversation prefix) is stored once.  Readers need only this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[2]                                      # workspace/engine-port
PH_DIR = ENG / "results" / "cap_char" / "policy_harness"   # ph_trace.py (existing trace logic, reused)
MG_DIR = ENG / "scripts" / "multigpu"                      # mg_x_pretokenize.py (existing arXiv token file reader)
for p in (str(PH_DIR), str(MG_DIR)):                       # scripts are run with python -I (no script dir on sys.path)
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- registered constants -------------------------------------------------------------------------------------------
TOKENIZER_SHA256 = "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f"   # Nano-9B-v2(-Base) tokenizer.json
SHAREGPT_V3_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"  # policy_runs/*/sharegpt.sha256
SHAREGPT_POOL_SHA256 = "f3bd82bc87826c9aa06d525c6296a7a3b0a503d14d18f86184d30c8100b92255"  # w1_lengths.json of the policy Jobs
ARXSUM_TOKENS_SHA256 = "4c32f1af8eff07d7d0ad39006423163895181381c0d83a73a5f0f5b1ce76888f"  # mg_const.X_TOKENS_SHA256
OPENCHAT_MAX_TOTAL = 8192
LOOGLE_MAX_INPUT = 94800                                   # chunk-less prefill limit, workload_comparison.html (2026-11-09)
LOOGLE_CTX = 96000                                         # separate boot with --context-length / --max-prefill-tokens ~96k
SEEDS = (20261111, 20261112, 20261113, 20261114)           # registered seeds (n >= 4)
SPARE_SEEDS = (20261115, 20261116)                         # declared re-runs only
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
WORKLOADS = ("sharegpt_first", "openchat_rounds", "arxiv_summ", "loogle_longdep")

TOK_MAGIC = b"PDMUX-WL-TOKENS v1\n"
TOK_FORMAT = "pdmux-wl-tokens/v1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def q(v, p):
    """Nearest-rank quantile, the rule of the earlier length checks (workload_comparison.html)."""
    s = sorted(v)
    return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]


def dist(v):
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return {"n": len(v), "mean": round(m, 3), "cv": round(sd / m, 4) if m else None, "min": min(v), "max": max(v),
            **{f"p{int(round(p * 100))}": q(v, p) for p in QUANTILES}}


def lengths_sha256(pairs):
    a = array("q")
    for i, o in pairs:
        a.append(int(i))
        a.append(int(o))
    if sys.byteorder != "little":
        a.byteswap()
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---- token file ------------------------------------------------------------------------------------------------------
def write_token_file(path, payload, records, extra):
    """payload: array('i'); records: [[record_id, [[offset, n], ...], n_input, n_output], ...]."""
    assert all(sum(n for _o, n in r[1]) == r[2] for r in records), "segment lengths must sum to n_input"
    header = dict(extra, format=TOK_FORMAT, n_records=len(records), n_payload_tokens=len(payload), records=records,
                  lengths_sha256=lengths_sha256([(r[2], r[3]) for r in records]))
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    if sys.byteorder != "little":
        payload.byteswap()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(TOK_MAGIC)
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(payload.tobytes())
    return header


def read_token_header(path):
    with open(path, "rb") as f:
        if f.read(len(TOK_MAGIC)) != TOK_MAGIC:
            raise ValueError(f"{path}: not a {TOK_FORMAT} file")
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h["_payload_start"] = len(TOK_MAGIC) + 8 + n
    return h


def record_input_ids(path, header, record_id):
    rec = next(r for r in header["records"] if r[0] == record_id)
    out = array("i")
    with open(path, "rb") as f:
        for off, n in rec[1]:
            f.seek(header["_payload_start"] + 4 * int(off))
            b = array("i")
            b.frombytes(f.read(4 * int(n)))
            out.extend(b)
    if sys.byteorder != "little":
        out.byteswap()
    if len(out) != rec[2]:
        raise ValueError(f"record {record_id}: short read")
    return out.tolist()


def load_pool(name, lengths_dir=None):
    """rows = [[record_id or None, n_in, n_out], ...] in the length-table order."""
    d = Path(lengths_dir or HERE / "lengths")
    data = json.load(open(d / f"{name}.json"))
    if name == "sharegpt_first":
        return [[None, int(i), int(o)] for i, o in data]
    return [[int(r[0]), int(r[-2]), int(r[-1])] for r in data]
