#!/usr/bin/env python3
"""Registered motivation workloads in the runner (2026-11-11).  CPU-side; no server.

A run's "workload" field selects how its segments are built:

  W1                    the sealed stock-baseline path (sb_runner.build_segments, ShareGPT length pool, [1]*n prompt) -- rehearsal
                        fixtures and the earlier tests keep using it
  sharegpt_first        user-first-turn ShareGPT lengths, [1]*n prompt (no text; manifest "harness convention")
  openchat_rounds       openchat_sharegpt4 rounds, real conversation-prefix token ids (mw token file)
  arxiv_summ            arXiv article token ids, output = abstract length (main condition, user decision 2026-11-10)
  arxiv_summ_out1024    the same arrivals and prompts, every request generates exactly 1,024 tokens (variant condition)

Segments come from workloads/gen_traces.instantiate(workload, seed, pool, rate, n=n_requests): the registered master stream of
(workload, seed) scaled by the rate, so every arm and every rate of one seed sees the same request sequence.  Records carry doc_id;
the trace hash (card, analyzer) is computed WITHOUT input_ids, and the token ids are attached only for the replay, from a token file
whose sha256 must equal the registered one (workloads/manifest.json, mw_lib.ARXSUM_TOKENS_SHA256).

  interference_with_long / interference_without_long
                        the paired interference cell: the sharegpt_first stream at a fixed short rate, with / without the LooGLE long
                        stream (5 % of all requests).  A run is defined by duration_s; LooGLE requests carry phase "LONG" and real
                        token ids, the short requests the [1]*n prompt.  Both members boot with context 96,000 (mo_intf_spec).

Token files are looked up in $PDMUX_MO_TOKEN_DIR (if set), else at their registered cluster-volume path.
LooGLE alone (no short stream) is not wired.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
WL = HERE / "workloads"
for _p in (str(WL),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mw_lib as L  # noqa: E402
import gen_traces as G  # noqa: E402
import mg_x_pretokenize as X  # noqa: E402  (arXiv token file reader; on sys.path through mw_lib)
import sb_runner as RU  # noqa: E402

_MANIFEST = json.loads((WL / "manifest.json").read_text())


def _tok(name):
    return (_MANIFEST["workloads"][name].get("token_file") or {})


WORKLOADS = {
    "sharegpt_first": {"pool": "sharegpt_first", "tokens": None, "fixed_output": None},
    "openchat_rounds": {"pool": "openchat_rounds", "fixed_output": None,
                        "tokens": {"format": "mw", "name": _tok("openchat_rounds").get("name"), "sha256": _tok("openchat_rounds").get("sha256"),
                                   "volume_path": "/data/datasets/motivation_wl/openchat_rounds_tokens_v1.bin"}},
    "arxiv_summ": {"pool": "arxiv_summ", "fixed_output": None,
                   "tokens": {"format": "arxsum", "name": "arxsum_x_tokens_v1.bin", "sha256": L.ARXSUM_TOKENS_SHA256,
                              "volume_path": "/data/datasets/arxsum_x/arxsum_x_tokens_v1.bin"}},
}
WORKLOADS["arxiv_summ_out1024"] = dict(WORKLOADS["arxiv_summ"], fixed_output=G.ARXIV_VARIANT_OUTPUT)
# Interference pair (2026-11-12): the SAME sharegpt_first stream at a fixed short rate, with / without the LooGLE long stream
# (gen_traces.interference: 5 % of all requests, count fixed).  Both members use the same token file (the "without" member sends none
# of its records but resolves the same file, so the two cells of a pair have one preflight) and report the same pool facts.
_LOOGLE_TOKENS = {"format": "mw", "name": _tok("loogle_longdep").get("name"), "sha256": _tok("loogle_longdep").get("sha256"),
                  "volume_path": "/data/datasets/motivation_wl/loogle_longdep_tokens_v1.bin"}
INTERFERENCE = {"interference_with_long": True, "interference_without_long": False}
for _n, _w in INTERFERENCE.items():
    WORKLOADS[_n] = {"pool": "sharegpt_first", "long_pool": "loogle_longdep", "with_long": _w, "fixed_output": None, "tokens": _LOOGLE_TOKENS}
LONG_PHASE = "LONG"                 # phase tag of the LooGLE requests in a trace / raw record (gen_traces.interference)
INTERFERENCE_CONTEXT = L.LOOGLE_CTX  # 96,000: the boot context of BOTH members (the pair differs in the long stream only)
SEEDS = tuple(L.SEEDS)
_POOLS = {}


def is_registered(workload):
    return workload in WORKLOADS


def pool(workload):
    name = WORKLOADS[workload]["pool"]
    if name not in _POOLS:
        _POOLS[name] = L.load_pool(name)
    return _POOLS[name]


_INTF_CACHE = {}


def long_pool():
    if "loogle_longdep" not in _POOLS:
        _POOLS["loogle_longdep"] = L.load_pool("loogle_longdep")
    return _POOLS["loogle_longdep"]


def n_interference(seed, rate, duration_s, with_long):
    """Request count of an interference cell (the plan records it; the analyzer checks the raw record against it)."""
    run = {"run_id": "count", "workload": "interference_with_long" if with_long else "interference_without_long", "seed": seed,
           "rate": rate, "duration_s": duration_s}
    return len(build_segments(run, None)[0]["requests"])


def build_segments(run, w1_pool):
    """Segments of one run, without token ids (the card and the trace hash see exactly this)."""
    wl = run["workload"]
    if wl == "W1":
        return RU.build_segments(run, w1_pool, [])
    if not is_registered(wl):
        raise ValueError(f"unregistered workload {wl!r}")
    if run.get("rate") is None:
        raise RuntimeError(f"ABORT_RATE_UNFILLED: {run['run_id']}")
    d = WORKLOADS[wl]
    if wl in INTERFERENCE:
        if run.get("duration_s") is None:
            raise ValueError(f"{run['run_id']}: an interference run is defined by its duration (duration_s), not by a request count")
        key = (run["seed"], float(run["rate"]), float(run["duration_s"]), d["with_long"], G.interference)
        if key not in _INTF_CACHE:                    # a plan, its card and its checker ask for the same cell many times
            _INTF_CACHE[key] = G.interference(run["seed"], pool(wl), long_pool(), run["rate"], float(run["duration_s"]), d["with_long"])[0]
        return [dict(s, segment_id=f"{run['run_id']}", requests=[dict(r) for r in s["requests"]]) for s in _INTF_CACHE[key]]
    segs = G.instantiate(d["pool"], run["seed"], pool(wl), run["rate"], n=run["n_requests"])
    if d["fixed_output"]:
        segs = G.fix_output(segs, d["fixed_output"])
    return [dict(s, segment_id=f"{run['run_id']}") for s in segs]


def pool_facts(workload, w1_pool, pool_facts_fn):
    if workload == "W1":
        return pool_facts_fn(w1_pool)
    rows = pool(workload)
    out = pool_facts_fn([[i, o] for _r, i, o in rows])
    if workload in INTERFERENCE:                      # one fact set for both members of the pair (short pool + long pool)
        out["long_pool"] = pool_facts_fn([[i, o] for _r, i, o in long_pool()])
        out.update(workload="interference", token_file_sha256=_LOOGLE_TOKENS["sha256"], fixed_output=None, long_share=G.LONG_SHARE,
                   context=INTERFERENCE_CONTEXT)
        return out
    out["workload"] = workload
    out["token_file_sha256"] = (WORKLOADS[workload]["tokens"] or {}).get("sha256")
    out["fixed_output"] = WORKLOADS[workload]["fixed_output"]
    return out


class TokenStore:
    """doc_id -> input_ids for one workload's token file; the file's sha256 is checked once, before any request is sent."""

    def __init__(self, workload, token_dir=None):
        self.workload = workload
        spec = WORKLOADS[workload]["tokens"]
        self.spec = spec
        if spec is None:
            self.path = None
            return
        d = token_dir or os.environ.get("PDMUX_MO_TOKEN_DIR")
        self.path = Path(d) / spec["name"] if d else Path(spec["volume_path"])
        if not self.path.exists():
            raise FileNotFoundError(f"ABORT_TOKEN_FILE_MISSING: {self.path}")
        got = L.sha256_file(self.path)
        if got != spec["sha256"]:
            raise ValueError(f"ABORT_TOKEN_FILE_SHA: {self.path} sha256 {got[:16]} != registered {spec['sha256'][:16]}")
        if spec["format"] == "mw":
            self.header = L.read_token_header(self.path)
            self.index = {int(r[0]): r for r in self.header["records"]}
        else:
            self.header = X.read_header(self.path)
            self.index = X.doc_index(self.header)

    def input_ids(self, doc_id, n_input):
        if self.path is None:
            raise ValueError(f"{self.workload}: no token file, doc_id {doc_id} cannot be resolved")
        if self.spec["format"] == "mw":
            ids = L.record_input_ids(self.path, {**self.header, "records": [self.index[int(doc_id)]]}, int(doc_id))
        else:
            ids = X.doc_input_ids(self.path, self.header, [int(doc_id)], self.index)[int(doc_id)]
        if len(ids) != int(n_input):
            raise ValueError(f"{self.workload} doc {doc_id}: {len(ids)} token ids != input_tokens {n_input}")
        return ids


def with_input_ids(segments, store):
    """Copies of the segments whose doc_id requests carry input_ids (what the client sends).  No doc_id -> unchanged ([1]*n)."""
    if store is None or store.path is None:
        return segments
    return [dict(s, requests=[dict(r, input_ids=store.input_ids(r["doc_id"], r["input_tokens"])) if "doc_id" in r else r
                              for r in s["requests"]]) for s in segments]


def store_for(workload, token_dir=None):
    if workload == "W1" or WORKLOADS[workload]["tokens"] is None:
        return None
    return TokenStore(workload, token_dir)
