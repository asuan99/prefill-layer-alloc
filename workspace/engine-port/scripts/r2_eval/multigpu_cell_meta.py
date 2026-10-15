#!/usr/bin/env python3
"""Per-cell metadata for multi-GPU (TP/DP) runs of engine_bench_runner.sh.

Stdlib only, CPU only.  Two subcommands:

  comm-env      Record the communication-related environment of this cell.  The DEFAULT is to
                record "unset" -- choosing a value is a PREREG matter (design
                reports/multigpu_scaling_design_2026-10-12.md section 5-2, audit C-10).  A value
                can be pinned from a PREREG with PDMUX_COMM_ENV_PIN="K=V,K=V" (runner exports
                it); the record says whether each value was pinned, inherited, or unset.
                SGLANG_USE_JIT_ALL_REDUCE set => exit 3 (jit all-reduce is opt-in, design R-1)
                unless --allow-jit-all-reduce.
  server-facts  After boot, extract the KV/mamba capacity the server really got from server.log
                (and /server_info JSON when present): max_total_num_tokens, mamba cache size,
                effective max_running_requests (audit C-5).  TP and DP cells differ here by
                construction (TP shards weights, so the pools grow), so every cell records them.
"""

import argparse
import json
import os
import re
import sys

# Registered list: the variables named by design section 5-2 / audit C-10, plus the neighbours
# that change the collective path.  Order is the record order.
COMM_ENV_VARS = (
    "NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS",
    "CUDA_DEVICE_MAX_CONNECTIONS", "SGLANG_USE_JIT_ALL_REDUCE", "SGLANG_CUSTOM_ALLREDUCE_ALGO",
    "NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE", "NCCL_NVLS_ENABLE", "NCCL_IB_DISABLE",
    "NCCL_SHM_DISABLE", "NCCL_DEBUG",
)
FALSY = ("", "0", "false", "False", "no")


def parse_pin(text):
    pins = {}
    for item in [x for x in (text or "").split(",") if x.strip()]:
        key, sep, val = item.partition("=")
        if not sep or key.strip() not in COMM_ENV_VARS:
            raise ValueError(f"PDMUX_COMM_ENV_PIN entry {item!r} is not K=V of a registered variable")
        pins[key.strip()] = val
    return pins


def comm_env_record(environ, pins=None):
    pins = pins or {}
    rec = {}
    for name in COMM_ENV_VARS:
        val = environ.get(name)
        rec[name] = {"value": val,
                     "state": "unset" if val is None else ("pinned" if name in pins else "inherited")}
    return rec


def jit_all_reduce_enabled(environ):
    return environ.get("SGLANG_USE_JIT_ALL_REDUCE", "") not in FALSY


def cmd_comm_env(a):
    pins = parse_pin(os.environ.get("PDMUX_COMM_ENV_PIN", ""))
    rec = {"vars": comm_env_record(os.environ, pins),
           "pinned": pins,
           "jit_all_reduce_enabled": jit_all_reduce_enabled(os.environ)}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    if rec["jit_all_reduce_enabled"] and not a.allow_jit_all_reduce:
        print("ERROR: SGLANG_USE_JIT_ALL_REDUCE is set; jit all-reduce is opt-in and not part of "
              "the registered baseline (set PDMUX_ALLOW_JIT_ALL_REDUCE=1 only if a PREREG says so)",
              file=sys.stderr)
        return 3
    return 0


def _ints(pattern, text):
    return [int(x) for x in re.findall(pattern, text)]


def server_facts(log_text, server_info=None, tp=1, dp=1, max_running_requested=None):
    """Pure function: server.log text (+ parsed /server_info) -> facts dict."""
    facts = {
        "tp_size": tp, "dp_size": dp,
        "max_running_requests_requested": max_running_requested,
        # one 'max_total_num_tokens=' line is printed per scheduler (tp_rank 0 of each DP replica)
        "max_total_num_tokens": _ints(r"max_total_num_tokens=([0-9]+)", log_text),
        "max_running_requests_logged": _ints(r"max_running_requests=([0-9]+)", log_text),
        "max_mamba_cache_size": _ints(r"max_mamba_cache_size: *([0-9]+)", log_text),
        "kv_cache_tokens": _ints(r"KV Cache is allocated\. #tokens: *([0-9]+)", log_text),
    }
    n_sched = len(facts["max_total_num_tokens"])
    facts["scheduler_lines"] = n_sched
    facts["scheduler_lines_expected"] = dp
    facts["scheduler_lines_match_dp"] = n_sched == dp
    facts["mamba_lines_note"] = ("mamba allocation is logged per TP rank and DP replica; "
                                 "compare across ranks, do not sum")
    if server_info:
        states = server_info.get("internal_states") or []
        facts["server_info"] = {
            "token_capacity": [s.get("memory_usage", {}).get("token_capacity") for s in states],
            "effective_max_running_requests_per_dp":
                [s.get("effective_max_running_requests_per_dp") for s in states],
            "max_total_num_tokens": server_info.get("max_total_num_tokens"),
            "max_req_input_len": server_info.get("max_req_input_len"),
            "dp_size": server_info.get("dp_size"), "tp_size": server_info.get("tp_size"),
            "load_balance_method": server_info.get("load_balance_method"),
            "max_running_requests": server_info.get("max_running_requests"),
        }
    facts["capacity_recorded"] = bool(facts["max_total_num_tokens"])
    return facts


def cmd_server_facts(a):
    log = open(a.log, errors="replace").read() if os.path.exists(a.log) else ""
    info = None
    if a.server_info and os.path.exists(a.server_info):
        try:
            info = json.load(open(a.server_info))
        except ValueError:
            info = None
    facts = server_facts(log, info, a.tp, a.dp, a.max_running_requested)
    facts["gpu_ids"] = a.gpu_ids or None
    with open(a.out, "w") as f:
        json.dump(facts, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("comm-env")
    s.add_argument("--out", required=True)
    s.add_argument("--allow-jit-all-reduce", action="store_true")
    s = sub.add_parser("server-facts")
    s.add_argument("--log", required=True)
    s.add_argument("--server-info", default="")
    s.add_argument("--out", required=True)
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--dp", type=int, default=1)
    s.add_argument("--max-running-requested", type=int, default=None)
    s.add_argument("--gpu-ids", default="")
    a = ap.parse_args(argv)
    return {"comm-env": cmd_comm_env, "server-facts": cmd_server_facts}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
