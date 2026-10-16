#!/usr/bin/env python3
"""M2g server-fact reference for rev12 KV_POOL_DRIFT / DTYPE_UNVERIFIED (PREREG rev12 §19-3, F-4).  CPU only.

  build --job J1=<m2g run dir> --job J2=... --job J4=... --job J8=... --out M2G_SERVER_REFERENCE.json
  extract-log server.log          print the Mamba allocation facts of one log

Per (Job, config, arm) of the M2g PASS run: max_total_num_tokens (cell_meta.json, as the boot recorded it) and the
`Mamba Cache is allocated` lines of server.log (memory_pool.py: max_mamba_cache_size, conv_state size GB, ssm_state size GB,
one line per TP rank / DP replica).  ssm_state size is the EFFECTIVE-dtype witness: float32 SSM state is twice bfloat16 at the
same slot count (F-4: the server_info `mamba_ssm_dtype` field only echoes the argument).  The reference is built ONCE from the
four PASS runs (M2G_PASS_TABLE_<J>.json sha matched to the run directory) and committed; the M3 labels compare against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MAMBA_RE = re.compile(r"Mamba Cache is allocated\. max_mamba_cache_size: *(\d+), conv_state size: *([0-9.]+)GB, "
                      r"ssm_state size: *([0-9.]+)GB")
BOOT_RE = re.compile(r"^m2g_(N1|TP\d+|DP\d+)_(fused_sym|fused|agnostic|d44)$")


def mamba_facts(log_text):
    rows = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in MAMBA_RE.finditer(log_text or "")]
    return {"mamba_lines": len(rows), "max_mamba_cache_size": [r[0] for r in rows], "conv_state_gb": [r[1] for r in rows],
            "ssm_state_gb": [r[2] for r in rows]}


def build(jobs, out):
    ref = {"schema": "mg_m2g_server_reference/v1", "entries": {}, "sources": {}}
    for J, d in sorted(jobs.items()):
        d = Path(d)
        pt = d / f"M2G_PASS_TABLE_{J}.json"
        ref["sources"][J] = {"dir": d.name, "pass_table_sha256": hashlib.sha256(pt.read_bytes()).hexdigest() if pt.exists() else None}
        for bd in sorted((d / "boots").iterdir()):
            m = BOOT_RE.match(bd.name)
            if not m:
                continue
            cfg, arm = m.groups()
            cm = json.loads((bd / "cell_meta.json").read_text()) if (bd / "cell_meta.json").exists() else {}
            log = bd / "server.log"
            e = {"max_total_num_tokens": cm.get("max_total_num_tokens"), "server_log_sha256": None}
            if log.exists():
                raw = log.read_bytes()
                e["server_log_sha256"] = hashlib.sha256(raw).hexdigest()
                e.update(mamba_facts(raw.decode("utf-8", "replace")))
            ref["entries"][f"{J}|{cfg}|{arm}"] = e
    Path(out).write_text(json.dumps(ref, indent=1, sort_keys=True) + "\n")
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--job", action="append", required=True)
    b.add_argument("--out", required=True)
    e = sub.add_parser("extract-log")
    e.add_argument("log")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        r = build(dict(x.split("=", 1) for x in a.job), a.out)
        print(len(r["entries"]), "entries")
        return 0
    print(json.dumps(mamba_facts(Path(a.log).read_text(errors="replace")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
