#!/usr/bin/env python3
"""1 s /proc sampler for FRONTEND_SATURATED (PREREG §4-3, MGP-19/30).  CPU only, stdlib only.

Writes CSV rows `t_mono,role,pid,tid,ticks` once per period.  `ticks` = utime+stime from
/proc/<pid>/stat (fields 14+15, clock ticks) -- the CUMULATIVE counter; the label code
(results/multigpu/mg_label.frontend_stats) differences consecutive rows, so the "1 s interval delta"
is computed from the real sample spacing.  `ps %cpu` (lifetime average) is never used.

Roles (found by walking descendants of --root-pid every tick, SGLang rewrites process titles):
  tokenizer_manager  the root launch_server process itself (TokenizerManager runs in its main thread tree)
  detokenizer        cmdline starts with `sglang::detokenizer`
  dp_controller      cmdline contains `sglang::data_parallel_controller` (DP only)
tid = -1 is the process total; tid >= 0 are the per-thread rows (/proc/<pid>/task/<tid>/stat) of the
same three roles -- recorded so the per-thread maximum (MGP-30: a single saturated thread hides inside
a many-thread process) can be reported; they never enter a label.

usage: mg_proc_sampler.py --root-pid PID --out FILE [--period 1.0]     (stops on SIGTERM / root gone)
"""

import argparse
import os
import signal
import sys
import time


def stat_ticks(path):
    try:
        with open(path) as f:
            data = f.read()
    except OSError:
        return None
    rest = data[data.rfind(")") + 2:].split()
    try:
        return int(rest[11]) + int(rest[12])
    except (IndexError, ValueError):
        return None


def children_map():
    cm = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                data = f.read()
            ppid = int(data[data.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        cm.setdefault(ppid, []).append(int(name))
    return cm


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def classify(pid, root):
    if pid == root:
        return "tokenizer_manager"
    c = cmdline(pid)
    if c.startswith("sglang::detokenizer"):
        return "detokenizer"
    if "sglang::data_parallel_controller" in c:
        return "dp_controller"
    return None


def descendants(root):
    cm, out, stack = children_map(), [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(cm.get(p, []))
    return out


def sample_once(root, writer, with_threads=True):
    now = time.monotonic()
    for pid in descendants(root):
        role = classify(pid, root)
        if role is None:
            continue
        t = stat_ticks(f"/proc/{pid}/stat")
        if t is not None:
            writer.write(f"{now:.6f},{role},{pid},-1,{t}\n")
        if with_threads:
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for tid in tids:
                tt = stat_ticks(f"/proc/{pid}/task/{tid}/stat")
                if tt is not None:
                    writer.write(f"{now:.6f},{role},{pid},{int(tid)},{tt}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--no-threads", action="store_true")
    a = ap.parse_args(argv)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    with open(a.out, "w", buffering=1) as w:
        w.write("t_mono,role,pid,tid,ticks\n")
        nxt = time.monotonic()
        while not stop and os.path.exists(f"/proc/{a.root_pid}"):
            sample_once(a.root_pid, w, not a.no_threads)
            nxt += a.period
            time.sleep(max(0.0, nxt - time.monotonic()))
        sample_once(a.root_pid, w, not a.no_threads) if os.path.exists(f"/proc/{a.root_pid}") else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
