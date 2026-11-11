#!/usr/bin/env python3
"""Boot probe for the Python 3.12 SGLang venv (non-measurement). Answers mechanical questions only; request times are probe values, not citable.
  1. Does piecewise CUDA graph capture work for Nano on py3.12 (default chunked, mixed-chunk, and PD-mux on the patched tree)?
  2. Which cap x mem-fraction combinations boot, and what memory does a full cap of short requests reach?
  3. Can a 2^18-token context be filled by chunked prefill (Nano declares 131072; timing use only)?
Each boot writes <out>/<boot>/{server.log,stage.json}; <out>/summary.json collects them. Usage: py312_probe.py --out-root DIR [--print-plan] [--budget-min N]"""
import argparse, json, os, random, re, signal, socket, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENG = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ENG / "results/stock_baseline"), str(ENG / "results/cap_char/policy_harness")]
import sb_spec as SP

PY312 = "/opt/sg312-venv/bin/python"
OFF = "--disable-piecewise-cuda-graph"
VOCAB, SEED = 100000, 1


def drop(cmd, *flags):
    return [x for x in cmd if x not in flags]


def setarg(cmd, flag, val):
    cmd = list(cmd)
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(val)
    else:
        cmd += [flag, str(val)]
    return cmd


def resolve(cmd):
    return [x.replace("{ENG}", str(ENG)) for x in cmd]


def plan():
    sg, mix, pm = SP.arm("SG-stock"), SP.arm("SG-mix"), SP.arm("PM-ref")
    base = resolve(sg["cmd"])
    short = [(8, 2048, 64), (1, 16000, 64)]
    B = [dict(id="piecewise_on_chunked", tree="pristine", cmd=drop(base, OFF), env={}, reqs=short, want_piecewise=True),
         dict(id="piecewise_on_mixed", tree="pristine", cmd=drop(resolve(mix["cmd"]), OFF), env={}, reqs=short, want_piecewise=True),
         dict(id="piecewise_off_chunked", tree="pristine", cmd=base, env={}, reqs=short, want_piecewise=False),
         dict(id="pdmux_piecewise_on", tree="patched", cmd=drop(resolve(pm["cmd"]), OFF), env=dict(pm["env"]), reqs=short, want_piecewise=True),
         dict(id="pdmux_piecewise_off", tree="patched", cmd=resolve(pm["cmd"]), env=dict(pm["env"]), reqs=short, want_piecewise=False)]
    for cap, mem in ((48, 0.6), (48, 0.75), (128, 0.75), (256, 0.75)):
        c = setarg(setarg(setarg(drop(base, OFF), "--max-running-requests", cap), "--max-mamba-cache-size", cap), "--mem-fraction-static", mem)
        B.append(dict(id=f"cap{cap}_mem{int(mem*100)}", tree="pristine", cmd=c, env={}, reqs=[(cap, 256, 32)], want_piecewise=True))
    c = drop(base, OFF)
    for f, v in (("--context-length", 262144), ("--max-prefill-tokens", 262144), ("--max-running-requests", 16), ("--mem-fraction-static", 0.82)):
        c = setarg(c, f, v)
    B.append(dict(id="ctx256k", tree="pristine", cmd=c, env={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
                  reqs=[(1, 65536, 16), (1, 131072, 16), (1, 262144, 16), (8, 131072, 16)], want_piecewise=True))
    return B


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    return int(out.split()[0])


def post(port, n_in, n_out, rng_seed, timeout):
    r = random.Random(rng_seed)
    body = {"input_ids": [r.randrange(1000, VOCAB) for _ in range(n_in)], "sampling_params": {"max_new_tokens": n_out, "ignore_eos": True, "temperature": 0}}
    t0 = time.time()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return {"ok": d["meta_info"]["completion_tokens"] == n_out, "s": round(time.time() - t0, 2), "out": d["meta_info"]["completion_tokens"]}
    except Exception as e:
        return {"ok": False, "s": round(time.time() - t0, 2), "error": repr(e)[:300]}


def run_boot(b, out, boot_timeout, req_timeout):
    d = Path(out) / b["id"]; d.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_") or k in ("PDMUX_ROOT", "PDMUX_PROJECT_ROOT")}
    env.update(b["env"], PDMUX_ENGINE_TREE=b["tree"], PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
    argv = [sys.executable, str(SP.ENGINE_TREE_PY), "boot", "--stamp", str(d / "tree.json"), "--", PY312] + b["cmd"] + ["--port", str(port)]
    res = dict(id=b["id"], tree=b["tree"], argv=b["cmd"], env=b["env"], want_piecewise=b["want_piecewise"], status="FAIL", requests=[])
    log = open(d / "server.log", "w")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=str(d))
    t0 = time.time(); up = False; peak = [0]; stop = threading.Event()
    def poll():
        while not stop.wait(2):
            try: peak[0] = max(peak[0], gpu_used_mib())
            except Exception: pass
    threading.Thread(target=poll, daemon=True).start()
    try:
        while time.time() - t0 < boot_timeout and p.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5); up = True; break
            except Exception:
                time.sleep(3)
        res["boot_s"] = round(time.time() - t0); res["boot_ok"] = up
        if up and b.get("driver"):          # mo_shortcheck.py: a stage-specific request driver replaces the fixed request list
            try:
                res["driver"] = b["driver"](port, d, req_timeout)
            except Exception as e:  # noqa: BLE001
                res["driver"] = {"ok": False, "error": repr(e)[:300]}
        if up and not b.get("driver"):
            sd = SEED
            with ThreadPoolExecutor(max_workers=300) as ex:
                for n, n_in, n_out in b["reqs"]:
                    fs = [ex.submit(post, port, n_in, n_out, sd + i, req_timeout) for i in range(n)]; sd += n
                    rs = [f.result() for f in fs]
                    res["requests"].append(dict(n=n, n_in=n_in, n_out=n_out, ok=sum(x["ok"] for x in rs), max_s=max(x["s"] for x in rs), errors=[x["error"] for x in rs if "error" in x][:3]))
    finally:
        stop.set()
        try: os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=60)
        except Exception:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
        log.close()
    txt = (d / "server.log").read_text(errors="replace")
    res["peak_gpu_mib"] = peak[0]; time.sleep(5); res["gpu_after_mib"] = gpu_used_mib()
    res["banner"] = [l[:400] for l in txt.splitlines() if re.search(r"max_total_num_tokens=|Mamba Cache is allocated|context_len|Piecewise CUDA Graph|Capture piecewise", l)][:14]
    res["piecewise_failed"] = "Piecewise CUDA Graph failed" in txt
    res["piecewise_captured"] = "Capture piecewise CUDA graph end" in txt
    res["oom"] = bool(re.search(r"out of memory", txt, re.I))
    good = up and not res["oom"] and all(r["ok"] == r["n"] for r in res["requests"]) and len(res["requests"]) == len(b["reqs"])
    if b.get("driver"):
        good = up and not res["oom"] and bool(res.get("driver", {}).get("ok"))
    if b["want_piecewise"]:
        good = good and res["piecewise_captured"] and not res["piecewise_failed"]
    res["status"] = "PASS" if good else "FAIL"
    (d / "stage.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root"); ap.add_argument("--print-plan", action="store_true"); ap.add_argument("--budget-min", type=float, default=150)
    a = ap.parse_args()
    B = plan()
    if a.print_plan:
        for b in B:
            print(b["id"], b["tree"], b["env"], b["reqs"], "want_piecewise" if b["want_piecewise"] else ""); print("  ", " ".join(b["cmd"]))
        return 0
    out = Path(a.out_root) / f"py312_probe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"; out.mkdir(parents=True)
    t0 = time.time(); R = []
    for b in B:
        left = a.budget_min * 60 - (time.time() - t0)
        if left < 420:
            R.append(dict(id=b["id"], status="SKIPPED", reason="budget")); continue
        print(f"[probe] {b['id']}", flush=True)
        R.append(run_boot(b, out, boot_timeout=min(1200, left - 120), req_timeout=max(300, min(900, left - 60))))
        print(f"[probe] {b['id']} -> {R[-1]['status']}", flush=True)
    (out / "summary.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({r["id"]: r["status"] for r in R}, indent=0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
