#!/usr/bin/env python3
"""Job runner of PREREG_STOCK_BASE rev2.1 (SEALED 2026-11-02; sb_job.sh gates on HARNESS_DIGEST.txt + PDMUX_SB_SEALED=1).

  sb_runner.py plan --job J2 [--addendum ADD.json]
  sb_runner.py run  --job J2 --out RAW --w1-lengths L.json [--w3-tokens F.bin] [--addendum ADD.json] [--resume]

Per boot: image check (both pip-freeze sha256 of the combined image; mismatch -> REFUSED_IMAGE_ENV, stamped) -> server start (SGLang
through engine_tree.py boot with PDMUX_ENGINE_TREE = the arm's tree; vLLM = /opt/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server,
stamp tree_<arm>.json with engine_tree="vllm", freeze sha, attention backend / CUDAGraphMode from the server log) -> warm-up 60 s
(discarded) + length check -> runs -> server stop.  Every run record carries job_name, gpu_uuid, the engine-tree
stamp of its boot, the trace hash, the load window (epoch + CLOCK_MONOTONIC: t_load_start/t_load_end) and per-request send / first-token /
finish stamps (t_sent_mono, token_t_mono[0], t_finished_mono).  Boot-specific steps:
  smoke    (J1)  : §5-0 sequential fixed requests + server TTFT histogram deltas
  sustain  (J1b) : after the boot's two points, the registered 0.5 x lambda* point runs in the same boot iff neither point passed
  refcell        : first and last boot of J1b/J2/J3 (J3 start boot also measures the W3 floor)
  w3ladder (J3)  : SG-stock W3 ladder; lambda*_W3 (pick rule, in-Job) fills the R3 rate of the stage-gated R3 boots
  instr    (J2)  : instrumentation boots after every measurement boot (C5); raw probe / telemetry files gzipped, reduced by sb_analyze;
                   the SG probe run adds a synthetic single-request segment after its W1 segment (t_synth_start/end, PREREG §5-1)
  instr_metrics (J2) / every vLLM run : server TTFT histogram (sum, count) scraped right before and right after the run (§5-0 (b))
Resume: a run whose record exists is skipped.  A dead / invalid boot makes the exit code non-zero (17).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import http.client
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import sb_spec as SP  # noqa: E402
import sb_loadgen as LG  # noqa: E402
import ph_trace as T  # noqa: E402
import ph_verdict as V  # noqa: E402

WARMUP_S = 60.0
KEEP_REQ = ("request_id", "phase", "input_tokens", "requested_output_tokens", "observed_output_tokens", "ttft_ms", "token_itl_ms",
            "error", "error_kind", "scheduled_arrival_s", "sent_s", "completion_s", "t_sent_mono", "t_finished_mono", "token_t_mono")
INFRA_PASS = ("PDMUX_ROOT", "PDMUX_PROJECT_ROOT", "PDMUX_JOB_NAME", "PDMUX_DATA", "PDMUX_IO", "PDMUX_COMMIT", "PDMUX_OPT",
              "PDMUX_ENERGY_SAMPLER")
PREFILL_RE = re.compile(r"Prefill batch.*?#running-req: (\d+)")


def dump(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, p)


# ---- stamps (Job, GPU) -------------------------------------------------------------------------------------------------
_GPU = {}


def job_name():
    return os.environ.get("PDMUX_JOB_NAME", "local")


def gpu_uuid():
    if "u" not in _GPU:
        u = os.environ.get("PDMUX_GPU_UUID")
        if not u:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
                u = r.stdout.strip().splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else "UNKNOWN"
            except Exception:  # noqa: BLE001
                u = "UNKNOWN"
        _GPU["u"] = u
    return _GPU["u"]


def stamp():
    return {"job_name": job_name(), "gpu_uuid": gpu_uuid()}


# ---- server env / command ----------------------------------------------------------------------------------------------
def resolve(a, bdir):
    a = json.loads(json.dumps(a))
    sub = lambda x: x.replace("{ENG}", str(SP.ENG)).replace("{dir}", str(bdir))
    a["cmd"] = [sub(x) for x in a["cmd"]]
    a["env"] = {k: sub(v) for k, v in a["env"].items()}
    return a


def server_env(a, bdir, base=None):
    """SGLang: every PDMUX_* is dropped except Job plumbing; the arm's own engine env is added.  Pristine boots get NO engine knob
    (engine_tree.py would refuse them); patched boots also get the policy-harness label env (PDMUX_RUN_ID / WORKLOAD_ID / HEARTBEAT_S).
    vLLM: EVERY PDMUX_* and PYTHONPATH dropped (the venv interpreter must not see the SGLang trees), CUDA_HOME / TORCH_CUDA_ARCH_LIST set,
    the venv bin first on PATH (JIT tools such as ninja are resolved from PATH; the base image has /opt/conda/bin there)."""
    base = dict(os.environ if base is None else base)
    keep = INFRA_PASS if a["engine"] == "sglang" else ()
    env = {k: v for k, v in base.items() if not k.startswith("PDMUX_") or k in keep}
    env.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
    env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
    if a["engine"] == "sglang":
        env["PDMUX_ENGINE_TREE"] = a["tree"]
        if a["tree"] == "patched":
            env.update(PDMUX_RUN_ID=job_name(), PDMUX_WORKLOAD_ID=f"stock_{Path(bdir).name}", PDMUX_HEARTBEAT_S="0")
    else:
        env.pop("PYTHONPATH", None)
        env["PATH"] = os.pathsep.join([str(Path(SP.VLLM_PY).parent)] + ([env["PATH"]] if env.get("PATH") else []))
    env.update(a["env"])
    return env


def server_cmd(a, bdir, port, py=None):
    py = py or sys.executable
    if a["engine"] == "sglang":
        return [py, str(SP.ENGINE_TREE_PY), "boot", "--stamp", str(Path(bdir) / f"tree_{a['name']}.json"), "--", py] + a["cmd"] + ["--port", str(port)]
    return list(a["cmd"]) + ["--port", str(port)]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class ProcessServer:
    def __init__(self, bdir, a, boot_timeout=1500):
        self.bdir, self.a, self.boot_timeout = Path(bdir), a, boot_timeout
        self.log = self.bdir / "server.log"
        self.proc = None

    def start(self):
        self.port = free_port()
        self.cmd = server_cmd(self.a, self.bdir, self.port)
        self.lf = open(self.log, "wb")
        self.proc = subprocess.Popen(self.cmd, stdout=self.lf, stderr=subprocess.STDOUT, env=server_env(self.a, self.bdir),
                                     start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.boot_timeout:
            if self.proc.poll() is not None:
                return "died"
            if _health(self.port):
                return "ok"
            time.sleep(2)
        return "timeout"

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        if self.proc is not None:
            for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                t0 = time.monotonic()
                while time.monotonic() - t0 < wait and self.proc.poll() is None:
                    time.sleep(0.1)
                if self.proc.poll() is not None:
                    break
            self.lf.close()
        return {"returncode": self.proc.poll() if self.proc else None}


def _health(port):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/health")
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except Exception:  # noqa: BLE001
        return False


# ---- server-log facts (recorded, never a decision input except MIX firing) ---------------------------------------------------
SG_LOG_KEYS = {k: re.compile(rf"\b{k}=([^,)]+)") for k in ("attention_backend", "mem_fraction_static", "chunked_prefill_size",
                                                              "enable_mixed_chunk", "disable_overlap_schedule", "disable_cuda_graph",
                                                              "cuda_graph_max_bs", "max_running_requests", "max_mamba_cache_size")}
VL_BACKEND_RE = re.compile(r"Using (?:AttentionBackendEnum\.)?([A-Za-z0-9_]+) (?:attention )?backend")
VL_CGMODE_RE = re.compile(r"cudagraph_mode['\"]?\s*[:=]\s*<?(?:CUDAGraphMode\.)?([A-Z_]+)")
VL_LOG_RE = re.compile(r"(?i)(attention backend|attn_backend|cudagraph_mode|CUDAGraphMode|chunked prefill|enable_chunked_prefill|"
                       r"max_num_batched_tokens|Using .*Backend)")


def parse_sglang_log(text):
    out = {}
    for k, rx in SG_LOG_KEYS.items():
        m = rx.search(text)
        out[k] = m.group(1).strip().strip("'\"") if m else None
    return out


def parse_vllm_log(text, keep=40):
    """Backend / CUDAGraphMode / chunked-prefill lines, verbatim (GPU-unverified patterns: bring-up (a)-(e) not run)."""
    lines = [ln.strip() for ln in text.splitlines() if VL_LOG_RE.search(ln)]
    uniq = lambda xs: list(dict.fromkeys(xs))
    be, cg = uniq(VL_BACKEND_RE.findall(text)), uniq(VL_CGMODE_RE.findall(text))
    return {"matched_lines": lines[:keep], "n_matched": len(lines), "attention_backend": be[0] if be else None, "attention_backends": be,
            "cudagraph_mode": cg[0] if cg else None, "cudagraph_modes": cg}


# ---- image identity (rev1: one combined image; both freezes checked before every boot) --------------------------------------
def _sha_file(p):
    import hashlib
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def image_env_check(opt=None, venv=SP.VLLM_VENV):
    """sha256 of the SGLang freeze (/opt/pdmux/pip_freeze.txt, PDMUX_OPT) and of the vLLM venv freeze (/opt/vllm-venv/pip_freeze.txt, also
    the sha recorded in the image at /opt/vllm-venv/pip_freeze.sha256), compared with the registered values."""
    opt = Path(opt or os.environ.get("PDMUX_OPT", "/opt/pdmux"))
    sg = _sha_file(opt / "pip_freeze.txt")
    vt = _sha_file(Path(venv) / "pip_freeze.txt")
    try:
        vf = (Path(venv) / "pip_freeze.sha256").read_text().split()[0]
    except (OSError, IndexError):
        vf = None
    return {"sglang_freeze_path": str(opt / "pip_freeze.txt"), "sglang_freeze_sha256": sg, "vllm_venv": str(venv),
            "vllm_venv_freeze_sha256": vt, "vllm_venv_freeze_sha256_file": vf,
            "sglang_ok": sg == SP.SGLANG_FREEZE_SHA256, "vllm_ok": vt == SP.VLLM_VENV_FREEZE_SHA256 and vf == SP.VLLM_VENV_FREEZE_SHA256}


def image_refusal(ic, engine):
    """-> None (boot allowed) or the refusal reason.  Every boot needs the SGLang side unchanged; a vLLM boot also needs the venv freeze."""
    why = []
    if not ic.get("sglang_ok"):
        why.append(f"SGLANG_FREEZE:{ic.get('sglang_freeze_sha256')}")
    if engine == "vllm" and not ic.get("vllm_ok"):
        why.append(f"VLLM_VENV_FREEZE:{ic.get('vllm_venv_freeze_sha256')}|{ic.get('vllm_venv_freeze_sha256_file')}")
    return ",".join(why) or None


def vllm_stamp(a, ic, log_text):
    f = parse_vllm_log(log_text)
    return {"engine_tree": "vllm", "python": a["cmd"][0], "argv": list(a["cmd"]), "env": dict(a["env"]),
            "vllm_venv_freeze_sha256": ic.get("vllm_venv_freeze_sha256"), "sglang_freeze_sha256": ic.get("sglang_freeze_sha256"),
            "attention_backend": f["attention_backend"], "attention_backends": f["attention_backends"],
            "cudagraph_mode": f["cudagraph_mode"], "cudagraph_modes": f["cudagraph_modes"]}


def mixing_opportunities(log_slice):
    """Number of MIXING OPPORTUNITIES (rev2, audit r1 C): prefill batches logged while >= 1 request was running.  The line is printed with
    or without --enable-mixed-chunk (v0.5.10 scheduler_metrics_mixin.py:389-394), so this is NOT a firing count of the treatment; it is
    reported for SG-mix and for SG-stock (negative control).  Firing = server_args (mix_fired_from_server_args)."""
    return sum(1 for m in PREFILL_RE.finditer(log_slice) if int(m.group(1)) >= 1)


def mix_fired_from_server_args(facts):
    """SG-mix treatment check (§6 rev2): the boot's server_args show enable_mixed_chunk=True AND chunked prefill active
    (chunked_prefill_size an integer > 0; v0.5.10 Scheduler.is_mixed_chunk = chunked_prefill_size is not None and enable_mixed_chunk)."""
    f = facts or {}
    try:
        cps = int(str(f.get("chunked_prefill_size")))
    except (TypeError, ValueError):
        cps = None
    return f.get("enable_mixed_chunk") == "True" and cps is not None and cps > 0


def _log_size(srv):
    p = getattr(srv, "log", None)
    return Path(p).stat().st_size if p and Path(p).exists() else 0


def _log_tail(srv, off):
    p = getattr(srv, "log", None)
    if not p or not Path(p).exists():
        return ""
    return Path(p).read_bytes()[off:].decode(errors="replace")


# ---- replay ------------------------------------------------------------------------------------------------------------
def default_replay(api, base_url, requests, out_path):
    return LG.replay(api, base_url, requests, out_path)


def replay_segment(replay_fn, api, base_url, seg, out_dir, w3_tokens=None):
    reqs = seg["requests"]
    if w3_tokens and any("doc_id" in r for r in reqs):
        T.attach_w3_ids(reqs, w3_tokens)
    e0, m0 = time.time(), time.monotonic()
    dur, recs = replay_fn(api, base_url, reqs, Path(out_dir) / f"{seg['segment_id']}.jsonl")
    e1, m1 = time.time(), time.monotonic()
    return {"segment_id": seg["segment_id"], "phase": seg["phase"], "round": seg.get("round", 0), "duration_s": float(dur),
            "t_load_start_epoch": e0, "t_load_end_epoch": e1, "t_load_start_mono": m0, "t_load_end_mono": m1,
            "requests": [{k: r[k] for k in KEEP_REQ if k in r} for r in recs]}


def build_segments(run, w1_pool, w3_pool):
    if run.get("rate") is None:
        raise RuntimeError(f"ABORT_RATE_UNFILLED: {run['run_id']}")
    pool = w1_pool if run["workload"] == "W1" else w3_pool
    segs = T.cap_trace(run["workload"], pool, run["rate"], run["seed"], run["n_requests"], run["run_id"])
    if run.get("synth"):
        # §5-1: separate segment AFTER the load segment has finished (replay returns once every request is done) -> its own time window
        segs.append({"segment_id": f"{run['run_id']}_SYNTH", "phase": SP.SYNTH_PHASE, "round": 0,
                     "requests": SP.synth_requests(run["seed"], run["run_id"], run["synth"])})
    return segs


def scrape_ttft(metrics_fn, base_url):
    try:
        s, c = LG.parse_ttft_metric(metrics_fn(base_url))
        return {"sum_s": s, "count": c}
    except Exception as e:  # noqa: BLE001
        return {"sum_s": None, "count": None, "error": repr(e)}


def wants_ttft_scrape(a, b):
    return a["engine"] == "vllm" or b["role"] == "instr_metrics"


def execute_run(run, b, srv, replay_fn, w1_pool, w3_pool, out, w3_tokens=None, metrics_fn=LG.fetch_metrics):
    a = b["arm_def"]
    segs = build_segments(run, w1_pool, w3_pool)
    rec = {"run_id": run["run_id"], "kind": run["kind"], "cell": run.get("cell"), "seed": run["seed"], "arm": b["arm"],
           "boot_id": b["boot_id"], "boot_role": b["role"], "abba_round": b.get("abba_round"), "trace_hash": T.trace_hash(segs),
           "spec": {k: v for k, v in run.items() if k != "run_id"}, "segments": [], **stamp(),
           "engine": a["engine"], "engine_tree": a["tree"]}
    off = _log_size(srv)
    scrape = wants_ttft_scrape(a, b)
    m0 = scrape_ttft(metrics_fn, srv.base_url) if scrape else None
    for s in segs:
        rec["segments"].append(replay_segment(replay_fn, a["api"], srv.base_url, s, Path(out) / "segments" / run["run_id"], w3_tokens))
    load = [x for x in rec["segments"] if x["phase"] != SP.SYNTH_PHASE]
    syn = [x for x in rec["segments"] if x["phase"] == SP.SYNTH_PHASE]
    rec["t_load_start_epoch"] = load[0]["t_load_start_epoch"]
    rec["t_load_end_epoch"] = load[-1]["t_load_end_epoch"]
    rec["t_load_start_mono"] = load[0]["t_load_start_mono"]
    rec["t_load_end_mono"] = load[-1]["t_load_end_mono"]
    if syn:                                                     # §5-1 synthetic window, recorded apart from the W1 load window
        rec["t_synth_start_epoch"], rec["t_synth_end_epoch"] = syn[0]["t_load_start_epoch"], syn[-1]["t_load_end_epoch"]
        rec["t_synth_start_mono"], rec["t_synth_end_mono"] = syn[0]["t_load_start_mono"], syn[-1]["t_load_end_mono"]
        sq = [q for x in segs if x["phase"] == SP.SYNTH_PHASE for q in x["requests"]]
        rec["synth"] = {"n_requests": len(sq), "lengths": [q["input_tokens"] for q in sq],
                        "input_ids_sha256": [q["input_ids_sha256"] for q in sq]}
    if scrape:
        rec["server_ttft_metric"] = {"before": m0, "after": scrape_ttft(metrics_fn, srv.base_url)}
    if a["name"] in SP.MIX_OPPORTUNITY_ARMS:
        rec["mixing_opportunities"] = mixing_opportunities(_log_tail(srv, off))
    return rec


def warmup_and_check(srv, a, replay_fn, w1_pool, bdir, boot_id):
    """Warm-up (60 s of W1 at 2 req/s, discarded) + correctness: every fixed-length warm-up request returns exactly its length."""
    ts = T.arrivals(2.0, ("warm", boot_id), duration=WARMUP_S)
    reqs = T.make_requests("W1", w1_pool, ts, ("warm", boot_id), f"{boot_id}_warm", "WARM")
    seg = replay_segment(replay_fn, a["api"], srv.base_url, {"segment_id": "warm", "phase": "WARM", "requests": reqs}, Path(bdir) / "warmup")
    bad = [q["request_id"] for q in seg["requests"] if q.get("error") or q.get("observed_output_tokens") != q.get("requested_output_tokens")]
    dump({"n": len(seg["requests"]), "bad": bad[:20], "n_bad": len(bad), "t_warm_end_epoch": seg["t_load_end_epoch"]},
         Path(bdir) / "correctness.json")
    return (not bad), f"{len(bad)}/{len(seg['requests'])} warm-up requests wrong length or errored"


def measure_floor(srv, a, replay_fn, w3_pool, out, w3_tokens=None, n=8, seed=7):
    """W3 floor = median TTFT of n sequential single-request segments (same Job, before the W3 cells; N3 rule)."""
    tt = []
    for k in range(n):
        seg = T.cap_trace("W3", w3_pool, 1000.0, seed + k, 1, f"floor{k}")[0]
        s = replay_segment(replay_fn, a["api"], srv.base_url, seg, Path(out) / "floor", w3_tokens)
        tt += [r["ttft_ms"] for r in s["requests"] if r.get("ttft_ms") is not None]
    f = statistics.median(tt) if tt else None
    dump({"n": len(tt), "ttft_ms": tt, "floor_ms": f, **stamp()}, Path(out) / "floor.json")
    return f


def smoke_requests(w1_pool):
    seg = T.cap_trace("W1", w1_pool, 1000.0, SP.SMOKE_SEED, SP.SMOKE_N, "SMOKE")[0]
    for r in seg["requests"]:
        r["output_tokens"] = min(int(r["output_tokens"]), SP.SMOKE_MAX_OUT)
    return seg["requests"]


# ---- registered in-boot rules ------------------------------------------------------------------------------------------
def sustain_done(r, e2e_max_s=SP.SUSTAIN_E2E_MAX_S):
    """rev2 (audit r1 A): a request counts toward its ARRIVAL window iff it finished without error and E2E = completion_s - sent_s
    (scheduled_arrival_s when the send stamp is absent; both on the run-start clock) <= 60 s."""
    if r.get("error") or r.get("completion_s") is None:
        return False
    t0 = r.get("sent_s") if r.get("sent_s") is not None else r["scheduled_arrival_s"]
    return float(r["completion_s"]) - float(t0) <= e2e_max_s


def sustain_eval(run_rec, window_s=SP.SUSTAIN_WINDOW_S, ttft_slo=SP.W1_TTFT_MS, itl_slo=SP.ITL_MS):
    """One sustained run -> per-120 s ARRIVAL-window rows and the pass rule of §3 (PREREG rev2, audit r1 A):
    every arrival window w (arrivals in [120 w, 120 (w+1)), arrival < 600 s) has
        completion_ratio = #(arrivals of w done with E2E <= 60 s) / #(arrivals of w) >= 0.95
    AND last-window TTFT p50 <= 2 x first-window p50.  Arrivals at or after 600 s are EXCLUDED (counted in n_late_excluded).
    The rev1 rule ("completed inside window w" / "arrived inside window w", late arrivals clamped into the last window) failed a healthy
    server structurally (first window: fill transient; last window: completions after 600 s not counted).  first_token_ratio (first tokens
    inside window w / arrivals of w) is the admission-speed column, reported only."""
    seg = run_rec["segments"][0]
    reqs = seg["requests"]
    rate = float(run_rec["spec"]["rate"])
    nwin = int(SP.SUSTAIN_S // window_s)
    first, by_arr, n_late = [0] * nwin, {}, 0
    for r in reqs:
        if float(r["scheduled_arrival_s"]) >= SP.SUSTAIN_S:
            n_late += 1
            continue
        by_arr.setdefault(int(float(r["scheduled_arrival_s"]) // window_s), []).append(r)
        if r.get("ttft_ms") is not None:
            w = int((r["sent_s"] + r["ttft_ms"] / 1000.0) // window_s) if r.get("sent_s") is not None else None
            if w is not None and 0 <= w < nwin:
                first[w] += 1
    comp = [sum(1 for r in by_arr.get(w, []) if sustain_done(r)) for w in range(nwin)]
    rows = []
    for w in range(nwin):
        arr = len(by_arr.get(w, []))
        tt = [x["ttft_ms"] for x in by_arr.get(w, []) if x.get("ttft_ms") is not None]
        good = sum(1 for x in by_arr.get(w, []) if V.req_good(x, ttft_slo, itl_slo))
        rows.append({"window": w, "arrivals": arr, "nominal_arrivals": rate * window_s,
                     "completion_ratio": comp[w] / arr if arr else 0.0, "first_token_ratio": first[w] / arr if arr else 0.0,
                     "attainment": good / len(by_arr[w]) if by_arr.get(w) else None, "ttft_p50_ms": V.percentile(tt, 0.5)})
    p0, pl = rows[0]["ttft_p50_ms"], rows[-1]["ttft_p50_ms"]
    growth = (pl / p0) if p0 and p0 == p0 and p0 > 0 else float("inf")
    ok = all(r["completion_ratio"] >= SP.SUSTAIN_COMPLETION_MIN for r in rows) and growth <= SP.SUSTAIN_TTFT_GROWTH_MAX
    return {"windows": rows, "ttft_growth": growth, "n_late_excluded": n_late, "e2e_max_s": SP.SUSTAIN_E2E_MAX_S,
            "rule": "rev2: arrival window, E2E <= 60 s, arrivals >= 600 s excluded", "pass": bool(ok)}


def lambda_pick(points, attain=SP.CAP_ATTAIN):
    """pick rule (implfix pick_p_rates): points = [(rate, attainment)] one per (rate, seed) run.  Walk ascending rate; at an equal rate the
    LOWER attainment comes first (tie rule: a rung where any seed fails stops the prefix -> the lower rate wins).  -> (lambda*, status)."""
    pts = sorted(points, key=lambda x: (x[0], x[1]))
    if not pts:
        return None, "NO_POINTS"
    last = None
    for rate, g in pts:
        if g >= attain:
            last = rate
        else:
            break
    if last is None:
        return None, "LADDER_BOTTOM_FAIL"
    if last == pts[-1][0]:
        return last, "LADDER_TOP_PASS"
    return last, "BRACKETED"


def attainment(run_rec, ttft_slo, itl_slo=SP.ITL_MS):
    reqs = [r for s in run_rec["segments"] for r in s["requests"]]
    return sum(1 for r in reqs if V.req_good(r, ttft_slo, itl_slo)) / len(reqs) if reqs else 0.0


# ---- boots -------------------------------------------------------------------------------------------------------------
def gzip_instr(bdir):
    for n in ("decode_probe.jsonl", "tel.jsonl"):
        f = Path(bdir) / n
        if f.exists():
            with open(f, "rb") as a, gzip.open(str(f) + ".gz", "wb") as b:
                shutil.copyfileobj(a, b)
            f.unlink()


def run_boot(b, out, make_server, replay_fn, w1_pool, w3_pool, w3_tokens=None, ctx=None, metrics_fn=LG.fetch_metrics,
             image_check_fn=image_env_check):
    ctx = ctx if ctx is not None else {}
    bdir = Path(out) / "boots" / b["boot_id"]
    bdir.mkdir(parents=True, exist_ok=True)
    a = resolve(b["arm_def"], bdir)
    if b.get("stage_gated") == "R3":
        if ctx.get("r3_budget_skip"):
            dump({"boot_id": b["boot_id"], "status": SP.R3_BUDGET_STATUS, "budget": ctx.get("r3_budget")}, bdir / "DONE.json")
            return {"boot_id": b["boot_id"], "status": SP.R3_BUDGET_STATUS, "runs": []}
        if ctx.get("r3_rate") is None:
            dump({"boot_id": b["boot_id"], "status": "SKIPPED_NO_W3_LAMBDA", "why": ctx.get("w3_status")}, bdir / "DONE.json")
            return {"boot_id": b["boot_id"], "status": "SKIPPED_NO_W3_LAMBDA", "runs": []}
        for r in b["runs"]:
            r["rate"] = ctx["r3_rate"]
    if b.get("role") == "w3ladder" and ctx.get("w3_slo_ms") is None:
        dump({"boot_id": b["boot_id"], "status": "SKIPPED_NO_W3_FLOOR"}, bdir / "DONE.json")
        return {"boot_id": b["boot_id"], "status": "SKIPPED_NO_W3_FLOOR", "runs": []}
    ic = image_check_fn()
    tp = bdir / f"tree_{a['name']}.json"
    refuse = image_refusal(ic, a["engine"])
    if refuse:
        if a["engine"] == "vllm":
            dump(vllm_stamp(a, ic, ""), tp)
        dump({"boot_id": b["boot_id"], "arm": b["arm"], "role": b["role"], "status": "REFUSED_IMAGE_ENV", "why": refuse, "image_env": ic,
              "engine": a["engine"], "engine_tree": a["tree"], **stamp(), "image_key": a["image_key"],
              "image": SP.image_digests().get(a["image_key"])}, bdir / "boot.json")
        dump({"boot_id": b["boot_id"], "status": "REFUSED_IMAGE_ENV", "why": refuse}, bdir / "DONE.json")
        return {"boot_id": b["boot_id"], "status": "REFUSED_IMAGE_ENV", "runs": []}
    srv = make_server(bdir, a)
    status = srv.start()
    if a["engine"] == "vllm":
        dump(vllm_stamp(a, ic, _log_tail(srv, 0)), tp)
    tree_stamp = {}
    if tp.exists():
        tree_stamp = json.loads(tp.read_text())
    sg_args = parse_sglang_log(_log_tail(srv, 0)) if a["engine"] == "sglang" else None
    dump({"boot_id": b["boot_id"], "arm": b["arm"], "role": b["role"], "status": status, "cmd": getattr(srv, "cmd", None),
          "engine": a["engine"], "engine_tree": a["tree"], "tree_stamp": tree_stamp, "server_args": sg_args, "image_env": ic, **stamp(),
          "image_key": a["image_key"], "image": SP.image_digests().get(a["image_key"])}, bdir / "boot.json")
    if status != "ok":
        srv.stop()
        return {"boot_id": b["boot_id"], "status": status, "runs": []}
    done = []
    try:
        ok, why = warmup_and_check(srv, a, replay_fn, w1_pool, bdir, b["boot_id"])
        if not ok:
            dump({"boot_id": b["boot_id"], "status": "INVALID_CORRECTNESS", "why": why}, bdir / "DONE.json")
            return {"boot_id": b["boot_id"], "status": "INVALID_CORRECTNESS", "runs": []}
        if b.get("floor") and not (Path(out) / "floor.json").exists():
            f = measure_floor(srv, a, replay_fn, w3_pool, out, w3_tokens)
            ctx["w3_slo_ms"] = SP.W3_TTFT_K * f if f else None
        runs = list(b["runs"])
        if b["role"] == "smoke":
            for run in runs:
                f = Path(out) / "runs" / f"{run['run_id']}.json"
                if not f.exists():
                    rows = LG.smoke_sequence(a["api"], srv.base_url, smoke_requests(w1_pool), bdir / "smoke", metrics_fn=metrics_fn,
                                             replay_fn=lambda api, url, reqs, p: replay_fn(api, url, reqs, p))
                    dump({"run_id": run["run_id"], "kind": "smoke", "engine": a["engine"], "arm": b["arm"], "rows": rows,
                          "boot_id": b["boot_id"], **stamp()}, f)
                done.append(run["run_id"])
        else:
            pending = list(runs)
            while pending:
                run = pending.pop(0)
                f = Path(out) / "runs" / f"{run['run_id']}.json"
                if f.exists():
                    rec = json.loads(f.read_text())
                else:
                    rec = execute_run(run, b, srv, replay_fn, w1_pool, w3_pool, out, w3_tokens, metrics_fn)
                    rec["tree_stamp"] = tree_stamp
                    rec["server_args"] = sg_args
                    dump(rec, f)
                done.append(run["run_id"])
                if b.get("extra_run") and not pending and run is not b["extra_run"]:
                    evs = [sustain_eval(json.loads((Path(out) / "runs" / f"{r['run_id']}.json").read_text())) for r in runs]
                    dump({"boot_id": b["boot_id"], "evals": {r["run_id"]: e for r, e in zip(runs, evs)},
                          "extra_needed": not any(e["pass"] for e in evs)}, bdir / "SUSTAIN_EVAL.json")
                    if not any(e["pass"] for e in evs):
                        pending.append(b["extra_run"])
        info_log = srv.log.read_text(errors="replace") if getattr(srv, "log", None) and Path(srv.log).exists() else ""
        dump({"sglang": parse_sglang_log(info_log)} if a["engine"] == "sglang" else {"vllm": parse_vllm_log(info_log)},
             bdir / "server_facts.json")
    finally:
        info = srv.stop()
        if b["role"] == "instr":
            gzip_instr(bdir)
    if b.get("role") == "w3ladder":
        recs = [json.loads((Path(out) / "runs" / f"{r}.json").read_text()) for r in done]
        lam, st = lambda_pick([(r["spec"]["rate"], attainment(r, ctx["w3_slo_ms"])) for r in recs])
        ctx["w3_status"] = st
        ctx["r3_rate"] = round(SP.W3_R3_FACTOR * lam, 6) if lam is not None and st == "BRACKETED" else None
        dump({"lambda_star_W3": lam, "status": st, "r3_rate": ctx["r3_rate"], "w3_slo_ms": ctx["w3_slo_ms"], **stamp()},
             Path(out) / "W3_LAMBDA.json")
        if ctx["r3_rate"] is not None:
            ctx["r3_budget"] = r3_budget(ctx, ctx["r3_rate"])
            ctx["r3_budget_skip"] = ctx["r3_budget"]["skip"]
            dump(ctx["r3_budget"], Path(out) / "R3_BUDGET.json")
    dump({"boot_id": b["boot_id"], "status": "done", "runs": done, "server": info, **stamp()}, bdir / "DONE.json")
    return {"boot_id": b["boot_id"], "status": "done", "runs": done}


def r3_budget(ctx, r3_rate):
    """rev2 (audit r1 ⑨): decided ONCE, right after the W3 ladder, before any R3 boot -> all R3 boots run or none (no half ABBA).
    need = time-model estimate of every R3 boot at the filled rate + every later boot of the plan (the end refcell); left = hard cap -
    elapsed (the Job's `timeout` starts with this process).  left < need -> R3_SKIPPED_BUDGET for every R3 boot."""
    boots = ctx.get("plan_boots") or []
    i = next((k for k, x in enumerate(boots) if x.get("role") == "w3ladder"), None)
    later = boots[i + 1:] if i is not None else []
    need = 0.0
    for x in later:
        y = json.loads(json.dumps(x))
        if y.get("stage_gated") == "R3":
            for r in y["runs"]:
                r["rate"] = r3_rate
        need += SP.boot_est_min(y)
    elapsed = (ctx["clock"]() - ctx["t0"]) / 60.0 if ctx.get("clock") and ctx.get("t0") is not None else 0.0
    cap = ctx.get("hard_cap_min")
    left = (cap - elapsed) if cap is not None else float("inf")
    return {"r3_rate": r3_rate, "need_min": round(need, 1), "elapsed_min": round(elapsed, 1), "hard_cap_min": cap,
            "left_min": round(left, 1) if left != float("inf") else None, "skip": bool(left < need),
            "status": SP.R3_BUDGET_STATUS if left < need else "R3_BUDGET_OK", **stamp()}


def run_job(plan, out, make_server, replay_fn, w1_pool, w3_pool, w3_tokens=None, resume=False, metrics_fn=LG.fetch_metrics,
            image_check_fn=image_env_check, clock=time.monotonic):
    problems = SP.plan_problems(plan)
    if problems:
        raise RuntimeError(f"ABORT_PLAN: {problems}")
    out = Path(out)
    if out.exists() and any((out / "runs").glob("*.json")) and not resume:
        raise RuntimeError("ABORT_RAW_NOT_EMPTY: raw directory has runs; set --resume")
    dump(plan, out / "plan.json")
    ctx = {"clock": clock, "t0": clock(), "hard_cap_min": plan.get("hard_cap_min"), "plan_boots": plan["boots"]}
    if (out / "floor.json").exists():
        f = json.loads((out / "floor.json").read_text()).get("floor_ms")
        ctx["w3_slo_ms"] = SP.W3_TTFT_K * f if f else None
    res = [run_boot(b, out, make_server, replay_fn, w1_pool, w3_pool, w3_tokens, ctx, metrics_fn, image_check_fn) for b in plan["boots"]]
    dump({"job": plan["job"], "plan_sha": SP.sha(plan), "boots": res, "resumed": bool(resume), "pending": plan["pending"], **stamp()},
         out / "RUN_DONE.json")
    return res


def exit_code(res):
    return 17 if any(b["status"] != "done" for b in res) else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "plan"):
        p = sub.add_parser(n)
        p.add_argument("--job", required=True)
        p.add_argument("--addendum")
    r = sub.choices["run"]
    r.add_argument("--out", required=True)
    r.add_argument("--w1-lengths", required=True)
    r.add_argument("--w3-tokens")
    r.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    add = json.load(open(a.addendum)) if a.addendum else None
    plan = SP.make_plan(a.job, add)
    if a.cmd == "plan":
        print(json.dumps({"plan_sha": SP.sha(plan), "problems": SP.plan_problems(plan), "boots": len(plan["boots"]),
                          "est_min": plan["est_min"], "hard_cap_min": plan["hard_cap_min"], "pending": plan["pending"]}))
        return 0
    want = (add or {}).get("w1_lengths_sha256")
    if want:
        import hashlib
        if hashlib.sha256(Path(a.w1_lengths).read_bytes()).hexdigest() != want:
            print("ABORT_DATASET: w1_lengths sha256 differs from the addendum", file=sys.stderr)
            return 13
    w1 = json.load(open(a.w1_lengths))
    w3 = T.load_w3_pool(a.w3_tokens)[0] if a.w3_tokens else []
    res = run_job(plan, a.out, lambda bdir, arm_def: ProcessServer(bdir, arm_def), default_replay, w1, w3, a.w3_tokens, a.resume)
    rc = exit_code(res)
    if rc:
        print("BOOT_FAILURES: " + ",".join(f"{b['boot_id']}={b['status']}" for b in res if b["status"] != "done"), file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
