"""
run_serving.py — vLLM V1 hybrid model serving evaluation (S1.1).

Launches an OpenAI-compatible vLLM server for a hybrid SSM/Transformer model
and drives it with ShareGPT or LongBench traces.  No SM partitioning applied:
this is pure observation of under-utilization, which motivates SMController.

Supported models (vLLM V1 hybrid support confirmed):
  zamba2     — Zyphra/Zamba2-7B-Instruct  (vLLM ≥ 0.6.3)
  nemotron_h — nvidia/Nemotron-H-8B-Base  (vLLM ≥ 0.8.x, verify model card)
  falcon_h1  — tiiuae/Falcon-H1-7B-Instruct (add after vLLM support confirmed)

Hardware guard (two-tier):
  A100 detected  → full run, numeric conclusions valid
  Non-A100       → "PIPELINE VALIDATION ONLY" warning + smoke run (10 req)

Prefill+decode mix is observable in server logs (default vLLM logging).

Usage:
    # Full ShareGPT run on A100:
    python workspace/serving-eval/run_serving.py --model zamba2 --trace sharegpt

    # Smoke run (pipeline validation, any GPU):
    python workspace/serving-eval/run_serving.py --model zamba2 --smoke

    # LongBench (long prefill, A100):
    python workspace/serving-eval/run_serving.py \\
        --model nemotron_h --trace longbench --concurrency 4 --n-requests 50

    # External server already running on port 8000:
    python workspace/serving-eval/run_serving.py --no-server --model zamba2
"""

from __future__ import annotations

import sys
import os

# workspace/ → shared.loaders accessible without installing the package
_here      = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.dirname(_here)
sys.path.insert(0, _workspace)
# serving-eval/ → _load_trace importable
sys.path.insert(0, _here)

import argparse
import asyncio
import json
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Hardware guard
# ---------------------------------------------------------------------------

_A100_KEYWORDS = frozenset({"a100", "a800"})


def _detect_gpu() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.  vLLM requires a CUDA device.")
    props = torch.cuda.get_device_properties(0)
    return {
        "name":      torch.cuda.get_device_name(0),
        "sm_count":  props.multi_processor_count,
        "memory_gb": props.total_memory / (1024 ** 3),
    }


def _is_a100(gpu: dict) -> bool:
    return any(kw in gpu["name"].lower() for kw in _A100_KEYWORDS)


def hardware_guard(force_smoke: bool) -> tuple[dict, bool]:
    """Return (gpu_info, is_full_run).  Prints tier decision.

    full_run=True  → A100, numeric conclusions valid.
    full_run=False → any GPU, pipeline validation only.
    """
    gpu = _detect_gpu()
    if _is_a100(gpu) and not force_smoke:
        print(f"  GPU: {gpu['name']}  ({gpu['sm_count']} SMs, {gpu['memory_gb']:.0f} GB)")
        print("  Tier: FULL RUN — A100 confirmed, numeric conclusions valid.")
        return gpu, True
    else:
        width = 70
        print("=" * width)
        if force_smoke:
            print("  --smoke flag: forcing smoke run.")
        else:
            print("  WARNING: Non-A100 GPU detected.")
        print(f"  GPU:  {gpu['name']}  ({gpu['sm_count']} SMs, {gpu['memory_gb']:.0f} GB)")
        print("  Tier: PIPELINE VALIDATION ONLY")
        print("  ⚠ Do NOT draw numeric conclusions from this run.")
        print("  ⚠ Re-run on A100 for results referenced in the paper.")
        print("=" * width)
        return gpu, False


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# HF repo IDs for each model.  get_model_config() is authoritative when available;
# this dict is the serving-eval fallback (no dependency on characterization/).
_HF_REPOS: dict[str, str] = {
    "zamba2":     "Zyphra/Zamba2-7B-Instruct",
    "nemotron_h": "nvidia/Nemotron-H-8B-Base-8K",
    "falcon_h1":  "tiiuae/Falcon-H1-7B-Instruct",
}

_GPU_MEM_UTIL: dict[str, float] = {
    "zamba2":     0.85,
    "nemotron_h": 0.90,
    "falcon_h1":  0.85,
}

# Per-model vLLM max_model_len.  None = let vLLM derive from model config.
# Zamba2-7B-Instruct: max_position_embeddings=4096 in config.json.
_MAX_MODEL_LEN: dict[str, int | None] = {
    "zamba2":     4096,
    "nemotron_h": None,
    "falcon_h1":  None,
}


def _resolve_hf_repo(model_name: str) -> str:
    try:
        from shared.loaders import get_model_config
        cfg = get_model_config(model_name)
        return cfg.get("hf_repo") or _HF_REPOS[model_name]
    except Exception:
        if model_name in _HF_REPOS:
            return _HF_REPOS[model_name]
        raise ValueError(
            f"Unknown model {model_name!r}.  Known: {sorted(_HF_REPOS)}.  "
            f"Add to shared/configs/models.yaml or _HF_REPOS."
        )


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def _build_server_cmd(
    hf_repo: str,
    port: int,
    gpu_mem_util: float,
    max_model_len: int,
    enable_chunked_prefill: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                   hf_repo,
        "--port",                    str(port),
        "--gpu-memory-utilization",  str(gpu_mem_util),
        "--max-model-len",           str(max_model_len),
        "--trust-remote-code",
        # vLLM V1 scheduler (enabled by default in ≥ 0.6.0)
        # Log per-request stats: prefill/decode token counts visible in server log
        "--max-num-seqs",            "64",
    ]
    if enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    cmd.extend(extra_args)
    return cmd


def _kill_port(port: int) -> None:
    """Kill any process already listening on the given port (zombie vLLM guard)."""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.status == "LISTEN":
                try:
                    proc = psutil.Process(conn.pid)
                    print(f"  [warn] Port {port} occupied by pid {conn.pid} "
                          f"({proc.name()!r}) — killing before launch.")
                    proc.terminate()
                    proc.wait(timeout=10)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except ImportError:
        # psutil not available: fall back to fuser
        subprocess.run(["fuser", "-k", f"{port}/tcp"],
                       check=False, capture_output=True)


def launch_server(
    hf_repo: str,
    *,
    port: int = 8000,
    gpu_mem_util: float = 0.85,
    max_model_len: int = 8192,
    enable_chunked_prefill: bool = False,
    extra_args: Optional[list[str]] = None,
    startup_timeout: int = 600,
    log_file: Optional[Path] = None,
) -> subprocess.Popen:
    # Kill any zombie vLLM server still holding the port from a prior job.
    _kill_port(port)

    cmd = _build_server_cmd(
        hf_repo, port, gpu_mem_util, max_model_len,
        enable_chunked_prefill, extra_args or [],
    )
    print(f"\n  Launching vLLM server (pid will appear after startup):")
    print(f"    {' '.join(cmd[:4])} ...")
    print(f"  Model:  {hf_repo}")
    print(f"  Port:   {port}")
    if log_file:
        print(f"  Log:    {log_file}")
        log_fh = open(log_file, "w")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.Popen(cmd, text=True)

    # Poll health endpoint until ready
    deadline = time.monotonic() + startup_timeout
    health_url = f"http://localhost:{port}/health"
    print("  Waiting for server to be ready ", end="", flush=True)
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print(f" OK  (pid {proc.pid})")
            return proc
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early (code {proc.returncode}).  "
                    f"Check log at {log_file or 'stderr'}."
                )
            print(".", end="", flush=True)
            time.sleep(5)

    proc.terminate()
    raise RuntimeError(
        f"vLLM server not ready after {startup_timeout}s.  "
        f"Check GPU memory ({hf_repo}) and model download status."
    )


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("  Server stopped.")


# ---------------------------------------------------------------------------
# Async load client
# ---------------------------------------------------------------------------

async def _request(
    session,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    req_id: int,
) -> dict:
    import aiohttp
    payload = {
        "model":       model,
        "prompt":      prompt,
        "max_tokens":  max_tokens,
        "temperature": 0.0,
        "stream":      False,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            body = await resp.json()
            t1   = time.perf_counter()
            usage = body.get("usage", {})
            return {
                "req_id":        req_id,
                "status":        resp.status,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "latency_s":     t1 - t0,
            }
    except Exception as exc:
        return {
            "req_id":    req_id,
            "status":    -1,
            "error":     str(exc)[:200],
            "latency_s": time.perf_counter() - t0,
        }


async def _drive(
    prompts: list[tuple[str, int]],
    model_id: str,
    port: int,
    concurrency: int,
) -> list[dict]:
    try:
        import aiohttp
        from tqdm.asyncio import tqdm as atqdm
    except ImportError:
        raise ImportError(
            "pip install aiohttp tqdm  # required for async load client"
        )

    url = f"http://localhost:{port}/v1/completions"
    sem = asyncio.Semaphore(concurrency)

    async def bounded(i: int, prompt: str, out_len: int) -> dict:
        async with sem:
            return await _request(session, url, model_id, prompt, out_len, i)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded(i, p, o) for i, (p, o) in enumerate(prompts)]
        return await atqdm.gather(*tasks, desc="  requests")


def run_load_client(
    prompts: list[tuple[str, int]],
    model_id: str,
    port: int,
    concurrency: int,
) -> list[dict]:
    return asyncio.run(_drive(prompts, model_id, port, concurrency))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NVML background monitor — delegated to src/nvml_monitor.py (S0.4 policy)
# ---------------------------------------------------------------------------
from src.nvml_monitor import _NVMLMonitor, save_nvml_csv, _nvml_summary  # noqa: E402


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(results: list[dict], is_full_run: bool, wall_s: float) -> dict:
    import numpy as np

    ok  = [r for r in results if r.get("status") == 200]
    err = [r for r in results if r.get("status") != 200]

    if not ok:
        return {"n_ok": 0, "n_err": len(err), "error": "No successful requests"}

    lats = [r["latency_s"] for r in ok]
    out  = [r.get("output_tokens", 0) for r in ok]
    inp  = [r.get("prompt_tokens", 0) for r in ok]

    return {
        "n_ok":                    len(ok),
        "n_err":                   len(err),
        "wall_time_s":             round(wall_s, 2),
        "latency_p50_s":           round(float(np.percentile(lats, 50)), 3),
        "latency_p95_s":           round(float(np.percentile(lats, 95)), 3),
        "latency_p99_s":           round(float(np.percentile(lats, 99)), 3),
        "output_tokens_mean":      round(float(np.mean(out)), 1),
        "prompt_tokens_mean":      round(float(np.mean(inp)), 1),
        "output_tok_per_s_mean":   round(
            float(np.mean([r.get("output_tokens", 0) / r["latency_s"]
                           for r in ok if r["latency_s"] > 0])), 1),
        "valid_for_conclusions":   is_full_run,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="zamba2",
                   choices=list(_HF_REPOS),
                   help="Hybrid model key (maps to HF repo)")
    p.add_argument("--trace", default="sharegpt",
                   choices=["sharegpt", "longbench", "smoke"],
                   help="Workload trace (smoke = synthetic 10 prompts)")
    p.add_argument("--n-requests", type=int, default=200,
                   help="Requests to send (full run).  Smoke always uses 10.")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Concurrent in-flight requests")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM max sequence length (prefill+decode combined)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "results",
                   help="Directory for result JSON")
    p.add_argument("--log-file", type=Path, default=None,
                   help="vLLM server log path (default: stdout of subprocess)")
    p.add_argument("--smoke", action="store_true",
                   help="Force smoke run (10 req) regardless of GPU type")
    p.add_argument("--no-server", action="store_true",
                   help="Skip server launch (assume server already running on --port)")
    p.add_argument("--enable-chunked-prefill", action="store_true",
                   help="Pass --enable-chunked-prefill to vLLM (experimental with SSM models)")
    p.add_argument("--monitor-nvml", action="store_true",
                   help=(
                       "Record device-level NVML metrics (sm_util_pct, mem_util_pct, power_w) "
                       "during serving.  Requires pynvml.  "
                       "Output: results/nvml_{model}_{trace}.csv  "
                       "Motivation figure input (S1.2)."
                   ))
    p.add_argument("--nvml-interval-ms", type=int, default=100,
                   help="NVML polling interval in ms (default: 100)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" serving-eval — vLLM V1 hybrid model serving  (S1.1)")
    print("=" * 70)

    gpu_info, is_full_run = hardware_guard(force_smoke=args.smoke)
    n_req = args.n_requests if is_full_run else 10

    # Load trace
    print(f"\n  Trace: {args.trace}  (n={n_req})")
    from src.trace import load_sharegpt, load_longbench_subset, make_smoke_trace
    import os
    from pathlib import Path as _Path
    _hf_cache = _Path(os.environ["HF_DATASETS_CACHE"]) if "HF_DATASETS_CACHE" in os.environ else None
    if not is_full_run or args.trace == "smoke":
        prompts = make_smoke_trace(n_req)
    elif args.trace == "sharegpt":
        prompts = load_sharegpt(n_samples=n_req, cache_dir=_hf_cache)
    else:
        prompts = load_longbench_subset(n_samples=n_req)
    print(f"  Loaded {len(prompts)} prompts  "
          f"(mean prompt len ≈ {sum(len(p) // 4 for p, _ in prompts) // len(prompts)} tok)")

    hf_repo      = _resolve_hf_repo(args.model)
    gpu_mem_util = _GPU_MEM_UTIL.get(args.model, 0.85)
    # CLI flag takes precedence; fall back to per-model table, then arg default.
    _model_max_len = _MAX_MODEL_LEN.get(args.model)
    max_model_len  = args.max_model_len if args.max_model_len != 8192 else (_model_max_len or args.max_model_len)

    log_path   = args.log_file or (args.output_dir / f"vllm_{args.model}.log")
    server_proc: Optional[subprocess.Popen] = None

    # NVML monitor setup (device-level aggregate; S0.4 policy: NOT per-layer)
    nvml_monitor: Optional[_NVMLMonitor] = None
    if args.monitor_nvml:
        nvml_monitor = _NVMLMonitor(device_id=0, interval_ms=args.nvml_interval_ms)

    try:
        if not args.no_server:
            server_proc = launch_server(
                hf_repo,
                port=args.port,
                gpu_mem_util=gpu_mem_util,
                max_model_len=max_model_len,
                enable_chunked_prefill=args.enable_chunked_prefill,
                log_file=log_path,
            )

        # Start NVML monitoring just before sending requests
        if nvml_monitor is not None:
            nvml_monitor.start()
            print("  NVML monitor: device-level aggregate (sm_util, mem_util, power)")

        # Run load client
        print(f"\n  Load client: concurrency={args.concurrency}  n_req={len(prompts)}")
        print("  (Server logs show prefill/decode token counts per request.)")
        t0      = time.perf_counter()
        results = run_load_client(prompts, hf_repo, args.port, args.concurrency)
        wall_s  = time.perf_counter() - t0

        # Stop NVML monitoring
        nvml_records: list[dict] = []
        if nvml_monitor is not None:
            nvml_records = nvml_monitor.stop()

        stats = analyze(results, is_full_run, wall_s)
        if nvml_records:
            stats.update(_nvml_summary(nvml_records))

        print(f"\n  Results:")
        print(f"    n_ok={stats['n_ok']}  n_err={stats['n_err']}")
        print(f"    latency  p50/p95/p99: "
              f"{stats['latency_p50_s']:.2f}s / "
              f"{stats['latency_p95_s']:.2f}s / "
              f"{stats['latency_p99_s']:.2f}s")
        print(f"    throughput (output): {stats['output_tok_per_s_mean']:.1f} tok/s/req")
        if nvml_records:
            print(f"    GPU util (device-level, NVML aggregate): "
                  f"mean={stats.get('sm_util_mean')}%  "
                  f"p95={stats.get('sm_util_p95')}%")
        print(f"    wall time: {wall_s:.1f}s")

        if not is_full_run:
            print("\n  ** PIPELINE VALIDATION ONLY — numeric conclusions must NOT be drawn. **")
            print(f"  ** Re-run on A100 with --trace {args.trace} for paper results.       **")

        # Save JSON
        tier = "smoke" if not is_full_run else args.trace
        out_path = args.output_dir / f"serving_{args.model}_{tier}.json"
        with open(out_path, "w") as f:
            json.dump({"stats": stats, "per_request": results}, f, indent=2)
        print(f"\n  Saved: {out_path}")

        # Save NVML CSV for plot_motivation.py
        if nvml_records:
            nvml_csv = args.output_dir / f"nvml_{args.model}_{tier}.csv"
            save_nvml_csv(nvml_records, nvml_csv)
            print(f"  NVML CSV for motivation figure: {nvml_csv}")
            print("  Caption note: sm_util_pct = device-level aggregate (NVML ~10Hz),")
            print("  NOT per-layer.  Suitable for serving-level under-utilization evidence.")

        if server_proc is not None:
            print(f"\n  Server log: {log_path}")
            print("  Inspect log for prefill_tokens / decode_tokens mix "
                  "(vLLM V1 logs per-request usage by default).")

    finally:
        if server_proc is not None:
            print("\n  Shutting down vLLM server ...")
            stop_server(server_proc)


if __name__ == "__main__":
    main()
