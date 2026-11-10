#!/usr/bin/env python3
"""Second short confirmation Job of the motivation track (2026-11-10, NON-MEASUREMENT).  Four mechanical questions, no performance claim.
Every time, size and ratio written here is a probe value: not citable, not an arm comparison.  Starting point: py312_probe.py (boot machinery).

  S1 LEN    Can a request just below 2^18 tokens be filled by chunked prefill?  The first probe sent input = context length and got HTTP 400
            (SGLang tokenizer_manager._validate_one_request rejects  len(input) + max_new_tokens >= context_len ).  Path: the sanctioned override
            SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 + --context-length 262144 (Nano declares 131072: TIMING USE ONLY, outputs beyond 131072
            are not claimed correct).  Requests: 2^18-16 (negative control, expected 400), 2^18-128, 2^18-17 (tightest accepted length for
            max_new_tokens 16).  Also checks meta_info.prompt_tokens == n_in (no silent truncation).
  S2 PMCAP  PD-mux bundle (patched tree, chunk -1, D44) with piecewise CUDA graph capture sizes stated explicitly.  The first probe: chunk -1 made the
            default capture list empty (piecewise_cuda_graph_max_tokens = chunked_prefill_size = -1), so nothing was captured.  Boots: main arm
            (piecewise OFF, control), ON default sizes (control, expected no capture), ON --piecewise-cuda-graph-max-tokens 8192 / 16384.
            The Job observes CAPTURE and boot/request success only.  By code reading the PD-mux prefill path runs ModelRunner.forward_split_prefill,
            which never calls the piecewise runner, so a captured graph is NOT evidence of replay (not testable without an engine change).
  S3 A31VLLM  Can the hybrid block alignment of vLLM 0.31.0 be switched off for the a31 composition variant (31 attention / 0 Mamba, dummy weights)?
            Boots: as-is, --block-size 16, --disable-hybrid-kv-cache-manager, config with mamba_num_heads 0.  Records every log line that mentions
            block size / page / hybrid / mamba and the parsed block sizes.  "Does not work" is a result and is recorded as such.
  S4 INSTR  Engine observation records (digest 8d8f6720, four default-OFF flags + the older probes) ON versus OFF on Nano, two configurations
            (fused mixed-chunk patched tree piecewise OFF = SG_PATCH_OFF; PD-mux bundle D44 piecewise OFF), ABBA per configuration in this Job.
            Equivalence = greedy outputs of sequential fixed requests (batch invariance not needed), noise floor = OFF_a vs OFF_b and the
            within-boot repeat.  Overhead = wall of fixed closed-loop bursts, reported with the OFF-pair / ON-pair floor.

  mo_shortcheck.py --out-root DIR [--budget-min N] [--only S1,S2,S3,S4]      run
  mo_shortcheck.py --print-plan                                              boots, argv, env (no GPU)
"""
import argparse, hashlib, json, os, random, re, statistics, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ENG / "results/motivation")]
import py312_probe as P            # noqa: E402  boot machinery (engine_tree boot, GPU memory poll, banner/capture facts)
import mo_spec as MS               # noqa: E402  arm table (build_arm) of the motivation harness
import sb_runner as RU             # noqa: E402  vLLM server env / process wrapper (reused unchanged)

SP = MS.SP
SCHEMA = "mo_shortcheck/v1"
ENGINE_DIGEST_EXPECTED = "8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858"   # env/dev_tree_edits.md #40
CTX_262144 = 2 ** 18
CTX_OUT = 16
SEED = f"mo_shortcheck|{SP.PLAN_SEED}"
STAGES = ("S1", "S2", "S3", "S4")
STAGE_CAP_MIN = {"S1": 15, "S2": 20, "S3": 35, "S4": 50}      # per-stage cap; the Job budget is --budget-min over all of them
# Boot-test constants (NOT measurement values): the inherited cap 48 / mem 0.6 / ctx 20480 of the sealed policy harness, used only so the argv equals
# the registered arm's argv; the cap measurement has not fixed them yet.
BOOT_FIXED = {"mem": 0.6, "cap": 48, "context": 20480, "vllm_chunk": 8192, "vllm_backend": "FLASHINFER", "sg_venv_freeze": "boot-test"}
ON_WINDOW_FLAGS = ("PDMUX_DECODE_PROBE_PATH", "PDMUX_PROBE_EXTEND_TIME", "PDMUX_PROBE_FORWARD_INTERIOR", "PDMUX_ADMISSION_PATH",
                   "PDMUX_ADMISSION_REQ_RECORDS", "PDMUX_ADMISSION_MIXED_RATIO")
PM_ONLY_FLAGS = ("PDMUX_TELEMETRY_PATH", "PDMUX_PHASE_EVENTS", "PDMUX_PHASE_DECODE_SHARE")
VOCAB = 100000
EQ_LENS = (512, 1024, 2048, 3072)
EQ_N, EQ_OUT = 8, 64
MIX_LENS, MIX_N, MIX_OUT, MIX_REPS = (512, 2048, 4096, 1024), 64, 128, 3
DEC_N, DEC_IN, DEC_OUT, DEC_REPS = 32, 256, 512, 2


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def rng(*salt):
    return random.Random("|".join([SEED, *map(str, salt)]))


def prompt_ids(seed, n):
    r = random.Random(seed)
    return [r.randrange(1000, VOCAB) for _ in range(n)]


def arm_params(key):
    return dict(next(p for p in MS.ARM_PARAMS if p["key"] == key))


# ---------------------------------------------------------------------------------------------------------------- argv builders (pure)
def flag_value(cmd, flag, default=None):
    return cmd[cmd.index(flag) + 1] if flag in cmd else default


def pm_cmd(piecewise, max_tokens=None):
    """PD-mux bundle argv of the motivation harness (non-sticky D44, chunk -1, overlap OFF) with piecewise OFF (main arm) or ON,
    optionally with an explicit --piecewise-cuda-graph-max-tokens."""
    p = arm_params("PDMUX_BUNDLE")
    p["piecewise"] = piecewise
    a = MS.build_arm(p, BOOT_FIXED)
    cmd = P.resolve(a["cmd"])
    if max_tokens is not None:
        cmd = P.setarg(cmd, "--piecewise-cuda-graph-max-tokens", max_tokens)
    return cmd, dict(a["env"])


def sg_patch_cmd():
    a = MS.build_arm(arm_params("SG_PATCH_OFF"), BOOT_FIXED)
    return P.resolve(a["cmd"]), dict(a["env"])


def ctx_cmd():
    """The first probe's ctx256k argv unchanged (only the requests change)."""
    return next(b for b in P.plan() if b["id"] == "ctx256k")


def ctx_requests():
    """(n_in, n_out, expected) -- the boundary rule is  n_in + n_out < context_len  (tokenizer_manager.py:795,815), strict."""
    return [(CTX_262144 - CTX_OUT, CTX_OUT, "http_400"),           # n_in + n_out == context_len: negative control
            (CTX_262144 - 128, CTX_OUT, "ok"),
            (CTX_262144 - CTX_OUT - 1, CTX_OUT, "ok")]            # n_in + n_out == context_len - 1: tightest accepted


def on_env(bdir, pm):
    d = str(bdir)
    env = {"PDMUX_DECODE_PROBE_PATH": f"{d}/decode_probe.jsonl", "PDMUX_PROBE_EXTEND_TIME": "1", "PDMUX_PROBE_FORWARD_INTERIOR": "1",
           "PDMUX_ADMISSION_PATH": f"{d}/admission.jsonl", "PDMUX_ADMISSION_REQ_RECORDS": "1", "PDMUX_ADMISSION_MIXED_RATIO": "1"}
    if pm:
        env.update({"PDMUX_TELEMETRY_PATH": f"{d}/tel.jsonl", "PDMUX_PHASE_EVENTS": "1", "PDMUX_PHASE_DECODE_SHARE": "1"})
    return env


def abba_orders():
    """-> {config: ['OFF','ON','ON','OFF'] or ['ON','OFF','OFF','ON']} (seeded coin per config) and the config order (seeded)."""
    cfgs = ["SG", "PM"]
    rng("s4", "cfg").shuffle(cfgs)
    orders = {c: (["OFF", "ON", "ON", "OFF"] if rng("s4", "abba", c).random() < 0.5 else ["ON", "OFF", "OFF", "ON"]) for c in cfgs}
    return cfgs, orders


def s2_order():
    ids = ["pm_off_main", "pm_on_default", "pm_on_max8192", "pm_on_max16384"]
    rng("s2").shuffle(ids)
    return ids


def a31_flags(cand):
    return {"as_is": [], "block16": ["--block-size", "16"], "nohybridkv": ["--disable-hybrid-kv-cache-manager"], "mamba0": [], "layer_types": []}[cand]


A31_CANDS = ("as_is", "block16", "nohybridkv", "mamba0", "layer_types")
A31_MODEL_DIR = {"mamba0": "a31_mamba0", "layer_types": "a31_nonhybrid"}          # candidates with their own config; the others share a31


def a31_order():
    ids = list(A31_CANDS)
    rng("s3").shuffle(ids)
    return ids


def vllm_cmd(model_dir, extra):
    base = SP.arm("VL-stock")["cmd"]
    cmd = list(base)
    i = cmd.index("--model")
    cmd[i + 1] = str(model_dir)
    if "--revision" in cmd:
        j = cmd.index("--revision")
        del cmd[j:j + 2]
    return cmd + ["--load-format", "dummy"] + list(extra)


def neutralize_mamba(cfg):
    """mamba0 candidate: the a31 config keeps every mamba_* field, so vLLM still sees a hybrid model with a Mamba page size.  Set the number of
    Mamba heads to 0 (the 'mamba fields 0' idea of the CPU prep); whether vLLM accepts it is what the boot answers."""
    c = dict(cfg)
    c["mamba_num_heads"] = 0
    return c


def attention_only(cfg, n_attn):
    """layer_types candidate: vLLM ModelConfig.is_hybrid is False when every entry of hf_config.layer_types is attention (its Granite micro branch).
    NemotronH builds its layers from hybrid_override_pattern and never reads layer_types, so only that hybrid decision changes."""
    return dict(cfg, layer_types=["attention"] * int(n_attn))


# ---------------------------------------------------------------------------------------------------------------- log facts (pure)
SG_LOG_PW = re.compile(r"(Disable piecewise CUDA graph[^\n]*|Capture piecewise CUDA graph (?:begin|end)[^\n]*|Piecewise CUDA Graph[^\n]*)")
TOKENS_RE = re.compile(r"piecewise_cuda_graph_tokens=\[([0-9, ]*)\]")
MAXTOK_RE = re.compile(r"piecewise_cuda_graph_max_tokens=(\d+|None|-1)")
BANNER_RE = re.compile(r"max_total_num_tokens=(\d+)")
AVAIL_RE = re.compile(r"available_gpu_mem=([0-9.]+) GB")


def sg_capture_facts(text):
    tok = TOKENS_RE.search(text)
    sizes = [int(x) for x in tok.group(1).replace(" ", "").split(",") if x] if tok else None
    mt = MAXTOK_RE.search(text)
    tot = BANNER_RE.search(text)
    av = AVAIL_RE.findall(text)
    return {"piecewise_tokens_list": sizes, "n_capture_sizes": None if sizes is None else len(sizes),
            "max_capture_size": None if not sizes else max(sizes),
            "piecewise_max_tokens_arg": mt.group(1) if mt else None,
            "pw_lines": [m.group(1)[:300] for m in SG_LOG_PW.finditer(text)][:12],
            "max_total_num_tokens": int(tot.group(1)) if tot else None,
            "available_gpu_mem_gb_last": float(av[-1]) if av else None,
            "captured": "Capture piecewise CUDA graph end" in text, "failed": "Piecewise CUDA Graph failed" in text,
            "disabled_reason": next((m.group(1)[:300] for m in SG_LOG_PW.finditer(text) if m.group(1).startswith("Disable")), None)}


VL_BLOCK_LINE = re.compile(r"(?i)block.?size|page.?size|hybrid|mamba|align|kv cache|attention backend")
VL_BLOCK_NUM = [re.compile(r"(?i)attention block size[^0-9\n]{0,40}(\d+)"),
                re.compile(r"(?i)block_size['\"]?\s*[=:]\s*(\d+)"),
                re.compile(r"(?i)\bblock size[^0-9\n]{0,20}(\d+)")]


def vllm_block_facts(text, keep=40):
    lines = [ln.strip()[:400] for ln in text.splitlines() if VL_BLOCK_LINE.search(ln)]
    nums = []
    for rx in VL_BLOCK_NUM:
        nums += [int(x) for x in rx.findall(text)]
    return {"matched_lines": list(dict.fromkeys(lines))[:keep], "n_matched": len(lines),
            "block_size_candidates": list(dict.fromkeys(nums)),
            "block_size_attn_line": next((int(m.group(1)) for m in (VL_BLOCK_NUM[0].search(ln) for ln in text.splitlines()) if m), None),
            "block_size_last": nums[-1] if nums else None,
            "attn_block_line_seen": "Setting attention block size" in text, "mamba_padding_line_seen": "Padding mamba page size" in text,
            "kv_cache_tokens": [int(x.replace(",", "")) for x in re.findall(r"(?i)GPU KV cache size:\s*([0-9,]+)\s*tokens", text)]}


def a31_verdict(default, cand):
    """Mechanical comparison of one candidate with the as-is boot.  CHANGED needs a parsed block size that differs from the as-is value."""
    if not cand.get("boot_ok"):
        return "BOOT_FAILED"
    if cand.get("attn_block_line_seen") is False and cand.get("mamba_padding_line_seen") is False:
        return "NO_ALIGNMENT_LINES"
    d = default.get("block_size_attn_line") or default.get("block_size_last")
    c = cand.get("block_size_attn_line") or cand.get("block_size_last")
    if d is None or c is None:
        return "UNPARSED"
    return "CHANGED" if c != d else "UNCHANGED"


# ---------------------------------------------------------------------------------------------------------------- S4 analysis (pure)
def eq_compare(a, b):
    """Mismatches between two lists of per-prompt records (text sha + completion_tokens); None when either side is missing."""
    if not a or not b or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if (x["sha"], x["n"]) != (y["sha"], y["n"]))


def equivalence(cfg_boots):
    """cfg_boots: {'OFF_a','OFF_b','ON_a','ON_b'} -> boot record with 'eq1' (first pass) and 'eq2' (repeat).  Verdict is mechanical."""
    g = lambda k: (cfg_boots.get(k) or {}).get("eq1")
    out = {"pairs": {}, "within_boot_repeat": {}}
    for x, y in (("OFF_a", "OFF_b"), ("ON_a", "ON_b"), ("ON_a", "OFF_a"), ("ON_a", "OFF_b"), ("ON_b", "OFF_a"), ("ON_b", "OFF_b")):
        out["pairs"][f"{x}~{y}"] = eq_compare(g(x), g(y))
    for k, v in cfg_boots.items():
        out["within_boot_repeat"][k] = eq_compare((v or {}).get("eq1"), (v or {}).get("eq2"))
    floor = [out["pairs"]["OFF_a~OFF_b"], out["pairs"]["ON_a~ON_b"]] + list(out["within_boot_repeat"].values())
    cross = [out["pairs"][k] for k in ("ON_a~OFF_a", "ON_a~OFF_b", "ON_b~OFF_a", "ON_b~OFF_b")]
    if any(v is None for v in floor + cross) or len(cfg_boots) < 4:
        out["verdict"] = "INCOMPLETE"
    elif any(floor):
        out["verdict"] = "FLOOR_NONZERO"          # the engine is not repeatable at this exactness: ON vs OFF equality is undecidable here
    elif any(cross):
        out["verdict"] = "EQUIV_FAIL"
    else:
        out["verdict"] = "EQUIV_PASS"
    out["note"] = "sequential greedy outputs, text sha256 + completion_tokens; PASS = zero mismatches ON vs OFF AND zero in every floor comparison"
    return out


def overhead(cfg_boots):
    """Ratios of burst walls ON/OFF with the pair floors.  Probe values; no performance claim."""
    def wall(rec, kind):
        w = (rec or {}).get(kind)
        return statistics.median(w) if w else None
    res = {}
    for kind in ("mix_wall_s", "dec_wall_s"):
        w = {k: wall(v, kind) for k, v in cfg_boots.items()}
        if any(w.get(k) is None for k in ("OFF_a", "OFF_b", "ON_a", "ON_b")):
            res[kind] = {"status": "INCOMPLETE", "boot_medians": w}
            continue
        off, on = (w["OFF_a"] + w["OFF_b"]) / 2, (w["ON_a"] + w["ON_b"]) / 2
        res[kind] = {"status": "PROBE_VALUE_NOT_CITABLE", "boot_medians": w, "ratio_on_over_off": on / off,
                     "floor_off_pair": abs(w["OFF_a"] / w["OFF_b"] - 1), "floor_on_pair": abs(w["ON_a"] / w["ON_b"] - 1)}
    return res


REC_EXPECT_SG = ("decode_forward", "extend_forward", "forward_interior", "admission_request", "mixed_opportunity")
REC_EXPECT_PM = REC_EXPECT_SG + ("decode_iteration",)


def read_jsonl(path):
    out = []
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def record_facts(bdir, pm, on):
    files = {n: (Path(bdir) / n).exists() for n in ("decode_probe.jsonl", "admission.jsonl", "tel.jsonl")}
    recs = []
    for n in ("decode_probe.jsonl", "admission.jsonl", "tel.jsonl"):
        recs += read_jsonl(Path(bdir) / n)
    kinds = {}
    for r in recs:
        kinds[r.get("event")] = kinds.get(r.get("event"), 0) + 1
    want = REC_EXPECT_PM if pm else REC_EXPECT_SG
    return {"files_exist": files, "kinds": kinds, "missing_kinds": [k for k in want if kinds.get(k, 0) == 0] if on else None,
            "t_issue_start_seen": any("t_issue_start" in r for r in recs if r.get("event") == "decode_iteration") if pm and on else None,
            "off_files_absent": (not any(files.values())) if not on else None}


# ---------------------------------------------------------------------------------------------------------------- HTTP drivers
def gen(port, n_in, n_out, seed, timeout):
    """/generate with input_ids, greedy, forced length.  Never raises: HTTP errors are data."""
    body = {"input_ids": prompt_ids(seed, n_in), "sampling_params": {"max_new_tokens": n_out, "ignore_eos": True, "temperature": 0}}
    t0 = time.time()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        mi = d["meta_info"]
        return {"ok": mi["completion_tokens"] == n_out, "status": 200, "s": round(time.time() - t0, 3), "n": mi["completion_tokens"],
                "prompt_tokens": mi.get("prompt_tokens"), "sha": sha_text(d.get("text", ""))}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "s": round(time.time() - t0, 3), "error": e.read()[:300].decode(errors="replace"), "n": 0, "sha": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": None, "s": round(time.time() - t0, 3), "error": repr(e)[:300], "n": 0, "sha": None}


def burst(port, reqs, timeout):
    """All requests at once (closed loop, one wave).  -> wall seconds, per-request results."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, len(reqs))) as ex:
        fs = [ex.submit(gen, port, n_in, n_out, sd, timeout) for (n_in, n_out, sd) in reqs]
        rs = [f.result() for f in fs]
    return round(time.time() - t0, 3), rs


def eq_requests():
    return [(EQ_LENS[i % len(EQ_LENS)], EQ_OUT, 5000 + i) for i in range(EQ_N)]


def mix_requests():
    return [(MIX_LENS[j % len(MIX_LENS)], MIX_OUT, 6000 + j) for j in range(MIX_N)]


def dec_requests():
    return [(DEC_IN, DEC_OUT, 7000 + j) for j in range(DEC_N)]


def drive_ctx(port, d, req_timeout):
    rows, ok = [], True
    for n_in, n_out, expect in ctx_requests():
        r = gen(port, n_in, n_out, 11, max(req_timeout, 900))
        good = (r["status"] == 400) if expect == "http_400" else (r["ok"] and r.get("prompt_tokens") == n_in)
        ok = ok and good
        rows.append(dict(n_in=n_in, n_out=n_out, total=n_in + n_out, expect=expect, matches_expectation=good, **r))
    return {"ok": ok, "rows": rows, "context_length": CTX_262144, "timing_use_only": True,
            "note": "Nano declares 131072; context 262144 is the SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN override; outputs beyond 131072 are not claimed correct"}


def drive_instr(port, d, req_timeout):
    """warm-up (discarded) -> sequential equivalence pass x2 -> mixed bursts -> decode bursts.  Returns raw per-request results."""
    t = max(req_timeout, 600)
    burst(port, [(1024, 32, 4000 + i) for i in range(16)], t)
    eq1 = [gen(port, a, b, c, t) for (a, b, c) in eq_requests()]
    eq2 = [gen(port, a, b, c, t) for (a, b, c) in eq_requests()]
    mix = [burst(port, mix_requests(), t) for _ in range(MIX_REPS)]
    dec = [burst(port, dec_requests(), t) for _ in range(DEC_REPS)]
    allr = eq1 + eq2 + [x for w, rs in mix + dec for x in rs]
    return {"ok": all(r["ok"] for r in allr), "eq1": [{"sha": r["sha"], "n": r["n"], "ok": r["ok"]} for r in eq1],
            "eq2": [{"sha": r["sha"], "n": r["n"], "ok": r["ok"]} for r in eq2],
            "mix_wall_s": [w for w, _ in mix], "dec_wall_s": [w for w, _ in dec],
            "mix_burst_sha": [sha_text("".join(str(r["sha"]) for r in rs)) for _, rs in mix],
            "mix_req_s_p50": [statistics.median(r["s"] for r in rs) for _, rs in mix],
            "n_failed": sum(1 for r in allr if not r["ok"]), "first_error": next((r.get("error") for r in allr if not r["ok"]), None)}


# ---------------------------------------------------------------------------------------------------------------- plan
def boot_dict(bid, tree, cmd, env, want_pw, reqs=None, driver=None):
    return dict(id=bid, tree=tree, cmd=cmd, env=env, reqs=reqs or [], want_piecewise=want_pw, driver=driver)


def plan(out_root):
    out = Path(out_root)
    P_ = {}
    ctx = ctx_cmd()
    P_["S1"] = [boot_dict("ctx262144", ctx["tree"], ctx["cmd"], ctx["env"], True, driver=drive_ctx)]
    short = [(8, 2048, 64), (1, 16000, 64)]
    s2 = {}
    c, e = pm_cmd("off")
    s2["pm_off_main"] = boot_dict("pm_off_main", "patched", c, e, False, reqs=short)
    c, e = pm_cmd("on")
    s2["pm_on_default"] = boot_dict("pm_on_default", "patched", c, e, False, reqs=short)
    for n in (8192, 16384):
        c, e = pm_cmd("on", n)
        s2[f"pm_on_max{n}"] = boot_dict(f"pm_on_max{n}", "patched", c, e, True, reqs=short)
    P_["S2"] = [s2[i] for i in s2_order()]
    cfgs, orders = abba_orders()
    s4 = []
    for cfg in cfgs:
        cmd, env0 = sg_patch_cmd() if cfg == "SG" else pm_cmd("off")
        want_pw = False                                  # both registered configurations are piecewise OFF
        tag = {"OFF": iter("ab"), "ON": iter("ab")}
        for pos, state in enumerate(orders[cfg]):
            bid = f"s4_{cfg}_{state}_{next(tag[state])}"
            bdir = out / bid
            env = dict(env0, PDMUX_RUN_ID="mo_shortcheck", PDMUX_WORKLOAD_ID=bid)
            if state == "ON":
                env.update(on_env(bdir, cfg == "PM"))
            s4.append(boot_dict(bid, "patched", cmd, env, want_pw, driver=drive_instr))
    P_["S4"] = s4
    return P_


# ---------------------------------------------------------------------------------------------------------------- run
def stage_boot(b, out, budget_left_s):
    t0 = time.time()
    boot_timeout = min(1200, budget_left_s - 120)
    req_timeout = max(300, min(900, budget_left_s - 60))
    res = P.run_boot(b, out, boot_timeout=boot_timeout, req_timeout=req_timeout)
    d = Path(out) / b["id"]
    txt = (d / "server.log").read_text(errors="replace") if (d / "server.log").exists() else ""
    res["capture"] = sg_capture_facts(txt)
    res["t_start_epoch"], res["t_end_epoch"] = round(t0, 1), round(time.time(), 1)
    res["driver_name"] = getattr(b.get("driver"), "__name__", None)
    if b["id"].startswith("s4_"):
        pm = "_PM_" in b["id"]
        on = "_ON_" in b["id"]
        res["records"] = record_facts(d, pm, on)
    (d / "stage.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def build_a31_variants(out, hf_home):
    import compo_probe_helper as CH
    ref, rev, snap = CH.LH.snapshot_dir(hf_home, CH.MV.NANO_ID)
    if rev != CH.MV.NANO_REV or not snap or not os.path.isdir(snap):
        raise RuntimeError(f"Nano snapshot missing or revision {rev!r} != pinned")
    vdir = Path(out) / "variants"
    rec = CH.MV.build_variant(snap, str(vdir / "a31"), 31, force=True, source_meta={"revision_observed": rev})
    m0 = vdir / "a31_mamba0"
    CH.MV.build_variant(snap, str(m0), 31, force=True, source_meta={"revision_observed": rev})
    cfg = json.load(open(m0 / "config.json"))
    (m0 / "config.json").write_text(json.dumps(neutralize_mamba(cfg), indent=2) + "\n")
    nh = vdir / "a31_nonhybrid"
    CH.MV.build_variant(snap, str(nh), 31, force=True, source_meta={"revision_observed": rev})
    cfg = json.load(open(nh / "config.json"))
    (nh / "config.json").write_text(json.dumps(attention_only(cfg, rec["n_attn"]), indent=2) + "\n")
    return {"a31": str(vdir / "a31"), "a31_mamba0": str(m0), "a31_nonhybrid": str(nh), "n_attn": rec["n_attn"], "n_mamba": rec["n_mamba"]}


def vllm_boot(cand, model_dir, out, boot_timeout):
    d = Path(out) / f"a31_{cand}"
    d.mkdir(parents=True, exist_ok=True)
    cmd = vllm_cmd(model_dir, a31_flags(cand))
    a = {"name": f"A31-{cand}", "engine": "vllm", "tree": "vllm", "cmd": cmd, "env": dict(SP.VL_ENV), "api": "openai", "image_key": SP.IMAGE_KEY}
    srv = RU.ProcessServer(d, a, boot_timeout=boot_timeout)
    res = {"id": f"a31_{cand}", "cand": cand, "argv": cmd, "status": "FAIL", "requests": []}
    t0 = time.time()
    st = srv.start()
    res["boot_state"], res["boot_s"], res["boot_ok"] = st, round(time.time() - t0), st == "ok"
    try:
        if res["boot_ok"]:
            for i in range(4):
                body = {"model": str(model_dir), "prompt": prompt_ids(8000 + i, 2048), "max_tokens": 32, "temperature": 0, "ignore_eos": True}
                tr = time.time()
                try:
                    rq = urllib.request.Request(f"{srv.base_url}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    dd = json.loads(urllib.request.urlopen(rq, timeout=300).read())
                    res["requests"].append({"ok": dd["usage"]["completion_tokens"] == 32, "s": round(time.time() - tr, 2)})
                except Exception as e:  # noqa: BLE001
                    res["requests"].append({"ok": False, "error": repr(e)[:300]})
    finally:
        srv.stop()
    txt = (d / "server.log").read_text(errors="replace") if (d / "server.log").exists() else ""
    res.update(vllm_block_facts(txt))
    res["died_tail"] = None if res["boot_ok"] else txt[-1500:]
    res["status"] = "PASS" if res["boot_ok"] and res["requests"] and all(r["ok"] for r in res["requests"]) else "FAIL"
    time.sleep(5)
    try:
        res["gpu_after_mib"] = P.gpu_used_mib()
    except Exception:  # noqa: BLE001
        res["gpu_after_mib"] = None
    (d / "stage.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def vllm_code_dump(out):
    """CPU, no boot: where the alignment lives and which flags exist (read by a human; no verdict)."""
    d = Path(out) / "a31_code"
    d.mkdir(parents=True, exist_ok=True)
    res = {}
    try:
        import glob
        hits = glob.glob("/opt/vllm-venv/lib/python3*/site-packages/vllm/platforms/interface.py")
        if hits:
            lines = open(hits[0]).read().splitlines()
            keep = [f"{i + 1}: {ln}" for i, ln in enumerate(lines) if re.search(r"(?i)block_size|mamba_page|attn_page|align|user_specified|hybrid", ln)]
            (d / "interface_align_lines.txt").write_text("\n".join(keep[:400]) + "\n")
            res["interface_py"] = hits[0]
            res["n_lines"] = len(keep)
    except Exception as e:  # noqa: BLE001
        res["interface_error"] = repr(e)[:200]
    try:
        r = subprocess.run([SP.VLLM_PY, "-m", "vllm.entrypoints.openai.api_server", "--help=all"], capture_output=True, text=True, timeout=240)
        keep = [ln for ln in (r.stdout + r.stderr).splitlines() if re.search(r"(?i)block.?size|hybrid|mamba", ln)]
        (d / "help_block_hybrid_mamba.txt").write_text("\n".join(keep[:200]) + "\n")
        res["help_rc"], res["help_lines"] = r.returncode, len(keep)
    except Exception as e:  # noqa: BLE001
        res["help_error"] = repr(e)[:200]
    (d / "stage.json").write_text(json.dumps(res, indent=1))
    return res


def finish_s4(results, out):
    summ = {}
    for cfg in ("SG", "PM"):
        boots = {}
        for bid, r in results.items():
            if not bid.startswith(f"s4_{cfg}_"):
                continue
            key = bid.split(f"s4_{cfg}_", 1)[1]            # OFF_a / OFF_b / ON_a / ON_b (letter = run order within the state)
            boots[key] = (r.get("driver") or {}) | {"status": r.get("status"), "records": r.get("records")}
        if boots:
            summ[cfg] = {"equivalence": equivalence(boots), "overhead": overhead(boots),
                         "records_on": {k: v.get("records") for k, v in boots.items() if k.startswith("ON")},
                         "records_off": {k: v.get("records") for k, v in boots.items() if k.startswith("OFF")},
                         "boot_status": {k: v.get("status") for k, v in boots.items()}}
    (Path(out) / "s4_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


def selected(order, arg, known):
    """Items of the seeded order named in the comma list arg; an unknown name is an error, not a silently smaller Job."""
    names = [x for x in arg.split(",") if x]
    bad = set(names) - set(known)
    if bad or not names:
        raise SystemExit(f"unknown or empty selection {arg!r}; known {known}")
    return [x for x in order if x in names]


def stage_boots(pl, stage, s4_cfgs):
    return [b for b in pl[stage] if stage != "S4" or b["id"].split("_")[1] in s4_cfgs]


def engine_digest_note():
    return {"expected": ENGINE_DIGEST_EXPECTED, "checked_by": "mo_shortcheck_job.sh (engine_digest.py check) before any boot"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root")
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument("--budget-min", type=float, default=130)
    ap.add_argument("--only", default=",".join(STAGES))
    ap.add_argument("--s3-cands", default=",".join(A31_CANDS), help="comma list of a31 candidates to boot")
    ap.add_argument("--s4-cfgs", default="SG,PM", help="comma list of S4 configurations to boot")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/data/hf"))
    a = ap.parse_args(argv)
    s3_cands = selected(a31_order(), a.s3_cands, A31_CANDS)
    s4_cfgs = selected(["SG", "PM"], a.s4_cfgs, ("SG", "PM"))
    stages = [s for s in a.only.split(",") if s in STAGES]
    if a.print_plan:
        pl = plan("<OUT>")
        for s in stages:
            print(f"== {s} (cap {STAGE_CAP_MIN[s]} min)")
            if s == "S3":
                for c in s3_cands:
                    print(f"  a31_{c}", "model=<variants>/" + A31_MODEL_DIR.get(c, "a31"), "env", SP.VL_ENV)
                    print("    ", " ".join(vllm_cmd("<variant>", a31_flags(c))))
                continue
            for b in stage_boots(pl, s, s4_cfgs):
                print(f"  {b['id']} tree={b['tree']} env={b['env']} driver={getattr(b['driver'], '__name__', None)} reqs={b['reqs']} want_piecewise={b['want_piecewise']}")
                print("    ", " ".join(b["cmd"]))
        return 0
    out = Path(a.out_root) / f"mo_shortcheck_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out.mkdir(parents=True)
    t0 = time.time()
    deadline = t0 + a.budget_min * 60
    pl = plan(out)
    results, stage_status = {}, {}
    (out / "plan.json").write_text(json.dumps({"schema": SCHEMA, "engine_digest": engine_digest_note(), "stages": stages, "boot_fixed": BOOT_FIXED,
                                                "s2_order": s2_order(), "s3_order": s3_cands, "s4_cfgs": s4_cfgs, "s4_orders": abba_orders(),
                                                "ctx_requests": ctx_requests(), "boot_ids": {s: [b["id"] for b in pl[s]] for s in pl}}, indent=1, default=str))
    for s in stages:
        sdead = min(deadline, time.time() + STAGE_CAP_MIN[s] * 60)
        print(f"[shortcheck] stage {s}", flush=True)
        if s == "S3":
            vllm_code_dump(out)
            try:
                vm = build_a31_variants(out, a.hf_home)
            except Exception as e:  # noqa: BLE001
                stage_status[s] = f"VARIANTS_FAILED {e!r}"[:300]
                continue
            for c in s3_cands:
                left = sdead - time.time()
                if left < 420:
                    results[f"a31_{c}"] = {"status": "SKIPPED", "reason": "budget"}
                    continue
                results[f"a31_{c}"] = vllm_boot(c, vm[A31_MODEL_DIR.get(c, "a31")], out, boot_timeout=min(1200, left - 120))
                print(f"[shortcheck] a31_{c} -> {results[f'a31_{c}']['status']}", flush=True)
            dflt = results.get("a31_as_is", {})
            for c in A31_CANDS:
                if c != "as_is" and "boot_ok" in results.get(f"a31_{c}", {}):
                    results[f"a31_{c}"]["verdict_vs_as_is"] = a31_verdict(dflt, results[f"a31_{c}"])
            stage_status[s] = "RAN"
            continue
        for b in stage_boots(pl, s, s4_cfgs):
            left = sdead - time.time()
            if left < 420:
                results[b["id"]] = {"id": b["id"], "status": "SKIPPED", "reason": "budget"}
                continue
            print(f"[shortcheck] {b['id']}", flush=True)
            try:
                results[b["id"]] = stage_boot(b, out, left)
            except Exception as e:  # noqa: BLE001
                results[b["id"]] = {"id": b["id"], "status": "ERROR", "error": repr(e)[:300]}
            print(f"[shortcheck] {b['id']} -> {results[b['id']].get('status')}", flush=True)
        stage_status[s] = "RAN"
    s4 = finish_s4(results, out) if "S4" in stages else {}
    (out / "summary.json").write_text(json.dumps({"schema": SCHEMA, "stage_status": stage_status, "elapsed_min": round((time.time() - t0) / 60, 1),
                                                   "boots": {k: {kk: vv for kk, vv in v.items() if kk in ("status", "boot_s", "boot_ok", "verdict_vs_as_is", "peak_gpu_mib")}
                                                             for k, v in results.items()}, "s4": s4}, indent=1, default=str))
    print(json.dumps({k: v.get("status") for k, v in results.items()}, indent=0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
