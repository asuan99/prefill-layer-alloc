"""G-gamma observer-effect DECOMPOSITION (decision 21, 2026-10-21): which instrument costs what.  Diagnostic (descriptive), NOT a gate.

Reuses the engine_gate_driver machinery (Gate, Server, boot_and_run, run_cell, digest checks); the driver binds itself with bind().
Engine bytes are NOT touched: the levels differ only in which PDMUX_* env vars the server is launched with.

LEVELS (agnostic arm, shape A, cap 48, mem 0.6, HI/LO cells exactly as G-gamma; PDMUX_TELEMETRY_PATH / PDMUX_PHASE_EVENTS never set)
  OFF  no probe env
  I5   PDMUX_ADMISSION_PATH only
  I6   PDMUX_DECODE_PROBE_PATH + PDMUX_DECODE_PROBE_GPU_EVENTS=0   (I-6 padding fields + host stamps; no CUDA events)
  I46  PDMUX_DECODE_PROBE_PATH + PDMUX_DECODE_PROBE_GPU_EVENTS=1   (I-4 CUDA events + I-6)
  TEL  PDMUX_TELEMETRY_PATH + PDMUX_PHASE_EVENTS=1   (the two other env vars of the G-gamma ON condition, same values; no probe env)
The G-gamma ON condition = I5 + I46 + TEL together ("ALL"); ALL is NOT run: 6 levels x 4 blocks x 650 s = 260 min of instances alone
does not fit a 270 min cap (budget = cap - 10 min, plus digests/overheads), so the 5-level design (20 x 650 s = 217 min, cap 250) is used.
I-4 alone does not exist: I-6 fields come with the same decode_probe hook (decode_probe.py: GPU_EVENTS=0 "keeps I-6 and drops the
CUDA events").  The I-4 effect is therefore the paired contrast I46 vs I6 ("I4_increment").

DESIGN: 4 blocks x 5 levels = 20 instances; block b uses seed SEEDS[b] for every level (same trace).  Order = a 4 x 5 Latin rectangle
(every level once per block, never twice at the same position => position-balanced as far as 4 blocks allow; a level occupies 4 of the
5 positions).  DISCLOSED LIMITS: a full Latin rectangle design for an odd number of levels needs 2 x 5 = 10 blocks; with 4 blocks the carry-over
(neighbour) balance is incomplete -- the registered rectangle has 15 distinct ordered neighbour pairs out of 16 (one pair occurs twice,
exhaustive search shows 0 repeats is impossible for 4 rows) and every level misses one position.  Residual time drift is therefore
not exactly cancelled; position_drift_log reports it.  Registered by ORDER_SHA.

PRE-REGISTERED SELECTION RULE (fixed before any data): metrics lambda*_thr (HI-cell achieved rate) and ITL p95 (LO cell), each as
ln(level/reference) per block (paired), mean/SD/95 % t-CI (n >= 3 valid blocks, else INSUFFICIENT).  Candidates = {I5 vs OFF,
I6 vs OFF, TEL vs OFF, I4_increment = I46 vs I6}.  A candidate is FLAGGED on a metric iff its CI excludes 0 in the degrading direction
(lambda down: ci_hi < 0; ITL up: ci_lo > 0).  Every flagged candidate is reported (ordered by point estimate); `largest` = the flagged
one with the largest degrading point estimate, no tie-breaking beyond that; none flagged -> NONE_RESOLVED.  I46 vs OFF (total) is
reported but is not a candidate.  No multiplicity correction (descriptive; 3 candidates x 2 metrics).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

D = None      # the bound engine_gate_driver module


def bind(drv):
    global D
    D = drv


LEVELS = ("OFF", "I5", "I6", "I46", "TEL")
LEVEL_ENV = {
    "OFF": {},
    "I5": {"PDMUX_ADMISSION_PATH": "{dir}/admission.jsonl"},
    "I6": {"PDMUX_DECODE_PROBE_PATH": "{dir}/decode_probe.jsonl", "PDMUX_DECODE_PROBE_GPU_EVENTS": "0"},
    "I46": {"PDMUX_DECODE_PROBE_PATH": "{dir}/decode_probe.jsonl", "PDMUX_DECODE_PROBE_GPU_EVENTS": "1"},
    "TEL": {"PDMUX_TELEMETRY_PATH": "{dir}/tel.jsonl", "PDMUX_PHASE_EVENTS": "1"},     # = the G-gamma ON values (driver Server.env)
}
EXPECT = {"OFF": dict(decode=False, events=None, admission=False, tel=False),
          "I5": dict(decode=False, events=None, admission=True, tel=False),
          "I6": dict(decode=True, events=False, admission=False, tel=False),
          "I46": dict(decode=True, events=True, admission=False, tel=False),
          "TEL": dict(decode=False, events=None, admission=False, tel=True)}
_RECT = ((0, 1, 2, 3, 4), (1, 3, 0, 4, 2), (2, 4, 1, 0, 3), (4, 0, 3, 2, 1))     # found by exhaustive search: min 1 repeated neighbour pair
PLAN = tuple(tuple(LEVELS[i] for i in row) for row in _RECT)        # PLAN[block-1][position-1]
SEEDS = (4386, 4162, 251, 2630)                                          # = G-gamma pairs 1-4
ORDER_SHA = "18d209665a5d81511d4c8afc625b2bf9039a38e48f3da1267ecb9e1dbf09fe41"   # sha256(json.dumps(PLAN)), registered with this harness
CANDIDATES = (("I5", "I5", "OFF"), ("I6", "I6", "OFF"), ("TEL", "TEL", "OFF"), ("I4_increment", "I46", "I6"))
REFERENCE = (("I46_total", "I46", "OFF"),)
METRICS = {"lambda_thr": ("hi", "achieved", -1), "itl_p95": ("lo", "itl_p95_ms", +1)}   # (cell, key, degrading sign of ln ratio)
MIN_N = 3
EXPORT_MB_DEFAULT = 40.0
_T975 = None


def order_sha():
    return hashlib.sha256(json.dumps(PLAN).encode()).hexdigest()


def balanced_ok(plan=PLAN):
    """Latin rectangle: each level once per block, never twice at the same position, <= 2 occurrences of any ordered neighbour pair
    (with 4 blocks of 5 levels exactly one pair repeats; see the module docstring)."""
    n = len(LEVELS)
    if any(sorted(r) != sorted(LEVELS) for r in plan):
        return False
    if any(len({plan[b][p] for b in range(len(plan))}) != len(plan) for p in range(n)):
        return False
    pairs = [(r[i], r[i + 1]) for r in plan for i in range(n - 1)]
    return max(pairs.count(x) for x in pairs) <= 2 and len(set(pairs)) >= len(pairs) - 1


def sequence():
    """Registered execution order: [(k, block, position, level, seed)], k = 1..20."""
    return [(b * len(LEVELS) + p + 1, b + 1, p + 1, lvl, SEEDS[b]) for b, row in enumerate(PLAN) for p, lvl in enumerate(row)]


def plan_lines():
    return [f"decomp   : 5 levels {{OFF, I5, I6, I46, TEL}} x 4 blocks = {len(sequence())} instances of ~{D.EST['inst']} s (est ~{len(sequence()) * D.EST['inst'] / 3600:.1f} h),"
            f" agnostic arm, shape A, seeds {SEEDS}",
            "order    : Latin rectangle (position-balanced), registered sha " + ORDER_SHA[:12] + "..; " +
            " | ".join(" ".join(r) for r in PLAN),
            "export   : summaries only (< %.0f MB); raw (requests, probe jsonl, server.log) stays on the cluster volume "
            "/data/runs/<job>/raw" % EXPORT_MB_DEFAULT,
            "ALL      : not run (6 levels x 4 x 650 s = 260 min > the 250 min budget of a 270 min cap incl. overhead)",
            "levels   : I-4 alone is not separable without an engine change (I-6 rides the same hook); I4_increment = I46 vs I6"]


# ---------------------------------------------------------------- level assertions
def level_check(level):
    """Instrumentation check for D._boot: exactly the instruments of `level` are on, nothing else."""
    ex = EXPECT[level]

    def chk(rec):
        why = []
        f = rec.get("files") or {}
        if bool(rec.get("log_decode_enabled")) != ex["decode"]:
            why.append(f"{level}: 'decode probe ENABLED' log line {'missing' if ex['decode'] else 'present'}")
        if bool(rec.get("log_admission_enabled")) != ex["admission"]:
            why.append(f"{level}: 'admission probe ENABLED' log line {'missing' if ex['admission'] else 'present'}")
        if ex["decode"] and rec.get("log_decode_events") is not ex["events"]:
            why.append(f"{level}: decode probe I-4 events={rec.get('log_decode_events')} in the log, expected {ex['events']}")
        if bool(f.get("decode_probe")) != ex["decode"]:
            why.append(f"{level}: decode_probe file {'missing' if ex['decode'] else 'present'}")
        if bool(f.get("admission")) != ex["admission"]:
            why.append(f"{level}: admission file {'missing' if ex['admission'] else 'present'}")
        if f.get("telemetry") and not ex["tel"]:
            why.append(f"{level}: telemetry file present (PDMUX_TELEMETRY_PATH must stay unset)")
        sf, tree = rec.get("sglang_file"), rec.get("tree")
        if not sf or not tree or not str(sf).startswith(str(tree).rstrip("/") + "/"):
            why.append(f"sglang imported from {sf!r}, not under the installed tree {tree!r}")
        return {"verdict": D.FAIL if why else D.PASS, "why": why}
    return chk


def post_level_check(bdir, level):
    """After the instance: only this level's probe files exist, I-4 field nullness matches, rows exist."""
    ex, why, bdir = EXPECT[level], [], Path(bdir)
    has = {"decode_probe": (bdir / "decode_probe.jsonl").exists(), "admission": (bdir / "admission.jsonl").exists(),
           "telemetry": (bdir / "tel.jsonl").exists()}
    if has["decode_probe"] != ex["decode"] or has["admission"] != ex["admission"] or has["telemetry"] != ex["tel"]:
        why.append(f"files after run {has} do not match level {level}")
    facts = dict(files=has)
    if ex["decode"] and has["decode_probe"]:
        rows = [r for r in D.read_jsonl(bdir / "decode_probe.jsonl") if r.get("event") == "decode_forward"]
        nonnull = sum(1 for r in rows if r.get("decode_fwd_gpu_ms") is not None)
        facts.update(n_decode_forward=len(rows), n_gpu_ms_nonnull=nonnull)
        if not rows:
            why.append("decode probe wrote no decode_forward row")
        if ex["events"] and rows and nonnull == 0:
            why.append("I46: no decode_fwd_gpu_ms resolved (CUDA events not active)")
        if (not ex["events"]) and nonnull:
            why.append(f"I6: {nonnull} rows carry decode_fwd_gpu_ms although GPU_EVENTS=0")
    if ex["tel"] and has["telemetry"]:
        ph = [r for r in D.read_jsonl(bdir / "tel.jsonl") if r.get("event") in ("decode_iteration", "prefill_span_start", "prefill_span_end")]
        facts["n_phase_events"] = len(ph)
        if not ph:
            why.append("TEL: telemetry file has no phase event (PDMUX_PHASE_EVENTS not active)")
        if "PDMUX_PHASE_EVENTS disabled after failure" in D.read_text(bdir / "server.log"):
            why.append("TEL: server log says phase events disabled after failure")
        boot = D.load(bdir / "boot.json", {}) or {}
        if "PDMUX_PHASE_EVENTS" not in (boot.get("pdmux_knobs") or []) or "PDMUX_TELEMETRY_PATH" not in (boot.get("pdmux_knobs") or []):
            why.append("TEL: boot.json does not list PDMUX_TELEMETRY_PATH + PDMUX_PHASE_EVENTS")
    if ex["admission"] and has["admission"]:
        wins = [r for r in D.read_jsonl(bdir / "admission.jsonl") if r.get("event") == "admission_window"]
        facts["n_admission_windows"] = len(wins)
        if not wins:
            why.append("I5: no admission_window row")
    return {"ok": not why, "why": why, "facts": facts, "level": level}


# ---------------------------------------------------------------- run
def _raw_root(gate):
    job = gate.meta["job"]
    return Path(os.environ.get("PDMUX_DECOMP_RAW") or (Path(os.environ.get("PDMUX_DATA", "/data")) / "runs" / job / "raw"))


SMALL = ("boot.json", "instrumentation.json", "teardown.json", "instance.json", "level_assert.json")


def _under(child, parent):
    c, p = Path(child).resolve(), Path(parent).resolve()
    return c == p or p in c.parents


def _excerpt(log: Path, out: Path):
    lines = D.read_text(log).splitlines()
    keep = [ln for ln in lines if "ENABLED" in ln or "max_total_num_tokens" in ln][:20] + ["--- tail ---"] + lines[-40:]
    out.write_text("\n".join(keep)[:20000] + "\n")


def run_instance(gate, raw, k, block, pos, level, seed, ctx):
    bid = f"b{block}p{pos}_{level}"
    if (Path(raw) / bid).exists() and any((Path(raw) / bid).iterdir()):
        raise D.GateAbort("ABORT_CONFIG", f"raw dir {Path(raw) / bid} is not empty (append-mode probe files would mix runs)")
    spec = dict(arm="agnostic", on=(level != "OFF"), ctx=None, level=level, level_env=LEVEL_ENV[level], ins_check=level_check(level))
    inst = {"k": k, "block": block, "position": pos, "level": level, "seed": seed, "arm": "agnostic"}

    def warm(srv):
        bad = 0
        for _ in range(8):
            r = D.gen_ids(srv.port, [1] * D.SHAPE_A[0], 64, timeout=D.TMO["warm"])
            bad += 0 if r["ok"] else 1
        return (D.PASS if not bad else D.FAIL), f"{8 - bad}/8 warm-up requests ok", {}

    def mk(rung):
        def fn(srv):
            inst[rung.lower()] = D.run_cell(srv, srv.dir / f"cell_{rung}", rung, seed)
            return D.PASS, f"{rung} cell done rc={inst[rung.lower()]['rc']}", {}
        return fn
    steps = [("warm", warm, 60, D.TMO["warm"]), ("hi", mk("HI"), 250, D.TMO["cell"] + 30), ("lo", mk("LO"), 250, D.TMO["cell"] + 30)]
    srv, _td, ok = D.boot_and_run(gate, bid, spec, "patched", steps, ctx)
    inst["abnormal_exit"] = any(D.M.alive_after_step(srv.dir, st) is False for st in ("warm", "hi", "lo"))
    inst["released"] = ok
    inst["valid"], inst["invalid_reasons"] = D.instance_valid(inst)
    boot_st = gate.stages.get(f"{bid}_boot", {})
    boot_level_fail = boot_st.get("status") == D.FAIL and "instrumentation" in (boot_st.get("facts") or {})
    post = post_level_check(srv.dir, level) if not boot_level_fail else {"ok": False, "why": ["boot level assertion failed"], "level": level}
    inst["level_ok"] = bool(post["ok"]) and not boot_level_fail
    if not inst["level_ok"]:
        inst["valid"] = False
        inst["invalid_reasons"] = list(inst["invalid_reasons"]) + ["level assertion failed: " + "; ".join(post["why"])]
    # export tree gets summaries only; the raw dir keeps everything (and an index with sizes + sha256)
    out = gate.root / bid
    out.mkdir(parents=True, exist_ok=True)
    D.dump(inst, srv.dir / "instance.json")
    D.dump(post, srv.dir / "level_assert.json")
    for nm in SMALL:
        if (srv.dir / nm).exists():
            shutil.copy2(srv.dir / nm, out / nm)
    for rung in ("HI", "LO"):
        if (srv.dir / f"cell_{rung}" / "cell_metrics.json").exists():
            (out / f"cell_{rung}").mkdir(exist_ok=True)
            shutil.copy2(srv.dir / f"cell_{rung}" / "cell_metrics.json", out / f"cell_{rung}" / "cell_metrics.json")
    _excerpt(srv.log, out / "server_excerpt.txt")
    D.dump({"raw_dir": str(srv.dir), "files": [{"path": str(f.relative_to(srv.dir)), "bytes": f.stat().st_size,
                                                 "sha256": D.sha256_file(f)} for f in sorted(srv.dir.rglob("*")) if f.is_file()]},
           out / "raw_index.json")
    if not inst["level_ok"]:
        raise D.GateAbort("ABORT_LEVEL", f"{bid}: " + "; ".join(post["why"]))
    return inst, ok


def export_check(gate, raw):
    """The exported tree (gate.root) must hold summaries only.  Over the limit => the biggest files are moved to the raw dir
    (so the export stays safe) AND the check FAILs (something wrote bulk data into the export tree).  Writes EXPORT_MANIFEST.json."""
    limit = float(os.environ.get("PDMUX_DECOMP_MAX_EXPORT_MB", EXPORT_MB_DEFAULT))

    def files():
        return [f for f in gate.root.rglob("*") if f.is_file() and "cache" not in f.relative_to(gate.root).parts]
    total = sum(f.stat().st_size for f in files()) / 1e6
    spilled = []
    if total > limit:
        spill = Path(raw) / "spilled_from_export"
        spill.mkdir(parents=True, exist_ok=True)
        for f in sorted(files(), key=lambda p: -p.stat().st_size):
            if total <= limit:
                break
            total -= f.stat().st_size / 1e6
            spilled.append(str(f.relative_to(gate.root)))
            shutil.move(str(f), str(spill / str(f.relative_to(gate.root)).replace("/", "__")))
    skip = {"GATE_RESULT.json", "SUMMARY.md", "EXPORT_MANIFEST.json"}
    man = [{"path": str(f.relative_to(gate.root)), "bytes": f.stat().st_size, "sha256": D.sha256_file(f)}
           for f in sorted(files()) if f.name not in skip]
    D.dump({"note": "sizes/sha256 of every exported summary file at write time (GATE_RESULT.json, SUMMARY.md rewritten afterwards); "
                    "compare with the fetched tree to detect export truncation", "files": man}, gate.root / "EXPORT_MANIFEST.json")
    return {"verdict": D.PASS if not spilled else D.FAIL, "export_mb": round(total, 2), "limit_mb": limit, "spilled": spilled,
            "n_files": len(man), "raw_dir": str(raw)}


def run_decomp(gate):
    if order_sha() != ORDER_SHA or not balanced_ok():
        raise D.GateAbort("ABORT_REG", f"decomp order sha {order_sha()} != registered {ORDER_SHA} (or plan not position-balanced)")
    raw = _raw_root(gate)
    if _under(raw, gate.root):
        raise D.GateAbort("ABORT_CONFIG", f"raw dir {raw} lies inside the exported tree {gate.root}")
    raw.mkdir(parents=True, exist_ok=True)
    gate.raw_root = raw
    gate.meta["raw_root"] = str(raw)
    D.dump({"levels": LEVEL_ENV, "plan": PLAN, "seeds": SEEDS, "order_sha": ORDER_SHA, "tel_and_phase_events": "TEL level only"}, gate.root / "design.json")
    ctx = D.Ctx()
    gate.check_digest("start", "applied")
    seq = sequence()
    for k, block, pos, level, seed in seq:
        if k == len(seq) // 2 + 1:
            gate.check_digest("middle", "applied")
        if not gate.fits(D.EST["inst"]):
            gate.record(f"inst{k:02d}_{level}", D.SKIPPED, "budget: instance not started")
            continue
        _inst, ok = run_instance(gate, raw, k, block, pos, level, seed, ctx)
        if not ok:
            gate.record(f"inst{k:02d}_halt", D.UNRESOLVED, "GPU memory not released: remaining instances skipped (HARNESS_UNRESOLVED)")
            break
    if not any(d["tag"] == "middle" for d in gate.digests):
        gate.check_digest("middle", "applied")
    gate.check_digest("end", "applied")
    chk = score_decomp(gate.root)
    chk["export_size"] = export_check(gate, raw)
    D.dump({"checks": chk["decomp"], "raw_root": str(raw)}, gate.root / "DECOMP_RESULT.json")
    return chk


# ---------------------------------------------------------------- scoring
def _t(df):
    from engine_gate_checks import _T975 as T
    return T[df - 1] if df - 1 < len(T) else 1.96


def log_ratio_stats(pairs):
    d = [math.log(a / b) for a, b in pairs if a and b and a > 0 and b > 0]
    n = len(d)
    if n < MIN_N:
        return {"n": n, "status": "INSUFFICIENT"}
    m, sd = statistics.mean(d), statistics.stdev(d)
    hw = _t(n - 1) * sd / math.sqrt(n)
    return {"n": n, "status": "OK", "mean_log": m, "sd_log": sd, "ci95_log": [m - hw, m + hw], "pct": math.expm1(m) * 100,
            "ci95_pct": [math.expm1(m - hw) * 100, math.expm1(m + hw) * 100], "sd_pct": sd * 100}


def _flag(st, sign):
    if st["status"] != "OK":
        return "INSUFFICIENT"
    lo, hi = st["ci95_log"]
    if (sign < 0 and hi < 0) or (sign > 0 and lo > 0):
        return "FLAGGED"
    if (sign < 0 and lo > 0) or (sign > 0 and hi < 0):
        return "OPPOSITE"
    return "CI_CONTAINS_0"


def collect(root):
    insts = {}
    for f in sorted(Path(root).glob("b*p*_*/instance.json")):
        i = D.load(f)
        if i:
            insts[(i["block"], i["level"])] = i
    return insts


def decomp_stats(insts):
    res, flagged = OrderedDict(), {}
    for mname, (cell, key, sign) in METRICS.items():
        res[mname] = {}
        for cname, a, b in CANDIDATES + REFERENCE:
            pairs = []
            for blk in range(1, len(PLAN) + 1):
                x, y = insts.get((blk, a)), insts.get((blk, b))
                if x and y and x.get("valid") and y.get("valid"):
                    pairs.append((x[cell][key], y[cell][key]))
            st = log_ratio_stats(pairs)
            st["flag"] = _flag(st, sign)
            st["contrast"] = f"{a} vs {b}"
            st["candidate"] = any(c[0] == cname for c in CANDIDATES)
            res[mname][cname] = st
        fl = [(c, res[mname][c]) for c, _a, _b in CANDIDATES if res[mname][c]["flag"] == "FLAGGED"]
        fl.sort(key=lambda kv: -sign * kv[1]["mean_log"])
        flagged[mname] = {"flagged": [c for c, _ in fl], "largest": fl[0][0] if fl else "NONE_RESOLVED"}
    # position drift (complete blocks only): mean ln(x / block geometric mean) per position
    drift = {}
    for mname, (cell, key, _s) in METRICS.items():
        acc = {p: [] for p in range(1, len(LEVELS) + 1)}
        for blk in range(1, len(PLAN) + 1):
            row = [insts.get((blk, lv)) for lv in PLAN[blk - 1]]
            if all(r and r.get("valid") for r in row):
                logs = [math.log(r[cell][key]) for r in row]
                g = statistics.mean(logs)
                for p, v in enumerate(logs, 1):
                    acc[p].append(v - g)
        drift[mname] = {str(p): (statistics.mean(v) if v else None) for p, v in acc.items()}
    return {"verdict": None, "metrics": res, "selection": flagged, "position_drift_log": drift,
            "rule": "FLAGGED = CI95 excludes 0 in the degrading direction; every flagged candidate is reported; largest = largest "
                    "degrading point estimate among them; none -> NONE_RESOLVED (n<=4 blocks: low power, not evidence of no effect)"}


def check_order(insts_all):
    """Executed sequence (by k) must equal the registered sequence (a prefix is allowed: budget stop)."""
    reg = {k: (b, p, lvl, seed) for k, b, p, lvl, seed in sequence()}
    ex = sorted((i["k"], (i["block"], i["position"], i["level"], i["seed"])) for i in insts_all)
    bad = [k for k, v in ex if reg.get(k) != v]
    ks = [k for k, _ in ex]
    gaps_ok = ks == sorted(ks) and len(set(ks)) == len(ks)
    return {"verdict": D.PASS if not bad and gaps_ok else D.FAIL, "n_executed": len(ex), "violations": bad, "unique_sorted": gaps_ok,
            "order_sha": order_sha(), "registered_sha": ORDER_SHA}


def score_decomp(root):
    root = Path(root)
    insts = collect(root)
    chk = OrderedDict()
    st = decomp_stats(insts)
    n_exec, n_valid = len(insts), {lv: sum(1 for (b, l), i in insts.items() if l == lv and i.get("valid")) for lv in LEVELS}
    complete = n_exec == len(sequence()) and all(v >= MIN_N for v in n_valid.values())
    st["verdict"] = "DIAGNOSTIC_COMPLETE" if complete else "DIAGNOSTIC_INCOMPLETE"
    st.update(n_executed=n_exec, n_valid_by_level=n_valid)
    chk["decomp"] = {"verdict": st["verdict"], **st}
    chk["order"] = check_order(list(insts.values()))
    bad = sorted(f"b{b}_{l}" for (b, l), i in insts.items() if not i.get("level_ok"))
    chk["level_assertions"] = {"verdict": D.PASS if not bad else D.FAIL, "bad": bad}
    chk["digest_3x"] = {"verdict": D.PASS if D._digests_ok(root, 3) else D.FAIL}
    if "export_size" not in chk:
        mb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and "cache" not in f.relative_to(root).parts) / 1e6
        lim = float(os.environ.get("PDMUX_DECOMP_MAX_EXPORT_MB", EXPORT_MB_DEFAULT))
        chk["export_size"] = {"verdict": D.PASS if mb <= lim else D.FAIL, "export_mb": round(mb, 2), "limit_mb": lim}
    return chk
