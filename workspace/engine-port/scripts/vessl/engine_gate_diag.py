#!/usr/bin/env python3
"""GPU correctness gate for PDMUX_PHASE_DIAG (diag Job instrumentation; PREREG_DIAG_stall_drain P-1 / P-3).

STATUS: scorers CPU-tested on synthetic records only (tests/test_engine_gate_diag.py).  THE DRIVER HAS NOT RUN ON A GPU
and has not been run against a fake server.  Preparing this file is not a PASS of anything; no performance claim.
Submission follows the VESSL operating model (launch.sh dry-run -> user approval -> --submit); this file does not submit.

  engine_gate_diag.py run                 inside a Job (entry: engine_gate_diag.sh)
  engine_gate_diag.py score DIR           re-score an exported result directory (CPU)

What the gate asks (P-3: "ON/OFF greedy output identity, no new synchronisation, CUDA event pair coherence"):
  D0  ENGINE_DIGEST of the installed tree == DIGEST_DIAG (the tree this instrumentation commit builds) -- else ABORT.
  D1  greedy identity: the SEQUENTIAL (one request in flight) outputs of boot ON == boot OFF, token for token.
      Concurrent outputs are recorded, not gated (batch composition is timing dependent in both arms).
  D2  OFF is really OFF: no telemetry / decode-probe file, no diag log line.  ON is really ON: both files exist.
  D3  record coherence of boot ON (`check_diag_records`): every prefill_chunk has a finite positive chunk_gpu_ms,
      t_issue_start <= t_issue_end, split_index + forward_count <= num_layers, a matching prefill_span_start; the chunks
      of every completed span tile [0, num_layers) contiguously; sum(chunk_gpu_ms) of a span <= its host span length
      (start record -> end record) + EPS_MS (GPU time cannot exceed the wall time that contains it); every
      prefill_span_start carries rids with len == n_new; drains have non-negative separate sync times and a known reason;
      realised (prefill_sms, decode_sms) is a configured division; no "PDMUX_PHASE_DIAG disabled" line in the log.
  D4  I-4 in the same boot: decode_probe decode_forward records with non-null decode_fwd_gpu_ms exist.
  D5  clocks: phase_diag_boot perf_counter is CLOCK_MONOTONIC (|t_monotonic - t_perf| < 1 ms and the implementation
      string names CLOCK_MONOTONIC); client (benchmarks/pdmux_eval/abs_mono_loadgen.py) stamps are causally ordered against
      the server records of the same rid: t_sent_mono <= span_start, span_start <= first token.
  D6  slo_switch drains (boot ON + PDMUX_SLO_SCHED=1, library defaults): >= 1 drain with reason slo_switch and
      from != to.  No switch under this load -> UNRESOLVED.  Two roll-ups (audit r1 M7c): `verdict_for_D1`
      tolerates D6 UNRESOLVED (D-1 runs no controller); `verdict_for_D3` requires D3_records_slo AND D6 = PASS.
  D7  no blocking call from the instrument on the GPU (audit r1 M7a): boot ON runs with PDMUX_PHASE_DIAG_SYNC_AUDIT=1,
      which wraps torch.cuda.Stream/Event.synchronize, torch.cuda.synchronize and Tensor.item process-wide and counts
      calls with a diag hook on the stack.  PASS iff every diag_* count == 0 AND total_stream_synchronize > 0 (the
      wrappers were live).  This boot is a gate boot, not a measurement boot (the wrappers walk the stack).
  D8  PHASE_EVENTS ON + DIAG OFF (audit r1 M8): phase events present, no diag event, no `rids`.
  D9  static NON-sticky arm (user decision 49; PDMUX_R2_POLICY=fixed, PDMUX_R2_FIXED_DSM=STATIC_D, sticky unset):
      decode steps co-resident with a prefill chunk run on decode_sms == STATIC_D, decode-only steps on 108, chunks
      without decode on prefill_sms == 108; all three states must be seen (missing state -> UNRESOLVED).
  D1 covers SEQUENTIAL requests only; concurrent requests are recorded (conc_requests.jsonl), never compared.
NOT in this gate: the observer cost (PREREG D-1 OFF arm measures it, paired); "no new synchronisation" on the GPU is
argued from code + CPU tests (fakes that raise on synchronize/wait, source scan), not measured here.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[1]

PASS, FAIL, UNRESOLVED = "PASS", "FAIL", "UNRESOLVED"
# [봉인 시 기입] ENGINE_DIGEST of a tree built from the instrumentation commit (sync -> manual edits -> sync).
DIGEST_DIAG = os.environ.get("PDMUX_DIAG_GATE_DIGEST", "")
EPS_MS = 2.0                        # event-record / host-stamp granularity allowance for the GPU <= wall check
CLOCK_EPS_S = 1e-3
DRAIN_REASONS = ("slo_switch", "adjust")
SEQ_LENGTHS = (512, 2048, 8192)     # sequential greedy identity inputs (synthetic ids, H.synth_ids)
SEQ_REPS = 2
SEQ_MAX_NEW = 64
CONC_N, CONC_LENGTHS, CONC_MAX_NEW, CONC_RATE = 24, (2048, 8192), 128, 2.0
STATIC_D = 34                       # D9: a division of pdmux_agnostic_manual.yml ([74, 34], [54, 54])


# ---------------------------------------------------------------- pure scorers (CPU-tested)
def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"event": "__unparseable__"})
    return out


def _finite_pos(x):
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def check_identity(off, on):
    """off/on: {key: [token ids]}.  PASS iff same keys, all non-empty, all equal."""
    why = []
    if set(off) != set(on):
        why.append(f"key sets differ: only OFF {sorted(set(off) - set(on))[:5]}, only ON {sorted(set(on) - set(off))[:5]}")
    empty = sorted(k for k in set(off) | set(on) if not off.get(k) or not on.get(k))
    if empty:
        why.append(f"empty outputs: {empty[:5]}")
    diff = sorted(k for k in set(off) & set(on) if off[k] and on[k] and list(off[k]) != list(on[k]))
    if diff:
        why.append(f"{len(diff)} outputs differ: {diff[:5]}")
    return {"verdict": PASS if not why and off else FAIL, "why": why, "n": len(off), "n_diff": len(diff)}


def check_diag_records(recs, sm_counts, log_text=""):
    """D3 on one ON boot's telemetry records.  -> {verdict, why, facts}."""
    why = []
    by = defaultdict(list)
    for r in recs:
        by[r.get("event")].append(r)
    boots = by["phase_diag_boot"]
    if len(boots) != 1:
        why.append(f"{len(boots)} phase_diag_boot records (expected 1)")
    nl = int(boots[0]["num_layers"]) if boots else None
    starts = {r["span_seq"]: r for r in by["prefill_span_start"]}
    ends = {r["span_seq"]: r for r in by["prefill_span_end"]}
    chunks = by["prefill_chunk"]
    drains = by["drain"]
    if not chunks:
        why.append("no prefill_chunk record")
    if not drains:
        why.append("no drain record")
    for s in starts.values():
        if not isinstance(s.get("rids"), list) or len(s["rids"]) != s.get("n_new"):
            why.append(f"span {s.get('span_seq')}: rids missing or len != n_new")
            break
    allowed = {tuple(x) for x in sm_counts}
    bad_chunk = []
    per_span = defaultdict(list)
    for c in chunks:
        ok = (_finite_pos(c.get("chunk_gpu_ms")) and c.get("t_issue_start", 1) <= c.get("t_issue_end", 0)
              and nl is not None and 0 <= c.get("split_index", -1)
              and c.get("split_index", 0) + c.get("forward_count", 0) <= nl and c.get("forward_count", 0) >= 1
              and c.get("span_seq") in starts
              and (c.get("prefill_sms"), c.get("decode_sms")) in allowed)
        if not ok:
            bad_chunk.append(c.get("chunk_seq"))
        per_span[c.get("span_seq")].append(c)
    if bad_chunk:
        why.append(f"{len(bad_chunk)} incoherent prefill_chunk records, e.g. chunk_seq {bad_chunk[:5]}")
    over, gaps = [], []
    for seq, end in ends.items():
        cs = sorted(per_span.get(seq, []), key=lambda c: c["split_index"])
        if not cs or seq not in starts:
            gaps.append(seq)
            continue
        cover = 0
        for c in cs:
            if c["split_index"] != cover:
                break
            cover += c["forward_count"]
        if nl is not None and cover != nl:
            gaps.append(seq)
        wall_ms = (end["timestamp_monotonic_s"] - starts[seq]["timestamp_monotonic_s"]) * 1000.0
        gpu_ms = sum(c["chunk_gpu_ms"] for c in cs if _finite_pos(c.get("chunk_gpu_ms")))
        if gpu_ms > wall_ms + EPS_MS:
            over.append((seq, round(gpu_ms, 3), round(wall_ms, 3)))
    if gaps:
        why.append(f"{len(gaps)} completed spans whose chunks do not tile [0, num_layers): {gaps[:5]}")
    if over:
        why.append(f"{len(over)} spans with sum(chunk_gpu_ms) > host span + {EPS_MS} ms: {over[:3]}")
    bad_drain = [d.get("drain_seq") for d in drains
                 if d.get("reason") not in DRAIN_REASONS
                 or not (isinstance(d.get("prefill_sync_ms"), (int, float)) and d["prefill_sync_ms"] >= 0)
                 or not (isinstance(d.get("decode_sync_ms"), (int, float)) and d["decode_sync_ms"] >= 0)
                 or (d.get("prefill_sms_to"), d.get("decode_sms_to")) not in allowed
                 or (d.get("split_present") and not (0 <= (d.get("split_index") or 0) <= (nl or 0)))]
    if bad_drain:
        why.append(f"{len(bad_drain)} incoherent drain records, e.g. {bad_drain[:5]}")
    if "PDMUX_PHASE_DIAG disabled" in log_text:
        why.append("the diag disabled itself during the boot (log)")
    facts = {"n_chunks": len(chunks), "n_drains": len(drains), "n_spans_started": len(starts),
             "n_spans_ended": len(ends), "n_slo_switch": sum(1 for d in drains if d.get("reason") == "slo_switch"),
             "n_adjust": sum(1 for d in drains if d.get("reason") == "adjust"), "num_layers": nl}
    return {"verdict": FAIL if why else PASS, "why": why, "facts": facts}


def check_decode_probe(recs, log_text=""):
    fw = [r for r in recs if r.get("event") == "decode_forward"]
    with_gpu = [r for r in fw if _finite_pos(r.get("decode_fwd_gpu_ms"))]
    why = []
    if not with_gpu:
        why.append(f"no decode_forward with decode_fwd_gpu_ms ({len(fw)} decode_forward records)")
    if "decode probe disabled" in log_text:
        why.append("decode probe disabled itself (log)")
    return {"verdict": FAIL if why else PASS, "why": why, "facts": {"n_decode_forward": len(fw), "n_gpu": len(with_gpu)}}


def check_clocks(recs, client_recs):
    """D5.  client_recs: abs_mono_loadgen request records (trace_loadgen records + *_mono stamps; rid = request_id)."""
    why = []
    boots = [r for r in recs if r.get("event") == "phase_diag_boot"]
    if not boots:
        return {"verdict": FAIL, "why": ["no phase_diag_boot"], "facts": {}}
    b = boots[0]
    if "CLOCK_MONOTONIC" not in str(b.get("perf_counter_impl")):
        why.append(f"perf_counter implementation {b.get('perf_counter_impl')!r}")
    if not abs(b["t_monotonic"] - b["t_perf"]) < CLOCK_EPS_S:
        why.append(f"|t_monotonic - t_perf| = {abs(b['t_monotonic'] - b['t_perf']):.6f} s")
    span_of = {}
    for r in recs:
        if r.get("event") == "prefill_span_start":
            for rid in r.get("rids") or []:
                span_of.setdefault(str(rid), []).append(r["timestamp_monotonic_s"])
    span_seq_of = {}
    for r in recs:
        if r.get("event") == "prefill_span_start":
            for rid in r.get("rids") or []:
                span_seq_of[str(rid)] = r.get("span_seq")
    end_t = {r.get("span_seq"): r["timestamp_monotonic_s"] for r in recs if r.get("event") == "prefill_span_end"}
    lags = []
    n_ok, bad, missing = 0, [], []
    for c in client_recs:
        rid = str(c.get("request_id"))
        if c.get("error") or not c.get("token_t_mono"):
            continue
        ts = span_of.get(rid)
        if not ts or len(ts) != 1:
            missing.append(rid)
            continue
        if not (c["t_sent_mono"] <= ts[0] <= c["token_t_mono"][0]):
            bad.append(rid)
        else:
            n_ok += 1
        te = end_t.get(span_seq_of.get(rid))
        if te is not None:
            lags.append((c["token_t_mono"][0] - te) * 1000.0)   # audit r1 M3: client-server lag, report only
    if missing:
        why.append(f"{len(missing)} client rids not in exactly one prefill_span_start.rids: {missing[:5]}")
    if bad:
        why.append(f"{len(bad)} rids violate t_sent_mono <= span_start <= first token: {bad[:5]}")
    if n_ok == 0:
        why.append("no successful client request to pair")
    lags.sort()
    q = (lambda f: lags[min(len(lags) - 1, int(f * len(lags)))] if lags else None)
    return {"verdict": FAIL if why else PASS, "why": why,
            "facts": {"n_paired": n_ok, "first_token_lag_after_span_end_ms": {"n": len(lags), "p50": q(0.5),
                                                                              "p99": q(0.99)}}}


def check_sync_audit(recs):
    a = [r for r in recs if r.get("event") == "phase_diag_sync_audit"]
    if not a:
        return {"verdict": FAIL, "why": ["no phase_diag_sync_audit record (audit not live)"], "facts": {}}
    last = a[-1]
    diag = {k: v for k, v in last.items() if k.startswith("diag_")}
    why = [f"{k} = {v}" for k, v in diag.items() if v]
    if not last.get("total_stream_synchronize"):
        why.append("total_stream_synchronize == 0: wrappers not live")
    return {"verdict": FAIL if why else PASS, "why": why, "facts": {k: v for k, v in last.items() if k != "event"}}


def check_phase_only(recs):
    ev = {r.get("event") for r in recs}
    why = []
    if "decode_iteration" not in ev or "prefill_span_start" not in ev:
        why.append("phase events missing")
    leaked = sorted(ev & {"prefill_chunk", "drain", "phase_diag_boot", "phase_diag_close", "phase_diag_sync_audit"})
    if leaked:
        why.append(f"diag events with DIAG OFF: {leaked}")
    if any("rids" in r for r in recs if r.get("event") == "prefill_span_start"):
        why.append("rids present with DIAG OFF")
    return {"verdict": FAIL if why else PASS, "why": why}


def check_static_nonsticky(recs, static_d, total_sm=108):
    dec = [r for r in recs if r.get("event") == "decode_iteration"]
    co = [r for r in dec if r.get("prefill_in_flight") and not r.get("prefill_kernel_done_wait")]
    solo = [r for r in dec if not r.get("prefill_in_flight")]
    pchunk = [r for r in recs if r.get("event") == "prefill_chunk" and not r.get("decode_launched")]
    why, missing = [], []
    for name, xs in (("co-resident decode", co), ("decode-only", solo), ("prefill-only chunk", pchunk)):
        if not xs:
            missing.append(name)
    bad_co = sum(1 for r in co if r.get("decode_sms") != static_d)
    bad_solo = sum(1 for r in solo if r.get("decode_sms") != total_sm)
    bad_p = sum(1 for r in pchunk if r.get("prefill_sms") != total_sm)
    if bad_co:
        why.append(f"{bad_co}/{len(co)} co-resident decode steps not on D{static_d}")
    if bad_solo:
        why.append(f"{bad_solo}/{len(solo)} decode-only steps not on {total_sm} SM")
    if bad_p:
        why.append(f"{bad_p}/{len(pchunk)} prefill-only chunks not on {total_sm} SM")
    facts = {"n_co": len(co), "n_decode_only": len(solo), "n_prefill_only_chunks": len(pchunk),
             "n_drain_adjust": sum(1 for r in recs if r.get("event") == "drain" and r.get("reason") == "adjust"),
             "n_drain_slo_switch": sum(1 for r in recs if r.get("event") == "drain" and r.get("reason") == "slo_switch"),
             "n_spans": sum(1 for r in recs if r.get("event") == "prefill_span_start")}
    if why:
        return {"verdict": FAIL, "why": why, "facts": facts}
    if missing:
        return {"verdict": UNRESOLVED, "why": [f"state not observed: {missing}"], "facts": facts}
    return {"verdict": PASS, "why": [], "facts": facts}


def check_slo_switch(recs):
    d = [r for r in recs if r.get("event") == "drain" and r.get("reason") == "slo_switch"]
    if not d:
        return {"verdict": UNRESOLVED, "why": ["no slo_switch drain under this load (path CPU-tested only)"], "facts": {}}
    bad = [x.get("drain_seq") for x in d if x.get("stream_idx_from") == x.get("stream_idx_to")]
    return {"verdict": FAIL if bad else PASS, "why": [f"slo_switch with from == to: {bad[:5]}"] if bad else [],
            "facts": {"n_slo_switch": len(d)}}


def rollup(verdicts):
    if any(v == FAIL for v in verdicts):
        return FAIL
    if verdicts and all(v == PASS for v in verdicts):
        return PASS
    return UNRESOLVED


def score_dir(root):
    """Re-score an exported result directory (layout written by `run`)."""
    root = Path(root)
    cfg = json.loads((root / "plan.json").read_text())
    sm = cfg["sm_counts"]
    res = {}
    seq_off = json.loads((root / "off" / "seq.json").read_text()) if (root / "off" / "seq.json").is_file() else {}
    seq_on = json.loads((root / "on" / "seq.json").read_text()) if (root / "on" / "seq.json").is_file() else {}
    res["D1_identity"] = check_identity(seq_off, seq_on)
    off_files = ([n for n in ("tel.jsonl", "decode_probe.jsonl") if (root / "off" / n).exists()]
                 if (root / "off").is_dir() else ["<no off dir>"])
    on_ok = (root / "on" / "tel.jsonl").is_file() and (root / "on" / "decode_probe.jsonl").is_file()
    res["D2_levels"] = {"verdict": PASS if (not off_files and on_ok) else FAIL,
                        "why": ([f"OFF wrote {off_files}"] if off_files else []) + ([] if on_ok else ["ON files missing"])}
    tel = read_jsonl(root / "on" / "tel.jsonl")
    log_on = (root / "on" / "server.log").read_text(errors="replace") if (root / "on" / "server.log").is_file() else ""
    res["D3_records"] = check_diag_records(tel, sm, log_on)
    res["D4_decode_probe"] = check_decode_probe(read_jsonl(root / "on" / "decode_probe.jsonl"), log_on)
    res["D5_clocks"] = check_clocks(tel, read_jsonl(root / "on" / "conc_requests.jsonl"))
    tel_slo = read_jsonl(root / "on_slo" / "tel.jsonl")
    log_slo = (root / "on_slo" / "server.log").read_text(errors="replace") if (root / "on_slo" / "server.log").is_file() else ""
    res["D3_records_slo"] = check_diag_records(tel_slo, sm, log_slo) if tel_slo else \
        {"verdict": UNRESOLVED, "why": ["no on_slo telemetry"]}
    res["D6_slo_switch"] = check_slo_switch(tel_slo)
    res["D7_sync_audit"] = check_sync_audit(tel)
    res["D8_phase_only"] = check_phase_only(read_jsonl(root / "phase_only" / "tel.jsonl"))
    tel_st = read_jsonl(root / "static_ns" / "tel.jsonl")
    log_st = (root / "static_ns" / "server.log").read_text(errors="replace") if (root / "static_ns" / "server.log").is_file() else ""
    res["D3_records_static_ns"] = check_diag_records(tel_st, sm, log_st) if tel_st else \
        {"verdict": UNRESOLVED, "why": ["no static_ns telemetry"]}
    res["D9_static_nonsticky"] = check_static_nonsticky(tel_st, int(cfg.get("static_d", STATIC_D)))
    common = [v["verdict"] for k, v in res.items() if k not in ("D3_records_slo", "D6_slo_switch")]
    v_d1 = rollup(common + [v for v in (res["D3_records_slo"]["verdict"], res["D6_slo_switch"]["verdict"])
                            if v != UNRESOLVED])
    v_d3 = rollup(common + [res["D3_records_slo"]["verdict"], res["D6_slo_switch"]["verdict"]])
    verdict = v_d1          # exit code = the D-1 roll-up; D-3 may be submitted only when verdict_for_D3 == PASS
    out = {"verdict": verdict, "verdict_for_D1": v_d1, "verdict_for_D3": v_d3, "checks": res,
           "note": "harness result on this Job's data; not a performance claim; observer cost not measured here"}
    (root / "GATE_RESULT.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


# ---------------------------------------------------------------- Job driver (GPU; NOT RUN)
def run():
    import asyncio
    import engine_gate_driver as GD          # reuse: Server lifecycle, synthetic ids, model/digest helpers
    sys.path.insert(0, str(ENG / "benchmarks"))
    from pdmux_eval import abs_mono_loadgen as AM

    for k in ("PDMUX_PHASE_DIAG", "PDMUX_PHASE_DIAG_SYNC_AUDIT"):
        os.environ.pop(k, None)
    job = os.environ.get("PDMUX_JOB_NAME", "local")
    root = Path(os.environ.get("PDMUX_GATE_ROOT") or ENG / "results" / "engine_gate" / job)
    root.mkdir(parents=True, exist_ok=True)
    gate = GD.Gate("diag", root, int(os.environ.get("PDMUX_GATE_HARDCAP_MIN", "120")))
    GD.check_models(gate)
    GD.write_substrate(gate)
    got, err = GD.tree_digest(GD.tree_path())
    if not DIGEST_DIAG or got != DIGEST_DIAG:
        (root / "GATE_RESULT.json").write_text(json.dumps(
            {"verdict": "ABORT_ENGINE_DIGEST", "got": got, "expected": DIGEST_DIAG, "err": err}) + "\n")
        return 13
    import yaml
    cfg_path = ENG / "benchmarks" / "configs" / "pdmux_agnostic_manual.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    divs = [[108, 0]] + [[p, d] for p, d, _t in cfg["manual_divisions"]] + [[0, 108]]
    (root / "plan.json").write_text(json.dumps({"sm_counts": divs, "config": str(cfg_path), "seq_lengths": SEQ_LENGTHS,
                                                "static_d": STATIC_D,
                                                "conc": [CONC_N, CONC_LENGTHS, CONC_MAX_NEW, CONC_RATE]}) + "\n")
    ctx = GD.Ctx()
    on_env = {"PDMUX_TELEMETRY_PATH": "{dir}/tel.jsonl", "PDMUX_PHASE_EVENTS": "1", "PDMUX_PHASE_DIAG": "1",
              "PDMUX_DECODE_PROBE_PATH": "{dir}/decode_probe.jsonl"}
    boots = (("off", {}), ("on", dict(on_env, PDMUX_PHASE_DIAG_SYNC_AUDIT="1")),
             ("phase_only", {"PDMUX_TELEMETRY_PATH": "{dir}/tel.jsonl", "PDMUX_PHASE_EVENTS": "1"}),
             ("static_ns", dict(on_env, PDMUX_R2_POLICY="fixed", PDMUX_R2_FIXED_DSM=str(STATIC_D))),
             ("on_slo", dict(on_env, PDMUX_SLO_SCHED="1")))

    def seq_step(srv):
        out = {}
        for L in SEQ_LENGTHS:
            for rep in range(SEQ_REPS):
                ids = ctx.ids(GD.NANO, L, 9000 + 17 * L + rep)
                r = GD.gen_ids(srv.port, ids, SEQ_MAX_NEW, timeout=600)
                out[f"L{L}_r{rep}"] = r["out"] if r["ok"] else []
        (srv.dir / "seq.json").write_text(json.dumps(out) + "\n")
        bad = [k for k, v in out.items() if not v]
        return (PASS if not bad else FAIL), f"{len(out)} sequential requests, empty {bad}", {}

    def conc_step(srv):
        trace = []
        for i in range(CONC_N):
            L = CONC_LENGTHS[i % len(CONC_LENGTHS)]
            ids = ctx.ids(GD.NANO, L, 7000 + i)
            trace.append({"request_id": f"diag-{srv.id}-{i:03d}", "phase": "gate", "arrival_s": i / CONC_RATE,
                          "input_tokens": len(ids), "output_tokens": CONC_MAX_NEW, "input_ids": ids})
        summ = asyncio.run(AM.replay_abs_mono(f"http://127.0.0.1:{srv.port}/generate", trace,
                                              srv.dir / "conc_requests.jsonl", 1e9, 1e9, 0))
        ok = summ.get("failed_requests", 1) == 0
        return (PASS if ok else FAIL), f"{CONC_N} concurrent requests, failed {summ.get('failed_requests')}", {}

    for bid, env in boots:
        spec = {"arm": "agnostic", "on": bool(env), "level_env": env,
                "ins_check": lambda ins: {"verdict": PASS, "why": []}}
        steps = ([("seq", seq_step, 600, 1800)] if bid in ("off", "on") else []) + [("conc", conc_step, 300, 1200)]
        srv = GD.Server(gate, bid, spec, "applied")   # dir = root/<bid>, log = root/<bid>/server.log
        st = gate.stage(f"{bid}_boot", lambda srv=srv: GD._boot(srv), GD.EST["boot"], GD.TMO["boot"] + 30)
        try:
            if st == PASS:
                for name, fn, est, tmo in steps:
                    gate.stage(f"{bid}_{name}", lambda fn=fn, srv=srv: fn(srv), est, tmo)
        finally:
            srv.stop()
    out = score_dir(root)
    print(json.dumps({"verdict": out["verdict"]}))
    return {PASS: 0, FAIL: 1}.get(out["verdict"], 4)


def main(argv):
    if len(argv) >= 2 and argv[1] == "run":
        return run()
    if len(argv) == 3 and argv[1] == "score":
        print(json.dumps(score_dir(argv[2])["verdict"]))
        return 0
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main(sys.argv))
