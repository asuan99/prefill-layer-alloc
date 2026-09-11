#!/usr/bin/env python3
"""Scorer for the R2 GPU correctness gate (legacy vs true dual-worker).

Reads, per boot label in `boots.txt` (e.g. L1 TD1 L2 TD2):
  srv_<label>.log   server log     (ignored by git -> signatures copied below)
  tel_<label>.jsonl PD-mux telemetry (ignored by git -> signatures copied)
  gen_<label>.json  client outputs (r2_correctness_client.py)
and writes r2_correctness_report.json (committable) + a printed summary.

VERDICT RULES -- fixed here, before any GPU data exists.  The scorer does not
pick thresholds after seeing results.

Per-boot checks (every one is required for PASS):
  B1 BOOTED        gen_<label>.json exists (server came up, client ran).
  B2 NO_CRASH      no "Traceback (most recent call last)" (except the one
                   routine optional-import traceback, see parse_log), no
                   "Scheduler hit an exception", no "unsafe ... split
                   transition" in the server log; no request error in
                   either phase.
  B3 CUDAGRAPH_ON  server_args has disable_cuda_graph=False, the log has
                   "Capture cuda graph end", and EVERY "Decode batch" line
                   (--decode-log-interval 1 => every decode step) reports
                   "cuda graph: True" with at least one such line.  Prefill
                   lines are excluded: extend is never graph-captured here.
                   Validated discriminator: results/cudagraph_probe
                   srv_pdmux_cgON_847391.log 98/98 True, cgOFF 0/41.
  B4 PATH          TD boots: log has "PD-mux true dual-worker enabled",
                   every runtime_snapshot has architecture=true_dual, and
                   max(prefill_host_tasks) > 0 and max(decode_host_tasks) > 0
                   (run_batch actually ran on the two worker threads).
                   L boots: none of that; architecture=legacy.
  B5 SAME_CONFIG   random_seed, attention_backend, context_length,
                   max_running_requests, pdmux_config_path, disable_cuda_graph
                   identical across all boots; every controller_decision is
                   policy=fixed with admission_limited=False, every SAFE one
                   targets DSM (unsafe ones legitimately hold the current
                   split and are counted, not failed); no r2_admission_*
                   events.
  B6 NOT_DEGENERATE  in the reference boot (first L boot) at most half the
                   outputs are empty or a single repeated token over their
                   first 32 tokens (otherwise every arm could agree on
                   garbage and "identical" would mean nothing).

Informational per boot (reported, never folded into the verdict):
  I1 overlap snapshots: runtime_snapshot with prefill_active_batch_size>0
     and decode_running_batch_size>0, and the fraction of them whose
     realized decode_sms equals DSM (target vs realized split).
  I2 TD: max host_worker_overlap_ratio, decode_host_tasks vs Decode lines.

Output equality (token ids, per request id, all boot pairs):
  S tier: all boots must be identical on every sequential request.
  C tier: within-arm pairs (L1-L2, TD1-TD2) are the null control.

Final labels:
  NO_VERDICT_INFRA   a boot never came up and its log has no Traceback
                     (harness/infrastructure failure -- not a gate failure).
  FAIL               any B2-B5 failure in any boot, B6 failure, any S-tier
                     mismatch while the L boots agree with each other on S,
                     or C-tier: within-arm pairs identical but some
                     cross-arm pair differs.
  INCONCLUSIVE_REFERENCE  the two L boots disagree on the S tier
                     (the reference itself is not reproducible).
  PASS_SEQ__CONC_INCONCLUSIVE  S identical everywhere; C tier has
                     within-arm mismatches (timing-dependent batch
                     composition), so equality under concurrency is not
                     certified; cross vs within counts are reported.
  PASS               every check holds, S and C identical across all boots,
                     and I1 shows overlap was exercised in every boot's C
                     phase (else PASS_SEQ__CONC_NOT_EXERCISED).
"""

import argparse
import itertools
import json
import os
import re
import sys

SERVER_ARG_KEYS = (
    "random_seed", "attention_backend", "context_length", "max_running_requests",
    "pdmux_config_path", "disable_cuda_graph", "disable_piecewise_cuda_graph",
    "piecewise_cuda_graph_tokens", "enable_pdmux", "chunked_prefill_size",
    "disable_overlap_schedule", "decode_log_interval", "mem_fraction_static",
)


def parse_log(path):
    sig = {"exists": os.path.exists(path)}
    if not sig["exists"]:
        return sig
    text = open(path, errors="replace").read()
    args = {}
    m = re.search(r"server_args=ServerArgs\((.*)\)\s*$", text, re.M)
    if m:
        blob = m.group(1)
        for key in SERVER_ARG_KEYS:
            km = re.search(rf"\b{key}=('[^']*'|\[[^\]]*\]|[^,]+)", blob)
            if km:
                args[key] = km.group(1).strip("'")
    sig["server_args"] = args
    decode = [ln for ln in text.splitlines() if "Decode batch" in ln]
    sig["decode_lines"] = len(decode)
    sig["decode_graph_true"] = sum("cuda graph: True" in ln for ln in decode)
    sig["decode_graph_false"] = sum("cuda graph: False" in ln for ln in decode)
    sig["capture_end"] = text.count("Capture cuda graph end")
    sig["true_dual_banner"] = "PD-mux true dual-worker enabled" in text
    sm = re.search(r"sm_counts \(prefill_sm, decode_sm\): (\[.*?\])", text)
    sig["sm_counts"] = sm.group(1) if sm else None
    # One traceback is routine and NOT a crash: the HTTP server's optional
    # Responses-API import (`Can not initialize OpenAIServingResponses ...
    # No module named 'openai_harmony'`) is logged at every boot of this venv
    # (e.g. results/cudagraph_probe/srv_pdmux_cgON_847391.log:76).  It is
    # excluded by its header line only and counted separately.
    tb_lines = [ln for ln in text.splitlines() if "Traceback (most recent call last)" in ln]
    benign = [ln for ln in tb_lines if "Can not initialize OpenAIServingResponses" in ln]
    sig["tracebacks_benign_responses_api"] = len(benign)
    sig["tracebacks"] = len(tb_lines) - len(benign)
    sig["scheduler_exceptions"] = text.count("Scheduler hit an exception")
    sig["unsafe_transition"] = len(re.findall(r"unsafe (automatic )?split transition", text))
    first_tb = next((text.find(ln) for ln in tb_lines if ln not in benign), -1)
    sig["first_traceback"] = text[first_tb:first_tb + 1500] if first_tb >= 0 else ""
    return sig


def parse_telemetry(path, dsm):
    sig = {"exists": os.path.exists(path)}
    if not sig["exists"]:
        return sig
    snaps, decisions, other = [], [], {}
    with open(path, errors="replace") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            ev = rec.get("event")
            if ev == "runtime_snapshot":
                snaps.append(rec)
            elif ev == "controller_decision":
                decisions.append(rec)
            elif ev in ("r2_admission_recheck", "r2_admission_released"):
                other[ev] = other.get(ev, 0) + 1
    sig["snapshots"] = len(snaps)
    sig["architectures"] = sorted({s.get("architecture") for s in snaps})
    for key in ("prefill_host_tasks", "decode_host_tasks", "host_worker_overlap_ratio",
                "dropped_events"):
        vals = [s[key] for s in snaps if isinstance(s.get(key), (int, float))]
        sig[f"max_{key}"] = max(vals) if vals else None
    overlap = [s for s in snaps
               if (s.get("prefill_active_batch_size") or 0) > 0
               and (s.get("decode_running_batch_size") or 0) > 0]
    sig["overlap_snapshots"] = len(overlap)
    sig["overlap_realized_dsm_frac"] = (
        sum(1 for s in overlap if s.get("decode_sms") == dsm) / len(overlap)
        if overlap else None
    )
    sig["decisions"] = len(decisions)
    sig["decision_policies"] = sorted({d.get("policy") for d in decisions})
    # FixedPolicy returns target=current on an unsafe boundary (controller.py
    # FixedPolicy.decide); in true-dual mode a lease release can lag the
    # future's result, so an unsafe decision is protocol, not a wiring fault.
    safe = [d for d in decisions if d.get("safe", True)]
    sig["decisions_unsafe"] = len(decisions) - len(safe)
    sig["decision_targets"] = sorted({d.get("target_decode_sms") for d in safe})
    sig["decision_admission_limited"] = sum(bool(d.get("admission_limited")) for d in decisions)
    sig["r2_admission_events"] = other
    return sig


def degenerate(ids):
    head = ids[:32]
    return (not head) or len(set(head)) == 1


def compare(gen, labels, phase):
    by = {lb: {r["id"]: r for r in gen[lb][phase]} for lb in labels if lb in gen}
    ids = sorted(set().union(*[set(v) for v in by.values()])) if by else []
    pairs = {}
    for a, b in itertools.combinations([lb for lb in labels if lb in by], 2):
        mism = []
        for rid in ids:
            x, y = by[a].get(rid, {}), by[b].get(rid, {})
            ox, oy = x.get("output_ids") or [], y.get("output_ids") or []
            if ox != oy:
                prefix = 0
                for u, v in zip(ox, oy):
                    if u != v:
                        break
                    prefix += 1
                mism.append({"id": rid, "first_divergence": prefix,
                             "len": [len(ox), len(oy)]})
        pairs[f"{a}-{b}"] = {"n": len(ids), "mismatches": len(mism), "detail": mism}
    prompt_tokens_consistent = all(
        len({by[lb][rid].get("prompt_tokens") for lb in by if rid in by[lb]}) == 1
        for rid in ids
    )
    return pairs, prompt_tokens_consistent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--dsm", type=int, default=44)
    args = ap.parse_args()
    out = args.out_dir
    labels = open(os.path.join(out, "boots.txt")).read().split()
    report = {"dsm": args.dsm, "boots": labels, "per_boot": {}, "checks": {}}
    gen = {}
    for lb in labels:
        path = os.path.join(out, f"gen_{lb}.json")
        if os.path.exists(path):
            gen[lb] = json.load(open(path))
    logs = {lb: parse_log(os.path.join(out, f"srv_{lb}.log")) for lb in labels}
    tels = {lb: parse_telemetry(os.path.join(out, f"tel_{lb}.jsonl"), args.dsm) for lb in labels}

    failures, infra = [], []
    for lb in labels:
        is_td = lb.startswith("TD")
        lg, tl = logs[lb], tels[lb]
        chk = {}
        chk["B1_BOOTED"] = lb in gen
        req_errors = sum(1 for ph in ("phase_s", "phase_c")
                         for r in gen.get(lb, {}).get(ph, []) if r.get("error"))
        chk["request_errors"] = req_errors
        chk["B2_NO_CRASH"] = (lg.get("tracebacks", 0) == 0
                              and lg.get("scheduler_exceptions", 0) == 0
                              and lg.get("unsafe_transition", 0) == 0
                              and req_errors == 0)
        sa = lg.get("server_args", {})
        chk["B3_CUDAGRAPH_ON"] = (sa.get("disable_cuda_graph") == "False"
                                  and lg.get("capture_end", 0) >= 1
                                  and lg.get("decode_graph_true", 0) >= 1
                                  and lg.get("decode_graph_false", 0) == 0)
        if is_td:
            chk["B4_PATH"] = (lg.get("true_dual_banner", False)
                              and tl.get("architectures") == ["true_dual"]
                              and (tl.get("max_prefill_host_tasks") or 0) > 0
                              and (tl.get("max_decode_host_tasks") or 0) > 0)
        else:
            chk["B4_PATH"] = (not lg.get("true_dual_banner", True)
                              and tl.get("architectures") == ["legacy"])
        # Subset checks: a boot whose policy never engaged is "not exercised"
        # (caught by I1 below), not a wiring failure.
        chk["B5_POLICY"] = (set(tl.get("decision_policies") or []) <= {"fixed"}
                            and set(tl.get("decision_targets") or []) <= {args.dsm}
                            and tl.get("decision_admission_limited") == 0
                            and not tl.get("r2_admission_events"))
        report["per_boot"][lb] = {"checks": chk, "log": lg, "telemetry": tl}
        if not chk["B1_BOOTED"]:
            crashed = lg.get("tracebacks", 0) or lg.get("scheduler_exceptions", 0)
            (failures if crashed else infra).append(f"{lb}:B1")
            continue
        for key in ("B2_NO_CRASH", "B3_CUDAGRAPH_ON", "B4_PATH", "B5_POLICY"):
            if not chk[key]:
                failures.append(f"{lb}:{key}")

    booted = [lb for lb in labels if lb in gen]
    same = {}
    for key in ("random_seed", "attention_backend", "context_length",
                "max_running_requests", "pdmux_config_path", "disable_cuda_graph"):
        vals = {logs[lb].get("server_args", {}).get(key) for lb in booted}
        same[key] = sorted(v for v in vals if v is not None) if len(vals) != 1 else list(vals)
        if len(vals) != 1:
            failures.append(f"B5_SAME_CONFIG:{key}")
    report["checks"]["server_args_across_boots"] = same

    l_boots = [lb for lb in booted if lb.startswith("L")]
    td_boots = [lb for lb in booted if lb.startswith("TD")]
    if l_boots:
        ref = gen[l_boots[0]]
        outs = [r.get("output_ids") or [] for ph in ("phase_s", "phase_c") for r in ref[ph]]
        deg = sum(degenerate(o) for o in outs)
        report["checks"]["B6_reference_degenerate"] = f"{deg}/{len(outs)}"
        if deg * 2 > len(outs):
            failures.append("B6_NOT_DEGENERATE")

    s_pairs, s_pt = compare(gen, booted, "phase_s")
    c_pairs, c_pt = compare(gen, booted, "phase_c")
    report["checks"]["S_pairs"] = s_pairs
    report["checks"]["C_pairs"] = c_pairs
    report["checks"]["prompt_tokens_consistent"] = {"S": s_pt, "C": c_pt}
    if not (s_pt and c_pt):
        failures.append("INPUT_IDENTITY")

    def pair_key(a, b):
        return f"{a}-{b}" if f"{a}-{b}" in c_pairs or f"{a}-{b}" in s_pairs else f"{b}-{a}"

    within = [pair_key(a, b) for grp in (l_boots, td_boots)
              for a, b in itertools.combinations(grp, 2)]
    cross = [pair_key(a, b) for a in l_boots for b in td_boots]
    s_ref_ok = all(s_pairs[pair_key(a, b)]["mismatches"] == 0
                   for a, b in itertools.combinations(l_boots, 2))
    s_all_ok = all(v["mismatches"] == 0 for v in s_pairs.values())
    c_within = {k: c_pairs[k]["mismatches"] for k in within if k in c_pairs}
    c_cross = {k: c_pairs[k]["mismatches"] for k in cross if k in c_pairs}
    report["checks"]["C_within_arm_mismatches"] = c_within
    report["checks"]["C_cross_arm_mismatches"] = c_cross
    overlap_ok = all((tels[lb].get("overlap_snapshots") or 0) > 0 for lb in booted)
    report["checks"]["I1_overlap_exercised_all_boots"] = overlap_ok

    if infra and not failures:
        verdict = "NO_VERDICT_INFRA"
    elif failures:
        verdict = "FAIL"
    elif not s_ref_ok:
        verdict = "INCONCLUSIVE_REFERENCE"
    elif not s_all_ok:
        verdict = "FAIL"
        failures.append("S_TIER_MISMATCH")
    elif not td_boots or not l_boots:
        verdict = "NO_VERDICT_INFRA"
    elif any(c_within.values()):
        verdict = "PASS_SEQ__CONC_INCONCLUSIVE"
    elif any(c_cross.values()):
        verdict = "FAIL"
        failures.append("C_TIER_CROSS_ARM_MISMATCH_WITH_DETERMINISTIC_NULL")
    elif not overlap_ok:
        verdict = "PASS_SEQ__CONC_NOT_EXERCISED"
    else:
        verdict = "PASS"
    report["failures"], report["infra"], report["verdict"] = failures, infra, verdict

    with open(os.path.join(out, "r2_correctness_report.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    print(f"boots: {labels}  (booted: {booted})  DSM={args.dsm}")
    for lb in labels:
        pb = report["per_boot"][lb]
        lg, tl, ck = pb["log"], pb["telemetry"], pb["checks"]
        print(f"  {lb:4s} booted={ck['B1_BOOTED']} crash_free={ck['B2_NO_CRASH']} "
              f"graph_on={ck['B3_CUDAGRAPH_ON']} "
              f"(decode True/False={lg.get('decode_graph_true')}/{lg.get('decode_graph_false')}, "
              f"capture_end={lg.get('capture_end')}) path={ck['B4_PATH']} "
              f"arch={tl.get('architectures')} host_tasks p/d="
              f"{tl.get('max_prefill_host_tasks')}/{tl.get('max_decode_host_tasks')} "
              f"policy_ok={ck['B5_POLICY']} unsafe_decisions={tl.get('decisions_unsafe')} "
              f"overlap_snaps={tl.get('overlap_snapshots')} "
              f"realized_D{args.dsm}_frac={tl.get('overlap_realized_dsm_frac')}")
    print(f"  S-tier mismatches: { {k: v['mismatches'] for k, v in s_pairs.items()} }")
    print(f"  C-tier within-arm: {c_within}  cross-arm: {c_cross}")
    print(f"  failures={failures} infra={infra}")
    print(f"VERDICT {verdict}")
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
