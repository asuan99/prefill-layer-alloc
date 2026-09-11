#!/usr/bin/env python3
"""Scorer for the R2 GPU correctness gate (legacy vs true dual-worker).

Reads, per boot label in `boots.txt` (e.g. L1 TD1 L2 TD2):
  srv_<label>.log     server log        (git-ignored -> signatures copied)
  tel_<label>.jsonl   PD-mux telemetry  (git-ignored -> signatures copied)
  green_<label>.json  driver green-context read-out (PDMUX_GREEN_READOUT)
  gen_<label>.json    client outputs (r2_correctness_client.py)
and writes the report (default <out_dir>/r2_correctness_report.json,
committable) plus a printed summary.

======================================================================
VERDICT RULE v2 -- fixed 2026-09-11, BEFORE any true-dual boot has
produced an output.  The scorer never picks a threshold after seeing data.
======================================================================
Tiers
  S  16 requests, one at a time.                           EQUALITY GATE
  O  8 overlap probes: each probe is sent while one long background request
     decodes, so its prefill runs on the fixed D division concurrently with
     that decode.  Probe token ids are compared; background outputs are
     excluded.                                             EQUALITY GATE
  C  32 staggered concurrent requests.                     DIAGNOSTIC ONLY
     (mismatch counts are reported and never enter the verdict)

Per-boot checks (every boot, both arms)
  B1 BOOTED        the client output exists.
  B2 NO_CRASH      no non-benign traceback, no "Scheduler hit an exception",
                   no "unsafe ... split transition", no request error.
  B3 CUDAGRAPH_ON  disable_cuda_graph=False, "Capture cuda graph end", and
                   every "Decode batch" line says "cuda graph: True" (>= 1).
  B4 PATH          TD: true-dual banner, every snapshot architecture=true_dual,
                   max prefill_host_tasks > 0 and max decode_host_tasks > 0.
                   L: no banner, every snapshot architecture=legacy.
  B5 POLICY        every controller_decision is policy=fixed with
                   admission_limited=False; every SAFE decision targets D;
                   no r2_admission_* events; server args identical across
                   boots (seed, backend, context, cap, config, cudagraph,
                   triton_attention_num_kv_splits).
  B6 NOT_DEGENERATE  the first L boot: at most half of all outputs are empty
                   or one repeated token over their first 32 tokens.
  B7 OVERLAP       >= 1 snapshot with prefill and decode active at once.
  B8 SPLIT_REALIZED  every such snapshot is on the D division index, AND the
                   driver read-out (green_<label>.json) reports that index's
                   prefill/decode green contexts non-null with smCount =
                   total-D / D.  (Realization, not target: the stream-group
                   label alone is the engine's target.)
O-tier confirmation (every probe, every boot)
  O1 protocol      probe and background finished without error; the
                   background had streamed its first token before the probe
                   was sent; the background finished after the probe; and
                   probe prompt_tokens > (K-1) * (bg prompt_tokens +
                   bg max_new), K = triton_attention_num_kv_splits (the
                   condition under which the probe's decode KV-split count
                   does not depend on the background; see the client).
  O2 overlap       >= 1 snapshot inside [probe sent, probe done] (perf_counter
                   = CLOCK_MONOTONIC, the telemetry clock) with prefill and
                   decode active.
  O3 division      every snapshot counted in O2 is on the D division index
                   (its realized SM counts are B8's read-out).
  O4 threads (TD)  prefill_host_tasks and decode_host_tasks both increase from
                   the last snapshot at or before the probe was sent to the
                   last snapshot inside the window, and some in-window
                   snapshot shows decode batch >= 2 (probe decoding alongside
                   the background on the decode worker).

Verdict, first matching line wins
  1 FAIL                   any B2-B6 failure in any boot; a boot missing (B1)
                           whose log shows a crash; input identity differs
                           (S or O prompt_tokens) across boots; an S mismatch
                           between any two boots while the two L boots agree
                           on S; an O mismatch between an L and a TD boot
                           while both within-arm O pairs (L-L and TD-TD) are
                           present and identical.
  2 NO_VERDICT_INFRA       a boot is missing and its log shows no crash, or
                           an arm has no booted boot at all.
  3 INCONCLUSIVE           the two L boots disagree on S (reference not
                           reproducible), or any within-arm O pair differs.
  4 NO_VERDICT_UNREALIZED  B7 or B8 fails in some boot, or O1-O4 is not
                           confirmed for some probe in some boot (the fixed D
                           split under overlap was not shown to be exercised
                           and realized, so equality there is not certified).
  5 PASS                   otherwise: every boot passes B1-B8, S and O are
                           token-identical across all boots, O1-O4 hold for
                           every probe in every boot.

Why this rule is not chosen after the fact
  * No true-dual boot has ever produced an output: in job 907032 both TD
    boots crashed on the server's own warm-up request.  Nothing about
    true-dual outputs could have informed this rule.
  * The C demotion applies the condition declared before job 907032 ("a
    cross-arm mismatch only counts when legacy-vs-legacy is identical"):
    job 907032 measured legacy-vs-legacy C mismatch 7/32 (one request at
    token 0), so C cannot meet that condition by construction.  The O tier
    replaces it with a concurrency test that is deterministic by construction
    and carries its own null control (rule line 3).
  * S alone never runs on the D division (prefill-only states use the plain
    group 0, decode-only states the last plain group; job 907032 L1: every
    overlap snapshot on index 4, none otherwise), hence the O tier and B8.
======================================================================
"""

import argparse
import ast
import itertools
import json
import os
import re
import sys

RULE_VERSION = "r2_correctness verdict rule v2 (2026-09-11)"

SERVER_ARG_KEYS = (
    "random_seed", "attention_backend", "context_length", "max_running_requests",
    "pdmux_config_path", "disable_cuda_graph", "disable_piecewise_cuda_graph",
    "piecewise_cuda_graph_tokens", "enable_pdmux", "chunked_prefill_size",
    "disable_overlap_schedule", "decode_log_interval", "mem_fraction_static",
    "triton_attention_num_kv_splits",
)
SAME_ACROSS_BOOTS = (
    "random_seed", "attention_backend", "context_length", "max_running_requests",
    "pdmux_config_path", "disable_cuda_graph", "triton_attention_num_kv_splits",
)
SNAP_KEYS = (
    "timestamp_monotonic_s", "architecture", "prefill_active_batch_size",
    "decode_running_batch_size", "stream_index", "prefill_sms", "decode_sms",
    "prefill_host_tasks", "decode_host_tasks", "host_worker_overlap_ratio",
    "dropped_events",
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
    # No module named 'openai_harmony'`), logged at boot in this venv
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


def division_index(sm_counts_text, dsm):
    """Index of the (total-D, D) division in the engine's sm_counts table."""
    if not sm_counts_text:
        return None, None
    try:
        table = [tuple(int(v) for v in row) for row in ast.literal_eval(sm_counts_text)]
    except (ValueError, SyntaxError, TypeError):
        return None, None
    total = table[0][0] if table else None
    for idx, row in enumerate(table):
        if total is not None and row == (total - dsm, dsm) and 0 < idx < len(table) - 1:
            return idx, total
    return None, total


def parse_green(path, div_idx, total, dsm):
    sig = {"exists": os.path.exists(path), "division_index": div_idx}
    if not sig["exists"] or div_idx is None or total is None:
        sig["confirmed"] = False
        return sig
    try:
        doc = json.load(open(path))
    except ValueError as exc:
        sig.update(confirmed=False, error=repr(exc))
        return sig
    row = next((g for g in doc.get("groups", []) if g.get("stream_index") == div_idx), None)
    if row is None:
        sig["confirmed"] = False
        return sig
    realized = {}
    for role in ("prefill", "decode"):
        rec = row.get(role) or {}
        realized[role] = {
            "green_ctx_is_null": rec.get("green_ctx_is_null"),
            "smCount": (rec.get("green_sm") or {}).get("smCount"),
            "error": rec.get("error"),
        }
    sig["realized"] = realized
    sig["target"] = [row.get("target_prefill_sm"), row.get("target_decode_sm")]
    sig["confirmed"] = (
        realized["prefill"]["green_ctx_is_null"] is False
        and realized["decode"]["green_ctx_is_null"] is False
        and realized["prefill"]["smCount"] == total - dsm
        and realized["decode"]["smCount"] == dsm
    )
    return sig


def parse_telemetry(path):
    sig = {"exists": os.path.exists(path)}
    snaps, decisions, other = [], [], {}
    if sig["exists"]:
        with open(path, errors="replace") as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                ev = rec.get("event")
                if ev == "runtime_snapshot":
                    snaps.append({k: rec.get(k) for k in SNAP_KEYS})
                elif ev == "controller_decision":
                    decisions.append(rec)
                elif ev in ("r2_admission_recheck", "r2_admission_released"):
                    other[ev] = other.get(ev, 0) + 1
    snaps.sort(key=lambda s: s.get("timestamp_monotonic_s") or 0.0)
    sig["snapshots"] = len(snaps)
    sig["architectures"] = sorted({s.get("architecture") for s in snaps})
    for key in ("prefill_host_tasks", "decode_host_tasks", "host_worker_overlap_ratio",
                "dropped_events"):
        vals = [s[key] for s in snaps if isinstance(s.get(key), (int, float))]
        sig[f"max_{key}"] = max(vals) if vals else None
    sig["decisions"] = len(decisions)
    sig["decision_policies"] = sorted({d.get("policy") for d in decisions})
    # FixedPolicy returns target=current on an unsafe boundary; in true-dual
    # mode that is protocol, not a wiring fault (counted, not failed).
    safe = [d for d in decisions if d.get("safe", True)]
    sig["decisions_unsafe"] = len(decisions) - len(safe)
    sig["decision_targets"] = sorted({d.get("target_decode_sms") for d in safe})
    sig["decision_admission_limited"] = sum(bool(d.get("admission_limited")) for d in decisions)
    sig["r2_admission_events"] = other
    return sig, snaps


def overlap(s):
    return (s.get("prefill_active_batch_size") or 0) > 0 and \
        (s.get("decode_running_batch_size") or 0) > 0


def o_tier_confirmation(doc, snaps, div_idx, is_td, kv_splits):
    """O1-O4 for every probe of one boot.  Returns (confirmed, per-probe list)."""
    probes = (doc or {}).get("phase_o") or []
    if not probes:
        return False, [{"id": None, "reason": "O tier not run"}]
    per = []
    for entry in probes:
        pr, bg = entry.get("probe") or {}, entry.get("bg") or {}
        r = {"id": entry.get("id")}
        lo, hi = pr.get("pc_start"), pr.get("pc_end")
        bg_len_max = None
        if bg.get("prompt_tokens") is not None and bg.get("max_new") is not None:
            bg_len_max = bg["prompt_tokens"] + bg["max_new"]
        r["O1_protocol"] = bool(
            not pr.get("error") and not bg.get("error")
            and bg.get("pc_first") is not None and lo is not None and hi is not None
            and bg["pc_first"] <= lo
            and bg.get("pc_end") is not None and bg["pc_end"] > hi
            and pr.get("prompt_tokens") is not None and bg_len_max is not None
            and kv_splits is not None
            and pr["prompt_tokens"] > (kv_splits - 1) * bg_len_max
        )
        win = [s for s in snaps if lo is not None and hi is not None
               and lo <= (s.get("timestamp_monotonic_s") or -1.0) <= hi]
        ov = [s for s in win if overlap(s)]
        r["overlap_snapshots"] = len(ov)
        r["O2_overlap"] = len(ov) > 0
        r["O3_division"] = bool(ov) and div_idx is not None and all(
            s.get("stream_index") == div_idx for s in ov
        )
        r["window_stream_indices"] = sorted({s.get("stream_index") for s in ov})
        if is_td:
            before = [s for s in snaps
                      if lo is not None and (s.get("timestamp_monotonic_s") or 0.0) <= lo]
            base = before[-1] if before else {}
            last = win[-1] if win else {}

            def delta(key):
                if not last or not isinstance(last.get(key), (int, float)):
                    return None
                return last[key] - (base.get(key) or 0)

            r["prefill_task_delta"] = delta("prefill_host_tasks")
            r["decode_task_delta"] = delta("decode_host_tasks")
            r["probe_in_decode_batch"] = any(
                (s.get("decode_running_batch_size") or 0) >= 2 for s in win
            )
            r["O4_threads"] = bool(
                (r["prefill_task_delta"] or 0) > 0 and (r["decode_task_delta"] or 0) > 0
                and r["probe_in_decode_batch"]
            )
        else:
            r["O4_threads"] = True  # not applicable to the legacy loop
        r["confirmed"] = all(r[k] for k in ("O1_protocol", "O2_overlap", "O3_division",
                                            "O4_threads"))
        per.append(r)
    return all(r["confirmed"] for r in per), per


def degenerate(ids):
    head = ids[:32]
    return (not head) or len(set(head)) == 1


def tier_records(doc, tier):
    if tier == "O":
        return [dict(e.get("probe") or {}, id=e.get("id")) for e in doc.get("phase_o") or []]
    return list(doc.get({"S": "phase_s", "C": "phase_c"}[tier]) or [])


def compare(gen, labels, tier):
    # A boot with NO records for a tier (tier not run) is not comparable on it;
    # that is an unconfirmed tier (O_TIER_CONFIRMED), not a null-control
    # disagreement.  A request missing inside a tier that did run still counts.
    by = {lb: {r["id"]: r for r in tier_records(gen[lb], tier)}
          for lb in labels if lb in gen and tier_records(gen[lb], tier)}
    ids = sorted(set().union(*[set(v) for v in by.values()])) if by else []
    pairs = {}
    for a, b in itertools.combinations([lb for lb in labels if lb in by], 2):
        mism = []
        for rid in ids:
            x, y = by[a].get(rid, {}), by[b].get(rid, {})
            ox, oy = x.get("output_ids") or [], y.get("output_ids") or []
            if ox != oy or rid not in by[a] or rid not in by[b]:
                prefix = 0
                for u, v in zip(ox, oy):
                    if u != v:
                        break
                    prefix += 1
                mism.append({"id": rid, "first_divergence": prefix,
                             "len": [len(ox), len(oy)]})
        pairs[f"{a}-{b}"] = {"n": len(ids), "mismatches": len(mism), "detail": mism}
    consistent = all(
        len({by[lb][rid].get("prompt_tokens") for lb in by if rid in by[lb]}) == 1
        for rid in ids
    )
    return pairs, consistent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--dsm", type=int, default=44)
    ap.add_argument("--report", default=None,
                    help="report path (default <out_dir>/r2_correctness_report.json)")
    args = ap.parse_args()
    out = args.out_dir
    report_path = args.report or os.path.join(out, "r2_correctness_report.json")
    labels = open(os.path.join(out, "boots.txt")).read().split()
    report = {"rule": RULE_VERSION, "dsm": args.dsm, "boots": labels,
              "per_boot": {}, "checks": {}}
    gen = {}
    for lb in labels:
        path = os.path.join(out, f"gen_{lb}.json")
        if os.path.exists(path):
            gen[lb] = json.load(open(path))
    logs = {lb: parse_log(os.path.join(out, f"srv_{lb}.log")) for lb in labels}

    failures, infra, unrealized = [], [], []
    for lb in labels:
        is_td = lb.startswith("TD")
        lg = logs[lb]
        tl, snaps = parse_telemetry(os.path.join(out, f"tel_{lb}.jsonl"))
        div_idx, total = division_index(lg.get("sm_counts"), args.dsm)
        green = parse_green(os.path.join(out, f"green_{lb}.json"), div_idx, total, args.dsm)
        sa = lg.get("server_args", {})
        try:
            kv_splits = int(sa.get("triton_attention_num_kv_splits"))
        except (TypeError, ValueError):
            kv_splits = None
        chk = {"B1_BOOTED": lb in gen}
        req_errors = sum(
            1 for tier in ("S", "O", "C") for r in tier_records(gen.get(lb, {}), tier)
            if r.get("error")
        ) + sum(1 for e in gen.get(lb, {}).get("phase_o") or [] if (e.get("bg") or {}).get("error"))
        chk["request_errors"] = req_errors
        chk["B2_NO_CRASH"] = (lg.get("tracebacks", 0) == 0
                              and lg.get("scheduler_exceptions", 0) == 0
                              and lg.get("unsafe_transition", 0) == 0
                              and req_errors == 0)
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
        chk["B5_POLICY"] = (set(tl.get("decision_policies") or []) <= {"fixed"}
                            and set(tl.get("decision_targets") or []) <= {args.dsm}
                            and tl.get("decision_admission_limited") == 0
                            and not tl.get("r2_admission_events"))
        ov_all = [s for s in snaps if overlap(s)]
        tl["overlap_snapshots"] = len(ov_all)
        tl["overlap_on_division"] = sum(1 for s in ov_all if s.get("stream_index") == div_idx)
        chk["B7_OVERLAP"] = len(ov_all) > 0
        chk["B8_SPLIT_REALIZED"] = (bool(ov_all) and div_idx is not None
                                    and tl["overlap_on_division"] == len(ov_all)
                                    and green.get("confirmed", False))
        o_ok, o_per = o_tier_confirmation(gen.get(lb), snaps, div_idx, is_td, kv_splits)
        chk["O_TIER_CONFIRMED"] = o_ok
        report["per_boot"][lb] = {"checks": chk, "log": lg, "telemetry": tl,
                                  "green": green, "division_index": div_idx,
                                  "o_tier": o_per}
        if not chk["B1_BOOTED"]:
            crashed = lg.get("tracebacks", 0) or lg.get("scheduler_exceptions", 0)
            (failures if crashed else infra).append(f"{lb}:B1")
            continue
        for key in ("B2_NO_CRASH", "B3_CUDAGRAPH_ON", "B4_PATH", "B5_POLICY"):
            if not chk[key]:
                failures.append(f"{lb}:{key}")
        for key in ("B7_OVERLAP", "B8_SPLIT_REALIZED", "O_TIER_CONFIRMED"):
            if not chk[key]:
                unrealized.append(f"{lb}:{key}")

    booted = [lb for lb in labels if lb in gen]
    same = {}
    for key in SAME_ACROSS_BOOTS:
        vals = {logs[lb].get("server_args", {}).get(key) for lb in booted}
        same[key] = sorted(str(v) for v in vals)
        if len(vals) > 1:
            failures.append(f"B5_SAME_CONFIG:{key}")
    report["checks"]["server_args_across_boots"] = same

    l_boots = [lb for lb in booted if lb.startswith("L")]
    td_boots = [lb for lb in booted if lb.startswith("TD")]
    if l_boots:
        ref = gen[l_boots[0]]
        outs = [r.get("output_ids") or [] for tier in ("S", "O", "C")
                for r in tier_records(ref, tier)]
        deg = sum(degenerate(o) for o in outs)
        report["checks"]["B6_reference_degenerate"] = f"{deg}/{len(outs)}"
        if deg * 2 > len(outs):
            failures.append("B6_NOT_DEGENERATE")

    pairs, consistent = {}, {}
    for tier in ("S", "O", "C"):
        pairs[tier], consistent[tier] = compare(gen, booted, tier)
        report["checks"][f"{tier}_pairs"] = pairs[tier]
    report["checks"]["prompt_tokens_consistent"] = consistent
    if not (consistent["S"] and consistent["O"]):
        failures.append("INPUT_IDENTITY")

    def key(a, b):
        return f"{a}-{b}" if f"{a}-{b}" in pairs["S"] else f"{b}-{a}"

    within = [key(a, b) for grp in (l_boots, td_boots) for a, b in itertools.combinations(grp, 2)]
    cross = [key(a, b) for a in l_boots for b in td_boots]
    mm = {t: {k: pairs[t][k]["mismatches"] for k in pairs[t]} for t in pairs}
    s_ref_ok = all(mm["S"][key(a, b)] == 0 for a, b in itertools.combinations(l_boots, 2))
    s_all_ok = all(v == 0 for v in mm["S"].values())
    o_within = {k: mm["O"][k] for k in within if k in mm["O"]}
    o_cross = {k: mm["O"][k] for k in cross if k in mm["O"]}
    report["checks"]["O_within_arm_mismatches"] = o_within
    report["checks"]["O_cross_arm_mismatches"] = o_cross
    report["checks"]["C_diagnostic_within_arm"] = {k: mm["C"][k] for k in within if k in mm["C"]}
    report["checks"]["C_diagnostic_cross_arm"] = {k: mm["C"][k] for k in cross if k in mm["C"]}

    if s_ref_ok and not s_all_ok:
        failures.append("S_TIER_MISMATCH")
    # An O-tier FAIL needs the complete null control: both within-arm pairs
    # present and identical.  With a pair missing the O tier is unconfirmed.
    o_null_complete = bool(within) and all(k in mm["O"] for k in within) and \
        len(l_boots) >= 2 and len(td_boots) >= 2
    report["checks"]["O_null_control_complete"] = o_null_complete
    if o_null_complete and not any(o_within.values()) and any(o_cross.values()):
        failures.append("O_TIER_CROSS_ARM_MISMATCH_WITH_IDENTICAL_NULL")

    if failures:
        verdict = "FAIL"
    elif infra or not l_boots or not td_boots:
        verdict = "NO_VERDICT_INFRA"
    elif not s_ref_ok or any(o_within.values()):
        verdict = "INCONCLUSIVE"
    elif unrealized:
        verdict = "NO_VERDICT_UNREALIZED"
    else:
        verdict = "PASS"
    report.update(failures=failures, infra=infra, unrealized=unrealized, verdict=verdict)

    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    print(f"{RULE_VERSION}  boots={labels} booted={booted} D={args.dsm}")
    for lb in labels:
        pb = report["per_boot"][lb]
        lg, tl, ck, gr = pb["log"], pb["telemetry"], pb["checks"], pb["green"]
        print(f"  {lb:4s} booted={ck['B1_BOOTED']} crash_free={ck['B2_NO_CRASH']} "
              f"graph_on={ck['B3_CUDAGRAPH_ON']} (decode T/F={lg.get('decode_graph_true')}/"
              f"{lg.get('decode_graph_false')}) path={ck['B4_PATH']} policy={ck['B5_POLICY']} "
              f"overlap={ck['B7_OVERLAP']} ({tl.get('overlap_on_division')}/"
              f"{tl.get('overlap_snapshots')} on idx {pb['division_index']}) "
              f"green_realized={gr.get('confirmed')} O_confirmed={ck['O_TIER_CONFIRMED']} "
              f"unsafe_decisions={tl.get('decisions_unsafe')}")
    print(f"  S mismatches: {mm['S']}")
    print(f"  O within-arm: {o_within}  cross-arm: {o_cross}")
    print(f"  C (diagnostic) within-arm: {report['checks']['C_diagnostic_within_arm']}  "
          f"cross-arm: {report['checks']['C_diagnostic_cross_arm']}")
    print(f"  failures={failures} infra={infra} unrealized={unrealized}")
    print(f"VERDICT {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
