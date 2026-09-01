#!/usr/bin/env python3
"""Score the seven P1 acceptance tests (PREREG_CP0_2026-08-28.md section 2.2).

The decision rule is code, not prose, and it is written BEFORE the measurement
PROJECT_STATUS.md "방법론 게이트" #34 ("사전등록은 규칙과 하네스를 한 번에
감사받으면 안 된다 -- 규칙 먼저 감사받고, 통과한 규칙에 대고 지은 하네스를 다시
감사받아라") and #61 ("규칙을 산문으로 고정하면 구멍이 난다 -- 결정 규칙은 코드로
고정하고 세계를 전수 열거하라").

FOUR-WAY FAILURE CLASSIFICATION (prereg section 2.3).  A test that does not pass
is NOT automatically "the instrument is broken":

    BOOT_FAILED             (i)   the server never came up.  Retry.  Says
                                  nothing about the arm and nothing about P1.
    EXTRACTION_EMPTY        (ii)  it came up but the counters are missing or
                                  self-inconsistent.  Instrument defect.
    ENGINE_BUDGET_EXCEEDED  (iii) P1-a's "<= cps" was violated.  This is a
                                  FINDING ABOUT THE ENGINE -- a path exists that
                                  overruns the prefill token budget -- and it is
                                  reported separately.  Folding it into
                                  "instrument failure" is the same error as
                                  labelling a measurement failure a gate failure
                                  (PROJECT_STATUS.md "방법론 게이트" #21, which
                                  the prereg notes is bidirectional).
    PERTURBED               (iv)  P1-c / P1-f violated: the probe is not an
                                  observer.
    FAIL                          the registered predicate is false and none of
                                  the four classes applies.
    NOT_RUN                       the input for this test was not supplied.

★THE VERDICT IS A PRODUCT OF TWO AXES (rule_rev 2, 2026-09-01)
--------------------------------------------------------------
Rule rev 1 ranked the labels on one ladder --
``ENGINE_BUDGET_EXCEEDED > PERTURBED > EXTRACTION_EMPTY > BOOT_FAILED >
NOT_RUN > else`` -- and ``FAIL`` only ever reached the ``else``.  So in job
899768 three ``EXTRACTION_EMPTY`` labels **swallowed P1-e's ``FAIL``**: an
arithmetic disagreement with hand-computed, pre-registered expectations
vanished from the verdict because some *other* test's plumbing had broken
(``RESULT_P1_899768_2026-08-28.md``, "부수 발견" 1).

``FAIL`` is not a fifth member of the section 2.3 classification; it is on a
different axis.  Section 2.3 answers *"did the measurement happen?"*; ``FAIL``
answers *"is the registered expectation true?"*.  Those are independent, so the
verdict reports both and neither may absorb the other:

    verdict_measurement   MEASUREMENT_CLEAN | P1_BLOCKED_ENGINE_FINDING |
                          P1_BLOCKED_NOT_AN_OBSERVER | P1_BLOCKED_INSTRUMENT |
                          P1_INCONCLUSIVE_BOOT | P1_INCOMPLETE
                          -- the section 2.3 ladder, UNCHANGED
    verdict_expectation   EXPECTATION_HELD | EXPECTATION_HELD_WHERE_EVALUATED |
                          P1_EXPECTATION_REFUTED | EXPECTATION_UNEVALUATED
    verdict               "P1_ACCEPTED" iff both are clean, else the two axes
                          joined by " AND "

This is the same discipline as PROJECT_STATUS.md "방법론 게이트" #21 ("측정
실패를 게이트 실패로 라벨링 마라"), read in its other direction: a measurement
failure must not be allowed to *hide* a gate failure either.

WHAT THIS SCRIPT DOES NOT DO.  It makes no performance judgement, ranks no arm,
and reads no goodput (prereg section 9).  P1-f's throughput numbers are used
only as an equivalence check on the instrument itself.
"""

import argparse
import json
import math
import sys
from pathlib import Path

RULE_REV = 2
SCHEMA = "cp0.p1-acceptance/v1"

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BOOT_FAILED = "BOOT_FAILED"
EXTRACTION_EMPTY = "EXTRACTION_EMPTY"
ENGINE_BUDGET_EXCEEDED = "ENGINE_BUDGET_EXCEEDED"
PERTURBED = "PERTURBED"

# axis 1 -- "did the measurement happen?"  (prereg section 2.3, unchanged)
MEASUREMENT_CLEAN = "MEASUREMENT_CLEAN"
# axis 2 -- "is the registered expectation true?"
EXPECTATION_HELD = "EXPECTATION_HELD"
EXPECTATION_HELD_WHERE_EVALUATED = "EXPECTATION_HELD_WHERE_EVALUATED"
EXPECTATION_REFUTED = "P1_EXPECTATION_REFUTED"
EXPECTATION_UNEVALUATED = "EXPECTATION_UNEVALUATED"

# The equivalence margin is NOT a new constant: it is this project's standard
# one, reused.  CLAUDE.md gate #3: "베이스라인 분산 먼저 측정, n>=4 없이 정책
# 결론 금지. 3% 미만 차이는 headline 아님".
NEUTRALITY_MARGIN = 0.03


def _kv(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"expected NAME=PATH, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = json.loads(Path(v).read_text())
    return out


def _boot_usable(rec):
    """-> (label, note) or (None, None) when the record is fit to score."""
    if rec is None:
        return NOT_RUN, "no boot record supplied"
    if not rec.get("boot_ok", True):
        return BOOT_FAILED, "harness recorded boot_ok=false"
    if rec.get("status") != "OK":
        return EXTRACTION_EMPTY, "; ".join(rec.get("problems", [])) or "status != OK"
    tot = rec.get("boot_total_from_events") or {}
    if (tot.get("extend_tokens_per_forward") or {}).get("n_forwards", 0) == 0:
        return EXTRACTION_EMPTY, "no prefill forward was observed in this boot"
    return None, None


def _tot(rec):
    return rec["boot_total_from_events"]


def _maxext(tot):
    return (tot.get("extend_tokens_per_forward") or {}).get("max")


def check_p1a(rec):
    """cps 512 boot: chunking happens, events >= requests, no forward over cps."""
    lab, note = _boot_usable(rec)
    if lab:
        return lab, note, {}
    t = _tot(rec)
    cps = rec.get("realized_chunked_prefill_size")
    obs = {
        "realized_chunked_prefill_size": cps,
        "n_requests_chunked": t["n_requests_chunked"],
        "n_chunk_events": t["n_chunk_events"],
        "max_extend_tokens": _maxext(t),
    }
    if cps != 512:
        return FAIL, f"expected a cps 512 boot, got {cps}", obs
    if _maxext(t) is not None and _maxext(t) > cps:
        return (
            ENGINE_BUDGET_EXCEEDED,
            f"a prefill forward carried {_maxext(t)} > cps {cps} extend tokens; "
            "this is a finding about the engine, not an instrument failure",
            obs,
        )
    if t["n_requests_chunked"] <= 0:
        return FAIL, "no request was chunked at cps 512", obs
    if t["n_chunk_events"] < t["n_requests_chunked"]:
        return FAIL, "n_chunk_events < n_requests_chunked (impossible)", obs
    return PASS, "", obs


def check_p1b(rec):
    """fused_mono (cps -1): zero chunking, and a forward larger than 512."""
    lab, note = _boot_usable(rec)
    if lab:
        return lab, note, {}
    t = _tot(rec)
    obs = {
        "requested_chunked_prefill_size": rec.get("requested_chunked_prefill_size"),
        "realized_chunked_prefill_size": rec.get("realized_chunked_prefill_size"),
        "n_requests_chunked": t["n_requests_chunked"],
        "n_chunk_events": t["n_chunk_events"],
        "max_extend_tokens": _maxext(t),
    }
    if rec.get("realized_chunked_prefill_size") is not None:
        return FAIL, "the engine did not disable chunking on this boot", obs
    if t["n_requests_chunked"] or t["n_chunk_events"]:
        return FAIL, "chunk events on an arm that cannot chunk", obs
    if not (_maxext(t) or 0) > 512:
        return (
            FAIL,
            "no forward exceeded 512 extend tokens, so this boot does not "
            "demonstrate that the counters would have fired if chunking existed",
            obs,
        )
    return PASS, "", obs


def check_p1c(on_doc, off_doc):
    """Bit-identical greedy output with the probe ON and OFF."""
    if on_doc is None or off_doc is None:
        return NOT_RUN, "greedy artifacts not supplied", {}
    on, off = on_doc.get("responses"), off_doc.get("responses")
    if not on or not off:
        return EXTRACTION_EMPTY, "greedy artifact has no responses", {}
    if len(on) != len(off):
        return PERTURBED, f"{len(on)} vs {len(off)} responses", {}
    def fingerprint(r):
        # ids when the engine returned them, text always: a difference in
        # either is a difference.  Comparing only text would let a retokenised
        # but differently-sampled continuation slip through.
        return (r.get("output_ids"), r.get("text"), r.get("finish_reason"))

    diffs = [
        i for i, (a, b) in enumerate(zip(on, off)) if fingerprint(a) != fingerprint(b)
    ]
    no_ids = sum(1 for r in on if r.get("output_ids") is None)
    obs = {"n_prompts": len(on), "n_differing": len(diffs),
           "differing_index": diffs[:8], "n_without_token_ids": no_ids}
    if diffs:
        return PERTURBED, "greedy decode differs with the probe on", obs
    return PASS, "", obs


def check_p1d(rec, phase, expectations):
    """Batch-boundary positive control: cps 4096, prompts all shorter than cps."""
    lab, note = _boot_usable(rec)
    if lab:
        return lab, note, {}
    if phase is None:
        return NOT_RUN, "phase D artifact not supplied", {}
    exp = expectations["p1d_positive_control"]
    t = _tot(rec)
    obs = {
        "realized_chunked_prefill_size": rec.get("realized_chunked_prefill_size"),
        "n_prompts": phase.get("n_prompts"),
        "n_prompts_longer_than_cps": phase.get("n_prompts_longer_than_cps"),
        "n_requests_chunked": t["n_requests_chunked"],
        "n_chunk_events": t["n_chunk_events"],
        "max_extend_tokens": _maxext(t),
    }
    if rec.get("realized_chunked_prefill_size") != exp["chunked_prefill_size"]:
        return FAIL, "wrong cps for the positive control", obs
    if phase.get("length_mismatch"):
        return (
            EXTRACTION_EMPTY,
            f"the engine served {phase.get('served_lens')} rather than the "
            f"requested {phase.get('requested_lens')}",
            obs,
        )
    if phase.get("n_prompts_longer_than_cps") != exp["expected"]["n_prompts_longer_than_cps"]:
        return (
            FAIL,
            "the burst contained a prompt longer than cps, so this control no "
            "longer separates the batch channel from the length channel",
            obs,
        )
    if _maxext(t) is not None and _maxext(t) > exp["expected"]["max_extend_tokens_max"]:
        return ENGINE_BUDGET_EXCEEDED, "forward exceeded the cps budget", obs
    if t["n_requests_chunked"] < exp["expected"]["n_requests_chunked_min"]:
        return (
            FAIL,
            "no request was chunked although every prompt was shorter than cps "
            "-- either the batch-budget mechanism did not fire (the burst did "
            "not build a backlog) or the instrument only counts long prompts. "
            "Re-run with a larger burst before concluding the second.",
            obs,
        )
    return PASS, "", obs


def _seg(rec, segment_id):
    for s in rec.get("segments", []):
        if s.get("segment_id") == segment_id:
            return s
    return None


def check_p1e(rec, phases_doc, expectations):
    """The separation test: (1), (2) and (3) must come out different."""
    lab, note = _boot_usable(rec)
    if lab:
        return lab, note, {}
    if phases_doc is None:
        return NOT_RUN, "phase artifact not supplied", {}
    obs, problems = {}, []
    for phase in expectations["phases"]:
        pid = phase["id"]
        seg = _seg(rec, f"p1e_{pid}")
        pdoc = (phases_doc.get("phases") or {}).get(pid)
        if seg is None or pdoc is None:
            return EXTRACTION_EMPTY, f"phase {pid} window missing from the boot record", obs
        if pdoc.get("length_mismatch"):
            return (
                EXTRACTION_EMPTY,
                f"phase {pid}: the engine served prompt lengths "
                f"{pdoc.get('served_lens')} but the registered expectations are "
                f"for {pdoc.get('requested_lens')}; re-derive before scoring",
                obs,
            )
        got = {
            "n_requests_chunked": seg["n_requests_chunked"],
            "n_chunk_events": seg["n_chunk_events"],
            "n_chunk_events_admission": seg["n_chunk_events_admission"],
            "n_chunk_events_continuation": seg["n_chunk_events_continuation"],
            "n_prompts_longer_than_cps": pdoc.get("n_prompts_longer_than_cps"),
            "n_prefill_forwards": seg["extend_tokens_per_forward"]["n_forwards"],
            "sum_extend_tokens": seg["extend_tokens_per_forward"]["sum"],
            "max_extend_tokens": seg["extend_tokens_per_forward"]["max"],
        }
        obs[pid] = {"expected": phase["expected"], "observed": got}
        if "gpu" in phase.get("exact_on", []):
            for k, want in phase["expected"].items():
                if k == "extend_tokens_per_forward_multiset" or k == "mean_extend_tokens":
                    continue
                if got.get(k) != want:
                    problems.append(f"phase {pid}: {k} expected {want}, got {got.get(k)}")
        else:
            g = phase.get("gpu_assertions", {})
            if got["n_prompts_longer_than_cps"] != g["n_prompts_longer_than_cps"]:
                problems.append(f"phase {pid}: the burst was not all-short")
            if got["n_requests_chunked"] < g["n_requests_chunked_min"]:
                problems.append(
                    f"phase {pid}: n_requests_chunked {got['n_requests_chunked']} < "
                    f"{g['n_requests_chunked_min']} (no backlog built, or the "
                    "instrument counts prompt length)"
                )
            if got["max_extend_tokens"] is not None and (
                got["max_extend_tokens"] > g["max_extend_tokens_max"]
            ):
                return ENGINE_BUDGET_EXCEEDED, f"phase {pid} exceeded the cps budget", obs

    d1 = sum(obs[p]["observed"]["n_requests_chunked"] for p in obs)
    d2 = sum(obs[p]["observed"]["n_chunk_events"] for p in obs)
    d3 = sum(obs[p]["observed"]["n_prompts_longer_than_cps"] for p in obs)
    obs["combined"] = {
        "n_requests_chunked": d1, "n_chunk_events": d2,
        "n_prompts_longer_than_cps": d3,
        "registered": expectations["separation_claim"]["combined_boot_S_plus_B"],
    }
    if len({d1, d2, d3}) != 3:
        problems.append(
            f"the three candidate definitions collided at ({d1}, {d2}, {d3}); "
            "this workload no longer separates them"
        )
    if problems:
        return FAIL, " | ".join(problems), obs
    return PASS, "", obs


def _paired_t_ci(diffs, conf=0.95):
    """Paired t CI.  prereg section 4.2 registers paired t over percentile
    bootstrap, because PROJECT_STATUS.md "다음 실험 gate" #8 recorded the
    bootstrap's undercoverage and prescribed replacing the estimator, not
    growing n.  With n = 3 this rests on a normality assumption and is reported
    as such."""
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    return {"n": n, "mean": mean, "lo": mean - tcrit * se, "hi": mean + tcrit * se,
            "t_crit": tcrit, "assumption": "normality; n is small"}


def check_p1f(doc):
    """Performance neutrality: paired ON/OFF throughput within the margin."""
    if doc is None:
        return NOT_RUN, "throughput artifact not supplied", {}
    pairs = doc.get("pairs") or []
    usable = [p for p in pairs if p.get("on") and p.get("off")]
    if len(usable) < 3:
        return (
            EXTRACTION_EMPTY,
            f"{len(usable)} usable paired boots, the prereg registers >= 3",
            {"n_pairs": len(usable)},
        )
    rel = [(p["on"] - p["off"]) / p["off"] for p in usable]
    ci = _paired_t_ci(rel)
    obs = {"n_pairs": len(usable), "rel_diff": rel, "mean_rel_diff": ci["mean"],
           "ci": ci, "margin": NEUTRALITY_MARGIN}
    if abs(ci["mean"]) >= NEUTRALITY_MARGIN:
        return PERTURBED, (
            f"mean paired throughput difference {ci['mean']:+.3%} is outside the "
            f"registered +/-{NEUTRALITY_MARGIN:.0%} margin"
        ), obs
    return PASS, "", obs


def check_p1g(records):
    """The piecewise banner must be on the record for every scored boot."""
    if not records:
        return NOT_RUN, "no boot records supplied", {}
    fields = ("disable_piecewise_cuda_graph", "piecewise_cuda_graph_tokens")
    obs, missing = {}, []
    for name, rec in records.items():
        if rec is None:
            continue
        got = {f: rec.get(f) for f in fields}
        got["piecewise_cuda_graph_max_tokens"] = rec.get("piecewise_cuda_graph_max_tokens")
        got["requested_chunked_prefill_size"] = rec.get("requested_chunked_prefill_size")
        obs[name] = got
        for f in fields:
            if rec.get(f) is None:
                missing.append(f"{name}.{f}")
        toks = rec.get("piecewise_cuda_graph_tokens") or {}
        if toks.get("len") is None:
            missing.append(f"{name}.piecewise_cuda_graph_tokens.len")
    if missing:
        return EXTRACTION_EMPTY, "missing banner fields: " + ", ".join(missing), obs
    # Derived observation, reported and NOT turned into a pass/fail of something
    # else: cps -1 is expected to empty the capture list, which is exactly why
    # the counters must not live in a graph-captured prefill path.
    obs["_note"] = (
        "the arm with requested cps -1 is expected to show an empty piecewise "
        "capture list (server_args.py:1254-1259 -> :1397-1415 -> "
        "model_executor/model_runner.py:2486-2491); the counters are in the scheduler layer so "
        "this cannot make the negative control pass for the wrong reason"
    )
    return PASS, "", obs


def measurement_verdict(labels):
    """Axis 1: the prereg section 2.3 ladder, verbatim.

    ``FAIL`` is deliberately absent: it belongs to axis 2.  In rule rev 1 it
    fell into this ladder's ``else`` branch and was therefore invisible whenever
    any other test carried a section 2.3 label.
    """
    if labels & {ENGINE_BUDGET_EXCEEDED}:
        return "P1_BLOCKED_ENGINE_FINDING"
    if labels & {PERTURBED}:
        return "P1_BLOCKED_NOT_AN_OBSERVER"
    if labels & {EXTRACTION_EMPTY}:
        return "P1_BLOCKED_INSTRUMENT"
    if labels & {BOOT_FAILED}:
        return "P1_INCONCLUSIVE_BOOT"
    if labels & {NOT_RUN}:
        return "P1_INCOMPLETE"
    return MEASUREMENT_CLEAN


def expectation_verdict(labels):
    """Axis 2: did the registered predicates come out true where they ran?

    A test is *evaluated* only when it reached its predicate, i.e. its label is
    PASS or FAIL.  A run in which every test was blocked says nothing about the
    expectations, and must not be reported as if the expectations held.
    """
    if FAIL in labels:
        return EXPECTATION_REFUTED
    if labels == {PASS}:
        return EXPECTATION_HELD
    if PASS in labels:
        return EXPECTATION_HELD_WHERE_EVALUATED
    return EXPECTATION_UNEVALUATED


def compose_verdict(measurement, expectation):
    """The product of the two axes.  Neither may absorb the other."""
    if measurement == MEASUREMENT_CLEAN and expectation == EXPECTATION_HELD:
        return "P1_ACCEPTED"
    if measurement == MEASUREMENT_CLEAN and expectation == EXPECTATION_REFUTED:
        # the rule rev 1 name for "the measurement was fine and the registered
        # expectation is false", kept so the vocabulary does not fork
        return "P1_REJECTED"
    parts = []
    if measurement != MEASUREMENT_CLEAN:
        parts.append(measurement)
    if expectation not in (EXPECTATION_HELD, EXPECTATION_HELD_WHERE_EVALUATED):
        parts.append(expectation)
    if not parts:
        # Unreachable today: a clean measurement means every label is PASS or
        # FAIL, so the expectation axis is HELD or REFUTED, both handled above.
        # Spelled out anyway, because a future label must never be able to fall
        # through into a silent "P1_ACCEPTED".
        return f"{measurement} AND {expectation}"
    return " AND ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expectations", required=True)
    ap.add_argument("--boot", action="append", metavar="NAME=PATH",
                    help="boot record from analyze_chunk_probe.py "
                         "(names: cps512, mono, cps4096)")
    ap.add_argument("--phase", action="append", metavar="NAME=PATH",
                    help="workload artifact (names: p1e, d)")
    ap.add_argument("--greedy-on", default=None)
    ap.add_argument("--greedy-off", default=None)
    ap.add_argument("--throughput", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    exp = json.loads(Path(args.expectations).read_text())
    boots = _kv(args.boot)
    phases = _kv(args.phase)
    greedy_on = json.loads(Path(args.greedy_on).read_text()) if args.greedy_on else None
    greedy_off = json.loads(Path(args.greedy_off).read_text()) if args.greedy_off else None
    thr = json.loads(Path(args.throughput).read_text()) if args.throughput else None

    tests = {}
    for tid, fn in (
        ("P1-a", lambda: check_p1a(boots.get("cps512"))),
        ("P1-b", lambda: check_p1b(boots.get("mono"))),
        ("P1-c", lambda: check_p1c(greedy_on, greedy_off)),
        ("P1-d", lambda: check_p1d(boots.get("cps4096"), phases.get("d"), exp)),
        ("P1-e", lambda: check_p1e(boots.get("cps512"), phases.get("p1e"), exp)),
        ("P1-f", lambda: check_p1f(thr)),
        ("P1-g", lambda: check_p1g(boots)),
    ):
        label, note, obs = fn()
        tests[tid] = {"label": label, "note": note, "observed": obs}

    labels = {t["label"] for t in tests.values()}
    verdict_measurement = measurement_verdict(labels)
    verdict_expectation = expectation_verdict(labels)
    verdict = compose_verdict(verdict_measurement, verdict_expectation)

    out = {
        "schema": SCHEMA,
        "rule_rev": RULE_REV,
        "verdict": verdict,
        "verdict_measurement": verdict_measurement,
        "verdict_expectation": verdict_expectation,
        "refuted_tests": sorted(
            tid for tid, t in tests.items() if t["label"] == FAIL
        ),
        "tests": tests,
        "reminder": (
            "P1 is instrumentation, not an experiment.  A P1 verdict says "
            "nothing about whether chunked prefill helps or hurts "
            "(PREREG_CP0_2026-08-28.md section 9), and a P1 failure does not "
            "kill the chunked-prefill axis (PROJECT_STATUS.md \"방법론 게이트\" #21). "
            "Read BOTH axes: verdict_measurement answers 'did the measurement "
            "happen', verdict_expectation answers 'is the registered "
            "expectation true'.  A blocked measurement never means the "
            "expectations held."
        ),
    }
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0 if verdict == "P1_ACCEPTED" else 4


if __name__ == "__main__":
    sys.exit(main())
