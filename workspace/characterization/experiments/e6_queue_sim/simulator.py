"""Iteration-level continuous-batching simulator (+ chunked prefill).

Each iteration runs: one prefill chunk (if a request is prefilling) overlapped with
a decode step for every in-flight decoding request. Iteration time comes from the
measured LatencyModel. See reports/queue_simulator_design.md.

v0 simplifications: one prefill slot at a time (chunked across iterations); single
representative (prefill_layer, decode_layer) kernel pair; FCFS admission.
"""
from __future__ import annotations


def simulate(reqs, policy, lm, pf_layer, dec_layer, ctx,
             max_batch=512, prefill_budget=1):
    """Run the serving loop. Mutates/returns reqs with ttft, itls, done filled.

    prefill_budget = chunks of prefill processed per iteration (Sarathi/vLLM token
    budget / chunk). The step-time LUT must be measured at prefill_batch == budget
    (k chunks ∥ decode), else step costs are wrong. Returns (reqs, stats)."""
    reqs = sorted(reqs, key=lambda r: r.arrival)
    n = len(reqs)
    ai = 0                       # next un-arrived request index
    clock = 0.0
    waiting, decoding = [], []
    prefill = None               # the single request currently prefilling
    done = 0
    notes = {}

    while done < n:
        while ai < n and reqs[ai].arrival <= clock:      # admit arrivals
            waiting.append(reqs[ai]); ai += 1
        if prefill is None and not decoding and not waiting:
            if ai < n:
                clock = reqs[ai].arrival; continue        # idle → jump to next arrival
            break
        if prefill is None and waiting:                   # start a prefill (FCFS)
            prefill = waiting.pop(0)

        B = min(len(decoding), max_batch)
        prefill_active = prefill is not None
        ms, note = lm.step_ms(policy, pf_layer, dec_layer, B, ctx, prefill_active)
        if note:
            notes[note] = notes.get(note, 0) + 1
        clock += ms

        for r in list(decoding):                          # every decoding req emits 1 token
            r.itls.append(ms); r.n_output -= 1
            if r.n_output <= 0:
                r.done = clock; decoding.remove(r); done += 1
        if prefill is not None:                            # prefill advances `budget` chunks
            prefill.n_chunks -= prefill_budget
            if prefill.n_chunks <= 0:
                prefill.ttft = clock - prefill.arrival
                decoding.append(prefill); prefill = None

    stats = {"makespan_ms": clock, "n": n, "notes": notes}
    return reqs, stats
