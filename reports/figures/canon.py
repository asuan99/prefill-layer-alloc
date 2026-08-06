#!/usr/bin/env python3
"""Canonical numbers for the prefill-layer-alloc research arc.

Every entry carries `src` (canonical document that owns it) and `grade`.
`grade` uses the project's own evidence vocabulary:

    serving      real-engine end-to-end serving measurement
    micro        forward-level / probe measurement, does NOT predict serving
    retracted    was published as a result, later withdrawn
    barred       measured, but citation-barred (instrumentation defect / confound)
    unaudited    measured, claims-auditor has not passed it -- canon citation banned
    scoped       confirmed only inside a stated scope

Nothing here is derived. Values are transcribed from the canon; the module-level
`SRC` map names the file each block came from so a figure can print provenance.
"""
from __future__ import annotations

# ── provenance ────────────────────────────────────────────────────────────────
SRC = {
    "consensus": "reports/CONSENSUS.md",
    "status": "PROJECT_STATUS.md",
    "arc": "reports/research_arc.md",
    "pos": "reports/layertype_dynamic_POSITIVE_2026-08-04.md",
    "neg": "reports/layertype_dynamic_NEGATIVE_2026-08-04.md",
    "c2": "workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md",
    "pf": "workspace/engine-port/results/s8p_prefill/FINDINGS_PREFILL_2026-07-29.md",
    "s2": "workspace/engine-port/results/s2_sticky/S2_ANALYSIS_2026-08-04.md",
}

SUBSTRATE = (
    "A100 80GB PCIe x1 - 108 SM green context - driver 580.105.08 / CUDA 13.0 - "
    "torch 2.9.1+cu130 - sglang v0.5.10 lineage (editable) - bf16"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. The arc: how the founding hypothesis died, stage by stage
# ═══════════════════════════════════════════════════════════════════════════════
# status: dead | reopened | reversed | open  (drives the marker in fig 1)
ARC = [
    dict(stage="S0", date="2026-07-05", axis="layer-type",
         title="decode-side layer-aware, 4-model serving",
         verdict="dead", grade="serving",
         number="agnostic 4/4 wins - Zamba2 goodput 0 from rate 3",
         why="sim/micro did not predict the serving batch"),
    dict(stage="S1", date="2026-07-06", axis="layer-type",
         title="'layer-aware is a binary strawman' (graduated per-type map)",
         verdict="reopened", grade="serving",
         number="coordinated 54 SM 42 ms vs uncoordinated 54 SM 121 ms",
         why="what was refuted was my measurement, not the hypothesis"),
    dict(stage="S2", date="2026-07-07", axis="layer-type",
         title="coordinated per-type layer-aware, actually implemented",
         verdict="dead", grade="serving",
         number="decode TPOT 42 -> 124 ms",
         why="(D) granularity: one decode step shatters into 19 windows"),
    dict(stage="S3", date="2026-07-13", axis="layer-type",
         title="prefill-side + (B,L) knee + sim reservation",
         verdict="dead", grade="micro",
         number="Diff B ~ 1.0 across the grid",
         why="mamba SSD-prefill is compute-bound too - cost differs, lever does not"),
    dict(stage="S5-S8", date="2026-07-15", axis="time",
         title="last refuge: SLO-aware dynamic PD-split (Step D/E/F, HE2)",
         verdict="dead", grade="serving",
         number="d16 5.81+-0.36 > bind-fine 5.35+-0.73",
         why="saturation slack is split-coupled - reactive control is powerless"),
    dict(stage="S9", date="2026-07-16", axis="time",
         title="real ShareGPT trace overturns 'optimum = d16, invariant'",
         verdict="reversed", grade="serving",
         number="d16 TTFT 7.24 s at 92 prefill SM vs d24 1.21 s",
         why="entanglement - prefill dies of decode's SM, not its own"),
    dict(stage="S10", date="2026-07-17", axis="time",
         title="n>=4 + noise hunt: HE0 confirmed robust",
         verdict="dead", grade="serving",
         number="d44 3.220+-0.013 > bind+GATE 3.132+-0.019 (5.4 sigma)",
         why="the two regimes' optima do not conflict - nothing to adapt to"),
    dict(stage="S11", date="2026-07-18", axis="time",
         title="re-score under tight SLO appears to revive dynamic",
         verdict="reversed", grade="retracted",
         number="apparent win at 0.335 s SLO - later REFUTED",
         why="re-scoring a 3s-tuned controller is post-hoc; retuning killed it"),
    dict(stage="V1", date="2026-07-25", axis="time",
         title="vector-1: short-ctx disjoint escape hatch",
         verdict="dead", grade="serving",
         number="companion collapse - d54 covers both phases (0.948 vs d44 0.932)",
         why="no conflict regime exists in the short-ctx band"),
    dict(stage="C2", date="2026-07-28", axis="space",
         title="8B decode-SM lever exists at the operating point",
         verdict="open", grade="scoped",
         number="ITL p50 2.36-2.91x over decode SM 16->92, 4 arms alike",
         why="iso-curve: establishes the lever, NOT a policy gain"),
    dict(stage="S0-ax", date="2026-08-03", axis="space",
         title="the 2.6x axis contradiction (C2 28.79 vs E1 11.09 ms)",
         verdict="open", grade="barred",
         number="same arm, same flags, matched batch - 2.6x apart",
         why="the floor everything downstream stands on"),
    dict(stage="S2s", date="2026-08-04", axis="space",
         title="sticky partition resolves the axis contradiction",
         verdict="reversed", grade="serving",
         number="d16 label sat at D108 96.2% of the time; sticky ON -> 99.90% at D16",
         why="the 2.6x was label dilution - the partition was never held"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Layer-aware death - the decisive TPOT numbers  (E2, Zamba2-2.7B)
# ═══════════════════════════════════════════════════════════════════════════════
LAYER_AWARE_TPOT = dict(
    src=SRC["consensus"] + " s1-3/s1-15 - " + SRC["arc"] + " S1/S2",
    grade="serving, no-cudagraph substrate",
    unit="decode TPOT (ms)",
    bars=[  # all four at decode = 54 SM, so the bars differ only in HOW it was run
        ("agnostic\ncoordinated", 42.0, "baseline"),
        ("layer-aware\nuncoordinated", 121.0, "artifact"),
        ("layer-aware\ncoordinated", 124.0, "verdict"),
        ("+ COORD_OPT\nwait_stream, no pin", 85.0, "recovered"),
    ],
    note=("S1 blamed the 121 ms on my own uncoordinated implementation and reopened "
          "the hypothesis. S2 built the coordinated version -- and it landed at 124 ms. "
          "COORD_OPT recovers 47% of the gap; the sign is robust."),
)

# 4-model layer-aware vs agnostic collapse is read from raw logs (extract_raw.py).

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HE0 - dynamic control never beats the best static
# ═══════════════════════════════════════════════════════════════════════════════
# Varying trace (ShareGPT, rate 3<->12, 3 rounds, durations SUMMED), cudagraph ON.
HE0_LENIENT = dict(
    src=SRC["neg"] + " s2.1 - " + SRC["consensus"] + " s0",
    grade="serving, n>=4",
    slo="TTFT <= 3 s (lenient)",
    unit="TRUE goodput (req/s)",
    rows=[  # label, mean, sd (None = not reported), n, kind
        ("d44  (decode-heavy static)", 3.220, 0.013, 4, "static"),
        ("d34  (static)", 3.171, 0.025, 4, "static"),
        ("bind + GATE  (dynamic)", 3.132, 0.019, 9, "dynamic"),
        ("d24  (static)", 3.081, None, 4, "static"),
        ("slo-aware  (dynamic)", 2.974, None, 4, "dynamic"),
        ("bind, no gate  (dynamic)", 2.934, 0.306, 4, "dynamic"),
        ("d16  (prefill-heavy static)", 2.846, None, 4, "static"),
    ],
    sigma="5.4 sigma between d44 and bind+GATE",
    note=("A canon discrepancy travels with this table: research_arc.md S10 lists "
          "d16 = 2.817 where NEGATIVE s2.1 and POSITIVE s2.3 list 2.846. "
          "2.846 is used here (the more recent, twice-repeated value)."),
)

HE0_TIGHT = dict(
    src=SRC["neg"] + " s2.1 - " + SRC["consensus"] + " s1-17",
    grade="serving, controller RETUNED for this SLO (not re-scored)",
    slo="chat SLO: TTFT 300 ms AND ITL 50 ms, rate 8",
    unit="SLO attainment (%)",
    rows=[("d44", 73.2, "static"), ("d34", 49.6, "static"),
          ("bind + GATE", 44.3, "dynamic"), ("bind", 40.6, "dynamic")],
    sigma="28.9 pp gap ~ 10 sigma",
    note=("s1-16 once read tight-SLO re-scoring as a dynamic win. s1-17 refuted it: "
          "re-scoring post-hoc grades where a 3s-tuned controller happened to settle. "
          "Methodology gate: change the SLO -> retune and measure, never re-score."),
)

# ── what the three "dynamic" arms in the HE0 chart actually are ───────────────
# All three are the SAME mechanism -- per prefill-layer-span, move the green-context
# split index by +-1 -- and differ ONLY in the decision rule. Source of truth is the
# engine itself, not a report.
CTRL_SRC = "workspace/engine-port/src/multiplex/multiplexing_mixin.py"

CONTROLLERS = [
    dict(
        id="slo-aware", full="SLO-aware (v7b)", fn="_slo_decide_idx", line=807,
        env="(default path)",
        signal="TPOT-EMA (decode latency) is PRIMARY; prefill backlog "
               "len(waiting_queue) is a GATED secondary",
        rule="tpot > 0.85·SLO -> +1 (more decode SM)  ·  "
             "tpot < 0.65·SLO AND qdepth > 4 -> −1 (more prefill SM)  ·  else hold",
        broke="Design flaw: at the cudagraph operating point TPOT has huge headroom and "
              "is NON-BINDING, so the prefill path almost never un-gates. The controller "
              "reads 'fine' (TPOT median 38.7 ms) while TTFT collapses to 6–10 s.",
        goodput=2.974,
    ),
    dict(
        id="bind", full="binding-first dual-slack (Step E + F)",
        fn="_slo_decide_idx_binding", line=726,
        env="PDMUX_SLO_MODE=binding",
        signal="prefill-slack = (TTFT_SLO − oldest-waiting age)/TTFT_SLO  and  "
               "decode-slack = (TPOT_SLO − TPOT-EMA)/TPOT_SLO, as PEERS",
        rule="push toward whichever slack is smaller AND urgent  ·  both slack -> drift "
             "to a tuned-static ANCHOR  ·  saturation -> hold at anchor and latch (Step F)",
        broke="Fixes the gating flaw and still loses. Being reactive + symmetric, it hands "
              "SM back to prefill the moment decode looks briefly fine, so it can never "
              "accumulate at the decode-heavy optimum. It oscillates at dec_sm 16–24 "
              "(mean 22) and NEVER reaches 44.",
        goodput=2.934,
    ),
    dict(
        id="bind + GATE", full="binding-first + Step G feasibility gate",
        fn="_slo_feasible", line=681,
        env="PDMUX_SLO_MODE=binding + PDMUX_SLO_FEAS_GATE=1",
        signal="same as bind, plus an ASYMMETRIC admission check on the move itself",
        rule="gates ONLY prefill-ward moves (decode SM down). Two guards, both must pass: "
             "(1) CONGESTION — refuse if running batch ≥ 0.85·max_running_requests; "
             "(2) ITL — predicted ITL at the candidate split < 0.9·TPOT_SLO",
        broke="Not intelligent control at all: a ONE-WAY RATCHET auto-tuner. It made one "
              "move (d24→d34) and then refused the prefill-ward return 113 times in a row "
              "(bs=47 ≥ 0.85×48 was permanently true) ⇒ frozen at d34 forever, and d34 is "
              "the WRONG static (the optimum is d44).",
        goodput=3.132,
    ),
]

# Why the gate is asymmetric, quoted from the guard's own docstring: starving decode
# PROPAGATES, starving prefill does not.
GATE_RATIONALE = dict(
    src=CTRL_SRC + ":681-716",
    asymmetry=("Starving decode propagates — residency up -> running batch saturates -> "
               "prefill admission blocked -> TTFT explodes (the entanglement trap). "
               "Starving prefill does not propagate back into decode. So only "
               "prefill-ward moves are gated."),
    littles_law=[("d44", 42, 0.12), ("d24", 50, 1.21), ("d16", 57, 7.24)],
    littles_note=("required_concurrency ≈ arrival_rate × output_len × ITL; when it reaches "
                  "max_running_requests the batch caps admission and TTFT collapses. "
                  "⚠ these TTFT magnitudes are from the RETIRED stationary bench (n=1) and "
                  "are citation-barred — they are shown as the guard's design rationale, "
                  "as recorded in its own docstring, not as a measurement."),
    why_congestion_primary=("The congestion guard is the one that actually catches the trap: "
                            "d16's ITL (33.5 ms p50) PASSES the 60 ms SLO, so an ITL-only "
                            "gate would have let the fatal move straight through."),
)

# The elimination that located the real cause.
DEATH_CAUSES = [
    dict(cause="switch overhead", verdict="refuted", src="CONSENSUS §1-8",
         evidence="a rep matched static with only 2 switches; and slo (5 switches) "
                  "scores BELOW bind (21 switches) — more switching is not the loser"),
    dict(cause="controller CPU cost", verdict="refuted", src="CONSENSUS §1-12",
         evidence="measured, not inferred: mean 32–36 µs per call, max 267 µs, "
                  "~34 ms cumulative = 0.9% of one decode step, 0.014% of wall clock"),
    dict(cause="positioning", verdict="THE CAUSE", src="CONSENSUS §1-17",
         evidence="the optimum is dec_sm 44 and the controller oscillates at 16–24 "
                  "(mean 22) — decode-STARVED. risk/reward 18:1: a prefill-ward move "
                  "gains ≤2.3% at low load and costs ≤41.5% at high load"),
    dict(cause="entanglement", verdict="the trap it falls into", src="CONSENSUS §1-4",
         evidence="that oscillation band is exactly the catastrophic band — "
                  "decode starvation -> ITL up -> batch stalls -> admission blocked"),
]

# Static goodput vs decode SM, and where each controller actually SAT.
# Same varying-trace campaign as HE0_LENIENT, so the two are directly comparable.
POSITIONING = dict(
    src=SRC["consensus"] + " §1-10/§1-17 · " + SRC["neg"] + " §2.2/§2.5",
    grade="serving, n>=4",
    unit="TRUE goodput (req/s)",
    static=[(16, 2.846), (24, 3.081), (34, 3.171), (44, 3.220)],
    optimum=44,
    realized=[
        # label, dwell band (lo, hi) or None, representative dec_sm, goodput, sd, note
        ("bind  (no gate)", (16, 24), 22, 2.934, 0.306,
         "oscillates, never reaches 44"),
        ("bind + GATE", None, 34, 3.132, 0.019,
         "1 move, then 113 refusals — frozen"),
    ],
    settle_cost=("bind+GATE 3.132 ≈ static d34 3.171 − 0.039. That 0.039 is the settling "
                 "cost: the dynamic arm scores BELOW the static split it froze at."),
    ceiling=("Every reactive fix — anchor, asymmetric penalty, stopping the moves — "
             "converges on 'just sit at 44', i.e. on the static. So the dynamic ceiling "
             "is a TIE with the best static, and §1-13 shows no regime where it could "
             "win. Footnote (claims-auditor): 'every fix' means the REACTIVE "
             "family; online feedforward was tested (Step D, HD0, n=3, underpowered) and "
             "only OFFLINE model-profile decode-floor prediction remains untested."),
)


# Why: the two regimes' optima do not conflict.
REGIME_SPLIT = dict(
    src=SRC["neg"] + " s2.3 - " + SRC["arc"] + " S10-**",
    grade="serving",
    unit="goodput (req/s)",
    phases=("LO (rate 3)", "HI (rate 12)"),
    rows=[("d44  (decode-heavy)", 2.858, 0.005, 3.924, 0.039),
          ("d16  (prefill-heavy)", 2.861, None, 2.737, None)],
    spread=[("LO (rate 3)", 0.067, 2.3), ("HI (rate 12)", 1.187, 43.0)],
    note=("~95% of the discrimination comes from the overload phase; the low-load "
          "phase is entirely indifferent to the split. HI's optimum is free at LO "
          "=> 'always use the HI optimum' is best by definition, and dynamic pays "
          "only transit. Dynamic needs the optima to CONFLICT. They never do here. "
          "Caveat: LO goodput is CEILING-CENSORED by the arrival rate "
          "(199-200/200 served) -- an indifferent metric, not an absent lever."),
)

# Escape hatches, all four sealed.
ESCAPE_HATCHES = [
    dict(stage="mix-swing trace", src=SRC["consensus"] + " s1-18",
         result="optimum swings only d24<->d34; middle static covers both",
         num="d24 3.930  >>  bind 3.419", verdict="sealed"),
    dict(stage="extreme disjoint", src=SRC["consensus"] + " s1-19",
         result="disjoint region is REAL, and dynamic still loses",
         num="ORACLE dynamic +2.1% - reactive bind -20.6%", verdict="sealed"),
    dict(stage="oracle reconstruction", src=SRC["consensus"] + " s1-20",
         result="real headroom needs 116 SM > 108 => impossible on one GPU",
         num="disaggregation +16% (n=1~2, UNCONFIRMED) - coupled ceiling +2%",
         verdict="sealed (evidence downgraded)"),
    dict(stage="vector-1 short-ctx", src=SRC["status"] + " vector-1",
         result="companion collapse - a single static (d54) covers both phases",
         num="rate 3.75: d54 0.948+-0.062 >= d44 0.932+-0.042", verdict="sealed"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. C2 - the strongest surviving positive: decode-SM lever at the operating point
# ═══════════════════════════════════════════════════════════════════════════════
C2 = dict(
    src=SRC["c2"],
    grade="CONFIRMED (scoped)",
    scope=("4 arms at 7-8B, A100 TP1, cudagraph-ON, --disable-overlap-schedule "
           "--chunked-prefill-size -1 --disable-radix-cache, cap 48, ctx1024, "
           "conc16, out512, n=4"),
    unit="decode ITL p50 (ms)",
    sm=[16, 24, 44, 92, 108],
    # batch=1, partition- and batch-matched intervals. None = cell too sparse.
    arms={
        "Ha8  Zamba2-7B (additive hybrid)":      [50.79, 36.78, 25.76, 21.95, 20.03],
        "M8   Mamba-Codestral-7B (pure SSM)":    [34.80, 25.61, 17.69, 14.14, None],
        "Hs8  Nemotron-H-8B (substitutive)":     [32.86, 23.91, 16.63, 13.37, 11.91],
        "T8   Qwen2.5-7B (pure Transformer)":    [29.93, 21.17, 14.64, 12.09, 10.75],
    },
    headline={  # clean prefill-fixed slice, SM16/SM92
        "M8": (58.44, 20.09, 2.91), "Ha8": (89.80, 31.49, 2.85),
        "Hs8": (40.63, 15.73, 2.58), "T8": (30.20, 12.77, 2.36),
    },
    # MANDATORY rider added by claims-auditor 2026-08-04.
    local_elasticity={"16->24": (0.77, 0.88), "44->92": (0.09, 0.35)},
    citation_rule=("Cite the range only. SM16->SM108 is BANNED (that cell removes the "
                   "split entirely, moving prefill too). Iso-curve: establishes the "
                   "lever's existence, NOT a policy gain, and does NOT revive HE0. "
                   "2.36-2.91x is an ENDPOINT ratio -- local elasticity differs 4x "
                   "between 16->24 and 44->92, so do NOT apply it above D=44. "
                   "Do not transplant to another grid."),
)

# Mirror campaign on the prefill axis -- decode pinned at 16 SM.
PREFILL_AXIS = dict(
    src=SRC["pf"],
    grade="REAL (scoped), UNAUDITED - canon citation banned",
    unit="TTFT ~ L slope (us / token)",
    sm=[16, 24, 44, 92],
    arms={
        "Ha8  Zamba2-7B":        [1059.16, 717.85, 398.95, 211.15],
        "M8   Mamba-Codestral":  [842.32, 563.51, 324.91, 163.14],
        "Hs8  Nemotron-H-8B":    [536.68, 360.07, 201.47, 110.43],
        "T8   Qwen2.5-7B":       [408.74, 275.97, 146.03, 86.24],
    },
    ratio={"M8": 5.163, "Ha8": 5.016, "Hs8": 4.860, "T8": 4.740},
    elasticity_16_92={"M8": 0.938, "Ha8": 0.922, "Hs8": 0.904, "T8": 0.890},
    gate={"M8": "4/4 PASS", "Ha8": "p16 FAIL", "Hs8": "p44,p92 FAIL", "T8": "p44,p92 FAIL"},
    refuted=("The tempting inference -- 'prefill slope (5.0x) > decode slope (2.4-2.9x), "
             "so prefill is the steeper axis at the frontier' -- is NOT supported. "
             "The local elasticity STRUCTURE differs: prefill epsilon is near-constant "
             "(0.71-1.05) while decode epsilon varies 9x (0.09-0.88, saturating above "
             "D~44). The sign of moving one SM flips with the operating point. "
             "Tension A closes only by direct measurement."),
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. The retraction that reshaped the mechanism story: Diff B
# ═══════════════════════════════════════════════════════════════════════════════
DIFF_B = dict(
    src=SRC["pos"] + " s1.3 - " + SRC["neg"] + " s1.4",
    grade="micro, no-cudagraph, n_indep = 1 per cell",
    unit="Diff B  =  attn / mamba SM-sensitivity ratio",
    L=[256, 512, 1024, 4096, 8192, 16384, 32768],
    series={
        "as reported (kernel unit)": [1.420, 1.219, 0.905, 1.032, 1.083, 1.070, None],
        "steady, defects removed":   [1.336, 1.198, 0.904, 1.011, 1.036, 1.022, 1.017],
        "policy unit (R_policy)":    [1.032, 1.031, 0.971, 1.007, 1.026, 1.019, 1.016],
    },
    defects=("(1) asymmetric buckets -- _zt('attn') times the RadixAttention core only "
             "while _zt('mamba') times the whole mixer; (2) the _zt_acc accumulator is "
             "never reset and the harness takes tail -1, so every reported value "
             "permanently includes cold start, and SM_LIST=(full 44 24 16 8) puts that "
             "bias systematically in the denominator."),
    kill=("'The lever opens at short L' is RETRACTED. Two independent jobs disagree "
          "4.5% on the reported value but agree to 0.08% on the steady value -- the "
          "1.32-vs-1.38 difference was how many times it had run, not physics. "
          "In the policy unit the lever vanishes entirely (0.97-1.03 flat)."),
    policy_note=("R_policy ~ 1 + w_attn*(DiffB - 1) with w_attn = 9.6%: the policy unit "
                 "is structurally pulled to 1 and is NOT independent evidence."),
    band_256=(1.24, 1.34),   # L=256 may ONLY be cited as this band, never a point
)

# ── the three Diff B series are NOT three measurements of the same thing ──────
# They differ along TWO ORTHOGONAL axes, applied in order:
#   as reported --(1) clean the SAMPLE--> steady --(2) change the UNIT--> policy
# Conflating the two is what made "the lever opens at short L" look confirmed.
DIFF_B_STEPS = [
    dict(
        step=1, frm="as reported (kernel)", to="steady (kernel)",
        axis="SAME estimand, SAME unit — a cleaned SAMPLE",
        what="drop the cold-start block from every cell",
        why=("`_zt_acc` is never reset when a value is emitted and the harness takes "
             "`tail -1`, so every printed number permanently averages in the first "
             "(cold) forward. `SM_LIST=(full 44 24 16 8)` runs `full` first, so that "
             "bias lands systematically in the DENOMINATOR of the ratio."),
        asymmetry=("The bias is not shared evenly. `layers_block_type[0..5]` is mamba, so "
                   "the first six forward events are all mamba, and `_zt('mamba')` "
                   "includes the mixer GEMM while `_zt('attn')` excludes qkv/o_proj — the "
                   "one-time forward-start cost is structurally attributable to mamba "
                   "alone. Result: Diff B is biased UPWARD."),
        proof=("Non-circular: two independent jobs measured the SAME cell. Reported "
               "disagree by 4.5% (111.578 vs 116.648 ms); steady agree to 0.08% "
               "(84.012 vs 83.941). The reported 1.32-vs-1.38 spread was how many times "
               "it had run, not physics."),
        decay="flat within ±0.4% after block 2 — one-shot, not exponential decay",
        effect="L=256  1.420 -> 1.336   ·   L=512  1.219 -> 1.198   ·   "
               "L=2000  1.38 -> 1.0081  [1.0078, 1.0084]",
    ),
    dict(
        step=2, frm="steady (kernel)", to="policy unit (R_policy)",
        axis="SAME sample — a DIFFERENT unit of aggregation",
        what="express the ratio over the module a policy actually moves SM for",
        why=("A kernel-level ratio is not what a policy can pull. Attention is only "
             "w_attn = 9.6% of the hybrid window at L=256, so "
             "R_policy ≈ 1 + w_attn·(DiffB − 1) — a 34% kernel-level gap becomes a "
             "3% module-level gap."),
        asymmetry=("This is the unit the founding hypothesis actually needed: it asked "
                   "for a lever a RUNTIME POLICY could pull, and a policy reallocates "
                   "SM for a whole window, not for one kernel."),
        proof=("Arithmetic, not a second measurement — which is exactly why it is NOT "
               "independent evidence: ~90% of the window is the same mamba mixer that "
               "sits in the denominator, so R_policy is structurally pulled toward 1 "
               "whatever the kernel ratio does."),
        decay="",
        effect="L=256  1.336 -> 1.032   ·   L=512  1.198 -> 1.031   ·   "
               "whole grid 0.97–1.03 flat",
    ),
]

# The mechanism behind step 1, measured: how much the FIRST block inflates each
# layer type (clean WIDE, B=1, SM=108). attn is inflated too -- but 1/20 to 1/2
# as much as mamba, which is why the ratio moves at all.
FIRST_BLOCK_INFLATION = dict(
    src=SRC["neg"] + " §1.4",
    L=[256, 512, 4096, 8192],
    attn=[0.980, 1.201, 1.030, 1.013],
    mamba=[5.100, 2.316, 1.356, 1.381],
    median_note="at reported level the medians are attn 1.0009 vs mamba 1.0215",
    closure=("The old L=2000 cell is the corpus maximum: attn 1.0162 vs mamba 1.3282. "
             "Their quotient 1.3282/1.0162 = 1.307 matches the observed correction "
             "1.3069 — so the whole correction is accounted for by the asymmetry. Had "
             "attn been inflated by the same factor, the quotient would have been 1.000 "
             "and nothing would have moved."),
)

DIFF_A = dict(
    src=SRC["pos"] + " s1.2",
    grade="micro, no-cudagraph",
    old=dict(attn=1.68, mamba=0.61, label="reported (2-point, contaminated endpoints)"),
    new=dict(attn=1.916, attn_se=0.021, mamba=0.954, mamba_se=0.008,
             label="steady OLS, ctx >= 2050, R2 >= 0.9996"),
    note=("Diff A (the COST ratio) is real and spans 21x across L. It was never the "
          "quantity the hypothesis needed -- that was Diff B (the SENSITIVITY ratio). "
          "The whole arc was built on confusing the two."),
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. The metric cliff - why the benchmark itself had to be retired
# ═══════════════════════════════════════════════════════════════════════════════
METRIC_CLIFF = dict(
    src=SRC["arc"] + " S10-* - " + SRC["consensus"] + " s2",
    grade="serving, reproduced by rate sweep",
    perturbation=[("throughput (req/s)", 6.36, 6.14, "3%"),
                  ("mean ITL (ms)", 28.0, 30.2, "8%"),
                  ("goodput (req/s)", 6.317, 3.102, "2x")],
    # good-request counts per repeat, same config, same prompt fingerprint
    stability={"rate 3": [200, 200, 200], "rate 8": [400, 400, 357, 206],
               "rate 12": [142, 141, 142]},
    totals={"rate 3": 200, "rate 8": 400, "rate 12": 400},
    static_scatter=("static d24, switch_count = 0, still scores 5.282 +- 1.302 "
                    "(1 of 4 reps collapses to 3.102)"),
    kill=("The amplifier is that rate 8 sits exactly at TTFT ~ SLO. The overload "
          "queue's TTFT plateau moves 1.5 s -> 3.7 s under a 3% deficit and crosses "
          "the threshold: 400/400 -> 206/400. Goodput was being evaluated at the "
          "steepest point of the CDF. => stationary ShareGPT r8 RETIRED as a policy "
          "benchmark; policy comparison uses the varying trace only."),
)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. The 2026-08 frontier: label dilution, and sticky partition resolving it
# ═══════════════════════════════════════════════════════════════════════════════
STICKY = dict(
    src=SRC["s2"],
    grade="serving, n = 8 blocks per cell, pre-registered",
    residency=[  # cell label -> where the decode partition actually was, time-weighted
        ("872077  sticky OFF", {"D16": 3.8, "D108": 96.2}),
        ("873015  sticky ON", {"D16": 99.90, "D108": 0.10}),
    ],
    ratio=[("872077  sticky OFF", 1.001, 0.002), ("873015  sticky ON", 2.401, 0.008)],
    ratio_label="block-paired d16 / d54 p50 (all-token population)",
    itl=[  # cell, pooled p50, ci_lo, ci_hi, prereg_lo, prereg_hi, hit
        ("T8 d16  (primary)", 28.92, 28.808, 29.020, 28, 34, True),
        ("T8 d54  (companion)", 12.03, 11.964, 12.116, 13, 16, False),
    ],
    c2_anchor=28.79,
    verdict=("PREREG_S2 row 1 fires: the d16 p50 lands inside [28, 34] with realization "
             ">= 0.90 in all 8 blocks. Reading: the 2.6x axis contradiction was LABEL "
             "DILUTION -- in 872077 the 'd16' cell was simply not at 16 SM (D108 96.2% "
             "of decode-busy time). The companion d54 MISSED low (12.03 < 13), which "
             "the pre-registration has no branch for."),
    scope=("Not a throughput, goodput, latency-policy, g, G_LEVER, G_FLAT or decode-SM "
           "elasticity claim. G_LEVER / G_FLAT remain UNDETERMINED."),
)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Evidence ladder + the retraction ledger
# ═══════════════════════════════════════════════════════════════════════════════
CLAIMS = [
    dict(id="A", text="composition / context / load-dependent decode demand",
         grade="partial", src=SRC["status"],
         rider="C2 (scoped) establishes the lever's existence only -- not a policy gain"),
    dict(id="B", text="layer-level reconfiguration damages the critical path",
         grade="strong", src=SRC["status"], rider="this substrate only"),
    dict(id="C", text="decode starvation entangles into TTFT",
         grade="strong/partial", src=SRC["status"],
         rider="running-batch path strong, KV path partial (occupancy = identity)"),
    dict(id="D", text="true dual-worker reduces coupling",
         grade="unverified", src=SRC["status"],
         rider="scoped down to control-plane by code review"),
    dict(id="E", text="hybrid-informed policy beats generic / static",
         grade="unverified", src=SRC["status"],
         rider="implementation complete != performance claim"),
]

# P-items: the positive axis, with 2026-08-04 audit outcomes.
POSITIVES = [
    dict(id="P1", text="PD separation itself is a gain (4 models)",
         grade="scoped", was="confirmed",
         rider="no-cudagraph NON-operating point, n=1/cell, tie at rate 1; "
               "cudagraph removes fused's cause of death (TPOT>60ms) -- may shrink or vanish"),
    dict(id="P2", text="operating point = cudagraph ON",
         grade="confirmed", was="confirmed",
         rider="decode wall removed, TPOT 41 -> 12 ms; all no-cudagraph numbers are lower bounds"),
    dict(id="P3", text="decode-axis layer-type sensitivity difference is real (Diff B ~ 4x)",
         grade="micro", was="micro",
         rider="(C1) lever exists; the same document rejected the policy under (C2)"),
    dict(id="P4", text="the optimum split is a function of load",
         grade="confirmed", was="confirmed",
         rider="direction only -- magnitude citation banned (retired benchmark)"),
    dict(id="P5", text="decode SM demand is predictable from ctx",
         grade="barred", was="confirmed",
         rider="3 instrumentation defects; correction moves ctx256 from 1.140x to 2.513x. "
               "This is Claim E's only mechanism basis"),
    dict(id="P6", text="disaggregation headroom +16% (116 SM > 108)",
         grade="unconfirmed", was="confirmed",
         rider="downgraded: n=1~2, overload-only, underpowered"),
    dict(id="P7", text="entanglement mechanism",
         grade="confirmed", was="confirmed",
         rider="direction confirmed by n>=4 varying trace; '7.24 s at 92 SM' magnitude banned"),
    dict(id="C2", text="decode-SM lever at the operating point, 4 arms",
         grade="confirmed", was="new",
         rider="scoped; iso-curve; endpoint ratio; do not apply above D=44; do not transplant"),
]

RETRACTIONS = [
    ("2026-07-15", "section-B '+18%' gain", "self-retracted",
     "double confound: no-cudagraph AND compared against a non-optimal static"),
    ("2026-07-16", "'optimum split = d16, invariant to load'", "refuted by real trace",
     "was a low-decode-load synthetic artifact"),
    ("2026-07-17", "stationary ShareGPT r8 as a policy benchmark", "benchmark retired",
     "static d24 alone scores 5.282 +- 1.302; the metric sits on a cliff"),
    ("2026-07-17", "'stationary variance = GPU clock throttling'", "unnecessary",
     "the whole effect is the metric cliff"),
    ("2026-07-17", "'bimodal because of the trap' / 'gate recovers performance'", "retracted",
     "baseline scatter was never measured first"),
    ("2026-07-18", "'tight SLO revives dynamic' (s1-16)", "refuted by s1-17",
     "re-scoring a 3s-tuned controller instead of retuning"),
    ("2026-07-24", "R1 dual-worker causal attribution", "retracted",
     "it was an observer-path A/B, not architecture separation"),
    ("2026-07-28", "Stage 0 'decode is SM-insensitive at the operating point'", "refuted",
     "the D108 uncontended anchor was in fact running at decode 16 SM"),
    ("2026-08-03", "g = A_free(d16)/A_free(d54), A_free, the pin gate", "retired",
     "estimand unidentified; the gate was an identity (conditional pin = 1.000)"),
    ("2026-08-03", "'dilution attenuated g' correction model", "refuted",
     "control-arm reductio: the same correction puts T8 at 21-29x, violating C2 tenfold"),
    ("2026-08-03", "C2 -> G_LEVER anchor derivation", "abandoned",
     "only claim 1 survived audit; claims 2-5 REFUTED / NOT-YET-SUPPORTED"),
    ("2026-08-04", "'the lever opens at short L' (Diff B 1.42 / 1.35)", "refuted",
     "asymmetric buckets + non-reset accumulator; steady L=2000 is 1.0081"),
    ("2026-08-04", "'attn share 5% -> 79%', 'ctx256 is SM-free', 'knee 16->44->108'",
     "citation-barred", "same defects + a physics-invariant violation in job 858811"),
    ("2026-08-04", "'C2 elasticity predicts vector-1'", "refuted",
     "used a pooled endpoint elasticity outside its own operating point (2-8x off)"),
    ("2026-08-04", "disaggregation +16% as 'confirmed'", "downgraded",
     "canon itself already recorded the campaign as n=1~2 and underpowered"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Methodology gates -- what the arc actually taught
# ═══════════════════════════════════════════════════════════════════════════════
GATES = [
    "1. A policy claim must be demonstrated in serving. Never promote a sim or "
    "micro-benchmark per-layer gain into a real-engine conclusion.",
    "2. Stationary ShareGPT r8 is retired for policy comparison (static alone "
    "scatters +-1.3). Compare policies on a VARYING trace.",
    "3. Measure baseline variance first. No policy conclusion without n >= 4. "
    "A difference under 3% is not a headline.",
    "4. goodput = request TTFT <= SLO AND in-request token-ITL p95 <= SLO.",
    "5. Dynamic results must ship switch_count, dwell distribution, and "
    "TTFT/ITL p50/p95/p99 together.",
    "6. Threshold indicator metrics must be measured away from the metric cliff. "
    "Measure capacity first.",
    "7. When combining rounds, SUM the durations (max() inflated goodput 3x -- "
    "that was a real harness bug).",
    "8. Changing the SLO means retuning the controller and measuring again -- "
    "never re-scoring the old controller.",
    "9. Verify a pin by the REALIZED partition, not the target (Stage 0 died here).",
    "10. Ask whether the gate is itself an identity, and hang the negative "
    "control on the COMPLEMENT class.",
]
