#!/usr/bin/env python3
"""span_factor(seed, N) -- pre-execution predictor of the realised arrival span.

sglang/bench_serving.py consumes np.random in exactly one place for a run
without LoRA:  :1706 np.random.seed(args.seed)  then  :948
np.random.exponential(1.0 / request_rate) once per emitted request.  The
`--dataset-name random --tokenize-prompt` path consumes no np.random draws
(verified by exhaustive grep of the file), so the arrival stream is the first
N-1 draws of the seeded stream and, because exponential(1/rate) is
(1/rate) * exponential(1) on the SAME uniforms, the normalised span

    span_factor = span / ((N-1)/rate) = mean of the first N-1 unit exponentials

depends only on (seed, N) -- NOT on the rate.  That makes the Q-A execution
cap  lambda <= 0.95 * s_min * mu_p  a deterministic, pre-computable rule.

VALIDATION (2026-09-10): reproduces the 12 realised span coefficients that
RESULT_P7_ITL_SIGMA_2026-09-10.md sec 3 measured at N=90 with mean |error|
0.21 %, max 1.08 % -- an independent 12-sample check of the predictor.

usage:  python3 span_factor.py            # validation + Q-A seed table
"""
import numpy as np

P7_OBSERVED = {41: 0.873, 42: 0.940, 43: 1.067, 44: 1.020, 45: 1.018, 46: 0.852,
               47: 0.994, 48: 0.873, 49: 1.154, 50: 1.118, 51: 0.770, 52: 1.049}


def span_factor(seed: int, n_prompts: int) -> float:
    """Mean of the first n_prompts-1 unit exponentials drawn from `seed`."""
    np.random.seed(seed)
    return float(np.random.exponential(1.0, size=n_prompts - 1).mean())


def validate_against_p7() -> float:
    errs = [abs(span_factor(s, 90) - o) / o * 100.0 for s, o in P7_OBSERVED.items()]
    return float(np.mean(errs)), float(np.max(errs))


if __name__ == "__main__":
    mean_err, max_err = validate_against_p7()
    print(f"P7 validation (N=90, 12 seeds): mean|err| {mean_err:.2f} %  max {max_err:.2f} %")
    for s, o in P7_OBSERVED.items():
        print(f"  seed {s}: pred {span_factor(s, 90):.4f}  obs {o:.3f}")

    ns = [15, 23, 39, 64, 106]           # N = round(lambda * 150) + 1 on Lambda
    rounds = {1: [61, 62, 63], 2: [64, 65, 66], 3: [67, 68, 69], 4: [70, 71, 72]}
    print("\nQ-A seeds -- span_factor by N")
    print("seed " + "".join(f"  N={n:<5}" for n in ns))
    for s in range(61, 73):
        print(f" {s} " + "".join(f"  {span_factor(s, n):.4f} " for n in ns))
    print("\nper-round s_min = min over that round's seeds")
    for r, ss in rounds.items():
        print(f"  R{r}: " + "  ".join(f"N={n}:{min(span_factor(s, n) for s in ss):.4f}" for n in ns))
    print("\n★ spread at N=15 is 0.4964..1.3052 (rel SD of a 14-draw mean is 1/sqrt(14)=27%),")
    print("  so a per-round s_min makes the executed rung set ROUND-DEPENDENT -- see")
    print("  audit_qa_rules_2026-09-10/ and PREREG_QA_REGRET rev3.")
