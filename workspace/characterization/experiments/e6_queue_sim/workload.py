"""Request stream for the queue simulator: Poisson arrivals + prompt/output lengths."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Request:
    id: int
    arrival: float           # arrival time (ms)
    n_chunks: int            # prefill chunks remaining (= ceil(prompt/chunk))
    n_output: int            # decode tokens remaining
    ttft: float = None       # set when prefill completes (ms, relative to arrival)
    done: float = None       # absolute completion time
    itls: list = field(default_factory=list)   # per-token inter-token latencies (ms)


def gen_requests(n, lam_per_ms, prompt_lens, output_lens, chunk, seed=0):
    """n requests, Poisson arrivals at `lam_per_ms` req/ms; lengths sampled from lists."""
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(lam_per_ms)
        pl = rng.choice(prompt_lens)
        ol = rng.choice(output_lens)
        reqs.append(Request(id=i, arrival=t,
                            n_chunks=max(1, math.ceil(pl / chunk)),
                            n_output=max(1, int(ol))))
    return reqs
