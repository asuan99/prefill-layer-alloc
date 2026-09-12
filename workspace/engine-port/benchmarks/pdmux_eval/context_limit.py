"""Model context limit + trace context demand -- R2 campaign pre-flight.

WHY THIS EXISTS
---------------
`scripts/r2_eval/engine_bench_runner.sh` boots the server with
`--context-length "${PDMUX_CONTEXT_LENGTH:-16384}"`, and nothing in the campaign
path ever set `PDMUX_CONTEXT_LENGTH`.  Zamba2-2.7B declares
`max_position_embeddings=4096`, so SGLang's `ModelConfig._derive_context_length`
raises (`model_config.py`, "User-specified context_length ... is greater than
the derived context_length") and the server never boots.  `results/r2_eval/` was
therefore never created.

The tempting fix -- `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` -- is
forbidden here: it does not give the model a longer context, it only silences
the check and extrapolates RoPE past what the model was trained on, which
contaminates the served tokens that a serving result is supposed to measure.
Instead we derive the limit from the model's own config and refuse anything
above it.

WHAT THE TWO PREDICATES MEAN (they are different failures)
----------------------------------------------------------
With `max_req_len = min(context_len - 1, pool - 1)` and
`max_req_input_len = max_req_len - 5` (`managers/tp_worker.py`), and
`init_req_max_new_tokens` clamping `max_new_tokens` to
`max_req_len - len(input_ids) - 1` (`managers/scheduler.py`):

  REJECTED      `input_tokens >= context_len - 6`
                the scheduler aborts the request outright.  `trace_loadgen.py`
                swallows the HTTP error into `record["error"]`, so this shows up
                only as a shrunken denominator -- a silent sample loss.

  TRUNCATED     `input_tokens + output_tokens > context_len - 2`
                the request runs but produces FEWER decode steps than the trace
                asked for.  This is the dangerous one: the run "succeeds", its
                ITL/TTFT are recorded, and the decode work an arm was supposed
                to be compared under is quietly different per arm composition.

Both are pre-flight refusals, not warnings.  A campaign that cannot be served
at the model's real context length is an experiment-design question (pick a
model with a longer context, or change the workload), not something a runner
may paper over.

The conservative branch: `pool - 1` is unknown before the server boots, so the
predicates use `context_len` alone.  That makes this check NECESSARY, not
sufficient -- a trace that passes here can still hit the memory-pool bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Mirrors sglang/srt/utils/hf_transformers_utils.py CONTEXT_LENGTH_KEYS, in
# order.  Kept as a copy on purpose: importing sglang here would drag torch into
# a pre-flight that must run on a login node before any GPU is allocated.
# tests/test_r2_eval_runner.py asserts this list still equals the engine's.
CONTEXT_LENGTH_KEYS = (
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
)

# Slack the engine reserves for a request's input (tp_worker.py: max_req_len - 5)
# plus the -1 in max_req_len itself.
INPUT_SLACK = 6
# `init_req_max_new_tokens`: max_new_tokens <= max_req_len - input - 1
#                                            = context_len - input - 2
TOTAL_SLACK = 2

DEFAULT_FALLBACK_CONTEXT = 2048   # hf_transformers_utils.get_context_length tail


def derived_context_length(config: Mapping[str, Any]) -> int:
    """Reproduce `hf_transformers_utils.get_context_length` on a raw config dict.

    Reads the *text* config when the top level is a wrapper, the same way
    `ModelConfig.hf_text_config` does.
    """
    text_config: Mapping[str, Any] = config
    inner = config.get("text_config")
    if isinstance(inner, Mapping) and any(k in inner for k in CONTEXT_LENGTH_KEYS):
        text_config = inner

    rope_scaling = text_config.get("rope_scaling")
    factor = 1.0
    if isinstance(rope_scaling, Mapping):
        factor = float(rope_scaling.get("factor", 1) or 1)
        if "original_max_position_embeddings" in rope_scaling:
            factor = 1.0
        if rope_scaling.get("rope_type") == "llama3":
            factor = 1.0

    for key in CONTEXT_LENGTH_KEYS:
        value = text_config.get(key)
        if value is not None:
            return int(factor * int(value))
    return DEFAULT_FALLBACK_CONTEXT


def load_model_config(model: str) -> Dict[str, Any]:
    """`config.json` for a local path or a hub id, without importing transformers.

    Hub ids resolve through the local HF cache; with `HF_HUB_OFFLINE=1` this is a
    pure cache read, which is what a login-node pre-flight needs.
    """
    local = Path(model) / "config.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download   # lazy: no torch import

    # Cache first.  A compute node may have no route to the hub, and a
    # pre-flight that reaches for the network turns a local question ("how long
    # is this model's context?") into a reason the job dies.  The weights have
    # to be cached for the server to boot anyway, so the config is too.
    try:
        path = hf_hub_download(model, "config.json", local_files_only=True)
    except Exception:                              # noqa: BLE001 - fall back
        path = hf_hub_download(model, "config.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_context_limit(model: str) -> int:
    return derived_context_length(load_model_config(model))


def read_trace_requests(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record") == "request":
            records.append(item)
    return records


def trace_violations(
    path: Path, context_length: int
) -> Tuple[List[str], List[str], int, int]:
    """(rejected_ids, truncated_ids, max_input, max_total) for one trace file."""
    rejected: List[str] = []
    truncated: List[str] = []
    max_input = 0
    max_total = 0
    for item in read_trace_requests(path):
        inp = int(item["input_tokens"])
        out = int(item["output_tokens"])
        rid = str(item.get("request_id", "?"))
        max_input = max(max_input, inp)
        max_total = max(max_total, inp + out)
        if inp >= context_length - INPUT_SLACK:
            rejected.append(rid)
        elif inp + out > context_length - TOTAL_SLACK:
            truncated.append(rid)
    return rejected, truncated, max_input, max_total


def check(
    model: str,
    context_length: Optional[int],
    traces: Sequence[Path],
) -> Tuple[int, List[str]]:
    """Resolve the context length and report every reason the campaign cannot run.

    Returns (resolved_context_length, problems).  An empty `problems` means the
    server will boot at that context length and every listed trace fits.
    """
    limit = model_context_limit(model)
    resolved = limit if context_length is None else int(context_length)
    problems: List[str] = []

    if resolved > limit:
        problems.append(
            f"context-length {resolved} exceeds {model}'s derived context "
            f"length {limit}; SGLang refuses to boot without "
            f"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN, and that override is "
            f"banned here because it extrapolates RoPE past training"
        )
    if resolved < 1:
        problems.append(f"context-length {resolved} is not positive")

    for trace in traces:
        rejected, truncated, max_input, max_total = trace_violations(trace, resolved)
        if rejected:
            problems.append(
                f"{trace.name}: {len(rejected)} request(s) have "
                f"input_tokens >= context_length-{INPUT_SLACK} (max input "
                f"{max_input} vs context {resolved}); the scheduler aborts them "
                f"and trace_loadgen records them only as failed_requests "
                f"(e.g. {rejected[0]})"
            )
        if truncated:
            problems.append(
                f"{trace.name}: {len(truncated)} request(s) have "
                f"input+output > context_length-{TOTAL_SLACK} (max total "
                f"{max_total} vs context {resolved}); the engine would silently "
                f"shorten max_new_tokens, so the decode work differs from the "
                f"trace (e.g. {truncated[0]})"
            )
    return resolved, problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="requested context length; default = the model's derived limit",
    )
    parser.add_argument("--trace", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    try:
        resolved, problems = check(args.model, args.context_length, args.trace)
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        print(f"context pre-flight failed: {exc!r}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"  [REFUSED] {problem}", file=sys.stderr)
    if problems:
        return 2
    print(resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
