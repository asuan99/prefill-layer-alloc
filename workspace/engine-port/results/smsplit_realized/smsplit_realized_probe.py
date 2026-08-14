#!/usr/bin/env python3
"""E-3 -- realized SM counts from the green-context split primitive.

`create_greenctx_stream_by_value` computes two realized SM counts after
`cuDevSmResourceSplitByCount` has rounded the request to hardware granularity
(`greenctx_stream.cu`: `resources[0].sm.smCount` / `resources[1].sm.smCount`),
and returns them.  `sgl_kernel.spatial.create_greenctx_stream_by_value` keeps
`res[0]`/`res[1]` (the two stream pointers) and discards everything after, so
no caller in this project has ever seen them.  This probe calls the op
directly and reads them.

SCOPE (read this before citing any number this emits)

  * The counts are a DRIVER SELF-REPORT of the partition the driver says it
    created.  In the Stage 0 D108 sense they are still target-layer: they
    answer "did the driver round my request", NOT "which physical SMs ran the
    work".  The latter needs an id-set probe (`%smid`), which is a separate,
    separately-audited campaign.
  * The split is two-stage: stage 1 carves `smA + smB` off the device and
    builds a green context; stage 2 splits that context into A and B.  So
    there are two rounding sites, and `realized_a + realized_b` can be less
    than `sm_a + sm_b` even when each half looks reasonable.
  * Nothing here is a performance measurement.

The probe does not assume the installed wheel returns the counts -- older
builds return a 2-element vector.  `len(res)` is recorded and drives the
verdict, so "the build does not expose it" is reported as its own outcome
rather than silently becoming a missing field.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import traceback

import torch


def _targets_from_producer(total_sm: int) -> list[tuple[int, int, str]]:
    """Partitions to probe.

    The pdmux ones come from the producer (`pdmux_context.divide_sm`), not from
    a hardcoded (74, 34): a probe that hardcodes the number it is checking is
    close to an identity (methodology gate #9 -- contrast on the producer).
    """
    out: list[tuple[int, int, str]] = []
    try:
        from sglang.srt.multiplex.pdmux_context import divide_sm

        for prefill_sm, decode_sm in divide_sm(total_sm, (8, 0), 2):
            out.append((int(prefill_sm), int(decode_sm), "divide_sm"))
    except Exception as exc:  # noqa: BLE001
        out.append((-1, -1, f"divide_sm_unavailable: {exc!r}"))

    # The decode-SM sweep points C2 was measured on, paired against the
    # complement the campaign used.  Included so that if the driver rounds,
    # we learn it for the exact denominators C2's elasticities were built on.
    for decode_sm in (16, 24, 44, 54, 92):
        prefill_sm = total_sm - decode_sm
        if prefill_sm > 0:
            out.append((prefill_sm, decode_sm, "c2_sweep"))
    return out


def probe_one(sm_a: int, sm_b: int, device: int) -> dict:
    rec: dict = {"requested_a": sm_a, "requested_b": sm_b}
    try:
        res = torch.ops.sgl_kernel.create_greenctx_stream_by_value(sm_a, sm_b, device)
        res = list(res)
        rec["n_returned"] = len(res)
        if len(res) >= 4:
            rec["realized_a"] = int(res[2])
            rec["realized_b"] = int(res[3])
            rec["delta_a"] = rec["realized_a"] - sm_a
            rec["delta_b"] = rec["realized_b"] - sm_b
            rec["requested_sum"] = sm_a + sm_b
            rec["realized_sum"] = rec["realized_a"] + rec["realized_b"]
            rec["unassigned_vs_request"] = rec["requested_sum"] - rec["realized_sum"]
            rec["exact"] = rec["delta_a"] == 0 and rec["delta_b"] == 0
        else:
            rec["note"] = "build returns <4 elements; realized counts not exposed"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device; this probe creates green contexts and "
              "cannot run on a login node", file=sys.stderr)
        return 2
    torch.cuda.init()
    torch.cuda.set_device(args.device)

    report: dict = {
        "probe": "smsplit_realized",
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "records": [],
    }

    try:
        from sgl_kernel import spatial as sgl_spatial

        report["sm_available"] = int(sgl_spatial.get_sm_available(args.device))
        report["sgl_kernel_spatial_file"] = sgl_spatial.__file__
    except Exception as exc:  # noqa: BLE001
        report["sm_available_error"] = repr(exc)

    total_sm = int(report.get("sm_available") or torch.cuda.get_device_properties(
        args.device).multi_processor_count)
    report["total_sm_used_for_targets"] = total_sm

    for sm_a, sm_b, origin in _targets_from_producer(total_sm):
        if sm_a <= 0 or sm_b <= 0:
            report["records"].append({"origin": origin, "skipped": True,
                                      "requested_a": sm_a, "requested_b": sm_b})
            continue
        rec = probe_one(sm_a, sm_b, args.device)
        rec["origin"] = origin
        report["records"].append(rec)

    scored = [r for r in report["records"] if "realized_a" in r]
    if not scored:
        exposed = [r.get("n_returned") for r in report["records"]
                   if "n_returned" in r]
        report["verdict"] = ("COUNTS_NOT_EXPOSED_BY_BUILD"
                             if exposed else "PROBE_FAILED")
    elif all(r["exact"] for r in scored):
        report["verdict"] = "REQUEST_EQUALS_DRIVER_REPORTED_PARTITION"
    else:
        report["verdict"] = "DRIVER_ROUNDS_REQUEST"
    report["n_scored"] = len(scored)
    report["n_exact"] = sum(1 for r in scored if r["exact"])
    report["any_unassigned"] = any(r["unassigned_vs_request"] != 0 for r in scored)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    # A verdict is never an exit code: every reachable outcome above is a
    # result, and only an unusable probe is a failure (gate #21).
    return 0 if report["verdict"] != "PROBE_FAILED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
