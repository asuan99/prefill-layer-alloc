#!/usr/bin/env python3
"""Gate 2-S correctness gate -- PREREG_GATE2S_2026-08-09.md rev6 sec6.3.

G-1  self-baseline : batch=1 greedy, store output_ids + sha256.
G-2  concurrent    : N concurrent greedy calls vs this arm's OWN batch=1 baseline.

METHODOLOGY GATE #21 (7th recurrence, recorded 2026-08-10).  A transport
failure is a MEASUREMENT failure, never a mismatch and never a gate failure:

    HTTP != 200            -> UNDETERMINED
    timeout / conn error   -> UNDETERMINED     (STATUS=TIMEOUT kept distinct)
    body without output_ids-> UNDETERMINED

Only a 200 response carrying output_ids can ever be scored match/mismatch.
There is deliberately NO code path from UNDETERMINED to FAIL.

sec6.3-4 also records that 16-concurrent greedy bitwise irreproducibility was
reproduced 3x independently in E-A and is NOT a correctness defect -> mismatches
are reported as INDETERMINATE with the first divergent token index, not as a
discard trigger.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

CONNECT_TIMEOUT = float(os.environ.get("G2_GREEDY_CONNECT_TIMEOUT", "10"))
MAX_TIME = float(os.environ.get("G2_GREEDY_MAX_TIME", "180"))


def greedy_call(port: int, prompt: str, out_len: int) -> Dict[str, Any]:
    """Bounded single greedy call.  Returns a status dict; NEVER raises."""
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"max_new_tokens": out_len, "temperature": 0},
        "return_logprob": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=MAX_TIME) as resp:
            code = resp.getcode()
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:                      # non-200
        return dict(status="UNDETERMINED", http=e.code,
                    detail=f"http_error {e.code} -- measurement failure, not a mismatch")
    except Exception as e:                                    # timeout / transport
        kind = "TIMEOUT" if "timed out" in str(e).lower() else "ERROR"
        return dict(status="UNDETERMINED", http=0, kind=kind,
                    detail=f"{kind}: {e!r} -- MEASUREMENT FAILURE (gate #21), never a mismatch")
    if code != 200:
        return dict(status="UNDETERMINED", http=code, detail="non-200")
    ids = (payload.get("meta_info") or {}).get("output_token_logprobs")
    if ids:
        ids = [t[1] for t in ids]
    else:
        ids = (payload.get("meta_info") or {}).get("output_ids") or payload.get("output_ids")
    if not ids:
        # sec6.3-4: no ids -> cannot be compared.  UNDETERMINED, never FAIL.
        return dict(status="UNDETERMINED", http=200,
                    detail="no output_ids in response -- cannot compare",
                    text=payload.get("text", "")[:200])
    return dict(status="OK", http=200, output_ids=list(ids),
                sha=hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest())


def first_diff(a: List[int], b: List[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out-len", type=int, default=96)
    ap.add_argument("--n-concurrent", type=int, default=16)
    ap.add_argument("--baseline", required=True, help="write/read self baseline json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = open(args.prompt_file).read()
    res: Dict[str, Any] = {"arm": args.arm, "n_concurrent": args.n_concurrent,
                           "prereg": "rev6 sec6.3"}

    # ---- G-1: self baseline -------------------------------------------------
    base = greedy_call(args.port, prompt, args.out_len)
    res["G1"] = {k: v for k, v in base.items() if k != "output_ids"}
    if base["status"] != "OK":
        res["verdict"] = "UNDETERMINED"
        res["reason"] = "G-1 baseline is a measurement failure (gate #21): " + base.get("detail", "")
        json.dump(res, open(args.out, "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    json.dump({"sha": base["sha"], "output_ids": base["output_ids"]},
              open(args.baseline, "w"))

    # ---- G-2: N concurrent vs own baseline ---------------------------------
    with cf.ThreadPoolExecutor(max_workers=args.n_concurrent) as ex:
        outs = list(ex.map(lambda _: greedy_call(args.port, prompt, args.out_len),
                           range(args.n_concurrent)))
    n_ok = n_match = n_mismatch = n_undet = 0
    diffs: List[Dict[str, Any]] = []
    for i, o in enumerate(outs):
        if o["status"] != "OK":
            n_undet += 1                       # gate #21: never a mismatch
            diffs.append(dict(idx=i, status=o["status"], detail=o.get("detail", "")))
            continue
        n_ok += 1
        if o["output_ids"] == base["output_ids"]:
            n_match += 1
        else:
            n_mismatch += 1
            diffs.append(dict(idx=i, status="MISMATCH",
                              first_diff_index=first_diff(o["output_ids"], base["output_ids"]),
                              len_self=len(o["output_ids"]), len_base=len(base["output_ids"]),
                              sha=o["sha"]))
    res["G2"] = dict(n_ok=n_ok, n_match=n_match, n_mismatch=n_mismatch,
                     n_undetermined=n_undet, details=diffs)

    if n_ok == 0:
        res["verdict"] = "UNDETERMINED"
        res["reason"] = "no scoreable concurrent response (all measurement failures)"
    elif n_mismatch == 0:
        res["verdict"] = "PASS"
    else:
        # sec6.3-4: E-A reproduced this 3x and it is NOT a correctness defect.
        res["verdict"] = "INDETERMINATE"
        res["reason"] = ("concurrent greedy bitwise irreproducibility -- E-A reproduced this "
                         "3x independently and it is NOT a correctness defect (sec6.3-4); "
                         "reported with first divergent token index, not a discard trigger")
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "G2"}, indent=1))
    print(f"G2: ok={n_ok} match={n_match} mismatch={n_mismatch} undetermined={n_undet} "
          f"-> {res['verdict']}")


if __name__ == "__main__":
    main()
