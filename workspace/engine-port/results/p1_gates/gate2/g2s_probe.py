#!/usr/bin/env python3
"""Gate 2-S micro probes -- PREREG_GATE2S_2026-08-09.md rev6 sec6.2.

ONE execution per boot (sec6.2.1 rev3 change 1).  The scoring campaign already
boots a fresh server per (arm, rep) -- g2_run.sbatch:375 -- so riding those 10
boots gives n_eff = 10 at ZERO extra boots.  Running 10 calls inside ONE boot
would be pseudoreplication (n_eff = 1) and biased toward the equivalence
conclusion; that is what rev2 did and rev3 fixed.

Two probes survive (rev3 deleted P-b: green-ctx count is perfectly collinear
with KV-pool/cudagraph-memory, sec2.3-a):

  P-pos    T-boot decode-only (idx1 = 34 SM) vs C-boot decode-only (idx3).
           sec6.2.1(1): HARNESS SANITY CHECK ONLY -- rev4 demoted it out of the
           anchor conditions because it cannot realistically fail.
  P-carve  C (unpartitioned pdmux) vs A2 (fused).  TOST, delta_probe = 5%.
           sec6.2.1(4): NO campaign veto.  Result enters as caveat only.

MANDATORY caveats emitted with every result (sec6.2.1):
  * micro-regime: 8 requests, 128/128 -- does NOT transfer to the serving
    operating point (confound #7 regime).
  * NO SM-unit detection floor is computed.  rev2's 15-60 SM figure came from
    transplanting the C2 elasticity across grids, which CONSENSUS sec3 items
    21/25/26 forbid ("prediction bands are covered by the transplant ban too").
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics as st
import time
import urllib.request
from typing import Any, Dict, List

MAX_TIME = float(os.environ.get("G2_GREEDY_MAX_TIME", "180"))

MICRO_CAVEAT = (
    "이 프로브는 8요청·input128/output128의 마이크로 체제에서 측정됐다(confound #7 체제). "
    "결과는 서빙 동작점(in2000/out96, rate 2-4)으로 이전되지 않는다. "
    "sec1.1 하향('무분할'=명시적 분할 부재, 잔여 차감 미측정)은 이 프로브가 있어도 유효하다.")
NO_FLOOR_NOTE = (
    "검출 하한을 SM 단위로 환산하지 않는다(sec6.2.1 변경 3: C2 elasticity 이식 금지, "
    "CONSENSUS sec3 items 21/25/26). 하한은 미정이거나 이 캠페인 내부에서만 산출한다.")


def one_request(port: int, prompt: str, out_len: int) -> Dict[str, Any]:
    body = json.dumps({"text": prompt,
                       "sampling_params": {"max_new_tokens": out_len, "temperature": 0}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=MAX_TIME) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:                       # gate #21: measurement failure
        return dict(status="UNDETERMINED", detail=repr(e))
    dt = time.perf_counter() - t0
    mi = payload.get("meta_info") or {}
    n_out = mi.get("completion_tokens") or out_len
    return dict(status="OK", e2e_s=dt, n_out=int(n_out),
                itl_mean_ms=1000.0 * dt / max(1, int(n_out) - 1))


def run_probe(port: int, n_conc: int, in_tok: int, out_len: int) -> Dict[str, Any]:
    """8 requests injected at once, NO further arrivals => prefill idle, matched batch."""
    prompt = " ".join(["token"] * in_tok)
    with cf.ThreadPoolExecutor(max_workers=n_conc) as ex:
        outs = list(ex.map(lambda _: one_request(port, prompt, out_len), range(n_conc)))
    ok = [o for o in outs if o["status"] == "OK"]
    if not ok:
        return dict(status="UNDETERMINED", n_ok=0,
                    detail="all requests were measurement failures (gate #21)")
    itls = sorted(o["itl_mean_ms"] for o in ok)
    return dict(status="OK", n_ok=len(ok),
                itl_p50_ms=st.median(itls), itl_mean_ms=sum(itls) / len(itls))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-concurrent", type=int, default=8)
    ap.add_argument("--in-tokens", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=128)
    args = ap.parse_args()

    r = run_probe(args.port, args.n_concurrent, args.in_tokens, args.out_len)
    rec = dict(arm=args.arm, rep=args.rep, prereg="rev6 sec6.2",
               probe_config=dict(n_concurrent=args.n_concurrent,
                                 in_tokens=args.in_tokens, out_len=args.out_len),
               result=r,
               micro_regime_caveat=MICRO_CAVEAT,
               detection_floor=NO_FLOOR_NOTE,
               note=("one execution per boot (n_eff = number of boots, sec6.2.1); "
                     "P-pos is a SANITY CHECK (not an anchor condition, rev4); "
                     "P-carve carries NO campaign veto (sec6.2.1(4))"))
    json.dump(rec, open(args.out, "w"), indent=1)
    print(f"[probe] arm={args.arm} rep={args.rep} status={r['status']} "
          f"itl_p50={r.get('itl_p50_ms', float('nan')):.3f}ms n_ok={r.get('n_ok', 0)}")


if __name__ == "__main__":
    main()
