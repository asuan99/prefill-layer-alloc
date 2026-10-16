#!/usr/bin/env python3
"""shape X small correctness step (U7, user decision 2026-10-16; REVIEW_U4_U12_2026-10-16.md).  CPU-testable.

  serve --endpoint URL --tokens F.bin --out xcheck.json      run inside a boot (cmd step "xcheck_<arm>"): 4 X documents, greedy 32 tokens,
                                                             two passes -- "c" (the 4 requests concurrently) and "s" (one at a time)
  score --gates DIR --boots DIR --pair BASE_BOOT_ID TREAT_BOOT_ID [--out XCHECK.json]

WHERE (implementation of U7, corrected 2026-10-16): the fused_sym and agnostic boot1 of every config -- N1 in J1, TP_N and DP_N in
J_N (on the N-GPU server) -- after the floor block and before the cells.  Verdict per (Job, config) -> gates/XCHECK_<config>.json;
mg_label marks the shape-X estimands of THAT config only X_CORRECTNESS_FAIL (or _NOT_TESTED / _NOT_RUN) when it is not PASS
(fail-closed); the other mode's X and every A/B estimand are untouched.  The xcheck requests are in mg_boot's fit pre-check.
DOCUMENTS [제안]: the XC_N_DOCS longest retained documents of the registered token file (ties: smaller doc_id) -- the long-prompt
  (split prefill) path is the one M2g never exercised.
REQUEST: POST /generate {"input_ids", "sampling_params": {"temperature": 0, "max_new_tokens": XC_MAX_NEW, "ignore_eos": true},
  "return_logprob": true, "logprob_start_len": len(input_ids)} (non-streaming).  logprobs only over the generated tokens, read for
  finiteness (NaN / inf check); both arms send the identical request bytes.
VERDICT (G-M2-style relaxed rule; the parts marked [제안] are proposals, not decisions):
  baseline noise D0 [제안] = exact-match count (all XC_MAX_NEW tokens) fused_sym pass "c" vs fused_sym pass "s"
                            (same boot, same requests, different batch composition)
  treatment D1           = exact-match count fused_sym pass "c" vs agnostic pass "c"
  PASS iff (a) D1 >= D0 - XC_DROP_TOL (1 request), (b) every document's first XC_FIRST_N [제안: 16] tokens agree between
  fused_sym "c" and agnostic "c", (c) no request error, every output exactly XC_MAX_NEW tokens, every request carrying exactly
  XC_MAX_NEW output-token logprobs (harness audit r1 R-H5: a response without logprobs would make the finiteness check vacuous) and
  all of them finite, and both servers alive right after their xcheck step (alive_after_xcheck_<arm>.json).  Any of (a)-(c)
  observed false -> FAIL.
  A missing record (output file, survival record) -> NOT_TESTED.  Only PASS keeps the Job's X cells citable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(ENGINE / "results" / "multigpu"))
sys.path.insert(0, str(HERE))
import mg_const as C  # noqa: E402
import mg_x_pretokenize as XT  # noqa: E402

PASS, FAIL, NOT_TESTED = "PASS", "FAIL", "NOT_TESTED"


def select_docs(header, n=None):
    """The n longest retained documents (ties: smaller doc_id), returned in doc_id order."""
    n = C.XC_N_DOCS if n is None else n
    docs = sorted(header["docs"], key=lambda d: (-int(d[2]), int(d[0])))[:n]
    return sorted(int(d[0]) for d in docs)


def request_body(ids):
    return {"input_ids": list(ids), "sampling_params": {"temperature": 0.0, "max_new_tokens": C.XC_MAX_NEW, "ignore_eos": True},
            "return_logprob": True, "logprob_start_len": len(ids)}


def _post(endpoint, body, timeout_s=600.0):
    import urllib.request
    req = urllib.request.Request(endpoint.rstrip("/") + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.load(resp)


def _one(endpoint, rid, ids, post):
    try:
        out = post(endpoint, request_body(ids))
    except Exception as exc:  # noqa: BLE001 - recorded, the verdict reads it
        return rid, {"output_ids": [], "error": repr(exc), "nonfinite": 0}
    if isinstance(out, dict) and out.get("error"):
        return rid, {"output_ids": [], "error": str(out.get("error"))[:300], "nonfinite": 0}
    mi = (out or {}).get("meta_info") or {}
    lps = [t[0] for t in (mi.get("output_token_logprobs") or []) if isinstance(t, (list, tuple)) and t]
    bad = sum(1 for v in lps if v is None or not math.isfinite(float(v)))
    return rid, {"output_ids": [int(x) for x in (out.get("output_ids") or [])], "error": "", "nonfinite": bad,
                 "n_logprobs": len(lps)}


def serve(endpoint, tokens, out, post=_post):
    h = XT.read_header(tokens)
    docs = select_docs(h)
    ids = XT.doc_input_ids(tokens, h, docs)
    rec = {"schema": "mg_xcheck/v1", "docs": docs, "input_tokens": {str(d): len(ids[d]) for d in docs}, "passes": {}, "t0": time.time()}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(docs)) as ex:
        rec["passes"]["c"] = dict(ex.map(lambda d: _one(endpoint, str(d), ids[d], post), docs))
    rec["passes"]["s"] = dict(_one(endpoint, str(d), ids[d], post) for d in docs)
    rec["t1"] = time.time()
    Path(out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    errs = [f"{p}:{d}" for p, r in rec["passes"].items() for d, v in r.items() if v["error"]]
    return 0 if not errs else 1


def _exact(a, b, docs):
    return sum(1 for d in docs if (a.get(d) or {}).get("output_ids") and a[d]["output_ids"] == (b.get(d) or {}).get("output_ids"))


def score(base, treat, base_alive, treat_alive, first_n=None, _mut=None):
    """base/treat: xcheck.json records (fused_sym, agnostic) or None; *_alive: True / False / None (gate alive record)."""
    first_n = C.XC_FIRST_N if first_n is None else first_n
    if base is None or treat is None:
        return {"verdict": NOT_TESTED, "why": "xcheck record missing"}
    docs = [str(d) for d in base.get("docs", [])]
    why = []
    if [str(d) for d in treat.get("docs", [])] != docs or len(docs) != C.XC_N_DOCS:
        return {"verdict": NOT_TESTED, "why": "document sets differ or incomplete"}
    for name, rec in (("fused_sym", base), ("agnostic", treat)):
        for p in ("c", "s") if name == "fused_sym" else ("c",):
            for d in docs:
                v = (rec.get("passes", {}).get(p) or {}).get(d)
                if v is None:
                    return {"verdict": NOT_TESTED, "why": f"{name}:{p}:{d} missing"}
                if v.get("error"):
                    why.append(f"{name}:{p}:{d}:ERROR")
                elif len(v.get("output_ids") or []) != C.XC_MAX_NEW:
                    why.append(f"{name}:{p}:{d}:LEN_{len(v.get('output_ids') or [])}")
                if int(v.get("nonfinite") or 0):
                    why.append(f"{name}:{p}:{d}:NONFINITE")
                if not v.get("error") and int(v.get("n_logprobs") or 0) != C.XC_MAX_NEW:
                    why.append(f"{name}:{p}:{d}:LOGPROBS_{int(v.get('n_logprobs') or 0)}")       # R-H5: absent / short logprobs -> FAIL
    for name, al in (("fused_sym", base_alive), ("agnostic", treat_alive)):
        if al is False:
            why.append(f"{name}:SERVER_DEAD_AFTER_XCHECK")
    bc, bs, tc = base["passes"]["c"], base["passes"]["s"], treat["passes"]["c"]
    d0, d1 = _exact(bc, bs, docs), _exact(bc, tc, docs)
    first_bad = [d for d in docs if (bc[d].get("output_ids") or [])[:first_n] != (tc[d].get("output_ids") or [])[:first_n]
                 or len(bc[d].get("output_ids") or []) < first_n]
    if _mut != "drop_rule" and d1 < d0 - C.XC_DROP_TOL:
        why.append(f"EXACT_DROP(D1={d1} < D0={d0} - {C.XC_DROP_TOL})")
    if _mut != "first_n_rule" and first_bad:
        why.append(f"FIRST_{first_n}_MISMATCH{first_bad}")
    out = {"D0_exact": d0, "D1_exact": d1, "first_n": first_n, "first_n_mismatch": first_bad, "n_docs": len(docs),
           "base_alive": base_alive, "treat_alive": treat_alive, "reasons": why}
    if why:
        return dict(out, verdict=FAIL)
    if base_alive is None or treat_alive is None:
        return dict(out, verdict=NOT_TESTED, why="survival record after the xcheck step missing")
    return dict(out, verdict=PASS)


def score_job(gates, boots, base_boot, treat_boot, config="N1"):
    import m2g_correctness_gate as G
    rd = lambda bid: (json.loads((Path(boots) / bid / "xcheck.json").read_text())
                      if (Path(boots) / bid / "xcheck.json").exists() else None)
    al = lambda bid, arm: G.alive_after_step(Path(boots) / bid, f"xcheck_{arm}")
    r = score(rd(base_boot), rd(treat_boot), al(base_boot, "fused_sym"), al(treat_boot, "agnostic"))
    r.update(base_boot=base_boot, treat_boot=treat_boot, config=config, rule="U7 G-M2-style: D1 >= D0-1, first-N agree, no crash/NaN/error")
    Path(gates).mkdir(parents=True, exist_ok=True)
    (Path(gates) / f"XCHECK_{config}.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--endpoint", required=True)
    s.add_argument("--tokens", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("score")
    c.add_argument("--gates", required=True)
    c.add_argument("--boots", required=True)
    c.add_argument("--pair", nargs=2, required=True)
    c.add_argument("--config", default="N1")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return serve(a.endpoint, a.tokens, a.out)
    r = score_job(a.gates, a.boots, *a.pair, config=a.config)
    print(json.dumps(r, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
