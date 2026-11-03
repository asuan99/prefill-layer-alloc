import sys, json, os, statistics, glob
sys.path.insert(0, os.environ.get("PDMUX_PROJECT_ROOT", "/home/wonho/Experiments/KISTI/prefill-layer-alloc") + "/workspace/engine-port/results/cap_char/policy_harness")
import ph_verdict as V
def stats(path):
    r = json.load(open(path))
    reqs = [q for s in r["segments"] for q in s["requests"]]
    tok = [x for q in reqs for x in (q.get("token_itl_ms") or [])]
    ok = [q for q in reqs if not q.get("error") and q.get("ttft_ms") is not None]
    rp50 = [V.percentile(q["token_itl_ms"], .5) for q in reqs if q.get("token_itl_ms")]
    rp95 = [V.percentile(q["token_itl_ms"], .95) for q in reqs if q.get("token_itl_ms")]
    n = len(tok)
    ttft = [q["ttft_ms"] for q in ok]
    att = sum(1 for q in reqs if V.req_good(q, 3000, 60)) / len(reqs)
    ttft_ok = sum(1 for q in reqs if not q.get("error") and q.get("ttft_ms") is not None and q["ttft_ms"] <= 3000) / len(reqs)
    return dict(run=r["run_id"], boot=r["boot_id"], trace_hash=r["trace_hash"], n_req=len(reqs), n_err=len(reqs) - len(ok), n_tok=n,
        req_itl_p50_med=statistics.median(rp50), req_itl_p95_med=statistics.median(rp95),
        tok_p50=V.percentile(tok, .5), tok_p95=V.percentile(tok, .95),
        frac_20_25=sum(1 for x in tok if 20 <= x < 25) / n, frac_gt40=sum(1 for x in tok if x > 40) / n,
        frac_60_65=sum(1 for x in tok if 60 <= x < 65) / n,
        ttft_p50=V.percentile(ttft, .5), ttft_p95=V.percentile(ttft, .95), ttft_attain=ttft_ok, attainment=att,
        n_pass=round(att * len(reqs)), gpu=r.get("gpu_uuid"))
if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(stats(p)))
