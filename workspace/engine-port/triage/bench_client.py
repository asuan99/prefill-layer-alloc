#!/usr/bin/env python3
"""Compact serving benchmark: Poisson arrivals, streaming TTFT/TPOT, goodput@SLO."""
import argparse, json, random, threading, time, urllib.request

def gen_prompt(n_tok):
    return "the robot walked slowly across the red planet " * max(1, n_tok // 9)

def one_request(port, prompt, out_len, results, idx):
    payload = json.dumps({
        "text": prompt,
        "sampling_params": {"max_new_tokens": out_len, "temperature": 0.0, "ignore_eos": True},
        "stream": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    last = t0
    itls = []
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                if "[DONE]" in line:
                    break
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last)
                last = now
                n_chunks += 1
        e2e = time.perf_counter() - t0
        tpot = (sum(itls) / len(itls)) if itls else 0.0
        results[idx] = {"ttft": ttft or e2e, "tpot": tpot, "e2e": e2e, "n": n_chunks, "ok": True}
    except Exception as e:
        results[idx] = {"ok": False, "err": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="30000")
    ap.add_argument("--rate", type=float, default=4.0)      # req/s (Poisson)
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--in-len", type=int, default=1500)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--ttft-slo", type=float, default=3.0)  # s
    ap.add_argument("--tpot-slo", type=float, default=0.06) # s
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    prompt = gen_prompt(a.in_len)
    results = {}
    threads = []
    t_start = time.perf_counter()
    for i in range(a.num):
        t = threading.Thread(target=one_request, args=(a.port, prompt, a.out_len, results, i))
        t.start(); threads.append(t)
        time.sleep(random.expovariate(a.rate))
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    ok = [r for r in results.values() if r.get("ok")]
    n_ok = len(ok)
    if not ok:
        print(f"BENCH {a.tag} rate={a.rate} ALL_FAILED"); return
    import statistics as st
    ttfts = sorted(r["ttft"] for r in ok)
    tpots = sorted(r["tpot"] for r in ok)
    tot_out = sum(r["n"] for r in ok)
    good = sum(1 for r in ok if r["ttft"] <= a.ttft_slo and r["tpot"] <= a.tpot_slo)
    def p(x, q): return x[min(len(x) - 1, int(q * len(x)))]
    print(f"BENCH {a.tag} rate={a.rate} n_ok={n_ok}/{a.num} wall={wall:.1f}s "
          f"out_tok/s={tot_out/wall:.1f} "
          f"ttft_med={st.median(ttfts)*1000:.0f}ms ttft_p99={p(ttfts,0.99)*1000:.0f}ms "
          f"tpot_med={st.median(tpots)*1000:.1f}ms tpot_p99={p(tpots,0.99)*1000:.1f}ms "
          f"goodput@SLO={good/wall:.2f}req/s ({good}/{n_ok})")

if __name__ == "__main__":
    main()
