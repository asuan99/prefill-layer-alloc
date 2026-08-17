"""Independent (no g16_analyze import) recomputation of the per-boot G16
estimands M_ttft / M_itl, plus block-level resampling of the decision
quantities that G16_RESULTS_2026-08-17.md headlines.

READ-ONLY.  Writes only into this audit directory.
"""
import json, math, statistics, glob, os, re, sys

SLO_DIR = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched"

def pct(values, frac):
    if not values:
        return math.nan
    o = sorted(float(v) for v in values)
    pos = (len(o) - 1) * frac
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return o[lo]
    return o[lo] + (o[hi] - o[lo]) * (pos - lo)

def boot(path):
    ttfts, itl_p95, dur, nreq, nrounds, empt = [], [], 0.0, 0, 0, 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            nrounds += 1
            rec = json.loads(line)
            dur += float(rec.get("duration") or 0.0)
            t = rec.get("ttfts") or []
            it = rec.get("itls") or []
            for i in range(len(t)):
                ttfts.append(1000.0 * float(t[i]))
                toks = it[i] if i < len(it) else []
                if toks:
                    itl_p95.append(pct([1000.0 * float(v) for v in toks], 0.95))
                else:
                    empt += 1
                    itl_p95.append(math.inf)
                nreq += 1
    return {"M_ttft": statistics.median(ttfts), "M_itl": statistics.median(itl_p95),
            "duration_s": dur, "requests": nreq, "rounds": nrounds,
            "empty_itl": empt}

def main():
    out = {}
    for phase in ("LO", "HI"):
        grid = {}
        for path in sorted(glob.glob(os.path.join(SLO_DIR, f"g16_*_{phase}.jsonl"))):
            name = os.path.basename(path)
            m = re.match(r"^g16_(?P<block>[^_]+)_(?P<arm>d\d+)_boot(?P<boot>\d+)_", name)
            if not m or m.group("block").startswith("smoke"):
                continue
            side = os.path.join(SLO_DIR, name[: -len(f"_{phase}.jsonl")] + "_sidecar.json")
            sc = json.load(open(side))
            assert sc.get("status") == "completed" and sc.get("exact") is True, name
            grid.setdefault(m.group("arm"), {})[m.group("block")] = boot(path)
        out[phase] = grid
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "indep_perboot.json"), "w"), indent=1)
    for phase in ("LO", "HI"):
        print("=" * 30, phase)
        arms = sorted(out[phase], key=lambda a: int(a[1:]))
        blocks = sorted(out[phase][arms[0]])
        print("arm  " + "  ".join(f"{b:>9}" for b in blocks) + "     mean_itl   mean_ttft")
        for a in arms:
            itls = [out[phase][a][b]["M_itl"] for b in blocks]
            ttfts = [out[phase][a][b]["M_ttft"] for b in blocks]
            print(f"{a}  " + "  ".join(f"{v:9.3f}" for v in itls)
                  + f"  {statistics.fmean(itls):9.4f}  {statistics.fmean(ttfts):9.2f}")

if __name__ == "__main__":
    main()
