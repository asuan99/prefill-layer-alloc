"""Independent reconstruction of the sec2.4 state-conditional decode-batch
numbers (33.04 split / 20.39 d64-unsplit / 19.49 d74-unsplit), which the
results document quotes with NO preserved computation path."""
import json, glob, os, statistics, sys

SLO = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched"
want = sys.argv[1:] or ["d64", "d74", "d44", "d54"]
out = {}
for arm in want:
    for path in sorted(glob.glob(os.path.join(SLO, f"g16_blk*_{arm}_boot1_*_telemetry.jsonl"))):
        blk = os.path.basename(path).split("_")[1]
        split_b, unsplit_b, zero_b = [], [], []
        with open(path) as fh:
            for line in fh:
                if '"runtime_snapshot"' not in line:
                    continue
                e = json.loads(line)
                if e.get("phase") != "benchmark":
                    continue
                b = int(e.get("decode_running_batch_size", 0) or 0)
                if b <= 0:
                    continue
                sms = int(e.get("decode_sms", 0) or 0)
                if sms == int(arm[1:]):
                    split_b.append(b)
                elif sms in (0, 108):
                    unsplit_b.append(b)
        out.setdefault(arm, {})[blk] = (
            statistics.fmean(split_b) if split_b else float("nan"),
            statistics.fmean(unsplit_b) if unsplit_b else float("nan"),
            len(split_b), len(unsplit_b))
        print(f"{arm} {blk}: split-state mean batch {out[arm][blk][0]:.3f} (n={len(split_b)})   "
              f"unsplit mean batch {out[arm][blk][1]:.3f} (n={len(unsplit_b)})")
print()
for arm in out:
    s = [v[0] for v in out[arm].values()]
    u = [v[1] for v in out[arm].values()]
    print(f"{arm}: split {statistics.fmean(s):.2f} +- {statistics.stdev(s):.2f}   "
          f"unsplit {statistics.fmean(u):.2f} +- {statistics.stdev(u):.2f}   "
          f"ratio {statistics.fmean(s)/statistics.fmean(u):.2f}x")
