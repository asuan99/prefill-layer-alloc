#!/usr/bin/env python3
"""A1 rev2 절차 3 — **Q3 채널 결정을 위한 비영 실증** (GPU 0, 기존 telemetry만).

감사 `audit_a1_rules_2026-08-25/VERDICT.md` §⑤ 절차 3:
    "Q3 채널 결정 — `decode_step_count`(⇒ 모드 등록 + 기판 불변 논증) vs
     `decode_iterations`(legacy에서 생존) 중 하나를 코드로 고정, 어느 쪽이든 **비영 실증 먼저**."

이 스크립트가 답하는 것 / 답하지 않는 것
--------------------------------------
답한다:
  (1) 두 후보 필드가 저장소의 실제 telemetry에서 **비영인가**.
  (2) 그 값이 런 안에서 **단조 비감소 카운터**로 행동하는가(런 단위 총계를 쓸 수 있는가).
  (3) 스냅샷 사이 Δ의 분포 — ★단위가 **step**임을 확인하고, "1 스냅샷 = k step"의 k가
      **아티팩트 성질**(샘플 그리드 stride)이지 상수가 아님을 보인다.
  (4) `architecture` 라벨별로 어느 필드가 사는가(= `decode_step_count`의 死 조건 재현).

★**답하지 않는다**: Q3 자체(= graph-launch RUNTIME row 수와 이 카운터가 **1:1인가**).
  그것은 nsys가 필요하고 **미측정**이다. 이 스크립트는 결정량이 **죽은 필드 위에 있지 않은지**만 본다.

★**사전 선언한 상한**: 저장소의 telemetry는 **25 GB / 996 파일**이라 전수 파싱이 I/O에 묶인다.
따라서 파일마다 **처음 `--max-snapshots`개의 `runtime_snapshot`만** 읽고, 잘린 파일은
`truncated: true`로 **센다**(무언의 절단 금지 — 게이트 #21 / "no silent caps").
결정량(필드가 살아 있는가·단조인가·Δ가 무엇인가)은 **접두부에서 답할 수 있다**;
런 단위 총계(span)는 잘린 파일에 대해 **답하지 않는다**고 표시한다.

두 부분으로 나눠 묻는다 (★각 질문에 맞는 모집단을 쓴다):

  **A. 저장소 전수(파일당 접두 상한)** — "필드가 살아 있는가 / 단조인가".
     이 두 성질은 **phase에 강건**하다(`decode_step_count`는 구조적으로 게이팅되고,
     카운터의 단조성은 어느 구간에서 보든 같다). 25 GB / 996 파일이라 파일마다
     처음 `--max-snapshots`개만 읽고 **잘린 파일 수를 센다**(무언의 절단 금지).

  **B. 사전 선언한 파일 목록의 전수 스캔(phase 필터)** — "Δ가 무엇인가".
     ★**A의 접두 표본으로 Δ를 말하면 안 된다** — 접두는 대부분 `phase="startup"`
     이고 그 구간의 `decode_iterations`는 0으로 고정이다(실측: s2_sticky b1에서
     startup 975 스냅샷 전부 0). Δ는 **`phase="benchmark"` 모집단에서만** 말한다.

★**답하지 않는다**: Q3 자체(= graph-launch RUNTIME row 수와 이 카운터가 **1:1인가**).
  그것은 nsys가 필요하고 **미측정**이다.

사용: python3 a1_q3_channel_probe.py [--root <dir>] [--out <json>] [--max-snapshots N]
"""
import argparse, glob, json, os, sys, collections

FIELDS = ("decode_iterations", "decode_step_count")


def scan_file(path, max_snapshots=0, phase=None):
    """한 telemetry 파일의 runtime_snapshot만 순서대로 읽어 필드별 통계를 낸다."""
    st = {f: {"n_present": 0, "n_nonzero": 0, "max": 0, "n_decrease": 0,
              "first": None, "last": None, "deltas": collections.Counter()}
          for f in FIELDS}
    arch = collections.Counter()
    prev = {f: None for f in FIELDS}
    n_snap = 0
    truncated = False
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("event") != "runtime_snapshot":
                    continue
                if phase is not None and rec.get("phase") != phase:
                    continue
                if max_snapshots and n_snap >= max_snapshots:
                    truncated = True
                    break
                n_snap += 1
                arch[rec.get("architecture")] += 1
                for f in FIELDS:
                    if f not in rec:
                        continue
                    try:
                        v = int(rec[f])
                    except (TypeError, ValueError):
                        continue
                    s = st[f]
                    s["n_present"] += 1
                    if v:
                        s["n_nonzero"] += 1
                    s["max"] = max(s["max"], v)
                    if s["first"] is None:
                        s["first"] = v
                    s["last"] = v
                    if prev[f] is not None:
                        d = v - prev[f]
                        if d < 0:
                            s["n_decrease"] += 1
                        else:
                            s["deltas"][d] += 1
                    prev[f] = v
    except OSError as e:
        return None, {"error": repr(e)}
    return n_snap, {"arch": dict(arch), "truncated": truncated,
                    "fields": {f: {k: (dict(v) if k == "deltas" else v)
                                   for k, v in st[f].items()} for f in FIELDS}}


# ★PRE-DECLARED (written before the numbers were read): the cells A1 rev2 cites.
# s2_sticky = the only boots where PDMUX_STICKY_PARTITION actually ran (job
# 873015); g16 d44 = the cell the NSL/kernel_mech analyses quote.
FULL_SCAN_GLOBS = (
    "s2_sticky/e1m3_T8_d16_b[1-4]_873015_telemetry.jsonl",
    "slo_sched/g16_blk[1-4]_d44_boot1_*_telemetry.jsonl",
)


def part_b(root):
    """Full scan, phase='benchmark' only, on the pre-declared file list."""
    out = {"population": "phase=='benchmark', FULL scan (no per-file cap)",
           "declared_globs": list(FULL_SCAN_GLOBS), "files": []}
    agg = collections.Counter()
    for pattern in FULL_SCAN_GLOBS:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            n, info = scan_file(path, max_snapshots=0, phase="benchmark")
            if not n:
                continue
            f = info["fields"]["decode_iterations"]
            deltas = collections.Counter({int(k): v for k, v in f["deltas"].items()})
            agg.update(deltas)
            nz = {d: c for d, c in deltas.items() if d}
            span = (f["last"] - f["first"]) if f["first"] is not None else None
            out["files"].append({
                "path": os.path.relpath(path, root),
                "n_benchmark_snapshots": n,
                "decode_iterations_first": f["first"],
                "decode_iterations_last": f["last"],
                "decode_steps_in_run": span,
                "steps_per_snapshot": (span / n) if (span is not None and n) else None,
                "n_pairs": sum(deltas.values()),
                "n_pairs_advancing": sum(nz.values()),
                "top_nonzero_deltas": [
                    {"delta": d, "count": c}
                    for d, c in sorted(nz.items(), key=lambda kv: -kv[1])[:5]
                ],
                "decode_step_count_max": info["fields"]["decode_step_count"]["max"],
            })
    nz_all = {d: c for d, c in agg.items() if d}
    out["pooled"] = {
        "n_pairs": sum(agg.values()),
        "n_pairs_advancing": sum(nz_all.values()),
        "top_nonzero_deltas": [
            {"delta": d, "count": c}
            for d, c in sorted(nz_all.items(), key=lambda kv: -kv[1])[:6]
        ],
    }
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")))
    ap.add_argument("--out", default=os.path.join(here, "a1_q3_channel_probe.json"))
    ap.add_argument("--max-snapshots", type=int, default=2000,
                    help="per-file cap on runtime_snapshot records (0 = no cap). "
                         "PRE-DECLARED and reported; truncated files are counted.")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.root, "**", "*telemetry*.jsonl"),
                             recursive=True))
    agg = {f: {"files_present": 0, "files_nonzero": 0, "snapshots_present": 0,
               "snapshots_nonzero": 0, "max": 0, "files_with_decrease": 0,
               "deltas": collections.Counter()} for f in FIELDS}
    arch_files = collections.Counter()
    per_file = []
    total_snap = 0
    n_truncated = 0
    for p in paths:
        n_snap, info = scan_file(p, max_snapshots=args.max_snapshots)
        if n_snap is None or not n_snap:
            continue
        total_snap += n_snap
        for a, c in info["arch"].items():
            arch_files[a] += c
        if info.get("truncated"):
            n_truncated += 1
        row = {"path": os.path.relpath(p, args.root), "n_snapshots": n_snap,
               "truncated": bool(info.get("truncated"))}
        for f in FIELDS:
            s = info["fields"][f]
            if not s["n_present"]:
                continue
            a = agg[f]
            a["files_present"] += 1
            a["snapshots_present"] += s["n_present"]
            a["snapshots_nonzero"] += s["n_nonzero"]
            a["max"] = max(a["max"], s["max"])
            if s["n_nonzero"]:
                a["files_nonzero"] += 1
            if s["n_decrease"]:
                a["files_with_decrease"] += 1
            for d, c in s["deltas"].items():
                a["deltas"][d] += c
            row[f] = {"max": s["max"], "nonzero": s["n_nonzero"],
                      "decrease": s["n_decrease"],
                      "span": (s["last"] - s["first"]) if s["first"] is not None else None}
        per_file.append(row)

    out = {
        "probe": "a1_q3_channel_probe",
        "question": "which Q3 counter channel is non-zero in this repository's telemetry",
        "does_not_answer": "Q3 itself (1:1 with nsys graph-launch rows) -- unmeasured",
        "root": args.root,
        "n_files_scanned": len(paths),
        "n_files_with_snapshots": len(per_file),
        "n_snapshots": total_snap,
        "max_snapshots_per_file": args.max_snapshots,
        "n_files_truncated_by_that_cap": n_truncated,
        "span_is_answerable_only_for_untruncated_files": True,
        "architecture_snapshot_counts": dict(arch_files),
        "fields": {},
        "per_file": per_file,
    }
    out["part_b_benchmark_phase_full_scan"] = part_b(args.root)
    for f in FIELDS:
        a = agg[f]
        top = sorted(a["deltas"].items(), key=lambda kv: -kv[1])[:8]
        nz = {d: c for d, c in a["deltas"].items() if d}
        nz_top = sorted(nz.items(), key=lambda kv: -kv[1])[:8]
        out["fields"][f] = {
            "files_present": a["files_present"],
            "files_nonzero": a["files_nonzero"],
            "snapshots_present": a["snapshots_present"],
            "snapshots_nonzero": a["snapshots_nonzero"],
            "max_observed": a["max"],
            "files_with_decrease": a["files_with_decrease"],
            "top_deltas_between_consecutive_snapshots": [
                {"delta": d, "count": c} for d, c in top
            ],
            # ★The repo-wide mode is 0: `dual_worker_trace_count` advances once
            # per SYNC and the event loop spins when idle, so most sampled pairs
            # contain no decode step at all.  The step-unit statement lives in
            # the NON-ZERO population, which is what this second list reports.
            "n_pairs_total": sum(a["deltas"].values()),
            "n_pairs_advancing": sum(nz.values()),
            "top_nonzero_deltas": [{"delta": d, "count": c} for d, c in nz_top],
        }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print(f"per-file cap = {args.max_snapshots} snapshots; "
          f"{n_truncated} file(s) hit it (span not answerable for those)")
    print(f"scanned {out['n_files_scanned']} telemetry files, "
          f"{out['n_files_with_snapshots']} with runtime_snapshot, "
          f"{total_snap} snapshots")
    print(f"architecture: {out['architecture_snapshot_counts']}")
    for f in FIELDS:
        d = out["fields"][f]
        print(f"\n{f}:")
        print(f"  present in {d['files_present']} files / {d['snapshots_present']} snapshots")
        print(f"  NONZERO   in {d['files_nonzero']} files / {d['snapshots_nonzero']} snapshots"
              f"   max={d['max_observed']}")
        print(f"  files with a DECREASE (counter reset): {d['files_with_decrease']}")
        print(f"  top deltas: {d['top_deltas_between_consecutive_snapshots'][:5]}")
    b = out["part_b_benchmark_phase_full_scan"]
    print("\n--- PART B: phase=='benchmark', FULL scan, pre-declared files ---")
    for row in b["files"]:
        print(f"  {row['path']}: n={row['n_benchmark_snapshots']} "
              f"steps={row['decode_steps_in_run']} "
              f"steps/snapshot={row['steps_per_snapshot']:.3f} "
              f"advancing={row['n_pairs_advancing']}/{row['n_pairs']} "
              f"top_nz={row['top_nonzero_deltas'][:2]}")
    print(f"  POOLED advancing={b['pooled']['n_pairs_advancing']}/{b['pooled']['n_pairs']} "
          f"top_nz={b['pooled']['top_nonzero_deltas'][:3]}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
