#!/usr/bin/env python3
"""기술 집계 · 판정 아님 (DESCRIPTIVE AGGREGATION ONLY -- NOT A VERDICT).

S3/S5 보조 집계.  어떤 결정 규칙에도 연결되지 않고, arm 순위·goodput·
SLO·운영점·인과 귀속을 일절 산출하지 않는다.  GPU 지출 0 (디스크에 이미
있는 telemetry 재집계).

무엇을 하는가.  `residency_census.py`의 PART 판정을 **그대로 import** 해서
(재구현 금지 -- 자기가 검증할 코드를 복사한 게이트는 항등식) PART를 두
갈래로 쪼갠 뒤 같은 다섯 규약으로 다시 집계한다:

    PART        <=> prefill_sms > 0 AND decode_sms > 0   (realized 필드)
    PART_pbusy  <=> PART AND prefill_active_batch_size > 0
    PART_pidle  <=> PART AND prefill_active_batch_size == 0

`prefill_active_batch_size`는 `dual_worker.py:570`
(`observe_scheduler`: `self.prefill.active_batch = scheduler.split_prefill_batch`)
을 거쳐 온 값이므로, **스냅샷 시점에 prefill 배치가 in-flight였는가**의
직접 프록시다.  단 스냅샷 격자 위에서만 관측되므로(median_gap_s 병기)
샘플 간격보다 짧은 상태는 해상하지 못한다.

자기검사(--selftest):
  (1) 가법성  PART_pbusy + PART_pidle == PART  (다섯 규약 전부, 분모 동일)
  (2) 음성 대조군  분할을 쓰지 않은 no-split 캠페인 파일에서 세 값 모두 0
  (3) 상류 selftest  residency_census tier1/tier1b 를 그대로 호출
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CENSUS_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "..",
    "workspace", "engine-port", "results", "residency_census_2026-09-21"))
sys.path.insert(0, _CENSUS_DIR)
import residency_census as RC  # noqa: E402  -- canonical PART predicate lives there

CONVENTIONS = RC.CONVENTIONS


def _pbusy(rec):
    return (rec.get("prefill_active_batch_size") or 0) > 0


def part_pbusy(rec):
    return RC.is_part(rec) and _pbusy(rec)


def part_pidle(rec):
    return RC.is_part(rec) and not _pbusy(rec)


def split_file(path):
    """세 번 census_file 을 돌린다 (part_pred 주입점만 바꾼다)."""
    base = RC.census_file(path, part_pred=RC.is_part)
    busy = RC.census_file(path, part_pred=part_pbusy)
    idle = RC.census_file(path, part_pred=part_pidle)
    out = {
        "path": path,
        "run_ids": base["run_ids"],
        "workload_ids": base["workload_ids"],
        "snapshots_all": base["snapshots_all"],
        "snapshots_busy": base["snapshots_busy"],
        "span_s": base["span_s"],
        "median_gap_s": base["median_gap_s"],
        "p95_gap_s": base["p95_gap_s"],
        "negative_gaps": base["negative_gaps"],
        "index_to_realized_sm": base["index_to_realized_sm"],
        "cohab_P_act": base["cohab_P_act"],
        "cohab_P_q": base["cohab_P_q"],
        "dwell_part_runs": base["dwell_part_runs"],
        "dwell_part_mean_s": base["dwell_part_mean_s"],
    }
    for c in CONVENTIONS:
        out[c] = {
            "PART": base[c], "PART_pbusy": busy[c], "PART_pidle": idle[c],
            "additive": abs((busy[c]["num"] + idle[c]["num"]) - base[c]["num"]) < 1e-6,
        }
    return out


def aggregate(rows):
    acc = {c: [0.0, 0.0, 0.0] for c in CONVENTIONS}   # busy_num, idle_num, den
    meta = {"files": 0, "snapshots_all": 0, "snapshots_busy": 0, "span_s_sum": 0.0,
            "nonmono_files": 0, "cohab_part_busy_n": 0, "cohab_P_act_num": 0,
            "dwell_part_runs": 0}
    for r in rows:
        nonmono = r["negative_gaps"] > 0
        meta["files"] += 1
        meta["snapshots_all"] += r["snapshots_all"]
        meta["snapshots_busy"] += r["snapshots_busy"]
        if (r["span_s"] or 0.0) >= 0:
            meta["span_s_sum"] += r["span_s"] or 0.0     # SUM, never max()
        meta["nonmono_files"] += 1 if nonmono else 0
        meta["cohab_part_busy_n"] += r["cohab_P_act"]["den"]
        meta["cohab_P_act_num"] += r["cohab_P_act"]["num"]
        meta["dwell_part_runs"] += r["dwell_part_runs"]
        for c in CONVENTIONS:
            if nonmono and c in RC.TIME_CONVENTIONS:
                continue
            acc[c][0] += r[c]["PART_pbusy"]["num"]
            acc[c][1] += r[c]["PART_pidle"]["num"]
            acc[c][2] += r[c]["PART"]["den"]
    out = dict(meta)
    for c in CONVENTIONS:
        b, i, d = acc[c]
        out[c] = {
            "den": round(d, 4),
            "PART_pct": (round(100.0 * (b + i) / d, 4) if d else None),
            "PART_pbusy_pct": (round(100.0 * b / d, 4) if d else None),
            "PART_pidle_pct": (round(100.0 * i / d, 4) if d else None),
        }
    return out


NEG_CONTROL_GLOB = "p1_gates/gate2"
NEG_CONTROL_MARK = "nosplit"


def selftest(results_root):
    fails = []
    print("# tier0: 상류 residency_census 자기검사 재사용")
    fails += ["upstream tier1: " + f for f in RC.selftest_tier1(verbose=True)]
    fails += ["upstream tier1b: " + f for f in RC.selftest_sum_not_max(verbose=True)]

    print("# tier A: 합성 fixture 가법성 (PART_pbusy + PART_pidle == PART)")
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "fixture.jsonl")
    RC._write_fixture(tmp)
    r = split_file(tmp)
    for c in CONVENTIONS:
        if not r[c]["additive"]:
            fails.append(f"fixture {c}: not additive")
        print(f"    {c:12} PART={r[c]['PART']['pct']}  "
              f"pbusy={r[c]['PART_pbusy']['pct']}  pidle={r[c]['PART_pidle']['pct']}  "
              f"additive={r[c]['additive']}")
    # fixture 는 PART 3 스냅샷 중 2개가 prefill_active>0 이므로 두 갈래 모두 비영이어야
    # 한다 -- 한쪽이 0이면 갈래 자체가 해상력이 없다는 뜻이다 (교훈 232).
    if not (r["R_cnt_all"]["PART_pbusy"]["num"] > 0 and r["R_cnt_all"]["PART_pidle"]["num"] > 0):
        fails.append("fixture: 두 갈래 중 하나가 0 -- 분해가 arm 차이를 해상하지 못한다")

    print("# tier B: 음성 대조군 (no-split 캠페인 -- 세 값 모두 0이어야)")
    neg = []
    gate2 = os.path.join(results_root, NEG_CONTROL_GLOB)
    if os.path.isdir(gate2):
        for name in sorted(os.listdir(gate2)):
            if NEG_CONTROL_MARK in name and name.endswith(".jsonl") and "telemetry" in name:
                neg.append(os.path.join(gate2, name))
    neg = neg[:4]
    if not neg:
        fails.append("음성 대조군 파일을 찾지 못했다 (확인 불가, 추측으로 메우지 않는다)")
    for p in neg:
        rr = split_file(p)
        vals = {c: (rr[c]["PART"]["pct"], rr[c]["PART_pbusy"]["pct"], rr[c]["PART_pidle"]["pct"])
                for c in CONVENTIONS}
        bad = [c for c, v in vals.items() if any((x or 0.0) != 0.0 for x in v)]
        print(f"    {os.path.basename(p)[:70]:70} realized_sm={rr['index_to_realized_sm']} "
              f"{'OK' if not bad else 'NONZERO:' + ','.join(bad)}")
        if bad:
            fails.append(f"negative control {os.path.basename(p)}: {bad}")

    if fails:
        print("\nSELFTEST_FAILED")
        for f in fails:
            print("  " + f)
        return 1
    print("\nSELFTEST_OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", default=os.path.normpath(os.path.join(
        _HERE, "..", "..", "..", "..", "workspace", "engine-port", "results")))
    ap.add_argument("--campaign", action="append", default=[],
                    help="results-root 하위 캠페인 디렉터리 (반복 가능)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.results_root)

    report = {"conventions": list(CONVENTIONS), "campaigns": {}, "files": []}
    for camp in args.campaign:
        root = os.path.join(args.results_root, camp)
        rows = []
        for dirpath, _dirs, names in os.walk(root):
            for n in sorted(names):
                if not n.endswith(".jsonl"):
                    continue
                p = os.path.join(dirpath, n)
                if os.path.getsize(p) < 1024:
                    continue
                with open(p, "r", errors="replace") as fh:
                    head = fh.read(200_000)
                if '"runtime_snapshot"' not in head:
                    continue
                rows.append(split_file(p))
        for r in rows:
            r["campaign"] = camp
        report["files"] += rows
        report["campaigns"][camp] = aggregate(rows)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1, sort_keys=True)
    for camp, a in report["campaigns"].items():
        print(f"\n== {camp}  files={a['files']} snapshots={a['snapshots_all']} "
              f"span_sum={a['span_s_sum']:.1f}s nonmono={a['nonmono_files']}")
        print(f"   {'convention':12} {'PART%':>9} {'PART&pbusy%':>12} {'PART&pidle%':>12} {'den':>14}")
        for c in CONVENTIONS:
            v = a[c]
            print(f"   {c:12} {v['PART_pct']:>9} {v['PART_pbusy_pct']:>12} "
                  f"{v['PART_pidle_pct']:>12} {v['den']:>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
