#!/usr/bin/env python3
"""Step 0 -- 전환 경계의 decode 간극을 **이미 있는** HOLB 아티팩트에서 재분석한다.

왜 이 파일이 있는가: `DESIGN_SWITCH_COST_2026-08-22.md`가 *"drain을 잰 아티팩트는 0건"* 이라
적었는데 **거짓**이었다(감사 D1). `holb_probe.py:428-430`이 직전 decode end → 이번 decode
start 를 device 이벤트로 이미 재고, `:203`이 `stream_key`(파티션별 decode 스트림 객체)를 함께
적는다.  ⇒ **stream_key 변화 = 전환 실행**이고, 조인 없이 라벨이 붙는다.
정확한 서술은 "잰 아티팩트 0건"이 아니라 **"그 아티팩트를 이 질문으로 분석한 적이 0건"**(교훈 #18).

GPU 0 · 새 측정 0 · 성능 판정 0건.  산출은 `c`가 아니라 **`c + confound`의 상한**이다
(전환이 컨트롤러 구동이라 "전환 비용"과 "전환을 촉발한 부하"가 교락 — 감사 D2/confound ①).

★층화가 이 분석의 핵심:
  gap_class == "strict"     : 그 간극 동안 **다른 스트림 forward가 없었다** → 기계 비용에 가깝다
  gap_class == "ambiguous"  : 간극 동안 prefill forward가 **실제로 돌았다**(n_other_fw>0)
                              → 이건 오버헤드가 아니라 **prefill 실행을 기다린 시간**이다
층을 섞으면 평균이 꼬리에 지배되고(총합의 ~91%), 그 꼬리는 대부분 ambiguous다.
⇒ **1차 = strict 층의 중앙값 대비**, 평균은 진단으로만 보고한다.
"""
from __future__ import annotations
import json, glob, math, os, statistics as st, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
GATE2 = os.path.join(HERE, "..", "p1_gates", "gate2")
SEED = 1
B = 10000


def boundaries(path):
    """연속한 decode 간극 이벤트를 (전환/비전환) × (strict/ambiguous)로 라벨."""
    prev_key = None
    out = []
    dropped = 0
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") != "holb_gap" or d.get("gap_ms") is None:
            continue
        if d.get("dropped_events"):
            dropped = max(dropped, int(d["dropped_events"]))
        key = d.get("stream_key")
        if key is None:
            continue
        if prev_key is not None:
            out.append({
                "gap_ms": float(d["gap_ms"]),
                "switch": key != prev_key,
                "strict": d.get("gap_class") == "strict",
                "n_other_fw": d.get("n_other_fw") or 0,
                "other_ms": float(d.get("other_stream_fw_ms") or 0.0),
                "bs": d.get("decode_bs"),
                "phase": d.get("phase"),
            })
        prev_key = key
    return out, dropped


def med(xs):
    return st.median(xs) if xs else float("nan")


def summarize(path):
    ev, dropped = boundaries(path)
    ev = [e for e in ev if e.get("phase") in (None, "measure")]   # 워밍업 제외
    if not ev:
        return None
    row = {"file": os.path.basename(path), "n_boundary": len(ev),
           "dropped_events": dropped,
           "n_switch": sum(e["switch"] for e in ev)}
    for strat, keep in (("strict", True), ("ambig", False)):
        sw = [e["gap_ms"] for e in ev if e["switch"] and e["strict"] is keep]
        no = [e["gap_ms"] for e in ev if not e["switch"] and e["strict"] is keep]
        row[strat] = {
            "n_sw": len(sw), "n_no": len(no),
            "med_sw": med(sw), "med_no": med(no),
            "delta_med": (med(sw) - med(no)) if sw and no else float("nan"),
            "mean_sw": st.fmean(sw) if sw else float("nan"),
            "mean_no": st.fmean(no) if no else float("nan"),
        }
    # 꼬리 지배도 (층 무시)
    allg = sorted((e["gap_ms"] for e in ev), reverse=True)
    tot = sum(allg)
    row["top5pct_share"] = (sum(allg[:max(1, len(allg) // 20)]) / tot) if tot else float("nan")
    # ambiguous 간극에서 다른 스트림 실행분을 뺀 잔차 (진단)
    amb = [e for e in ev if not e["strict"] and e["switch"]]
    row["ambig_sw_residual_med"] = med([max(0.0, e["gap_ms"] - e["other_ms"]) for e in amb]) if amb else float("nan")
    return row


def boot_ci(vals, stat=st.median, b=B, seed=SEED):
    """아티팩트(rep)를 1차 단위로 하는 부트스트랩. 경계는 독립 반복이 아니다."""
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 3:
        return (float("nan"), float("nan"), len(vals))
    rng = random.Random(seed)
    draws = sorted(stat([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(b))
    return (draws[int(0.025 * b)], draws[int(0.975 * b)], len(vals))


def main():
    pats = sys.argv[1:] or ["g2_*_agnostic_rep*.holb.jsonl"]
    files = sorted(f for p in pats for f in glob.glob(os.path.join(GATE2, p)))
    rows = [r for r in (summarize(f) for f in files) if r]
    if not rows:
        print("no artifacts matched"); return 2

    print(f"아티팩트 {len(rows)}개 · 경계 총 {sum(r['n_boundary'] for r in rows)}개 "
          f"· 전환 {sum(r['n_switch'] for r in rows)}개 · dropped_events max="
          f"{max(r['dropped_events'] for r in rows)}")
    print(f"\n{'stratum':8s} {'n_art':>5s} {'med(sw)':>9s} {'med(no)':>9s} {'Δmed':>8s} "
          f"{'Δmed 95%CI (rep-boot)':>26s}")
    out = {"frame": "Step 0 재분석. GPU 0 · 새 측정 0 · 성능 판정 0건. "
                    "산출은 c가 아니라 c+confound의 상한(전환이 컨트롤러 구동).",
           "n_artifacts": len(rows), "per_artifact": rows, "strata": {}}
    for strat in ("strict", "ambig"):
        d = [r[strat]["delta_med"] for r in rows]
        lo, hi, n = boot_ci(d)
        msw = [r[strat]["med_sw"] for r in rows if not math.isnan(r[strat]["med_sw"])]
        mno = [r[strat]["med_no"] for r in rows if not math.isnan(r[strat]["med_no"])]
        print(f"{strat:8s} {n:5d} {med(msw):9.3f} {med(mno):9.3f} {med(d):8.3f} "
              f"   [{lo:.3f}, {hi:.3f}] ms")
        out["strata"][strat] = {"n_artifacts": n, "med_of_med_sw": med(msw),
                                "med_of_med_no": med(mno), "delta_med_median": med(d),
                                "delta_med_ci95": [lo, hi]}
    ts = [r["top5pct_share"] for r in rows]
    print(f"\n꼬리 지배도(상위 5%가 총합서 차지): 중앙 {med(ts):.3f} · 범위 [{min(ts):.3f}, {max(ts):.3f}]")
    ar = [r["ambig_sw_residual_med"] for r in rows if not math.isnan(r["ambig_sw_residual_med"])]
    if ar:
        print(f"ambiguous 전환 간극에서 다른 스트림 실행분을 뺀 잔차(중앙): {med(ar):.3f} ms")
    out["top5pct_share_median"] = med(ts)
    out["ambig_sw_residual_med_median"] = med(ar) if ar else None
    p = os.path.join(HERE, "STEP0_SWITCH_GAP_2026-08-22.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
