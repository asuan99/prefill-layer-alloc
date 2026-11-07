#!/usr/bin/env python3
"""Scores one nsys sqlite against the load's expected counts. Gates fixed before the first GPU run:
  G1 kernel rows per arm == expected (in graph-mode trace, graph arms instead need one graph row per replay)
  G2 plain / prefill / decode kernels carry distinct green context identity
  G3 GEMM mean duration prefill / decode >= 2.0 at a 16/92 split (the lane reflects the partition)
  G4 graph rows or node rows carry graphId / graphNodeId
  G5 concurrent arm: kernels on the two partitions overlap in time
Usage: gpu_obs_green_score.py <trace.sqlite> <load.json> <graph|node>"""
import json
import sqlite3
import sys

GEMM_RATIO_MIN = 2.0


def score(db_path, load, mode):
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    windows = {}
    for text, tid, start, end in db.execute("select text, textId, start, end from NVTX_EVENTS"):
        if tid is not None:
            text = db.execute("select value from StringIds where id=?", (tid,)).fetchone()[0]
        if text and text.startswith("arm:") and end is not None:
            windows[text[4:]] = (start, end)
    rows = {}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        q = """select k.start, k.end, k.contextId, k.greenContextId, k.streamId, k.graphId, k.graphNodeId, s.value
               from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id"""
        all_k = db.execute(q).fetchall()
    else:
        all_k = []
    graph_rows = []
    for t in sorted(t for t in tables if "GRAPH_TRACE" in t):
        graph_rows += db.execute(f'select start, end, graphId from "{t}"').fetchall()
    arms = {}
    for name, (s, e) in sorted(windows.items()):
        k = [r for r in all_k if s <= r[0] and r[1] <= e]
        rows[name] = k
        durs = [r[1] - r[0] for r in k]
        arms[name] = {
            "kernels": len(k),
            "expected": load["arms"].get(name, {}).get("expected_kernels"),
            "graph_rows": sum(1 for g in graph_rows if s <= g[0] and g[1] <= e),
            "ctx": sorted({r[2] for r in k}),
            "green_ctx": sorted({r[3] for r in k}, key=str),
            "streams": sorted({r[4] for r in k}),
            "with_graph_node": sum(1 for r in k if r[6] is not None),
            "mean_dur_us": round(sum(durs) / len(durs) / 1e3, 2) if durs else None,
            "names": sorted({r[7][:40] for r in k})[:4],
        }

    gates = {}
    bad = []
    for name, a in arms.items():
        replays = load.get("replays", 50)
        is_graph = "graph" in name
        if is_graph and mode == "graph":
            if a["graph_rows"] != replays:
                bad.append(f"{name}: graph_rows {a['graph_rows']} != {replays}")
        elif a["expected"] is not None and a["kernels"] != a["expected"]:
            bad.append(f"{name}: kernels {a['kernels']} != {a['expected']}")
    missing = sorted(set(load["arms"]) - set(arms))
    if missing:
        bad.append(f"arms absent from trace: {missing}")
    gates["G1"] = {"pass": not bad, "detail": bad}

    ident = [(arms.get(n) or {}).get("green_ctx") for n in ("plain_eager", "green_p_eager", "green_d_eager")]
    gates["G2"] = {"pass": all(i is not None and len(i) == 1 for i in ident) and len({str(i) for i in ident}) == 3,
                   "detail": ident}

    def gemm(n):
        d = [r[1] - r[0] for r in rows.get(n, []) if "gemm" in r[7]]
        return sum(d) / len(d) if d else None
    gp, gd = gemm("green_p_eager"), gemm("green_d_eager")
    ratio = gp / gd if gp and gd else None
    gates["G3"] = {"pass": ratio is not None and ratio >= GEMM_RATIO_MIN, "detail": ratio}

    gr = [a for n, a in arms.items() if "graph" in n]
    if mode == "node":
        ok = bool(gr) and all(a["with_graph_node"] == a["kernels"] > 0 for a in gr)
    else:
        ok = bool(gr) and all(a["graph_rows"] > 0 for a in gr)
    gates["G4"] = {"pass": ok, "detail": [(a["kernels"], a["with_graph_node"], a["graph_rows"]) for a in gr]}

    c = rows.get("green_concurrent", [])
    by = {}
    for r in c:
        by.setdefault(r[3], []).append((r[0], r[1]))
    overlap = 0
    if len(by) == 2:
        (x, y) = [sorted(v) for v in by.values()]
        i = j = 0
        while i < len(x) and j < len(y):
            overlap += max(0, min(x[i][1], y[j][1]) - max(x[i][0], y[j][0]))
            if x[i][1] < y[j][1]:
                i += 1
            else:
                j += 1
    gates["G5"] = {"pass": overlap > 0, "detail": {"partitions": len(by), "overlap_ms": overlap / 1e6}}
    return {"mode": mode, "tables": sorted(t for t in tables if "GRAPH" in t), "arms": arms, "gates": gates}


if __name__ == "__main__":
    load = json.load(open(sys.argv[2]))
    print(json.dumps(score(sys.argv[1], load, sys.argv[3]), indent=1, default=str))
