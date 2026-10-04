#!/usr/bin/env python3
"""λ0 VESSL substrate port -- registration constants, variance block, budget.

WHAT THIS FILE IS.  The CPU-side companion of `lambda0_vessl.sh`.  It does NOT
re-implement any rev5 rule: the plan, cell rows, analyzer and label are the
rev5 files in `../lambda0_prereg/`, imported here (never copied).  This file adds
only what the substrate port needs and rev5 did not have:

  1. REGISTERED DIGESTS.  The rev5 decision path must be byte-identical to the
     bytes job 908623 ran (`lam0_908623/REGISTRATION_SHA256.txt`), and the three
     git-ignored decision INPUTS (job 907959) must be byte-identical to the
     ones `LAMBDA_INF_DECISION.txt` of 908623 hashed.  `--verify-digests` exits 3
     on any drift -- the runner aborts before a GPU second is spent.
  2. VARIANCE BLOCK (PREREG_LAMBDA0_VESSL sec 5).  Seeds 3 and 4 of the SAME
     ranking rev5 uses for seeds 1 and 2 (`lambda0_plan.ebar` over the plan's N
     multiset), applied to the SAME top rung rev5 repeats.  Asserted against
     the registered literals (251, 2630) so a scorer drift cannot change them.
  3. BUDGET / ADMISSION (prereg sec 6).  rev5's own `cell_duration_at` + `warmup_s`
     with a VESSL per-boot allowance instead of the KISTI 110 s residual.
  4. DESCRIPTIVE summaries: variance of the top-rung achieved rate (n=4 per
     shape) and a side-by-side table against job 908623 that NEVER pools.

Nothing here feeds the rev5 label.  `LAMBDA0_LABEL.json` is produced by the
unmodified `lambda0_label.py` over exactly the 11 rev5 cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV5 = HERE.parent / "lambda0_prereg"
sys.path.insert(0, str(REV5))
import lambda0_plan as P  # noqa: E402  (rev5, unmodified)
import lambda0_cells as C  # noqa: E402  (rev5, unmodified)
import lambda0_label as L  # noqa: E402  (rev5, unmodified)

# ------------------------------------------------------------------ registered
# (1) rev5 decision path + renderer + runner + prereg, exactly as job 908623
#     recorded them (lam0_908623/REGISTRATION_SHA256.txt, ADDENDUM sec B-2).
REV5_DIGESTS = {
    "lambda0_lambda_inf.py": "2c9c80111aa5c0bf9f35f5156cdbcfe7d5248a65199952953b3a95f19f94f861",
    "lambda0_plan.py": "2e51280c42e61f0e3ff7fdeccb1e7184a5091e228328d66d9d6d482e2893f86d",
    "lambda0_label.py": "75cc7a67898f49b298dfb0bd313e34194dc5faf7eeeb303a6c1c2c7f58c6efc7",
    "lambda0_analyze.py": "3b17a162272f614f40f68f53045db38a7e55b8457dafb7829423f57b659d409d",
    "lambda0_cells.py": "3dd4a1a0a78ab239acb209ca0bc4169108a2d279fa36f6075cafea19e00d4205",
    "lambda0_reachability.py": "8b49ca716748f6831b0cc8468cc1628ab2e8002134eb9869fcfbf4b647d4b303",
    "lambda0_mutation_check.py": "44b41cffef5229fe1b5375f2e462fe45db57ac478ef631f76da7ba20c374946e",
    "lambda0_cellprint.py": "222519d4812b0d90ac5ab53324c48e0a800e3aac41fb80d13a54a5d26d57f93b",
    "lambda0.sbatch": "c789af6ce8861ba56faeda96c3daf662f0bdcf1b380269c831bec8092d553416",
    "PREREG_LAMBDA0_REV5_2026-09-14.md": "8367b6ecdb6b4d795fac2e8aa9960805887f2512fe623fb01c1a87020082c214",
    "PREREG_LAMBDA0_REV5_ADDENDUM_2026-09-14.md": "632c2d7804f7fef03622f5287c4f7ae2dad019f605511bb896acba061061ab6d",
}
# (1b) the λ_inf predicate's inputs (job 907959), as 908623 hashed them
#      (lam0_908623/LAMBDA_INF_DECISION.txt `LAMBDA0_INPUT_SHA256_*`).  The first
#      three are git-ignored and reach the Job through /io/staging (prereg sec 2-5).
DECISION_INPUT_DIGESTS = {
    "srv_warmup.log": "18400ea7bf90db5851abe6ddd58f2d2b6af78e286168a358e67514aba561690a",
    "instrument/I3a_shapeA.jsonl": "8539b43a3e67d36cbb271757d2635f8281f5f11e241af18c8cc03ca90b6b12c8",
    "instrument/I3b_shapeB.jsonl": "7fbc81e7633114fd62b6d965441c415b99cba2501ee6e3388bfb615e97764b55",
    "instrument/I_log_offsets.txt": "29ac5e6ae4e411e5c21a5a03324f61360fa31af3faeddf2d20ec624a449cee50",
    "instrument/I3_max_running_req.txt": "654ee9da442fa353f59f11beb688fc7f76c8de62a6c18b2a181fdde2a27cc3ef",
}
STAGED_INPUTS = ("srv_warmup.log", "instrument/I3a_shapeA.jsonl",
                 "instrument/I3b_shapeB.jsonl")
# (1c) served config (unchanged since 7e924a2, 2026-09-09) and the descriptive
#      realized-partition estimator (E2 track, audited; not a decision input here).
CONFIG_REL = "results/longctx_conflict/probes/pdmux_homog5.yml"
CONFIG_DIGEST = "be6dd6473ff8884f429b4d0d825e9cbf0b42b59da0ad4b136b542105c533625c"
# (1d) model revision every KISTI campaign served (hf_hub_revisions.tsv) and that
#      stage_data.sh pinned on /data/hf (operating model sec 9).
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
MODEL_REVISION = "dc0661c829b14e5b9246c05cfa89094a0875e052"

# (2) variance block
VARIANCE_SEED_RANKS = (3, 4)
VARIANCE_SEEDS_REGISTERED = (251, 2630)
VARIANCE_SUFFIXES = ("_s3", "_s4")
N_VARIANCE_MIN = 4                     # CLAUDE.md methodology gate 3

# (3) budget -- PREREG_LAMBDA0_VESSL sec 6
RATE_USD_H = 1.48
HARD_CAP_S = 16200                     # 4.5 h from job_entry start = rev5 `--time 04:30:00`
TAIL_RESERVE_S = 600                   # label + summaries after the last cell
VESSL_BOOT_ALLOW_S = 240.0             # B4 boot+12 req 151 s (2026-10-04) + teardown 15 s + margin
ADMISSION_RATIO = 0.25                 # rev5's worst registered corner (BUDGET_RATIO_GRID[-1])
PREFLIGHT_ALLOW_S = 900.0              # rev5 sec 7: ~10 min CPU gates (8 workers)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ digests
def verify_digests(prereg_dir: Path, i3_job_dir: Path, engine_root: Path):
    """Return a list of drift strings (empty = byte-identical to 908623's inputs)."""
    bad = []
    for name, want in sorted(REV5_DIGESTS.items()):
        p = prereg_dir / name
        got = sha256(p) if p.exists() else "MISSING"
        if got != want:
            bad.append(f"rev5 {name}: {got} != registered {want}")
    for rel, want in sorted(DECISION_INPUT_DIGESTS.items()):
        p = i3_job_dir / rel
        got = sha256(p) if p.exists() else "MISSING"
        if got != want:
            bad.append(f"decision input {rel}: {got} != registered {want}")
    cfg = engine_root / CONFIG_REL
    got = sha256(cfg) if cfg.exists() else "MISSING"
    if got != CONFIG_DIGEST:
        bad.append(f"config {CONFIG_REL}: {got} != registered {CONFIG_DIGEST}")
    return bad


def manifest_entries(path: Path) -> dict:
    """runtime_source_manifest -> {path relative to the SGLang python/ dir: sha}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, p = line.split(None, 1)
        key = p.split("/python/", 1)[1] if "/python/" in p else p
        out[key] = h
    return out


def engine_tree_diff(this_manifest: Path, ref_manifest: Path):
    a, b = manifest_entries(this_manifest), manifest_entries(ref_manifest)
    return [f"{k}: this={a.get(k, 'ABSENT')[:16]} 908623={b.get(k, 'ABSENT')[:16]}"
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


# ------------------------------------------------------------------ variance block
def ranked_seeds(plan: dict, k: int = 4):
    """Same scorer as `lambda0_plan.choose_seeds` (max |Ebar-1| over the plan's
    N multiset), extended to rank k.  Ranks 1-2 are asserted equal to the plan's."""
    ns = [c["num_prompts"] for s in ("A", "B") for c in plan["shapes"][s]["cells"]]
    scored = sorted((max(abs(P.ebar(s, n) - 1.0) for n in ns), s)
                    for s in P.SEED_CANDIDATES)
    top = [s for _, s in scored[:k]]
    assert top[0] == plan["seed1"] and top[1] == plan["seed2"], (top, plan["seed1"], plan["seed2"])
    return top, [sc for sc, _ in scored[:k]]


def variance_rows(plan: dict):
    top, _ = ranked_seeds(plan, max(VARIANCE_SEED_RANKS))
    seeds = tuple(top[r - 1] for r in VARIANCE_SEED_RANKS)
    assert seeds == VARIANCE_SEEDS_REGISTERED, (seeds, VARIANCE_SEEDS_REGISTERED)
    rows = []
    # interleave shapes so neither shape's extra repeats sit at the very end
    for suf, seed in zip(VARIANCE_SUFFIXES, seeds):
        for s in ("A", "B"):
            top_rung = plan["shapes"][s]["cells"][P.SEED_REPEAT_RUNG]
            rows.append(C._row(dict(top_rung, name=top_rung["name"] + suf), s, seed))
    return rows


def _base_cell(plan: dict, name: str) -> dict:
    base = name
    for suf in ("_s2",) + VARIANCE_SUFFIXES:
        if name.endswith(suf):
            base = name[: -len(suf)]
    for s in ("A", "B"):
        for c in plan["shapes"][s]["cells"]:
            if c["name"] == base:
                return c
    raise KeyError(name)


def cell_cost_s(plan: dict, name: str, ratio: float, boot_s: float = VESSL_BOOT_ALLOW_S) -> float:
    c = _base_cell(plan, name)
    lam_star = ratio * plan["lambda_inf"][c["shape"]]
    return P.cell_duration_at(c, lam_star) + P.warmup_s(c["shape"]) + boot_s


def budget(plan: dict, ratio: float, boot_s: float = VESSL_BOOT_ALLOW_S) -> dict:
    rev5 = [r.split()[0] for r in C.rows(plan)]
    var = [r.split()[0] for r in variance_rows(plan)]
    t5 = sum(cell_cost_s(plan, n, ratio, boot_s) for n in rev5)
    tv = sum(cell_cost_s(plan, n, ratio, boot_s) for n in var)
    tot = PREFLIGHT_ALLOW_S + t5 + tv
    return {"ratio": ratio, "boot_allow_s": boot_s,
            "preflight_s": PREFLIGHT_ALLOW_S, "rev5_cells_s": t5,
            "variance_cells_s": tv, "total_s": tot, "total_gpu_h": tot / 3600.0,
            "usd": tot / 3600.0 * RATE_USD_H,
            "fits_hard_cap": tot <= HARD_CAP_S - TAIL_RESERVE_S,
            "rev5_part_fits_hard_cap": PREFLIGHT_ALLOW_S + t5 <= HARD_CAP_S - TAIL_RESERVE_S}


def budget_table(plan: dict) -> dict:
    tab = {f"lambda_star_over_lambda_inf={r:.2f}": budget(plan, r)
           for r in P.BUDGET_RATIO_GRID}
    # reference scenario only (NOT a prior for this substrate): the KISTI 908623
    # λ* values expressed as ratios of the registered λ_inf (3.053/2.10, 0.697/0.675)
    ref = {}
    for s, lam in (("A", 3.053130), ("B", 0.697240)):
        ref[s] = lam / plan["lambda_inf"][s]
    tab["reference_908623_lambda_star(A,B ratios differ)"] = {
        "note": "per-shape ratios; computed cell by cell",
        "total_s": PREFLIGHT_ALLOW_S + sum(
            cell_cost_s(plan, r.split()[0], ref[r.split()[1]])
            for r in C.rows(plan) + variance_rows(plan)),
    }
    t = tab["reference_908623_lambda_star(A,B ratios differ)"]
    t["total_gpu_h"] = t["total_s"] / 3600.0
    t["usd"] = t["total_gpu_h"] * RATE_USD_H
    return {"hard_cap_s": HARD_CAP_S, "hard_cap_usd": HARD_CAP_S / 3600.0 * RATE_USD_H,
            "tail_reserve_s": TAIL_RESERVE_S, "admission_ratio": ADMISSION_RATIO,
            "rate_usd_h": RATE_USD_H, "scenarios": tab}


# ------------------------------------------------------------------ summaries
def variance_summary(out_dir: Path) -> dict:
    """n=4 top-rung repeats per shape: s1 (rev5 top rung), s2 (rev5 repeat),
    s3/s4 (variance block).  DESCRIPTIVE; never fed to the rev5 label."""
    res = {"definition": "achieved_rate of the TOP rung at seeds 1..4 "
                         "(rev5 a_r4/b_r3 + _s2 + variance _s3/_s4); each seed its own boot",
           "n_min": N_VARIANCE_MIN, "shapes": {}}
    label_p = out_dir / "LAMBDA0_LABEL.json"
    label = json.loads(label_p.read_text()) if label_p.exists() else None
    resumed = {}
    rp = out_dir / "RESUMED_CELLS.tsv"
    if rp.exists():
        for line in rp.read_text().splitlines()[1:]:
            f = line.split("\t")
            if f and f[0]:
                resumed[f[0]] = f[1] if len(f) > 1 else "?"
    res["resumed_cells"] = resumed
    res["mixed_jobs"] = bool(resumed)
    for s, top in (("A", "a_r4"), ("B", "b_r3")):
        recs = []
        for name, sub in ((top, ""), (top + "_s2", ""), (top + "_s3", "variance"),
                          (top + "_s4", "variance")):
            p = (out_dir / sub / f"cell_{name}.json") if sub else (out_dir / f"cell_{name}.json")
            if p.exists():
                c = json.loads(p.read_text())
                guard = L._cell_invalid(c)
                recs.append({"cell": name, "seed": c["client_seed"],
                             "from_job": resumed.get(name, "this"),
                             "achieved_rate": c["achieved_rate"],
                             "ratio": c[L.RATIO_KEY],
                             "saturated": c[L.RATIO_KEY] <= L.ACH_LO,
                             "r0_invalid": guard})
            else:
                recs.append({"cell": name, "missing": True})
        ok = [r for r in recs if not r.get("missing") and not r["r0_invalid"] and r["saturated"]]
        x = [r["achieved_rate"] for r in ok]
        d = {"cells": recs, "n_usable": len(x)}
        if len(x) >= 2:
            m = statistics.fmean(x)
            sd = statistics.stdev(x)
            d.update(mean=m, sd=sd, cv_pct=100.0 * sd / m,
                     range_pct_of_mean=100.0 * (max(x) - min(x)) / m)
        d["verdict"] = ("N_MET" if len(x) >= N_VARIANCE_MIN else
                        "UNRESOLVED_N_BELOW_4 (measurement shortfall, not a finding)")
        if label:
            lam = label["R0_R1_R2_R3_by_shape"].get(s, {}).get("lambda_star")
            d["rev5_lambda_star"] = lam
            if lam:
                d["max_rel_dev_vs_rev5_lambda_star_pct"] = (
                    100.0 * max(abs(v - lam) / lam for v in x) if x else None)
        res["shapes"][s] = d
    return res


def compare_908623(out_dir: Path, ref_dir: Path) -> dict:
    """Side by side ONLY.  No pooled mean, no ratio is a verdict (prereg sec 4)."""
    rows = []
    for p in sorted(ref_dir.glob("cell_*.json")):
        name = p.stem[len("cell_"):]
        r = json.loads(p.read_text())
        q = out_dir / p.name
        v = json.loads(q.read_text()) if q.exists() else None
        rows.append({"cell": name,
                     "kisti_908623_achieved": r["achieved_rate"],
                     "vessl_achieved": v["achieved_rate"] if v else None,
                     "kisti_ratio": r[L.RATIO_KEY],
                     "vessl_ratio": v[L.RATIO_KEY] if v else None})
    out = {"POLICY": "COMPARE ONLY -- different substrate; never pooled, averaged "
                     "or used to fill a missing cell (PREREG_LAMBDA0_VESSL sec 4)",
           "cells": rows}
    for tag, d in (("kisti_908623", ref_dir), ("vessl", out_dir)):
        lp = d / "LAMBDA0_LABEL.json"
        if lp.exists():
            lab = json.loads(lp.read_text())["R0_R1_R2_R3_by_shape"]
            out[tag + "_labels"] = {s: {"verdict": v["verdict"], "lambda_star": v["lambda_star"],
                                        "seed_repeat": v["seed_repeat"].get("verdict")}
                                    for s, v in lab.items()}
    return out


# ------------------------------------------------------------------ selftest
def selftest() -> None:
    p = json.loads((REV5 / "lam0_908623" / "plan.json").read_text())
    # the plan rev5 renders today is the plan 908623 ran
    p_now = P.plan(2.10, 0.675, ladders={k: tuple(v) for k, v in P.FALLBACK_LADDER.items()})
    assert json.dumps(p_now, sort_keys=True) == json.dumps(p, sort_keys=True)
    vr = variance_rows(p)
    assert [r.split()[0] for r in vr] == ["a_r4_s3", "b_r3_s3", "a_r4_s4", "b_r3_s4"], vr
    assert [int(r.split()[6]) for r in vr] == [251, 251, 2630, 2630], vr
    # the extra rows are the top rung's rows with only name+seed changed
    rev5 = {r.split()[0]: r.split() for r in C.rows(p)}
    for r in vr:
        f = r.split()
        base = rev5[f[0][:-3]]
        assert f[1:6] == base[1:6] and f[7:] == base[7:], (f, base)
    # every variance seed clears the analyzer's Ebar guard at its N (R0 validity)
    for r in vr:
        f = r.split()
        eb = P.ebar(int(f[6]), int(f[5]))
        assert 0.95 <= 1.0 / eb <= 1.05, (r, eb)
    # budget: rev5 part fits the cap even at the worst registered corner
    b = budget(p, ADMISSION_RATIO)
    assert b["rev5_part_fits_hard_cap"], b
    # admission cost is monotone in the ratio (lower true capacity => costlier)
    assert cell_cost_s(p, "a_r4", 0.25) > cell_cost_s(p, "a_r4", 1.0)
    # rev5's own registered budget is reproduced when the boot allowance is 110 s
    want = P.budget(p, 1.0)["total_s"]
    got = sum(cell_cost_s(p, r.split()[0], 1.0, P.BOOT_TEARDOWN_S) for r in C.rows(p))
    assert abs(got - want) < 1e-6, (got, want)
    print("LAM0V PLAN SELFTEST OK (908623 plan reproduced; variance rows = top rung x "
          "seeds 251/2630, Ebar-guard clean; rev5 budget reproduced at 110 s/boot; "
          "rev5 part fits the 4.5 h cap at the 0.25x corner)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-digests", nargs=3, metavar=("PREREG_DIR", "I3_JOB_DIR", "ENGINE_ROOT"))
    ap.add_argument("--variance-rows", metavar="PLAN_JSON")
    ap.add_argument("--budget", metavar="PLAN_JSON")
    ap.add_argument("--need", nargs=2, metavar=("PLAN_JSON", "CELL"))
    ap.add_argument("--engine-diff", nargs=2, metavar=("THIS_MANIFEST", "REF_MANIFEST"))
    ap.add_argument("--variance-summary", metavar="OUT_DIR")
    ap.add_argument("--compare-908623", nargs=2, metavar=("OUT_DIR", "REF_DIR"))
    ap.add_argument("--print-digests", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.verify_digests:
        bad = verify_digests(*(Path(x) for x in a.verify_digests))
        for b in bad:
            print("DRIFT " + b)
        print(f"DIGESTS {'OK' if not bad else 'DRIFT'} "
              f"({len(REV5_DIGESTS)} rev5 files, {len(DECISION_INPUT_DIGESTS)} decision inputs, 1 config)")
        return 3 if bad else 0
    if a.print_digests:
        for k, v in sorted(REV5_DIGESTS.items()):
            print(f"{v}  rev5/{k}")
        for k, v in sorted(DECISION_INPUT_DIGESTS.items()):
            print(f"{v}  job_907959/{k}")
        print(f"{CONFIG_DIGEST}  {CONFIG_REL}")
        return 0
    if a.variance_rows:
        for r in variance_rows(json.loads(Path(a.variance_rows).read_text())):
            print(r)
        return 0
    if a.budget:
        print(json.dumps(budget_table(json.loads(Path(a.budget).read_text())),
                         indent=2, sort_keys=True))
        return 0
    if a.need:
        p = json.loads(Path(a.need[0]).read_text())
        print(int(math.ceil(cell_cost_s(p, a.need[1], ADMISSION_RATIO))))
        return 0
    if a.engine_diff:
        d = engine_tree_diff(Path(a.engine_diff[0]), Path(a.engine_diff[1]))
        for x in d:
            print("ENGINE_DIFF " + x)
        print(f"ENGINE_TREE {'IDENTICAL' if not d else 'DIFFERS'} vs 908623 ({len(d)} entries differ)")
        return 3 if d else 0
    if a.variance_summary:
        print(json.dumps(variance_summary(Path(a.variance_summary)), indent=2, sort_keys=True))
        return 0
    if a.compare_908623:
        print(json.dumps(compare_908623(Path(a.compare_908623[0]), Path(a.compare_908623[1])),
                         indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
