#!/usr/bin/env python3
"""Quick stock-engine SLO-attainment ladders (user request 2026-11-02): default SGLang (and optionally default vLLM) on Nano,
same W1/W3 rate grids as CAP1-R/CAP2.  Descriptive only.  NOT PREREG STOCK-BASE: no paired goodput labels, no comparison arm in
this Job, so nothing here may be cited as a same-Job comparison.  Reuses the sealed harness modules unchanged (arm definitions,
trace, runner); the sealed sb_job.sh is not used because its Job list is fixed.

  sq_plan.py plan --arms SG-stock[,VL-stock]
  sq_plan.py run  --arms ... --out DIR --w1-lengths F --w3-tokens F [--resume]
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SB = HERE.parent / "stock_baseline"
sys.path.insert(0, str(SB))
import sb_spec as S      # noqa: E402
import sb_runner as R    # noqa: E402

ALLOWED = ("SG-stock", "VL-stock")
JOB = "SQ"
def make_plan(arms):
    bad = [a for a in arms if a not in ALLOWED]
    if bad or not arms or arms[0] != "SG-stock":
        raise SystemExit(f"ABORT_CONFIG: arms must start with SG-stock and be within {ALLOWED}; got {arms}")
    rng = S._rng(JOB)
    boots = []
    if "VL-stock" in arms:   # same-definition TTFT check between the engines (PREREG §5-0(a))
        boots += [S.boot(f"{JOB}_smoke_{e}", S.smoke_arm(e), [{"run_id": f"{JOB}_SMOKE_{e}", "kind": "smoke", "seed": S.SMOKE_SEED,
                                                              "n_requests": S.SMOKE_N, "cell": "SMOKE"}], role="smoke") for e in ("sglang", "vllm")]
    for i, name in enumerate(arms):
        w1 = [S.cap_run(f"{JOB}_{name}_W1lad_r{r}_s{s}", "W1", r, s, S.CAP_N, cell="LADDER") for r in S.ladder_rates() for s in S.CAP_SEEDS]
        rng.shuffle(w1)
        # W3 floor (SLO = 3 x floor) is measured once, on SG-stock, in its W1 boot; every arm is scored against that one SLO
        boots.append(S.boot(f"{JOB}_w1_{name}", S.arm(name), w1, floor=(i == 0)))
    for name in arms:
        w3 = [S.cap_run(f"{JOB}_{name}_W3lad_r{r}_s{s}", "W3", r, s, S.W3_LADDER_N, cell="W3LADDER") for r in S.w3_ladder_rates() for s in S.CAP_SEEDS]
        rng.shuffle(w3)
        boots.append(S.boot(f"{JOB}_w3_{name}", S.arm(name), w3, role="w3ladder"))
    return S.finish({"job": JOB, "kind": "LADDER", "arms": list(arms), "boots": boots, "pending": []})

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out"); ap.add_argument("--w1-lengths"); ap.add_argument("--w3-tokens"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    plan = make_plan(a.arms.split(","))
    problems = S.plan_problems(plan)
    if a.cmd == "plan":
        print(json.dumps({"plan_sha": S.sha(plan), "problems": problems, "boots": [b["boot_id"] for b in plan["boots"]],
                          "n_runs": sum(len(b["runs"]) for b in plan["boots"]), "est_min": plan["est_min"], "hard_cap_min": plan["hard_cap_min"]}))
        return 1 if problems else 0
    w1 = json.load(open(a.w1_lengths))
    w3 = R.T.load_w3_pool(a.w3_tokens)[0]
    res = R.run_job(plan, a.out, lambda bdir, arm_def: R.ProcessServer(bdir, arm_def), R.default_replay, w1, w3, a.w3_tokens, a.resume)
    rc = R.exit_code(res)
    if rc:
        print("BOOT_FAILURES: " + ",".join(f"{b['boot_id']}={b['status']}" for b in res if b["status"] != "done"), file=sys.stderr)
    return rc

if __name__ == "__main__":
    sys.exit(main())
