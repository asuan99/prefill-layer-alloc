import sys, json, os
from pathlib import Path
SB = Path(os.environ["PDMUX_PROJECT_ROOT"]) / "workspace/engine-port/results/stock_baseline"
sys.path.insert(0, str(SB))
import sb_spec as SP
import sb_runner as SR

v = sys.argv[1]; out = Path(sys.argv[2]); w1 = json.load(open(sys.argv[3]))
plan = SP.make_plan("J1")
boots = {b["boot_id"]: b for b in plan["boots"]}
sg = json.loads(json.dumps(boots["J1_lad_SG-stock"])); pm = json.loads(json.dumps(boots["J1_lad_PM-ref"]))
run = next(r for r in sg["runs"] if r["rate"] == 5.516 and r["seed"] == 201)
b = sg; cmd = list(sg["arm_def"]["cmd"])
if v == "B2":
    cmd.remove("--disable-piecewise-cuda-graph")
elif v == "B5":
    b = pm; cmd = list(pm["arm_def"]["cmd"])
    i = cmd.index("--pdmux-config-path"); del cmd[i:i + 2]; cmd.remove("--enable-pdmux")
elif v == "B1":
    cmd += ["--disable-overlap-schedule"]
elif v == "B3":
    cmd += ["--chunked-prefill-size", "-1"]
elif v == "B4":
    cmd += ["--mem-fraction-static", "0.6", "--max-mamba-cache-size", "48"]
else:
    raise SystemExit("variant?")
b["arm_def"]["cmd"] = cmd
b["boot_id"] = f"DIAG_{v}"; b["runs"] = [run]; b["role"] = "meas"; b.pop("floor", None); b.pop("extra_run", None)
out.mkdir(parents=True, exist_ok=True)
(out / "argv.json").write_text(json.dumps({"variant": v, "cmd": cmd, "env": b["arm_def"]["env"], "tree": b["arm_def"]["tree"], "run": run}, indent=1))
res = SR.run_boot(b, out, lambda bdir, a: SR.ProcessServer(bdir, a), SR.default_replay, w1, [], None, {})
print(json.dumps(res))
