"""C-5(1) regression: a perfectly saturated grid must NOT identify a donor,
and must reach ITL_SATURATED.  Also checks the fix cannot CREATE an
identification (monotone) on a clean grid."""
import sys, math
sys.path.insert(0,'/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched')
import g16_analyze as G

ARMS = ["d16","d24","d34","d44","d54","d64","d74"]

def mk(vals_by_arm, blocks=4):
    """vals_by_arm: arm -> list of per-block M_itl values."""
    recs=[]
    for a in ARMS:
        for b in range(blocks):
            v = vals_by_arm[a][b]
            recs.append(G.BootRecord(arm=a, decode_sm=G.arm_sm(a),
                block=f"blk{b+1}", boot=b+1, phase="HI", path=f"synthetic/{a}/{b}",
                est={"M_itl": v, "M_ttft": 1000.0 + 10*G.arm_sm(a),
                     "itl_pass": 1.0, "ttft_pass": 1.0, "joint_pass": 1.0,
                     "n_req": 200, "residency_fraction": {}, "switch_count": 0}))
    return recs

# --- case 1: PERFECT TIE (every arm, every block identical)
tie = {a:[50.0]*4 for a in ARMS}
d = G.decide(mk(tie), label="perfect tie")
print("=== case 1: perfect tie ===")
print("  D_itl identified:", d["D_itl"]["identified"], "| arm:", d["D_itl_arm"])
print("  rank blocks_won:", d["D_itl"]["rank_rule"]["blocks_won"],
      "n_blocks_tied:", d["D_itl"]["rank_rule"]["n_blocks_tied"])
print("  bootstrap frac:", d["D_itl"]["bootstrap_rule"]["fraction"],
      "tied_frac:", d["D_itl"]["bootstrap_rule"]["tied_fraction"])
print("  gap_upper:", d["saturation"]["gap_upper_ms"], "basis:", d["saturation"]["upper_arm_basis"])
print("  verdict:", d["verdict"], "| delta_citable:", d["delta_citable"])
ok1 = (not d["D_itl"]["identified"]) and d["verdict"]=="ITL_SATURATED"
print("  EXPECT: not identified AND ITL_SATURATED ->", "PASS" if ok1 else "FAIL")

# --- case 2: clean separation, no ties -> must still identify
clean = {a:[60.0 - 2.0*i + 0.01*b for b in range(4)] for i,a in enumerate(ARMS)}
d2 = G.decide(mk(clean), label="clean")
print("=== case 2: clean separation (no ties) ===")
print("  D_itl identified:", d2["D_itl"]["identified"], "| arm:", d2["D_itl_arm"],
      "| blocks_won:", d2["D_itl"]["rank_rule"]["blocks_won"],
      "| n_tied:", d2["D_itl"]["rank_rule"]["n_blocks_tied"],
      "| boot frac:", d2["D_itl"]["bootstrap_rule"]["fraction"])
ok2 = d2["D_itl"]["identified"] and d2["D_itl_arm"]=="d74"
print("  EXPECT: identified d74 ->", "PASS" if ok2 else "FAIL")

# --- case 3: two arms exactly tied at the minimum in every block
part = {a:[60.0 - 2.0*i + 0.01*b for b in range(4)] for i,a in enumerate(ARMS)}
part["d64"] = list(part["d74"])           # exact tie at the min
d3 = G.decide(mk(part), label="tied minimum")
print("=== case 3: two arms exactly tied at the minimum ===")
print("  identified:", d3["D_itl"]["identified"], "| blocks_won:",
      d3["D_itl"]["rank_rule"]["blocks_won"], "| n_tied:",
      d3["D_itl"]["rank_rule"]["n_blocks_tied"], "| verdict:", d3["verdict"])
ok3 = not d3["D_itl"]["identified"]
print("  EXPECT: not identified (win came from tie-break) ->", "PASS" if ok3 else "FAIL")

print("\nOVERALL:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
