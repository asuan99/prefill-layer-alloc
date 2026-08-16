# Fragment E — `S_itl` 충분점 민감도 (claims-auditor 자기정정 C2' 의 근거)
#
# PROVENANCE: claims-auditor 독립 재구현, 2026-08-16 (메인 세션 요청으로 추가 산출).
#   원문 그대로 — 리팩터링 금지.
# RUN: cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched \
#      && python3 audit_g16_independent_2026-08-16/fragE_sitl_sensitivity.py
# ★이 스크립트가 보인 것: `S_itl`(= min{decode SM : M_itl <= SLO})은 60 ms 운영점에서
#   knife-edge다 — 34-밴드 상단 경계까지 0.203 ms(+0.339%)뿐. 따라서 `Δ_SLO`를 단일
#   60 ms 점추정으로 공동 1차에 올리면 metric cliff(확인 #5)가 새 결정량에 재발한다.
#   ⇒ 사전등록 rev3은 `Δ_SLO(·)`를 **SLO 사다리의 함수**로 등록한다.
# 소비처: PREREG_G16_RULES_REV3_2026-08-16.md §4 1차-A · §6 S_ITL_UNREACHED · K8

# Fragment E: S_itl = min{decode SM : M_itl <= SLO} sensitivity on the OLD 4-arm grid.
# INDEPENDENT reimplementation (does not import pdmux_eval).
import json, glob, statistics
def pctl(v, q):
    o = sorted(float(x) for x in v)
    if not o: return float('nan')
    p = (len(o) - 1) * q; lo = int(p // 1); hi = min(lo + 1, len(o) - 1)
    return o[lo] + (o[hi] - o[lo]) * (p - lo)
def M_itl_of(fn):
    p95 = []
    for ln in open(fn):
        ln = ln.strip()
        if not ln: continue
        r = json.loads(ln); T = r.get("ttfts") or []; I = r.get("itls") or []
        for i, _t in enumerate(T):
            tok = [1000.0 * x for x in (I[i] if i < len(I) else [])]
            p95.append(pctl(tok, 0.95) if tok else float('inf'))
    return statistics.median(p95)
ARMS = [("d16",16),("d24",24),("d34",34),("d44",44)]
per_boot = {}
for arm,_sm in ARMS:
    fs = sorted(glob.glob(f"sgptvHi_{arm}_rep*_L3H12_*.jsonl"))
    per_boot[arm] = [(f.split("_")[-1].replace(".jsonl",""), M_itl_of(f)) for f in fs]
print("=== [E1] HI per-boot M_itl (median per-request ITL p95, ms), with job id ===")
for arm,_ in ARMS:
    print(f"  {arm}: " + "  ".join(f"{j}:{v:.4f}" for j,v in per_boot[arm]))
    print(f"        mean {statistics.mean(v for _,v in per_boot[arm]):.4f}  sd {statistics.stdev([v for _,v in per_boot[arm]]):.4f}  max {max(v for _,v in per_boot[arm]):.4f}")
def S_itl(slo, mode):
    for arm, sm in ARMS:
        vals = [v for _, v in per_boot[arm]]
        ok = (statistics.mean(vals) <= slo) if mode == "mean" else all(v <= slo for v in vals)
        if ok: return sm
    return None
print("\n=== [E2] S_itl on old 4-arm grid; D_ttft = 44 (canonical, ORACLE 3-4 bootstrap 10000/10000) ===")
print(f"{'ITL SLO':>8} {'S_itl(mean)':>12} {'S_itl(all4)':>12} {'D_SLO(mean)':>12} {'D_SLO(all4)':>12}")
for slo in (55, 60, 70):
    a, b = S_itl(slo, "mean"), S_itl(slo, "all")
    f = lambda s: ("UNREACHED" if s is None else f"{s}")
    g = lambda s: ("undef" if s is None else f"{s-44:+d}")
    print(f"{slo:>8} {f(a):>12} {f(b):>12} {g(a):>12} {g(b):>12}")
print("\n=== [E3] exact breakpoints of S_itl (mean convention) = the arm means themselves ===")
for arm, sm in ARMS:
    print(f"  S_itl = {sm:>2}  iff  SLO in [{statistics.mean(v for _,v in per_boot[arm]):.4f}, ...)  until next arm's mean")
print("  -> bands: [58.6470,58.8500)->44 (width 0.203 ms) | [58.8500,60.2030)->34 (1.353) | [60.2030,83.5720)->24 | >=83.5720 ->16 | <58.6470 -> UNREACHED")
print("\n=== [E4] knife-edge at the 60 ms operating point ===")
d24 = [v for _, v in per_boot["d24"]]; d34 = [v for _, v in per_boot["d34"]]
print(f"  d24 misses 60 ms by {min(v-60 for v in d24):.4f}..{max(v-60 for v in d24):.4f} ms  (0/4 boots pass)")
print(f"  d34 clears 60 ms by {min(60-v for v in d34):.4f}..{max(60-v for v in d34):.4f} ms  (4/4 boots pass)")
print(f"  S_itl 34->24 flips at SLO = {statistics.mean(d24):.4f} ms (mean conv.) = +{100*(statistics.mean(d24)-60)/60:.3f}% of 60")
print(f"  S_itl 34->24 flips at SLO = {max(d24):.4f} ms (all-4 conv.)  = +{100*(max(d24)-60)/60:.3f}% of 60")
print(f"  S_itl 34->44 flips at SLO = {statistics.mean(d34):.4f} ms (mean conv.) = {100*(statistics.mean(d34)-60)/60:+.3f}% of 60")
