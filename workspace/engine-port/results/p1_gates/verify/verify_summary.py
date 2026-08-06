"""Consolidated tables: where do the percentile bootstrap and the paired t
disagree, on the canonical predicate and across the C-3 threshold ladder?"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    res = json.loads((HERE / "c2_c3_results.json").read_text())
    pv = {(r["model"], r["rate"]): r for r in json.loads((HERE / "c2_pvalues.json").read_text())}
    bd = {(r["model"], r["rate"]): r for r in json.loads((HERE / "c3_boundary.json").read_text())["boundary"]}

    print("### TABLE 1 -- canonical predicate (T=60ms), n=5 paired reps")
    print(f"{'cell':28s}{'eff%':>9s}{'signs':>7s}{'boot CI95':>21s}{'t CI95':>21s}"
          f"{'p_t':>8s}{'p_perm':>8s}{'verdict change':>16s}")
    for r in res["c2"]:
        k = (r["model"], r["rate"])
        p = pv[k]
        change = ("boot YES -> t NO" if r["boot_excludes_0"] and not r["t_excludes_0"]
                  else ("agree" if r["boot_excludes_0"] == r["t_excludes_0"] else "boot NO -> t YES"))
        print(f"{r['model'][:18]+' r'+str(int(r['rate'])):28s}{r['effect_percent']:+9.2f}"
              f"{p['n_pos_diffs']:>5d}/5"
              f" [{r['boot_lo']:+7.4f},{r['boot_hi']:+7.4f}]"
              f" [{r['t_lo']:+7.4f},{r['t_hi']:+7.4f}]"
              f"{p['p_t']:8.4f}{p['p_perm_exact']:8.4f}{change:>16s}")

    print("\n### TABLE 2 -- C-3 ladder: sign and vacuity")
    print(f"{'cell':28s}{'sign@40..100':>13s}{'sign@150':>9s}{'sign@300':>9s}"
          f"{'flip T(ms)':>11s}{'both-arm ITL-viol=0 above':>26s}{'viol@300 (p/a)':>16s}")
    for model in sorted({r["model"] for r in res["c3_ladder_effects"]}):
        for rate in (2.0, 3.0, 4.0, 6.0):
            rows = {r["T"]: r for r in res["c3_ladder_effects"]
                    if r["model"] == model and r["rate"] == rate}
            s_low = {rows[T]["sign"] for T in (40.0, 48.0, 50.0, 60.0, 80.0, 100.0)}
            s = lambda x: "+" if x > 0 else ("-" if x < 0 else "0")  # noqa: E731
            b = bd[(model, rate)]
            print(f"{model[:18]+' r'+str(int(rate)):28s}"
                  f"{('+' if s_low == {1} else str(s_low)):>13s}"
                  f"{s(rows[150.0]['sign']):>9s}{s(rows[300.0]['sign']):>9s}"
                  f"{(('%.1f' % b['T_signflip_ms']) if b['T_signflip_ms'] else 'none'):>11s}"
                  f"{b['T_vacuous_ms']:26.2f}"
                  f"{str(rows[300.0]['viol_plain']) + '/' + str(rows[300.0]['viol_agn']):>16s}")

    print("\n### TABLE 3 -- ladder cells where the two intervals disagree")
    n_dis = 0
    for r in res["c3_ladder_effects"]:
        if r["boot_excludes_0"] != r["t_excludes_0"]:
            n_dis += 1
            print(f"  {r['model'][:18]:20s} r{r['rate']:.0f} T={r['T']:5.0f}  "
                  f"eff {r['effect_percent']:+8.2f}%  boot[{r['boot_lo']:+7.4f},{r['boot_hi']:+7.4f}]"
                  f"  t[{r['t_lo']:+7.4f},{r['t_hi']:+7.4f}]  "
                  f"({'boot fires, t does not' if r['boot_excludes_0'] else 't fires, boot does not'})")
    tot = len(res["c3_ladder_effects"])
    print(f"  -> {n_dis}/{tot} ladder cells disagree; all in the direction "
          f"'bootstrap fires, t does not'"
          if all(r["boot_excludes_0"] or not r["t_excludes_0"]
                 for r in res["c3_ladder_effects"]) else "")


if __name__ == "__main__":
    main()
