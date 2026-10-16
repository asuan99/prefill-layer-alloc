"""M3 PREREG rev12 harness (DRAFT r2 §19-12 + user decisions U7 / U12 of 2026-10-16) -- CPU only, no GPU, no model.

Every rule is pinned by a positive test AND by a mutation test (lesson 53): the mutant is the REAL source file with one unique
substitution (the site must occur exactly once, else the mutant would be vacuous), executed as a fresh module; each mutant must make
its check fail (KILLED) while the unmutated control passes.  Mutants (names = PREREG §19-12 / VR2 / U7 / U12 wording):

  E1  "서버 0.9% + 클라이언트 0.9%인데 라벨 그대로"            per-side 1 % instead of the summed 1 % (D-1' (a))
  E2  "우세 분류가 서버인데 CLIENT_ERROR로 셈"
  E3  "OUTPUT_SHORT를 성공으로 셈"
  E4  "한 arm X HI 셀의 가장 긴 3%가 서버 측 오류인데 라벨 그대로"   (server side does not count, the r1 C5 reading)
  J1  "H-3 X판" (X reference cell not saturated, no JOB_EFFECT_UNTESTED(J_N, X))
  J2  "앵커에 X 없음" (U14)          J3  F-5 anchor SERVER_ERROR_HI ignored
  P1  "X BROKEN이 A·B를 무효화"       P2  "A BROKEN이 B를 무효화하지 않음"     P3  "X를 빠뜨린 P10"          (U12)
  X1  "X 정확성 불통과인데 X 셀 라벨이 그대로" (U7)   X2/X3 the U7 D1>=D0-1 / first-N rules removed
  F1  "J1 floor로 J_N 채점"           F2  "config boot1 agnostic floor 블록 누락"     S1  "shape 2개로 계획"
  L1  "X에 [1]*L 송신"                L2  "A·B payload가 rev11과 다름"                L3  "X 경로에서 error 청크를 토큰으로 셈" (F-3)
  B1  "mg_boot context 검사 제거"     B2  "floor 요청 fit 검사 누락" (F-3)            B3  torch_dtype fallback (R-4)
Harness audit r1 (VERDICT_m3_rev12_harness_r1_2026-10-16 §5) repairs, auditor mutant names:
  R-H1  G1 "옛 diff --quiet HEAD로 되돌림" (mg_plan gate (i)), G2 the same in m3_job.sh check_committed, G3 "gate 깨끗함 검사 제거" (M6 re-sited)
  R-H2  G4 "FAIL_NEW를 PASS로 기록" (preflight_judge record)
  R-H3  M20 ref_x from A cells, M21 anchor_x from A cells, M22 X trend reads the A Job effect / anchors, M23 X correction from A arrays
  R-H4  M1 "출력 32토큰 미만 PASS", M12 DTYPE_UNVERIFIED dropped, M13 floor median over a partial set
  R-H5  M24 "logprob 부재 PASS"         (M8, equivalent end to end, is pinned at the job_effect() contract)
"""

import asyncio
import hashlib
import shutil
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
for p in (ENGINE / "results" / "multigpu", ENGINE / "scripts" / "multigpu", ENGINE / "benchmarks"):
    sys.path.insert(0, str(p))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import mg_boot as B  # noqa: E402
import mg_const as C  # noqa: E402
import mg_label as L  # noqa: E402
import mg_plan as P  # noqa: E402
import mg_x_pretokenize as XT  # noqa: E402
import mg_xcheck as XC  # noqa: E402
from pdmux_eval import trace_loadgen as TL  # noqa: E402

LABEL_SRC = ENGINE / "results/multigpu/mg_label.py"
PLAN_SRC = ENGINE / "results/multigpu/mg_plan.py"
BOOT_SRC = ENGINE / "scripts/multigpu/mg_boot.py"
XC_SRC = ENGINE / "scripts/multigpu/mg_xcheck.py"
M3_JOB_SH = ENGINE / "results/multigpu/m3_job.sh"
PF_TOOLS = ENGINE / "results/handoff_review_2026-10-15/merge_trial_tools"
PF_JUDGE_SRC = PF_TOOLS / "preflight_judge.py"
LOADGEN_SRC = ENGINE / "benchmarks/pdmux_eval/trace_loadgen.py"

try:
    from aiohttp import web
    HAVE_AIOHTTP = True
except ImportError:                                  # pragma: no cover
    HAVE_AIOHTTP = False


def variant(src_path, old, new, name, package=None):
    """The real module with ONE unique substitution, as a fresh module (old must occur exactly once)."""
    src = Path(src_path).read_text()
    n = src.count(old)
    if n != 1:
        raise AssertionError(f"mutation site for {name} occurs {n} times (must be 1)")
    mod = types.ModuleType(f"_mut_{name}")
    mod.__file__ = str(src_path)
    if package:
        mod.__package__ = package
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src.replace(old, new) if new is not None else src, str(src_path), "exec"), mod.__dict__)
    finally:
        del sys.modules[mod.__name__]
    return mod


def control(src_path, name, package=None):
    mod = types.ModuleType(f"_ctl_{name}")
    mod.__file__ = str(src_path)
    if package:
        mod.__package__ = package
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(Path(src_path).read_text(), str(src_path), "exec"), mod.__dict__)
    finally:
        del sys.modules[mod.__name__]
    return mod


class MutationCase(unittest.TestCase):
    """check(mod) -> None (pass) or a failure string.  The control (real source) must pass, the mutant must fail."""

    def assert_killed(self, src, name, old, new, check, package=None):
        ctl = check(control(src, name, package))
        self.assertIsNone(ctl, f"CONTROL fails for {name}: {ctl}")
        mut = check(variant(src, old, new, name, package))
        self.assertIsNotNone(mut, f"mutant {name} SURVIVED")
        if os.environ.get("MG_MUT_REPORT"):
            sys.stderr.write(f"MUTANT {name} KILLED ({Path(src).name}): {mut} | CONTROL passed\n")
        return mut


# =========================================================================== synthetic requests
def reqs_of(n, n_server=0, n_client=0, server_kind="SSE_ERROR", longest_fail=False, lengths=None, late=0):
    """n requests 0.1 s apart; ok requests: ttft 100 ms, completion +2 s.  Errors: server (error_side server) / client (exception)."""
    lens = lengths or [256] * n
    order = sorted(range(n), key=lambda i: -lens[i]) if longest_fail else list(range(n))
    srv, cli = set(order[:n_server]), set(order[n_server:n_server + n_client])
    out = []
    for i in range(n):
        r = {"request_id": f"r{i:05d}", "sent_s": 0.1 * i, "scheduled_arrival_s": 0.1 * i, "ttft_ms": 100.0,
             "completion_s": 0.1 * i + 2.0 + lens[i] / 4000.0, "error": "", "error_kind": "", "error_side": "", "input_tokens": lens[i]}
        if i in srv:
            if server_kind == "OUTPUT_SHORT":
                r.update(error="OUTPUT_SHORT: observed 3 != requested 9 (ignore_eos)", error_kind="OUTPUT_SHORT", error_side="server")
            else:
                r.update(error=f"{server_kind}: refused", error_kind=server_kind, error_side="server", ttft_ms=None)
        elif i in cli:
            r.update(error="ClientTimeout()", error_kind="CLIENT_EXCEPTION", error_side="client", ttft_ms=None)
        out.append(r)
    for i in range(late):                                  # successful but sent > 0.5 s late (client side, SP R7-3)
        k = n - 1 - i
        if not out[k]["error"]:
            out[k]["sent_s"] = out[k]["scheduled_arrival_s"] + 0.6
    return out


META_HI = {"cell_id": "c", "rung": "HI", "config": "N1", "arm": "agnostic", "shape": "X", "seed": 4386}


class ErrorRule(MutationCase):
    """§19-1b D-1' (a): summed 1 %, dominant label, both counts."""

    def test_summed_rule_dominant_label_and_counts(self):
        cases = [((9, 9), "CLIENT_ERROR"),         # 0.9 % + 0.9 % = 1.8 % > 1 %; a tie -> CLIENT_ERROR [제안]
                 ((12, 3), "SERVER_ERROR_HI"), ((3, 12), "CLIENT_ERROR"),
                 ((10, 0), "OK"), ((0, 10), "OK"), ((5, 5), "OK"),         # exactly 1 % is not "> 1 %"
                 ((6, 5), "SERVER_ERROR_HI")]
        for (s, c), want in cases:
            rec = L.analyze_cell(reqs_of(1000, s, c), META_HI)
            self.assertEqual(rec["status"], want, (s, c))
            self.assertEqual((rec["n_server_errors"], rec["n_client_errors"]), (s, c))

    def test_lo_cell_server_errors_are_descriptive_client_errors_invalidate(self):           # MG12-14
        lo = dict(META_HI, rung="LO")
        self.assertEqual(L.analyze_cell(reqs_of(1000, 50, 0), lo)["status"], "OK")
        self.assertEqual(L.analyze_cell(reqs_of(1000, 0, 20), lo)["status"], "CLIENT_ERROR")

    def test_late_send_counts_client_side(self):
        rec = L.analyze_cell(reqs_of(1000, 0, 0, late=11), META_HI)
        self.assertEqual((rec["n_client_errors"], rec["status"]), (11, "CLIENT_ERROR"))
        self.assertEqual(L.analyze_cell(reqs_of(1000, 0, 0, late=10), META_HI)["status"], "OK")

    def test_unclassified_error_is_client_side(self):
        r = reqs_of(100, 0, 0)
        r[0].update(error="legacy", error_side=None)
        r[1].update(ttft_ms=None)                                    # no TTFT, no error string -> unclassified -> client
        self.assertEqual(L.error_split(r), (0, 2))

    def _check_e1(self, mod):
        st = mod.analyze_cell(reqs_of(1000, 9, 9), META_HI)["status"]
        return None if st == "CLIENT_ERROR" else f"0.9%+0.9% labelled {st}"

    def test_mutant_E1_per_side_threshold_killed(self):
        self.assert_killed(LABEL_SRC, "E1", "    if (n_server + n_client) / n > C.CLIENT_ERROR_FRAC:",
                           "    if max(n_server, n_client) / n > C.CLIENT_ERROR_FRAC:", self._check_e1)

    def test_mutant_E2_dominant_server_counted_as_client_killed(self):
        def chk(mod):
            st = mod.analyze_cell(reqs_of(1000, 12, 3), META_HI)["status"]
            return None if st == "SERVER_ERROR_HI" else f"server-dominant labelled {st}"
        self.assert_killed(LABEL_SRC, "E2", '"SERVER_ERROR_HI" if n_server > n_client else "CLIENT_ERROR"', '"CLIENT_ERROR"', chk)

    def test_mutant_E3_output_short_as_success_killed(self):
        def chk(mod):
            rec = mod.analyze_cell(reqs_of(1000, 30, 0, server_kind="OUTPUT_SHORT"), META_HI)
            return None if rec["status"] == "SERVER_ERROR_HI" else f"3 % OUTPUT_SHORT labelled {rec['status']}"
        self.assert_killed(LABEL_SRC, "E3", '    return bool(r.get("error")) or r.get("ttft_ms") is None',
                           '    return (bool(r.get("error")) and r.get("error_kind") != "OUTPUT_SHORT") or r.get("ttft_ms") is None', chk)

    def _x_hi_cells(self, mod, fail_longest):
        lens = [200 + 37 * ((i * 7919) % 400) for i in range(200)]          # varied X-like input lengths
        cells = []
        for arm in ("fused_sym", "agnostic"):
            for sd in C.HI_SEEDS:
                rq = reqs_of(200, 6 if (arm == "agnostic" and fail_longest) else 0, 0, longest_fail=True, lengths=lens)
                rec = mod.analyze_cell(rq, dict(META_HI, config="DP2", arm=arm, seed=sd))
                rec["g"] = 0.4                                             # saturated by construction (the rule under test is the error rule)
                rec.update(meta=dict(config="DP2", arm=arm, shape="X", rung="HI", seed=sd, ext=0, round=1, block="main"),
                           harness_limited=False, frontend_saturated=False)
                cells.append(rec)
            for sd in C.LO_SEEDS:
                cells.append({"cell_id": f"lo{arm}{sd}", "status": "OK", "g": 0.01, "achieved": 1.0, "truncated": False,
                              "harness_limited": False, "frontend_saturated": False,
                              "meta": dict(config="DP2", arm=arm, shape="X", rung="LO", seed=sd, ext=0, round=1, block="main")})
        return cells

    def _check_e4(self, mod):
        doc = mod.aggregate(self._x_hi_cells(mod, True), xcheck={"verdict": "PASS"}, job="J2")
        e = doc["estimands"]["G_sym"]["DP2"]["X"]
        if e["defined"]:
            return f"G_sym(DP2, X) defined ({e.get('point')}) although the agnostic X HI cells lost their longest 3 % server-side"
        if not any("SERVER_ERROR_HI" in r for r in e["undefined_reasons"]):
            return f"no SERVER_ERROR_HI reason: {e['undefined_reasons']}"
        return None

    def test_x_hi_longest_3pct_server_errors_void_the_pair_and_control_pair_defined(self):
        self.assertIsNone(self._check_e4(L))
        doc = L.aggregate(self._x_hi_cells(L, False), xcheck={"verdict": "PASS"}, job="J2")
        self.assertTrue(doc["estimands"]["G_sym"]["DP2"]["X"]["defined"], doc["estimands"]["G_sym"]["DP2"]["X"])
        key = doc["keys"]["DP2|agnostic|X|r1"]
        self.assertIn("error_counts_hi", key)

    def test_mutant_E4_server_side_not_counted_killed(self):
        self.assert_killed(LABEL_SRC, "E4", "    if (n_server + n_client) / n > C.CLIENT_ERROR_FRAC:",
                           "    if n_client / n > C.CLIENT_ERROR_FRAC:", self._check_e4)


# =========================================================================== synthetic Job documents with shape X
def make_docs(dp, tp, x_ref=None, x_anchor=None, a_anchor=None, xcheck_reason=None):
    """dp/tp: {shape: [G(1), G(2), G(4), G(8)]} (G(1) from J1 N1).  x_ref / x_anchor / a_anchor: overrides."""
    docs = {"J1": {"estimands": {"G_sym": {"N1": {sh: L._syn_ent(v[0]) for sh, v in dp.items()}}}, "ref": L._syn_ref(),
                   "ref_x": x_ref if x_ref is not None else L._syn_ref(), "n_cells": 1, "anchor": {}}}
    for i, n in enumerate((2, 4, 8), start=1):
        est = {f"DP{n}": {sh: L._syn_ent(v[i]) for sh, v in dp.items()}, f"TP{n}": {sh: L._syn_ent(v[i]) for sh, v in tp.items()}}
        docs[f"J{n}"] = {"estimands": {"G_sym": est}, "ref": {}, "n_cells": 1,
                         "anchor": (a_anchor or {}).get(n, L._syn_anchor(1.0)),
                         "anchor_x": (x_anchor or {}).get(n, L._syn_anchor(1.0))}
    return docs


FLAT, UP = [1.0, 1.0, 1.0, 1.0], [1.0, 1.1, 1.2, 1.3]


class P10Groups(MutationCase):
    """U12 (user decision 2026-10-16): {A, B} bound as rev11, {X} alone."""

    def _scen1(self):        # X BROKEN only
        return make_docs({"A": FLAT, "B": FLAT, "X": UP}, {"A": UP, "B": UP, "X": UP})

    def _scen2(self):        # A BROKEN, B / X held
        return make_docs({"A": UP, "B": FLAT, "X": FLAT}, {"A": UP, "B": UP, "X": UP})

    @staticmethod
    def _nc(doc, sh):
        return bool(doc["trends"][f"G_sym|TP|{sh}"].get("MGP13_not_citable"))

    def test_x_broken_voids_only_x_and_a_broken_voids_b(self):
        d1 = L.aggregate_jobs(self._scen1())
        self.assertEqual((d1["P_DP"]["groups"]["AB"]["status"], d1["P_DP"]["groups"]["X"]["status"]), ("HELD", "BROKEN"))
        self.assertEqual((self._nc(d1, "A"), self._nc(d1, "B"), self._nc(d1, "X")), (False, False, True))
        self.assertEqual(d1["P_DP"]["status"], "HELD")                                        # the rev11-layout top level = the A/B group
        d2 = L.aggregate_jobs(self._scen2())
        self.assertEqual((self._nc(d2, "A"), self._nc(d2, "B"), self._nc(d2, "X")), (True, True, False))

    def _chk1(self, mod):
        d = mod.aggregate_jobs(self._scen1())
        got = (self._nc(d, "A"), self._nc(d, "B"), self._nc(d, "X"))
        return None if got == (False, False, True) else f"X-BROKEN scenario: TP not-citable (A, B, X) = {got}"

    def _chk2(self, mod):
        d = mod.aggregate_jobs(self._scen2())
        got = (self._nc(d, "A"), self._nc(d, "B"), self._nc(d, "X"))
        return None if got == (True, True, False) else f"A-BROKEN scenario: TP not-citable (A, B, X) = {got}"

    SITE = 'P10_GROUPS = {"AB": ("A", "B"), "X": ("X",)}'

    def test_mutant_P1_x_broken_voids_ab_killed(self):
        self.assert_killed(LABEL_SRC, "P1", self.SITE, 'P10_GROUPS = {"ABX": ("A", "B", "X")}', self._chk1)

    def test_mutant_P2_a_broken_does_not_void_b_killed(self):
        self.assert_killed(LABEL_SRC, "P2", self.SITE, 'P10_GROUPS = {"A": ("A",), "B": ("B",), "X": ("X",)}', self._chk2)

    def test_mutant_P3_p10_without_x_killed(self):
        self.assert_killed(LABEL_SRC, "P3", self.SITE, 'P10_GROUPS = {"AB": ("A", "B")}', self._chk1)


class XJobEffect(MutationCase):
    """U14 + F-5: the X anchor against J1 boot1 X; X reference cell not saturated -> every J_N JOB_EFFECT_UNTESTED(J_N, X)."""

    def test_clear_fired_and_untested(self):
        d = L.aggregate_jobs(make_docs({"A": FLAT, "B": FLAT, "X": FLAT}, {"A": FLAT, "B": FLAT, "X": FLAT}))
        self.assertEqual({J: v["status"] for J, v in d["job_effect_x"].items()}, {"J2": "CLEAR", "J4": "CLEAR", "J8": "CLEAR"})
        d = L.aggregate_jobs(make_docs({"A": FLAT, "B": FLAT, "X": FLAT}, {"A": FLAT, "B": FLAT, "X": FLAT},
                                       x_anchor={4: L._syn_anchor(1.05)}))
        self.assertEqual(d["job_effect_x"]["J4"]["status"], "FIRED")
        self.assertEqual(d["job_effect"]["J4"]["status"], "CLEAR")                       # the shape-A Job effect is separate
        self.assertTrue(any(x.startswith("SHAPE_A_JOB_EFFECT(") for x in d["trends"]["G_sym|DP|X"]["disclosures"]))

    def _not_sat_docs(self):
        return make_docs({"A": FLAT, "B": FLAT, "X": FLAT}, {"A": FLAT, "B": FLAT, "X": FLAT},
                         x_ref=L._syn_ref(fused_sym=dict(boot1_hi_label="HI_NOT_SATURATED")))

    def _chk_h3x(self, mod):
        d = mod.aggregate_jobs(self._not_sat_docs())
        st = {J: v["status"] for J, v in d["job_effect_x"].items()}
        if set(st.values()) != {"UNTESTED"}:
            return f"X reference not saturated but JOB_EFFECT(J_N, X) = {st}"
        if d["job_effect"]["J2"]["status"] != "CLEAR":
            return "shape-A Job effect changed"
        return None

    def test_mutant_J1_h3_x_killed(self):
        self.assert_killed(LABEL_SRC, "J1", '        if e.get("boot1_hi_label") != "HI_SATURATED":', "        if False:", self._chk_h3x)

    def _chk_f5(self, mod):
        anc = L._syn_anchor(1.0, server_error_hi=True)
        d = mod.aggregate_jobs(make_docs({"A": FLAT, "B": FLAT, "X": FLAT}, {"A": FLAT, "B": FLAT, "X": FLAT}, a_anchor={2: anc}))
        st = d["job_effect"]["J2"]
        return None if st["status"] == "UNTESTED" and any("SERVER_ERROR_HI" in r for r in st["reasons"]) else f"F-5: {st}"

    def test_mutant_J3_anchor_server_error_ignored_killed(self):
        self.assert_killed(LABEL_SRC, "J3", '        if e.get("server_error_hi"):', "        if False:", self._chk_f5)

    def test_reference_server_error_and_x_correctness_make_x_untested(self):
        d = L.aggregate_jobs(make_docs({"A": FLAT, "B": FLAT, "X": FLAT}, {"A": FLAT, "B": FLAT, "X": FLAT},
                                       x_ref=L._syn_ref(agnostic=dict(boot1_server_error_hi=True))))
        self.assertTrue(all(v["status"] == "UNTESTED" for v in d["job_effect_x"].values()))
        anc = L._syn_anchor(1.0)
        for a in anc["arms"].values():
            a["x_correctness"] = "X_CORRECTNESS_FAIL"
        d = L.aggregate_jobs(make_docs({"A": FLAT, "B": FLAT, "X": FLAT}, {"A": FLAT, "B": FLAT, "X": FLAT}, x_anchor={2: anc}))
        self.assertEqual(d["job_effect_x"]["J2"]["status"], "UNTESTED")
        self.assertEqual(d["job_effect_x"]["J4"]["status"], "CLEAR")

    def test_mutant_J2_anchor_without_x_killed(self):
        def chk(mod):
            c = mod.m3_plan_counts()
            return None if (c["x_anchor_cells"], c["cells"]) == (24, 560) else f"plan {c}"
        self.assert_killed(PLAN_SRC, "J2", "C.ANCHOR_SHAPES, ext=0", "(C.ANCHOR_SHAPE,), ext=0", chk)


# =========================================================================== plan
class Plan12(MutationCase):
    def test_counts_560_floor_20_xcheck_and_order(self):
        c = P.assert_plan_registered()
        self.assertEqual((c["cells"], c["floor_blocks"], c["x_anchor_cells"], c["xcheck_steps"], c["boots"]), (560, 20, 24, 14, 70))
        for job in ("J2", "J4", "J8"):
            for b in P.boots_anchor(job):                                                    # anchors: floor, no xcheck (N=1)
                self.assertEqual(b["steps"][0]["type"], "floor")
                self.assertFalse(any(str(s.get("name", "")).startswith("xcheck") for s in b["steps"]))
        for cfg in ("N1", "TP2", "DP2", "TP4", "DP4", "TP8", "DP8"):                          # U7 on every config boot1 (N-GPU server)
            for b in P.boots_boot1(cfg):
                kinds = [s["type"] if s["type"] != "cmd" else s["name"] for s in b["steps"][:2]]
                want = ["floor", f"xcheck_{b['arm']}"] if b["arm"] in ("fused_sym", "agnostic") else ["warmup", "cell"]
                self.assertEqual(kinds, want, (cfg, b["arm"]))
                self.assertEqual(b["gpu_ids"], C.GPU_SUBSETS[cfg])
        self.assertTrue(all(sorted(P.shape_order(c_, "fused", "t")) == ["A", "B", "X"] for c_ in C.CONFIGS))

    def test_mutant_X4_xcheck_only_at_n1_killed(self):
        def chk(mod):
            tp = [s for b in mod.boots_boot1("TP8") if b["arm"] == "agnostic" for s in b["steps"] if str(s.get("name", "")).startswith("xcheck")]
            n = mod.m3_plan_counts()["xcheck_steps"]
            return None if (tp and n == 14) else f"xcheck steps {n}, TP8 agnostic boot1 xcheck {len(tp)}"
        self.assert_killed(PLAN_SRC, "X4", '        if kind == "boot1":\n            st.append(xcheck_step(arm))',
                           '        if kind == "boot1" and cfg == "N1":\n            st.append(xcheck_step(arm))', chk)

    def test_mutant_S1_two_shapes_killed(self):
        chk = lambda mod: None if mod.m3_plan_counts()["cells"] == 560 else f"cells {mod.m3_plan_counts()['cells']}"
        self.assert_killed(PLAN_SRC, "S1", "    s = list(C.SHAPES)", '    s = ["A", "B"]', chk)

    def test_mutant_F2_agnostic_floor_missing_killed(self):
        chk = lambda mod: None if mod.m3_plan_counts()["floor_blocks"] == 20 else f"floor blocks {mod.m3_plan_counts()['floor_blocks']}"
        self.assert_killed(PLAN_SRC, "F2", '    if arm in C.FLOOR_ARMS and kind in ("boot1", "anchor"):',
                           '    if arm == "fused_sym" and kind in ("boot1", "anchor"):', chk)

    def test_budget_model_and_caps(self):
        h = {j: P.m3_budget(j)["job_h"] for j in C.JOBS}
        for j, (lo, hi) in {"J1": (5.8, 6.0), "J2": (13.1, 13.3), "J4": (13.0, 13.2), "J8": (12.3, 12.5)}.items():
            self.assertTrue(lo <= h[j] <= hi, (j, h[j]))
        self.assertEqual({j: C.hard_cap("m3", j) for j in C.JOBS}, {"J1": 14 * 3600, "J2": 16 * 3600, "J4": 16 * 3600, "J8": 16 * 3600})
        rows = {(r["scenario"], r["calib"]): r for r in P.m3_forecast_table("J2")}
        s3 = rows[("S3 extension: agnostic 1.48", "audited")]
        self.assertEqual(s3["configs_defined"], ["DP2"])                                   # DP completes before the cap, TP is cut
        self.assertLess(s3["end_h"], 16.0 + 1e-6)


# =========================================================================== floor (descriptive) / log knobs / KV / clock
def floor_doc(scale=1.0, complete=True):
    pts = {str(L_): {"floor_s": 0.05 * scale * (L_ / 1024) ** 0.9, "complete": True} for L_ in C.FLOOR_LENGTHS}
    return {"complete": complete, "points": pts}


class FloorAndKnobs(MutationCase):
    def test_floor_T_interpolation_and_no_extrapolation(self):
        f = floor_doc()
        self.assertAlmostEqual(L.floor_T(f, 256, 3), 3 * 0.05, 9)                           # L < 1k -> floor(1k)
        self.assertAlmostEqual(L.floor_T(f, 8192, 5), 5 * 0.05 * 8 ** 0.9, 9)
        mid = L.floor_T(f, 3000, 1)
        self.assertAlmostEqual(mid, 0.05 * (3000 / 1024) ** 0.9, 9)                          # log-log exact on a power law
        self.assertIsNone(L.floor_T(f, 20000, 3))
        self.assertIsNone(L.floor_T(floor_doc(complete=False), 2048, 3))

    def test_floor_job_effect(self):
        docs = {"J1": {"floor": floor_doc()}, "J2": {"floor": floor_doc(1.02)}, "J4": {"floor": floor_doc(1.04)}, "J8": {"floor": None}}
        fj = L.floor_job_effects(docs)
        self.assertEqual({J: v["status"] for J, v in fj.items()}, {"J2": "CLEAR", "J4": "FIRED", "J8": "UNTESTED"})

    def test_mutant_F1_j1_floor_scores_j_n_killed(self):
        docs = {"J1": {"floor": floor_doc()}, "J2": {"floor": floor_doc(1.5)}}
        chk = lambda mod: None if mod.job_floor(docs, "J2") is docs["J2"]["floor"] else "J2 scored with another Job's floor"
        self.assert_killed(LABEL_SRC, "F1", '    return (docs.get("J1" if _mut == "j1" else J) or {}).get("floor")',
                           '    return (docs.get("J1") or {}).get("floor")', chk)

    def test_log_knob_mismatch_blocks_every_trend(self):
        base = make_docs({"A": UP, "B": UP, "X": UP}, {"A": UP, "B": UP, "X": UP})
        same = {J: dict(d, log_knobs={"log_mirror": "1", "heartbeat_s": "0"}) for J, d in base.items()}
        d = L.aggregate_jobs(same)
        self.assertFalse(d["LOG_KNOB_MISMATCH"])
        self.assertTrue(d["trends"]["G_sym|DP|A"]["direction_citable"])
        diff = dict(same, J4=dict(same["J4"], log_knobs={"log_mirror": "0", "heartbeat_s": "0"}))
        d = L.aggregate_jobs(diff)
        self.assertTrue(d["LOG_KNOB_MISMATCH"])
        self.assertTrue(all(not t["direction_citable"] and "LOG_KNOB_MISMATCH" in t["disclosures"] for t in d["trends"].values()))
        missing = dict(same, J8=dict(same["J8"], log_knobs=None))
        self.assertTrue(L.aggregate_jobs(missing)["LOG_KNOB_MISMATCH"])

    def test_kv_pool_drift_and_dtype_unverified(self):
        ref = L.load_reference()
        self.assertEqual(len(ref["entries"]), 40)
        e = ref["entries"]["J2|TP2|agnostic"]
        self.assertEqual((e["max_total_num_tokens"], sorted(set(e["ssm_state_gb"]))), ([6389744], [6.39]))
        cm = {"max_total_num_tokens": [6389744]}
        mf = {"ssm_state_gb": [6.39, 6.39]}
        self.assertEqual(L.kv_dtype_flags("J2", "TP2", "agnostic", cm, mf, ref), ([], False))
        dr, _ = L.kv_dtype_flags("J2", "TP2", "agnostic", {"max_total_num_tokens": [6389744 * 1.002]}, mf, ref)
        self.assertTrue(dr and "max_total_num_tokens" in dr[0])                             # 0.2 % > 0.1 %
        dr, _ = L.kv_dtype_flags("J2", "TP2", "agnostic", cm, {"ssm_state_gb": [3.2, 3.2]}, ref)
        self.assertTrue(dr and "ssm_state_gb" in dr[0])                                     # bfloat16-sized state
        self.assertEqual(L.kv_dtype_flags("J2", "TP2", "agnostic", cm, {}, ref), ([], True))   # no Mamba line -> DTYPE_UNVERIFIED
        self.assertEqual(L.kv_dtype_flags("J2", "TP2", "agnostic", cm, mf, None)[1], True)

    def test_mutant_M13_floor_partial_set_killed(self):
        """R-H4 M13: fewer than FLOOR_REPEATS successes at a length -> floor_s None (never a median over a partial set)."""
        L0, rep = C.FLOOR_LENGTHS[0], C.FLOOR_REPEATS
        ok = [{"input_tokens": L0, "ttft_ms": 50.0, "error": ""} for _ in range(rep - 1)]
        one_err = ok + [{"input_tokens": L0, "ttft_ms": None, "error": "SSE_ERROR: x", "error_kind": "SSE_ERROR", "error_side": "server"}]
        full = ok + [{"input_tokens": L0, "ttft_ms": 50.0, "error": ""}]

        def chk(mod):
            if mod.floor_summary(full, [L0], rep)["points"][str(L0)]["floor_s"] != 0.05:
                return "control: 16 successes did not give the median"
            for name, reqs in (("15 ok + 1 server error", one_err), ("15 sent, all ok", ok)):
                pt = mod.floor_summary(reqs, [L0], rep)["points"][str(L0)]
                if pt["floor_s"] is not None or pt["complete"]:
                    return f"{name}: floor_s {pt['floor_s']} complete {pt['complete']}"
            return None
        self.assert_killed(BOOT_SRC, "M13",
                           '"floor_s": (statistics.median(float(r["ttft_ms"]) for r in ok) / 1000.0) if complete else None,',
                           '"floor_s": (statistics.median(float(r["ttft_ms"]) for r in ok) / 1000.0) if ok else None,', chk)

    def _dtype_root(self, td, with_ssm):
        """Two M3 rev12 boots (fused_sym, agnostic) of J2 DP2 whose cell_meta equals the M2g reference; mamba_facts has the ssm_state
        line only if `with_ssm`."""
        root = Path(td)
        ref = L.load_reference()
        for arm in ("fused_sym", "agnostic"):
            bid = f"m3_DP2_{arm}_boot1"
            (root / "specs").mkdir(parents=True, exist_ok=True)
            (root / "specs" / f"{bid}.json").write_text(json.dumps({"boot_id": bid, "config": "DP2", "arm": arm, "block": "main", "steps": []}))
            bd = root / "boots" / bid
            bd.mkdir(parents=True)
            e = ref["entries"][f"J2|DP2|{arm}"]
            (bd / "cell_meta.json").write_text(json.dumps({"max_mamba_cache_size": e["max_mamba_cache_size"],
                                                            "max_total_num_tokens": e["max_total_num_tokens"],
                                                            "max_running_requests_requested": 48}))
            (bd / "precheck.json").write_text("{}")
            (bd / "mamba_facts.json").write_text(json.dumps({"ssm_state_gb": e["ssm_state_gb"]} if with_ssm else {}))
        return root, ref

    def _chk_dtype(self, mod):
        cells = xw_cells("DP2", {"A": 1.0, "X": 1.0})
        for with_ssm, want in ((True, False), (False, True)):
            with tempfile.TemporaryDirectory() as td:
                root, ref = self._dtype_root(td, with_ssm)
                fl = mod.build_config_flags(root, cells, reference=ref, job="J2")
                doc = mod.aggregate(cells, config_flags=fl, job="J2", xcheck={"DP2": {"verdict": "PASS"}})
                for shape in ("A", "X"):
                    e = doc["estimands"]["G_sym"]["DP2"][shape]
                    if not e["defined"]:
                        return f"{shape} undefined: {e['undefined_reasons']}"
                    if ("DTYPE_UNVERIFIED" in e["disclosures"]) != want:
                        return f"mamba line {'present' if with_ssm else 'absent'}: {shape} disclosures {e['disclosures']}"
        return None

    def test_mutant_M12_dtype_unverified_reaches_the_estimand_killed(self):
        """R-H4 M12: boot without a Mamba ssm_state line -> build_config_flags dtype_unverified -> DTYPE_UNVERIFIED in the estimand
        disclosures (MG12-15); with the line (= the M2g reference) the disclosure is absent."""
        self.assert_killed(LABEL_SRC, "M12", '            d.append("DTYPE_UNVERIFIED")', "            pass", self._chk_dtype)

    def test_memory_check_x_bound(self):
        f = {a: dict(mamba=[48], max_total=[48 * 17000], eff_cap=48) for a in C.ARMS}
        self.assertFalse(L.memory_check(f, shapes=C.SHAPES)["memory_mismatch"])
        f = {a: dict(mamba=[48], max_total=[48 * 17000 - 1], eff_cap=48) for a in C.ARMS}
        self.assertTrue(L.memory_check(f, shapes=C.SHAPES)["memory_mismatch"])           # (ii) with X = 16384 + 616 (M3 labels)
        self.assertFalse(L.memory_check(f)["memory_mismatch"])                            # default = A/B: the sealed M2g G5(c) scorer is unchanged

    def test_clock_by_cell(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "clk.csv"
            rows = ["timestamp, clocks.current.sm [MHz], power.draw [W], temperature.gpu, clocks_event_reasons.active, index"]
            import datetime as dt
            t0 = dt.datetime(2026, 10, 20, 1, 0, 0, tzinfo=dt.timezone.utc)
            for k in range(20):
                ts = (t0 + dt.timedelta(seconds=0.1 * k)).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                for g in (0, 1):
                    rows.append(f"{ts}, {1410 if g == 0 else 1200} MHz, {300 + g} W, 40, {'0x0000000000000004' if (g == 1 and k % 2) else '0x0000000000000001'}, {g}")
            p.write_text("\n".join(rows) + "\n")
            e0 = t0.timestamp()
            cells = [{"cell_id": "a", "meta": {"t_start_epoch": e0, "t_end_epoch": e0 + 1.0, "gpu_ids": "0", "arm": "fused"}},
                     {"cell_id": "b", "meta": {"t_start_epoch": e0, "t_end_epoch": e0 + 1.95, "gpu_ids": "1", "arm": "agnostic"}}]
            out = L.clock_by_cell(p, cells)
            self.assertEqual(out["cells"]["a"]["sm_clock_mhz_median"], 1410)
            self.assertAlmostEqual(out["cells"]["b"]["throttle_frac"], 0.5, 6)
            self.assertEqual(L.clock_by_cell(Path(td) / "none.csv", cells)["status"], "CLK_ABSENT")


# =========================================================================== R-H3: X wiring aggregate() -> aggregate_jobs()
def xw_cell(cfg, arm, shape, rung, seed, achieved, block="main"):
    return {"cell_id": f"{block}|{cfg}|{arm}|{shape}|{rung}|{seed}", "status": "OK", "g": 0.4 if rung == "HI" else 0.01, "achieved": achieved,
            "truncated": False, "harness_limited": False, "frontend_saturated": False, "server_error_hi": False,
            "meta": dict(config=cfg, arm=arm, shape=shape, rung=rung, seed=seed, ext=0, round=1, block=block)}


def xw_cells(cfg, g_by_shape, block="main", lam_fs=1.0):
    """HI (4 seeds, g 0.4) + LO (2 seeds, g 0.01) cells of fused_sym (achieved lam_fs) and agnostic (achieved G * lam_fs) per shape."""
    out = []
    for shape, G in g_by_shape.items():
        for arm, ach in (("fused_sym", lam_fs), ("agnostic", G * lam_fs)):
            out += [xw_cell(cfg, arm, shape, "HI", sd, ach, block) for sd in C.HI_SEEDS]
            out += [xw_cell(cfg, arm, shape, "LO", sd, ach / 4.0, block) for sd in C.LO_SEEDS]
    return out


def jobs_from_cells(mod, g1, series, anchors):
    """The four Job documents built by mod.aggregate from synthetic CELLS (not injected ref/anchor dicts), joined by mod.aggregate_jobs.
    g1 {shape: G(1)} (J1 N1 boot1), series {shape: [G(2), G(4), G(8)]} (DP and TP alike), anchors {shape: G of the J_N N=1 anchor}."""
    pas = {"verdict": "PASS"}
    docs = {"J1": mod.aggregate(xw_cells("N1", g1), job="J1", xcheck={"N1": pas})}
    for i, n in enumerate((2, 4, 8)):
        cells = xw_cells("N1", anchors, block="anchor")
        for m in ("DP", "TP"):
            cells += xw_cells(f"{m}{n}", {sh: v[i] for sh, v in series.items()})
        docs[f"J{n}"] = mod.aggregate(cells, job=f"J{n}", xcheck={f"DP{n}": pas, f"TP{n}": pas})
    return mod.aggregate_jobs(docs)


class XWiringEndToEnd(MutationCase):
    """U14 / F-5 wiring (harness audit r1 R-H3): the X Job effect, X anchors and the X correction must be built from shape-X cells.
    The X anchor/reference values differ from the A ones, so reading A anywhere on the X path changes the result."""

    # S1: G(1, X) = 1.2 = every X anchor; A everywhere 1.0  ->  every Job effect (A and X) CLEAR
    S1 = dict(g1={"A": 1.0, "B": 1.0, "X": 1.2}, series={"A": [1.0] * 3, "B": [1.0] * 3, "X": [1.2] * 3}, anchors={"A": 1.0, "X": 1.2})
    # S2: G(1, X) = 1.2, X anchors 1.0 (FIRED, -16.7 %), X points 1.0  ->  raw X trend decreases, the corrected one (x 1.2 / 1.0) is flat
    S2 = dict(g1={"A": 1.0, "B": 1.0, "X": 1.2}, series={"A": [1.0] * 3, "B": [1.0] * 3, "X": [1.0] * 3}, anchors={"A": 1.02, "X": 1.0})

    def _chk_s1(self, mod):
        d = jobs_from_cells(mod, **self.S1)
        sx = {J: v["status"] for J, v in d["job_effect_x"].items()}
        sa = {J: v["status"] for J, v in d["job_effect"].items()}
        if set(sx.values()) != {"CLEAR"}:
            return f"X Job effect {sx} (X anchor 1.2 == X reference 1.2 must be CLEAR)"
        if set(sa.values()) != {"CLEAR"}:
            return f"A Job effect {sa}"
        return None

    def _chk_s2(self, mod):
        d = jobs_from_cells(mod, **self.S2)
        sx = {J: v["status"] for J, v in d["job_effect_x"].items()}
        if set(sx.values()) != {"FIRED"}:
            return f"X Job effect {sx} (X anchor 1.0 vs X reference 1.2 must FIRE)"
        if {J: v["status"] for J, v in d["job_effect"].items()} != {"J2": "CLEAR", "J4": "CLEAR", "J8": "CLEAR"}:
            return "A Job effect not CLEAR"
        tx, ta = d["trends"]["G_sym|DP|X"], d["trends"]["G_sym|DP|A"]
        if tx["label"] != L.TREND_JOB_SENSITIVE or tx["direction_citable"] or tx["raw_label"] != "G_DECREASES_WITH_N":
            return f"X trend {tx['label']} raw {tx.get('raw_label')} citable {tx['direction_citable']} (must be TREND_JOB_SENSITIVE)"
        if (tx.get("corrected") or {}).get("label") != "NO_TREND_DETECTED":
            return f"X corrected trend {(tx.get('corrected') or {}).get('label')} (the J1 X reference must correct it to flat)"
        if ta["label"] != "NO_TREND_DETECTED":
            return f"A trend {ta['label']}"
        return None

    def test_controls(self):
        self.assertIsNone(self._chk_s1(L))
        self.assertIsNone(self._chk_s2(L))

    def _chk_m8(self, mod):
        ref = L._syn_ref(g_ref=1.2, fused_sym=dict(x_correctness="X_CORRECTNESS_FAIL"), agnostic=dict(x_correctness="X_CORRECTNESS_FAIL"))
        st = mod.job_effect(ref, L._syn_anchor(1.2))
        return None if st["status"] == "UNTESTED" and any("J1_REFERENCE_X_CORRECTNESS_FAIL" in r for r in st["reasons"]) else f"{st}"

    def test_mutant_M8_reference_x_correctness_ignored_killed(self):
        """M8 is EQUIVALENT end to end (J1 xcheck not PASS -> G_sym(1, X) undefined -> P9-a UNTESTED first, auditor's reproduction);
        pinned here at the job_effect() contract so that the P9-a path is not its only guard."""
        self.assert_killed(LABEL_SRC, "M8", '        if e.get("x_correctness"):\n            why.append(f"{a}:J1_REFERENCE_',
                           '        if False:\n            why.append(f"{a}:J1_REFERENCE_', self._chk_m8)

    def test_mutant_M20_ref_x_from_A_killed(self):
        self.assert_killed(LABEL_SRC, "M20",
                           'doc["ref_x"] = {arm: dict(ref_entry(keys.get(("N1", arm, "X", 1))), x_correctness=xr)',
                           'doc["ref_x"] = {arm: dict(ref_entry(keys.get(("N1", arm, "A", 1))), x_correctness=xr)', self._chk_s1)

    def test_mutant_M21_anchor_x_from_A_killed(self):
        self.assert_killed(LABEL_SRC, "M21",
                           'doc["anchor_x"] = {"arms": {arm: anchor_entry(keys.get((C.ANCHOR_CONFIG, arm, "X", 1)))',
                           'doc["anchor_x"] = {"arms": {arm: anchor_entry(keys.get((C.ANCHOR_CONFIG, arm, "A", 1)))', self._chk_s1)

    def test_mutant_M22_x_trend_reads_A_job_effect_killed(self):
        self.assert_killed(LABEL_SRC, "M22",
                           '                st_of, anc_src = (status_x, anchors_x) if sx else (status, anchors)',
                           '                st_of, anc_src = (status, anchors)', self._chk_s2)

    def test_mutant_M23_x_correction_from_A_arrays_killed(self):
        self.assert_killed(LABEL_SRC, "M23",
                           't = trend(pts, anchors=anc_by_N, job_flags=flags_series, ref_g1=(ref_g1_x if sx else ref_g1), _mut=_mut)',
                           't = trend(pts, anchors=anc_by_N, job_flags=flags_series, ref_g1=ref_g1, _mut=_mut)', self._chk_s2)


# =========================================================================== U7 X correctness
def xrec(outs_c, outs_s=None, err=None, nonfinite=0, n_logprobs=None):
    """n_logprobs: None = one logprob per output token (what mg_xcheck._one records); an int = that count on every request."""
    docs = sorted(outs_c)
    mk = lambda o: {str(d): {"output_ids": list(o[d]), "error": (err if d == docs[0] and err else ""), "nonfinite": nonfinite,
                             "n_logprobs": len(o[d]) if n_logprobs is None else n_logprobs}
                    for d in docs}
    return {"docs": docs, "passes": {"c": mk(outs_c), "s": mk(outs_s or outs_c)}}


def outs(seed, n=4, diverge_at=None, docs=(11, 22, 33, 44)):
    o = {}
    for d in docs[:n]:
        ids = [(d * 31 + k * seed) % 997 for k in range(C.XC_MAX_NEW)]
        if diverge_at is not None and d == docs[0]:
            ids = ids[:diverge_at] + [(x + 1) % 997 for x in ids[diverge_at:]]
        o[d] = ids
    return o


class XCorrectness(MutationCase):
    def test_score_verdicts(self):
        base = xrec(outs(3))
        self.assertEqual(XC.score(base, xrec(outs(3)), True, True)["verdict"], "PASS")
        self.assertEqual(XC.score(base, xrec(outs(3, diverge_at=20)), True, True)["verdict"], "PASS")    # D1 = 3 >= D0 - 1, first 16 agree
        self.assertEqual(XC.score(base, xrec(outs(3, diverge_at=5)), True, True)["verdict"], "FAIL")     # first-16 mismatch
        two = outs(3)
        two[22] = [(x + 1) % 997 for x in two[22]]
        two[11] = two[11][:20] + [(x + 1) % 997 for x in two[11][20:]]
        self.assertEqual(XC.score(base, xrec(two), True, True)["verdict"], "FAIL")                       # D1 2 < D0 4 - 1
        self.assertEqual(XC.score(base, xrec(outs(3), err="HTTPError 400"), True, True)["verdict"], "FAIL")
        self.assertEqual(XC.score(base, xrec(outs(3), nonfinite=1), True, True)["verdict"], "FAIL")
        self.assertEqual(XC.score(base, xrec(outs(3)), True, False)["verdict"], "FAIL")                  # server died
        self.assertEqual(XC.score(base, xrec(outs(3)), True, None)["verdict"], "NOT_TESTED")
        self.assertEqual(XC.score(None, xrec(outs(3)), True, True)["verdict"], "NOT_TESTED")
        noisy = xrec(outs(3), outs_s={**outs(3), 11: [(x + 1) % 997 for x in outs(3)[11]]})                # D0 = 3
        two_b = outs(3)
        two_b[33] = two_b[33][:20] + [(x + 1) % 997 for x in two_b[33][20:]]
        two_b[11] = two_b[11][:20] + [(x + 1) % 997 for x in two_b[11][20:]]
        self.assertEqual(XC.score(noisy, xrec(two_b), True, True)["verdict"], "PASS")                    # D1 2 >= D0 3 - 1

    def test_mutants_X2_X3_rules_removed_killed(self):
        base = xrec(outs(3))
        two = outs(3)
        two[22] = two[22][:20] + [(x + 1) % 997 for x in two[22][20:]]
        two[11] = two[11][:20] + [(x + 1) % 997 for x in two[11][20:]]
        drop = lambda mod: None if mod.score(base, xrec(two), True, True)["verdict"] == "FAIL" else "exact-match drop 2 passed"
        self.assert_killed(XC_SRC, "X2", '    if _mut != "drop_rule" and d1 < d0 - C.XC_DROP_TOL:', "    if False:", drop)
        first = lambda mod: None if mod.score(base, xrec(outs(3, diverge_at=5)), True, True)["verdict"] == "FAIL" else "first-N mismatch passed"
        self.assert_killed(XC_SRC, "X3", '    if _mut != "first_n_rule" and first_bad:', "    if False:", first)

    def _cells_x(self, cfgs=("DP2",)):
        cells = []
        for cfg, arm in [(c, a) for c in cfgs for a in ("fused_sym", "agnostic")]:
            for sh in ("A", "X"):
                for sd in C.HI_SEEDS:
                    cells.append({"cell_id": f"{cfg}{arm}{sh}{sd}", "status": "OK", "g": 0.4, "achieved": 2.0, "truncated": False,
                                  "harness_limited": False, "frontend_saturated": False, "server_error_hi": False,
                                  "meta": dict(config=cfg, arm=arm, shape=sh, rung="HI", seed=sd, ext=0, round=1, block="main")})
                for sd in C.LO_SEEDS:
                    cells.append({"cell_id": f"lo{cfg}{arm}{sh}{sd}", "status": "OK", "g": 0.01, "achieved": 1.0, "truncated": False,
                                  "harness_limited": False, "frontend_saturated": False,
                                  "meta": dict(config=cfg, arm=arm, shape=sh, rung="LO", seed=sd, ext=0, round=1, block="main")})
        return cells

    def _chk_x1(self, mod):
        doc = mod.aggregate(self._cells_x(), job="J2", xcheck={"verdict": "FAIL"})
        e, a = doc["estimands"]["G_sym"]["DP2"]["X"], doc["estimands"]["G_sym"]["DP2"]["A"]
        if e["defined"]:
            return "X estimand defined although the Job's X correctness step FAILED"
        if "X_CORRECTNESS_FAIL" not in e["undefined_reasons"]:
            return f"reasons {e['undefined_reasons']}"
        if not a["defined"]:
            return "shape A voided by the X correctness step"
        return None

    def test_fail_and_not_run_void_x_only(self):
        self.assertIsNone(self._chk_x1(L))
        self.assertTrue(L.aggregate(self._cells_x(), job="J2", xcheck={"verdict": "PASS"})["estimands"]["G_sym"]["DP2"]["X"]["defined"])
        e = L.aggregate(self._cells_x(), job="J2", xcheck=None)["estimands"]["G_sym"]["DP2"]["X"]
        self.assertIn("X_CORRECTNESS_NOT_RUN", e["undefined_reasons"])                    # fail-closed when the step never ran

    def _chk_scope(self, mod):
        doc = mod.aggregate(self._cells_x(("DP2", "TP2")), job="J2", xcheck={"DP2": {"verdict": "PASS"}, "TP2": {"verdict": "FAIL"}})
        e = doc["estimands"]["G_sym"]
        got = (e["TP2"]["X"]["defined"], e["DP2"]["X"]["defined"], e["TP2"]["A"]["defined"], e["DP2"]["A"]["defined"])
        return None if got == (False, True, True, True) else f"(TP2 X, DP2 X, TP2 A, DP2 A) defined = {got}"

    def test_scope_is_the_failing_mode_only(self):
        self.assertIsNone(self._chk_scope(L))
        e = L.aggregate(self._cells_x(("DP2", "TP2")), job="J2", xcheck={"DP2": {"verdict": "PASS"}})["estimands"]["G_sym"]
        self.assertIn("X_CORRECTNESS_NOT_RUN", e["TP2"]["X"]["undefined_reasons"])           # a config without its record: fail-closed
        self.assertTrue(e["DP2"]["X"]["defined"])

    def test_mutant_X5_failure_voids_every_mode_killed(self):
        self.assert_killed(LABEL_SRC, "X5",
                           '        if shape == "X" and x_correctness_reason(xcheck_of(xcheck, cfg)):\n'
                           '            reasons.append(x_correctness_reason(xcheck_of(xcheck, cfg)))',
                           '        if shape == "X" and any(x_correctness_reason(v) for v in (xcheck or {}).values()):\n'
                           '            reasons.append("X_CORRECTNESS_FAIL")', self._chk_scope)

    def test_mutant_X1_label_unchanged_killed(self):
        self.assert_killed(LABEL_SRC, "X1", '        if shape == "X" and x_correctness_reason(xcheck_of(xcheck, cfg)):', "        if False:",
                           self._chk_x1)

    def test_mutant_M1_short_output_killed(self):
        """R-H4 M1: every pass returns the SAME 31 tokens (D0 = D1 = 4, first 16 agree); the length rule alone must FAIL it.  The
        logprob count is held at XC_MAX_NEW so that the R-H5 rule cannot mask this mutant."""
        short = {d: v[:C.XC_MAX_NEW - 1] for d, v in outs(3).items()}
        mk = lambda: xrec(short, n_logprobs=C.XC_MAX_NEW)
        chk = lambda mod: None if mod.score(mk(), mk(), True, True)["verdict"] == "FAIL" else "31-token outputs passed"
        self.assert_killed(XC_SRC, "M1", '                elif len(v.get("output_ids") or []) != C.XC_MAX_NEW:', "                elif False:", chk)

    def _chk_logprobs(self, mod):
        full = xrec(outs(3))
        if mod.score(full, xrec(outs(3)), True, True)["verdict"] != "PASS":
            return "control record with 32 logprobs per request did not PASS"
        for n_lp in (0, C.XC_MAX_NEW - 1):
            r = mod.score(full, xrec(outs(3), n_logprobs=n_lp), True, True)
            if r["verdict"] != "FAIL" or not any(f"LOGPROBS_{n_lp}" in w for w in r["reasons"]):
                return f"{n_lp} logprobs per request: {r['verdict']} {r.get('reasons')}"
        legacy = xrec(outs(3))
        for v in legacy["passes"]["c"].values():
            v.pop("n_logprobs")
        if mod.score(legacy, xrec(outs(3)), True, True)["verdict"] != "FAIL":
            return "a record without n_logprobs passed"
        return None

    def test_mutant_M24_logprobs_absent_pass_killed(self):
        """R-H5: every request must carry exactly XC_MAX_NEW output-token logprobs, else FAIL (the NaN check is vacuous otherwise)."""
        self.assert_killed(XC_SRC, "M24", '                if not v.get("error") and int(v.get("n_logprobs") or 0) != C.XC_MAX_NEW:',
                           "                if False:", self._chk_logprobs)

    def test_serve_records_one_logprob_per_token(self):
        def post(_ep, body):
            out = list(range(body["sampling_params"]["max_new_tokens"]))
            return {"output_ids": out, "meta_info": {"output_token_logprobs": [[-0.2, t, None] for t in out]}}
        rid, v = XC._one("http://x", "1", [1, 2, 3], post)
        self.assertEqual((v["n_logprobs"], v["nonfinite"]), (C.XC_MAX_NEW, 0))
        rid, v = XC._one("http://x", "1", [1, 2, 3], lambda _e, b: {"output_ids": [1] * C.XC_MAX_NEW, "meta_info": {}})
        self.assertEqual(v["n_logprobs"], 0)

    def test_select_docs_longest_four(self):
        h = {"docs": [[1, 0, 100, 5], [2, 0, 900, 5], [3, 0, 900, 5], [4, 0, 50, 5], [5, 0, 800, 5], [6, 0, 700, 5]]}
        self.assertEqual(XC.select_docs(h), [2, 3, 5, 6])


# =========================================================================== fake SGLang servers
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeGen:
    """/generate: stream (SSE) or not; logs every raw request body; rid prefix 'err' -> SSE error chunk."""

    def __init__(self, diverge=False, error=None):
        self.port, self.bodies, self.diverge, self.error = free_port(), [], diverge, error
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    async def generate(self, req):
        raw = await req.read()
        self.bodies.append(raw)
        body = json.loads(raw)
        n = int(body["sampling_params"]["max_new_tokens"])
        ids = body.get("input_ids") or []
        if not body.get("stream") and self.error == "json":
            return web.json_response({"error": {"message": "context too long"}})
        if not body.get("stream") and self.error == "http":
            return web.json_response({"object": "error", "message": "bad"}, status=400)
        if not body.get("stream"):
            out = [(sum(ids) + 7 * k + (1 if (self.diverge and k > 2) else 0)) % 1000 for k in range(n)]
            return web.json_response({"output_ids": out, "meta_info": {"output_token_logprobs": [[-0.5, t, None] for t in out]}})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        if str(body.get("rid", "")).startswith("err"):
            await resp.write(b'data: {"error": {"message": "too long"}}\n\n')
        else:
            for k in range(1, n + 1):
                await resp.write(b"data: " + json.dumps({"text": "t", "meta_info": {"completion_tokens": k}}).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    def _run(self):
        asyncio.set_event_loop(self.loop)
        app = web.Application(client_max_size=1 << 26)
        app.router.add_post("/generate", self.generate)
        self.runner = web.AppRunner(app, access_log=None)
        self.loop.run_until_complete(self.runner.setup())
        self.loop.run_until_complete(web.TCPSite(self.runner, "127.0.0.1", self.port).start())
        self.ready.set()
        self.loop.run_forever()

    def __enter__(self):
        self.thread.start()
        self.ready.wait(10)
        return self

    def __exit__(self, *a):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


def x_item(rid, ids, out=3):
    return {"request_id": rid, "phase": "mg", "arrival_s": 0.0, "input_tokens": len(ids), "output_tokens": out, "input_ids": list(ids)}


@unittest.skipUnless(HAVE_AIOHTTP, "aiohttp not installed")
class ReplayPayload(MutationCase):
    """F-3: X real tokens are sent INSIDE replay (StreamState); A/B payload bytes are rev11's."""

    @staticmethod
    def _run(mod, items):
        with tempfile.TemporaryDirectory() as td, FakeGen() as srv:
            asyncio.run(mod.replay(f"http://127.0.0.1:{srv.port}/generate", items, Path(td) / "r.jsonl", 3000.0, 50.0, 0))
            recs = [json.loads(x) for x in (Path(td) / "r.jsonl").read_text().splitlines()]
            return srv.bodies, recs

    def _chk_x(self, mod):
        bodies, _ = self._run(mod, [x_item("x0", [5, 6, 7, 8])])
        got = json.loads(bodies[0])["input_ids"]
        return None if got == [5, 6, 7, 8] else f"X payload input_ids {got}"

    def _chk_ab(self, mod):
        item = {"request_id": "a0", "phase": "mg", "arrival_s": 0.0, "input_tokens": 4, "output_tokens": 2}
        bodies, _ = self._run(mod, [item])
        rev11 = json.dumps({"input_ids": [1] * 4, "sampling_params": {"max_new_tokens": 2, "temperature": 0.0, "ignore_eos": True},
                            "stream": True, "rid": "a0"}).encode()
        return None if bodies[0] == rev11 else f"A/B payload bytes changed: {bodies[0][:120]!r}"

    def _chk_err(self, mod):
        _, recs = self._run(mod, [x_item("err0", [9] * 10, 4), x_item("x1", [3] * 10, 4)])
        r = {x["request_id"]: x for x in recs}
        e = r["err0"]
        if (e["error_kind"], e["error_side"], e["observed_output_tokens"]) != ("SSE_ERROR", "server", 0):
            return f"X error chunk: {e['error_kind']}/{e['error_side']} observed {e['observed_output_tokens']}"
        return None if r["x1"]["error"] == "" else "X success misclassified"

    def test_controls(self):
        self.assertIsNone(self._chk_x(TL))
        self.assertIsNone(self._chk_ab(TL))
        self.assertIsNone(self._chk_err(TL))

    def test_token_count_mismatch_refused_before_send(self):
        bad = dict(x_item("x0", [1, 2, 3]), input_tokens=4)
        with self.assertRaises(ValueError):
            self._run(TL, [bad])

    SITE = '''                "input_ids": (list(item["input_ids"]) if "input_ids" in item
                              else [1] * int(item["input_tokens"])),'''

    def test_mutant_L1_x_sends_ones_killed(self):
        self.assert_killed(LOADGEN_SRC, "L1", self.SITE, '''                "input_ids": [1] * int(item["input_tokens"]),''', self._chk_x,
                           package="pdmux_eval")

    def test_mutant_L2_ab_payload_changed_killed(self):
        self.assert_killed(LOADGEN_SRC, "L2", self.SITE, '''                "input_ids": (list(item["input_ids"]) if "input_ids" in item
                              else [0] * int(item["input_tokens"])),''', self._chk_ab, package="pdmux_eval")

    def test_mutant_L3_x_error_chunk_counted_as_token_killed(self):
        self.assert_killed(LOADGEN_SRC, "L3", '        if isinstance(chunk, dict) and "error" in chunk:\n',
                           '        if False and isinstance(chunk, dict) and "error" in chunk:\n', self._chk_err, package="pdmux_eval")

    def test_xcheck_serve_and_score_against_fake_servers(self):
        with tempfile.TemporaryDirectory() as td:
            tok = make_token_file(Path(td) / "t.bin")
            with FakeGen() as a, FakeGen() as b, FakeGen(diverge=True) as c:
                for srv, name in ((a, "fs"), (b, "ag"), (c, "bad")):
                    self.assertEqual(XC.serve(f"http://127.0.0.1:{srv.port}", tok, Path(td) / f"{name}.json"), 0)
            rd = lambda n: json.loads((Path(td) / f"{n}.json").read_text())
            self.assertEqual(rd("fs")["docs"], XC.select_docs(XT.read_header(tok)))
            self.assertEqual(XC.score(rd("fs"), rd("ag"), True, True)["verdict"], "PASS")
            self.assertEqual(XC.score(rd("fs"), rd("bad"), True, True)["verdict"], "FAIL")
            body = json.loads(a.bodies[0])
            self.assertEqual(body["sampling_params"], {"temperature": 0.0, "max_new_tokens": 32, "ignore_eos": True})

    def test_xcheck_error_responses_fail(self):
        """An error answer (JSON error body or HTTP 400) is recorded and the verdict is FAIL (never PASS, never silently NOT_TESTED)."""
        with tempfile.TemporaryDirectory() as td:
            tok = make_token_file(Path(td) / "t.bin")
            with FakeGen() as a, FakeGen(error="json") as e1, FakeGen(error="http") as e2:
                self.assertEqual(XC.serve(f"http://127.0.0.1:{a.port}", tok, Path(td) / "fs.json"), 0)
                for srv, name in ((e1, "e1"), (e2, "e2")):
                    self.assertEqual(XC.serve(f"http://127.0.0.1:{srv.port}", tok, Path(td) / f"{name}.json"), 1)
            rd = lambda n: json.loads((Path(td) / f"{n}.json").read_text())
            for name in ("e1", "e2"):
                rec = rd(name)
                self.assertTrue(all(v["error"] for v in rec["passes"]["c"].values()), name)
                r = XC.score(rd("fs"), rec, True, True)
                self.assertEqual(r["verdict"], "FAIL", name)
                self.assertTrue(any(":ERROR" in w for w in r["reasons"]), r)
                self.assertEqual(XC.score(rec, rd("fs"), True, True)["verdict"], "FAIL", name)


# =========================================================================== token file + mg_boot pre-checks
def make_token_file(path, n_docs=12, long_len=None, floor_n=16384):
    docs = {}
    for d in range(n_docs):
        L_ = (long_len if (long_len and d == 0) else 40 + 13 * d)
        docs[100 + d] = ([(d * 17 + k) % 50000 for k in range(L_)], 5 + d)
    XT.write_file(path, docs, {"doc_id": 999}, [(k * 3) % 50000 for k in range(floor_n)], {"tokenizers_version": "test"})
    return str(path)


class TokenFileAndTraces(unittest.TestCase):
    def test_roundtrip_verify_and_prefix_property(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_token_file(Path(td) / "t.bin")
            v = XT.verify(p)
            self.assertEqual(v["problems"], [])
            h = XT.read_header(p)
            self.assertEqual(XT.doc_input_ids(p, h, [103])[103], [(3 * 17 + k) % 50000 for k in range(40 + 39)])
            self.assertEqual(XT.floor_input_ids(p, h, 1024), [(k * 3) % 50000 for k in range(1024)])
            B.gen_trace_x(Path(td) / "a", 2.0, 5, 4386, h)
            B.gen_trace_x(Path(td) / "b", 2.0, 9, 4386, h)
            da = [json.loads(x)["doc_id"] for x in (Path(td) / "a").read_text().splitlines()[1:]]
            db = [json.loads(x)["doc_id"] for x in (Path(td) / "b").read_text().splitlines()[1:]]
            self.assertEqual(db[:5], da)                                                    # R-8: pi_s prefix -> nested samples across N
            self.assertEqual(B.gen_trace_x(Path(td) / "c", 2.0, 5, 4386, h), B.gen_trace_x(Path(td) / "d", 2.0, 5, 4386, h))
            with self.assertRaises(ValueError):
                B.gen_trace_x(Path(td) / "e", 2.0, 13, 4386, h)                           # no sampling with replacement

    @unittest.skipUnless(os.environ.get("MG_TEST_X_TOKENS"), "set MG_TEST_X_TOKENS=<arxsum_x_tokens_v1.bin> to check the registered file")
    def test_registered_token_file(self):
        p = os.environ["MG_TEST_X_TOKENS"]
        v = XT.verify(p, C.X_TOKENS_SHA256)
        self.assertEqual(v["problems"], [])
        self.assertEqual((v["n_retained"], v["lengths_sha256"], v["floor"]["doc_id"], v["tokenizers_version"]),
                         (C.X_RETAINED_N, C.X_LENGTHS_SHA256, C.FLOOR_DOC_ID, C.X_TOKENIZERS_VERSION))
        st = v["stats"]
        self.assertEqual((st["input_mean"], st["input_p50"], st["input_p99"], st["output_mean"], st["output_p50"], st["output_p99"],
                          st["output_max"], st["max_input_plus_output"]), (7116, 6484, 15722, 203, 195, 399, 616, 16744))
        h = XT.read_header(p)
        for s in C.HI_SEEDS:
            B.x_perm(h, s)                                                                  # registered permutation sha256s


def boot_spec(td, steps, arm="fused_sym", kind="anchor", tok=None, sha=None):
    s = dict(P.config_spec("N1"), arm=arm, kind=kind, block="anchor" if kind == "anchor" else "main",
             boot_id=f"m3_N1_{arm}_{kind}", steps=steps, out_root=str(Path(td) / "out"), mamba_ssm_dtype="float32", health_wait_s=60)
    if tok:
        s.update(x_tokens_path=tok, x_tokens_sha256=sha or XT.sha256_file(tok))
    return s


def floor_step():
    return P.floor_step()


def a_cell():
    return dict(type="cell", shape="A", rung="HI", seed=4386, offered=20.0, n_req=40, ext=0, round=1, block="anchor")


def x_cell(n=8):
    return dict(type="cell", shape="X", rung="HI", seed=4386, offered=20.0, n_req=n, ext=0, round=1, block="anchor")


class BootPrechecks(MutationCase):
    def _rc(self, mod, steps, ctx, tok_long=None):
        with tempfile.TemporaryDirectory() as td:
            tok = make_token_file(Path(td) / "t.bin", long_len=tok_long)
            with mock.patch.object(C, "CONTEXT_LENGTH", ctx):
                return mod.run_boot(boot_spec(td, steps, tok=tok), dry=True)

    def test_context_fit_at_20480_and_refusal_at_16384(self):
        steps = [floor_step(), a_cell(), x_cell()]
        self.assertEqual(self._rc(B, steps, 20480, tok_long=16379), 0)
        # §19-2: a token file whose max input+output exceeds ctx-2 is ABORT_DATASET before the per-request fit check
        self.assertEqual(self._rc(B, steps, 16384, tok_long=16379), B.RC_DATASET)
        self.assertEqual(self._rc(B, [a_cell(), x_cell()], 16384, tok_long=16379), B.RC_DATASET)
        self.assertEqual(self._rc(B, [a_cell(), x_cell()], 16384), 0)                      # short X docs fit 16384
        self.assertEqual(self._rc(B, [floor_step(), a_cell()], 16384), B.RC_CONTEXT)       # floor 16k alone refused (16,392 > 16,382)

    def test_dataset_sha_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            tok = make_token_file(Path(td) / "t.bin")
            rc = B.run_boot(boot_spec(td, [x_cell()], tok=tok, sha="0" * 64), dry=True)
            self.assertEqual(rc, B.RC_DATASET)
            self.assertEqual(json.loads((Path(td) / "out/boots/m3_N1_fused_sym_anchor/boot.json").read_text())["status"], "ABORT_DATASET")

    def test_span_budget_drift_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "c.yml"
            cfg.write_text("split_forward_token_budget: 32768\n")
            spec = boot_spec(td, [a_cell()], arm="agnostic")
            spec["config_rel"] = str(cfg)
            with mock.patch.object(B, "ENGINE", Path("/")):
                self.assertEqual(B.run_boot(spec, dry=True), B.RC_CONFIG_DRIFT)
            self.assertEqual(B.config_span_budget(ENGINE / C.AGNOSTIC_CONFIG_REL), 65536)
            self.assertEqual(B.config_span_budget(ENGINE / C.D44_CONFIG_REL), 65536)

    def test_xcheck_requests_are_fit_checked(self):
        """U7 requests (the 4 longest documents + 32 tokens) are in mg_boot's pre-check: with a 20,460-token document and ctx 20480 the
        xcheck request (20,460 + 32 > 20,478) is refused (ABORT_CONTEXT_TOO_SHORT) even when the file-level bound is not the trigger."""
        with tempfile.TemporaryDirectory() as td:
            tok = make_token_file(Path(td) / "t.bin", long_len=20460)
            spec = boot_spec(td, [P.xcheck_step("fused_sym")], kind="boot1", tok=tok)
            with mock.patch.object(C, "CONTEXT_LENGTH", 20480 + 40):
                self.assertEqual(B.run_boot(dict(spec, boot_id="a"), dry=True), 0)          # control: 20,492 <= 20,518
            with mock.patch.object(B, "load_x_header", lambda sp: (XT.read_header(tok), None)):
                self.assertEqual(B.run_boot(dict(spec, boot_id="b"), dry=True), B.RC_CONTEXT)
                pre = json.loads((Path(td) / "out/boots/b/precheck.json").read_text())
                self.assertTrue(any("input+output" in p_ for p_ in pre["fit_problems"]), pre)

    def test_mutant_B1_check_removed_killed(self):
        chk = lambda mod: None if self._rc(mod, [floor_step(), a_cell(), x_cell()], 16384) == mod.RC_CONTEXT else "ctx 16384 not refused"
        self.assert_killed(BOOT_SRC, "B1", "    if spec.get(\"kind\") in M3_KINDS:\n        prc, pre = rev12_prechecks(",
                           "    if False:\n        prc, pre = rev12_prechecks(", chk)

    def test_mutant_B2_floor_not_fit_checked_killed(self):
        chk = lambda mod: None if self._rc(mod, [floor_step(), a_cell()], 16384) == mod.RC_CONTEXT else "floor 16k not refused at ctx 16384"
        self.assert_killed(BOOT_SRC, "B2", '''        elif st["type"] == "floor":
            gen_floor_trace(p, st["lengths"], st["repeats"], st["max_new"])''', '''        elif st["type"] == "floor":
            continue''', chk)

    def test_server_args_knobs_and_dtype_rule(self):
        a = B.build_server_args(boot_spec("/tmp", [], kind="boot1"), 1)
        for k, v in (("--context-length", "20480"), ("--max-prefill-tokens", "16384"), ("--max-mamba-cache-size", "48"),
                     ("--mamba-ssm-dtype", "float32")):
            self.assertEqual(a[a.index(k) + 1], v)
        tp = dict(P.config_spec("TP4"), arm="agnostic", kind="boot1", mamba_ssm_dtype="float32")
        a = B.build_server_args(tp, 1, "/c.yml")
        self.assertEqual(a[a.index("--max-mamba-cache-size") + 1], "192")
        with self.assertRaises(ValueError):
            B.build_server_args(dict(tp, mamba_ssm_dtype=""), 1, "/c.yml")
        m2g = dict(P.config_spec("TP2"), arm="fused", kind="g0")                          # sealed M2g boots: no rev12 knobs
        self.assertNotIn("--mamba-ssm-dtype", B.build_server_args(m2g, 1))

    def _dtype(self, mod, cfg):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.json"
            p.write_text(json.dumps(cfg))
            return mod.resolve_mamba_ssm_dtype(p)[0]

    def test_resolve_dtype(self):
        self.assertEqual(self._dtype(B, {"torch_dtype": "bfloat16"}), "float32")
        self.assertEqual(self._dtype(B, {"torch_dtype": "bfloat16", "mamba_ssm_dtype": "bfloat16"}), "bfloat16")
        self.assertEqual(self._dtype(B, {"text_config": {"mamba_ssm_dtype": "float16"}}), "float16")

    def test_mutant_B3_torch_dtype_fallback_killed(self):
        chk = lambda mod: None if self._dtype(mod, {"torch_dtype": "bfloat16"}) == "float32" else "torch_dtype fallback"
        self.assert_killed(BOOT_SRC, "B3", '    return C.MAMBA_SSM_DTYPE_DEFAULT, "default (key absent; torch_dtype not consulted)"',
                           '    return str(cfg.get("torch_dtype") or C.MAMBA_SSM_DTYPE_DEFAULT), "x"', chk)


FAKE_LAUNCH_SERVER_X = r"""
import asyncio, json, os, sys
from aiohttp import web
args = sys.argv[1:]
port = int(args[args.index("--port") + 1]); cap = int(args[args.index("--max-running-requests") + 1])
arg = lambda k: args[args.index(k) + 1] if k in args else None
log = os.environ.get("FAKE_PAYLOAD_LOG")
print(f"max_total_num_tokens=900000, max_running_requests={cap}", flush=True)
print(f"Mamba Cache is allocated. max_mamba_cache_size: {cap}, conv_state size: 0.09GB, ssm_state size: 6.46GB ", flush=True)
info_d = {"max_total_num_tokens": 900000, "max_running_requests": cap, "context_length": int(arg("--context-length")),
          "max_prefill_tokens": int(arg("--max-prefill-tokens")), "max_mamba_cache_size": int(arg("--max-mamba-cache-size")),
          "mamba_ssm_dtype": "bfloat16" if os.environ.get("FAKE_MODE") == "drift" else arg("--mamba-ssm-dtype")}
async def health(_r): return web.Response(text="ok")
async def info(_r): return web.json_response(info_d)
async def generate(req):
    body = await req.json(); n = int(body["sampling_params"]["max_new_tokens"]); ids = body.get("input_ids") or []
    if log:
        with open(log, "a") as f: f.write(json.dumps({"rid": body.get("rid"), "n": len(ids), "head": ids[:4], "sum": sum(ids)}) + "\n")
    if not body.get("stream"):
        out = [(sum(ids) + k) % 1000 for k in range(n)]
        return web.json_response({"output_ids": out, "meta_info": {"output_token_logprobs": [[-0.1, t, None] for t in out]}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await resp.prepare(req)
    for k in range(1, n + 1):
        await resp.write(b"data: " + json.dumps({"text": "t", "meta_info": {"completion_tokens": k}}).encode() + b"\n\n")
        await asyncio.sleep(0.001)
    await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
app = web.Application(client_max_size=1 << 26); app.router.add_get("/health", health); app.router.add_get("/model_info", info)
app.router.add_get("/server_info", info); app.router.add_post("/generate", generate)
web.run_app(app, host="127.0.0.1", port=port, print=None, access_log=None)
"""


@unittest.skipUnless(HAVE_AIOHTTP, "aiohttp not installed")
class BootWithFakeServer(unittest.TestCase):
    """The real mg_boot.run_boot: floor block, U7 xcheck cmd step, X warm-up + X cell with real tokens, through a fake server."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        root = Path(self.td.name)
        (root / "fake" / "sglang").mkdir(parents=True)
        (root / "fake" / "sglang" / "__init__.py").write_text("")
        (root / "fake" / "sglang" / "launch_server.py").write_text(FAKE_LAUNCH_SERVER_X)
        self.root = root
        clean = {"available": True, "gpus": {"0": {"used_mib": 10.0, "n_compute_procs": 0}}}
        p = mock.patch.object(B, "nvml_snapshot", return_value=clean)
        p.start()
        self.addCleanup(p.stop)
        e = mock.patch.dict(os.environ, {"PYTHONPATH": str(root / "fake") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                                         "FAKE_PAYLOAD_LOG": str(root / "payloads.jsonl")})
        e.start()
        self.addCleanup(e.stop)
        for k in [k for k in os.environ if k.startswith(("PDMUX_", "SGLANG_USE", "NCCL_"))]:
            os.environ.pop(k)
        cwd = os.getcwd()
        os.chdir(REPO)                                                              # cmd argv are repo-relative (the Job cd's to ROOT)
        self.addCleanup(os.chdir, cwd)

    def test_floor_xcheck_and_x_cell_end_to_end(self):
        tok = make_token_file(self.root / "t.bin")
        fl = dict(P.floor_step(), lengths=[1024, 2048], repeats=3)
        spec = boot_spec(self.root, [fl, P.xcheck_step("fused_sym"), {"type": "warmup", "shape": "X"}, x_cell(6)], tok=tok)
        rc = B.run_boot(spec)
        bd = self.root / "out/boots/m3_N1_fused_sym_anchor"
        self.assertEqual(rc, 0, (bd / "boot.json").read_text())
        boot = json.loads((bd / "boot.json").read_text())
        self.assertEqual(boot["status"], "OK", boot)
        self.assertEqual(boot["knobs_server_info"]["drift"], [])
        self.assertEqual(json.loads((bd / "precheck.json").read_text())["fit_problems"], [])
        self.assertEqual(json.loads((bd / "mamba_facts.json").read_text())["ssm_state_gb"], [6.46])
        floor = json.loads((bd / "FLOOR.json").read_text())
        self.assertTrue(floor["complete"], floor)
        self.assertEqual(hashlib.sha256((bd / "FLOOR.json").read_bytes()).hexdigest(), (bd / "FLOOR.json.sha256").read_text().strip())
        xc = json.loads((bd / "xcheck.json").read_text())
        self.assertEqual(len(xc["docs"]), 4)
        self.assertTrue(all(len(v["output_ids"]) == 32 for v in xc["passes"]["c"].values()))
        h = XT.read_header(tok)
        idx = XT.doc_index(h)
        pay = [json.loads(x) for x in (self.root / "payloads.jsonl").read_text().splitlines()]
        floor_pay = [p for p in pay if str(p["rid"]).startswith("floor-")]
        self.assertEqual(len(floor_pay), 6)
        self.assertEqual(floor_pay[0]["head"], [0, 3, 6, 9])                              # the floor document's prefix, not [1]*L
        cells = sorted((self.root / "out/cells").glob("*__X_HI_s4386"))
        self.assertEqual(len(cells), 1)
        reqs = [json.loads(x) for x in (cells[0] / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(reqs), 6)
        self.assertTrue(all(r["error"] == "" for r in reqs), reqs[:2])
        trace_docs = {}
        for p in pay:
            if str(p["rid"]).startswith("mg-X-4386-"):
                trace_docs[p["rid"]] = p
        perm = B.x_perm(h, 4386, check=False)
        for i in range(6):
            d = perm[i]
            ids = XT.doc_input_ids(tok, h, [d], idx)[d]
            got = trace_docs[f"mg-X-4386-{i:06d}"]
            self.assertEqual((got["n"], got["head"], got["sum"]), (len(ids), ids[:4], sum(ids)))   # real tokens of pi_s[i]
            self.assertEqual(reqs[i]["requested_output_tokens"], idx[d][2])

    def test_server_info_knob_drift_aborts(self):
        tok = make_token_file(self.root / "t.bin")
        with mock.patch.dict(os.environ, {"FAKE_MODE": "drift"}):
            rc = B.run_boot(boot_spec(self.root, [a_cell()], tok=tok))
        self.assertEqual(rc, B.RC_CONFIG_DRIFT)
        boot = json.loads((self.root / "out/boots/m3_N1_fused_sym_anchor/boot.json").read_text())
        self.assertIn("mamba_ssm_dtype", boot["abort"])


# =========================================================================== §19-6 submission gate + Job script
def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo)] + list(a), capture_output=True, text=True, check=True).stdout.strip()


class PreflightGate(unittest.TestCase):
    def _repo(self, td):
        r = Path(td) / "repo"
        (r / "workspace/engine-port/src").mkdir(parents=True)
        (r / "workspace/engine-port/results/multigpu").mkdir(parents=True)
        (r / "workspace/engine-port/src/a.py").write_text("x = 1\n")
        git(r, "init", "-q")
        git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
        git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "post")
        post = git(r, "rev-parse", "HEAD")
        ed = r / "workspace/engine-port/results/multigpu/ENGINE_DIGEST.txt"
        ed.write_text("abc123\n")
        pf = r / "workspace/engine-port/results/multigpu/PREFLIGHT_PASS.json"
        raw = json.dumps({"verdict": "PREFLIGHT_PASS", "post_commit": post, "merge_tree_engine_digest": "abc123",
                          "image": "reg/img@sha256:feed"}).encode()
        pf.write_bytes(raw)
        Path(str(pf) + ".sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
        git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
        git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pass")
        return r, pf, ed

    def test_gate_i_ii_iii(self):
        with tempfile.TemporaryDirectory() as td:
            r, pf, ed = self._repo(td)
            ok, why = P.preflight_gate(r, pf, "reg/img@sha256:feed", engine_digest_file=ed)
            self.assertTrue(ok, why)
            ok, why = P.preflight_gate(r, pf, "reg/img@sha256:beef", engine_digest_file=ed)
            self.assertFalse(ok)
            self.assertTrue(any(w.startswith("(ii)") for w in why))
            ed.write_text("other\n")
            ok, why = P.preflight_gate(r, pf, "reg/img@sha256:feed", engine_digest_file=ed)
            self.assertTrue(any(w.startswith("(i)") and "digest" in w for w in why))
            ed.write_text("abc123\n")
            (r / "workspace/engine-port/src/a.py").write_text("x = 2\n")                     # a gate path changed after the POST
            git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "change")
            ok, why = P.preflight_gate(r, pf, "reg/img@sha256:feed", engine_digest_file=ed)
            self.assertFalse(ok)
            self.assertTrue(any(w.startswith("(iii)") for w in why), why)
            out = subprocess.run([sys.executable, str(PLAN_SRC), "preflight-gate", str(r), str(pf), "reg/img@sha256:feed"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 3)
            self.assertIn("ABORT_PREFLIGHT_NOT_PASSED", out.stdout)

    def test_uncommitted_pass_file_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r, pf, ed = self._repo(td)
            raw = pf.read_bytes().replace(b"PREFLIGHT_PASS", b"PREFLIGHT_PASS ")
            pf.write_bytes(raw)
            Path(str(pf) + ".sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
            ok, why = P.preflight_gate(r, pf, "reg/img@sha256:feed", engine_digest_file=ed)
            self.assertFalse(ok)
            self.assertTrue(any("not tracked/clean" in w for w in why), why)


def job_layout(src_repo, td):
    """job_entry.sh step 1 (bundle clone + checkout of the exact commit; results/ copied to the run volume and REPLACED BY A SYMLINK).
    -> (project root, run-volume results dir)."""
    td = Path(td)
    td.mkdir(parents=True, exist_ok=True)
    head = git(src_repo, "rev-parse", "HEAD")
    bundle = td / "repo.bundle"
    git(src_repo, "bundle", "create", str(bundle), "HEAD")
    proj = td / "opt" / "project"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(bundle), str(proj)], check=True, capture_output=True)
    git(proj, "checkout", "-q", head)
    run = td / "data" / "runs" / "job" / "results"
    res = proj / "workspace/engine-port/results"
    shutil.copytree(res, run, symlinks=True)
    shutil.rmtree(res)
    res.symlink_to(run)
    return proj, run


def commit_all(r, msg):
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg)


class PreflightGateJobLayout(MutationCase):
    """Harness audit r1 R-H1: gate (i) and m3_job.sh check_committed must hold in the Job layout (results/ is a symlink)."""
    IMG = "reg/img@sha256:feed"
    OLD_GATE = ('        ok_q, why_q = (_git(["ls-files", "--error-unmatch", rel(q)])[0] == 0 and '
                '_git(["diff", "--quiet", "HEAD", "--", rel(q)])[0] == 0), ""')

    def _gate(self, mod, root):
        m = root / "workspace/engine-port/results/multigpu"
        return mod.preflight_gate(root, m / "PREFLIGHT_PASS.json", self.IMG, engine_digest_file=m / "ENGINE_DIGEST.txt")

    @staticmethod
    def _tamper(results_multigpu):
        """verdict text changed AND the .sha256 rewritten to match: only the committed-bytes check can see it."""
        pf = results_multigpu / "PREFLIGHT_PASS.json"
        raw = pf.read_bytes().replace(b'"post"', b'"post" ')
        rec = json.loads(raw)
        rec["note"] = "edited after commit"
        raw = json.dumps(rec).encode()
        pf.write_bytes(raw)
        Path(str(pf) + ".sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")

    def _chk(self, mod, layout):
        with tempfile.TemporaryDirectory() as td:
            r, _pf, _ed = PreflightGate()._repo(td)
            root = r if layout == "plain" else job_layout(r, Path(td) / "job")[0]
            ok, why = self._gate(mod, root)
            if not ok:
                return f"{layout} layout: valid committed PREFLIGHT_PASS rejected: {why}"
            self._tamper(root / "workspace/engine-port/results/multigpu")
            ok, why = self._gate(mod, root)
            if ok or not any("not tracked/clean" in w for w in why):
                return f"{layout} layout: PREFLIGHT_PASS edited after the commit accepted ({why})"
        return None

    def test_job_layout_valid_true_tampered_false(self):
        self.assertIsNone(self._chk(P, "plain"))
        self.assertIsNone(self._chk(P, "job"))
        with tempfile.TemporaryDirectory() as td:                    # the defect itself: the old git check is false in this layout
            r, _pf, _ed = PreflightGate()._repo(td)
            proj, run = job_layout(r, Path(td) / "job")
            self.assertTrue((proj / "workspace/engine-port/results").is_symlink())
            rel = "workspace/engine-port/results/multigpu/PREFLIGHT_PASS.json"
            self.assertNotEqual(subprocess.run(["git", "-C", str(proj), "diff", "--quiet", "HEAD", "--", rel]).returncode, 0)
            self.assertEqual(P.committed_matches_head(proj, proj / rel), (True, ""))
            (run / "multigpu" / "NEW.json").write_text("{}")
            self.assertFalse(P.committed_matches_head(proj, proj / "workspace/engine-port/results/multigpu/NEW.json")[0])   # untracked

    def test_mutant_G1_old_diff_quiet_killed(self):
        self.assert_killed(PLAN_SRC, "G1", "        ok_q, why_q = committed_matches_head(repo, q, head)", self.OLD_GATE,
                           lambda mod: self._chk(mod, "job"))

    def test_mutant_G3_clean_check_removed_killed(self):
        """the auditor's M6 ("gate 깨끗함 검사 제거"), re-sited on the repaired code."""
        self.assert_killed(PLAN_SRC, "G3", "        if not ok_q:", "        if False:", lambda mod: self._chk(mod, "plain"))

    # ---- gate (i) "PASS file sha256 == committed .sha256" (harness audit r2 §5-3 A14 test gap, 2026-10-16)
    A14_OLD = "    if shaf.read_text().split()[:1] != [hashlib.sha256(raw).hexdigest()]:"
    SHA_MISMATCH = "(i) PREFLIGHT_PASS sha256 != committed .sha256"

    @staticmethod
    def _commit_mismatch(r, which):
        """Commit a .json and a .sha256 that disagree, BOTH tracked and clean at HEAD (so committed_matches_head holds for each and
        only the cross-check can see it).  which = "json": the record gains a field, its .sha256 is left at the old digest;
        which = "sha": the record is untouched, its .sha256 names another digest."""
        pf = r / "workspace/engine-port/results/multigpu/PREFLIGHT_PASS.json"
        if which == "json":
            rec = json.loads(pf.read_bytes())
            rec["note"] = "json re-committed without its .sha256"
            pf.write_bytes(json.dumps(rec).encode())
        else:
            Path(str(pf) + ".sha256").write_text("ab" * 32 + "\n")
        commit_all(r, f"mismatch-{which}")

    def _chk_mismatch(self, mod, layout):
        for which in ("json", "sha"):
            with tempfile.TemporaryDirectory() as td:
                r, _pf, _ed = PreflightGate()._repo(td)
                self._commit_mismatch(r, which)
                root = r if layout == "plain" else job_layout(r, Path(td) / "job")[0]
                m = root / "workspace/engine-port/results/multigpu"
                for n in ("PREFLIGHT_PASS.json", "PREFLIGHT_PASS.json.sha256"):            # precondition: both committed and clean
                    if P.committed_matches_head(root, m / n) != (True, ""):
                        return f"{layout}/{which}: precondition broken, {n} not committed-clean"
                ok, why = self._gate(mod, root)
                if ok or why != [self.SHA_MISMATCH]:
                    return f"{layout}/{which}: committed .json/.sha256 mismatch not rejected by the cross-check alone ({ok}, {why})"
        return None

    def test_committed_json_sha256_mismatch_rejected_both_layouts(self):
        self.assertIsNone(self._chk(P, "plain"))                      # the valid committed pair passes (same fixture)
        self.assertIsNone(self._chk_mismatch(P, "plain"))
        self.assertIsNone(self._chk_mismatch(P, "job"))

    def test_mutant_A14_sha_cross_check_removed_killed_plain(self):
        self.assert_killed(PLAN_SRC, "A14p", self.A14_OLD, "    if False:", lambda mod: self._chk_mismatch(mod, "plain"))

    def test_mutant_A14_sha_cross_check_removed_killed_job(self):
        self.assert_killed(PLAN_SRC, "A14j", self.A14_OLD, "    if False:", lambda mod: self._chk_mismatch(mod, "job"))

    # ---- m3_job.sh check_committed (rev11 F17, inherited): the same layout defect, the same repair
    OLD_SH = '  git -C "$ROOT" ls-files --error-unmatch "$1" > /dev/null 2>&1 && git -C "$ROOT" diff --quiet HEAD -- "$1" 2>/dev/null'
    NEW_SH = '  python3 "$HERE/mg_plan.py" committed-clean "$ROOT" "$1" > /dev/null 2>&1'

    @staticmethod
    def _check_committed_rc(script_text, root, f):
        fn = re.search(r"^check_committed \(\) \{.*?^\}", script_text, re.M | re.S).group(0)
        env = dict(os.environ, ROOT=str(root), HERE=str(ENGINE / "results/multigpu"))
        return subprocess.run(["bash", "-c", fn + '\ncheck_committed "$1"', "_", str(f)], env=env, capture_output=True).returncode

    def _chk_sh(self, script_text):
        with tempfile.TemporaryDirectory() as td:
            r, _pf, _ed = PreflightGate()._repo(td)
            m = r / "workspace/engine-port/results/multigpu"
            (m / "M2G_PASS_TABLE_J2.json.sha256").write_text("ab" * 32 + "\n")
            commit_all(r, "table")
            proj, run = job_layout(r, Path(td) / "job")
            f = proj / "workspace/engine-port/results/multigpu/M2G_PASS_TABLE_J2.json.sha256"
            if self._check_committed_rc(script_text, proj, f) != 0:
                return "Job layout: committed, unmodified .sha256 rejected (would be exit 15 ABORT_M2G_NOT_PASSED)"
            (run / "multigpu" / "M2G_PASS_TABLE_J2.json.sha256").write_text("cd" * 32 + "\n")
            if self._check_committed_rc(script_text, proj, f) == 0:
                return "Job layout: .sha256 edited after the commit accepted"
            (run / "multigpu" / "M2G_PASS_TABLE_J4.json.sha256").write_text("ef" * 32 + "\n")
            if self._check_committed_rc(script_text, proj, proj / "workspace/engine-port/results/multigpu/M2G_PASS_TABLE_J4.json.sha256") == 0:
                return "Job layout: untracked .sha256 accepted"
        return None

    def test_m3_job_check_committed_and_mutant_G2_killed(self):
        src = M3_JOB_SH.read_text()
        self.assertIsNone(self._chk_sh(src), "CONTROL")                     # behaviour of the shipped m3_job.sh first
        self.assertEqual(src.count(self.NEW_SH), 1, "mutation site for G2 must occur once")
        self.assertIsNotNone(self._chk_sh(src.replace(self.NEW_SH, self.OLD_SH)), "mutant G2 (old git diff --quiet) SURVIVED")


class PreflightToolRecord(MutationCase):
    """Harness audit r1 R-H2: the preflight tool writes PREFLIGHT_PASS.json itself, in the shape preflight_gate reads, and those bytes,
    once committed, pass the gate in BOTH layouts; a FAIL_NEW / UNVERIFIABLE judge label never does."""
    IMG = "reg/img@sha256:feed"

    @staticmethod
    def _manifest(td):
        spec = __import__("importlib.util").util.spec_from_file_location("_ed_t", ENGINE / "scripts/bootstrap/engine_digest.py")
        ed = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(ed)
        rels = ed.registered_relpaths()
        mf = Path(td) / "arm_post_manifest.sha256"
        mf.write_text("".join(f"{hashlib.sha256(r.encode()).hexdigest()}  /opt/x/python/{r}\n" for r in rels))
        return mf, ed.digest_of_entries(ed.load_entries(manifest=str(mf)))

    def _repo_with(self, td, write_record):
        """POST commit (gate paths) -> the tool writes the record -> its bytes + ENGINE_DIGEST are committed beside the harness."""
        r = Path(td) / "repo"
        (r / "workspace/engine-port/src").mkdir(parents=True)
        (r / "workspace/engine-port/results/multigpu").mkdir(parents=True)
        (r / "workspace/engine-port/src/a.py").write_text("x = 1\n")
        git(r, "init", "-q")
        commit_all(r, "post")
        post = git(r, "rev-parse", "HEAD")
        out = Path(td) / "pf_out"
        out.mkdir()
        digest = write_record(post, out / "PREFLIGHT_PASS.json")
        m = r / "workspace/engine-port/results/multigpu"
        for n in ("PREFLIGHT_PASS.json", "PREFLIGHT_PASS.json.sha256"):
            shutil.copyfile(out / n, m / n)                                  # copied byte for byte, never retyped
        (m / "ENGINE_DIGEST.txt").write_text(digest + "\n")
        commit_all(r, "pass")
        return r

    def _gates(self, r, td):
        res = {}
        for layout in ("plain", "job"):
            root = r if layout == "plain" else job_layout(r, Path(td) / "job")[0]
            m = root / "workspace/engine-port/results/multigpu"
            res[layout] = P.preflight_gate(root, m / "PREFLIGHT_PASS.json", self.IMG, engine_digest_file=m / "ENGINE_DIGEST.txt")
        return res

    def _cli_writer(self, td, label):
        jj = Path(td) / f"judge_{label}.json"
        jj.write_text(json.dumps({"label": label}))
        mf, digest = self._manifest(td)

        def write(post, out):
            p_ = subprocess.run([sys.executable, str(PF_JUDGE_SRC), "record", "--judge-json", str(jj), "--post-commit", post,
                                 "--merge-manifest", str(mf), "--engine-digest-py", str(ENGINE / "scripts/bootstrap/engine_digest.py"),
                                 "--image", self.IMG, "--job-name", "pf-job", "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(p_.returncode, 0 if label == "PREFLIGHT_PASS" else 3, p_.stdout + p_.stderr)
            return digest
        return write

    def test_tool_output_passes_the_gate_in_both_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            g = self._gates(self._repo_with(td, self._cli_writer(td, "PREFLIGHT_PASS")), td)
            self.assertEqual(g, {"plain": (True, []), "job": (True, [])})
        for label in ("PREFLIGHT_FAIL_NEW", "PREFLIGHT_UNVERIFIABLE"):
            with tempfile.TemporaryDirectory() as td:
                g = self._gates(self._repo_with(td, self._cli_writer(td, label)), td)
                for layout, (ok, why) in g.items():
                    self.assertFalse(ok, f"{label} {layout}")
                    self.assertTrue(any(f"verdict '{label}'" in w for w in why), why)

    def _chk_g4(self, mod):
        with tempfile.TemporaryDirectory() as td:
            jj = Path(td) / "judge.json"
            jj.write_text(json.dumps({"label": "PREFLIGHT_FAIL_NEW"}))

            def write(post, out):
                mod.write_pass_record(str(out), mod.pass_record(str(jj), post, "c" * 64, self.IMG, "pf-job"))
                return "c" * 64
            g = self._gates(self._repo_with(td, write), td)
            return None if not any(ok for ok, _ in g.values()) else f"FAIL_NEW judge passed the gate: {g}"

    def test_mutant_G4_fail_new_recorded_as_pass_killed(self):
        self.assert_killed(PF_JUDGE_SRC, "G4", "    verdict = label if label in LABELS else 'PREFLIGHT_UNVERIFIABLE'",
                           "    verdict = PASS_LABEL", self._chk_g4)

    def test_shell_tool_registered_paths_post_and_targets(self):
        sh = (PF_TOOLS / "preflight_cl3.sh").read_text()
        self.assertFalse("POST_DEFAULT" in sh or "PDMUX_PF_POST:-$" in sh, "POST must have no default")       # R-H2
        self.assertIn('POST="${PDMUX_PF_POST:-}"', sh)
        blk = re.search(r'diff --stat "\$\{POST\}" HEAD -- (.*?)> "\$\{OUT\}/meta/post_vs_job_head.txt"', sh, re.S).group(1)
        self.assertEqual(sorted(re.findall(r"workspace/engine-port/([\w/]+)", blk)), sorted(C.PREFLIGHT_GATE_PATHS))
        tg = [l.split("\t") for l in (PF_TOOLS / "preflight_cl3_targets.txt").read_text().splitlines() if l and not l.startswith("#")]
        self.assertIn(["E_rev12", "module", "test_loadgen_outcome"], tg)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
        p_ = subprocess.run(["bash", str(PF_TOOLS / "preflight_cl3.sh"), "--dry-run"], env=env, capture_output=True, text=True)
        self.assertEqual(p_.returncode, 2)
        self.assertIn("PDMUX_PF_POST must be the full 40-hex sha", p_.stderr)
        sel = subprocess.run([sys.executable, str(PF_TOOLS / "preflight_judge_selftest.py")], capture_output=True, text=True)
        self.assertEqual(sel.returncode, 0, sel.stdout[-2000:])


class M3JobScript(unittest.TestCase):
    def run_job(self, job="J2", data_setup=None):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"
            if data_setup:
                data_setup(data)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_MG_DRYRUN="1", PDMUX_MG_TEST_OUT_ROOT=td, PDMUX_MG_JOB=job,
                       PDMUX_JOB_NAME="unit", PDMUX_DATA=str(data))
            p = subprocess.run(["bash", str(ENGINE / "results/multigpu/m3_job.sh")], env=env, capture_output=True, text=True)
            out = Path(td) / "m3_unit"
            notes = (out / "DRYRUN_NOTES.txt").read_text() if (out / "DRYRUN_NOTES.txt").exists() else ""
            files = {f.name for f in out.iterdir()} if out.exists() else set()
            reg = (out / "REGISTRATION_SHA256.txt").read_text() if (out / "REGISTRATION_SHA256.txt").exists() else ""
            return p, notes, files, reg

    def test_dry_run_notes_rev12_gates(self):
        p, notes, files, reg = self.run_job()
        self.assertEqual(p.returncode, 0, p.stdout[-1500:] + p.stderr[-1500:])
        for a in ("ABORT_LOG_KNOB", "ABORT_PREFLIGHT_NOT_PASSED", "ABORT_DATASET"):
            self.assertIn(a, notes)
        self.assertIn("MAMBA_SSM_DTYPE.txt", files)
        for f in ("PREREG_MULTIGPU_rev12_DRAFT_2026-10-15.md", "context_limit.py", "test_loadgen_outcome.py", "mg_x_pretokenize.py",
                  "M2G_SERVER_REFERENCE.json", "test_mg_rev12.py", "mg_xcheck.py"):
            self.assertIn(f, reg)
        self.assertIn("rev12", reg)

    def test_heartbeat_zero_clears_the_log_knob_note(self):
        def setup(data):
            m = data / "runs" / "unit" / "meta"
            m.mkdir(parents=True)
            (m / "log_mirror.txt").write_text("log_mirror=1 heartbeat_s=0\n")
        p, notes, files, _ = self.run_job(data_setup=setup)
        self.assertEqual(p.returncode, 0)
        self.assertNotIn("ABORT_LOG_KNOB", notes)
        self.assertIn("LOG_KNOBS.txt", files)


if __name__ == "__main__":
    unittest.main()
