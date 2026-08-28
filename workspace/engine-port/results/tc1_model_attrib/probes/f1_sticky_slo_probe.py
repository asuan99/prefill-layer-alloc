#!/usr/bin/env python3
"""TC1 F1 probe -- does the engine actually refuse PDMUX_STICKY_PARTITION + PDMUX_SLO_SCHED?

GPU 0.  This is a CODE FACT probe, not a measurement: no model, no server, no CUDA.

It imports the REAL `_init_sticky_partition` and calls it.  It does not
re-implement the guard -- PROJECT_STATUS "방법론 게이트" #9 (자기가 검증하려는
코드를 복사한 검사는 그 코드를 검증하지 않는다) is the reason.

Registered outcome labels:
  REFUSED_AT_INIT   -- the combination raises at scheduler init  (TC1 rev1 F1 CONFIRMED)
  ACCEPTED          -- it does not raise                          (F1 REFUTED)
  PROBE_INVALID     -- the real function could not be reached     (no verdict)
"""
import json, os, sys, hashlib, importlib.util, traceback

SRC = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/multiplexing_mixin.py"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_verdict.json")

res = {"probe": "TC1-F1", "gpu_hours": 0.0, "source": SRC}
res["source_sha256"] = hashlib.sha256(open(SRC, "rb").read()).hexdigest()

# --- reach the real function without importing the whole engine -------------
fn = None
try:
    sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-dummy")  # not present; harmless
    import multiplex.multiplexing_mixin as M           # normal path if installed
    fn = M.MultiplexingMixin._init_sticky_partition
    res["import_path"] = "package import"
except Exception:
    # fall back: exec ONLY the function's own source, taken verbatim from the file
    src = open(SRC, encoding="utf-8").read()
    start = src.index("    def _init_sticky_partition(")
    # end at the next top-level `    def ` after the start
    nxt = src.index("\n    def ", start + 10)
    body = src[start:nxt]
    class _Logger:
        def info(self, *a, **k): pass
    class _FixedPolicy:            # only used by isinstance(); the canonical
        pass                       # bind+GATE run has r2_policy None, so the
                                   # branch is not taken and nothing is faked
    ns = {"os": os, "Optional": __import__("typing").Optional, "Scheduler": object,
          "logger": _Logger(), "FixedPolicy": _FixedPolicy}
    exec("class _Shim:\n" + body, ns)
    fn = ns["_Shim"]._init_sticky_partition
    res["import_path"] = "verbatim function extract (no re-implementation)"
    res["extracted_lines"] = body.count("\n") + 1
    res["extract_sha256"] = hashlib.sha256(body.encode()).hexdigest()

class Stub:
    # values the canonical bind+GATE run actually has
    r2_policy_name = ""                    # PDMUX_R2_POLICY unset
    r2_policy = None                       # -> the FixedPolicy branch is skipped
    real_sm_group_num = 7                  # pdmux_slo.yml sm_group_num: 7
    sm_counts = [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (54, 54), (0, 108)]
    pdmux_config = None

def attempt(env):
    for k in ("PDMUX_STICKY_PARTITION", "PDMUX_SLO_SCHED", "PDMUX_LA_COORD",
              "PDMUX_FIXED_DECODE_SM_FILE"):
        os.environ.pop(k, None)
    os.environ.update(env)
    s = Stub()
    try:
        fn(s)
        return {"raised": False, "exc": None, "msg": None,
                "sticky_enabled": getattr(s, "sticky_partition_enabled", None)}
    except Exception as e:
        return {"raised": True, "exc": type(e).__name__, "msg": str(e)}

cases = {
    # the exact combination TC1 rev1 sec8-2 + sec10 registered
    "sticky1_slosched1": {"PDMUX_STICKY_PARTITION": "1", "PDMUX_SLO_SCHED": "1"},
    # controls: each half alone must be fine, else the probe proves nothing
    "sticky1_only":      {"PDMUX_STICKY_PARTITION": "1"},
    "slosched1_only":    {"PDMUX_SLO_SCHED": "1"},
    "neither":           {},
}
res["cases"] = {k: attempt(v) for k, v in cases.items()}

tgt  = res["cases"]["sticky1_slosched1"]
ctl1 = res["cases"]["sticky1_only"]
ctl2 = res["cases"]["slosched1_only"]

if fn is None:
    res["verdict"] = "PROBE_INVALID"
elif not (ctl1["raised"] is False and ctl2["raised"] is False):
    # if a control also raises, the target raise is not attributable to the COMBINATION
    res["verdict"] = "PROBE_INVALID"
    res["why"] = "a single-flag control also raised; the refusal is not attributable to the pair"
elif tgt["raised"] and tgt["exc"] == "RuntimeError" and "PDMUX_SLO_SCHED" in (tgt["msg"] or ""):
    res["verdict"] = "REFUSED_AT_INIT"
else:
    res["verdict"] = "ACCEPTED"

res["reading"] = {
    "REFUSED_AT_INIT": "TC1 rev1 F1 CONFIRMED — the two knobs rev1 registered as both-required "
                       "cannot coexist; rev2's withdrawal of the sticky requirement is necessary, "
                       "not a convenience.",
}.get(res["verdict"], "see verdict")
res["scope"] = ("Code fact at scheduler init. Says NOTHING about performance, about whether "
                "sticky OFF is the right substrate, or about realized residency magnitudes.")

json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False, sort_keys=True)
print(json.dumps(res, indent=2, ensure_ascii=False))
