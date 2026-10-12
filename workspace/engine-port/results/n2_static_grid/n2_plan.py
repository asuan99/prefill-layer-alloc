#!/usr/bin/env python3
"""N2(3-a) Job plan + registration integrity (PREREG_N2_3A rev16 §3, §4-2, §5, §14-3, §15 `n2_plan.py`). CPU only.

HARNESS-SEAL file (§18-2-a "하네스 봉인(미구현)" row). Built against RULE SEAL digest RULE_SEAL_DIGEST below.

SINGLE SOURCE (rev14 / §15): every registered number this runner uses -- c_in, O_P, O_EA, K, L_lag, delta, per-boot
worst cost, cell-section hard cap -- is read from N2_REGISTERED.json ONLY. This module never imports or re-runs the
calculator (n2_feasibility.py) and never reads K_POST_PROBE.json / GATE_PREVIEW.json.

ABORT_REG_SOURCE (§15, rev15 H14-2, rev16 H15-5) -- `load_registered()` refuses to return constants when:
  (1) the file is missing / unreadable / has duplicate JSON keys,
  (2) the sha256 of the FILE BYTES != REGISTERED_JSON_SHA256 (the value the rule seal recorded for N2_REGISTERED.json
      in REGISTRATION_SHA256.txt, commit fcc0b9e) -- the only condition that refuses a hand-edited file whose
      self-reported fields are intact,
  (3) (auxiliary, self-reported fields, N2C-69) K_matches_derivation is not true, calculator_sha256 != the sealed
      calculator sha (RULE_FILES), limits.ok is not true, or K != derive_fixed_k recorded in the same file.
`verify_rule_seal()` additionally re-hashes the 7 rule-seal files and recomputes the rule-seal digest
(sha256 of the `sha256sum` listing of the 7 files in registration order) against REGISTRATION_SHA256.txt.
`verify_harness_seal(scope=...)` reads the "## HARNESS SEAL" sections of REGISTRATION_SHA256.txt (written at the
harness seals -- format in HARNESS_SECTION_FORMAT). The harness seal is TWO-STAGE (rev17 R17-2 / H17-6): every section
carries a scope tag `scope PROFILE` or `scope EA`; a Job-PROFILE needs the PROFILE section, a Job-EA needs BOTH the
PROFILE and the EA section (REQUIRED_SCOPES). Every required section must be well formed, be the only section of its
scope, carry the current rule-seal digest and list harness files whose CURRENT bytes match (so a harness change after
the PROFILE seal needs a re-issued PROFILE section -- fail-closed, HARNESS CHOICE reported). A missing required
section is UNSEALED (ABORT_HARNESS_UNSEALED; PROFILE dry-run/test mode only notes it, Job-EA is refused always). The harness digest is NOT a constant in this file (this
file is itself a harness file -- a self-referencing digest is impossible).

Units (§5-1): Job-PROFILE = 28 boots, round-major registered permutation; cells inside a boot ordered by the §5-3 /
§14-3 alternation (b order r1·r2 ascending, r3·r4 descending; c order r1·r3 ascending, r2·r4 descending) with
**b as the outer loop and c as the inner loop** (HARNESS CHOICE -- the registration does not state the nesting; the
Job-EA fixed order (8,2048),(8,8192),(32,2048),(32,8192) is b-outer, so the same nesting is used; reported).
Job-EA units (§5-2): 16 boots, round-major EA_ORDER, the fixed E-alpha cell order (R-7) on every boot, K / delta /
c_in / O_EA from N2_REGISTERED only (ON and OFF use c_in(d), R5-2); `plan_problems("EA", units, reg)` checks the
structure and `ea_trace_problems` the per-cell registered values (ABORT_TRACE). The Job-EA runner exists (EA harness
work, not yet sealed): n2_capacity.sh still refuses a Job-EA without BOTH harness-seal sections (H17-6).

CLI (used by n2_capacity.sh): --verify-registration, --verify-harness-seal, --units JOB, --budget JOB,
--need JOB UNIT, --const NAME, --check-config CFG, --check-substrate CSV SM, --check-green JSON CFG KIND D,
--check-wiring LOG KIND D, --p3-verdict FILE, --engine-check MANIFEST, --cg-bs LOG, --emit-plan OUT JOB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Job layout (lesson 270): workspace/engine-port/results is a SYMLINK to /data/runs/<job>/results, and a module
# imported via the cwd ('' on sys.path, e.g. the preflight `cd $HERE && python3 -m unittest`) gets a __file__ under
# the PHYSICAL run dir (getcwd resolves symlinks) -> HERE/../.. leaves the clone and loses benchmarks/ (Job
# pdmux-n2-static-grid-52a3b97-20261011182211, ABORT_PREFLIGHT).  PDMUX_PROJECT_ROOT (exported by job_entry.sh,
# required by n2_capacity.sh) names the clone; without it (local runs) the file-relative root is used.
ENG = (os.path.join(os.path.abspath(os.environ["PDMUX_PROJECT_ROOT"]), "workspace", "engine-port")
       if os.environ.get("PDMUX_PROJECT_ROOT") else os.path.abspath(os.path.join(HERE, "..", "..")))

# ---------------------------------------------------------------------------- rule seal (rev18 2026-10-13; rev17 5e29990; rev16 fcc0b9e)
REGISTERED_JSON = os.path.join(HERE, "N2_REGISTERED.json")
REGISTRATION_TXT = os.path.join(HERE, "REGISTRATION_SHA256.txt")
REGISTERED_JSON_SHA256 = "e7d743c85e6adb007aa940342a3188cdb35568b2135ecd678f3ee6e4ae3087e7"
RULE_SEAL_DIGEST = "56473d6fc1954abdf56f20ccf6cd5cd88d7ea4eeac281124d02803b677645cad"
RULE_FILES = (                                   # registration order (the digest is order-sensitive)
    ("PREREG_N2_3A_2026-10-07.md", "1e2f54df3ae2f7012b5abc1a18f5b9e7c410816d48bc3c70df040023e3ce36fc"),
    ("n2_feasibility.py", "1ee3338dd1f2be7eceaa5e514510199a9d5ac25176b41c5edad27ae205ab0c60"),
    ("n2_probe_reduce.py", "8ef0c8cc6b0ba273ab684a8e297441598ffd70453138ee1f10713a8948d4282d"),
    ("N2_REGISTERED.json", REGISTERED_JSON_SHA256),
    ("test_n2_feasibility.py", "331c7b0ff6efaa0b7d4cc2f5f92e30597455bbb068d165cddb93930c4d786da6"),
    ("test_n2_probe_reduce.py", "a796980c71604d8b7ef94610c2d007d11c6c05e9c3dc899373b894573f2ca171"),
    ("n2_mutants.py", "12b2726b5932fb05a5883059e50dabe2bd747ad2792c85a0c17dc937877c51c7"),
)
CALCULATOR_SHA256 = dict(RULE_FILES)["n2_feasibility.py"]
REDUCER_FILE = os.path.join(HERE, "n2_probe_reduce.py")

# harness-seal files (§18-2-a: the 6 existing harness files + the rev17 Job harness). The harness seal records the
# sha256 of each, in this order, plus the rule-seal digest it was built and tested on (§18-2-c).
HARNESS_FILES = ("n2_marks.py", "n2_loadgen.py", "n2_probe_run.py", "fake_n2_engine.py", "test_n2_harness.py",
                 "n2_harness_mutants.py", "n2_capacity.sh", "n2_plan.py", "n2_job.py", "n2_window.py",
                 "n2_signature.py", "n2_realized_mix.py", "n2_analyze.py", "n2_reach_tests.py", "test_n2_job.py",
                 "n2_job_mutants.py")
HARNESS_SCOPES = ("PROFILE", "EA")
REQUIRED_SCOPES = {"PROFILE": ("PROFILE",), "EA": ("PROFILE", "EA")}   # rev17 R17-2: Job-EA needs both sections
HARNESS_SECTION_FORMAT = """## HARNESS SEAL
scope PROFILE|EA                   (rev17 H17-6 two-stage seal: one section per scope, PROFILE first)
rule_seal_digest <64 hex>          (the RULE SEAL DIGEST this harness was built and tested on)
<64 hex>  <harness file>           (one line per HARNESS_FILES entry, this order)
harness_digest <64 hex>            (sha256 of the listing lines above, sha256sum format)"""

# ---------------------------------------------------------------------------- §3 conditions (registered text)
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
MODEL_REVISION = "dc0661c829b14e5b9246c05cfa89094a0875e052"
BACKEND = "flashinfer"
CTX = 16384
MEM_FRACTION = "0.82"
MAX_RUNNING = 48
SERVER_SEED = 1
CONFIG_REL = "results/n3_capacity/configs/pdmux_n3_A_agn.yml"
CONFIG_DIGEST = "01675b2d7fa8ff23bb1cb142ff66fa4dfd0c021722ae9b8e7acedc2f4d38fca1"
# external digests (§18-2-b "config·분석"): the rule seal file does NOT carry them (reported); pinned here at the
# bytes the rule seal was built on (commit fcc0b9e).
ANALYZE_REL, ANALYZE_DIGEST = "benchmarks/pdmux_eval/analyze.py", \
    "1be495110ea67c95f275b6c8ab7e7af25d0778c2910bf6a830170dbc078aeaa2"
PROFILE_PY_REL, PROFILE_PY_DIGEST = "src/multiplex/profile.py", \
    "d7b6a807f141dc07e706480c1be88887c00f2e154e8a3a60b6ac26ebaf0a766a"
PAIR = {16: (92, 16), 24: (84, 24), 34: (74, 34), 44: (64, 44), 108: (0, 108)}
IDX = {16: 1, 24: 2, 34: 3, 44: 4, 108: 5}           # A-agn stream index of each realized pair (§3-1)
STICKY_IDX = {16: 1, 24: 2, 34: 3, 44: 4}             # §3-1 expected sticky target index
STREAM_GROUPS = [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (0, 108)]
CAPTURE_BS_REQUIRED = (1, 8, 32, 48)                  # §3-2 ABORT_CG_BS: registered b subset of the capture list
P3_MAMBA, P3_TOKEN_FLOOR, P3_REL_TOL = 48, 48 * 16384, 1e-4   # §3-2 P3 (ii) (same text/constants as E2 A14 / N3 R6)

# ENGINE MANIFEST (§15 "엔진 트리 = HEAD 등록 digest 일치"): the 25-entry installed-tree manifest registered by N3
# (n3_plan.ENGINE_MANIFEST, = λ0 VESSL + PHASE_EVENTS mixin); the src/ mirror at fcc0b9e equals it (mirror check).
ENGINE_MANIFEST = {
    "sglang/srt/distributed/parallel_state.py": "167424ad6fcf8b57cd17f19ef46bb0e6ce792bd2fa36bb2b024309d8a5f1f78e",
    "sglang/srt/multiplex/dual_worker.py": "7a5022f8681446de3a04b2b868a475951594c5e766f9dbeef632ba40ade716ab",
    "sglang/srt/multiplex/multiplexing_mixin.py": "ba5835aa9f935652860f73c2e38fdb550705f3d462b7ed25015c57d28d2aff8b",
    "sglang/srt/multiplex/profile.py": "d7b6a807f141dc07e706480c1be88887c00f2e154e8a3a60b6ac26ebaf0a766a",
    "sglang/srt/multiplex/controller.py": "45fd29678b2468bc95a561afc887c0479c9ade806d2b0563858026309845eba7",
    "sglang/srt/multiplex/telemetry.py": "e4399835717ff13a50bf8c8720c40d39b308c5ec8cb7df8aff65b4cfba88b0f1",
    "sglang/srt/multiplex/holb_probe.py": "97cd7392287158523337860a334f88b1924ab272cc1d6eb6f41f083912fd1f3f",
    "sglang/srt/multiplex/chunk_probe.py": "7dfa76a9097244a8b9021dd4fe552046ee1cf3d0dbb8a3eafe9e2ad110b2c4b9",
    "sglang/srt/multiplex/green_readout.py": "84b3e728f40cf5244bf3d460bf3f33a68bdc3f7a84e85a9ce7aa54ee96e58672",
    "sglang/srt/managers/scheduler.py": "023b9f1a70592261484c629c43df88d9dfa0b5c41d35a4b96cd13e1587478321",
    "sglang/srt/configs/mamba2.py": "e0488b2035af9091a32058afabb85d65fcdefd911f118f5a8802f846fbb91501",
    "sglang/srt/models/mamba2.py": "43b2a9e4f02fa65df326fb87735a3d96fde5ac0b39ee49f1e85b03f74fbc3ca7",
    "sglang/srt/configs/zamba2.py": "97e91d40908c956014691f64aa03acaf11aa9ebd0bc84cc129a96c37b5bbe437",
    "sglang/srt/models/zamba2.py": "035a7d758ada757a76af4b76332dd1f8e769ea24195b2169fce8b7b7b7a1e264",
    "sglang/srt/server_args.py": "f2e89a1051d4537d959b48768693a21e6b713a7ecbf29d7315329242b09be388",
    "sglang/srt/model_executor/model_runner_kv_cache_mixin.py": "b673f3fb378791065b97b74f03ca68ab4f5ad0a5e1aad70d7fe101d85a69d514",
    "sglang/srt/mem_cache/memory_pool.py": "0e99909a421f004cf25d7f79ad693882301a0de75395c05566ff859ff103e1b2",
    "sglang/srt/models/nemotron_h.py": "713333e87b4a697ad9004860633a8a6fbc57f98358c9d55f2f7f4333293571b0",
    "sglang/srt/models/falcon_h1.py": "3fb851ad84cdd22b4d37fb0147e4a4c3ed24063fce026ea23e1087751f526794",
    "sglang/srt/models/granitemoehybrid.py": "b97f83121ae215b1f6b8e6771ea0bf8093311a4f40cc79b09c22ed8d58ef5ad3",
    "sglang/srt/configs/nemotron_h.py": "bcdface1473d506208d48adca7bbe21eed0d558498da9d5b389fb941351f2f76",
    "sglang/srt/configs/falcon_h1.py": "8057f56c915654f83eed5df0a8dce720744ebb33723fcf08e4f8c4c76a83e930",
    "sglang/srt/configs/granitemoehybrid.py": "f103f9bc0a893d1bbfb644f6f8137a3b252460acaf8b26803e3fb2943e1b509d",
    "sglang/srt/configs/mamba_utils.py": "58e6f716e7dec0d538777442b3e912b48aac4ae4832d1e0977d8e5676c79451a",
    "sglang/srt/layers/attention/flashinfer_backend.py": "9181648bb265f130bdf72d95bce0c2396288e9212cc2d7321c59fe8d24f86983",
}
SUBSTRATE_SM_COUNT, SUBSTRATE_CC, SUBSTRATE_MIN_MEM_MIB, SUBSTRATE_MIN_DRIVER_MAJOR = 108, "8.0", 79000, 550

# ---------------------------------------------------------------------------- §4 / §5 grid + structure
B_GRID = (1, 8, 32, 48)
C_GRID = (512, 2048, 8192)
D_ON = (16, 24, 34, 44)
OBS_CELLS = ((1, 2048), (32, 2048))
EA_CELLS = ((8, 2048), (8, 8192), (32, 2048), (32, 8192))     # §5-2 R-7 fixed order
U1 = "b"                                                     # §18-1 U-1 (b): client cancel after the window
# §5-1 registered permutation (structure test fixed)
PROFILE_ORDER = {
    1: ("OBS44", "ON44", "OFF44", "ON24", "ON16", "OFF16", "ON34"),
    2: ("ON34", "OFF16", "ON16", "ON24", "OFF44", "ON44", "OBS44"),
    3: ("OBS44", "ON44", "OFF44", "ON34", "OFF16", "ON16", "ON24"),
    4: ("ON24", "ON34", "ON16", "OFF16", "OFF44", "ON44", "OBS44"),
}
EA_ORDER = {
    1: ("ON44", "OFF44", "ON16", "OFF16"),
    2: ("OFF16", "ON16", "OFF44", "ON44"),
    3: ("OFF44", "ON44", "ON16", "OFF16"),
    4: ("OFF16", "ON16", "ON44", "OFF44"),
}
GREEDY_BOOTS = {"r1_ON16", "r1_ON24", "r1_ON34", "r1_ON44", "r1_OFF16"}   # §5-1 / §11-2


class RegSourceError(SystemExit):
    """ABORT_REG_SOURCE (exit 3)."""

    def __init__(self, msg):
        super().__init__(f"ABORT_REG_SOURCE {msg}")
        self.msg = msg


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def _no_dup(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON keys")
    return dict(pairs)


# ---------------------------------------------------------------------------- ABORT_REG_SOURCE
def load_registered(path=REGISTERED_JSON, expected_sha256=REGISTERED_JSON_SHA256, calculator_sha256=CALCULATOR_SHA256):
    """Return the registered constants dict, or raise RegSourceError. `expected_sha256` / `calculator_sha256` are
    parameters only so the tests can load a synthetic registered file with ITS OWN digest (the runner always uses the
    sealed constants)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise RegSourceError(f"N2_REGISTERED.json unreadable: {e!r}")
    got = sha256_bytes(raw)
    if got != expected_sha256:
        raise RegSourceError(f"N2_REGISTERED.json sha256 {got} != rule-seal value {expected_sha256}")
    try:
        reg = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dup)
    except ValueError as e:
        raise RegSourceError(f"N2_REGISTERED.json unparseable: {e!r}")
    bad = []
    if reg.get("K_matches_derivation") is not True:
        bad.append("K_matches_derivation is not true")
    if reg.get("calculator_sha256") != calculator_sha256:
        bad.append(f"calculator_sha256 {reg.get('calculator_sha256')} != sealed {calculator_sha256}")
    if (reg.get("limits") or {}).get("ok") is not True:
        bad.append("limits.ok is not true")
    if reg.get("K") != reg.get("derive_fixed_k"):
        bad.append(f"K {reg.get('K')} != derive_fixed_k {reg.get('derive_fixed_k')}")
    for key in ("derived", "budget", "K", "L_LAG", "DELTA_FIXED"):
        if key not in reg:
            bad.append(f"missing {key}")
    if bad:
        raise RegSourceError("; ".join(bad))
    return reg


def rule_listing(here=HERE):
    lines = []
    for fn, _want in RULE_FILES:
        lines.append(f"{sha256_file(os.path.join(here, fn))}  {fn}\n")
    return "".join(lines)


def verify_rule_seal(here=HERE, registration_txt=None):
    """Problems (empty = OK): every rule-seal file sha == RULE_FILES, the listing digest == RULE_SEAL_DIGEST, and
    REGISTRATION_SHA256.txt's RULE SEAL DIGEST line == RULE_SEAL_DIGEST."""
    bad = []
    for fn, want in RULE_FILES:
        p = os.path.join(here, fn)
        got = sha256_file(p) if os.path.exists(p) else "MISSING"
        if got != want:
            bad.append(f"{fn}: {got} != sealed {want}")
    if not bad and sha256_bytes(rule_listing(here).encode()) != RULE_SEAL_DIGEST:
        bad.append("rule listing digest != RULE_SEAL_DIGEST")
    txt_path = registration_txt or os.path.join(here, "REGISTRATION_SHA256.txt")
    try:
        with open(txt_path) as fh:
            txt = fh.read()
    except OSError:
        txt = ""
    m = re.search(r"## RULE SEAL DIGEST[^\n]*\n([0-9a-f]{64})", txt)
    if not m or m.group(1) != RULE_SEAL_DIGEST:
        bad.append("REGISTRATION_SHA256.txt RULE SEAL DIGEST missing or != RULE_SEAL_DIGEST")
    return bad


def harness_listing(here=HERE, files=HARNESS_FILES):
    return "".join(f"{sha256_file(os.path.join(here, fn))}  {fn}\n" for fn in files)


def parse_harness_sections(txt):
    """-> list of dict(scope, rule_digest, files {file: sha}, harness_digest, problems) -- one per '## HARNESS SEAL'
    section, file order; empty when there is none."""
    out = []
    for blk in re.split(r"^## ", txt, flags=re.M)[1:]:
        if not blk.startswith("HARNESS SEAL"):
            continue
        sc = re.findall(r"^scope (\S+)$", blk, re.M)
        rd = re.search(r"^rule_seal_digest ([0-9a-f]{64})$", blk, re.M)
        hd = re.search(r"^harness_digest ([0-9a-f]{64})$", blk, re.M)
        files = dict((m.group(2), m.group(1)) for m in re.finditer(r"^([0-9a-f]{64})  (\S+)$", blk, re.M))
        probs = []
        if len(sc) != 1:
            probs.append(f"section needs exactly one 'scope' line (got {sc})")
        elif sc[0] not in HARNESS_SCOPES:
            probs.append(f"unknown scope {sc[0]!r}")
        out.append(dict(scope=sc[0] if len(sc) == 1 else None, rule_digest=rd.group(1) if rd else None, files=files,
                        harness_digest=hd.group(1) if hd else None, problems=probs))
    return out


def _verify_section(here, sec):
    bad = list(sec["problems"])
    rd, files, hd = sec["rule_digest"], sec["files"], sec["harness_digest"]
    if rd != RULE_SEAL_DIGEST:
        bad.append(f"harness built on rule digest {rd} != current {RULE_SEAL_DIGEST} (§18-2-c: re-run the harness ledger)")
    if list(files) != list(HARNESS_FILES):
        bad.append(f"harness file list {list(files)} != {list(HARNESS_FILES)}")
    for fn, want in files.items():
        p = os.path.join(here, fn)
        got = sha256_file(p) if os.path.exists(p) else "MISSING"
        if got != want:
            bad.append(f"{fn}: {got} != sealed {want}")
    listing = "".join(f"{files[fn]}  {fn}\n" for fn in files)
    if hd != sha256_bytes(listing.encode()):
        bad.append("harness_digest != sha256 of the listed lines")
    return [f"[scope {sec['scope']}] {b}" for b in bad]


def verify_harness_seal(here=HERE, registration_txt=None, scope="PROFILE"):
    """(status, problems) for a Job of `scope`: status 'SEALED' / 'UNSEALED' (a required section is absent) /
    'BROKEN' (any section malformed, a duplicated or unknown scope, or a required section that does not verify)."""
    if scope not in REQUIRED_SCOPES:
        return "BROKEN", [f"unknown Job scope {scope!r}"]
    try:
        with open(registration_txt or os.path.join(here, "REGISTRATION_SHA256.txt")) as fh:
            txt = fh.read()
    except OSError:
        return "UNSEALED", ["REGISTRATION_SHA256.txt unreadable"]
    secs = parse_harness_sections(txt)
    bad = [f"[section {i}] {p}" for i, s_ in enumerate(secs) for p in s_["problems"]]
    by = {}
    for s_ in secs:
        if s_["scope"] is not None:
            by.setdefault(s_["scope"], []).append(s_)
    for sc, lst in by.items():
        if len(lst) != 1:
            bad.append(f"scope {sc} sealed {len(lst)} times (one section per scope)")
    missing = [sc for sc in REQUIRED_SCOPES[scope] if sc not in by]
    for sc in REQUIRED_SCOPES[scope]:
        if sc in by and len(by[sc]) == 1:
            bad += _verify_section(here, by[sc][0])
    if bad:
        return "BROKEN", bad
    if missing:
        return "UNSEALED", [f"no '## HARNESS SEAL' section with 'scope {sc}' (required for a Job-{scope}; "
                            f"harness seal pending)" for sc in missing]
    return "SEALED", []


# ---------------------------------------------------------------------------- units
def _boot_tuple(tag):
    kind = tag.rstrip("0123456789")
    return kind, int(tag[len(kind):])


def profile_cell_order(r):
    bs = list(B_GRID) if r in (1, 2) else list(reversed(B_GRID))
    cs = list(C_GRID) if r in (1, 3) else list(reversed(C_GRID))
    return [(b, c) for b in bs for c in cs]                       # b outer, c inner (HARNESS CHOICE, reported)


def obs_cell_order(r):
    return sorted(OBS_CELLS, key=lambda bc: bc[0], reverse=r not in (1, 2))


def units(job, reg):
    """The registered boot list of one Job, in execution order. Every number comes from `reg` (N2_REGISTERED)."""
    out = []
    der = reg["derived"]
    if job == "PROFILE":
        for r in (1, 2, 3, 4):
            for tag in PROFILE_ORDER[r]:
                kind, d = _boot_tuple(tag)
                name = f"r{r}_{tag}"
                alone_d = 108 if kind == "OFF" else d
                cells_bc = obs_cell_order(r) if kind == "OBS" else profile_cell_order(r)
                cells = []
                for b, c in cells_bc:
                    p = der["profile"][f"{alone_d},{b},{c}"]
                    cells.append(dict(cell_id=f"{name}-b{b}-c{c}", kind="OBS" if kind == "OBS" else "PROFILE",
                                      seed_kind="PROFILE", b=b, c=c, c_in=int(p["c_in"]), O=int(math.ceil(p["O_P"])),
                                      O_raw=p["O_P"], s_hat=p["s_hat"], r=r))
                out.append(dict(name=name, job="PROFILE", round=r, kind=kind, D=d, fixed_dsm=d, alone_D=alone_d,
                                sticky=kind in ("ON", "OBS"), phase_events=kind != "OBS", cost_key=tag,
                                greedy=name in GREEDY_BOOTS, cells=cells))
    elif job == "EA":
        for r in (1, 2, 3, 4):
            for tag in EA_ORDER[r]:
                kind, d = _boot_tuple(tag)
                name = f"r{r}_{tag}"
                cells = []
                for b, c in EA_CELLS:
                    e = der["ealpha"][f"{d},{b},{c}"]
                    cells.append(dict(cell_id=f"{name}-ea-b{b}-c{c}", kind="EALPHA", seed_kind="EALPHA", b=b, c=c,
                                      c_in=int(e["c_in"]), O=int(math.ceil(e["O_EA"])), O_raw=e["O_EA"],
                                      K=int(reg["K"][str(d)]), delta=int(reg["DELTA_FIXED"]), r=r))
                out.append(dict(name=name, job="EA", round=r, kind=kind, D=d, fixed_dsm=d,
                                alone_D=108 if kind == "OFF" else d, sticky=kind == "ON", phase_events=True,
                                cost_key=tag, greedy=False, cells=cells))
    else:
        raise ValueError(f"job must be PROFILE or EA, got {job!r}")
    return out


def budget(job, reg):
    col = f"{job}_{U1}"
    b = reg["budget"][col]
    return dict(column=col, cap_s=float(b["cap"]), per_boot_worst=dict(b["per_boot_worst"]),
                per_boot_nom=dict(b["per_boot_nom"]))


def need_s(job, reg, unit_name):
    """§14-3: the boot's WORST-corner cost (N2_REGISTERED budget, U-1 (b) column)."""
    tag = unit_name.split("_", 1)[1]
    return float(budget(job, reg)["per_boot_worst"][tag])


def admission(remaining_s, need):
    """§14-3 STOP rule: admit iff remaining >= worst-corner cost. The caller stops at the FIRST refusal."""
    return remaining_s >= need


def trace_problems(unit):
    """ABORT_TRACE checks of one boot's cells (PROFILE: decode-only payloads)."""
    bad = []
    for c in unit["cells"]:
        if c["O"] < 1:
            bad.append(f"{c['cell_id']}: O {c['O']} < 1")
        if c["c_in"] < 1 or c["c_in"] + c["O"] > CTX:
            bad.append(f"{c['cell_id']}: c_in {c['c_in']} + O {c['O']} outside (0, {CTX}]")
        if c["c_in"] % 8192 == 0:
            bad.append(f"{c['cell_id']}: c_in {c['c_in']} is a multiple of 8192 (co-type span)")
        if c["b"] > MAX_RUNNING:
            bad.append(f"{c['cell_id']}: b {c['b']} > cap {MAX_RUNNING}")
    return bad


def ea_trace_problems(unit, reg):
    """ABORT_TRACE checks specific to a Job-EA boot (§4-2, §10-1, §3-1 co-load row): every cell is an E-alpha cell of
    the registered fixed order (R-7), its K is the registered K_d of the boot's d (the SAME K on ON and OFF boots),
    its delta is the registered fixed allowance, and c_in / O_EA are the registered E-alpha values of d (ON and OFF
    use c_in(d), R5-2)."""
    bad = []
    d = unit["D"]
    if [(c["b"], c["c"]) for c in unit["cells"]] != list(EA_CELLS):
        bad.append(f"{unit['name']}: E-alpha cell order {[(c['b'], c['c']) for c in unit['cells']]} != {list(EA_CELLS)}")
    for c in unit["cells"]:
        e = reg["derived"]["ealpha"][f"{d},{c['b']},{c['c']}"]
        if c.get("kind") != "EALPHA":
            bad.append(f"{c['cell_id']}: kind {c.get('kind')} != EALPHA")
        if c.get("K") != int(reg["K"][str(d)]):
            bad.append(f"{c['cell_id']}: K {c.get('K')} != registered K_{d} {reg['K'][str(d)]}")
        if c.get("delta") != int(reg["DELTA_FIXED"]):
            bad.append(f"{c['cell_id']}: delta {c.get('delta')} != registered {reg['DELTA_FIXED']}")
        if c["c_in"] != int(e["c_in"]) or c["O"] != int(math.ceil(e["O_EA"])):
            bad.append(f"{c['cell_id']}: c_in/O {c['c_in']}/{c['O']} != registered E-alpha {e['c_in']}/{e['O_EA']}")
    return bad


def plan_problems(job, units_, reg=None):
    """Structure checks (§5): boot count, round permutation, OBS44 adjacent to ON44 in every round, cell count;
    Job-EA (§5-2): 16 boots / 64 cells, the registered round permutation, every E-alpha pair (ON-d, OFF-d) adjacent
    in every round, ON first twice and OFF first twice per d, and (with `reg`) the EA trace checks."""
    bad = []
    if job == "EA":
        if len(units_) != 16:
            bad.append(f"EA boots {len(units_)} != 16")
        if sum(len(u["cells"]) for u in units_) != 64:
            bad.append("EA cells != 64")
        first = {16: [], 44: []}
        for r in (1, 2, 3, 4):
            tags = [u["cost_key"] for u in units_ if u["round"] == r]
            if tags != list(EA_ORDER[r]):
                bad.append(f"EA round {r} order {tags} != registered")
                continue
            for d in (16, 44):
                i, j = tags.index(f"ON{d}"), tags.index(f"OFF{d}")
                if abs(i - j) != 1:
                    bad.append(f"EA round {r}: ON{d} not adjacent to OFF{d}")
                first[d].append("ON" if i < j else "OFF")
        for d, seq in first.items():
            if len(seq) == 4 and sorted(seq) != ["OFF", "OFF", "ON", "ON"]:
                bad.append(f"EA d={d}: ON first {seq.count('ON')} times (registered 2 / 2)")
        if reg is not None:
            for u in units_:
                bad += ea_trace_problems(u, reg)
    if job == "PROFILE":
        if len(units_) != 28:
            bad.append(f"PROFILE boots {len(units_)} != 28")
        if sum(len(u["cells"]) for u in units_) != 296:
            bad.append("PROFILE cells != 296")
        for r in (1, 2, 3, 4):
            tags = [u["cost_key"] for u in units_ if u["round"] == r]
            if tags != list(PROFILE_ORDER[r]):
                bad.append(f"round {r} order {tags} != registered")
            i, j = tags.index("OBS44"), tags.index("ON44")
            if abs(i - j) != 1:
                bad.append(f"round {r}: OBS44 not adjacent to ON44")
    for u in units_:
        bad += trace_problems(u)
    ids = [c["cell_id"] for u in units_ for c in u["cells"]]
    if len(ids) != len(set(ids)):
        bad.append("duplicate cell ids")
    return bad


# ---------------------------------------------------------------------------- config / substrate / green / wiring
def parse_config(path):
    try:
        import yaml
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
    except ImportError:                                            # pragma: no cover -- minimal subset fallback
        cfg = {"manual_divisions": []}
        for raw in open(path).read().splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.lstrip().startswith("- ["):
                cfg["manual_divisions"].append([int(v) for v in line.split("[", 1)[1].split("]", 1)[0].split(",")])
            elif ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                if v.strip():
                    cfg[k.strip()] = int(v.strip())
    return cfg


def check_config(path):
    """§8-1 (parser, lesson 258): 6 stream groups = [(108,0)] + divisions + [(0,108)]; every D in D_ON is exactly
    one division row (unambiguous _sticky_fixed_idx = divisions[0]) at the expected index."""
    bad = []
    if sha256_file(path) != CONFIG_DIGEST:
        bad.append("config bytes != registered digest (ABORT_REG_DRIFT)")
    cfg = parse_config(path)
    divs = [tuple(d[:2]) for d in cfg.get("manual_divisions") or []]
    groups = [(108, 0)] + divs + [(0, 108)]
    if groups != STREAM_GROUPS:
        bad.append(f"stream groups {groups} != {STREAM_GROUPS}")
    if cfg.get("sm_group_num") != len(groups):
        bad.append(f"sm_group_num {cfg.get('sm_group_num')} != {len(groups)}")
    for d in D_ON:
        hits = [i for i, g in enumerate(groups) if g[1] == d and 1 <= i <= len(groups) - 2]
        if hits != [STICKY_IDX[d]]:
            bad.append(f"D{d}: division indices {hits} != [{STICKY_IDX[d]}]")
    return bad


def check_substrate(csv_line, sm_count):
    """B0 (§3-3, same rule as N3 check_substrate): A100 80GB, cc 8.0, >= 79,000 MiB, driver >= 550, MIG off, SM 108."""
    f = [x.strip() for x in csv_line.split(",")]
    if len(f) < 5:
        return [f"unparseable nvidia-smi line: {csv_line!r}"]
    name, cc, mem, drv, mig = f[:5]
    bad = []
    if "A100" not in name or "80GB" not in name:
        bad.append(f"gpu name {name!r} is not an A100 80GB")
    if cc != SUBSTRATE_CC:
        bad.append(f"compute_cap {cc} != {SUBSTRATE_CC}")
    try:
        if float(mem) < SUBSTRATE_MIN_MEM_MIB:
            bad.append(f"memory.total {mem} < {SUBSTRATE_MIN_MEM_MIB}")
    except ValueError:
        bad.append(f"memory.total {mem!r} unparseable")
    try:
        if int(drv.split(".")[0]) < SUBSTRATE_MIN_DRIVER_MAJOR:
            bad.append(f"driver {drv} < {SUBSTRATE_MIN_DRIVER_MAJOR}")
    except ValueError:
        bad.append(f"driver {drv!r} unparseable")
    if mig not in ("Disabled", "[N/A]", "N/A"):
        bad.append(f"MIG mode {mig!r}")
    try:
        if int(sm_count) != SUBSTRATE_SM_COUNT:
            bad.append(f"SM count {sm_count} != {SUBSTRATE_SM_COUNT}")
    except (TypeError, ValueError):
        bad.append(f"SM count {sm_count!r} unparseable")
    return bad


def check_green(green, sticky, d):
    """§11-1 green read-out: ON/OBS -> sticky_target_index == STICKY_IDX[d], only that group is_sticky_target;
    OFF -> sticky_target_index null and no sticky group; every boot: 6/6 groups target the registered pairs, the
    division groups' green smCount == target, the plain groups (index 0, 5) carry no green context."""
    bad = []
    want_idx = STICKY_IDX[d] if sticky else None
    if green.get("sticky_target_index") != want_idx:
        bad.append(f"sticky_target_index {green.get('sticky_target_index')!r} != {want_idx!r}")
    groups = {g.get("stream_index"): g for g in green.get("groups", [])}
    if sorted(groups) != list(range(len(STREAM_GROUPS))):
        bad.append(f"stream indices {sorted(groups)} != 0..{len(STREAM_GROUPS) - 1}")
    for i, (p, dd) in enumerate(STREAM_GROUPS):
        g = groups.get(i)
        if g is None:
            continue
        if bool(g.get("is_sticky_target")) != (want_idx == i):
            bad.append(f"group {i} is_sticky_target {g.get('is_sticky_target')} (expected {want_idx == i})")
        if (g.get("target_prefill_sm"), g.get("target_decode_sm")) != (p, dd):
            bad.append(f"group {i} targets {(g.get('target_prefill_sm'), g.get('target_decode_sm'))} != {(p, dd)}")
        division = 1 <= i <= len(STREAM_GROUPS) - 2
        for side, sm in (("prefill", p), ("decode", dd)):
            rec = g.get(side) or {}
            got = (rec.get("green_sm") or {}).get("smCount")
            if division and (rec.get("green_ctx_is_null", True) or got != sm):
                bad.append(f"group {i} {side}: green smCount {got} != target {sm}")
            if not division and not rec.get("green_ctx_is_null", True):
                bad.append(f"group {i} {side}: plain group carries a green context")
    return bad


STICKY_LINE = re.compile(r"PD-mux sticky partition ENABLED.*\(fixed target index=(\d+)\)")


def check_wiring(log_text, sticky, d):
    """§11-1 server log: ON/OBS -> the ENABLED line exactly once with the expected index; OFF -> no ENABLED line."""
    hits = STICKY_LINE.findall(log_text)
    n_any = log_text.count("PD-mux sticky partition ENABLED")
    if sticky:
        if n_any != 1 or hits != [str(STICKY_IDX[d])]:
            return [f"sticky ENABLED lines {n_any}, indices {hits} != exactly one with index {STICKY_IDX[d]}"]
        return []
    return [f"sticky ENABLED line present on a sticky-OFF boot ({n_any})"] if n_any else []


CAPTURE_RE = re.compile(r"Capture cuda graph bs \[([0-9, ]+)\]")


def capture_bs(log_text):
    m = CAPTURE_RE.findall(log_text)
    return [int(x) for x in m[-1].split(",")] if m else None


def cg_bs_problems(log_text):
    got = capture_bs(log_text)
    if got is None:
        return ["no 'Capture cuda graph bs [...]' line"]
    miss = [b for b in CAPTURE_BS_REQUIRED if b not in got]
    return [f"registered b {miss} not in the capture list {got}"] if miss else []


def banners(log_text):
    mm = re.findall(r"max_mamba_cache_size: *([0-9]+)", log_text)
    mt = re.findall(r"max_total_num_tokens=([0-9]+)", log_text)
    return (int(mm[-1]) if mm else None), (int(mt[-1]) if mt else None)


def p3_verdict(rows):
    """§3-2 P3 (ii) over all boots so far: rows = [(mamba, tokens)] (None = banner missing). (ok, line)."""
    if not rows:
        return False, "FAIL no_keys"
    if any(m is None or t is None for m, t in rows):
        return False, "FAIL banner_missing"
    mamba = sorted({int(m) for m, _ in rows})
    tok = [int(t) for _, t in rows]
    if mamba != [P3_MAMBA]:
        return False, f"FAIL mamba={mamba} != [{P3_MAMBA}]"
    if min(tok) < P3_TOKEN_FLOOR:
        return False, f"FAIL tokens_min={min(tok)} < floor={P3_TOKEN_FLOOR}"
    rel = (max(tok) - min(tok)) / max(tok)
    ok = rel <= P3_REL_TOL
    return ok, (("PASS" if ok else "FAIL") + f" mamba={P3_MAMBA} tokens_min={min(tok)} tokens_max={max(tok)} "
                f"rel={rel:.3e} tol={P3_REL_TOL:g} n_boots={len(rows)}")


def manifest_entries(path):
    out = {}
    with open(path) as fh:
        text = fh.read()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, p = line.split(None, 1)
        out[p.split("/python/", 1)[1] if "/python/" in p else p] = h
    return out


def engine_check(manifest_path):
    a = manifest_entries(manifest_path)
    return [f"{k}: installed={a.get(k, 'ABSENT')[:16]} registered={ENGINE_MANIFEST.get(k, 'ABSENT')[:16]}"
            for k in sorted(set(a) | set(ENGINE_MANIFEST)) if a.get(k) != ENGINE_MANIFEST.get(k)]


def external_digest_problems(eng=ENG):
    bad = []
    for rel, want in ((CONFIG_REL, CONFIG_DIGEST), (ANALYZE_REL, ANALYZE_DIGEST), (PROFILE_PY_REL, PROFILE_PY_DIGEST)):
        p = os.path.join(eng, rel)
        got = sha256_file(p) if os.path.exists(p) else "MISSING"
        if got != want:
            bad.append(f"{rel}: {got} != {want}")
    return bad


def server_flags(cfg_path):
    """§3-1: the N3 rev6 server text (cudagraph ON)."""
    return ["--model-path", MODEL, "--trust-remote-code", "--dtype", "bfloat16", "--attention-backend", BACKEND,
            "--enable-pdmux", "--pdmux-config-path", cfg_path, "--disable-overlap-schedule",
            "--chunked-prefill-size", "-1", "--disable-radix-cache", "--mem-fraction-static", MEM_FRACTION,
            "--max-running-requests", str(MAX_RUNNING), "--context-length", str(CTX),
            "--random-seed", str(SERVER_SEED)]


def boot_env(base_env, unit, tel_path, green_path, run_id):
    """§3-1 per-boot knobs: legacy loop (TRUE_DUAL unset), fixed policy, FIXED_DSM = boot D, sticky ON/OBS = 1 and
    OFF unset, PHASE_EVENTS = 1 on ON/OFF and unset on OBS, green read-out on, telemetry one file per boot."""
    keep = ("PDMUX_ROOT", "PDMUX_PROJECT_ROOT", "PDMUX_DATA", "PDMUX_IO", "PDMUX_COMMIT", "PDMUX_JOB_NAME",
            "PDMUX_ENERGY_SAMPLER")
    env = {k: v for k, v in base_env.items() if not (k.startswith("PDMUX_") and k not in keep)}
    env.update(PDMUX_TELEMETRY_PATH=tel_path, PDMUX_RUN_ID=run_id, PDMUX_WORKLOAD_ID=f"n2_{unit['name']}",
               PDMUX_R2_POLICY="fixed", PDMUX_R2_FIXED_DSM=str(unit["fixed_dsm"]), PDMUX_GREEN_READOUT="1",
               PDMUX_GREEN_READOUT_PATH=green_path, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1")
    if unit["sticky"]:
        env["PDMUX_STICKY_PARTITION"] = "1"
    if unit["phase_events"]:
        env["PDMUX_PHASE_EVENTS"] = "1"
    env.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
    return env


# ---------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-registration", action="store_true")
    ap.add_argument("--verify-harness-seal", action="store_true")
    ap.add_argument("--scope", default="PROFILE", choices=sorted(REQUIRED_SCOPES),
                    help="Job scope for --verify-harness-seal (PROFILE: PROFILE section; EA: PROFILE + EA sections)")
    ap.add_argument("--units", metavar="JOB")
    ap.add_argument("--emit-plan", nargs=2, metavar=("OUT", "JOB"))
    ap.add_argument("--budget", metavar="JOB")
    ap.add_argument("--need", nargs=2, metavar=("JOB", "UNIT"))
    ap.add_argument("--const", metavar="NAME")
    ap.add_argument("--check-config", metavar="CFG")
    ap.add_argument("--check-substrate", nargs=2, metavar=("CSV", "SM"))
    ap.add_argument("--engine-check", metavar="MANIFEST")
    ap.add_argument("--external-digests", action="store_true")
    a = ap.parse_args(argv)
    if a.verify_registration:
        try:
            reg = load_registered()
        except RegSourceError as e:
            print(e.code)
            return 3
        bad = verify_rule_seal()
        for b in bad:
            print("ABORT_REG_SOURCE " + b)
        if bad:
            return 3
        print(f"REG_OK N2_REGISTERED sha {REGISTERED_JSON_SHA256} rule_seal {RULE_SEAL_DIGEST} K {reg['K']}")
        return 0
    if a.verify_harness_seal:
        st, bad = verify_harness_seal(scope=a.scope)
        print(f"{st} scope={a.scope}" + ("" if not bad else " " + "; ".join(bad)))
        return 0 if st == "SEALED" else 3
    if a.const:
        print(globals()[a.const])
        return 0
    if a.check_config:
        bad = check_config(a.check_config)
        print("CONFIG_OK" if not bad else "\n".join("ABORT_REG_DRIFT " + b for b in bad))
        return 0 if not bad else 3
    if a.check_substrate:
        bad = check_substrate(*a.check_substrate)
        print("B0_OK" if not bad else "\n".join(bad))
        return 0 if not bad else 3
    if a.engine_check:
        bad = engine_check(a.engine_check)
        print("ENGINE_TREE_OK" if not bad else "\n".join(bad))
        return 0 if not bad else 3
    if a.external_digests:
        bad = external_digest_problems()
        print("EXTERNAL_DIGESTS_OK" if not bad else "\n".join(bad))
        return 0 if not bad else 3
    reg = load_registered()
    if a.units:
        us = units(a.units, reg)
        bad = plan_problems(a.units, us, reg)
        if bad:
            print("ABORT_TRACE " + "; ".join(bad))
            return 3
        for u in us:
            print(f"{u['name']} kind={u['kind']} D={u['D']} sticky={int(u['sticky'])} pe={int(u['phase_events'])} "
                  f"cells={len(u['cells'])} greedy={int(u['greedy'])}")
        return 0
    if a.emit_plan:
        out, job = a.emit_plan
        us = units(job, reg)
        bad = plan_problems(job, us, reg)
        if bad:
            print("ABORT_TRACE " + "; ".join(bad))
            return 3
        with open(out, "x") as fh:
            json.dump(dict(job=job, units=us, budget=budget(job, reg), registered_sha256=REGISTERED_JSON_SHA256,
                           rule_seal_digest=RULE_SEAL_DIGEST), fh, indent=1, sort_keys=True)
        return 0
    if a.budget:
        print(json.dumps(budget(a.budget, reg), indent=1, sort_keys=True))
        return 0
    if a.need:
        print(int(math.ceil(need_s(a.need[0], reg, a.need[1]))))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
