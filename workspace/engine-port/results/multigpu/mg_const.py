#!/usr/bin/env python3
"""Registered constants of PREREG_MULTIGPU_2026-10-12.md (rev6) -- ONE place, section-numbered.

Every value here is copied from the PREREG text; none is tuned.  Values the PREREG does NOT
register but the harness needs are in the IMPLEMENTATION block at the bottom and are labelled
as such (they are not rules; changing one is a harness change, not a pre-registration change).
Importers: mg_plan.py, mg_label.py, mg_drive.py, scripts/multigpu/mg_*.{py,sh}.
Shell access: `python3 mg_const.py --get KEY` prints one value (lists space-separated).
"""

import json
import sys

PREREG = "PREREG_MULTIGPU_2026-10-12.md"
PREREG_REV = 6

# ---- §1 fixed conditions ---------------------------------------------------------------------
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
MODEL_REVISION = "dc0661c829b14e5b9246c05cfa89094a0875e052"            # §1 (ABORT_MODEL_REVISION)
SHAPE_IO = {"A": (256, 512), "B": (8192, 64)}                            # §1 (lambda0 SHAPE_IO)
ATTENTION_BACKEND = "flashinfer"                                         # §1 / H4
CAP_PER_GPU = 48                                                         # §1 (TP: 48*N, DP: 48/replica)
LOAD_BALANCE_METHOD = "round_robin"                                      # §1
CONNECTOR_LIMIT = 0                                                      # §1 (unlimited)
SOCK_READ_S = 600                                                        # §1 / H5 (trace_loadgen.py hard-codes it)
FD_SOFT_MIN = 8192                                                       # §1 / H6
DECODE_LOG_INTERVAL = 40                                                 # §1
SGLANG_LOG_MS = "1"                                                      # §1
HEALTH_WAIT_S = 1800                                                     # §3 G0(i) / H2
COMM_ENV_UNSET = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS",
                  "CUDA_DEVICE_MAX_CONNECTIONS", "SGLANG_CUSTOM_ALLREDUCE_ALGO")   # §1 (record only)
JIT_ALL_REDUCE_ENV = "SGLANG_USE_JIT_ALL_REDUCE"                         # §1: must be unset -> ABORT_COMM_ENV
GPU_SUBSETS = {"N1": "0", "TP2": "0,1", "DP2": "0,1", "TP4": "0,1,2,3", "DP4": "0,1,2,3",
               "TP8": "0,1,2,3,4,5,6,7", "DP8": "0,1,2,3,4,5,6,7"}       # §1 (sequential, never concurrent)
NUMA_SUBSET_TP4 = "0,1,4,5"                                              # §4-6
EXPECTED_GPUS = 8                                                        # §1 substrate gate

# ---- §2 arms ---------------------------------------------------------------------------------
ARMS = ("fused", "fused_sym", "agnostic", "d44")
PDMUX_ARMS = ("agnostic", "d44")
# the two scheduler knobs shared BYTE-IDENTICALLY by fused_sym and both PD-mux arms (§2, H3)
SCHED_KNOBS = ("--chunked-prefill-size", "-1", "--disable-overlap-schedule")
D44_FIXED_DSM = 44                                                       # §2
AGNOSTIC_CONFIG_REL = "benchmarks/configs/pdmux_agnostic_manual.yml"     # §2 (engine-porter's file)
D44_CONFIG_REL = "benchmarks/configs/pdmux_r2.yml"                       # §2
AUTO_GRID_CONFIG_REL = "triage/pdmux_a100_smoke.yml"                     # §3 G0(iii) (no manual_divisions)

# ---- §3 M2g ----------------------------------------------------------------------------------
G0_REFUSED_ENVS = (("PDMUX_R2_POLICY", "generic"), ("PDMUX_R2_POLICY", "hybrid"),
                   ("PDMUX_SLO_SCHED", "1"), ("PDMUX_LA_COORD", "1"), ("PDMUX_TRUE_DUAL_WORKER", "1"))
G3B_OFFER_MULT = 1.5                                                     # §3 G3b: 1.5 x (N x 2.99)
G3B_DURATION_S = 600                                                     # §3 G3b
G3B_ARMS = ("agnostic", "d44")
G1_N_PROMPTS = 64                                                        # §3 G1 (informational)
G4_NSYS_TPS = (2, 8)                                                     # §3 G4 (x d44, 30 s)
G4_NSYS_S = 30
MEMORY_REL_TOL = 0.01                                                    # §3 G5(c) (iii)
NVML_EMPTY_GPU_MIB = 1024                                                # §3 G5(c) (가): < 1 GiB
M2G_PASS_TABLE = "M2G_PASS_TABLE.json"                                   # §3-1

# ---- §4 M3 -----------------------------------------------------------------------------------
PRIOR_PER_GPU = {"A": 2.99, "B": 0.690}                                  # §4-2 (lambda*_VESSL)
HI_MULT = 1.6                                                            # §4-2
LO_MULT = 0.60                                                           # §4-2
WINDOW_S = 170                                                           # §4-2
N_REQ_CLAMP = (40, 4000)                                                 # §4-2
HI_SEEDS = (4386, 4162, 251, 2630)                                       # §4-2
LO_SEEDS = (4386, 4162)                                                  # §4-2
EXT_MULT = 2                                                             # §4-2 (once, joint)
ARM_ORDER_SEED = 20261012                                                # §4-1
CONFIGS = ("N1", "TP2", "TP4", "TP8", "DP2", "DP4", "DP8")               # §4-1 (7 configs)
# §4-1 run order after the GPU-deviation block and before NUMA/nsys/N1 round 2
CONFIG_ORDER = ("N1", "TP8", "DP8", "TP2", "DP2", "TP4", "DP4")
G_HI = 0.10                                                              # §4-3 (P-D3)
G_LO = 0.05                                                              # §4-3 (P-D4)
MIN_WINDOW_REQS = 20                                                     # §4-3 SATURATION_UNRESOLVED
CLIENT_ERROR_FRAC = 0.01                                                 # §4-3
LAG_P99_MS = 5.0                                                         # §4-3 HARNESS_LIMITED
THROTTLE_FRAC = 0.01                                                     # §4-3 (nr_throttled / nr_periods increase)
FRONTEND_CORE_FRAC = 0.90                                                # §4-3 FRONTEND_SATURATED (per 1 s delta)
FRONTEND_SECOND_FRAC = 0.20                                              # §4-3
PARTITION_ETIME_MIN = 0.5                                                # §4-3 PARTITION_NOT_EXERCISED
BOOTSTRAP_REPS = 10000                                                   # §4-4
BOOTSTRAP_SEED = 1                                                       # §4-4
CI_PCT = (2.5, 97.5)                                                     # §4-4 (95 % percentile)
TREND_MIN_POINTS = 3                                                     # §4-4
MAGNITUDE_BAND = 0.03                                                    # §4-4 (|G(Nmax)/G(1)-1| < 3 %)
ANCHOR_DRIFT_TOL = 0.03                                                  # §4-5
GPU_DEV_TOL = 0.03                                                       # §4-6
GPU_DEV_SEED = 4386                                                      # §4-6

# ---- §6 budget / abort -----------------------------------------------------------------------
HARD_CAP_S = {"m2g": 8 * 3600, "m3": 30 * 3600}                          # §6-2
ABORT_CODES = ("ABORT_SUBSTRATE", "ABORT_MODEL_REVISION", "ABORT_ENGINE_DIGEST", "ABORT_COMM_ENV",
               "ABORT_M2G_NOT_PASSED")                                   # §6-3
# §6-1 audit forecast (rev5 delta audit §3 "추가 예보"), used only to print the comparison
AUDIT_FORECAST_H = {"lambda_eq_prior": (28.0, 31.0), "tp_lambda_below_prior": (31.0, 35.0)}

# ---- IMPLEMENTATION block (NOT registered; harness choices, listed in the hand-off report) ----
SERVER_SEED = 1                          # N2/N3 convention (n3_plan.SERVER_SEED); PREREG says only "server seed fixed"
MEM_FRACTION = 0.82                      # runner default, unchanged
CONTEXT_LENGTH = 16384                   # runner default, unchanged
CLK_TCK = 100                            # /proc stat tick rate (os.sysconf at runtime is preferred)
PROC_SAMPLE_PERIOD_S = 1.0               # §4-3 "1 s 구간 델타"
WARMUP_REQS_PER_GPU = 8                  # warm-up phase (N3 used 8 sequential requests); not registered
# budget model (estimate only): §6-1 / rev5 audit "boot 4 min, n/lambda, drain"
EST_BOOT_S = 240.0
EST_DRAIN_S = {"A": 25.0, "B": 8.0}
EST_WARMUP_S = {"A": 12.0, "B": 6.0}
EST_CELL_GAP_S = 10.0                    # loadgen start-up + sampler flush between cells


def get(key):
    return globals()[key]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--get":
        v = get(sys.argv[2])
        print(" ".join(map(str, v)) if isinstance(v, (list, tuple)) else (json.dumps(v) if isinstance(v, dict) else v))
    else:
        print(__doc__)
