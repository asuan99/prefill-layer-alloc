"""figures_v4 (decode-side 발표 덱) 공용: e8_engine_figs._common 재활용 + OUT=figures_v4."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e8_engine_figs"))
from _common import (EP, ATT, SSM, AGN, LA, FUSED, TUNED, OUT_V4,  # noqa: F401
                     read_rows, tpot_at, read_knee, save as _save)


def save(fig, name):
    _save(fig, name, outdir=OUT_V4)
