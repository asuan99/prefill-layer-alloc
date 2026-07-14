"""Shared helpers for engine-port paper figures (figures_v3).
모든 값은 workspace/engine-port/results/ 아래 실측 결과 파일에서 직접 읽는다.
팔레트: reports/sm_policy_report.html CSS 변수 + 기존 characterization attn/ssm 색과 일치."""
import os, re, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))          # .../workspace
EP = os.path.join(WS, "engine-port", "results")                       # engine-port results
OUT = os.path.abspath(os.path.join(WS, "..", "reports", "figures_v3"))  # prefill-layer-alloc/reports/figures_v3
os.makedirs(OUT, exist_ok=True)

# ---- palette (measured-figure consistency) ----
ATT   = "#e8710a"   # attn  (matches characterization attn_ssm_diff)
SSM   = "#1a73e8"   # mamba/ssm
AGN   = "#158b7f"   # agnostic          (html --agn)
LA    = "#c6790f"   # layer-aware       (html --la)
FUSED = "#c0503a"   # fused             (html --fused)
TUNED = "#5b3f8a"   # tuned-uniform

# 19-col standard rows schema (no header in _rows_*.csv)
IX = dict(label=1, in_len=5, out_len=6, rate=7, ttft50=11, tpot50=13, good=15, gp=16)


def read_rows(path):
    """_rows_*.csv (headerless 19-col) -> list[dict]. header가 있으면 스킵."""
    out = []
    with open(path) as fh:
        for r in csv.reader(fh):
            if len(r) < 18 or r[0].strip() in ("model", ""):
                continue
            try:
                out.append(dict(rate=float(r[IX["rate"]]), tpot50=float(r[IX["tpot50"]]),
                                ttft50=float(r[IX["ttft50"]]), gp=float(r[IX["gp"]]),
                                in_len=int(float(r[IX["in_len"]])), out_len=int(float(r[IX["out_len"]]))))
            except ValueError:
                continue
    return out


def tpot_at(path, rate):
    """단일 rate의 tpot_p50(ms) measured 값."""
    for row in read_rows(path):
        if abs(row["rate"] - rate) < 1e-6:
            return row["tpot50"]
    raise KeyError(f"rate={rate} not in {path}")


_KNEE = re.compile(r"sm=(\S+).*?per-attn\((\d+)\)=([\d.]+)\s+per-mamba\((\d+)\)=([\d.]+)")


def read_knee(path):
    """knee_result_*.txt -> dict(sm=[...108..8], attn=[...], mamba=[...], n_attn, n_mamba).
    sm=full -> 108."""
    sm, attn, mamba, na, nm = [], [], [], None, None
    for line in open(path):
        m = _KNEE.search(line)
        if not m:
            continue
        s = 108 if m.group(1) == "full" else int(m.group(1))
        sm.append(s); na = int(m.group(2)); attn.append(float(m.group(3)))
        nm = int(m.group(4)); mamba.append(float(m.group(5)))
    order = sorted(range(len(sm)), key=lambda i: -sm[i])
    return dict(sm=[sm[i] for i in order], attn=[attn[i] for i in order],
                mamba=[mamba[i] for i in order], n_attn=na, n_mamba=nm)


def save(fig, name, outdir=OUT):
    """PNG(150dpi) 렌더 후 PIL로 PDF 변환 (기존 reports/figures 관례). outdir로 대상 변경 가능."""
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{name}.png")
    pdf = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    Image.open(png).convert("RGB").save(pdf, "PDF", resolution=150.0)
    print(f"wrote {pdf}  (+{name}.png)")


OUT_V4 = os.path.abspath(os.path.join(WS, "..", "reports", "figures_v4"))
