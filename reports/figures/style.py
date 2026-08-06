#!/usr/bin/env python3
"""Chart styling: the reference data-viz palette, both modes.

The categorical slots are used in fixed order and never cycled. Slots 1-4 were
validated with the skill's own validator on the ADJACENT pairlist (the correct
list for bars and lines) and pass every hard gate in both modes:

    light  worst adjacent CVD dE 9.1 (protan) / normal-vision 22.9
    dark   worst adjacent CVD dE 8.4 (protan) / normal-vision 19.8

In light mode two slots (aqua 2.74:1, yellow 2.11:1) sit below 3:1 against the
surface, so the relief rule applies -- every multi-series chart here carries
direct labels, and the dashboard ships a table view. Under the ALL-PAIRS list
four slots FAIL (yellow vs orange, normal-vision 13.7), so no chart form that
needs all-pairs separation uses more than three colour-carrying series.
"""
from __future__ import annotations

import matplotlib as mpl

LIGHT = dict(
    surface="#fcfcfb", plane="#f9f9f7",
    ink="#0b0b0b", secondary="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
    band="#eceae3",
)

DARK = dict(
    surface="#1a1a19", plane="#0d0d0d",
    ink="#ffffff", secondary="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835",
    series=["#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767"],
    good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
    band="#232322",
)

MODES = {"light": LIGHT, "dark": DARK}

# Verdict -> status colour. Status colours are reserved and always ship with a
# label or a distinct marker, never hue alone.
VERDICT_COLOR = {
    "dead": "critical", "reopened": "warning", "reversed": "serious", "open": "good",
}
VERDICT_MARKER = {"dead": "X", "reopened": "o", "reversed": "^", "open": "D"}

GRADE_COLOR = {
    "serving": "good", "scoped": "good",
    "micro": "warning", "unaudited": "warning", "partial": "warning",
    "strong": "good", "strong/partial": "warning", "confirmed": "good",
    "retracted": "critical", "barred": "critical", "refuted": "critical",
    "unverified": "serious", "unconfirmed": "serious",
}


def apply(mode: str) -> dict:
    """Install the mode's rcParams and hand back its colour dict."""
    c = MODES[mode]
    mpl.rcParams.update({
        "figure.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "savefig.edgecolor": c["surface"],
        "text.color": c["ink"],
        "axes.labelcolor": c["secondary"],
        "axes.edgecolor": c["axis"],
        "axes.titlecolor": c["ink"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelcolor": c["secondary"],
        "ytick.labelcolor": c["secondary"],
        "grid.color": c["grid"],
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.constrained_layout.use": False,
    })
    return c


def title(ax, main: str, sub: str | None, c: dict, pad: float = 14.0,
          subwidth: int = 96):
    """Title in primary ink; the scope/grade line under it in secondary ink.

    `subwidth` wraps the sub line so a long scope note cannot run off the figure
    edge -- panel titles are drawn from the axes' left edge and are not clipped
    to it, so an unwrapped sub line silently overflows the canvas.
    """
    import textwrap
    lines = textwrap.wrap(sub, subwidth) if sub else []
    # the title pad has to reserve room for EVERY wrapped sub line, otherwise the
    # sub block grows upward through the title
    ax.set_title(main, color=c["ink"], loc="left", pad=pad + 10.5 * len(lines))
    if lines:
        ax.annotate("\n".join(lines), xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, pad - 5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=c["secondary"],
                    linespacing=1.4)


def footer(fig, text: str, c: dict, y: float = 0.012, width: int = 205):
    import textwrap
    fig.text(0.008, y, "\n".join(textwrap.wrap(text, width)), fontsize=6.6,
             color=c["muted"], ha="left", va="bottom", linespacing=1.45)


def no_minor_labels(ax, which: str = "both"):
    """Log axes otherwise sprout '4 x 10^1' style minor labels over the data."""
    from matplotlib.ticker import NullFormatter
    for axis in ((ax.xaxis, ax.yaxis) if which == "both"
                 else (ax.xaxis,) if which == "x" else (ax.yaxis,)):
        axis.set_minor_formatter(NullFormatter())


def grade_chip(ax, text: str, c: dict, kind: str = "warning",
               xy=(1.0, 1.0), ha="right"):
    """A small state chip. Always icon-or-text plus colour, never colour alone."""
    ax.annotate(text, xy=xy, xycoords="axes fraction", xytext=(0, 6),
                textcoords="offset points", ha=ha, va="bottom",
                fontsize=7.5, color=c[kind], weight="bold")
