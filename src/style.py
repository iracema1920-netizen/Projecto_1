"""
Project style - one place for every colour and chart setting.
===============================================================
Import it at the top of each script:

    from style import apply_style, save, BLUE, CRITICAL ...
    apply_style()

Palette: a modern, colour-blind-checked categorical set (keep the ORDER - it is
what keeps neighbouring colours distinguishable), plus reserved status colours
for maintenance states. Status colours are never reused for ordinary series.
"""
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# --- Categorical palette (series colours, use in this order) ---------------
BLUE    = "#2A78D6"
ORANGE  = "#EB6834"
AQUA    = "#1BAF7A"
YELLOW  = "#EDA100"
MAGENTA = "#E87BA4"
GREEN   = "#008300"
VIOLET  = "#6250D6"
RED     = "#E34948"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# --- Status colours (maintenance state - reserved meaning) -----------------
OK       = "#0CA30C"   # healthy / normal
WARNING  = "#FAB219"   # watch / degradation period
SERIOUS  = "#EC835A"   # alarm limit H
CRITICAL = "#D03B3B"   # failure / trip / limit HH

# --- Neutrals (text, grid, background) -------------------------------------
INK       = "#1F1F1E"   # titles and main text
TEXT      = "#52514E"   # axis labels and tick labels
MUTED     = "#898781"   # secondary annotations, reference lines
LIGHT     = "#D6D5CE"   # "background" bars (e.g. maximum possible value)
GRID      = "#E9E8E2"
BASELINE  = "#C3C2B7"
SURFACE   = "#FCFCFB"

# Diverging colour map for correlations: blue (-1) -> grey (0) -> red (+1)
DIVERGING = LinearSegmentedColormap.from_list(
    "ogp_diverging", ["#184F95", BLUE, "#9EC5F4", "#F0EFEC", "#F4B3B2", RED, "#A12B2B"])

# Excel header colour (hex without '#', used by openpyxl)
EXCEL_HEADER = BLUE.lstrip("#")


def apply_style() -> None:
    """Apply the project look to every matplotlib figure."""
    plt.rcParams.update({
        # colours
        "axes.prop_cycle": cycler(color=PALETTE),
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": INK,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        # frame and grid
        "axes.edgecolor": BASELINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        # lines and markers
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        # output
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, path) -> None:
    """Save a figure and free its memory."""
    fig.savefig(path)
    plt.close(fig)