"""One look for every figure: sans text matched to the paper's caption size, thin marks, recessive axes.

Colours (validated with the dataviz palette checker on white: CVD ΔE 24.7, contrast >= 3:1):
the encoder family is categorical slot 1 (blue), the decoder family slot 2 (orange).
"""
import matplotlib as mpl

ENCODER = "#2a78d6"
DECODER = "#eb6834"
FAMILY = {"encoder": ENCODER, "decoder": DECODER}
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text, ticks
RULE = "#d9d8d4"         # grid and zero lines

COLUMN = 3.25            # inches: one column of the 6.5-inch text block (two figures side by side)
FULL = 6.5               # inches: the full text width


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Libertinus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
        "axes.edgecolor": RULE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 3, "ytick.major.pad": 4,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType: selectable text, accepted by arXiv
    })
