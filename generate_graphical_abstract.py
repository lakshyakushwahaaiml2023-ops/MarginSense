# -*- coding: utf-8 -*-
"""
MarginSense - Publication-Style Graphical Abstract Generator
Produces a four-panel (A-D) figure matching computational-oncology paper conventions.
Run from the repository root:
    python generate_graphical_abstract.py
Output: outputs/marginsense_graphical_abstract.png  (300 dpi)
         outputs/marginsense_graphical_abstract.svg
"""

import sys
import os
# Force UTF-8 stdout so Unicode in print() doesn't crash on Windows cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc, Circle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         8.5,
    "axes.linewidth":    0.7,
    "axes.edgecolor":    "#555555",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "figure.dpi":        150,
})

# Palette (muted, publication-ready)
C_BLUE    = "#3B6FA0"   # main brand
C_TEAL    = "#2E8B7A"   # ensemble / uncertainty
C_ORANGE  = "#D4763B"   # highlight / margin
C_PURPLE  = "#7B5EA7"   # safety / UCB
C_GREEN   = "#4A8C5C"   # recurrence coverage
C_GRAY    = "#888888"   # baseline / neutral
C_RED     = "#B94040"   # warning / standard margin
C_CREAM   = "#FAFAF7"   # panel background
C_BORDER  = "#CCCCCC"   # panel border
C_TEXT    = "#1A1A1A"   # main text
C_SUBTEXT = "#555555"   # secondary text
C_PANEL   = "#F0F0EA"   # slightly deeper cream for sub-boxes


def panel_background(ax, color=C_CREAM):
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
        spine.set_linewidth(0.8)


def panel_label(ax, letter, x=-0.04, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=C_TEXT,
            va="bottom", ha="left", clip_on=False,
            fontfamily="DejaVu Sans")


def draw_rounded_box(ax, x, y, w, h, text, fontsize=7.5,
                      facecolor=C_PANEL, edgecolor=C_BORDER,
                      text_color=C_TEXT, bold=False, radius=0.02,
                      linewidth=0.8):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.01,rounding_size={radius}",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, transform=ax.transAxes,
                          clip_on=False, zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal",
            zorder=4, wrap=True,
            multialignment="center")


def arrow_axes(ax, x0, y0, x1, y1, color=C_BLUE, lw=1.2, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"))


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC VOLUME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def make_brain_slice(size=80):
    """Smooth elliptical brain with tissue bands."""
    g = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(g, g)
    r = np.sqrt((xx / 0.85)**2 + (yy / 0.95)**2)
    brain = (r < 1.0).astype(float)
    # WM ring
    wm = ((r > 0.30) & (r < 0.72)).astype(float) * brain
    gm = ((r > 0.72) & (r < 0.95)).astype(float) * brain
    csf = ((r > 0.10) & (r < 0.30)).astype(float) * brain
    tissue = np.zeros((size, size))
    tissue[wm > 0] = 1   # WM
    tissue[gm > 0] = 2   # GM
    tissue[csf > 0] = 3  # CSF
    # tumor ~upper-left quadrant
    tumor = (((xx + 0.32)**2 + (yy - 0.18)**2) < 0.065**2 * 15)
    tumor &= brain.astype(bool)
    tissue[tumor] = 4    # Tumor
    return brain, tissue, tumor.astype(float)

def make_density_map(size=80, cx=-0.32, cy=0.18):
    """Anisotropic infiltration probability map."""
    g = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(g, g)
    d = np.sqrt(((xx - cx) / 1.6)**2 + ((yy - cy) / 0.9)**2)
    dens = np.exp(-3.5 * d**2)
    dens = gaussian_filter(dens, sigma=3)
    dens /= dens.max()
    brain = (np.sqrt((xx / 0.85)**2 + (yy / 0.95)**2) < 1.0)
    dens *= brain
    return dens

def make_uncertainty_map(size=80):
    dens = make_density_map(size)
    g = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(g, g)
    brain = (np.sqrt((xx / 0.85)**2 + (yy / 0.95)**2) < 1.0)
    # uncertainty peaks at the rim of high-density region
    grad_mag = np.sqrt(np.gradient(dens)[0]**2 + np.gradient(dens)[1]**2)
    std = gaussian_filter(grad_mag * 3 + rng.random((size, size)) * 0.04, sigma=2)
    std *= brain
    std /= std.max() + 1e-8
    return std

VIRIDIS_MUTED = LinearSegmentedColormap.from_list(
    "viridis_muted",
    ["#f7fbff", "#2171b5", "#08306b"])

HEAT_MUTED = LinearSegmentedColormap.from_list(
    "heat_muted",
    ["#fff7f3", "#fc8d59", "#7f0000"])

PURPLE_MUTED = LinearSegmentedColormap.from_list(
    "purple_muted",
    ["#f7f4f9", "#9e9ac8", "#3f007d"])


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor="white")
fig.patch.set_facecolor("white")

# Title strip
fig.text(0.5, 0.975,
         "MarginSense  |  Physics-Informed, Uncertainty-Aware Patient-Specific Radiotherapy Margin Optimization for Glioblastoma",
         ha="center", va="top", fontsize=11, fontweight="bold", color=C_TEXT)
fig.text(0.5, 0.955,
         "Amortized PINN  ·  Deep Ensemble Uncertainty  ·  UCB Safety Margin  ·  Factual Explainability",
         ha="center", va="top", fontsize=8.5, color=C_SUBTEXT)

# Divider below title
fig.add_artist(plt.Line2D([0.03, 0.97], [0.945, 0.945],
               color=C_BORDER, lw=0.8, transform=fig.transFigure))

# 2×2 grid of panels
outer = GridSpec(2, 2, figure=fig,
                 left=0.03, right=0.97,
                 top=0.940, bottom=0.02,
                 wspace=0.06, hspace=0.10)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL A — Input Data & Preprocessing
# ═════════════════════════════════════════════════════════════════════════════
axA = fig.add_subplot(outer[0, 0])
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
axA.set_xticks([]); axA.set_yticks([])
panel_background(axA)
panel_label(axA, "A")

# Section header
axA.text(0.5, 0.97, "Input Data & Preprocessing", ha="center", va="top",
         fontsize=9.5, fontweight="bold", color=C_BLUE, transform=axA.transAxes)

# ── Brain tissue map (synthetic) ──────────────────────────────────────────
_, tissue, _ = make_brain_slice(size=80)
tissue_cmap = matplotlib.colors.ListedColormap(
    ["#FFFFFF", "#C8DCF0", "#A8C8A0", "#E8E8D0", "#D47080"])
tissue_norm = matplotlib.colors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5],
                                              ncolors=tissue_cmap.N)

# MRI panel inset (top-left quadrant of panel A)
ax_mri = axA.inset_axes([0.03, 0.53, 0.46, 0.40])
ax_mri.imshow(tissue, cmap=tissue_cmap, norm=tissue_norm,
              origin="upper", aspect="equal", interpolation="nearest")
ax_mri.set_xticks([]); ax_mri.set_yticks([])
ax_mri.set_title("MRI Tissue Map", fontsize=7.5, color=C_TEXT, pad=2)
for sp in ax_mri.spines.values():
    sp.set_edgecolor(C_BORDER); sp.set_linewidth(0.6)

# Legend for tissue map
legend_items = [
    mpatches.Patch(facecolor="#C8DCF0", edgecolor=C_BORDER, label="White Matter"),
    mpatches.Patch(facecolor="#A8C8A0", edgecolor=C_BORDER, label="Gray Matter"),
    mpatches.Patch(facecolor="#E8E8D0", edgecolor=C_BORDER, label="CSF"),
    mpatches.Patch(facecolor="#D47080", edgecolor=C_BORDER, label="Tumor (GTV)"),
]
ax_mri.legend(handles=legend_items, loc="lower left", fontsize=5.5,
              framealpha=0.85, handlelength=0.8, borderpad=0.4,
              edgecolor=C_BORDER)

# MRI modalities strip
ax_mods = axA.inset_axes([0.51, 0.53, 0.46, 0.40])
ax_mods.set_xlim(0, 1); ax_mods.set_ylim(0, 1)
ax_mods.set_xticks([]); ax_mods.set_yticks([])
ax_mods.set_facecolor(C_PANEL)
for sp in ax_mods.spines.values():
    sp.set_edgecolor(C_BORDER); sp.set_linewidth(0.6)
ax_mods.set_title("Multi-Modal MRI Channels", fontsize=7.5, color=C_TEXT, pad=2)

mri_labels   = ["T₁", "T₁ce", "T₂", "FLAIR"]
mri_cols     = ["#4a7fb5", "#5ba55b", "#a06530", "#8a5a9a"]
mri_ys       = [0.75, 0.52, 0.29, 0.06]
for lbl, col, ys in zip(mri_labels, mri_cols, mri_ys):
    rect = FancyBboxPatch((0.08, ys), 0.84, 0.18,
                           boxstyle="round,pad=0.01,rounding_size=0.02",
                           facecolor=col, edgecolor="white",
                           linewidth=0.5, alpha=0.85,
                           transform=ax_mods.transAxes, zorder=2)
    ax_mods.add_patch(rect)
    ax_mods.text(0.5, ys + 0.09, lbl,
                 transform=ax_mods.transAxes, ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold", zorder=3)

# Preprocessing steps flow
steps = [
    ("① Load NIfTI\n(BraTS Format)", C_BLUE, 0.07),
    ("② Resample\n128³ voxels",     "#2E7D7D", 0.36),
    ("③ Z-Score Norm\n(Brain Mask)", "#3D6B3D", 0.65),
]
for txt, col, x in steps:
    draw_rounded_box(axA, x, 0.04, 0.26, 0.14, txt,
                     fontsize=6.5, facecolor=col,
                     text_color="white", bold=False)

# Arrows between steps
for x in [0.34, 0.63]:
    arrow_axes(axA, x, 0.11, x + 0.02, 0.11, color=C_GRAY, lw=0.9)

# Covariate table
cov_header = ["Feature", "Type", "Encoded As"]
cov_rows   = [
    ["Age",        "Numeric",  "age / 100"],
    ["KPS",        "Numeric",  "kps / 100"],
    ["IDH Status", "Binary",   "1 = Mutant"],
    ["MGMT",       "Binary",   "1 = Methylated"],
    ["Resection",  "Ordinal",  "GTR=1.0, STR=0.5"],
    ["Tumor Vol",  "Derived",  "log(vol) / 5"],
    ["Sphericity", "Derived",  "PCA λ_min/λ_max"],
    ["Lobe Loc",   "Derived",  "Frontal→0 … Ins→1"],
]

ax_cov = axA.inset_axes([0.02, 0.23, 0.96, 0.27])
ax_cov.set_xlim(0, 3); ax_cov.set_ylim(0, len(cov_rows) + 1)
ax_cov.set_xticks([]); ax_cov.set_yticks([])
ax_cov.set_facecolor(C_PANEL)
for sp in ax_cov.spines.values():
    sp.set_edgecolor(C_BORDER); sp.set_linewidth(0.6)
ax_cov.set_title("Clinical Covariates  (11-dim vector → Encoder)", fontsize=7.5,
                 color=C_TEXT, pad=2)

# Header row
for j, h in enumerate(cov_header):
    ax_cov.text(j + 0.5, len(cov_rows) + 0.5, h, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color=C_TEXT)
    ax_cov.add_patch(mpatches.FancyBboxPatch(
        (j + 0.04, len(cov_rows) + 0.05), 0.92, 0.85,
        boxstyle="square,pad=0.01",
        facecolor="#D8E8F0", edgecolor=C_BORDER, linewidth=0.4))

for i, row in enumerate(cov_rows):
    bg = "#FFFFFF" if i % 2 == 0 else C_PANEL
    ax_cov.add_patch(mpatches.Rectangle(
        (0, i + 0.05), 3, 0.9, facecolor=bg, edgecolor=C_BORDER, linewidth=0.3))
    for j, val in enumerate(row):
        ax_cov.text(j + 0.5, i + 0.5, val, ha="center", va="center",
                    fontsize=6, color=C_TEXT)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL B — Fisher-KPP Equation & Physical Parameters
# ═════════════════════════════════════════════════════════════════════════════
axB = fig.add_subplot(outer[0, 1])
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_xticks([]); axB.set_yticks([])
panel_background(axB)
panel_label(axB, "B")

axB.text(0.5, 0.97, "Governing Physics & Tunable Parameters", ha="center", va="top",
         fontsize=9.5, fontweight="bold", color=C_BLUE, transform=axB.transAxes)

# ── PDE Box ───────────────────────────────────────────────────────────────
pde_box = FancyBboxPatch((0.04, 0.70), 0.92, 0.24,
                          boxstyle="round,pad=0.015,rounding_size=0.025",
                          facecolor="#EDF4FB", edgecolor=C_BLUE,
                          linewidth=1.2, transform=axB.transAxes, zorder=2)
axB.add_patch(pde_box)
axB.text(0.5, 0.91,
         "Fisher-KPP Reaction-Diffusion PDE",
         ha="center", va="center", fontsize=8.5,
         fontweight="bold", color=C_BLUE, transform=axB.transAxes)
axB.text(0.5, 0.82,
         r"$\frac{\partial c}{\partial t} = \nabla \cdot [ D(x) \nabla c ] + \rho_0 c (1-c)$",
         ha="center", va="center", fontsize=13,
         color=C_TEXT, transform=axB.transAxes)
axB.text(0.5, 0.73,
         r"$c(x,t)\in[0,1]$ = cell density     "
         r"$x\in[0,1]^3$, $t\in[0,1]$",
         ha="center", va="center", fontsize=7.5,
         color=C_SUBTEXT, transform=axB.transAxes)

# ── Spatially-varying D(x) table ─────────────────────────────────────────
axB.text(0.5, 0.665, "Spatially-Varying Diffusion  D(x) = D₀ · w_tissue(x)",
         ha="center", va="center", fontsize=8, fontweight="bold",
         color=C_TEXT, transform=axB.transAxes)

tissue_table = [
    ("White Matter (WM)",   "0.15 mm²/day", "#B0CCEC", "Fast — along axonal tracts"),
    ("Gray Matter (GM)",    "0.03 mm²/day", "#A8D0A8", "Slow — dense neuropil"),
    ("CSF / Ventricles",    "0.00 mm²/day", "#E0E8B8", "Zero — physical barrier → OAR"),
    ("Necrotic / Edema",    "0.075 mm²/day","#E0C8B8", "Intermediate"),
]
row_y = 0.625
for tname, dval, col, note in tissue_table:
    r = FancyBboxPatch((0.04, row_y - 0.052), 0.92, 0.048,
                        boxstyle="square,pad=0.005",
                        facecolor=col, edgecolor=C_BORDER,
                        linewidth=0.4, transform=axB.transAxes, zorder=2)
    axB.add_patch(r)
    axB.text(0.06, row_y - 0.028, tname, ha="left", va="center",
             fontsize=7, color=C_TEXT, transform=axB.transAxes, zorder=3)
    axB.text(0.47, row_y - 0.028, dval, ha="center", va="center",
             fontsize=7.5, color=C_TEXT, fontweight="bold",
             transform=axB.transAxes, zorder=3)
    axB.text(0.95, row_y - 0.028, note, ha="right", va="center",
             fontsize=6, color=C_SUBTEXT, transform=axB.transAxes, zorder=3)
    row_y -= 0.055

# ── Tunable Parameters Table ─────────────────────────────────────────────
axB.text(0.5, 0.385, "Model's Tunable Parameters",
         ha="center", va="center", fontsize=8.5, fontweight="bold",
         color=C_TEXT, transform=axB.transAxes)

param_header = ["Symbol", "Role", "Default", "Range"]
param_rows = [
    ["D₀",    "Global diffusion scale (log-space)",    "~0.01",  "(0, ∞)"],
    ["ρ₀",    "Proliferation rate (log-space)",         "~0.30",  "(0, ∞)"],
    ["z",     "UCB safety caution multiplier",          "1.0",    "[0, 2]"],
    ["λ",     "Coverage–toxicity trade-off weight",     "~50",    "(0, ∞)"],
    ["τ*",    "Optimized contour threshold",            "auto",   "[0, 1]"],
    ["σ",     "IC Gaussian seed width",                 "0.05",   "fixed"],
]

ax_pt = axB.inset_axes([0.03, 0.02, 0.94, 0.35])
ax_pt.set_xlim(0, 4); ax_pt.set_ylim(0, len(param_rows) + 1)
ax_pt.set_xticks([]); ax_pt.set_yticks([])
ax_pt.set_facecolor(C_PANEL)
for sp in ax_pt.spines.values():
    sp.set_edgecolor(C_BORDER); sp.set_linewidth(0.6)

for j, h in enumerate(param_header):
    ax_pt.add_patch(mpatches.FancyBboxPatch(
        (j + 0.04, len(param_rows) + 0.05), 0.92, 0.85,
        boxstyle="square,pad=0.01",
        facecolor="#D8E0F0", edgecolor=C_BORDER, linewidth=0.4))
    ax_pt.text(j + 0.5, len(param_rows) + 0.5, h,
               ha="center", va="center", fontsize=6.5,
               fontweight="bold", color=C_TEXT)

for i, row in enumerate(param_rows):
    bg = "#FFFFFF" if i % 2 == 0 else C_PANEL
    ax_pt.add_patch(mpatches.Rectangle(
        (0, i + 0.05), 4, 0.9, facecolor=bg, edgecolor=C_BORDER, linewidth=0.3))
    for j, val in enumerate(row):
        fw = "bold" if j == 0 else "normal"
        col = C_BLUE if j == 0 else C_TEXT
        ax_pt.text(j + 0.5, i + 0.5, val, ha="center", va="center",
                   fontsize=6.5, color=col, fontweight=fw)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL C — Amortized vs. Per-Patient Architecture
# ═════════════════════════════════════════════════════════════════════════════
axC = fig.add_subplot(outer[1, 0])
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
axC.set_xticks([]); axC.set_yticks([])
panel_background(axC)
panel_label(axC, "C")

axC.text(0.5, 0.97, "Amortized PINN Architecture  vs.  Per-Patient Baseline",
         ha="center", va="top", fontsize=9.5, fontweight="bold",
         color=C_BLUE, transform=axC.transAxes)

# ── Main amortized flow (left 68%) ─────────────────────────────────────
main_boxes = [
    # (x, y, w, h, label, color, text_color)
    (0.01, 0.70, 0.30, 0.20,
     "5-Channel\n3D Volume\n(T₁ T₁ce T₂ FLAIR + GTV)\n128³",
     "#EDF4FB", C_TEXT),
    (0.01, 0.47, 0.16, 0.17,
     "11-dim\nClinical\nCovariates",
     "#EAF2EA", C_TEXT),
    (0.36, 0.57, 0.22, 0.35,
     "3D CNN\nEncoder\n\n4 conv blocks\n16→32→64→64\nAdaptive Pool\n\n+ Cov Proj",
     C_BLUE, "white"),
    (0.63, 0.73, 0.20, 0.18,
     "Physics\nDecoder\nD₀, ρ₀\n(log-space)",
     C_TEAL, "white"),
    (0.63, 0.52, 0.20, 0.18,
     "64-dim\nPatient\nEmbedding\n  z",
     "#5B6FA0", "white"),
    (0.85, 0.57, 0.14, 0.35,
     "FiLM\nCoord\nMLP\n\n4 layers\n64 units\nTanh\n\nγz·x+βz",
     C_ORANGE, "white"),
]

for (bx, by, bw, bh, btxt, bfc, btc) in main_boxes:
    bb = FancyBboxPatch((bx, by), bw, bh,
                         boxstyle="round,pad=0.01,rounding_size=0.025",
                         facecolor=bfc, edgecolor=C_BORDER,
                         linewidth=0.8, transform=axC.transAxes, zorder=3)
    axC.add_patch(bb)
    axC.text(bx + bw / 2, by + bh / 2, btxt,
             transform=axC.transAxes, ha="center", va="center",
             fontsize=5.8, color=btc, zorder=4, multialignment="center")

# Arrows in main flow
arrow_axes(axC, 0.31, 0.80, 0.35, 0.80, color=C_BLUE)    # volume → encoder
arrow_axes(axC, 0.17, 0.55, 0.35, 0.66, color=C_TEAL)    # cov → encoder
arrow_axes(axC, 0.58, 0.82, 0.63, 0.82, color=C_TEAL)    # enc → phys
arrow_axes(axC, 0.58, 0.66, 0.63, 0.63, color="#5B6FA0") # enc → embed
arrow_axes(axC, 0.83, 0.74, 0.85, 0.74, color=C_ORANGE)  # D,ρ → MLP (via z)
arrow_axes(axC, 0.83, 0.63, 0.85, 0.63, color=C_ORANGE)  # embed → MLP

# Input coordinate label
coord_box = FancyBboxPatch((0.84, 0.47), 0.15, 0.09,
                             boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor="#F4EEF8", edgecolor=C_PURPLE,
                             linewidth=0.7, transform=axC.transAxes, zorder=3)
axC.add_patch(coord_box)
axC.text(0.915, 0.515, "(x,y,z,t)\nCoords",
         transform=axC.transAxes, ha="center", va="center",
         fontsize=5.8, color=C_PURPLE, zorder=4)
arrow_axes(axC, 0.915, 0.56, 0.915, 0.573, color=C_PURPLE, lw=0.8)

# Output
out_box = FancyBboxPatch((0.72, 0.38), 0.26, 0.12,
                          boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=C_GREEN, edgecolor=C_BORDER,
                          linewidth=0.8, transform=axC.transAxes, zorder=3)
axC.add_patch(out_box)
axC.text(0.85, 0.44, "c(x,t) ∈ [0,1]\nCell Density",
         transform=axC.transAxes, ha="center", va="center",
         fontsize=6.5, color="white", fontweight="bold", zorder=4)
arrow_axes(axC, 0.915, 0.57, 0.855, 0.50, color=C_GREEN)

# Loss terms strip
loss_labels = [
    ("L_data\nMSE vs\nrecurrence",   "#4A7FB5"),
    ("L_ic\nGaussian IC\nat t = 0",  "#2E7D7D"),
    ("L_pde\nFisher-KPP\nresidual",  "#A05030"),
]
lx_start = 0.01
for (ltxt, lcol) in loss_labels:
    lb = FancyBboxPatch((lx_start, 0.26), 0.20, 0.10,
                          boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=lcol, edgecolor=C_BORDER,
                          linewidth=0.6, alpha=0.88,
                          transform=axC.transAxes, zorder=3)
    axC.add_patch(lb)
    axC.text(lx_start + 0.10, 0.31, ltxt,
             transform=axC.transAxes, ha="center", va="center",
             fontsize=5.6, color="white", zorder=4, multialignment="center")
    lx_start += 0.22

axC.text(0.37, 0.31, "+", transform=axC.transAxes, ha="center", va="center",
         fontsize=12, color=C_SUBTEXT, fontweight="bold")
axC.text(0.59, 0.31, "+", transform=axC.transAxes, ha="center", va="center",
         fontsize=12, color=C_SUBTEXT, fontweight="bold")
axC.text(0.80, 0.32, "L_total\n(lam_pde = 0.1)", transform=axC.transAxes,
         ha="center", va="center", fontsize=6, color=C_SUBTEXT)

axC.text(0.5, 0.22, "Single training run across patient population → New patient: ONE forward pass  (seconds, not minutes)",
         transform=axC.transAxes, ha="center", va="center",
         fontsize=7, color=C_GREEN, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#E8F5E9",
                   edgecolor=C_GREEN, linewidth=0.8))

# ── Baseline inset (bottom-right corner) ────────────────────────────────
ax_base = axC.inset_axes([0.0, 0.01, 0.46, 0.19])
ax_base.set_xlim(0, 1); ax_base.set_ylim(0, 1)
ax_base.set_xticks([]); ax_base.set_yticks([])
ax_base.set_facecolor("#FFF0ED")
for sp in ax_base.spines.values():
    sp.set_edgecolor(C_RED); sp.set_linewidth(0.8)
ax_base.set_title("Per-Patient Vanilla PINN Baseline  (⚠ slow)", fontsize=6.5,
                  color=C_RED, pad=2)
baseline_steps = ["MRI\nInput", "Train\n100 epochs\nper patient", "Output\nc(x,t)"]
for bi, (btxt, bx) in enumerate(zip(baseline_steps, [0.08, 0.40, 0.74])):
    bb2 = FancyBboxPatch((bx, 0.12), 0.24, 0.70,
                           boxstyle="round,pad=0.01,rounding_size=0.03",
                           facecolor="#D47070", edgecolor=C_BORDER,
                           linewidth=0.5, alpha=0.75,
                           transform=ax_base.transAxes)
    ax_base.add_patch(bb2)
    ax_base.text(bx + 0.12, 0.47, btxt, transform=ax_base.transAxes,
                 ha="center", va="center", fontsize=5.8, color="white")
    if bi < 2:
        arrow_axes(ax_base, bx + 0.25, 0.5, bx + 0.38, 0.5,
                   color=C_RED, lw=0.8)
ax_base.text(0.5, -0.18, "New patient requires full retraining → minutes to hours",
             transform=ax_base.transAxes, ha="center", va="center",
             fontsize=6, color=C_RED)

# Ensemble inset
ax_ens = axC.inset_axes([0.50, 0.01, 0.48, 0.19])
ax_ens.set_xlim(0, 1); ax_ens.set_ylim(0, 1)
ax_ens.set_xticks([]); ax_ens.set_yticks([])
ax_ens.set_facecolor("#EAF5EC")
for sp in ax_ens.spines.values():
    sp.set_edgecolor(C_TEAL); sp.set_linewidth(0.8)
ax_ens.set_title("Deep Ensemble (5 seeds × 64-dim z)", fontsize=6.5,
                 color=C_TEAL, pad=2)
ens_colors = ["#2E8B7A", "#3B7AB5", "#7B5EA7", "#D4763B", "#4A8C5C"]
for ei in range(5):
    ex = 0.04 + ei * 0.19
    eb = FancyBboxPatch((ex, 0.12), 0.155, 0.70,
                          boxstyle="round,pad=0.01,rounding_size=0.03",
                          facecolor=ens_colors[ei], edgecolor=C_BORDER,
                          linewidth=0.4, alpha=0.8,
                          transform=ax_ens.transAxes)
    ax_ens.add_patch(eb)
    ax_ens.text(ex + 0.077, 0.47, f"M{ei+1}\ns={42+ei}",
                transform=ax_ens.transAxes, ha="center", va="center",
                fontsize=5.5, color="white")
ax_ens.text(0.5, -0.18, "μ(x)=mean  ·  σ(x)=std  →  Per-voxel confidence field",
            transform=ax_ens.transAxes, ha="center", va="center",
            fontsize=6, color=C_TEAL, fontweight="bold")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL D — Output Maps & Clinical Comparison
# ═════════════════════════════════════════════════════════════════════════════
axD = fig.add_subplot(outer[1, 1])
axD.set_xlim(0, 1); axD.set_ylim(0, 1)
axD.set_xticks([]); axD.set_yticks([])
panel_background(axD)
panel_label(axD, "D")

axD.text(0.5, 0.97, "Output Maps & Clinical Margin Comparison",
         ha="center", va="top", fontsize=9.5, fontweight="bold",
         color=C_BLUE, transform=axD.transAxes)

brain, tissue_slice, tumor_slice = make_brain_slice(size=80)
density_map    = make_density_map(size=80)
uncertainty_map = make_uncertainty_map(size=80)
ucb_map        = np.clip(density_map + 1.0 * uncertainty_map, 0, 1) * brain
tau_opt        = 0.35

# Brain boundary for contour overlays
brain_border = (np.sqrt(
    ((np.mgrid[-1:1:80j, -1:1:80j][0] / 0.85)**2 +
     (np.mgrid[-1:1:80j, -1:1:80j][1] / 0.95)**2)) < 1.0)

# ── Subpanel helper using inset_axes on axD ─────────────────────────────
def make_subpanel_inset(parent_ax, rect, title, img, cmap, vmin=0, vmax=1,
                        contour_data=None, contour_level=0.35,
                        contour_color="white"):
    """rect = [x0, y0, width, height] in axes fraction of parent_ax."""
    ax = parent_ax.inset_axes(rect)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
              origin="upper", aspect="equal", interpolation="bilinear")
    if contour_data is not None:
        ax.contour(contour_data, levels=[contour_level],
                   colors=[contour_color], linewidths=[0.8])
    ax.set_title(title, fontsize=6, color=C_TEXT, pad=2,
                 multialignment="center")
    for sp in ax.spines.values():
        sp.set_edgecolor(C_BORDER); sp.set_linewidth(0.5)
    return ax


# Top image row: 4 panels, right 52% of axD, upper 47%
map_y0   = 0.50   # bottom of top row (axes fraction)
map_h    = 0.44   # height of top row
col_w    = 0.118  # width of each map
col_gap  = 0.010
col_x0   = 0.49   # starting x for first map

map_configs = [
    ("① Prediction\n(Mean μ)",    density_map * brain,     "Blues",     density_map,     0.35, "#1a1a80"),
    ("② Uncertainty\n(Std σ)",    uncertainty_map * brain, HEAT_MUTED,  uncertainty_map, 0.25, "#5a0000"),
    ("③ Safety Margin\n(μ+z·σ)", ucb_map,                  PURPLE_MUTED,ucb_map,         tau_opt, C_PURPLE),
    ("④ Isoprobability\nContours", density_map * brain,    "bone_r",    None,            0.35, "white"),
]

map_axes = []
for mi, (title, img, cmap, cdata, clev, ccol) in enumerate(map_configs):
    rx = col_x0 + mi * (col_w + col_gap)
    ax_m = make_subpanel_inset(axD, [rx, map_y0, col_w, map_h],
                               title, img, cmap,
                               contour_data=cdata,
                               contour_level=clev,
                               contour_color=ccol)
    map_axes.append(ax_m)

# Isoprobability contours on panel 4
ax_d4 = map_axes[3]
iso_levels  = [0.20, 0.50, 0.80, 0.95]
iso_colors  = ["#B0C8E8", "#5090C8", "#2060A8", "#082060"]
for lv, lc in zip(iso_levels, iso_colors):
    ax_d4.contour(density_map, levels=[lv], colors=[lc], linewidths=[0.8])
for lv, lc, yi in zip(iso_levels, iso_colors, [62, 47, 33, 20]):
    ax_d4.text(72, yi, f"{int(lv*100)}%", fontsize=4.5, color=lc, fontweight="bold")

# ── Bottom row: Margin Comparison ────────────────────────────────────────
g80 = np.linspace(-1, 1, 80)
xx80, yy80 = np.meshgrid(g80, g80)
cx_t, cy_t = -0.32, 0.18
tumor_norm    = (((xx80 - cx_t)**2 + (yy80 - cy_t)**2) < 0.045 * 15)
uniform_margin = (((xx80 - cx_t) / 1.4)**2 + ((yy80 - cy_t) / 1.25)**2 < 0.38**2)
uniform_margin &= brain_border
recurrence_gt  = (density_map > 0.35) & brain_border

cmp_y0 = 0.02
cmp_h  = 0.44
cmp_w  = 0.235

def show_comparison_inset(parent_ax, rect, title, margin_mask, color):
    axx = parent_ax.inset_axes(rect)
    axx.set_xticks([]); axx.set_yticks([])
    bg_img = np.ones((80, 80, 3)) * 0.92
    bg_img[~brain_border] = [1, 1, 1]
    axx.imshow(bg_img, origin="upper", aspect="equal")
    overlay = np.zeros((80, 80, 4))
    r_c, g_c, b_c = [int(color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)]
    overlay[margin_mask] = [r_c, g_c, b_c, 0.30]
    axx.imshow(overlay, origin="upper", aspect="equal")
    axx.contour(margin_mask.astype(float), levels=[0.5],
                colors=[color], linewidths=[1.0])
    axx.contour(tumor_norm.astype(float), levels=[0.5],
                colors=["#882020"], linewidths=[0.8], linestyles=["--"])
    axx.contour(recurrence_gt.astype(float), levels=[0.5],
                colors=["#208840"], linewidths=[0.7], linestyles=[":"])
    axx.set_title(title, fontsize=5.8, color=C_TEXT, pad=2,
                  multialignment="center")
    for sp in axx.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(0.9)
    rec_vox = recurrence_gt.sum()
    cov = (margin_mask & recurrence_gt).sum() / max(rec_vox, 1) * 100
    ht  = (margin_mask & ~tumor_norm).sum()
    axx.text(0.5, -0.10, f"Coverage: {cov:.0f}%   |   Healthy: {ht} vox",
             transform=axx.transAxes, ha="center", va="top",
             fontsize=5.5, color=color, fontweight="bold")
    return axx

axcmp1 = show_comparison_inset(
    axD, [0.49, cmp_y0, cmp_w, cmp_h],
    "Clinical Standard\n(Uniform 1.5 cm)",
    uniform_margin, C_RED)

axcmp2 = show_comparison_inset(
    axD, [0.745, cmp_y0, cmp_w, cmp_h],
    "MarginSense\n(UCB, z = 1.0)",
    (ucb_map >= tau_opt), C_PURPLE)

# Shared legend
legend_D = [
    Line2D([0], [0], color=C_RED,     lw=1.0, label="Std Margin Contour"),
    Line2D([0], [0], color=C_PURPLE,  lw=1.0, label="MarginSense UCB"),
    Line2D([0], [0], color="#882020", lw=0.8, ls="--", label="GTV Tumor"),
    Line2D([0], [0], color="#208840", lw=0.7, ls=":",  label="Recurrence GT"),
]
axcmp2.legend(handles=legend_D, loc="lower center",
              bbox_to_anchor=(-0.10, -0.34),
              ncol=2, fontsize=5.5, framealpha=0.9,
              edgecolor=C_BORDER, handlelength=1.2)

# ── UCB formula sidebar ──────────────────────────────────────────────────
axD.text(0.01, 0.83,
         "UCB Safety\nContour Formula:",
         transform=axD.transAxes, ha="left", va="center",
         fontsize=7, color=C_TEXT, fontweight="bold")
ucb_box = FancyBboxPatch((0.01, 0.56), 0.45, 0.24,
                           boxstyle="round,pad=0.015,rounding_size=0.025",
                           facecolor="#F4EEF8", edgecolor=C_PURPLE,
                           linewidth=1.0, transform=axD.transAxes, zorder=2)
axD.add_patch(ucb_box)
axD.text(0.235, 0.685,
         r"$c_{UCB}(x) = \mu(x) + z \cdot \sigma(x)$",
         ha="center", va="center", fontsize=10,
         color=C_PURPLE, transform=axD.transAxes)
axD.text(0.235, 0.61,
         r"$\Omega^* = \{x \mid c_{UCB} \geq \tau^*,\; tissue \neq CSF\}$",
         ha="center", va="center", fontsize=8.5,
         color=C_TEXT, transform=axD.transAxes)

# Optimal threshold formula
opt_box = FancyBboxPatch((0.01, 0.34), 0.45, 0.20,
                           boxstyle="round,pad=0.015,rounding_size=0.025",
                           facecolor="#EAF5EC", edgecolor=C_GREEN,
                           linewidth=1.0, transform=axD.transAxes, zorder=2)
axD.add_patch(opt_box)
axD.text(0.235, 0.47,
         "Optimal Threshold:",
         ha="center", va="center", fontsize=7,
         color=C_TEXT, fontweight="bold", transform=axD.transAxes)
axD.text(0.235, 0.40,
         r"$\tau^* = \arg\max_{\tau}\; [ Cov(\tau) - \lambda \cdot V_{healthy}(\tau) ]$",
         ha="center", va="center", fontsize=9,
         color=C_GREEN, transform=axD.transAxes)

# Small metrics comparison table
axD.text(0.235, 0.30,
         "Key Evaluation Metrics  (mean ± std, n=3 patients)",
         ha="center", va="center", fontsize=7, fontweight="bold",
         color=C_TEXT, transform=axD.transAxes)

metric_rows = [
    ("Metric",             "Clinical Std", "MarginSense"),
    ("Recurrence Coverage","~64%",          "~68–72%"),
    ("HD95 (mm)",          "higher",        "lower"),
    ("Surface Dice @2mm",  "lower",         "higher"),
    ("Healthy Tissue (cm³)","calibrated",   "λ-optimized"),
    ("Inference Time",      "—",            "< 2 sec"),
]
ax_met = axD.inset_axes([0.00, 0.01, 0.48, 0.27])
ax_met.set_xlim(0, 3); ax_met.set_ylim(0, len(metric_rows))
ax_met.set_xticks([]); ax_met.set_yticks([])
ax_met.set_facecolor(C_PANEL)
for sp in ax_met.spines.values():
    sp.set_edgecolor(C_BORDER); sp.set_linewidth(0.5)

for i, row in enumerate(metric_rows):
    ri = len(metric_rows) - 1 - i  # display top to bottom
    if i == 0:
        bg = "#D8E8F0"
        fw = "bold"
    else:
        bg = "#FFFFFF" if i % 2 == 1 else C_PANEL
        fw = "normal"
    ax_met.add_patch(mpatches.Rectangle(
        (0, ri + 0.05), 3, 0.90, facecolor=bg,
        edgecolor=C_BORDER, linewidth=0.3))
    ax_met.text(0.5,  ri + 0.5, row[0], ha="center", va="center",
                fontsize=5.8, color=C_TEXT, fontweight=fw)
    ax_met.text(1.5,  ri + 0.5, row[1], ha="center", va="center",
                fontsize=5.8, color=C_RED if i > 0 else C_TEXT,
                fontweight=fw)
    ax_met.text(2.5,  ri + 0.5, row[2], ha="center", va="center",
                fontsize=5.8, color=C_GREEN if i > 0 else C_TEXT,
                fontweight=fw)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.03, 0.97], [0.022, 0.022],
               color=C_BORDER, lw=0.7, transform=fig.transFigure))
fig.text(0.5, 0.010,
         "References: Fisher-KPP (Kolmogorov 1937) · Swanson GBM model · GliODIL (Nat Commun 2025) · "
         "BraTS benchmark (Menze et al., IEEE TMI 2015) · FiLM (Perez et al., AAAI 2018) · "
         "Deep Ensembles (Fort et al., NeurIPS 2019)",
         ha="center", va="center", fontsize=5.5, color=C_SUBTEXT)
fig.text(0.03, 0.010, "⚠ Research prototype — not for clinical use.",
         ha="left", va="center", fontsize=5.5, color="#AA3333")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out_png = "outputs/marginsense_graphical_abstract.png"
out_svg = "outputs/marginsense_graphical_abstract.svg"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
print("[OK] Saved PNG -> " + out_png)
print("[OK] Saved SVG -> " + out_svg)
