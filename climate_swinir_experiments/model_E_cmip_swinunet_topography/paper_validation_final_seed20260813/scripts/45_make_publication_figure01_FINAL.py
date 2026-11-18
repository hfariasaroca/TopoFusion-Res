from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
)
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
FINAL = SCRIPT.parents[1]

OUT = FINAL / "final_publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

PNG = OUT / "Figure01_architecture_FINAL.png"
PDF = OUT / "Figure01_architecture_FINAL.pdf"
SVG = OUT / "Figure01_architecture_FINAL.svg"
META = OUT / "Figure01_architecture_FINAL_metadata.json"


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    # Source figure is intentionally large.
    # At manuscript width these become publication-scale fonts.
    "font.size": 14,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 300,
})


# ------------------------------------------------------------
# Restrained, colorblind-friendly palette
# ------------------------------------------------------------

BLUE = "#356FA5"
BLUE_FILL = "#EAF2F8"

ORANGE = "#C9782C"
ORANGE_FILL = "#FDF2E9"

GRAY = "#666666"
GRAY_FILL = "#F4F4F4"

GREEN = "#4D8062"
GREEN_FILL = "#EAF3ED"

RED = "#B54A4A"
RED_FILL = "#FAEAEA"

BLACK = "#20252B"

STAGE1_BORDER = "#B8CEDF"
STAGE2_BORDER = "#E5BABA"

LIGHT_TEXT = "#5F6368"


# ============================================================
# CANVAS
# ============================================================

# Final aspect ratio ~2.1:1 rather than the previous ~4:1.
fig, ax = plt.subplots(
    figsize=(14.0, 6.6)
)

ax.set_xlim(0, 14)
ax.set_ylim(0, 6.6)
ax.axis("off")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ============================================================
# HELPERS
# ============================================================

def rounded_box(
    x,
    y,
    w,
    h,
    title,
    lines=None,
    edge=GRAY,
    face="white",
    title_fs=15.0,
    body_fs=12.8,
    title_color=BLACK,
    body_color=BLACK,
    lw=1.45,
    radius=0.08,
    title_y=0.69,
):
    """
    Draw a publication-style rounded rectangle.

    lines:
        list[str] rendered below the title.
        Mathtext can be used.
    """

    patch = FancyBboxPatch(
        (x, y),
        w, h,
        boxstyle=(
            f"round,pad=0.018,"
            f"rounding_size={radius}"
        ),
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_fs,
        weight="semibold",
        color=title_color,
        zorder=5,
    )

    if lines:
        body = "\n".join(lines)

        ax.text(
            x + w / 2,
            y + h * 0.31,
            body,
            ha="center",
            va="center",
            fontsize=body_fs,
            color=body_color,
            linespacing=1.20,
            zorder=5,
        )

    return patch


def arrow(
    x1,
    y1,
    x2,
    y2,
    color=GRAY,
    lw=1.6,
    mutation_scale=14,
    zorder=2,
):
    """Straight arrow."""

    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )

    ax.add_patch(a)

    return a


def elbow_arrow(
    points,
    color=GRAY,
    lw=1.45,
    mutation_scale=14,
    zorder=2,
):
    """
    Orthogonal polyline terminated by an arrow.
    points = [(x0,y0), (x1,y1), ...]
    """

    if len(points) < 2:
        raise ValueError("Need at least two points.")

    verts = points[:-1]

    codes = [MplPath.MOVETO] + [
        MplPath.LINETO
        for _ in range(len(verts) - 1)
    ]

    path = MplPath(
        verts,
        codes,
    )

    patch = PathPatch(
        path,
        fill=False,
        color=color,
        linewidth=lw,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )

    ax.add_patch(patch)

    arrow(
        points[-2][0],
        points[-2][1],
        points[-1][0],
        points[-1][1],
        color=color,
        lw=lw,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )


def stage_frame(
    x,
    y,
    w,
    h,
    title,
    color,
    subtitle=None,
):
    """Subtle stage container."""

    frame = FancyBboxPatch(
        (x, y),
        w, h,
        boxstyle=(
            "round,pad=0.018,"
            "rounding_size=0.10"
        ),
        linewidth=1.15,
        edgecolor=color,
        facecolor="white",
        zorder=0,
    )

    ax.add_patch(frame)

    ax.text(
        x + 0.28,
        y + h - 0.23,
        title,
        ha="left",
        va="center",
        fontsize=17.5,
        weight="bold",
        color=color,
        zorder=6,
    )

    if subtitle:
        ax.text(
            x + w - 0.28,
            y + h - 0.23,
            subtitle,
            ha="right",
            va="center",
            fontsize=11.5,
            color=LIGHT_TEXT,
            style="italic",
            zorder=6,
        )


# ============================================================
# STAGE FRAMES
# ============================================================

stage_frame(
    0.25,
    2.35,
    13.50,
    3.95,
    "Stage 1  ·  TopoFusion-Base",
    STAGE1_BORDER,
)

stage_frame(
    0.25,
    0.25,
    13.50,
    1.72,
    "Stage 2  ·  Residual refinement",
    STAGE2_BORDER,
    subtitle="TopoFusion-Base frozen",
)


# ============================================================
# STAGE 1 — ROW 1: ATMOSPHERIC BRANCH
# ============================================================

Y_ATM = 5.00
H_ATM = 0.90


# ------------------------------------------------------------
# Coarse atmospheric inputs
# ------------------------------------------------------------

rounded_box(
    0.62,
    Y_ATM,
    2.25,
    H_ATM,
    "Coarse atmospheric inputs",
    [
        r"$X_t$: 24 × 22 × 22",
        r"$S^c$: 1 × 22 × 22",
    ],
    edge=BLUE,
    face=BLUE_FILL,
    title_fs=14.2,
    body_fs=12.1,
)


# ------------------------------------------------------------
# Atmospheric encoder
# ------------------------------------------------------------

rounded_box(
    3.25,
    Y_ATM,
    1.95,
    H_ATM,
    "Atmospheric encoder",
    [
        "ConvBlock + ResBlock",
        "25 × 22 × 22 → 64 × 22 × 22",
    ],
    edge=BLUE,
    face=BLUE_FILL,
    title_fs=14.0,
    body_fs=11.3,
)


# ------------------------------------------------------------
# Conditioning addition
# ------------------------------------------------------------

PLUS_X = 5.63
PLUS_Y = Y_ATM + H_ATM / 2

plus = Circle(
    (PLUS_X, PLUS_Y),
    radius=0.16,
    edgecolor=GRAY,
    facecolor="white",
    linewidth=1.35,
    zorder=4,
)

ax.add_patch(plus)

ax.text(
    PLUS_X,
    PLUS_Y - 0.005,
    "+",
    ha="center",
    va="center",
    fontsize=16,
    color=BLACK,
    zorder=5,
)


# ------------------------------------------------------------
# Multi-resolution decoder
# ------------------------------------------------------------

rounded_box(
    6.02,
    Y_ATM,
    2.58,
    H_ATM,
    "Multi-resolution decoder",
    [
        "22×22  →  44×44  →  64×64",
        "upsampling + ConvBlocks / ResBlocks",
    ],
    edge=BLUE,
    face=BLUE_FILL,
    title_fs=14.0,
    body_fs=11.6,
)


# ------------------------------------------------------------
# Atmospheric representation
# ------------------------------------------------------------

rounded_box(
    9.03,
    Y_ATM,
    2.25,
    H_ATM,
    "Atmospheric representation",
    [
        r"$H_t^a$",
        "64 × 64 × 64",
    ],
    edge=BLUE,
    face=BLUE_FILL,
    title_fs=13.7,
    body_fs=12.0,
)


# ------------------------------------------------------------
# Atmospheric arrows
# ------------------------------------------------------------

arrow(
    2.87,
    PLUS_Y,
    3.25,
    PLUS_Y,
    color=BLUE,
)

arrow(
    5.20,
    PLUS_Y,
    PLUS_X - 0.17,
    PLUS_Y,
    color=BLUE,
)

arrow(
    PLUS_X + 0.17,
    PLUS_Y,
    6.02,
    PLUS_Y,
    color=BLUE,
)

arrow(
    8.60,
    PLUS_Y,
    9.03,
    PLUS_Y,
    color=BLUE,
)


# ============================================================
# STAGE 1 — ROW 2: GLOBAL CONDITIONING
# ============================================================

Y_CONTEXT = 4.02
H_CONTEXT = 0.72

rounded_box(
    0.62,
    Y_CONTEXT,
    2.25,
    H_CONTEXT,
    "Global context",
    [
        r"$\tau_t$: 3 temporal features   ·   $f_t$: 4 forcings",
    ],
    edge=GRAY,
    face=GRAY_FILL,
    title_fs=13.6,
    body_fs=10.9,
    title_y=0.67,
)

rounded_box(
    3.25,
    Y_CONTEXT,
    1.95,
    H_CONTEXT,
    "Global conditioning",
    [
        r"MLP: 7 → 64   ·   $C(c_t)$: 64 × 1 × 1",
    ],
    edge=GRAY,
    face=GRAY_FILL,
    title_fs=13.4,
    body_fs=10.8,
    title_y=0.67,
)


arrow(
    2.87,
    Y_CONTEXT + H_CONTEXT / 2,
    3.25,
    Y_CONTEXT + H_CONTEXT / 2,
    color=GRAY,
    lw=1.4,
)


# Conditioning → plus
elbow_arrow(
    [
        (
            5.20,
            Y_CONTEXT + H_CONTEXT / 2,
        ),
        (
            PLUS_X,
            Y_CONTEXT + H_CONTEXT / 2,
        ),
        (
            PLUS_X,
            PLUS_Y - 0.17,
        ),
    ],
    color=GRAY,
    lw=1.35,
)


# ============================================================
# STAGE 1 — ROW 3: FINE STATIC + FUSION + BASE
# ============================================================

Y_FUS = 2.73
H_FUS = 0.90


# ------------------------------------------------------------
# Fine static input
# ------------------------------------------------------------

rounded_box(
    0.62,
    Y_FUS,
    2.42,
    H_FUS,
    "Fine-scale static features",
    [
        r"$S^f$: 6 × 64 × 64",
        "elevation · land fraction · lat/lon",
        "local orographic gradient · roughness",
    ],
    edge=ORANGE,
    face=ORANGE_FILL,
    title_fs=13.5,
    body_fs=10.6,
)


# ------------------------------------------------------------
# Fine encoder
# ------------------------------------------------------------

rounded_box(
    3.37,
    Y_FUS,
    1.92,
    H_FUS,
    "Fine-scale encoder",
    [
        "ConvBlock + ResBlock",
        "6 × 64 × 64 → 32 × 64 × 64",
    ],
    edge=ORANGE,
    face=ORANGE_FILL,
    title_fs=13.5,
    body_fs=10.9,
)


# ------------------------------------------------------------
# Fusion
# ------------------------------------------------------------

rounded_box(
    5.67,
    Y_FUS,
    1.95,
    H_FUS,
    "Multi-source fusion",
    [
        r"$[H_t^a, H^f]$",
        "96 × 64 × 64",
    ],
    edge=GREEN,
    face=GREEN_FILL,
    title_fs=13.5,
    body_fs=11.4,
    lw=1.55,
)


# ------------------------------------------------------------
# Fused representation
# ------------------------------------------------------------

rounded_box(
    8.00,
    Y_FUS,
    1.80,
    H_FUS,
    "Fused representation",
    [
        r"$H_t^{fusion}$",
        "64 × 64 × 64",
    ],
    edge=GREEN,
    face=GREEN_FILL,
    title_fs=13.0,
    body_fs=11.3,
)


# ------------------------------------------------------------
# Base head
# ------------------------------------------------------------

rounded_box(
    10.18,
    Y_FUS,
    1.55,
    H_FUS,
    "Base head",
    [
        "Conv3×3 → GELU",
        "→ Conv1×1",
    ],
    edge=BLUE,
    face=BLUE_FILL,
    title_fs=13.4,
    body_fs=11.0,
)


# ------------------------------------------------------------
# TopoFusion-Base
# ------------------------------------------------------------

rounded_box(
    12.10,
    Y_FUS,
    1.25,
    H_FUS,
    "TopoFusion-Base",
    [
        r"$T_{base}$",
        "1 × 64 × 64",
    ],
    edge=BLUE,
    face="#DFEDF7",
    title_fs=12.0,
    body_fs=11.0,
    lw=1.75,
)


# ------------------------------------------------------------
# Fine branch arrows
# ------------------------------------------------------------

arrow(
    3.04,
    Y_FUS + H_FUS / 2,
    3.37,
    Y_FUS + H_FUS / 2,
    color=ORANGE,
)

arrow(
    5.29,
    Y_FUS + H_FUS / 2,
    5.67,
    Y_FUS + H_FUS / 2,
    color=ORANGE,
)


# ------------------------------------------------------------
# Atmospheric representation → fusion
# ------------------------------------------------------------

elbow_arrow(
    [
        (
            10.155,
            Y_ATM,
        ),
        (
            10.155,
            3.86,
        ),
        (
            6.645,
            3.86,
        ),
        (
            6.645,
            Y_FUS + H_FUS,
        ),
    ],
    color=BLUE,
    lw=1.45,
)


# ------------------------------------------------------------
# Fusion → base output
# ------------------------------------------------------------

arrow(
    7.62,
    Y_FUS + H_FUS / 2,
    8.00,
    Y_FUS + H_FUS / 2,
    color=GREEN,
)

arrow(
    9.80,
    Y_FUS + H_FUS / 2,
    10.18,
    Y_FUS + H_FUS / 2,
    color=GREEN,
)

arrow(
    11.73,
    Y_FUS + H_FUS / 2,
    12.10,
    Y_FUS + H_FUS / 2,
    color=BLUE,
)


# ============================================================
# STAGE 2 — ONE CLEAN HORIZONTAL FLOW
# ============================================================

Y_RES = 0.62
H_RES = 0.88


# ------------------------------------------------------------
# Residual inputs
# ------------------------------------------------------------

rounded_box(
    0.78,
    Y_RES,
    3.00,
    H_RES,
    "Residual inputs",
    [
        r"$H^{fusion}$ (64) + $S^f$ (6) + $T_{base}$ (1)",
        "71 × 64 × 64",
    ],
    edge=RED,
    face="white",
    title_fs=14.3,
    body_fs=11.8,
    lw=1.50,
)


# ------------------------------------------------------------
# Residual head
# ------------------------------------------------------------

rounded_box(
    4.17,
    Y_RES,
    2.60,
    H_RES,
    "Residual head",
    [
        "Conv3×3 → GN → GELU",
        "ResBlock → Conv1×1",
    ],
    edge=RED,
    face=RED_FILL,
    title_fs=14.3,
    body_fs=11.5,
)


# ------------------------------------------------------------
# Bounded correction
# ------------------------------------------------------------

rounded_box(
    7.16,
    Y_RES,
    2.75,
    H_RES,
    "Bounded correction",
    [
        r"$\Delta T = 2.5\,\tanh(raw/2.5)$",
        r"$-2.5^\circ C \leq \Delta T \leq 2.5^\circ C$",
    ],
    edge=RED,
    face=RED_FILL,
    title_fs=14.1,
    body_fs=11.3,
)


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

rounded_box(
    10.31,
    Y_RES,
    3.00,
    H_RES,
    "TopoFusion-Res",
    [
        r"$T_{final}=T_{base}+\Delta T$",
        "1 × 64 × 64",
    ],
    edge=RED,
    face="#F7DDDD",
    title_fs=14.4,
    body_fs=11.8,
    lw=1.85,
)


# ------------------------------------------------------------
# Residual flow
# ------------------------------------------------------------

arrow(
    3.78,
    Y_RES + H_RES / 2,
    4.17,
    Y_RES + H_RES / 2,
    color=RED,
    lw=1.55,
)

arrow(
    6.77,
    Y_RES + H_RES / 2,
    7.16,
    Y_RES + H_RES / 2,
    color=RED,
    lw=1.55,
)

arrow(
    9.91,
    Y_RES + H_RES / 2,
    10.31,
    Y_RES + H_RES / 2,
    color=RED,
    lw=1.55,
)


# ============================================================
# STAGE-TRANSITION CUE
# ============================================================

# A single subtle cue instead of three long dashed cables.
ax.annotate(
    "",
    xy=(2.28, 1.51),
    xytext=(2.28, 2.34),
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=1.0,
        color="#9B9B9B",
        linestyle=(0, (3, 3)),
        mutation_scale=12,
    ),
    zorder=1,
)

ax.text(
    2.48,
    1.91,
    "reuse frozen Stage-1 tensors",
    ha="left",
    va="center",
    fontsize=10.5,
    color="#777777",
    style="italic",
    zorder=5,
)


# ============================================================
# AUDIT
# ============================================================

audit_text = """
TopoFusion-Base
TopoFusion-Res
24 × 22 × 22
1 × 22 × 22
25 × 22 × 22
22×22 → 44×44 → 64×64
6 × 64 × 64
local orographic gradient
32 × 64 × 64
96 × 64 × 64
64 × 64 × 64
71 × 64 × 64
Delta T = 2.5 tanh(raw/2.5)
1 × 64 × 64
"""

required = [
    "TopoFusion-Base",
    "TopoFusion-Res",
    "24 × 22 × 22",
    "1 × 22 × 22",
    "25 × 22 × 22",
    "22×22 → 44×44 → 64×64",
    "6 × 64 × 64",
    "local orographic gradient",
    "32 × 64 × 64",
    "96 × 64 × 64",
    "64 × 64 × 64",
    "71 × 64 × 64",
    "Delta T = 2.5 tanh(raw/2.5)",
    "1 × 64 × 64",
]

missing = [
    s
    for s in required
    if s not in audit_text
]

if missing:
    raise RuntimeError(
        "Architecture audit failed. Missing:\n"
        + "\n".join(missing)
    )


for forbidden in [
    "SwinUNet",
    "SwinIR",
    "E2 v2",
]:
    if forbidden in audit_text:
        raise RuntimeError(
            f"Legacy label found: {forbidden}"
        )


if re.search(
    r"\bcal orographic gradient\b",
    audit_text,
    flags=re.IGNORECASE,
):
    raise RuntimeError(
        "Truncated label found: cal orographic gradient"
    )


# ============================================================
# SAVE
# ============================================================

fig.subplots_adjust(
    left=0.01,
    right=0.99,
    bottom=0.015,
    top=0.985,
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

fig.savefig(
    SVG,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

plt.close(fig)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "figure": "Figure 1",
    "model": "TopoFusion-Res",

    "stage_1": {
        "name": "TopoFusion-Base",

        "dynamic_predictors":
            "24 x 22 x 22",

        "coarse_orography":
            "1 x 22 x 22",

        "coarse_concat":
            "25 x 22 x 22",

        "decoder":
            "22x22 -> 44x44 -> 64x64",

        "fine_static":
            "6 x 64 x 64",

        "fine_static_channels": [
            "elevation",
            "land fraction",
            "latitude",
            "longitude",
            "local orographic gradient",
            "roughness",
        ],

        "fine_representation":
            "32 x 64 x 64",

        "fusion_input":
            "96 x 64 x 64",

        "fused_representation":
            "64 x 64 x 64",

        "base_prediction":
            "1 x 64 x 64",
    },

    "stage_2": {
        "base_frozen": True,

        "residual_input_channels":
            "64 + 6 + 1 = 71",

        "residual_input":
            "71 x 64 x 64",

        "bounded_correction":
            "Delta T = 2.5 tanh(raw/2.5)",

        "correction_range_C": [
            -2.5,
            2.5,
        ],

        "final_prediction":
            "1 x 64 x 64",
    },

    "design": {
        "layout":
            "two aligned stages",

        "stage_transition":
            "single conceptual reuse cue",

        "output_formats": [
            "PNG",
            "PDF",
            "SVG",
        ],
    },
}

with open(
    META,
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# DONE
# ============================================================

print("=" * 90)
print("FIGURE 1 — FINAL PUBLICATION VERSION")
print("=" * 90)

print("PNG :", PNG)
print("PDF :", PDF)
print("SVG :", SVG)
print("META:", META)

print()
print("Architecture audit : OK")
print("Legacy-label audit : OK")
print("Topography label    : local orographic gradient")
print()
print("DONE")
