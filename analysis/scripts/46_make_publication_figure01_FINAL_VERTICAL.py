from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()

FINAL = SCRIPT.parents[1]

OUT = FINAL / "final_publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

PNG = OUT / "Figure01_architecture_FINAL_VERTICAL.png"
PDF = OUT / "Figure01_architecture_FINAL_VERTICAL.pdf"
SVG = OUT / "Figure01_architecture_FINAL_VERTICAL.svg"

META = OUT / "Figure01_architecture_FINAL_VERTICAL_metadata.json"


# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": 13,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 300,
})


# ============================================================
# COLORS
# ============================================================

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
MUTED = "#666A70"

STAGE1_BORDER = "#B8CEDF"
STAGE2_BORDER = "#E5BABA"


# ============================================================
# CANVAS
#
# Deliberately much taller than previous versions.
# Aspect ratio ~1.5 rather than ~4.
# ============================================================

FIG_W = 16.0
FIG_H = 10.6

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

ax.set_xlim(0, 16)
ax.set_ylim(0, 10.6)

ax.axis("off")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ============================================================
# DRAWING HELPERS
# ============================================================

def stage_frame(
    x,
    y,
    w,
    h,
    title,
    edge,
    title_color,
    subtitle=None,
):
    """
    Draw subtle rounded stage container.
    """

    frame = FancyBboxPatch(
        (x, y),
        w,
        h,

        boxstyle=(
            "round,pad=0.018,"
            "rounding_size=0.10"
        ),

        linewidth=1.25,
        edgecolor=edge,
        facecolor="white",

        zorder=0,
    )

    ax.add_patch(frame)

    ax.text(
        x + 0.34,
        y + h - 0.27,

        title,

        ha="left",
        va="center",

        fontsize=18.5,
        fontweight="bold",
        color=title_color,

        zorder=10,
    )

    if subtitle is not None:

        ax.text(
            x + w - 0.34,
            y + h - 0.27,

            subtitle,

            ha="right",
            va="center",

            fontsize=12.2,
            fontstyle="italic",
            color=MUTED,

            zorder=10,
        )


def box(
    x,
    y,
    w,
    h,

    title,
    body_lines,

    edge,
    fill,

    title_fs=14.2,
    body_fs=12.0,

    linewidth=1.55,

    title_fraction=0.76,
    body_fraction=0.35,
):
    """
    Draw publication box with separated title/body hierarchy.
    """

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,

        boxstyle=(
            "round,pad=0.020,"
            "rounding_size=0.075"
        ),

        linewidth=linewidth,
        edgecolor=edge,
        facecolor=fill,

        zorder=3,
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * title_fraction,

        title,

        ha="center",
        va="center",

        fontsize=title_fs,
        fontweight="semibold",
        color=BLACK,

        linespacing=1.05,

        zorder=5,
    )

    if body_lines:

        ax.text(
            x + w / 2,
            y + h * body_fraction,

            "\n".join(body_lines),

            ha="center",
            va="center",

            fontsize=body_fs,
            color=BLACK,

            linespacing=1.18,

            zorder=5,
        )

    return patch


def arrow(
    x1,
    y1,
    x2,
    y2,

    color,
    linewidth=1.6,
    size=15,
    linestyle="-",
):
    """
    Straight arrow.
    """

    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),

        arrowstyle="-|>",

        mutation_scale=size,

        linewidth=linewidth,
        linestyle=linestyle,

        color=color,

        shrinkA=1.0,
        shrinkB=1.0,

        zorder=2,
    )

    ax.add_patch(arr)

    return arr


def elbow(
    points,
    color,
    linewidth=1.45,
    size=14,
    linestyle="-",
):
    """
    Orthogonal route:
        points = [(x0,y0), ..., (xn,yn)]

    All intermediate segments are plain lines.
    Last segment gets arrowhead.
    """

    if len(points) < 2:
        raise ValueError(
            "elbow() requires at least two points."
        )

    # intermediate segments
    for i in range(len(points) - 2):

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        ax.plot(
            [x1, x2],
            [y1, y2],

            color=color,
            linewidth=linewidth,
            linestyle=linestyle,

            solid_capstyle="round",

            zorder=1,
        )

    # final segment
    x1, y1 = points[-2]
    x2, y2 = points[-1]

    arrow(
        x1,
        y1,
        x2,
        y2,

        color=color,

        linewidth=linewidth,
        size=size,
        linestyle=linestyle,
    )


# ============================================================
# STAGE 1 FRAME
# ============================================================

STAGE1_X = 0.25
STAGE1_Y = 3.45
STAGE1_W = 15.50
STAGE1_H = 6.75

stage_frame(
    STAGE1_X,
    STAGE1_Y,
    STAGE1_W,
    STAGE1_H,

    "Stage 1  ·  TopoFusion-Base",

    edge=STAGE1_BORDER,
    title_color=BLUE,
)


# ============================================================
# STAGE 1 — ATMOSPHERIC BRANCH
# ============================================================

Y_ATM = 8.02
H_ATM = 1.42


# ------------------------------------------------------------
# COARSE INPUTS
# ------------------------------------------------------------

box(
    0.70,
    Y_ATM,

    2.55,
    H_ATM,

    "Coarse atmospheric\ninputs",

    [
        r"$X_t$: 24 × 22 × 22",
        r"$S^c$: 1 × 22 × 22",
    ],

    edge=BLUE,
    fill=BLUE_FILL,

    title_fs=14.8,
    body_fs=12.8,

    title_fraction=0.73,
    body_fraction=0.29,
)


# ------------------------------------------------------------
# ATMOSPHERIC ENCODER
# ------------------------------------------------------------

box(
    3.68,
    Y_ATM,

    2.20,
    H_ATM,

    "Atmospheric\nencoder",

    [
        "ConvBlock + ResBlock",
        "25 × 22 × 22",
        "→ 64 × 22 × 22",
    ],

    edge=BLUE,
    fill=BLUE_FILL,

    title_fs=14.8,
    body_fs=11.7,

    title_fraction=0.74,
    body_fraction=0.28,
)


# ------------------------------------------------------------
# CONDITIONING ADDITION
# ------------------------------------------------------------

PLUS_X = 6.30
PLUS_Y = Y_ATM + H_ATM / 2

plus = Circle(
    (PLUS_X, PLUS_Y),

    radius=0.18,

    edgecolor=GRAY,
    facecolor="white",

    linewidth=1.45,

    zorder=4,
)

ax.add_patch(plus)

ax.text(
    PLUS_X,
    PLUS_Y - 0.008,

    "+",

    ha="center",
    va="center",

    fontsize=17,
    color=BLACK,

    zorder=5,
)


# ------------------------------------------------------------
# MULTI-RESOLUTION DECODER
# ------------------------------------------------------------

box(
    6.68,
    Y_ATM,

    3.00,
    H_ATM,

    "Multi-resolution\ndecoder",

    [
        "22×22 → 44×44 → 64×64",
        "upsampling +",
        "ConvBlocks / ResBlocks",
    ],

    edge=BLUE,
    fill=BLUE_FILL,

    title_fs=14.8,
    body_fs=11.6,

    title_fraction=0.74,
    body_fraction=0.27,
)


# ------------------------------------------------------------
# ATMOSPHERIC REPRESENTATION
# ------------------------------------------------------------

box(
    10.18,
    Y_ATM,

    3.05,
    H_ATM,

    "Atmospheric\nrepresentation",

    [
        r"$H_t^a$",
        "64 × 64 × 64",
    ],

    edge=BLUE,
    fill=BLUE_FILL,

    title_fs=14.8,
    body_fs=12.7,

    title_fraction=0.74,
    body_fraction=0.29,
)


# ------------------------------------------------------------
# ATMOSPHERIC FLOW
# ------------------------------------------------------------

ATM_CY = Y_ATM + H_ATM / 2

arrow(
    3.25,
    ATM_CY,
    3.68,
    ATM_CY,

    color=BLUE,
)

arrow(
    5.88,
    ATM_CY,
    PLUS_X - 0.18,
    ATM_CY,

    color=BLUE,
)

arrow(
    PLUS_X + 0.18,
    ATM_CY,
    6.68,
    ATM_CY,

    color=BLUE,
)

arrow(
    9.68,
    ATM_CY,
    10.18,
    ATM_CY,

    color=BLUE,
)


# ============================================================
# STAGE 1 — GLOBAL CONDITIONING ROW
# ============================================================

Y_CTX = 6.55
H_CTX = 1.02


box(
    0.70,
    Y_CTX,

    2.55,
    H_CTX,

    "Global context",

    [
        r"$\tau_t$: 3 temporal features",
        r"$f_t$: 4 forcing features",
    ],

    edge=GRAY,
    fill=GRAY_FILL,

    title_fs=14.6,
    body_fs=11.9,

    title_fraction=0.70,
    body_fraction=0.27,
)


box(
    3.68,
    Y_CTX,

    2.20,
    H_CTX,

    "Global conditioning",

    [
        "MLP: 7 → 64",
        r"$C(c_t)$: 64 × 1 × 1",
    ],

    edge=GRAY,
    fill=GRAY_FILL,

    title_fs=14.2,
    body_fs=11.7,

    title_fraction=0.70,
    body_fraction=0.27,
)


CTX_CY = Y_CTX + H_CTX / 2

arrow(
    3.25,
    CTX_CY,
    3.68,
    CTX_CY,

    color=GRAY,

    linewidth=1.4,
)


# ------------------------------------------------------------
# CONDITIONING → ADDITION
# ------------------------------------------------------------

elbow(
    [
        (5.88, CTX_CY),
        (PLUS_X, CTX_CY),
        (PLUS_X, PLUS_Y - 0.19),
    ],

    color=GRAY,

    linewidth=1.35,
)


# ============================================================
# STAGE 1 — FINE STATIC BRANCH
# ============================================================

Y_FINE = 4.95
H_FINE = 1.28


# ------------------------------------------------------------
# FINE STATIC INPUTS
# ------------------------------------------------------------

box(
    0.70,
    Y_FINE,

    3.30,
    H_FINE,

    "Fine-scale static\nfeatures",

    [
        r"$S^f$: 6 × 64 × 64",
        "elevation · land fraction",
        "latitude · longitude",
        "local orographic gradient · roughness",
    ],

    edge=ORANGE,
    fill=ORANGE_FILL,

    title_fs=14.6,
    body_fs=10.7,

    title_fraction=0.76,
    body_fraction=0.28,
)


# ------------------------------------------------------------
# FINE ENCODER
# ------------------------------------------------------------

box(
    4.38,
    Y_FINE,

    2.48,
    H_FINE,

    "Fine-scale\nencoder",

    [
        "ConvBlock + ResBlock",
        "6 × 64 × 64",
        "→ 32 × 64 × 64",
    ],

    edge=ORANGE,
    fill=ORANGE_FILL,

    title_fs=14.6,
    body_fs=11.4,

    title_fraction=0.75,
    body_fraction=0.28,
)


FINE_CY = Y_FINE + H_FINE / 2

arrow(
    4.00,
    FINE_CY,
    4.38,
    FINE_CY,

    color=ORANGE,
)


# ============================================================
# STAGE 1 — FUSION / BASE ROW
# ============================================================

Y_FUS = 3.70
H_FUS = 1.00


# ------------------------------------------------------------
# FUSION
# ------------------------------------------------------------

box(
    6.70,
    Y_FUS,

    2.38,
    H_FUS,

    "Multi-source\nfusion",

    [
        r"$[H_t^a, H^f]$",
        "96 × 64 × 64",
    ],

    edge=GREEN,
    fill=GREEN_FILL,

    title_fs=14.2,
    body_fs=11.8,

    linewidth=1.65,

    title_fraction=0.72,
    body_fraction=0.28,
)


# ------------------------------------------------------------
# FUSED REPRESENTATION
# ------------------------------------------------------------

box(
    9.45,
    Y_FUS,

    2.12,
    H_FUS,

    "Fused\nrepresentation",

    [
        r"$H_t^{fusion}$",
        "64 × 64 × 64",
    ],

    edge=GREEN,
    fill=GREEN_FILL,

    title_fs=13.8,
    body_fs=11.8,

    title_fraction=0.72,
    body_fraction=0.28,
)


# ------------------------------------------------------------
# BASE HEAD
# ------------------------------------------------------------

box(
    11.92,
    Y_FUS,

    1.65,
    H_FUS,

    "Base head",

    [
        "Conv3×3 → GELU",
        "→ Conv1×1",
    ],

    edge=BLUE,
    fill=BLUE_FILL,

    title_fs=13.4,
    body_fs=10.9,

    title_fraction=0.70,
    body_fraction=0.28,
)


# ------------------------------------------------------------
# BASE OUTPUT
# ------------------------------------------------------------

box(
    13.88,
    Y_FUS,

    1.52,
    H_FUS,

    "TopoFusion-\nBase",

    [
        r"$T_{base}$",
        "1 × 64 × 64",
    ],

    edge=BLUE,
    fill="#DFEDF7",

    title_fs=12.6,
    body_fs=10.9,

    linewidth=1.85,

    title_fraction=0.72,
    body_fraction=0.27,
)


FUS_CY = Y_FUS + H_FUS / 2


# ------------------------------------------------------------
# FINE ENCODER → FUSION
# ------------------------------------------------------------

elbow(
    [
        (
            5.62,
            Y_FINE,
        ),
        (
            5.62,
            Y_FUS + H_FUS / 2,
        ),
        (
            6.70,
            FUS_CY,
        ),
    ],

    color=ORANGE,

    linewidth=1.45,
)


# ------------------------------------------------------------
# ATMOSPHERIC REPRESENTATION → FUSION
# ------------------------------------------------------------

elbow(
    [
        (
            11.70,
            Y_ATM,
        ),
        (
            11.70,
            4.82,
        ),
        (
            7.89,
            4.82,
        ),
        (
            7.89,
            Y_FUS + H_FUS,
        ),
    ],

    color=BLUE,

    linewidth=1.45,
)


# ------------------------------------------------------------
# FUSION FLOW
# ------------------------------------------------------------

arrow(
    9.08,
    FUS_CY,
    9.45,
    FUS_CY,

    color=GREEN,
)

arrow(
    11.57,
    FUS_CY,
    11.92,
    FUS_CY,

    color=GREEN,
)

arrow(
    13.57,
    FUS_CY,
    13.88,
    FUS_CY,

    color=BLUE,
)


# ============================================================
# STAGE 2 FRAME
# ============================================================

STAGE2_X = 0.25
STAGE2_Y = 0.22
STAGE2_W = 15.50
STAGE2_H = 2.72

stage_frame(
    STAGE2_X,
    STAGE2_Y,
    STAGE2_W,
    STAGE2_H,

    "Stage 2  ·  Residual refinement",

    edge=STAGE2_BORDER,
    title_color=RED,

    subtitle=(
        "uses frozen "
        r"$H^{fusion}$, $S^f$ and $T_{base}$"
    ),
)


# ============================================================
# STAGE 2 — CLEAN FULL-WIDTH FLOW
# ============================================================

Y_RES = 0.67
H_RES = 1.33


# ------------------------------------------------------------
# RESIDUAL INPUTS
# ------------------------------------------------------------

box(
    0.72,
    Y_RES,

    3.42,
    H_RES,

    "Residual inputs",

    [
        r"$H^{fusion}$  ·  64 channels",
        r"$S^f$  ·  6 channels",
        r"$T_{base}$  ·  1 channel",
        "71 × 64 × 64",
    ],

    edge=RED,
    fill="white",

    title_fs=15.0,
    body_fs=11.3,

    linewidth=1.65,

    title_fraction=0.78,
    body_fraction=0.32,
)


# ------------------------------------------------------------
# RESIDUAL HEAD
# ------------------------------------------------------------

box(
    4.55,
    Y_RES,

    3.02,
    H_RES,

    "Residual head",

    [
        "Conv3×3 → GN → GELU",
        "ResBlock",
        "→ Conv1×1",
    ],

    edge=RED,
    fill=RED_FILL,

    title_fs=15.0,
    body_fs=11.7,

    title_fraction=0.76,
    body_fraction=0.31,
)


# ------------------------------------------------------------
# BOUNDED CORRECTION
# ------------------------------------------------------------

box(
    7.98,
    Y_RES,

    3.20,
    H_RES,

    "Bounded correction",

    [
        r"$\Delta T = 2.5\,\tanh(raw/2.5)$",
        "",
        r"$-2.5^\circ C \leq \Delta T \leq 2.5^\circ C$",
    ],

    edge=RED,
    fill=RED_FILL,

    title_fs=14.8,
    body_fs=11.5,

    title_fraction=0.76,
    body_fraction=0.30,
)


# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

box(
    11.60,
    Y_RES,

    3.68,
    H_RES,

    "TopoFusion-Res",

    [
        r"$T_{final}=T_{base}+\Delta T$",
        "",
        "1 × 64 × 64",
    ],

    edge=RED,
    fill="#F7DDDD",

    title_fs=15.2,
    body_fs=12.0,

    linewidth=1.95,

    title_fraction=0.76,
    body_fraction=0.30,
)


RES_CY = Y_RES + H_RES / 2


# ------------------------------------------------------------
# STAGE 2 FLOW
# ------------------------------------------------------------

arrow(
    4.14,
    RES_CY,
    4.55,
    RES_CY,

    color=RED,

    linewidth=1.65,
)

arrow(
    7.57,
    RES_CY,
    7.98,
    RES_CY,

    color=RED,

    linewidth=1.65,
)

arrow(
    11.18,
    RES_CY,
    11.60,
    RES_CY,

    color=RED,

    linewidth=1.65,
)


# ============================================================
# SINGLE STAGE TRANSITION CUE
#
# Not data flow — conceptual transition only.
# ============================================================

TRANS_X = 2.43

arrow(
    TRANS_X,
    STAGE1_Y,
    TRANS_X,
    STAGE2_Y + STAGE2_H,

    color="#A0A0A0",

    linewidth=1.05,
    size=12,

    linestyle=(0, (3, 3)),
)


# ============================================================
# SCIENTIFIC AUDIT
# ============================================================

audit_text = """
TopoFusion-Base
TopoFusion-Res
24 × 22 × 22
1 × 22 × 22
25 × 22 × 22
64 × 22 × 22
22×22 → 44×44 → 64×64
64 × 64 × 64
6 × 64 × 64
local orographic gradient
32 × 64 × 64
96 × 64 × 64
71 × 64 × 64
Delta T = 2.5 tanh(raw/2.5)
-2.5 C <= Delta T <= 2.5 C
1 × 64 × 64
"""


required = [
    "TopoFusion-Base",
    "TopoFusion-Res",

    "24 × 22 × 22",
    "1 × 22 × 22",
    "25 × 22 × 22",

    "64 × 22 × 22",

    "22×22 → 44×44 → 64×64",

    "64 × 64 × 64",

    "6 × 64 × 64",

    "local orographic gradient",

    "32 × 64 × 64",

    "96 × 64 × 64",

    "71 × 64 × 64",

    "Delta T = 2.5 tanh(raw/2.5)",

    "1 × 64 × 64",
]


missing = [
    item
    for item in required
    if item not in audit_text
]

if missing:

    raise RuntimeError(
        "Architecture audit failed.\n"
        "Missing:\n"
        + "\n".join(missing)
    )


for forbidden in [
    "SwinUNet",
    "SwinIR",
    "E2 v2",
]:

    if forbidden in audit_text:

        raise RuntimeError(
            f"Forbidden legacy label: {forbidden}"
        )


if re.search(
    r"\bcal orographic gradient\b",
    audit_text,
    flags=re.IGNORECASE,
):

    raise RuntimeError(
        "Forbidden truncated topography label."
    )


# ============================================================
# SAVE
# ============================================================

fig.subplots_adjust(
    left=0.008,
    right=0.992,

    bottom=0.008,
    top=0.992,
)


fig.savefig(
    PNG,

    dpi=300,

    bbox_inches="tight",
    pad_inches=0.035,

    facecolor="white",
)


fig.savefig(
    PDF,

    bbox_inches="tight",
    pad_inches=0.035,

    facecolor="white",
)


fig.savefig(
    SVG,

    bbox_inches="tight",
    pad_inches=0.035,

    facecolor="white",
)


plt.close(fig)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "figure": "Figure 1",

    "model": "TopoFusion-Res",

    "version":
        "FINAL_VERTICAL",

    "design": {

        "aspect_ratio":
            FIG_W / FIG_H,

        "orientation":
            "vertically expanded two-stage architecture",

        "goal":
            (
                "publication-scale readability with "
                "larger internal typography"
            ),
    },

    "stage_1": {

        "name":
            "TopoFusion-Base",

        "dynamic_predictors":
            "24 x 22 x 22",

        "coarse_orography":
            "1 x 22 x 22",

        "coarse_input_concat":
            "25 x 22 x 22",

        "encoded_coarse_state":
            "64 x 22 x 22",

        "spatial_reconstruction":
            "22x22 -> 44x44 -> 64x64",

        "atmospheric_representation":
            "64 x 64 x 64",

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

        "base_network_frozen":
            True,

        "inputs": {
            "fused_representation_channels": 64,
            "fine_static_channels": 6,
            "base_prediction_channels": 1,
        },

        "residual_input":
            "71 x 64 x 64",

        "bounded_residual":
            "Delta T = 2.5 tanh(raw/2.5)",

        "residual_range_C": [
            -2.5,
            2.5,
        ],

        "final_prediction":
            "1 x 64 x 64",
    },

    "output_files": {
        "png": str(PNG),
        "pdf": str(PDF),
        "svg": str(SVG),
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
print("FIGURE 1 — FINAL VERTICAL PUBLICATION VERSION")
print("=" * 90)

print("PNG :", PNG)
print("PDF :", PDF)
print("SVG :", SVG)
print("META:", META)

print()
print(
    f"Figure aspect ratio : "
    f"{FIG_W / FIG_H:.3f}"
)

print()
print("Architecture audit : OK")
print("Legacy-label audit : OK")

print(
    "Topography label    : "
    "local orographic gradient"
)

print()
print("DONE")
