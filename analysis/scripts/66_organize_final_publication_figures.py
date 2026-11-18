#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()
PAPER = SCRIPT.parents[1]

FIG_DIR = (
    PAPER
    / "final_publication_figures"
)

OFFICIAL_DIR = (
    FIG_DIR
    / "official"
)

ARCHIVE_DIR = (
    FIG_DIR
    / "archive_versions"
)

MANIFEST = (
    OFFICIAL_DIR
    / "manifest.csv"
)


OFFICIAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARCHIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OFFICIAL FIGURES
#
# source stem -> clean final stem
# ============================================================

OFFICIAL = [
    (
        "Figure01_architecture",
        "Figure01_architecture",
    ),

    (
        "Figure02_spatial_rmse",
        "Figure02_spatial_rmse",
    ),

    (
        "Figure03_residual_refinement",
        "Figure03_residual_refinement",
    ),

    (
        "Figure04_fine_topography",
        "Figure04_fine_topography",
    ),

    (
        "Figure05_temporal_evolution",
        "Figure05_temporal_evolution",
    ),

    (
        "Figure06_variance_wasserstein",
        "Figure06_variance_wasserstein",
    ),

    (
        "Figure07_lag1_persistence",
        "Figure07_lag1_persistence",
    ),

    (
        "Figure08_long_term_trend_signal",
        "Figure08_long_term_trend_signal",
    ),

    (
        "Figure09_distribution_by_orographic_band_v6",
        "Figure09_distribution_by_orographic_band",
    ),

    (
        "Figure10_seasonal_extremes_v3",
        "Figure10_seasonal_thermal_tail_fidelity",
    ),
]


EXTENSIONS = [
    ".png",
    ".pdf",
]


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("ORGANIZE FINAL PUBLICATION FIGURES")
print("=" * 100)

print("Source:")
print(" ", FIG_DIR)

print()

print("Official:")
print(" ", OFFICIAL_DIR)

print()

print("Archive:")
print(" ", ARCHIVE_DIR)

print()


# ============================================================
# VERIFY OFFICIAL SOURCES FIRST
# ============================================================

missing = []

for source_stem, _ in OFFICIAL:

    for ext in EXTENSIONS:

        src = (
            FIG_DIR
            / f"{source_stem}{ext}"
        )

        if not src.exists():

            missing.append(
                src
            )


if missing:

    print("=" * 100)
    print("WARNING — MISSING EXPECTED FILES")
    print("=" * 100)

    for p in missing:
        print(
            "MISSING:",
            p.name,
        )

    print()
    print(
        "The script will continue with the files that exist."
    )
    print()


# ============================================================
# COPY OFFICIAL FIGURES
# ============================================================

manifest_rows = []

print("=" * 100)
print("OFFICIAL FIGURES")
print("=" * 100)


for fig_number, (
    source_stem,
    final_stem,
) in enumerate(
    OFFICIAL,
    start=1,
):

    print()
    print(
        f"Figure {fig_number:02d}"
    )

    for ext in EXTENSIONS:

        src = (
            FIG_DIR
            / f"{source_stem}{ext}"
        )

        dst = (
            OFFICIAL_DIR
            / f"{final_stem}{ext}"
        )


        if not src.exists():

            print(
                f"  SKIP {ext}: "
                f"{src.name} not found"
            )

            continue


        shutil.copy2(
            src,
            dst,
        )


        print(
            f"  {src.name}"
        )

        print(
            f"    -> {dst.name}"
        )


        manifest_rows.append({
            "figure":
                f"Figure {fig_number:02d}",

            "source_file":
                src.name,

            "official_file":
                dst.name,

            "format":
                ext.lstrip("."),
        })


# ============================================================
# ARCHIVE NON-OFFICIAL FIGURE VERSIONS
#
# COPY ONLY — nothing is deleted or moved.
# ============================================================

official_source_names = set()

for source_stem, _ in OFFICIAL:

    for ext in EXTENSIONS:

        official_source_names.add(
            f"{source_stem}{ext}"
        )


archive_candidates = []

for ext in EXTENSIONS:

    archive_candidates.extend(
        FIG_DIR.glob(
            f"Figure*{ext}"
        )
    )


archive_candidates = sorted(
    archive_candidates,
    key=lambda p:
        p.name.lower(),
)


print()
print("=" * 100)
print("ARCHIVING OTHER VERSIONS")
print("=" * 100)


n_archived = 0

for src in archive_candidates:

    if src.name in official_source_names:

        continue


    dst = (
        ARCHIVE_DIR
        / src.name
    )


    shutil.copy2(
        src,
        dst,
    )


    print(
        "  ",
        src.name,
    )

    n_archived += 1


# ============================================================
# MANIFEST
# ============================================================

with open(
    MANIFEST,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "figure",
            "source_file",
            "official_file",
            "format",
        ],
    )

    writer.writeheader()

    writer.writerows(
        manifest_rows
    )


# ============================================================
# FINAL AUDIT
# ============================================================

official_pngs = sorted(
    OFFICIAL_DIR.glob(
        "Figure*.png"
    )
)

official_pdfs = sorted(
    OFFICIAL_DIR.glob(
        "Figure*.pdf"
    )
)


print()
print("=" * 100)
print("FINAL AUDIT")
print("=" * 100)

print(
    f"Official PNG files : {len(official_pngs)}"
)

print(
    f"Official PDF files : {len(official_pdfs)}"
)

print(
    f"Archived versions  : {n_archived}"
)

print()


print("OFFICIAL SET:")
print()


for p in official_pngs:

    print(
        " ",
        p.name,
    )


print()
print("Manifest:")
print(
    " ",
    MANIFEST
)

print()


if (
    len(official_pngs) == 10
    and len(official_pdfs) == 10
):

    print(
        "OK — complete official set: "
        "10 PNG + 10 PDF."
    )

else:

    print(
        "ATTENTION — official set is incomplete. "
        "Review the missing-file warnings above."
    )


print()
print("DONE")
