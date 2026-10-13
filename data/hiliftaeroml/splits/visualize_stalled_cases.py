"""Visualize stall onset across all HiLiftAeroML geometries.

For each of the 180 LHC geometries, loads CL at all 10 AoA values, fits
a cubic spline, and identifies the first AoA where dCL/dalpha <= 0
(stall onset). Everything from that AoA onward is marked post-stall.

Produces a scatter plot: x = geometry ID (1-180), y = AoA, colored
gray (pre-stall) or red (post-stall).

Usage:
    uv run splits/visualize_stalled_cases.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_splits import (
    AOA_ARRAY,
    AOA_VALUES,
    N_AOA,
    N_GEOMETRIES,
    build_stall_mask,
    detect_stall,
    load_cl_matrix,
)


def main() -> None:
    print("Loading CL data for 1800 cases...")
    cl = load_cl_matrix()
    print(f"  CL range: [{cl.min():.4f}, {cl.max():.4f}]")

    ### Detect stall onset for each geometry
    print("Detecting stall onset via cubic spline dCL/dalpha...")
    stall_mask = build_stall_mask()

    onset_aoas = [detect_stall(cl[i]) for i in range(N_GEOMETRIES)]
    n_stalled_geos = sum(1 for o in onset_aoas if o is not None)
    n_stalled_cases = int(stall_mask.sum())
    print(f"  {n_stalled_geos}/{N_GEOMETRIES} geometries stall within AoA range")
    print(f"  {n_stalled_cases}/{N_GEOMETRIES * N_AOA} total post-stall cases")

    ### Distribution of stall onset AoA
    stalled_indices = [o for o in onset_aoas if o is not None]
    if stalled_indices:
        print("\n  Stall onset AoA distribution:")
        for j, aoa in enumerate(AOA_VALUES):
            count = sum(1 for idx in stalled_indices if idx == j)
            bar = "█" * count
            print(f"    {aoa:>2d} deg: {count:>3d}  {bar}")

    ### Build scatter plot
    geo_nums = np.arange(1, N_GEOMETRIES + 1)
    aoa_grid, geo_grid = np.meshgrid(AOA_ARRAY, geo_nums)
    aoa_flat = aoa_grid.ravel()
    geo_flat = geo_grid.ravel()
    stall_flat = stall_mask.ravel()

    fig, ax = plt.subplots(figsize=(16, 5))

    pre = ~stall_flat
    post = stall_flat
    ax.scatter(
        geo_flat[pre], aoa_flat[pre],
        c="0.70", s=22, marker="o", edgecolors="none",
        label=f"Pre-stall ({int(pre.sum())})", zorder=2,
    )
    ax.scatter(
        geo_flat[post], aoa_flat[post],
        c="red", s=22, marker="o", edgecolors="none",
        label=f"Post-stall ({int(post.sum())})", zorder=3,
    )

    ax.set_xlabel("Geometry ID (LHC number)", fontsize=13)
    ax.set_ylabel("Angle of Attack (deg)", fontsize=13)
    ax.set_title(
        "HiLiftAeroML: Stall Onset Detection\n"
        r"Post-stall $\equiv$ first $\alpha$ where $dC_L/d\alpha \leq 0$ (cubic spline)",
        fontsize=13,
    )
    ax.set_yticks(AOA_ARRAY.astype(int))
    ax.set_ylim(2, 24)
    ax.set_xlim(0, N_GEOMETRIES + 1)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "stall_onset.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  Plot saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
