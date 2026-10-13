"""Generate deterministic train/val/test splits for the HiLiftAeroML dataset.

Produces a manifest.json containing 14 split types, each with
train/val/test keys:

  1. full            - random case-level 70/10/20 split
  2. medium          - same val/test as full, train has 510 nested cases
  3. scarce          - same val/test as full, train is 1/6 subsample
  4. super_scarce    - same val/test as full, train is 1/36 subsample
  5. geometry        - hold out 36 random geometries for test
  6. geometry_medium - 51 complete train geometries, same geometry val/test
  7. geometry_scarce - 21 complete train geometries, same geometry val/test
  8. geometry_super_scarce - 4 complete train geometries, same geometry val/test
  9. aoa             - train on AoA <= 12, test on AoA >= 14
 10. deflection      - train on low-deflection geometries, test on top 20%
 11. stall           - train on pre-stall, test on post-stall (per-geometry)
 12. single_aoa_4    - per-AoA geometry split at 4 deg (pre-stall)
 13. single_aoa_12   - per-AoA geometry split at 12 deg (mid-range)
 14. single_aoa_22   - per-AoA geometry split at 22 deg (post-stall)

For every split, the validation set is drawn from the **same distribution
as training** so that hyperparameter tuning never sees out-of-distribution
data.

Usage:
    uv run splits/generate_splits.py
"""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

### ──── Dataset constants ────

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset"
GEOMETRY_IDS = [f"LHC{i:03d}" for i in range(1, 181)]
AOA_VALUES = list(range(4, 23, 2))  # [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
AOA_ARRAY = np.asarray(AOA_VALUES, dtype=float)
N_GEOMETRIES = len(GEOMETRY_IDS)    # 180
N_AOA = len(AOA_VALUES)             # 10
N_CASES = N_GEOMETRIES * N_AOA      # 1800


### ──── Split parameters ────

SEED = 42
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
VAL_FRACTION_OF_POOL = VAL_FRACTION / (1 - TEST_FRACTION)

N_TEST_GEOS = round(N_GEOMETRIES * TEST_FRACTION)                        # 36
N_VAL_GEOS = round((N_GEOMETRIES - N_TEST_GEOS) * VAL_FRACTION_OF_POOL)  # 18
N_TRAIN_GEOS = N_GEOMETRIES - N_TEST_GEOS - N_VAL_GEOS                   # 126

SCARCE_FRACTION = 1 / 6         # fraction of full_train for scarce_train
SUPER_SCARCE_FRACTION = 1 / 36  # fraction of full_train for super_scarce_train
N_MEDIUM_CASES = 510             # near geometric midpoint of 1260 and 210

# Nested whole-geometry data-efficiency ladder. Every selected geometry keeps
# all 10 AoA cases, so geometry coverage is varied without also varying the
# within-geometry AoA trajectory.
N_GEOMETRY_MEDIUM = 51
N_GEOMETRY_SCARCE = 21
N_GEOMETRY_SUPER_SCARCE = 4

# AoA split: train/val on pre-stall regime, test on stall/post-stall.
# The 12/14 cutoff is physics-motivated: the paper documents a regime
# transition around 14-16 deg where dominant aerodynamic sensitivity
# shifts from flap deflection (camber) to slat deflection (leading-edge
# separation control).
AOA_TRAIN = [4, 6, 8, 10, 12]
AOA_TEST = [14, 16, 18, 20, 22]

# Deflection parameters used to compute mean deflection per geometry
DEFLECTION_PARAMS = [
    "IB_Flap_Deflection", "OB_Flap_Deflection",
    "IB_Slat_Deflection", "OB_Slat_Deflection",
]

# Per-AoA evaluation: pre-stall / mid-range / post-stall
PER_AOA_VALUES = [4, 12, 22]


### ──── Helpers ────


def case_id(geo: str, aoa: int) -> str:
    """Construct a case ID matching the on-disk directory name."""
    return f"geo_{geo}_AoA_{aoa}"


def case_sort_key(cid: str) -> tuple[int, int]:
    """Sort key giving numerical order: (geometry_number, aoa)."""
    parts = cid.split("_")          # ["geo", "LHC042", "AoA", "12"]
    return int(parts[1][3:]), int(parts[3])


def make_case_ids(geos: list[str], aoas: list[int]) -> list[str]:
    """Generate sorted case IDs for all (geometry, AoA) combinations."""
    return sorted(
        [case_id(g, a) for g in geos for a in aoas],
        key=case_sort_key,
    )


def _rng(salt: str) -> random.Random:
    """Create a deterministic RNG independent of other splits.

    Each split derives its own seed from the master SEED and a salt string,
    so adding or modifying one split never affects another.
    """
    seed_bytes = hashlib.sha256(f"{SEED}:{salt}".encode()).digest()[:8]
    return random.Random(int.from_bytes(seed_bytes, "big"))


def _split_pool(
    pool: list[str], *, salt: str,
) -> tuple[list[str], list[str]]:
    """Split a case-level pool into (train, val) by random subsample.

    Holds out VAL_FRACTION_OF_POOL of the pool as val, returns the rest
    as train. Both lists are returned sorted by case_sort_key.
    """
    rng = _rng(salt)
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    n_val = round(len(pool) * VAL_FRACTION_OF_POOL)
    val = sorted(shuffled[:n_val], key=case_sort_key)
    train = sorted(shuffled[n_val:], key=case_sort_key)
    return train, val


### ──── Core: geometry selection ────


def select_geometry_splits() -> tuple[list[str], list[str], list[str]]:
    """Select train/val/test geometries, returning (train, val, test).

    The val and test geometry sets are shared across the `geometry` split
    and all `single_aoa_*` splits, enabling direct comparison across AoA
    regimes on identical held-out geometries.
    """
    rng = _rng("geometry_selection")
    shuffled = GEOMETRY_IDS.copy()
    rng.shuffle(shuffled)
    test_geos = sorted(shuffled[:N_TEST_GEOS])
    val_geos = sorted(shuffled[N_TEST_GEOS : N_TEST_GEOS + N_VAL_GEOS])
    train_geos = sorted(shuffled[N_TEST_GEOS + N_VAL_GEOS :])
    return train_geos, val_geos, test_geos


def order_training_geometries(train_geos: list[str]) -> list[str]:
    """Return one deterministic priority ordering of training geometries.

    Prefixes of this ordering define the nested geometry data-efficiency
    ladder without changing the existing geometry train/val/test assignment.
    """
    ordered = train_geos.copy()
    _rng("geometry_data_efficiency").shuffle(ordered)
    return ordered


### ──── Stall detection ────


def load_cl_matrix() -> np.ndarray:
    """Load CL for all geometries and AoA from per-case force_mom CSVs.

    Returns:
        Array of shape (N_GEOMETRIES, N_AOA) where entry [i, j] is the lift
        coefficient for geometry i at AOA_VALUES[j].
    """
    cl = np.zeros((N_GEOMETRIES, N_AOA))
    for i, geo in enumerate(GEOMETRY_IDS):
        for j, aoa in enumerate(AOA_VALUES):
            case = f"geo_{geo}_AoA_{aoa}"
            csv_path = DATA_ROOT / case / f"force_mom_{case}.csv"
            with open(csv_path) as f:
                row = next(csv.DictReader(f))
            cl[i, j] = float(row["cl"])
    return cl


def detect_stall(cl_row: np.ndarray) -> int | None:
    """Find the first AoA index where dCL/dalpha <= 0 via cubic spline.

    Fits a cubic spline to CL(alpha), differentiates it analytically,
    and evaluates at each data point to find the onset of stall.

    Args:
        cl_row: CL values at each of the N_AOA data points for one geometry.

    Returns:
        Index into AOA_VALUES of stall onset, or None if CL is monotonically
        increasing (no stall detected within the AoA range).
    """
    dcl_dalpha = CubicSpline(AOA_ARRAY, cl_row).derivative()(AOA_ARRAY)
    nonpositive = np.where(dcl_dalpha <= 0.0)[0]
    if len(nonpositive) == 0:
        return None
    return int(nonpositive[0])


def build_stall_mask() -> np.ndarray:
    """Build a boolean mask marking post-stall cases.

    For each geometry, the stall onset AoA is the first angle where
    dCL/dalpha <= 0 (from the cubic spline fit). Everything from that
    AoA onward is marked post-stall.

    Returns:
        Boolean array of shape (N_GEOMETRIES, N_AOA), True = post-stall.
        Row order matches GEOMETRY_IDS, column order matches AOA_VALUES.
    """
    cl = load_cl_matrix()
    mask = np.zeros((N_GEOMETRIES, N_AOA), dtype=bool)
    for i in range(N_GEOMETRIES):
        idx = detect_stall(cl[i])
        if idx is not None:
            mask[i, idx:] = True
    return mask


### ──── Deflection analysis ────


def load_mean_deflections() -> dict[str, float]:
    """Compute mean deflection angle for each geometry from the master CSV.

    Averages the 4 deflection angles (IB/OB flap and slat) per geometry.
    Gap multipliers are excluded since they are dimensionless scale factors,
    not angular deflections.

    Returns:
        Dict mapping geometry ID (e.g. "LHC001") to mean deflection in degrees.
    """
    csv_path = DATA_ROOT / "geo_parameters_all.csv"
    lhc_set = set(GEOMETRY_IDS)
    result: dict[str, float] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            geo = row["GeoID"]
            if geo in lhc_set:
                angles = [float(row[p]) for p in DEFLECTION_PARAMS]
                result[geo] = sum(angles) / len(angles)
    return result


### ──── Split generation ────


def generate_splits() -> dict[str, list[str]]:
    """Generate all 14 split types with train/val/test keys.

    Returns:
        Dict mapping split keys to sorted lists of case ID strings.
        Keys follow the pattern ``{split_name}_{train|val|test}``.
    """
    train_geos, val_geos, test_geos = select_geometry_splits()
    splits: dict[str, list[str]] = {}

    ### 1. Full random case-level split (70/10/20)
    rng = _rng("full_case_shuffle")
    all_cases = make_case_ids(GEOMETRY_IDS, AOA_VALUES)
    shuffled = all_cases.copy()
    rng.shuffle(shuffled)
    n_test = round(N_CASES * TEST_FRACTION)                       # 360
    n_val = round((N_CASES - n_test) * VAL_FRACTION_OF_POOL)      # 180
    splits["full_train"] = sorted(shuffled[n_test + n_val :], key=case_sort_key)
    splits["full_val"] = sorted(shuffled[n_test : n_test + n_val], key=case_sort_key)
    splits["full_test"] = sorted(shuffled[:n_test], key=case_sort_key)

    ### 2-4. Case-level data-efficiency splits
    # Same val/test as full; train is a subsample of full_train.
    # super_scarce ⊂ scarce ⊂ medium ⊂ full by construction: a single
    # shuffle determines the priority order, and each level takes a prefix.
    rng_scarce = _rng("scarce_subsample")
    full_train_shuffled = splits["full_train"].copy()
    rng_scarce.shuffle(full_train_shuffled)
    n_scarce = round(len(splits["full_train"]) * SCARCE_FRACTION)
    n_super_scarce = round(len(splits["full_train"]) * SUPER_SCARCE_FRACTION)

    splits["medium_train"] = sorted(full_train_shuffled[:N_MEDIUM_CASES], key=case_sort_key)
    splits["medium_val"] = splits["full_val"]
    splits["medium_test"] = splits["full_test"]

    splits["scarce_train"] = sorted(full_train_shuffled[:n_scarce], key=case_sort_key)
    splits["scarce_val"] = splits["full_val"]
    splits["scarce_test"] = splits["full_test"]

    splits["super_scarce_train"] = sorted(full_train_shuffled[:n_super_scarce], key=case_sort_key)
    splits["super_scarce_val"] = splits["full_val"]
    splits["super_scarce_test"] = splits["full_test"]

    ### 5. Geometry-level split (126/18/36 geometries)
    splits["geometry_train"] = make_case_ids(train_geos, AOA_VALUES)
    splits["geometry_val"] = make_case_ids(val_geos, AOA_VALUES)
    splits["geometry_test"] = make_case_ids(test_geos, AOA_VALUES)

    ### 6-8. Whole-geometry data-efficiency splits
    # Reuse the geometry split's held-out val/test geometries and take nested
    # prefixes from one deterministic ordering of geometry_train. All 10 AoAs
    # are retained for each selected training geometry.
    ordered_train_geos = order_training_geometries(train_geos)
    geometry_levels = {
        "geometry_medium": N_GEOMETRY_MEDIUM,
        "geometry_scarce": N_GEOMETRY_SCARCE,
        "geometry_super_scarce": N_GEOMETRY_SUPER_SCARCE,
    }
    for name, n_geos in geometry_levels.items():
        splits[f"{name}_train"] = make_case_ids(ordered_train_geos[:n_geos], AOA_VALUES)
        splits[f"{name}_val"] = splits["geometry_val"]
        splits[f"{name}_test"] = splits["geometry_test"]

    ### 9. AoA extrapolation (low → high)
    # Val is drawn from the pre-stall pool (same distribution as train).
    aoa_pool = make_case_ids(GEOMETRY_IDS, AOA_TRAIN)
    aoa_train, aoa_val = _split_pool(aoa_pool, salt="aoa_val_shuffle")
    splits["aoa_train"] = aoa_train
    splits["aoa_val"] = aoa_val
    splits["aoa_test"] = make_case_ids(GEOMETRY_IDS, AOA_TEST)

    ### 10. Deflection-based geometry split
    # Sort geometries by mean deflection; train/val on the bottom 80%,
    # test on the top 20%. Val geos are a random subset of the low-
    # deflection pool (same distribution as train).
    mean_defls = load_mean_deflections()
    sorted_by_defl = sorted(GEOMETRY_IDS, key=lambda g: mean_defls[g])
    defl_test_geos = sorted_by_defl[N_GEOMETRIES - N_TEST_GEOS :]     # top 36
    defl_pool_geos = sorted_by_defl[: N_GEOMETRIES - N_TEST_GEOS]     # bottom 144

    rng_defl = _rng("deflection_val_selection")
    pool_shuffled = defl_pool_geos.copy()
    rng_defl.shuffle(pool_shuffled)
    defl_train_geos = pool_shuffled[N_VAL_GEOS:]
    defl_val_geos = pool_shuffled[:N_VAL_GEOS]

    splits["deflection_train"] = make_case_ids(defl_train_geos, AOA_VALUES)
    splits["deflection_val"] = make_case_ids(defl_val_geos, AOA_VALUES)
    splits["deflection_test"] = make_case_ids(defl_test_geos, AOA_VALUES)

    ### 12-14. Per-AoA geometry splits (same geo split as geometry)
    for aoa in PER_AOA_VALUES:
        splits[f"single_aoa_{aoa}_train"] = make_case_ids(train_geos, [aoa])
        splits[f"single_aoa_{aoa}_val"] = make_case_ids(val_geos, [aoa])
        splits[f"single_aoa_{aoa}_test"] = make_case_ids(test_geos, [aoa])

    ### 11. Stall-based split (per-geometry, data-driven)
    # Val is drawn from the pre-stall pool (same distribution as train).
    stall_mask = build_stall_mask()
    all_cases = make_case_ids(GEOMETRY_IDS, AOA_VALUES)
    stall_pool, stall_test = [], []
    for cid, is_stalled in zip(all_cases, stall_mask.ravel()):
        (stall_test if is_stalled else stall_pool).append(cid)
    stall_train, stall_val = _split_pool(stall_pool, salt="stall_val_shuffle")
    splits["stall_train"] = stall_train
    splits["stall_val"] = stall_val
    splits["stall_test"] = stall_test

    return splits


### ──── Validation ────


def validate_splits(splits: dict[str, list[str]]) -> None:
    """Verify structural correctness of all generated splits.

    Checks: pairwise disjointness of train/val/test, correct totals,
    matching held-out geometries between geometry-level splits.
    """
    split_names = sorted({k.rsplit("_", 1)[0] for k in splits})

    for name in split_names:
        train_set = set(splits[f"{name}_train"])
        val_set = set(splits[f"{name}_val"])
        test_set = set(splits[f"{name}_test"])

        assert not (train_set & val_set), f"{name}: train/val overlap"
        assert not (train_set & test_set), f"{name}: train/test overlap"
        assert not (val_set & test_set), f"{name}: val/test overlap"

        for cid in train_set | val_set | test_set:
            parts = cid.split("_")
            assert len(parts) == 4 and parts[0] == "geo" and parts[2] == "AoA", (
                f"Malformed case ID: {cid!r}"
            )

    ### Total sizes (partitioning splits should sum to N_CASES)
    for prefix in ["full", "geometry", "aoa", "deflection", "stall"]:
        total = (
            len(splits[f"{prefix}_train"])
            + len(splits[f"{prefix}_val"])
            + len(splits[f"{prefix}_test"])
        )
        assert total == N_CASES, f"{prefix}: expected {N_CASES} total, got {total}"

    ### Case-level data efficiency: super_scarce ⊂ scarce ⊂ medium ⊂ full
    assert set(splits["super_scarce_train"]) < set(splits["scarce_train"]), (
        "super_scarce_train must be a proper subset of scarce_train"
    )
    assert set(splits["scarce_train"]) < set(splits["medium_train"]), (
        "scarce_train must be a proper subset of medium_train"
    )
    assert set(splits["medium_train"]) < set(splits["full_train"]), (
        "medium_train must be a proper subset of full_train"
    )
    for prefix in ["medium", "scarce", "super_scarce"]:
        assert splits[f"{prefix}_val"] == splits["full_val"], (
            f"{prefix}_val must be identical to full_val"
        )
        assert splits[f"{prefix}_test"] == splits["full_test"], (
            f"{prefix}_test must be identical to full_test"
        )

    ### Whole-geometry data efficiency: preserve complete AoA trajectories.
    geometry_ladder = [
        ("geometry_super_scarce", N_GEOMETRY_SUPER_SCARCE),
        ("geometry_scarce", N_GEOMETRY_SCARCE),
        ("geometry_medium", N_GEOMETRY_MEDIUM),
        ("geometry", N_TRAIN_GEOS),
    ]
    for (smaller, expected_geos), (larger, _) in zip(
        geometry_ladder, geometry_ladder[1:]
    ):
        assert set(splits[f"{smaller}_train"]) < set(splits[f"{larger}_train"]), (
            f"{smaller}_train must be a proper subset of {larger}_train"
        )
        selected_geos = {
            cid.split("_AoA_")[0].removeprefix("geo_")
            for cid in splits[f"{smaller}_train"]
        }
        assert len(selected_geos) == expected_geos, (
            f"{smaller}: expected {expected_geos} training geometries, "
            f"got {len(selected_geos)}"
        )
        assert len(splits[f"{smaller}_train"]) == expected_geos * N_AOA, (
            f"{smaller}: every selected geometry must retain all {N_AOA} AoAs"
        )
    for prefix, _ in geometry_ladder[:-1]:
        assert splits[f"{prefix}_val"] == splits["geometry_val"], (
            f"{prefix}_val must be identical to geometry_val"
        )
        assert splits[f"{prefix}_test"] == splits["geometry_test"], (
            f"{prefix}_test must be identical to geometry_test"
        )

    for aoa in PER_AOA_VALUES:
        n = (
            len(splits[f"single_aoa_{aoa}_train"])
            + len(splits[f"single_aoa_{aoa}_val"])
            + len(splits[f"single_aoa_{aoa}_test"])
        )
        assert n == N_GEOMETRIES, f"single_aoa_{aoa}: expected {N_GEOMETRIES}, got {n}"

    ### Val and test geometries are shared across geometry-level splits
    for role in ["val", "test"]:
        geo_set = {
            cid.split("_AoA_")[0].removeprefix("geo_")
            for cid in splits[f"geometry_{role}"]
        }
        for aoa in PER_AOA_VALUES:
            per_aoa_set = {
                cid.split("_AoA_")[0].removeprefix("geo_")
                for cid in splits[f"single_aoa_{aoa}_{role}"]
            }
            assert per_aoa_set == geo_set, (
                f"single_aoa_{aoa} {role} geometries differ from geometry split"
            )


### ──── Main ────


def main() -> None:
    splits = generate_splits()
    validate_splits(splits)

    ### Summary header
    print("HiLiftAeroML Splits")
    print("=" * 60)
    print(f"  Dataset: {N_GEOMETRIES} geometries x {N_AOA} AoA = {N_CASES} cases")
    print(f"  Seed: {SEED}")
    print()

    ### Geometry splits
    train_geos, val_geos, test_geos = select_geometry_splits()
    print(f"  Test geometries ({len(test_geos)}):")
    for row_start in range(0, len(test_geos), 9):
        row = test_geos[row_start : row_start + 9]
        print(f"    {', '.join(row)}")
    print(f"  Val geometries ({len(val_geos)}):")
    for row_start in range(0, len(val_geos), 9):
        row = val_geos[row_start : row_start + 9]
        print(f"    {', '.join(row)}")
    print()

    ### Split sizes
    split_names = sorted({k.rsplit("_", 1)[0] for k in splits})
    print(f"  {'Split':<28s} {'Train':>6s} {'Val':>6s} {'Test':>6s} {'Total':>6s}")
    print(f"  {'-' * 56}")
    for name in split_names:
        n_train = len(splits[f"{name}_train"])
        n_val = len(splits[f"{name}_val"])
        n_test = len(splits[f"{name}_test"])
        total = n_train + n_val + n_test
        print(f"  {name:<28s} {n_train:>6d} {n_val:>6d} {n_test:>6d} {total:>6d}")
    print()

    ### Write manifest
    output = Path(__file__).parent / "manifest.json"
    output.write_text(json.dumps(splits, indent=4) + "\n")
    print(f"  Manifest: {output}")
    print(f"  Keys: {len(splits)}")
    print()
    print("All validations passed.")


if __name__ == "__main__":
    main()
