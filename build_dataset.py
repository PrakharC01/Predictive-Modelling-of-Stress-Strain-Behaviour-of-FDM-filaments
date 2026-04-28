"""
Steps 1-6: Load JSONs, parse filenames, compute mechanical properties, build dataset.
"""
import json
import os
import re
import numpy as np
import pandas as pd

DATA_DIR = "data/Datapoints"
OUT_CSV = "ml_pipeline/dataset.csv"


def parse_filename(fname: str):
    """
    Filename pattern: FDM_<MATERIAL>_<BRAND>_<WALLS>_<INFILL>.json
    Special case: FDM_<MATERIAL>_<BRAND>_SOLID_SOLID.json -> walls=NaN-like marker, infill=100
    Material may itself contain '+' (e.g. PLA+).
    Brand may be a single token (OVERTURE, DURAMIC, etc.).
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    # parts[0] should be FDM (process). The last two are walls/infill or SOLID/SOLID.
    process = parts[0]
    last2 = parts[-2:]
    middle = parts[1:-2]  # material + brand tokens
    if len(middle) < 2:
        material = middle[0] if middle else ""
        brand = ""
    else:
        # Heuristic: brand is the last token in middle, material is everything else joined.
        material = "_".join(middle[:-1])
        brand = middle[-1]

    if last2 == ["SOLID", "SOLID"]:
        is_solid = 1
        # treat solid specimens as fully dense single-block geometry
        # walls/infill are not meaningful here; encode infill as 100, walls as 0 sentinel
        walls = 0
        infill = 100
    else:
        is_solid = 0
        try:
            walls = int(last2[0])
            infill = int(last2[1])
        except ValueError:
            walls, infill = np.nan, np.nan

    return {
        "process": process,
        "material": material,
        "brand": brand,
        "walls": walls,
        "infill": infill,
        "is_solid": is_solid,
    }


def compute_properties(strain: np.ndarray, stress: np.ndarray):
    """
    Young's modulus: slope of the initial linear portion of the stress-strain curve.
    UTS: max stress.
    Toughness: area under curve up to failure (trapezoidal integration over full curve).
    """
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)

    # sort by strain in case data is unordered
    order = np.argsort(strain)
    strain = strain[order]
    stress = stress[order]

    # Young's modulus: steepest secant slope over the loading region.
    # The curves are downsampled to 100 points across the full strain-to-failure range,
    # so a fixed-fraction "first 20%" window can miss the actual elastic region.
    # Use a small sliding window (3 consecutive points) within the loading region
    # (up to peak stress) and take the maximum secant slope.
    peak_idx = int(np.argmax(stress))
    load_end = max(peak_idx, 3)
    win = 3
    best_slope = float("nan")
    if load_end >= win and len(strain) >= win:
        for i in range(load_end - win + 1):
            dx = strain[i + win - 1] - strain[i]
            dy = stress[i + win - 1] - stress[i]
            if dx > 0:
                s = dy / dx
                if np.isnan(best_slope) or s > best_slope:
                    best_slope = s
    youngs_modulus = float(best_slope)

    uts = float(np.max(stress))
    toughness = float(np.trapezoid(stress, strain))

    return youngs_modulus, uts, toughness


def main():
    rows = []
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    print(f"Found {len(files)} JSON files")

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            d = json.load(f)

        feat = parse_filename(fname)

        # Cross-check filename parse with JSON contents and prefer JSON when present.
        feat_from_json = {
            "process": d.get("process", feat["process"]),
            "material": d.get("material", feat["material"]),
            "brand": d.get("brand", feat["brand"]),
        }
        # walls / infill: prefer numeric JSON values when present
        json_walls = d.get("perim", None)
        json_infill = d.get("infill", None)

        walls = feat["walls"]
        infill = feat["infill"]
        if isinstance(json_walls, (int, float)) and not feat["is_solid"]:
            walls = int(json_walls)
        if isinstance(json_infill, (int, float)) and not feat["is_solid"]:
            infill = int(json_infill)

        infill_type = d.get("infill_type", None)
        cost = d.get("cost", None)
        layer_strength = d.get("layer_strength", None)

        stress = d.get("stress", [])
        strain = d.get("strain", [])
        if not stress or not strain or len(stress) != len(strain):
            print(f"Skipping {fname} (bad curves)")
            continue

        E, uts, toughness = compute_properties(strain, stress)

        rows.append({
            "file": fname,
            "process": feat_from_json["process"],
            "material": feat_from_json["material"],
            "brand": feat_from_json["brand"],
            "walls": walls,
            "infill": infill,
            "is_solid": feat["is_solid"],
            "infill_type": infill_type,
            "cost": cost,
            "layer_strength": layer_strength,
            "youngs_modulus": E,
            "uts": uts,
            "toughness": toughness,
        })

    df = pd.DataFrame(rows)
    print("\n=== Dataset (head) ===")
    print(df.head(10).to_string(index=False))
    print(f"\nShape: {df.shape}")

    print("\n=== Materials ===")
    print(df["material"].value_counts().to_string())
    print("\n=== Brands ===")
    print(df["brand"].value_counts().to_string())
    print("\n=== Walls ===")
    print(df["walls"].value_counts().sort_index().to_string())
    print("\n=== Infill ===")
    print(df["infill"].value_counts().sort_index().to_string())

    # Step 6: identify duplicate conditions
    cond_cols = ["process", "material", "brand", "walls", "infill", "is_solid"]
    dup_counts = df.groupby(cond_cols).size().reset_index(name="n_replicates")
    print("\n=== Replicates per condition ===")
    print(dup_counts.to_string(index=False))
    n_dup = (dup_counts["n_replicates"] > 1).sum()
    print(f"\nConditions with replicates: {n_dup}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
