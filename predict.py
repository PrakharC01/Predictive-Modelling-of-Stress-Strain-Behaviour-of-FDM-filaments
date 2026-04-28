"""
Step 12: Make predictions on new inputs.

Usage:
  python3 ml_pipeline/predict.py \\
      --material PLA --brand INLAND --walls 4 --infill 35 \\
      [--infill-type Tri] [--cost 18] [--solid]
"""
import argparse
import pandas as pd
from joblib import load

MODEL_PATH = "ml_pipeline/model.joblib"
TARGETS = ["youngs_modulus", "uts", "toughness"]
CAT_FEATURES = ["material", "brand", "infill_type"]
NUM_FEATURES = ["walls", "infill", "is_solid", "cost"]


def predict(material, brand, walls, infill, infill_type="Tri", cost=20, solid=False):
    model = load(MODEL_PATH)
    is_solid = 1 if solid else 0
    if solid:
        infill_type = "SOLID"
        walls = 0
        infill = 100
    row = pd.DataFrame([{
        "material": material,
        "brand": brand,
        "infill_type": infill_type,
        "walls": walls,
        "infill": infill,
        "is_solid": is_solid,
        "cost": cost,
    }])
    pred = model.predict(row[CAT_FEATURES + NUM_FEATURES])[0]
    return dict(zip(TARGETS, [float(x) for x in pred]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material", required=True)
    p.add_argument("--brand", required=True)
    p.add_argument("--walls", type=int, default=2)
    p.add_argument("--infill", type=int, default=35)
    p.add_argument("--infill-type", default="Tri")
    p.add_argument("--cost", type=float, default=20)
    p.add_argument("--solid", action="store_true",
                   help="Solid block (overrides walls/infill/infill-type).")
    args = p.parse_args()

    result = predict(args.material, args.brand, args.walls, args.infill,
                     args.infill_type, args.cost, args.solid)
    print("Inputs:")
    print(f"  material={args.material}  brand={args.brand}  walls={args.walls}  "
          f"infill={args.infill}  infill_type={args.infill_type}  solid={args.solid}")
    print("Predictions:")
    for k, v in result.items():
        print(f"  {k:<16}  {v:8.3f}")


if __name__ == "__main__":
    main()
