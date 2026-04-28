"""
Steps 7-13: Encode features, train models, validate (group-aware),
evaluate, save, demo predictions, feature importances.
"""
import json
import warnings
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DATA_CSV = "ml_pipeline/dataset.csv"
MODEL_PATH = "ml_pipeline/model.joblib"
METRICS_PATH = "ml_pipeline/metrics.json"

CAT_FEATURES = ["material", "brand", "infill_type"]
NUM_FEATURES = ["walls", "infill", "is_solid", "cost"]
TARGETS = ["youngs_modulus", "uts", "toughness"]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )


def make_models():
    return {
        "Ridge": Pipeline([("pre", make_preprocessor()),
                           ("reg", MultiOutputRegressor(Ridge(alpha=1.0)))]),
        "RandomForest": Pipeline([("pre", make_preprocessor()),
                                  ("reg", RandomForestRegressor(
                                      n_estimators=400,
                                      max_depth=None,
                                      min_samples_leaf=1,
                                      random_state=42,
                                      n_jobs=-1))]),
        "GradientBoosting": Pipeline([("pre", make_preprocessor()),
                                      ("reg", MultiOutputRegressor(
                                          GradientBoostingRegressor(
                                              n_estimators=300,
                                              learning_rate=0.05,
                                              max_depth=3,
                                              random_state=42)))]),
    }


def cv_eval(model, X, y, groups, cv, label):
    """Run cross-validation; return per-target MAE / RMSE / R²."""
    per_target = {t: {"mae": [], "rmse": [], "r2": []} for t in TARGETS}
    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        for j, t in enumerate(TARGETS):
            yt, yp = y.iloc[te, j].values, pred[:, j]
            per_target[t]["mae"].append(mean_absolute_error(yt, yp))
            per_target[t]["rmse"].append(np.sqrt(mean_squared_error(yt, yp)))
            per_target[t]["r2"].append(r2_score(yt, yp))
    summary = {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in per_target.items()}
    return summary


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} samples")

    # Drop rows with NaN targets just in case
    df = df.dropna(subset=TARGETS).reset_index(drop=True)

    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    y = df[TARGETS].copy()
    groups = df["brand"].values  # group-aware CV: hold out entire brands

    # Validation strategies
    n_brands = df["brand"].nunique()
    gkf = GroupKFold(n_splits=min(5, n_brands))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== Cross-validation results ===")
    print(f"Random KFold (n=5) — may leak across brands")
    print(f"GroupKFold by brand (n={gkf.get_n_splits()}) — generalization to unseen brands\n")

    all_metrics = {}
    for name, model in make_models().items():
        kfold_metrics = cv_eval(model, X, y, None, kf, name)
        gkf_metrics = cv_eval(model, X, y, groups, gkf, name)
        all_metrics[name] = {"kfold": kfold_metrics, "groupkfold_brand": gkf_metrics}

        print(f"--- {name} ---")
        for t in TARGETS:
            k = kfold_metrics[t]; g = gkf_metrics[t]
            print(f"  {t:>16}  KFold: MAE={k['mae']:.3f}  RMSE={k['rmse']:.3f}  R²={k['r2']:.3f}   "
                  f"|  GroupKFold(brand): MAE={g['mae']:.3f}  RMSE={g['rmse']:.3f}  R²={g['r2']:.3f}")
        print()

    # Pick best model by averaged R² over targets under random KFold (in-distribution).
    def avg_r2(metrics, kind):
        return np.mean([metrics[kind][t]["r2"] for t in TARGETS])
    best_name = max(all_metrics, key=lambda n: avg_r2(all_metrics[n], "kfold"))
    print(f"Best in-distribution model: {best_name}")
    print(f"Best out-of-brand   model: "
          f"{max(all_metrics, key=lambda n: avg_r2(all_metrics[n], 'groupkfold_brand'))}")

    # Step 11: refit best on full data and save
    best_model = make_models()[best_name]
    best_model.fit(X, y)
    dump(best_model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump({"models": all_metrics, "best_model": best_name}, f, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

    # Step 13 (partial): feature importances if available
    reg = best_model.named_steps["reg"]
    pre = best_model.named_steps["pre"]
    feat_names = list(pre.get_feature_names_out())
    print("\n=== Feature importances (best model, per target if available) ===")
    if hasattr(reg, "feature_importances_"):
        # Single multi-output regressor (e.g. RF)
        imps = reg.feature_importances_
        order = np.argsort(imps)[::-1][:15]
        for i in order:
            print(f"  {feat_names[i]:<35}  {imps[i]:.4f}")
    elif hasattr(reg, "estimators_"):
        for j, est in enumerate(reg.estimators_):
            if hasattr(est, "feature_importances_"):
                imps = est.feature_importances_
            elif hasattr(est, "coef_"):
                imps = np.abs(est.coef_)
            else:
                continue
            order = np.argsort(imps)[::-1][:8]
            print(f"\n  -> Target: {TARGETS[j]}")
            for i in order:
                print(f"     {feat_names[i]:<35}  {imps[i]:.4f}")

    # Step 12: example predictions
    print("\n=== Example predictions ===")
    examples = pd.DataFrame([
        # PLA, INLAND brand, 4 walls, 35% infill (Tri infill)
        {"material": "PLA",  "brand": "INLAND",   "infill_type": "Tri",
         "walls": 4, "infill": 35, "is_solid": 0, "cost": 18},
        # PLA+, ESUN brand, 6 walls, 35% infill
        {"material": "PLA+", "brand": "ESUN",     "infill_type": "Tri",
         "walls": 6, "infill": 35, "is_solid": 0, "cost": 22},
        # ASA, OVERTURE, solid block
        {"material": "ASA",  "brand": "OVERTURE", "infill_type": "SOLID",
         "walls": 0, "infill": 100, "is_solid": 1, "cost": 29},
    ])
    preds = best_model.predict(examples[CAT_FEATURES + NUM_FEATURES])
    out = examples.copy()
    for j, t in enumerate(TARGETS):
        out[t + "_pred"] = preds[:, j].round(3)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
