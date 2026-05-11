"""
train_rating.py
---------------
Trains an XGBoost model to predict a user's likely star rating
for a product given their persona vector + category.

The model does NOT predict "what will this specific user rate this
specific product" — that requires product features we don't have.

It predicts: "given this user's personality and rating tendencies,
what star rating would they TYPICALLY give something in this category
when they interact with it?"

This is an ordinal regression problem (1-5 stars).

Input features:
  - OCEAN scores (5 dims) — personality profile
  - generosity_score      — derived rating tendency
  - rating_std            — how volatile their ratings are
  - pct_*_star (5 dims)   — their historical rating distribution
  - avg_review_length     — verbosity
  - category (encoded)    — which domain

Target:
  - avg_rating (float 1-5) — rounded to nearest 0.5 for ordinal bins

Output:
  - core/models/artifacts/rating_model.joblib
  - core/models/artifacts/rating_scaler.joblib
  - core/models/artifacts/rating_model_meta.json

Usage:
    python core/models/train_rating.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "data" / "processed"
ART_DIR   = Path(__file__).resolve().parent / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

PROFILES_PATH = DATA_DIR / "user_profiles.csv"
MODEL_PATH    = ART_DIR / "rating_model.joblib"
SCALER_PATH   = ART_DIR / "rating_scaler.joblib"
META_PATH     = ART_DIR / "rating_model_meta.json"

# ── feature config ────────────────────────────────────────────────────────────
OCEAN_FEATURES = ["ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N"]

# Remove ALL rating-derived features — these leak the target
# Keep only text behaviour and personality signals
TEXT_FEATURES = [
    "avg_review_length",
    "avg_token_count",
    "vocab_richness",
    "pos_word_ratio",
    "neg_word_ratio",
    "exclamation_ratio",
    "first_person_ratio",
    "certainty_ratio",
    "hedge_ratio",
]

ALL_FEATURES = OCEAN_FEATURES + TEXT_FEATURES + ["category_encoded"]
RATING_FEATURES = [
    "generosity_score",
    "rating_std",
    "pct_1_star",
    "pct_2_star",
    "pct_3_star",
    "pct_4_star",
    "pct_5_star",
]


TARGET = "rating"


# ── data prep ─────────────────────────────────────────────────────────────────

def load_and_prepare(path: Path) -> pd.DataFrame:
    """
    Loads user_profiles.csv and joins back to get individual
    review-level ratings as the target instead of avg_rating.
    Uses OCEAN scores (derived from text) to predict individual ratings.
    """
    profiles = pd.read_csv(path)
    print(f"  Loaded {len(profiles):,} user profiles")

    # load raw review files to get individual ratings
    raw_files = {
        "books":       ROOT / "data" / "raw" / "books_final.csv",
        "electronics": ROOT / "data" / "raw" / "electronics_final.csv",
        "movies":      ROOT / "data" / "raw" / "movies_final.csv",
    }

    frames = []
    for category, fpath in raw_files.items():
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df["category_name"] = category
        frames.append(df[["user_id", "rating", "category_name"]])

    reviews = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(reviews):,} individual reviews")

    # join profiles to individual reviews
    merged = reviews.merge(
        profiles[OCEAN_FEATURES + TEXT_FEATURES + ["user_id"]],
        on="user_id",
        how="inner"
    )
    print(f"  After join: {len(merged):,} rows")

    # encode category
    le = LabelEncoder()
    merged["category_encoded"] = le.fit_transform(merged["category_name"])

    merged = merged.dropna(subset=ALL_FEATURES + ["rating"])
    print(f"  After dropping nulls: {len(merged):,} rows")

    return merged, le

# ── training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, label_encoder: LabelEncoder):
    """
    Trains XGBoost regressor on the prepared dataset.
    Uses sample weights to account for category imbalance.
    """
    X = df[ALL_FEATURES].values
    y = df[TARGET].values

    # scale features
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split — stratify by rounded rating to keep distribution
    y_binned = np.round(y * 2) / 2  # round to nearest 0.5
    X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=0.15,
                random_state=42,
            )

    print(f"\n  Training set: {len(X_train):,} rows")
    print(f"  Test set:     {len(X_test):,} rows")
    print(f"  Rating distribution (train):")
    for star in [1.0, 2.0, 3.0, 4.0, 5.0]:
        pct = (y_train == star).mean() * 100
        if pct > 0:
            print(f"    {star:.0f} star: {pct:.1f}%")

    # XGBoost regressor
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # evaluate
    y_pred       = model.predict(X_test)
    y_pred_clip  = np.clip(y_pred, 1.0, 5.0)

    mae          = mean_absolute_error(y_test, y_pred_clip)
    within_1     = np.mean(np.abs(y_test - y_pred_clip) <= 1.0)
    within_half  = np.mean(np.abs(y_test - y_pred_clip) <= 0.5)

    # round predictions to nearest star for accuracy
    y_pred_round = np.round(y_pred_clip).clip(1, 5)
    y_test_round = np.round(y_test).clip(1, 5)
    exact_acc    = accuracy_score(y_test_round, y_pred_round)

    print(f"\n  ── Evaluation ───────────────────────────────")
    print(f"  MAE:                {mae:.4f} stars")
    print(f"  Within 0.5 stars:   {within_half*100:.1f}%")
    print(f"  Within 1.0 star:    {within_1*100:.1f}%")
    print(f"  Exact star match:   {exact_acc*100:.1f}%")

    # feature importance
    importances = model.feature_importances_
    feat_imp    = sorted(
        zip(ALL_FEATURES, importances),
        key=lambda x: -x[1]
    )
    print(f"\n  Top 5 features:")
    for feat, imp in feat_imp[:5]:
        print(f"    {feat:<30} {imp:.4f}")

    return model, scaler, {
        "mae":          round(float(mae), 4),
        "within_1_star": round(float(within_1), 4),
        "within_half_star": round(float(within_half), 4),
        "exact_accuracy": round(float(exact_acc), 4),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "features":     ALL_FEATURES,
        "categories":   label_encoder.classes_.tolist(),
        "target":       TARGET,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Rating Prediction Model — Training")
    print("=" * 55)

    print(f"\nLoading data from {PROFILES_PATH}...")
    df, le = load_and_prepare(PROFILES_PATH)

    print("\nTraining XGBoost model...")
    model, scaler, meta = train(df, le)

    # save artifacts
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ rating_model.joblib  → {MODEL_PATH}")
    print(f"✓ rating_scaler.joblib → {SCALER_PATH}")
    print(f"✓ rating_model_meta.json → {META_PATH}")
    print("\n── Day 4a complete — model trained and saved ────────")
    print("   Next → rating_predictor.py")


if __name__ == "__main__":
    main()