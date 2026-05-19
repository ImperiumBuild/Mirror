"""
rating_predictor.py
-------------------
Loads the trained XGBoost rating model and exposes a clean
predict() interface for the review engine to call at runtime.

Given a user's persona profile (from profile_builder.py) and
a target category, returns a predicted star rating (1-5) and
a confidence note.

Usage:
    from core.models.rating_predictor import RatingPredictor

    predictor = RatingPredictor()  # loads model once

    prediction = predictor.predict(
        profile={
            "ocean": {"O": 0.72, "C": 0.45, "E": 0.60, "A": 0.80, "N": 0.25},
            "liwc_features": {
                "avg_review_length": 180,
                "avg_token_count": 35,
                "vocab_richness": 0.65,
                "pos_word_ratio": 0.04,
                "neg_word_ratio": 0.01,
                "exclamation_rate": 0.02,
                "first_person_ratio": 0.05,
                "certainty_ratio": 0.01,
                "hedge_ratio": 0.01,
            },
            "rating_calibration": {"generosity_score": 0.75},
        },
        category="movies",
    )
    # {"predicted_rating": 4, "confidence": "high", "raw_score": 4.23}
"""

from __future__ import annotations

import json
import numpy as np
import joblib
from pathlib import Path
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version warnings from models trained on different sklearn versions
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# Suppress transformers tokenization warning
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

from core.ocean.Liwc import analyse_chosen_reviews

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
ART_DIR   = _HERE / "artifacts"
MODEL_PATH  = ART_DIR / "rating_model.joblib"
SCALER_PATH = ART_DIR / "rating_scaler.joblib"
META_PATH   = ART_DIR / "rating_model_meta.json"

# ── category encoding ─────────────────────────────────────────────────────────
# Must match the LabelEncoder used during training.
# Order: books=0, electronics=1, movies=2
CATEGORY_ENCODING = {
    "books":       0,
    "electronics": 1,
    "movies":      2,
    "apps":        1,   # map apps → electronics (closest domain)
    "products":    1,   # map generic products → electronics
    "restaurants": 2,   # map restaurants → movies (closest: entertainment)
    "food":        2,
}

# Feature order must match ALL_FEATURES in train_rating.py exactly
OCEAN_FEATURES = ["ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N"]
TEXT_FEATURES  = [
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

# ── predictor ─────────────────────────────────────────────────────────────────

class RatingPredictor:
    """
    Loads the trained rating model once and exposes predict().
    Designed to be instantiated once and reused across requests.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Rating model not found at {MODEL_PATH}. "
                "Run core/models/train_rating.py first."
            )

        self._model  = joblib.load(MODEL_PATH)
        self._scaler = joblib.load(SCALER_PATH)

        with open(META_PATH) as f:
            self._meta = json.load(f)

        print(f"RatingPredictor loaded — "
              f"MAE={self._meta.get('mae', '?')} stars, "
              f"within_1_star={self._meta.get('within_1_star', '?'):.0%}")

    def predict(self, profile: dict, category: str, product_name: str | None = None) -> dict:
    
        # Primary signal — user's own calibration average
        calibration  = profile.get("rating_calibration", {})
        avg_given    = calibration.get("avg_given_rating", None)
        generosity   = calibration.get("generosity_score", 0.5)

        # ML prediction as secondary signal
        feature_vector = self._build_feature_vector(profile, category)
        scaled_vector  = self._scaler.transform([feature_vector])
        ml_score       = float(self._model.predict(scaled_vector)[0])

        if avg_given is not None:
            # blend: 60% user's own avg, 40% ML
            raw_score = (avg_given * 0.60) + (ml_score * 0.40)
        else:
            raw_score = ml_score

        raw_clipped      = max(1.0, min(5.0, raw_score))
        predicted_rating = int(round(raw_clipped))

        # if review tone is positive, don't predict below 3
        user_review = profile.get("user_review", "")
        if user_review:
            features = analyse_chosen_reviews([user_review])
            if features.get("pos_affect_ratio", 0) > features.get("neg_affect_ratio", 0):
                raw_clipped = max(3.0, raw_clipped)
                predicted_rating = int(round(raw_clipped))

        distance         = abs(raw_clipped - round(raw_clipped))
        confidence       = "high" if distance >= 0.35 else "medium" if distance >= 0.20 else "low"
        reasoning        = self._build_reasoning(profile, category, predicted_rating, raw_clipped)

        return {
            "predicted_rating": predicted_rating,
            "raw_score":        round(raw_clipped, 3),
            "confidence":       confidence,
            "reasoning":        reasoning,
        }
    def predict_from_ocean(
        self,
        ocean: dict[str, float],
        category: str,
        liwc_features: dict | None = None,
    ) -> dict:
        """
        Lighter interface — predict directly from OCEAN scores
        without needing the full profile JSON.
        Useful for quick lookups during recommendations.
        """
        profile = {
            "ocean": ocean,
            "liwc_features": liwc_features or {},
        }
        return self.predict(profile, category)

    # ── private ───────────────────────────────────────────────────────────────

    def _build_feature_vector(
        self,
        profile: dict,
        category: str,
    ) -> list[float]:
        """
        Assembles the feature vector in the exact order
        the model was trained on.
        """
        ocean        = profile.get("ocean", {})
        liwc         = profile.get("liwc_features", {})
        cat_encoded  = CATEGORY_ENCODING.get(category.lower(), 1)

        # OCEAN features
        ocean_vals = [ocean.get(k, 0.5) for k in ["O", "C", "E", "A", "N"]]

        # text features — use liwc_features if available, else neutral defaults
        text_vals = [
            liwc.get("avg_review_length",  150.0),
            liwc.get("avg_token_count",     28.0),
            liwc.get("vocab_richness",       0.60),
            liwc.get("pos_word_ratio",       0.04),
            liwc.get("neg_word_ratio",       0.01),
            liwc.get("exclamation_rate",     0.01),
            liwc.get("first_person_ratio",   0.05),
            liwc.get("certainty_ratio",      0.01),
            liwc.get("hedge_ratio",          0.01),
        ]

        return ocean_vals + text_vals + [cat_encoded]

    def _build_reasoning(
        self,
        profile: dict,
        category: str,
        predicted: int,
        raw: float,
        source: str = "ml",
    ) -> str:
        """
        Builds a plain-English explanation of the predicted rating.
        """
        ocean       = profile.get("ocean", {})
        calibration = profile.get("rating_calibration", {})
        archetype   = profile.get("dominant_archetype", "reviewer")
        tendency    = calibration.get("harsh_or_generous", "balanced")

        # agreeableness and neuroticism are the dominant predictors
        agreeableness = ocean.get("A", 0.5)
        neuroticism   = ocean.get("N", 0.5)

        if agreeableness >= 0.70:
            a_desc = "generous with ratings"
        elif agreeableness <= 0.35:
            a_desc = "demanding with ratings"
        else:
            a_desc = "fair with ratings"

        if neuroticism >= 0.65:
            n_desc = "amplifies negative experiences"
        elif neuroticism <= 0.30:
            n_desc = "stays even-keeled"
        else:
            n_desc = "notices but doesn't dwell on issues"

        insight = f"Based on your profile — you tend to be {a_desc} and {n_desc}."
        if source.startswith("hybrid"):
            insight += f" This item has been highly rated by other {archetype}s."

        return (
            f"{insight} Your personal tendency is {tendency}. "
            f"Predicting {predicted} stars for this {category.rstrip('s')}."
        )

    @property
    def model_meta(self) -> dict:
        """Returns training metadata for debugging."""
        return self._meta


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("rating_predictor.py — self test")
    print("=" * 50)

    try:
        predictor = RatingPredictor()
        print()

        test_cases = [
            (
                "Generous enthusiast",
                {
                    "ocean": {"O": 0.75, "C": 0.45, "E": 0.80, "A": 0.85, "N": 0.15},
                    "liwc_features": {
                        "avg_review_length": 120, "avg_token_count": 22,
                        "vocab_richness": 0.62, "pos_word_ratio": 0.08,
                        "neg_word_ratio": 0.00, "exclamation_rate": 0.04,
                        "first_person_ratio": 0.06, "certainty_ratio": 0.02,
                        "hedge_ratio": 0.01,
                    },
                    "rating_calibration": {"generosity_score": 0.85},
                    "dominant_archetype": "enthusiast",
                },
                "movies",
            ),
            (
                "Harsh critic",
                {
                    "ocean": {"O": 0.70, "C": 0.75, "E": 0.35, "A": 0.20, "N": 0.85},
                    "liwc_features": {
                        "avg_review_length": 320, "avg_token_count": 58,
                        "vocab_richness": 0.72, "pos_word_ratio": 0.01,
                        "neg_word_ratio": 0.08, "exclamation_rate": 0.01,
                        "first_person_ratio": 0.04, "certainty_ratio": 0.03,
                        "hedge_ratio": 0.00,
                    },
                    "rating_calibration": {"generosity_score": 0.15},
                    "dominant_archetype": "critic",
                },
                "electronics",
            ),
            (
                "Balanced pragmatist",
                {
                    "ocean": {"O": 0.45, "C": 0.80, "E": 0.40, "A": 0.55, "N": 0.45},
                    "liwc_features": {
                        "avg_review_length": 95, "avg_token_count": 18,
                        "vocab_richness": 0.58, "pos_word_ratio": 0.03,
                        "neg_word_ratio": 0.02, "exclamation_rate": 0.00,
                        "first_person_ratio": 0.04, "certainty_ratio": 0.02,
                        "hedge_ratio": 0.01,
                    },
                    "rating_calibration": {"generosity_score": 0.50},
                    "dominant_archetype": "pragmatist",
                },
                "books",
            ),
        ]

        for label, profile, category in test_cases:
            result = predictor.predict(profile, category)
            print(f"{label} ({category}):")
            print(f"  Predicted: {result['predicted_rating']} stars "
                  f"(raw: {result['raw_score']}, confidence: {result['confidence']})")
            print(f"  Reasoning: {result['reasoning']}")
            print()

    except FileNotFoundError as e:
        print(f"ERROR: {e}")

