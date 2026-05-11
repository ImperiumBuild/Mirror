# test_with_context.py
import sys
sys.path.insert(0, ".")

from core.review_engine import ReviewEngine

profile = {
    "ocean": {"O": 0.7173, "C": 0.84, "E": 0.6548, "A": 0.666, "N": 0.6807},
    "liwc_features": {
        "avg_sentence_len": 11.5,
        "ttr": 0.8312,
        "pos_affect_ratio": 0.0128,
        "neg_affect_ratio": 0.0303,
        "exclamation_rate": 0.0256,
        "review_length": 147.3333
    },
    "dominant_archetype":  "pragmatist",
    "secondary_archetype": "skeptic",
    "rating_calibration": {
        "harsh_or_generous": "slightly generous",
        "avg_given_rating":  4.0,
        "generosity_score":  0.6,
    },
    "style_tags": ["doesn't shy away from calling out problems", "expressive and enthusiastic in tone"],
    "voice_profile": {
        "length":      "short to medium",
        "tone":        "casual and direct",
        "rating_bias": "slightly generous",
        "focus":       "immediate functionality and bugs",
        "negatives":   "upfront with urgency",
    },
    "pairwise_picks": [
        {"chosen_text": "This app needs to be checked. The transaction feature should at least refund a failed transaction. Otherwise the app has been good.", "tag": "tone"},
        {"chosen_text": "WhatsApp needs to watch their updates, Video calls are not even possible at this point. This needs to be resolved as soon as possible.", "tag": "focus"},
    ],
    "user_review": "I cant enjoy the app if you keep on rolling bugs. This needs to be fixed ASAP. I would hold off updating if I were you!!",
    "profile_summary": "You write short reviews with high urgency. You use exclamations and direct commands.",
}

engine = ReviewEngine(provider="gemini")

print("Testing Kuda (apps)...")
result = engine.generate_review(
    profile=profile,
    product_name="Kuda",
    category="apps",
)

print(f"\nFocus areas found: {result['product_context'].get('focus_areas', [])}")
print(f"Predicted rating:  {result['predicted_rating']}/5")
print(f"Reasoning:         {result['reasoning_shown']}")
print(f"\nGenerated review:\n{result['review']}")