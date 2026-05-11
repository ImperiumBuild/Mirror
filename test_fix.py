from core.review_engine import ReviewEngine
import json

# Your actual profile data from the previous run
profile = {
    "ocean": {"O": 0.7173, "C": 1.0, "E": 0.6548, "A": 0.666, "N": 0.6807},
    "liwc_features": {
        "avg_sentence_len": 11.5,
        "ttr": 0.8312,
        "pos_affect_ratio": 0.0128,
        "neg_affect_ratio": 0.0303,
        "certainty_ratio": 0.009,
        "hedge_ratio": 0.0,
        "exclamation_rate": 0.0256,
        "review_length": 147.3333
    },
    "dominant_archetype": "pragmatist",
    "secondary_archetype": "skeptic",
    "rating_calibration": {
        "harsh_or_generous": "slightly generous",
        "avg_given_rating": 4.0
    },
    "style_tags": ["doesn't shy away from calling out problems", "expressive and enthusiastic in tone"],
    "voice_profile": {
        "length": "short and punchy",
        "tone": "casual and direct",
        "rating_bias": "slightly generous",
        "focus": "immediate functionality and bugs",
        "negatives": "upfront with urgency (ASAP, ASAP!!)"
    },
    # THESE ARE THE SAMPLES IT MUST CLONE
    "pairwise_picks": [
        {"chosen_text": "This app needs to be checked. The transaction feature should at least refund a failed transaction. Otherwise the app has been good.", "tag": "tone"},
        {"chosen_text": "WhatsApp needs to watch their updates, Video calls are not even possible at this point. This needs to be resolved as soon as possible.", "tag": "focus"}
    ],
    "user_review": "I cant enjoy the app if you keep on rolling bugs. This needs to be fixed ASAP. I would hold off updating if I were you!!",
    "profile_summary": "You write short reviews with high urgency. You use exclamations and direct commands (ASAP, hold off)."
}

print("=" * 60)
print("RE-GENERATING REVIEW WITH FIXED VOICE")
print("=" * 60)

product_name = "Telegram"
category = "apps"
optional_note = "I am just a neutral user"

print(f"\nTargeting voice: 'Short, direct, high-energy, uses ASAP!!'\n")

# Use the existing engine but we've tuned the profile to be MUCH more aggressive
engine = ReviewEngine(provider="gemini")
result = engine.generate_review(
    profile=profile,
    product_name=product_name,
    category=category,
    optional_note=optional_note,
)

print(f"PREDICTED RATING: {result['predicted_rating']} / 5")
print(f"REASONING:        {result['reasoning_shown']}")
print(f"\nGENERATED REVIEW:\n{'-' * 20}")
print(result['review'])
print(f"{'-' * 20}\n")
