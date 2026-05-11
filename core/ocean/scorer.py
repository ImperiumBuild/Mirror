"""
scorer.py
---------
Layer 1 of the Train My LLM flow.

Contains a bank of 12 scenario questions. At runtime, 3 are randomly
selected per user so no two sessions are identical.

Each question probes specific OCEAN dimensions through behaviour
scenarios — no personality jargon shown to the user.

Usage:
    from core.ocean.scorer import get_random_questions, score_answers, describe_ocean

    questions = get_random_questions(n=3)
    scores = score_answers({"q_damage": "b", "q_restaurant": "a", "q_app_bug": "c"})
    # {"O": 0.58, "C": 0.62, "E": 0.45, "A": 0.71, "N": 0.38}
"""

from __future__ import annotations
import random

QUESTION_BANK = [
    {
        "id":   "q_damage",
        "text": "You bought something and it arrived damaged. What do you do next?",
        "options": {
            "a": "Leave a 1-star review immediately and warn others",
            "b": "Contact support first, then review based on how they handle it",
            "c": "Request a replacement and only review if the issue isn't resolved",
            "d": "Move on — these things happen, not worth the energy",
        },
        "scores": {
            "a": {"N": +0.20, "A": -0.15, "E": +0.08},
            "b": {"C": +0.15, "A": +0.08, "N": +0.08},
            "c": {"C": +0.20, "A": +0.15, "N": -0.08},
            "d": {"N": -0.15, "A": +0.12, "O": +0.08},
        },
        "probes": ["N", "A", "C"],
    },
    {
        "id":   "q_mixed_book",
        "text": "You finish a book that started slowly but had an incredible ending. How do you rate it?",
        "options": {
            "a": "5 stars — the ending made it all worth it",
            "b": "4 stars — great ending but the slow start can't be ignored",
            "c": "3 stars — I balance the slow parts against the ending equally",
            "d": "2 stars — I nearly quit, and most people will too",
        },
        "scores": {
            "a": {"A": +0.20, "E": +0.12, "N": -0.12},
            "b": {"C": +0.15, "A": +0.08, "N": +0.08},
            "c": {"C": +0.20, "N": +0.00, "A": +0.00},
            "d": {"N": +0.20, "A": -0.15, "C": +0.08},
        },
        "probes": ["A", "N", "C"],
    },
    {
        "id":   "q_restaurant",
        "text": "A friend asks you to recommend a restaurant. How do you respond?",
        "options": {
            "a": "Give them the full rundown — ambiance, food, service, price, what to order",
            "b": "Name your favourite and say why you like it in a sentence or two",
            "c": "Ask what they're in the mood for first, then tailor your answer",
            "d": "Send them a link to the reviews and let them decide",
        },
        "scores": {
            "a": {"O": +0.20, "E": +0.12, "C": +0.08},
            "b": {"E": +0.08, "O": -0.08, "A": +0.08},
            "c": {"A": +0.20, "O": +0.12, "E": +0.08},
            "d": {"E": -0.15, "A": -0.08, "O": -0.08},
        },
        "probes": ["O", "E", "A"],
    },
    {
        "id":   "q_review_length",
        "text": "You watched an excellent movie. How long is your review likely to be?",
        "options": {
            "a": "A few words — 'amazing, just watch it'",
            "b": "A short paragraph covering the highlights",
            "c": "Several paragraphs — I want to do it justice",
            "d": "As long as it needs to be — I write until I've said everything",
        },
        "scores": {
            "a": {"O": -0.15, "C": -0.08, "E": +0.08},
            "b": {"O": +0.08, "C": +0.08, "E": +0.08},
            "c": {"O": +0.15, "C": +0.15, "E": +0.08},
            "d": {"O": +0.20, "E": +0.12, "C": +0.12},
        },
        "probes": ["O", "C"],
    },
    {
        "id":   "q_app_bug",
        "text": "You try an app that mostly works but has one annoying bug. Your review focuses on...",
        "options": {
            "a": "The bug — I warn others so they know what they're getting into",
            "b": "Both — I give credit where it's due but the bug needs to be mentioned",
            "c": "The overall experience — the bug is a footnote",
            "d": "I don't review it until the bug is fixed",
        },
        "scores": {
            "a": {"N": +0.20, "A": -0.08, "C": +0.08},
            "b": {"C": +0.20, "A": +0.08, "N": +0.08},
            "c": {"A": +0.20, "N": -0.12, "O": +0.08},
            "d": {"C": +0.15, "N": +0.08, "A": -0.08},
        },
        "probes": ["N", "A", "C"],
    },
    {
        "id":   "q_new_product",
        "text": "You try a product everyone has been raving about. It's just okay. What do you do?",
        "options": {
            "a": "Write an honest review explaining why it didn't live up to the hype",
            "b": "Give it a neutral rating and move on — not everything is for everyone",
            "c": "Ask around to see if others feel the same before reviewing",
            "d": "Say nothing — who am I to go against the crowd",
        },
        "scores": {
            "a": {"E": +0.20, "O": +0.15, "N": +0.08},
            "b": {"C": +0.15, "A": +0.12, "N": -0.08},
            "c": {"A": +0.15, "O": +0.12, "E": -0.08},
            "d": {"E": -0.20, "A": +0.08, "N": -0.08},
        },
        "probes": ["E", "O", "A"],
    },
    {
        "id":   "q_before_buying",
        "text": "Before buying something new, you typically...",
        "options": {
            "a": "Read every review you can find, compare alternatives, then decide",
            "b": "Skim the top reviews and check the star rating",
            "c": "Ask someone you trust what they think",
            "d": "Just buy it — you'll figure out if it's good soon enough",
        },
        "scores": {
            "a": {"C": +0.20, "O": +0.15, "N": +0.08},
            "b": {"C": +0.08, "O": +0.08, "E": +0.08},
            "c": {"A": +0.20, "E": +0.08, "O": -0.08},
            "d": {"O": +0.15, "N": -0.12, "C": -0.15},
        },
        "probes": ["C", "O", "A"],
    },
    {
        "id":   "q_service",
        "text": "You had great food but terrible service at a restaurant. Your review...",
        "options": {
            "a": "Focuses mainly on the service — that ruined the experience",
            "b": "Gives equal weight to both — readers deserve the full picture",
            "c": "Highlights the food and mentions service as a caveat",
            "d": "I'd probably not leave a review — too conflicted",
        },
        "scores": {
            "a": {"N": +0.20, "A": -0.12, "E": +0.08},
            "b": {"C": +0.20, "O": +0.12, "E": +0.08},
            "c": {"A": +0.20, "N": -0.12, "O": +0.08},
            "d": {"E": -0.20, "N": +0.08, "A": +0.08},
        },
        "probes": ["N", "A", "E"],
    },
    {
        "id":   "q_expectations",
        "text": "A product works perfectly but isn't what you expected based on the description. You rate it...",
        "options": {
            "a": "1-2 stars — misleading descriptions are inexcusable",
            "b": "3 stars — it works, but I didn't get what I came for",
            "c": "4 stars — good product, I just misjudged what I needed",
            "d": "5 stars — not what I expected but I love it anyway",
        },
        "scores": {
            "a": {"N": +0.20, "C": +0.12, "A": -0.15},
            "b": {"C": +0.20, "N": +0.08, "A": +0.00},
            "c": {"A": +0.15, "N": -0.12, "C": +0.08},
            "d": {"A": +0.20, "N": -0.20, "O": +0.12},
        },
        "probes": ["N", "C", "A"],
    },
    {
        "id":   "q_storytelling",
        "text": "When writing a review, which feels most natural to you?",
        "options": {
            "a": "Telling the story — why I bought it, what happened, how I feel now",
            "b": "Listing the pros and cons clearly",
            "c": "Giving an overall verdict and a couple of reasons",
            "d": "Just the rating — it speaks for itself",
        },
        "scores": {
            "a": {"O": +0.20, "E": +0.15, "C": -0.08},
            "b": {"C": +0.20, "O": +0.08, "E": -0.08},
            "c": {"E": +0.12, "C": +0.08, "O": +0.08},
            "d": {"O": -0.20, "E": -0.15, "C": -0.08},
        },
        "probes": ["O", "E", "C"],
    },
    {
        "id":   "q_repeat_bad",
        "text": "You've had a bad experience with a brand twice now. What happens next?",
        "options": {
            "a": "Never buying from them again and making sure others know why",
            "b": "One more chance — two bad experiences could still be bad luck",
            "c": "Contacting them directly before going public with a review",
            "d": "Giving them a fair review that reflects both experiences",
        },
        "scores": {
            "a": {"N": +0.20, "A": -0.20, "E": +0.12},
            "b": {"A": +0.20, "N": -0.12, "O": +0.08},
            "c": {"C": +0.20, "A": +0.12, "N": -0.08},
            "d": {"C": +0.15, "A": +0.15, "N": -0.08},
        },
        "probes": ["N", "A", "C"],
    },
    {
        "id":   "q_who_for",
        "text": "When you write a review, who are you mainly writing it for?",
        "options": {
            "a": "Future buyers — I want to save them from my mistake or share a great find",
            "b": "The brand — feedback they should hear",
            "c": "Myself — processing my own experience",
            "d": "No one really — I just feel like I should",
        },
        "scores": {
            "a": {"A": +0.20, "E": +0.15, "O": +0.08},
            "b": {"E": +0.12, "C": +0.12, "N": +0.08},
            "c": {"O": +0.15, "E": -0.12, "N": +0.08},
            "d": {"E": -0.20, "A": -0.08, "C": -0.08},
        },
        "probes": ["A", "E", "O"],
    },
]

CALIBRATION_BANK = [
    {
        "id":             "c_delivery",
        "text":           "You order food delivery. It arrives 20 minutes late but tastes exactly as expected. Your rating is...",
        "neutral_rating": 3,
    },
    {
        "id":             "c_movie",
        "text":           "You watch a highly anticipated movie. The first hour is slow but the last 30 minutes are excellent. Your rating is...",
        "neutral_rating": 3,
    },
    {
        "id":             "c_app_update",
        "text":           "An app you use daily releases an update that improves one feature but breaks another you rarely use. Your rating is...",
        "neutral_rating": 4,
    },
    {
        "id":             "c_hotel",
        "text":           "A hotel room is clean and comfortable but the wifi was slow and breakfast mediocre. Your rating is...",
        "neutral_rating": 3,
    },
    {
        "id":             "c_book_ending",
        "text":           "A book you loved for 80% of its length had a disappointing ending. Your rating is...",
        "neutral_rating": 3,
    },
    {
        "id":             "c_product_packaging",
        "text":           "A product works perfectly but arrived in damaged packaging with no instructions. Your rating is...",
        "neutral_rating": 4,
    },
]

BASE_SCORES = {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5}


def get_random_questions(n: int = 3, seed: int | None = None) -> list[dict]:
    """Returns n randomly selected questions from the bank."""
    rng      = random.Random(seed)
    selected = rng.sample(QUESTION_BANK, min(n, len(QUESTION_BANK)))
    return [
        {"id": q["id"], "text": q["text"], "options": q["options"]}
        for q in selected
    ]


def get_random_calibration(n: int = 2, seed: int | None = None) -> list[dict]:
    """Returns n randomly selected calibration scenarios."""
    rng      = random.Random(seed)
    selected = rng.sample(CALIBRATION_BANK, min(n, len(CALIBRATION_BANK)))
    return [{"id": s["id"], "text": s["text"]} for s in selected]


def score_answers(answers: dict[str, str]) -> dict[str, float]:
    """Maps {question_id: option_letter} to OCEAN scores."""
    scores       = BASE_SCORES.copy()
    question_map = {q["id"]: q for q in QUESTION_BANK}

    for q_id, chosen_option in answers.items():
        if q_id not in question_map:
            continue
        question      = question_map[q_id]
        chosen_option = chosen_option.lower().strip()
        if chosen_option not in question["scores"]:
            continue
        for dimension, delta in question["scores"][chosen_option].items():
            scores[dimension] += delta

    return {k: round(max(0.0, min(1.0, v)), 4) for k, v in scores.items()}


def score_calibration(calibration_answers: dict[str, int]) -> dict:
    """Converts calibration scenario ratings into harsh/generous profile."""
    scenario_map  = {s["id"]: s for s in CALIBRATION_BANK}
    deviations    = []
    given_ratings = []

    for scenario_id, given_rating in calibration_answers.items():
        given_rating = int(given_rating)
        given_ratings.append(given_rating)
        if scenario_id in scenario_map:
            deviations.append(
                given_rating - scenario_map[scenario_id]["neutral_rating"])

    if not deviations:
        return {
            "generosity_score":  0.5,
            "avg_given_rating":  3.0,
            "rating_variance":   0.0,
            "harsh_or_generous": "balanced",
        }

    avg_deviation = sum(deviations) / len(deviations)
    avg_given     = sum(given_ratings) / len(given_ratings)
    variance      = (
        sum((r - avg_given) ** 2 for r in given_ratings) / len(given_ratings)
        if len(given_ratings) > 1 else 0.0
    )
    generosity = round(max(0.0, min(1.0, (avg_deviation + 2) / 4)), 4)

    if generosity >= 0.70:
        label = "generous"
    elif generosity >= 0.55:
        label = "slightly generous"
    elif generosity >= 0.45:
        label = "balanced"
    elif generosity >= 0.30:
        label = "slightly harsh"
    else:
        label = "harsh"

    return {
        "generosity_score":  generosity,
        "avg_given_rating":  round(avg_given, 2),
        "rating_variance":   round(variance, 4),
        "harsh_or_generous": label,
    }


def describe_ocean(scores: dict[str, float]) -> str:
    """Plain-English personality description from OCEAN scores."""
    lines = []

    if scores["O"] >= 0.60:
        lines.append("You're curious and open-minded, with a wide range of interests.")
    elif scores["O"] <= 0.40:
        lines.append("You're practical and grounded, preferring the tried and tested.")
    else:
        lines.append("You balance curiosity with practicality.")

    if scores["C"] >= 0.60:
        lines.append("You're thorough and structured — you don't cut corners.")
    elif scores["C"] <= 0.40:
        lines.append("You're spontaneous and flexible, going with the flow.")
    else:
        lines.append("You're reasonably organised without being rigid about it.")

    if scores["E"] >= 0.60:
        lines.append("You're expressive and energetic in how you communicate.")
    elif scores["E"] <= 0.40:
        lines.append("You're measured and reserved — you say what you mean and no more.")
    else:
        lines.append("You're comfortable in both expressive and reserved modes.")

    if scores["A"] >= 0.60:
        lines.append("You're generous and fair, giving the benefit of the doubt.")
    elif scores["A"] <= 0.40:
        lines.append("You hold high standards and don't soften your opinions.")
    else:
        lines.append("You're balanced — fair but honest when things fall short.")

    if scores["N"] >= 0.60:
        lines.append("When something disappoints you, you feel it strongly and say so.")
    elif scores["N"] <= 0.40:
        lines.append("You're even-keeled — setbacks don't throw you off easily.")
    else:
        lines.append("You notice problems but don't let them dominate your view.")

    return " ".join(lines)


if __name__ == "__main__":
    print("scorer.py — self test")
    print("=" * 50)

    print("\nRandom questions (n=3, seed=42):")
    for q in get_random_questions(n=3, seed=42):
        print(f"  [{q['id']}] {q['text'][:60]}...")

    print("\nRandom calibration (n=2, seed=42):")
    for s in get_random_calibration(n=2, seed=42):
        print(f"  [{s['id']}] {s['text'][:60]}...")

    print("\nScoring tests:")
    harsh = {"q_damage": "a", "q_storytelling": "a", "q_who_for": "a"}
    print(f"  Harsh:    {score_answers(harsh)}")

    generous = {"q_damage": "d", "q_mixed_book": "a", "q_who_for": "a"}
    print(f"  Generous: {score_answers(generous)}")

    balanced = {"q_damage": "b", "q_app_bug": "b", "q_restaurant": "c"}
    print(f"  Balanced: {score_answers(balanced)}")

    print("\nCalibration test:")
    print(f"  Harsh:    {score_calibration({'c_delivery': 2, 'c_movie': 2})}")
    print(f"  Generous: {score_calibration({'c_delivery': 5, 'c_movie': 5})}")