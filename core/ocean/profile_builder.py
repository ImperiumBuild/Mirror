"""
profile_builder.py
------------------
Aggregates all three layers of the Train My LLM flow into a
single persona JSON stored in the taste_profiles table.

Updated flow:
  Layer 1: 3 random scenario answers (from scorer.py bank of 12)
  Layer 2: 3 pairwise review picks + one tap tag per pick
           ("what made you pick this?" → tone/detail/focus/negatives)
  Layer 3: 2 random rating calibration scenarios

LIWC weight increased to 0.40 — pairwise picks are revealed
preference and carry more signal than stated answers.

Usage:
    from core.ocean.profile_builder import build_profile

    profile = build_profile(
        layer1_answers={"q_damage": "b", "q_restaurant": "a", "q_app_bug": "c"},
        pairwise_picks=[
            {"chosen_text": "review text...", "tag": "tone"},
            {"chosen_text": "review text...", "tag": "detail"},
            {"chosen_text": "review text...", "tag": "focus"},
        ],
        calibration_answers={"c_delivery": 3, "c_movie": 4},
        user_meta={"age_range": "25-34", "occupation": "engineer"},
    )
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from core.ocean.scorer import score_answers, score_calibration, describe_ocean
from core.ocean.Liwc import analyse_chosen_reviews, ocean_from_liwc, tag_reasoning
from core.ocean.archetypes import ArchetypeMatcher


# ── pairwise tag → OCEAN signal ───────────────────────────────────────────────

TAG_OCEAN_SIGNALS = {
    "tone":      {"E": +0.08, "A": +0.05},
    "detail":    {"O": +0.08, "C": +0.08},
    "focus":     {"O": +0.10, "C": +0.05},
    "negatives": {"N": +0.10, "A": -0.05},
}


# ── ocean merger ──────────────────────────────────────────────────────────────

def merge_ocean(
    layer1_scores:    dict[str, float],
    liwc_adjustments: dict[str, float],
    tag_adjustments:  dict[str, float],
    liwc_weight: float = 0.40,
    tag_weight:  float = 0.20,
) -> dict[str, float]:
    """
    Weighted merge of three OCEAN signal sources.
    Split: 40% Layer 1 / 40% LIWC / 20% Tags
    Pairwise picks carry 60% of total weight.
    """
    merged        = {}
    layer1_weight = 1.0 - liwc_weight - tag_weight

    for dim in ["O", "C", "E", "A", "N"]:
        base     = layer1_scores.get(dim, 0.5)
        liwc_adj = liwc_adjustments.get(dim, 0.0)
        tag_adj  = tag_adjustments.get(dim, 0.0)

        value = (
            base * layer1_weight
            + (base + liwc_adj) * liwc_weight
            + (base + tag_adj)  * tag_weight
        )
        merged[dim] = round(max(0.0, min(1.0, value)), 4)

    return merged


def aggregate_tag_signals(tags: list[str]) -> dict[str, float]:
    """Converts list of tag choices into averaged OCEAN adjustments."""
    totals = {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0}
    for tag in tags:
        for dim, delta in TAG_OCEAN_SIGNALS.get(tag, {}).items():
            totals[dim] += delta
    n = max(len(tags), 1)
    return {dim: round(v / n, 4) for dim, v in totals.items()}


# ── summary builder ───────────────────────────────────────────────────────────

def build_profile_summary(
    ocean:              dict[str, float],
    dominant_archetype: dict,
    secondary_name:     str,
    confidence:         float,
    calibration:        dict,
    style_tags:         list[str],
) -> str:
    voice_traits = dominant_archetype.get("voice_traits", {})
    length       = voice_traits.get("length", "medium-length")
    tone         = voice_traits.get("tone", "balanced")
    bias         = voice_traits.get("rating_bias", "balanced")
    focus        = voice_traits.get("focus", "overall experience")
    generosity   = calibration["harsh_or_generous"]
    avg_rating   = calibration["avg_given_rating"]
    ocean_desc   = describe_ocean(ocean)

    parts = []

    if confidence >= 0.70:
        parts.append(
            f"Your reviewer personality is closest to "
            f"{dominant_archetype['name']}. "
            f"{dominant_archetype['description']}"
        )
    else:
        parts.append(
            f"Your reviewer personality blends "
            f"{dominant_archetype['name']} and {secondary_name} traits."
        )

    parts.append(
        f"You write {length} reviews in a {tone} voice, "
        f"tending to be {bias} with ratings and focusing on {focus}."
    )

    parts.append(
        f"When rating products, you tend to be {generosity} — "
        f"your average across scenarios was {avg_rating:.1f} stars."
    )

    if style_tags:
        parts.append(
            f"The reviews you identified with suggest you: "
            f"{', '.join(style_tags[:3])}."
        )

    parts.append(ocean_desc)
    return " ".join(parts)


# ── main builder ──────────────────────────────────────────────────────────────

def build_profile(
    layer1_answers:      dict[str, str],
    pairwise_picks:      list[dict],
    calibration_answers: dict[str, int],
    user_reviews:        list[str] | None = None,
    user_meta:           dict | None = None,
    archetypes_path:     str | None  = None,
) -> dict:
    """
    Full pipeline: three layers → persona JSON.

    Args:
        layer1_answers: {question_id: option_letter}
        pairwise_picks: List of chosen reviews and tags
        calibration_answers: {scenario_id: star_rating}
        user_reviews: Optional list of texts written by the user themselves
        user_meta: Optional metadata
        archetypes_path: Optional override path
    """
    # Layer 1 — scenario answers
    layer1_ocean = score_answers(layer1_answers)

    # Layer 2 — LIWC from chosen review texts AND user's own writing
    chosen_texts = [p["chosen_text"] for p in pairwise_picks if p.get("chosen_text")]
    if user_reviews:
        for rev in user_reviews:
            if rev.strip():
                chosen_texts.append(rev.strip())
        
    liwc_features    = analyse_chosen_reviews(chosen_texts) if chosen_texts else {}
    liwc_adjustments = ocean_from_liwc(liwc_features) if liwc_features else {"O":0,"C":0,"E":0,"A":0,"N":0}
    style_tags       = tag_reasoning(liwc_features) if liwc_features else []

    # Layer 2 — tag signals
    tags            = [p["tag"] for p in pairwise_picks if p.get("tag")]
    tag_adjustments = aggregate_tag_signals(tags)

    # Merge
    final_ocean = merge_ocean(layer1_ocean, liwc_adjustments, tag_adjustments)

    # Layer 3 — calibration
    calibration = score_calibration(calibration_answers)

    # Archetype matching
    matcher = ArchetypeMatcher(archetypes_path) if archetypes_path else ArchetypeMatcher()
    archetype_result = matcher.match(final_ocean)
    dominant_def     = archetype_result["archetype"]
    secondary_def    = matcher.get_archetype(archetype_result["secondary"])
    secondary_name   = (
        secondary_def["name"] if secondary_def
        else archetype_result["secondary"].title()
    )

    summary = build_profile_summary(
        ocean=final_ocean,
        dominant_archetype=dominant_def,
        secondary_name=secondary_name,
        confidence=archetype_result["confidence"],
        calibration=calibration,
        style_tags=style_tags,
    )

    return {
        "ocean":                   final_ocean,
        "layer1_ocean":            layer1_ocean,
        "liwc_adjustments":        liwc_adjustments,
        "tag_adjustments":         tag_adjustments,
        "liwc_features": {
            k: v for k, v in liwc_features.items()
            if k in ["ttr", "pos_affect_ratio", "neg_affect_ratio",
                     "certainty_ratio", "hedge_ratio",
                     "exclamation_rate", "review_length", "avg_sentence_len"]
        },
        "dominant_archetype":      archetype_result["dominant"],
        "secondary_archetype":     archetype_result["secondary"],
        "archetype_confidence":    archetype_result["confidence"],
        "archetype_probabilities": archetype_result["probabilities"],
        "voice_profile": {
            "length":      dominant_def["voice_traits"]["length"],
            "tone":        dominant_def["voice_traits"]["tone"],
            "rating_bias": dominant_def["voice_traits"]["rating_bias"],
            "focus":       dominant_def["voice_traits"]["focus"],
            "negatives":   dominant_def["voice_traits"]["negatives"],
        },
        "rating_calibration": calibration,
        "style_tags":         style_tags,
        "pairwise_tags":      tags,
        "user_meta":          user_meta or {},
        "profile_summary":    summary,
        "category_priorities": {
            "books":       {"focus": [], "weights": {}},
            "movies":      {"focus": [], "weights": {}},
            "electronics": {"focus": [], "weights": {}},
            "apps":        {"focus": [], "weights": {}},
        },
        "feedback_count":   0,
        "review_count":     0,
        "last_updated":     datetime.now(timezone.utc).isoformat(),
        "training_version": "1.0",
    }


# ── feedback updater ──────────────────────────────────────────────────────────

def update_profile_from_feedback(
    existing_profile: dict,
    feedback_score:   str,
    generated_review: str,
    edited_review:    str | None,
    category:         str,
) -> dict:
    """Called by Django Q2 background job after user rates a review."""
    profile = copy.deepcopy(existing_profile)
    profile["feedback_count"] = profile.get("feedback_count", 0) + 1
    profile["review_count"]   = profile.get("review_count", 0) + 1
    profile["last_updated"]   = datetime.now(timezone.utc).isoformat()

    if feedback_score == "spot_on" or not edited_review:
        return profile

    liwc_features    = analyse_chosen_reviews([edited_review])
    liwc_adjustments = ocean_from_liwc(liwc_features)
    strength         = 0.08 if feedback_score == "off" else 0.04
    current_ocean    = profile.get("ocean", {})

    for dim in ["O", "C", "E", "A", "N"]:
        delta = liwc_adjustments.get(dim, 0.0) * strength
        current_ocean[dim] = round(
            max(0.0, min(1.0, current_ocean.get(dim, 0.5) + delta)), 4)

    profile["ocean"] = current_ocean

    try:
        matcher          = ArchetypeMatcher()
        archetype_result = matcher.match(current_ocean)
        dominant_def     = archetype_result["archetype"]
        profile["dominant_archetype"]      = archetype_result["dominant"]
        profile["secondary_archetype"]     = archetype_result["secondary"]
        profile["archetype_confidence"]    = archetype_result["confidence"]
        profile["archetype_probabilities"] = archetype_result["probabilities"]
        profile["voice_profile"] = {
            "length":      dominant_def["voice_traits"]["length"],
            "tone":        dominant_def["voice_traits"]["tone"],
            "rating_bias": dominant_def["voice_traits"]["rating_bias"],
            "focus":       dominant_def["voice_traits"]["focus"],
            "negatives":   dominant_def["voice_traits"]["negatives"],
        }
    except Exception:
        pass

    new_tags = tag_reasoning(liwc_features)
    if new_tags:
        profile["style_tags"] = new_tags

    return profile


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("profile_builder.py — self test")
    print("=" * 55)

    harsh = build_profile(
        layer1_answers={
            "q_damage":       "a",
            "q_storytelling": "a",
            "q_who_for":      "a",
        },
        pairwise_picks=[
            {"chosen_text": "Terrible. Arrived damaged, support was useless. Avoid.", "tag": "negatives"},
            {"chosen_text": "Complete waste of money. Returning immediately. Do not buy.", "tag": "tone"},
            {"chosen_text": "Three of these, all failed within a month. Pattern is clear.", "tag": "detail"},
        ],
        calibration_answers={"c_delivery": 2, "c_movie": 2},
        user_meta={"age_range": "35-44"},
    )

    print("\nHarsh reviewer:")
    print(f"  Layer 1 OCEAN: {harsh['layer1_ocean']}")
    print(f"  Tag signals:   {harsh['tag_adjustments']}")
    print(f"  Final OCEAN:   {harsh['ocean']}")
    print(f"  Dominant:      {harsh['dominant_archetype']} ({harsh['archetype_confidence']:.2f})")
    print(f"  Secondary:     {harsh['secondary_archetype']}")
    print(f"  Calibration:   {harsh['rating_calibration']['harsh_or_generous']}")
    print(f"\n  Summary: {harsh['profile_summary']}")

    print("\n" + "=" * 55)

    generous = build_profile(
        layer1_answers={
            "q_damage":     "d",
            "q_mixed_book": "a",
            "q_who_for":    "a",
        },
        pairwise_picks=[
            {"chosen_text": "Absolutely love this! Best purchase I've made all year!!!", "tag": "tone"},
            {"chosen_text": "Amazing quality. Arrived quickly. Will definitely buy again!", "tag": "focus"},
            {"chosen_text": "Great product, highly recommend to everyone. 5 stars!", "tag": "tone"},
        ],
        calibration_answers={"c_delivery": 5, "c_movie": 5},
        user_meta={"age_range": "25-34"},
    )

    print("\nGenerous reviewer:")
    print(f"  Layer 1 OCEAN: {generous['layer1_ocean']}")
    print(f"  Tag signals:   {generous['tag_adjustments']}")
    print(f"  Final OCEAN:   {generous['ocean']}")
    print(f"  Dominant:      {generous['dominant_archetype']} ({generous['archetype_confidence']:.2f})")
    print(f"  Secondary:     {generous['secondary_archetype']}")
    print(f"  Calibration:   {generous['rating_calibration']['harsh_or_generous']}")
    print(f"\n  Summary: {generous['profile_summary']}")

    print("\n" + "=" * 55)
    print("Feedback update:")
    updated = update_profile_from_feedback(
        existing_profile=harsh,
        feedback_score="off",
        generated_review="This product was bad.",
        edited_review="Broken on arrival. No support response. Avoid this brand entirely.",
        category="electronics",
    )
    print(f"  Before: {harsh['ocean']}")
    print(f"  After:  {updated['ocean']}")
    print(f"  Feedback count: {updated['feedback_count']}")