from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
"""
core/llm/service.py
-------------------
Orchestrates the two-call LLM chain for review generation
and the single-call chain for recommendations.

This is the main entry point that Django views will call.
It combines providers.py + prompt_builder.py into clean functions.

Usage:
    from core.llm.service import LLMService

    service = LLMService(provider_name="gemini")

    result = service.generate_review(
        profile=profile_json,
        product_name="Spotify",
        category="apps",
        predicted_rating=4,
        optional_note="mention the offline mode",
    )

    recommendations = service.generate_recommendations(
        profile=profile_json,
        category="movies",
        candidates=[{"title": "...", "description": "...", "avg_rating": 4.2}],
    )
"""

import json
import re

from core.llm.providers import get_provider, BaseLLMProvider
from core.llm.prompt_builder import (
    build_reasoning_prompt,
    build_review_prompt,
    build_recommendation_prompt,
)


class LLMService:
    """
    Orchestrates LLM calls for review generation and recommendations.
    Instantiate once per request in Django views.
    """

    def __init__(
        self,
        provider_name: str = "gemini",
        api_key: str | None = None,
    ):
        self._provider: BaseLLMProvider = get_provider(provider_name, api_key)

    # ── review generation ─────────────────────────────────────────────────────

    def generate_review(
        self,
        profile:          dict,
        product_name:     str,
        category:         str,
        predicted_rating: int,
        optional_note:    str | None = None,
        product_context:  dict | None = None,
    ) -> dict:

        # Call 1 — persona + product context → reasoning
        reasoning_prompt = build_reasoning_prompt(
            profile, category, product_context=product_context)
        reasoning = self._provider.reason(reasoning_prompt)

        # sample reviews from profile — user's own writing first
        sample_reviews = []
        user_review = profile.get("user_review", "")
        if user_review:
            sample_reviews.append(user_review)
        for pick in profile.get("pairwise_picks", []):
            if isinstance(pick, dict) and pick.get("chosen_text"):
                sample_reviews.append(pick["chosen_text"])

        # Call 2 — reasoning + product focus → review in user's voice
        review_prompt = build_review_prompt(
            reasoning=reasoning,
            product_name=product_name,
            category=category,
            predicted_rating=predicted_rating,
            optional_note=optional_note,
            sample_reviews=sample_reviews,
            product_context=product_context,
        )
        review = self._provider.generate(review_prompt)
        review = _strip_markdown(review)

        return {
            "review":           review,
            "reasoning":        _summarise_reasoning(reasoning),
            "full_reasoning":   reasoning,
            "predicted_rating": predicted_rating,
            "provider":         self._provider.name(),
        }

    # ── recommendations ───────────────────────────────────────────────────────

    def generate_recommendations(
        self,
        profile:    dict,
        category:   str,
        candidates: list[dict],
    ) -> list[dict]:
        """
        Single-call recommendation ranking.

        Args:
            profile:    Full persona JSON
            category:   Category being recommended
            candidates: List of dicts with title, description, avg_rating

        Returns:
            List of recommendation dicts with rank, title, confidence, reasoning
        """
        if not candidates:
            return []

        prompt   = build_recommendation_prompt(profile, category, candidates)
        response = self._provider.generate(prompt)

        return _parse_recommendations(response)

    @property
    def provider_name(self) -> str:
        return self._provider.name()


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting from generated review text."""
    # remove bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # remove bullet points
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    # collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _summarise_reasoning(full_reasoning: str) -> str:
    sentences = re.split(r"\n", full_reasoning.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if sentences:
        return sentences[-1]
    return full_reasoning[:150].strip() + "..."


def _parse_recommendations(response: str) -> list[dict]:
    """
    Parses the JSON response from the recommendation prompt.
    Falls back gracefully if the LLM returns malformed JSON.
    """
    # strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", response).strip()
    cleaned = cleaned.rstrip("`").strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        # sometimes wrapped in a key
        if isinstance(data, dict):
            for key in ("recommendations", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    except json.JSONDecodeError:
        pass

    # fallback — extract anything that looks like a ranked item
    fallback = []
    lines    = response.split("\n")
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            title = line.split(".", 1)[-1].strip()
            if title:
                fallback.append({
                    "rank":      len(fallback) + 1,
                    "title":     title,
                    "confidence": 0.5,
                    "reasoning": "Recommended based on your profile.",
                })
    return fallback[:5]


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    print("LLMService — self test")
    print("=" * 55)

    # sample profile — harsh critic
    sample_profile = {
        "ocean": {"O": 0.70, "C": 0.75, "E": 0.45, "A": 0.28, "N": 0.82},
        "voice_profile": {
            "length":      "long and structured",
            "tone":        "analytical and direct",
            "rating_bias": "harsh",
            "focus":       "defects, unmet expectations, value for money",
            "negatives":   "front and centre, specific and detailed",
        },
        "dominant_archetype":  "critic",
        "secondary_archetype": "skeptic",
        "rating_calibration": {
            "harsh_or_generous": "harsh",
            "avg_given_rating":  2.0,
        },
        "style_tags": [
            "doesn't shy away from calling out problems",
            "uses direct, certain language",
        ],
        "profile_summary": (
            "You blend Critic and Skeptic traits. You write long, structured "
            "reviews focusing on flaws and unmet expectations. You rate harshly."
        ),
    }

    provider_name = "gemini" if os.environ.get("GEMINI_API_KEY") else "anthropic"

    try:
        service = LLMService(provider_name=provider_name)
        print(f"\nUsing provider: {service.provider_name}")

        print("\n── Review Generation Test ───────────────────────────")
        result = service.generate_review(
            profile=sample_profile,
            product_name="Opay",
            category="apps",
            predicted_rating=2,
            optional_note="mention the failed transaction",
        )

        print(f"\nReasoning shown to user:\n  {result['reasoning']}")
        print(f"\nGenerated review (rating: {result['predicted_rating']} stars):")
        print(f"\n{result['review']}")

        print("\n── Recommendation Test ──────────────────────────────")
        candidates = [
            {"title": "Inception", "description": "A mind-bending thriller about dreams within dreams.", "avg_rating": 4.8},
            {"title": "The Notebook", "description": "A romantic drama about young love and devotion.", "avg_rating": 4.2},
            {"title": "Gone Girl", "description": "A psychological thriller with an unreliable narrator.", "avg_rating": 4.5},
            {"title": "Parasite", "description": "A dark Korean thriller about class inequality.", "avg_rating": 4.9},
            {"title": "La La Land", "description": "A musical romance set in modern Los Angeles.", "avg_rating": 4.3},
        ]

        recs = service.generate_recommendations(
            profile=sample_profile,
            category="movies",
            candidates=candidates,
        )

        print("\nRecommendations:")
        for r in recs:
            print(f"  {r.get('rank', '?')}. {r.get('title', '?')} "
                  f"(confidence: {r.get('confidence', '?')})")
            print(f"     {r.get('reasoning', '')}")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure GEMINI_API_KEY is set in your .env file")