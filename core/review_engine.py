"""
core/review_engine.py
---------------------
Orchestrates full review generation pipeline.
Now includes product context fetching before LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model.rating_predictor import RatingPredictor
from core.llm.service import LLMService
from core.product_context import get_product_context


class ReviewEngine:

    def __init__(
        self,
        provider:  str = "gemini",
        api_key:   str | None = None,
    ):
        self._predictor = RatingPredictor()
        self._llm       = LLMService(provider_name=provider, api_key=api_key)

    def generate_review(
        self,
        profile:       dict,
        product_name:  str,
        category:      str,
        optional_note: str | None = None,
    ) -> dict:
        """
        Full pipeline:
          1. Fetch product context (description + what reviewers focus on)
          2. Predict rating from persona
          3. Call 1 — persona + product context → behavioural reasoning
          4. Call 2 — reasoning + product focus → review in user's voice
        """

        # ── Step 1: Fetch product context ──
        product_context = get_product_context(product_name, category)

        # ── Step 2: Predict rating ──
        rating_result    = self._predictor.predict(
            profile=profile, category=category, product_name=product_name)
        predicted_rating  = rating_result["predicted_rating"]
        rating_confidence = rating_result["confidence"]
        rating_reasoning  = rating_result["reasoning"]

        # User Override: if optional_note is strongly positive or negative, adjust the rating
        if optional_note:
            note_lower = optional_note.lower()
            positive_words = ["love", "great", "excellent", "best", "good", "happy", "amazing"]
            negative_words = ["hate", "terrible", "bad", "worst", "bug", "crazy", "crash", "freeze"]
            
            has_pos = any(w in note_lower for w in positive_words)
            has_neg = any(w in note_lower for w in negative_words)
            
            if has_pos and not has_neg:
                # user likes it, ensure rating is high (4-5)
                predicted_rating = max(predicted_rating, 4)
            elif has_neg and not has_pos:
                # user is complaining, ensure rating is low (1-2)
                predicted_rating = min(predicted_rating, 2)

        # ── Step 3 + 4: Two-call LLM chain ──
        llm_result = self._llm.generate_review(
            profile=profile,
            product_name=product_name,
            category=category,
            predicted_rating=predicted_rating,
            optional_note=optional_note,
            product_context=product_context,
        )

        return {
            "product_name":      product_name,
            "category":          category,
            "predicted_rating":  predicted_rating,
            "rating_confidence": rating_confidence,
            "rating_reasoning":  rating_reasoning,
            "reasoning_shown":   llm_result["reasoning"],
            "review":            llm_result["review"],
            "provider":          llm_result["provider"],
            "product_context":   {
                "focus_areas":  product_context.get("common_focus_areas", []),
                "avg_rating":   product_context.get("avg_public_rating", None),
                "source":       product_context.get("source", "none"),
            } if product_context else {},
        }