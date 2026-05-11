"""
reviews/views.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_q.tasks import async_task

from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

from .models import Review
from users.models import TasteProfile
from core.review_engine import ReviewEngine

@extend_schema(
    summary="Generate a review in the user's voice",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "item_name":     {"type": "string",  "example": "Telegram"},
                "category":      {"type": "string",  "enum": ["apps", "products", "books", "movies", "restaurants"], "example": "apps"},
                "optional_note": {"type": "string",  "example": "mention the update bugs"},
            },
            "required": ["item_name", "category"],
        }
    },
    responses={
        201: {
            "description": "Generated review",
            "example": {
                "review_id":        1,
                "item_name":        "Telegram",
                "category":         "apps",
                "predicted_rating": 4,
                "rating_confidence": "high",
                "reasoning":        "You tend to be slightly generous...",
                "review":           "Telegram is great! But the updates need work ASAP!!",
                "provider":         "gemini",
            }
        }
    },
    tags=["Reviews"],
)
@extend_schema(
    summary="Generate a review in the user's voice",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "item_name":     {"type": "string",  "example": "Telegram"},
                "category":      {"type": "string",  "enum": ["apps", "products", "books", "movies", "restaurants"], "example": "apps"},
                "optional_note": {"type": "string",  "example": "mention the update bugs"},
            },
            "required": ["item_name", "category"],
        }
    },
    responses={
        201: {
            "description": "Generated review",
            "example": {
                "review_id":        1,
                "item_name":        "Telegram",
                "category":         "apps",
                "predicted_rating": 4,
                "rating_confidence": "high",
                "reasoning":        "You tend to be slightly generous...",
                "review":           "Telegram is great! But the updates need work ASAP!!",
                "provider":         "gemini",
            }
        }
    },
    tags=["Reviews"],
)
 
class GenerateReviewView(APIView):
    """
    POST /api/reviews/generate/
    Generates a review in the user's voice.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        item_name     = request.data.get("item_name", "").strip()
        category      = request.data.get("category", "").strip()
        optional_note = request.data.get("optional_note", "").strip() or None

        if not item_name or not category:
            return Response(
                {"error": "item_name and category are required"},
                status=status.HTTP_400_BAD_REQUEST)

        # get user profile
        try:
            tp = TasteProfile.objects.get(user=request.user)
        except TasteProfile.DoesNotExist:
            return Response(
                {"error": "Please complete Train My LLM first"},
                status=status.HTTP_400_BAD_REQUEST)

        profile  = tp.profile_json
        provider = request.user.preferred_llm
        api_key  = request.user.anthropic_api_key_enc or None

        try:
            engine = ReviewEngine(provider=provider, api_key=api_key or None)
            result = engine.generate_review(
                profile=profile,
                product_name=item_name,
                category=category,
                optional_note=optional_note,
            )
        except Exception as e:
            return Response(
                {"error": f"Generation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # save to database
        review = Review.objects.create(
            user             = request.user,
            category         = category,
            item_name        = item_name,
            optional_note    = optional_note or "",
            predicted_rating = result["predicted_rating"],
            rating_confidence = result["rating_confidence"],
            reasoning_shown  = result["reasoning_shown"],
            generated_review = result["review"],
            provider         = result["provider"],
        )

        return Response({
            "review_id":       review.id,
            "item_name":       item_name,
            "category":        category,
            "predicted_rating": result["predicted_rating"],
            "rating_confidence": result["rating_confidence"],
            "reasoning":       result["reasoning_shown"],
            "review":          result["review"],
            "provider":        result["provider"],
        }, status=status.HTTP_201_CREATED)

@extend_schema(
    summary="Submit feedback on a generated review",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "feedback_score": {"type": "string", "enum": ["off", "close", "spot_on"]},
                "edited_review":  {"type": "string", "example": "Telegram is good but the bugs are frustrating!!"},
            },
            "required": ["feedback_score"],
        }
    },
    responses={200: {"description": "Feedback saved, profile updating"}},
    tags=["Reviews"],
)
class FeedbackView(APIView):
    """
    POST /api/reviews/<review_id>/feedback/
    Stores user feedback and triggers profile update.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, review_id):
        feedback_score = request.data.get("feedback_score", "").strip()
        edited_review  = request.data.get("edited_review", "").strip()

        if feedback_score not in ("off", "close", "spot_on"):
            return Response(
                {"error": "feedback_score must be off, close, or spot_on"},
                status=status.HTTP_400_BAD_REQUEST)

        try:
            review = Review.objects.get(id=review_id, user=request.user)
        except Review.DoesNotExist:
            return Response(
                {"error": "Review not found"},
                status=status.HTTP_404_NOT_FOUND)

        review.feedback_score = feedback_score
        review.edited_review  = edited_review
        review.save()

        # trigger async profile update via Django Q2
        async_task(
            "reviews.tasks.update_profile_task",
            request.user.id,
            feedback_score,
            review.generated_review,
            edited_review,
            review.category,
        )

        return Response({"message": "Feedback saved. Profile updating."})

@extend_schema(
    summary="Get user's review history",
    responses={200: {"description": "List of past reviews"}},
    tags=["Reviews"],
)
class ReviewHistoryView(APIView):
    """GET /api/reviews/ — user's review history"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        reviews = Review.objects.filter(user=request.user).values(
            "id", "item_name", "category",
            "predicted_rating", "feedback_score", "created_at"
        )
        return Response(list(reviews))