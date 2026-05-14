"""
recommendations/views.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import traceback
from datetime import timedelta
from django.utils import timezone
from recommendations.models import RecommendationReaction, RecommendationCache


from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

from .models import RecommendationReaction
from users.models import TasteProfile
from core.recommendation_engine import RecommendationEngine
import traceback
import logging

logger = logging.getLogger(__name__)

@extend_schema(
    summary="Get personalised recommendations for a category",
    parameters=[
        OpenApiParameter(
            name="category",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Category to recommend for",
            enum=["apps", "products", "books", "movies"],
            required=True,
            examples=[
                OpenApiExample("Apps", value="apps"),
                OpenApiExample("Products", value="products"),
            ],
        )
    ],
    responses={200: {"description": "Ranked recommendations with reasoning"}},
    tags=["Recommendations"],
)
class RecommendView(APIView):
    """
    GET /api/recommendations/?category=apps
    Returns personalised recommendations for a category.
    For apps, returns sub-categories.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        category = request.query_params.get("category", "").strip()
        force    = request.query_params.get("force", "false") == "true"

        if not category:
            return Response(
                {"error": "category query param required"},
                status=status.HTTP_400_BAD_REQUEST)

        try:
            tp = TasteProfile.objects.get(user=request.user)
        except TasteProfile.DoesNotExist:
            return Response(
                {"error": "Please complete Train My LLM first"},
                status=status.HTTP_400_BAD_REQUEST)

        # check cache first
        if not force:
            cached = RecommendationCache.objects.filter(
                user            = request.user,
                category        = category,
                profile_version = tp.feedback_count,
                generated_at__gte = timezone.now() - timedelta(hours=24),
            ).first()
            if cached:
                return Response({
                    "category":        category,
                    "recommendations": cached.results_json,
                    "cached":          True,
                })

        profile  = tp.profile_json
        provider = request.user.preferred_llm
        api_key  = request.user.anthropic_api_key_enc or None

        try:
            engine = RecommendationEngine(
                provider=provider, api_key=api_key or None)

            if category.lower() == "apps":
                result = engine.recommend_apps_by_subcategory(
                    profile=profile, n_per_subcategory=3)
            else:
                result = engine.recommend(
                    profile=profile, category=category, n=5)

        except Exception as e:
            logger.error("RecommendView crashed", exc_info=True)
            print("🔥 FULL TRACEBACK:")
            traceback.print_exc()

            return Response(
                {
                    "error": str(e),
                    "type": type(e).__name__,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # save to cache
        RecommendationCache.objects.update_or_create(
            user     = request.user,
            category = category,
            defaults = {
                "results_json":    result,
                "profile_version": tp.feedback_count,
            }
        )

        return Response({
            "category":        category,
            "recommendations": result,
            "cached":          False,
        })


@extend_schema(
    summary="Get recommendations across all categories for the dashboard",
    responses={
        200: {
            "description": "Dashboard recommendations",
            "example": {
                "apps": {
                    "productivity": [{"rank": 1, "title": "Notion", "reasoning": "..."}],
                    "social":       [{"rank": 1, "title": "WhatsApp", "reasoning": "..."}],
                },
                "books":    [{"rank": 1, "title": "Atomic Habits", "reasoning": "..."}],
                "products": [{"rank": 1, "title": "AirPods Pro", "reasoning": "..."}],
                "movies":   [{"rank": 1, "title": "Parasite", "reasoning": "..."}],
            }
        }
    },
    tags=["Recommendations"],
)
class DashboardView(APIView):
    """
    GET /api/recommendations/dashboard/
    Returns recommendations across all categories for the dashboard.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
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
            engine = RecommendationEngine(
                provider=provider, api_key=api_key or None)
            result = engine.recommend_dashboard(
                profile=profile, n_per_category=3)
        except Exception as e:
            return Response(
                {"error": f"Dashboard failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(result)

@extend_schema(
    summary="Submit a reaction to a recommendation",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "item_name":    {"type": "string", "example": "Notion"},
                "category":     {"type": "string", "example": "apps"},
                "sub_category": {"type": "string", "example": "productivity"},
                "reaction":     {"type": "string", "enum": ["interested", "not_for_me", "already_tried"]},
            },
            "required": ["item_name", "category", "reaction"],
        }
    },
    responses={200: {"description": "Reaction saved"}},
    tags=["Recommendations"],
)
class ReactionView(APIView):
    """
    POST /api/recommendations/react/
    Stores user reaction to a recommendation.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        item_name    = request.data.get("item_name", "").strip()
        category     = request.data.get("category", "").strip()
        sub_category = request.data.get("sub_category", "").strip()
        reaction     = request.data.get("reaction", "").strip()

        if not all([item_name, category, reaction]):
            return Response(
                {"error": "item_name, category, and reaction are required"},
                status=status.HTTP_400_BAD_REQUEST)

        if reaction not in ("interested", "not_for_me", "already_tried"):
            return Response(
                {"error": "Invalid reaction"},
                status=status.HTTP_400_BAD_REQUEST)

        obj, created = RecommendationReaction.objects.update_or_create(
            user      = request.user,
            item_name = item_name,
            category  = category,
            defaults  = {
                "sub_category": sub_category,
                "reaction":     reaction,
            }
        )

        return Response({
            "message": "Reaction saved",
            "reaction": reaction,
            "created": created,
        })