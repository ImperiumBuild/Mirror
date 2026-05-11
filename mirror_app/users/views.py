"""
users/views.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken

from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

from .models import User, TasteProfile
from .serializers import (
    RegisterSerializer, UserSerializer,
    TasteProfileSerializer, TrainSerializer,
    LLMProviderSerializer,
)
from core.ocean.profile_builder import build_profile

@extend_schema(
    summary="Register a new user",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "username":   {"type": "string", "example": "thimmie"},
                "email":      {"type": "string", "example": "thimmie@email.com"},
                "password":   {"type": "string", "example": "securepass123"},
                "age_range":  {"type": "string", "example": "25-34"},
                "occupation": {"type": "string", "example": "Data Engineer"},
            },
            "required": ["username", "password"],
        }
    },
    responses={201: {"description": "User created with JWT tokens"}},
    tags=["Auth"],
)
class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user    = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                "user":    UserSerializer(user).data,
                "refresh": str(refresh),
                "access":  str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@extend_schema(
    summary="Login and get JWT tokens",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "username": {"type": "string", "example": "thimmie"},
                "password": {"type": "string", "example": "securepass123"},
            },
            "required": ["username", "password"],
        }
    },
    responses={200: {"description": "JWT access and refresh tokens"}},
    tags=["Auth"],
)
class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        from django.contrib.auth import authenticate
        username = request.data.get("username")
        password = request.data.get("password")
        user     = authenticate(username=username, password=password)

        if not user:
            return Response(
                {"error": "Invalid credentials"},
                status=status.HTTP_401_UNAUTHORIZED)

        refresh = RefreshToken.for_user(user)
        return Response({
            "user":    UserSerializer(user).data,
            "refresh": str(refresh),
            "access":  str(refresh.access_token),
        })

@extend_schema(
    summary="Get current user profile and taste profile",
    responses={200: {"description": "User info + taste profile JSON"}},
    tags=["Users"],
)
class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        data = UserSerializer(user).data
        try:
            tp   = TasteProfile.objects.get(user=user)
            data["taste_profile"] = TasteProfileSerializer(tp).data
        except TasteProfile.DoesNotExist:
            data["taste_profile"] = None
        return Response(data)

@extend_schema(
    summary="Submit Train My LLM answers to build persona",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "layer1_answers": {
                    "type": "object",
                    "example": {
                        "q_damage": "b",
                        "q_restaurant": "a",
                        "q_app_bug": "c"
                    }
                },
                "pairwise_picks": {
                    "type": "array",
                    "example": [
                        {"chosen_text": "This app needs to be checked. The transaction feature should at least refund a failed transaction.", "tag": "tone"},
                        {"chosen_text": "WhatsApp needs to watch their updates. Video calls are not possible at this point.", "tag": "focus"}
                    ]
                },
                "calibration_answers": {
                    "type": "object",
                    "example": {"c_delivery": 3, "c_movie": 4}
                },
                "user_review": {
                    "type": "string",
                    "example": "I cant enjoy the app if you keep rolling bugs. Fix this ASAP!!"
                }
            },
            "required": ["layer1_answers", "pairwise_picks", "calibration_answers"],
        }
    },
    responses={201: {"description": "Persona profile built and stored"}},
    tags=["Users"],
)
class TrainView(APIView):
    """
    Receives Train My LLM answers and builds the user's persona profile.
    POST /api/users/train/
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = TrainSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        try:
            profile = build_profile(
                layer1_answers      = data["layer1_answers"],
                pairwise_picks      = data["pairwise_picks"],
                calibration_answers = data["calibration_answers"],
                user_review         = data.get("user_review", ""),
                user_meta           = {
                    "age_range":  request.user.age_range,
                    "occupation": request.user.occupation,
                },
            )
        except Exception as e:
            return Response(
                {"error": f"Profile build failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        ocean = profile.get("ocean", {})

        # save to TasteProfile
        tp, _ = TasteProfile.objects.update_or_create(
            user=request.user,
            defaults={
                "ocean_o":             ocean.get("O", 0.5),
                "ocean_c":             ocean.get("C", 0.5),
                "ocean_e":             ocean.get("E", 0.5),
                "ocean_a":             ocean.get("A", 0.5),
                "ocean_n":             ocean.get("N", 0.5),
                "dominant_archetype":  profile.get("dominant_archetype", ""),
                "secondary_archetype": profile.get("secondary_archetype", ""),
                "archetype_confidence":profile.get("archetype_confidence", 0),
                "profile_json":        profile,
                "profile_summary":     profile.get("profile_summary", ""),
                "generosity_score":    profile.get("rating_calibration", {}).get("generosity_score", 0.5),
                "harsh_or_generous":   profile.get("rating_calibration", {}).get("harsh_or_generous", "balanced"),
            }
        )

        request.user.training_complete = True
        request.user.save()

        return Response({
            "message":       "Profile built successfully",
            "taste_profile": TasteProfileSerializer(tp).data,
        }, status=status.HTTP_201_CREATED)

@extend_schema(
    summary="Update preferred LLM provider",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "preferred_llm":     {"type": "string", "enum": ["gemini", "anthropic"]},
                "anthropic_api_key": {"type": "string", "example": "sk-ant-..."},
            },
            "required": ["preferred_llm"],
        }
    },
    responses={200: {"description": "Provider updated"}},
    tags=["Users"],
)
class LLMProviderView(APIView):
    """Update user's preferred LLM provider."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = LLMProviderSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data                  = serializer.validated_data
        request.user.preferred_llm = data["preferred_llm"]

        if data.get("anthropic_api_key"):
            request.user.anthropic_api_key_enc = data["anthropic_api_key"]

        request.user.save()
        return Response({"message": "LLM provider updated"})