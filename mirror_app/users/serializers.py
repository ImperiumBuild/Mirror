"""
users/serializers.py
"""
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import User, TasteProfile


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True, required=True, validators=[validate_password])

    class Meta:
        model  = User
        fields = ["username", "email", "password", "age_range", "occupation"]

    def create(self, validated_data):
        user = User.objects.create_user(
            username   = validated_data["username"],
            email      = validated_data.get("email", ""),
            password   = validated_data["password"],
            age_range  = validated_data.get("age_range", ""),
            occupation = validated_data.get("occupation", ""),
        )
        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model  = User
        fields = ["id", "username", "email", "age_range", "occupation",
                  "training_complete", "preferred_llm"]


class TasteProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model  = TasteProfile
        fields = ["dominant_archetype", "secondary_archetype",
                  "archetype_confidence", "profile_summary",
                  "generosity_score", "harsh_or_generous",
                  "feedback_count", "review_count",
                  "ocean_o", "ocean_c", "ocean_e", "ocean_a", "ocean_n",
                  "updated_at"]


class TrainSerializer(serializers.Serializer):
    layer1_answers      = serializers.DictField(child=serializers.CharField())
    pairwise_picks      = serializers.ListField(child=serializers.DictField())
    calibration_answers = serializers.DictField(child=serializers.IntegerField())
    user_reviews        = serializers.ListField(child=serializers.CharField(), required=False)

class LLMProviderSerializer(serializers.Serializer):
    preferred_llm         = serializers.ChoiceField(choices=["gemini", "anthropic"])
    anthropic_api_key     = serializers.CharField(required=False, allow_blank=True)