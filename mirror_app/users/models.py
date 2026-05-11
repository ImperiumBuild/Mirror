"""
users/models.py
"""

from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Custom user model extending Django's AbstractUser.
    Adds profile fields collected during onboarding.
    """
    # onboarding fields
    age_range        = models.CharField(max_length=20, blank=True)
    occupation       = models.CharField(max_length=100, blank=True)
    training_complete = models.BooleanField(default=False)

    # LLM provider preference
    preferred_llm    = models.CharField(
        max_length=20,
        choices=[("gemini", "Gemini"), ("anthropic", "Anthropic")],
        default="gemini",
    )
    anthropic_api_key_enc = models.TextField(blank=True)

    class Meta:
        db_table = "users"

    def __str__(self):
        return self.username


class TasteProfile(models.Model):
    """
    Stores the full persona JSON built from Train My LLM.
    One per user. Updated by feedback background jobs.
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="taste_profile",
    )

    # OCEAN scores
    ocean_o = models.FloatField(default=0.5)
    ocean_c = models.FloatField(default=0.5)
    ocean_e = models.FloatField(default=0.5)
    ocean_a = models.FloatField(default=0.5)
    ocean_n = models.FloatField(default=0.5)

    # archetype
    dominant_archetype    = models.CharField(max_length=50, blank=True)
    secondary_archetype   = models.CharField(max_length=50, blank=True)
    archetype_confidence  = models.FloatField(default=0.0)

    # full profile JSON — single source of truth for LLM calls
    profile_json          = models.JSONField(default=dict)

    # plain-English summary shown to user
    profile_summary       = models.TextField(blank=True)

    # rating calibration
    generosity_score      = models.FloatField(default=0.5)
    harsh_or_generous     = models.CharField(max_length=30, default="balanced")

    # feedback tracking
    feedback_count        = models.IntegerField(default=0)
    review_count          = models.IntegerField(default=0)

    created_at            = models.DateTimeField(auto_now_add=True)
    updated_at            = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "taste_profiles"

    def __str__(self):
        return f"{self.user.username} — {self.dominant_archetype}"