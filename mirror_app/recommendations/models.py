"""
recommendations/models.py
"""

from django.db import models
from users.models import User


class RecommendationReaction(models.Model):
    """
    Stores user reactions to recommended items.
    Used to refine future recommendations.
    """
    REACTION_CHOICES = [
        ("interested",   "Interested"),
        ("not_for_me",   "Not For Me"),
        ("already_tried","Already Tried"),
    ]

    user         = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="reactions")
    item_name    = models.CharField(max_length=255)
    category     = models.CharField(max_length=50)
    sub_category = models.CharField(max_length=50, blank=True)
    reaction     = models.CharField(max_length=20, choices=REACTION_CHOICES)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "recommendation_reactions"
        unique_together = ["user", "item_name", "category"]

    def __str__(self):
        return f"{self.user.username} — {self.reaction} — {self.item_name}"
    
class RecommendationCache(models.Model):
    user         = models.ForeignKey(User, on_delete=models.CASCADE)
    category     = models.CharField(max_length=50)
    sub_category = models.CharField(max_length=50, blank=True)
    results_json = models.JSONField(default=list)
    generated_at = models.DateTimeField(auto_now=True)
    profile_version = models.IntegerField(default=0)  # tracks feedback_count

    class Meta:
        unique_together = ["user", "category"]

class ReviewBank(models.Model):
    category     = models.CharField(max_length=50, db_index=True)
    sub_category = models.CharField(max_length=50, blank=True, db_index=True)
    item_id      = models.CharField(max_length=255, blank=True)
    title        = models.CharField(max_length=500, blank=True, db_index=True)
    text         = models.TextField()
    rating       = models.FloatField()
    ocean_o      = models.FloatField(default=0.5)
    ocean_c      = models.FloatField(default=0.5)
    ocean_e      = models.FloatField(default=0.5)
    ocean_a      = models.FloatField(default=0.5)
    ocean_n      = models.FloatField(default=0.5)
    generosity_score     = models.FloatField(default=0.5)
    avg_review_length    = models.FloatField(default=0.0)
    source       = models.CharField(max_length=50, default="amazon")

    class Meta:
        db_table = "review_bank"
        indexes  = [
            models.Index(fields=["category", "ocean_a"]),
            models.Index(fields=["category", "title"]),
        ]


class UserReviewProfile(models.Model):
    """36k Amazon user profiles for archetype training reference."""
    user_id              = models.CharField(max_length=100, unique=True)
    source               = models.CharField(max_length=50, default="amazon")
    dominant_archetype   = models.CharField(max_length=50, db_index=True)
    secondary_archetype  = models.CharField(max_length=50, blank=True)
    archetype_confidence = models.FloatField(default=0.0)
    ocean_o              = models.FloatField(default=0.5)
    ocean_c              = models.FloatField(default=0.5)
    ocean_e              = models.FloatField(default=0.5)
    ocean_a              = models.FloatField(default=0.5)
    ocean_n              = models.FloatField(default=0.5)
    avg_rating           = models.FloatField(default=0.0)
    generosity_score     = models.FloatField(default=0.5)
    avg_review_length    = models.FloatField(default=0.0)

    class Meta:
        db_table = "user_review_profiles"
        indexes  = [models.Index(fields=["dominant_archetype", "ocean_a"])]



class Movie(models.Model):

    REGION_CHOICES = (
        ("hollywood", "Hollywood"),
        ("nollywood", "Nollywood"),
    )

    CATEGORY_CHOICES = (
        ("action", "Action"),
        ("adventure", "Adventure"),
        ("romance", "Romance"),
        ("thriller", "Thriller"),
        ("tragedy", "Tragedy"),
        ("mind_bending", "Mind Bending"),
    )
    tmdb_id = models.IntegerField(unique=True, null=True, blank=True)
    vote_count = models.IntegerField(default=0)

    title = models.CharField(max_length=255)

    region = models.CharField(max_length=20, choices=REGION_CHOICES)

    category = models.CharField(max_length=50)

    overview = models.TextField()

    vote_average = models.FloatField(default=0)

    popularity = models.FloatField(default=0)

    emotional_tone = models.TextField(blank=True)

    top_praises = models.JSONField(default=list)

    poster_path = models.CharField(max_length=255, null=True, blank=True)
    top_critics = models.JSONField(default=list)

class ArchetypeAffinity(models.Model):
    """
    Stores pre-computed weights for how much an archetype likes a specific item/category.
    This is the 'Secret Sauce' that powers the data-driven recommendation engine.
    """
    archetype_id = models.CharField(max_length=50, db_index=True)
    item_title   = models.CharField(max_length=255, db_index=True)
    category     = models.CharField(max_length=50, db_index=True)

    # The score is calculated based on frequency and rating within that cluster
    affinity_score = models.FloatField(default=0.0)

    # Metadata for transparency
    review_count   = models.IntegerField(default=0)
    avg_rating     = models.FloatField(default=0.0)

    class Meta:
        db_table = "archetype_affinities"
        unique_together = ["archetype_id", "item_title", "category"]
        indexes = [
            models.Index(fields=["archetype_id", "category", "-affinity_score"]),
        ]