"""
reviews/models.py
"""

from django.db import models
from users.models import User


class Review(models.Model):
    """
    Stores every generated review and the user's feedback on it.
    """
    CATEGORY_CHOICES = [
        ("apps",         "Apps"),
        ("products",     "Products"),
        ("books",        "Books"),
        ("movies",       "Movies"),
        ("restaurants",  "Restaurants"),
    ]

    FEEDBACK_CHOICES = [
        ("off",      "Off"),
        ("close",    "Close"),
        ("spot_on",  "Spot On"),
    ]

    user             = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="reviews")

    category         = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    item_name        = models.CharField(max_length=255)
    optional_note    = models.TextField(blank=True)

    # LLM outputs
    predicted_rating = models.IntegerField()
    rating_confidence = models.CharField(max_length=20, blank=True)
    reasoning_shown  = models.TextField(blank=True)
    generated_review = models.TextField()

    # user feedback
    edited_review    = models.TextField(blank=True)
    feedback_score   = models.CharField(
        max_length=20, choices=FEEDBACK_CHOICES, blank=True)
    
    # Task A Validation Fields
    user_written_review = models.TextField(blank=True, help_text="The user's own review for comparison")
    rouge_l_score       = models.FloatField(null=True, blank=True)
    bert_score          = models.FloatField(null=True, blank=True)
    actual_user_rating  = models.IntegerField(null=True, blank=True, help_text="The rating the user actually gave")

    # provider used
    provider         = models.CharField(max_length=20, default="gemini")

    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "reviews"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} — {self.item_name} ({self.predicted_rating}★)"