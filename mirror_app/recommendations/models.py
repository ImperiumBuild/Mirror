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