"""
reviews/urls.py
"""
from django.urls import path
from .views import GenerateReviewView, FeedbackView, ReviewHistoryView

urlpatterns = [
    path("",                          ReviewHistoryView.as_view(),  name="review_history"),
    path("generate/",                 GenerateReviewView.as_view(), name="generate_review"),
    path("<int:review_id>/feedback/", FeedbackView.as_view(),       name="review_feedback"),
]