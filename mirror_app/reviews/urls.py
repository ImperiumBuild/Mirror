"""
reviews/urls.py
"""
from django.urls import path
from .views import GenerateReviewView, FeedbackView, ReviewHistoryView, ReviewMetricsView, HumanValidationView

urlpatterns = [
    path("",                          ReviewHistoryView.as_view(),  name="review_history"),
    path("metrics/",                  ReviewMetricsView.as_view(),  name="review_metrics"),
    path("generate/",                 GenerateReviewView.as_view(), name="generate_review"),
    path("<int:review_id>/human-validation/", HumanValidationView.as_view(), name="human_validation"),
    path("<int:review_id>/feedback/", FeedbackView.as_view(),       name="review_feedback"),
]