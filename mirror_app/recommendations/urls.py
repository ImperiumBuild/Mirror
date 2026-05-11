"""
recommendations/urls.py
"""
from django.urls import path
from .views import RecommendView, DashboardView, ReactionView

urlpatterns = [
    path("",           RecommendView.as_view(),  name="recommend"),
    path("dashboard/", DashboardView.as_view(),  name="dashboard"),
    path("react/",     ReactionView.as_view(),   name="react"),
]