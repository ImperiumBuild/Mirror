"""
users/urls.py
"""
from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views import RegisterView, LoginView, ProfileView, TrainView, LLMProviderView

urlpatterns = [
    path("register/",  RegisterView.as_view(),     name="register"),
    path("login/",     LoginView.as_view(),         name="login"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("profile/",   ProfileView.as_view(),       name="profile"),
    path("train/",     TrainView.as_view(),          name="train"),
    path("llm/",       LLMProviderView.as_view(),    name="llm_provider"),
]