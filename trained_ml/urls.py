from django.contrib import admin
from django.urls import path, include
from .views import TrainedModels

urlpatterns = [
    path("trained_models/", TrainedModels.as_view(), name="trained_models"),

]
